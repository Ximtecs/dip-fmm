// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>

#ifdef CDFMM_USE_OPENMP
#include <omp.h>
#endif

#include "cdfmm/geometry/primitives/rectangular_prism.hpp"
#include "cdfmm/geometry/primitives/tetrahedron.hpp"
#include "cdfmm/math/vec3.hpp"

namespace cdfmm::detail::exact_reuse {

// Exact operator classification.
//
// Every exact pair tensor is a pure function of the displacement and of the
// participating body records; where a periodic image shift applies it is
// folded into the displacement before any tensor call.  Pairs whose inputs
// agree bit for bit therefore describe one and the same operator exactly, so
// the expensive exact evaluation can run once per distinct input and be
// scattered to the pairs that share it.  Repeated geometry makes this
// decisive: a regular lattice reaches only a few hundred distinct
// displacements however many bodies it holds.
//
// The key stores raw bit patterns and is never compared with a tolerance.
// `-0.0` stays distinct from `+0.0`, which can at worst repeat one build,
// whereas merging them would assume a continuity the exact corner formulas do
// not have.
//
// This header owns the definition of that equivalence so the near-field P2P
// builder and the dense all-to-all builder cannot drift apart on what "the
// same operator" means.  It owns no mathematics and no storage layout: which
// values enter a key, and where a built tensor is written, stay with each
// caller.

/// @brief Widest exact key a pair loop needs: a displacement, two tetrahedron
///        records, and the tetrahedron pair's coincident-geometry selector.
inline constexpr std::size_t exact_operator_key_capacity = 3 + 12 + 12 + 1;

/// @brief Bitwise description of everything one exact pair tensor depends on.
struct ExactOperatorKey {
    std::array<std::uint64_t, exact_operator_key_capacity> words{};
    std::size_t used{0};

    void push(const double value) noexcept
    {
        assert(used < exact_operator_key_capacity);
        words[used] = std::bit_cast<std::uint64_t>(value);
        ++used;
    }

    void push(const Vec3& value) noexcept
    {
        push(value.x);
        push(value.y);
        push(value.z);
    }

    void push(const RectangularPrism& prism) noexcept
    {
        push(prism.hx);
        push(prism.hy);
        push(prism.hz);
    }

    void push(const Tetrahedron& tetrahedron) noexcept
    {
        for (const Vec3& vertex : tetrahedron.vertices) {
            push(vertex);
        }
    }

    [[nodiscard]] bool operator==(const ExactOperatorKey& other) const noexcept
    {
        return used == other.used &&
            std::equal(words.begin(),
                       words.begin() + static_cast<std::ptrdiff_t>(used),
                       other.words.begin());
    }
};

struct ExactOperatorKeyHash {
    [[nodiscard]] std::size_t operator()(
        const ExactOperatorKey& key) const noexcept
    {
        std::uint64_t hash = 0x9e3779b97f4a7c15ULL ^ key.used;
        for (std::size_t word = 0; word < key.used; ++word) {
            hash ^= key.words[word];
            hash *= 0x00000100000001b3ULL;
            hash ^= hash >> 29;
        }
        return static_cast<std::size_t>(hash);
    }
};

/// @brief Pairs grouped by bitwise-identical exact operator inputs.
struct ExactOperatorClasses {
    /// @brief Class of each pair; empty when classification was abandoned.
    std::vector<std::uint32_t> class_of_pair;
    /// @brief Lowest pair index in each class, which is the pair that builds
    ///        it.  Serial construction would have reached that pair first, so
    ///        selecting it keeps a failure's reported cause unchanged.
    std::vector<std::uint32_t> representative;
    /// @brief False when the inputs were too diverse to repay classification.
    bool classified{false};

    /// @brief Transient bytes the class map and representative list hold.
    [[nodiscard]] std::size_t transient_bytes() const noexcept
    {
        return class_of_pair.size() * sizeof(std::uint32_t) +
            representative.size() * sizeof(std::uint32_t);
    }
};

/// @brief How eagerly a caller pays for classification before giving up.
struct ExactReuseGate {
    /// @brief Pairs examined before the reuse estimate is acted on.
    std::size_t sample_pairs{65536};
    /// @brief Minimum duplicates per distinct operator worth classifying for.
    std::size_t sample_reuse_factor{2};
    /// @brief Hard cap on distinct operators, for a caller that must bound
    ///        the transient storage its representatives will occupy.  The
    ///        sample is an estimate; this is a guarantee.
    std::size_t max_classes{std::numeric_limits<std::size_t>::max()};
};

/// @brief Group pairs whose exact operator inputs agree bit for bit.
///
/// `key_of_pair(index)` returns the complete set of values the tensor of that
/// pair depends on.  Classes are numbered in first-seen order, so the classes,
/// their representatives, and therefore every built value are independent of
/// thread count and of hash iteration order.
///
/// Classification stops when an initial sample shows too few duplicates to
/// repay it, which bounds both the table and the wasted lookups on irregular
/// geometry.  That is only ever a performance decision: the caller builds the
/// same operators either way.  A caller whose per-pair tensor is cheap enough
/// that the lookup itself would dominate should not call this at all.
///
/// It also stops when the pair indices do not fit the 32-bit words the class
/// map and the representative list store, which a dense all-to-all plan can
/// reach on a large-memory machine at roughly 65536 bodies per side.
template <typename KeyOfPair>
[[nodiscard]] ExactOperatorClasses classify_exact_operators(
    const std::size_t pair_count, const KeyOfPair& key_of_pair,
    const ExactReuseGate gate = {})
{
    // `representative` stores pair indices and `class_of_pair` stores class
    // numbers, both as `std::uint32_t`; the largest value either can hold is
    // `pair_count - 1`.  Abandon rather than narrow: widening the vectors
    // instead would double `class_of_pair`, which holds one entry per pair
    // and is already the largest transient allocation of a big build.
    if (pair_count != 0 &&
        pair_count - 1 > std::numeric_limits<std::uint32_t>::max()) {
        return {};
    }
    using Map = std::unordered_map<ExactOperatorKey, std::uint32_t,
                                   ExactOperatorKeyHash>;

    // The sample gate, serial and first: an initial prefix with too few
    // duplicates means the geometry is irregular and no classification is
    // worth paying for.  Its map is discarded; the prefix is rehashed below.
    if (gate.sample_pairs < pair_count) {
        Map probe;
        std::size_t distinct = 0;
        for (std::size_t index = 0; index < gate.sample_pairs; ++index) {
            if (probe.try_emplace(key_of_pair(index), 0U).second) {
                ++distinct;
            }
        }
        if (distinct * gate.sample_reuse_factor > gate.sample_pairs) {
            return {};
        }
    }

    ExactOperatorClasses result;
    result.class_of_pair.resize(pair_count);

    // Parallel first-seen classification.  The pairs are cut into contiguous
    // ranges; every range numbers the classes it meets in its own first-seen
    // order and records the pair index of each first occurrence.  The merge
    // walks the ranges in order and, within a range, its classes in that
    // order, so a class receives the rank of its first occurrence over ALL
    // pairs -- exactly the number the serial loop gave it -- whatever the
    // number of ranges: a class first met in a later range cannot have an
    // earlier first occurrence than any class of an earlier range, and two
    // classes of one range are ordered by their first occurrences.  Class
    // numbers, representatives and every built value are therefore
    // independent of the thread count, as before.
    struct Range {
        std::size_t begin{0};
        std::size_t end{0};
        Map ids{};
        std::vector<std::uint32_t> first{};      // pair index per local class
        std::vector<ExactOperatorKey> keys{};    // key per local class
        bool overflowed{false};
    };
    std::size_t range_count = 1;
#ifdef CDFMM_USE_OPENMP
    range_count = static_cast<std::size_t>(std::max(1, omp_get_max_threads()));
#endif
    // A range below the sample size gains nothing from another thread.
    range_count = std::max<std::size_t>(
        1, std::min(range_count, pair_count / std::max<std::size_t>(gate.sample_pairs, 1)));
    std::vector<Range> ranges(range_count);
    for (std::size_t r = 0; r < range_count; ++r) {
        ranges[r].begin = pair_count * r / range_count;
        ranges[r].end = pair_count * (r + 1) / range_count;
    }
#pragma omp parallel for schedule(static) if (range_count > 1)
    for (std::ptrdiff_t raw = 0; raw < static_cast<std::ptrdiff_t>(range_count);
         ++raw) {
        Range& range = ranges[static_cast<std::size_t>(raw)];
        for (std::size_t index = range.begin; index < range.end; ++index) {
            ExactOperatorKey key = key_of_pair(index);
            const auto [entry, inserted] = range.ids.try_emplace(
                key, static_cast<std::uint32_t>(range.first.size()));
            if (inserted) {
                if (range.first.size() >= gate.max_classes) {
                    range.overflowed = true;
                    break;
                }
                range.first.push_back(static_cast<std::uint32_t>(index));
                range.keys.push_back(key);
            }
            result.class_of_pair[index] = entry->second;
        }
    }
    for (const Range& range : ranges) {
        if (range.overflowed) {
            return {};
        }
    }

    // Serial merge in range order: local class -> global class.
    Map global;
    std::vector<std::vector<std::uint32_t>> global_of_local(range_count);
    for (std::size_t r = 0; r < range_count; ++r) {
        Range& range = ranges[r];
        global_of_local[r].resize(range.keys.size());
        for (std::size_t local = 0; local < range.keys.size(); ++local) {
            const auto [entry, inserted] = global.try_emplace(
                range.keys[local],
                static_cast<std::uint32_t>(result.representative.size()));
            if (inserted) {
                if (result.representative.size() >= gate.max_classes) {
                    return {};
                }
                result.representative.push_back(range.first[local]);
            }
            global_of_local[r][local] = entry->second;
        }
        // The range's map and keys are no longer needed; free them before
        // the next range's are merged so the peak stays one range's worth.
        range.ids = {};
        range.keys = {};
    }
#pragma omp parallel for schedule(static) if (range_count > 1)
    for (std::ptrdiff_t raw = 0; raw < static_cast<std::ptrdiff_t>(range_count);
         ++raw) {
        const Range& range = ranges[static_cast<std::size_t>(raw)];
        const std::vector<std::uint32_t>& table =
            global_of_local[static_cast<std::size_t>(raw)];
        for (std::size_t index = range.begin; index < range.end; ++index) {
            result.class_of_pair[index] = table[result.class_of_pair[index]];
        }
    }
    result.classified = true;
    return result;
}

/// @brief Exact operators built so far, keyed by their inputs.
///
/// A chunked near-field build classifies each chunk on its own, so a
/// displacement that recurs in every chunk (every one on a lattice) would be
/// rebuilt once per chunk.  The memo carries the built tensors from one call
/// to the next; because a tensor is a pure function of its key, a remembered
/// tensor is bit for bit the one the build would produce.
template <typename Tensor>
struct ExactOperatorMemo {
    std::unordered_map<ExactOperatorKey, Tensor, ExactOperatorKeyHash> tensors{};
    /// @brief Entries beyond which nothing further is remembered; an entry
    ///        is about 300 bytes, so the default bounds the memo near 1.3 GB.
    std::size_t max_entries{std::size_t{1} << 22};
};

/// @brief The failure a parallel exact-operator loop reports.
///
/// The lowest failing index wins, because that is the item a serial build
/// would have reached first, so the reported cause does not depend on how the
/// iterations were scheduled.  Work above a known failure is skipped: it can no
/// longer win that comparison.
class FirstFailure {
public:
    [[nodiscard]] bool superseded(const std::size_t index) const noexcept
    {
        return index > index_.load(std::memory_order_relaxed);
    }

    /// @brief Record the exception currently being handled for @p index.
    void record(const std::size_t index)
    {
        std::size_t previous = index_.load(std::memory_order_relaxed);
        while (index < previous &&
               !index_.compare_exchange_weak(previous, index,
                                             std::memory_order_relaxed)) {
        }
#pragma omp critical(cdfmm_exact_operator_build_exception)
        {
            if (index_.load(std::memory_order_relaxed) == index) {
                exception_ = std::current_exception();
            }
        }
    }

    void rethrow_any() const
    {
        if (exception_) {
            std::rethrow_exception(exception_);
        }
    }

private:
    std::atomic<std::size_t> index_{std::numeric_limits<std::size_t>::max()};
    std::exception_ptr exception_{};
};

/// @brief One tensor per class: from @p memo where its key is remembered,
///        otherwise built from the class representative.
///
/// The builds run in parallel over the classes the memo does not hold.  A
/// class representative is the lowest pair index of its class and the
/// representatives are listed in first-seen order, so the lowest failing
/// representative is the pair a serial build would have failed on first;
/// @p build_of_pair takes a representative as `classes.representative` lists
/// it.  A null @p memo builds every class.
template <typename Tensor, typename KeyOfPair, typename BuildOfPair>
[[nodiscard]] std::vector<Tensor> build_exact_operator_classes(
    const ExactOperatorClasses& classes, const KeyOfPair& key_of_pair,
    const BuildOfPair& build_of_pair, ExactOperatorMemo<Tensor>* memo)
{
    const std::size_t class_count = classes.representative.size();
    std::vector<Tensor> tensors(class_count);
    // Classes the memo does not hold, with their keys for the insertion
    // below so no key is derived twice.
    std::vector<std::uint32_t> pending;
    std::vector<ExactOperatorKey> pending_keys;
    if (memo == nullptr) {
        pending.resize(class_count);
        std::iota(pending.begin(), pending.end(), std::uint32_t{0});
    } else {
        pending.reserve(class_count);
        pending_keys.reserve(class_count);
        for (std::size_t entry = 0; entry < class_count; ++entry) {
            ExactOperatorKey key = key_of_pair(classes.representative[entry]);
            const auto found = memo->tensors.find(key);
            if (found != memo->tensors.end()) {
                tensors[entry] = found->second;
            } else {
                pending.push_back(static_cast<std::uint32_t>(entry));
                pending_keys.push_back(key);
            }
        }
    }

    FirstFailure failure;
    const std::ptrdiff_t pending_count =
        static_cast<std::ptrdiff_t>(pending.size());
#pragma omp parallel for schedule(dynamic, 1) if (pending_count >= 8)
    for (std::ptrdiff_t raw = 0; raw < pending_count; ++raw) {
        const std::size_t entry = pending[static_cast<std::size_t>(raw)];
        const std::size_t representative = classes.representative[entry];
        if (failure.superseded(representative)) {
            continue;
        }
        try {
            tensors[entry] = build_of_pair(representative);
        } catch (...) {
            failure.record(representative);
        }
    }
    failure.rethrow_any();

    if (memo != nullptr) {
        for (std::size_t raw = 0; raw < pending.size(); ++raw) {
            if (memo->tensors.size() >= memo->max_entries) {
                break;
            }
            memo->tensors.emplace(pending_keys[raw], tensors[pending[raw]]);
        }
    }
    return tensors;
}

} // namespace cdfmm::detail::exact_reuse
