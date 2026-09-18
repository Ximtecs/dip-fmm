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
#include <unordered_map>
#include <vector>

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

    ExactOperatorClasses result;
    result.class_of_pair.resize(pair_count);
    std::unordered_map<ExactOperatorKey, std::uint32_t, ExactOperatorKeyHash>
        classes;
    for (std::size_t index = 0; index < pair_count; ++index) {
        const auto [entry, inserted] = classes.try_emplace(
            key_of_pair(index),
            static_cast<std::uint32_t>(result.representative.size()));
        if (inserted) {
            if (result.representative.size() >= gate.max_classes) {
                return {};
            }
            result.representative.push_back(static_cast<std::uint32_t>(index));
        }
        result.class_of_pair[index] = entry->second;

        const std::size_t sampled = index + 1;
        if (sampled == gate.sample_pairs && sampled < pair_count &&
            result.representative.size() * gate.sample_reuse_factor >
                sampled) {
            return {};
        }
    }
    result.classified = true;
    return result;
}

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

} // namespace cdfmm::detail::exact_reuse
