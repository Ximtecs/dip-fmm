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
// Every exact operator built at construction is a pure function of a few
// geometry values: a pair tensor of the displacement and the two body records
// (with any periodic image shift folded into the displacement before the
// tensor call), a finite P2M or L2P operator of the body shapes and their
// offsets from the leaf centre.  Items whose inputs agree bit for bit
// therefore describe one and the same operator exactly, so the expensive
// exact evaluation can run once per distinct input and be scattered to the
// items that share it.  Repeated geometry makes this decisive: a regular
// lattice reaches only a few hundred distinct displacements however many
// bodies it holds, and every leaf of a lattice has the same internal layout.
//
// The key stores raw bit patterns and is never compared with a tolerance.
// `-0.0` stays distinct from `+0.0`, which can at worst repeat one build,
// whereas merging them would assume a continuity the exact corner formulas do
// not have.
//
// This header owns the definition of that equivalence so the near-field P2P
// builder, the dense all-to-all builder and the finite endpoint (P2M/L2P)
// builders cannot drift apart on what "the same operator" means.  It owns no
// mathematics and no storage layout: which values enter a key, and where a
// built operator is written, stay with each caller.

/// @brief Widest exact pair key a pair loop needs: a displacement, two
///        tetrahedron records, and the tetrahedron pair's coincident-geometry
///        selector.
inline constexpr std::size_t exact_operator_key_capacity = 3 + 12 + 12 + 1;

/// @brief Bitwise description of everything one exact pair tensor depends on.
///
/// Fixed capacity and no heap allocation, because a pair loop builds one key
/// per near-field or dense pair.
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

    [[nodiscard]] const std::uint64_t* begin() const noexcept
    {
        return words.data();
    }

    [[nodiscard]] std::size_t size() const noexcept { return used; }

    [[nodiscard]] bool operator==(const ExactOperatorKey& other) const noexcept
    {
        return used == other.used &&
            std::equal(words.begin(),
                       words.begin() + static_cast<std::ptrdiff_t>(used),
                       other.words.begin());
    }
};

/// @brief Growable counterpart of `ExactOperatorKey` for operators whose input
///        count is not bounded in advance.
///
/// A leaf P2M operator depends on every body in the leaf, so its key grows
/// with the leaf occupancy.  The endpoint loops build one key per leaf or per
/// target, so the heap allocation per key is affordable there; pair loops
/// keep the fixed-capacity key.  Same bit semantics, same hash.
struct GrowableExactOperatorKey {
    std::vector<std::uint64_t> words{};

    void reserve(const std::size_t count) { words.reserve(count); }

    void push(const double value)
    {
        words.push_back(std::bit_cast<std::uint64_t>(value));
    }

    void push(const Vec3& value)
    {
        push(value.x);
        push(value.y);
        push(value.z);
    }

    void push(const RectangularPrism& prism)
    {
        push(prism.hx);
        push(prism.hy);
        push(prism.hz);
    }

    void push(const Tetrahedron& tetrahedron)
    {
        for (const Vec3& vertex : tetrahedron.vertices) {
            push(vertex);
        }
    }

    [[nodiscard]] const std::uint64_t* begin() const noexcept
    {
        return words.data();
    }

    [[nodiscard]] std::size_t size() const noexcept { return words.size(); }

    [[nodiscard]] bool operator==(
        const GrowableExactOperatorKey& other) const noexcept
    {
        return words == other.words;
    }
};

/// @brief Hash of either key type over its used words.
///
/// Classes are numbered in first-seen order (see `classify_exact_operators`),
/// so the hash affects only lookup speed, never which items share a class.
struct ExactOperatorKeyHash {
    template <typename Key>
    [[nodiscard]] std::size_t operator()(const Key& key) const noexcept
    {
        std::uint64_t hash = 0x9e3779b97f4a7c15ULL ^ key.size();
        const std::uint64_t* word = key.begin();
        for (std::size_t index = 0; index < key.size(); ++index) {
            hash ^= word[index];
            hash *= 0x00000100000001b3ULL;
            hash ^= hash >> 29;
        }
        return static_cast<std::size_t>(hash);
    }
};

/// @brief Items grouped by bitwise-identical exact operator inputs.
///
/// An item is whatever the caller classifies: a near-field pair, a dense
/// pair, a source leaf (one P2M operator) or a target (one L2P evaluator).
struct ExactOperatorClasses {
    /// @brief Class of each item; empty when classification was abandoned.
    std::vector<std::uint32_t> class_of_item;
    /// @brief Lowest item index in each class, which is the item that builds
    ///        it.  Serial construction would have reached that item first, so
    ///        selecting it keeps a failure's reported cause unchanged.
    std::vector<std::uint32_t> representative;
    /// @brief False when the inputs were too diverse to repay classification.
    bool classified{false};

    /// @brief Transient bytes the class map and representative list hold.
    [[nodiscard]] std::size_t transient_bytes() const noexcept
    {
        return class_of_item.size() * sizeof(std::uint32_t) +
            representative.size() * sizeof(std::uint32_t);
    }
};

/// @brief How eagerly a caller pays for classification before giving up.
struct ExactReuseGate {
    /// @brief Items examined before the reuse estimate is acted on.  Pair
    ///        loops use the default; the endpoint loops, whose items are
    ///        leaves or targets rather than pairs, sample 4096.
    std::size_t sample_items{65536};
    /// @brief Minimum duplicates per distinct operator worth classifying for.
    std::size_t sample_reuse_factor{2};
    /// @brief Hard cap on distinct operators, for a caller that must bound
    ///        the transient storage its representatives will occupy.  The
    ///        sample is an estimate; this is a guarantee.
    std::size_t max_classes{std::numeric_limits<std::size_t>::max()};
};

/// @brief Group items whose exact operator inputs agree bit for bit.
///
/// `key_of_item(index)` returns the complete set of values the operator of
/// that item depends on, as an `ExactOperatorKey` or a
/// `GrowableExactOperatorKey`.  Classes are numbered in first-seen order, so
/// the classes, their representatives, and therefore every built value are
/// independent of thread count and of hash iteration order.
///
/// Classification stops when an initial sample shows too few duplicates to
/// repay it, which bounds both the table and the wasted lookups on irregular
/// geometry.  That is only ever a performance decision: the caller builds the
/// same operators either way.  A caller whose per-item operator is cheap
/// enough that the lookup itself would dominate should not call this at all.
///
/// It also stops when the item indices do not fit the 32-bit words the class
/// map and the representative list store, which a dense all-to-all plan can
/// reach on a large-memory machine at roughly 65536 bodies per side.
template <typename KeyOfItem>
[[nodiscard]] ExactOperatorClasses classify_exact_operators(
    const std::size_t item_count, const KeyOfItem& key_of_item,
    const ExactReuseGate gate = {})
{
    using Key = std::decay_t<decltype(key_of_item(std::size_t{0}))>;

    // `representative` stores item indices and `class_of_item` stores class
    // numbers, both as `std::uint32_t`; the largest value either can hold is
    // `item_count - 1`.  Abandon rather than narrow: widening the vectors
    // instead would double `class_of_item`, which holds one entry per item
    // and is already the largest transient allocation of a big build.
    if (item_count != 0 &&
        item_count - 1 > std::numeric_limits<std::uint32_t>::max()) {
        return {};
    }

    ExactOperatorClasses result;
    result.class_of_item.resize(item_count);
    std::unordered_map<Key, std::uint32_t, ExactOperatorKeyHash> classes;
    for (std::size_t index = 0; index < item_count; ++index) {
        const auto [entry, inserted] = classes.try_emplace(
            key_of_item(index),
            static_cast<std::uint32_t>(result.representative.size()));
        if (inserted) {
            if (result.representative.size() >= gate.max_classes) {
                return {};
            }
            result.representative.push_back(static_cast<std::uint32_t>(index));
        }
        result.class_of_item[index] = entry->second;

        const std::size_t sampled = index + 1;
        if (sampled == gate.sample_items && sampled < item_count &&
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
