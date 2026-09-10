// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "cdfmm/plan/p2p/canonical.hpp"
#include "cdfmm/plan/p2p/memory.hpp"

namespace cdfmm {

/** @brief Particle ranges defining one dense target/source leaf pair. */
struct StaticP2PLeafPair {
    int target_begin{0};
    int target_count{0};
    int source_begin{0};
    int source_count{0};
};

/** @brief Metadata for one dense target/source tensor block. */
struct StaticP2PLeafBlock {
    int source_begin{0};
    int source_count{0};
    std::size_t tensor_offset{0};
};

/** @brief Dense leaf-pair packing derived from canonical FP64 target rows. */
struct StaticP2PLeafPlan {
    int source_count{0};
    int target_count{0};
    std::vector<int> target_begins{};
    std::vector<int> target_counts{};
    std::vector<int> leaf_row_offsets{};
    std::vector<StaticP2PLeafBlock> blocks{};
    std::array<std::vector<double>, 6> tensors{};
    int minimum_occupancy{0};
    int maximum_occupancy{0};
    double mean_occupancy{0.0};
    int unique_occupancies{0};
    bool uniform_occupancy{false};

    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/** @brief Dense leaf-pair packing at FP32 execution precision. */
struct FloatStaticP2PLeafPlan {
    int source_count{0};
    int target_count{0};
    std::vector<int> target_begins{};
    std::vector<int> target_counts{};
    std::vector<int> leaf_row_offsets{};
    std::vector<StaticP2PLeafBlock> blocks{};
    std::array<std::vector<float>, 6> tensors{};
    int minimum_occupancy{0};
    int maximum_occupancy{0};
    double mean_occupancy{0.0};
    int unique_occupancies{0};
    bool uniform_occupancy{false};

    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/**
 * @brief Deterministically packs dense leaf rectangles from canonical rows.
 *
 * The supplied rectangles must cover every canonical block exactly once and
 * must preserve the canonical interaction order.
 */
[[nodiscard]] StaticP2PLeafPlan build_static_p2p_leaf_plan(
    const StaticP2POperator& operator_map,
    std::span<const StaticP2PLeafPair> leaf_pairs);

} // namespace cdfmm
