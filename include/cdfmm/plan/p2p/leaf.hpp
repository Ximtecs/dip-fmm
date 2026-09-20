// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "cdfmm/plan/p2p/canonical.hpp"
#include "cdfmm/plan/p2p/memory.hpp"

namespace cdfmm {

/**
 * @brief Particle ranges defining one dense target/source leaf pair.
 *
 * A periodic plan lists the same (target leaf, source leaf) ranges once per
 * image. The canonical rows keep the images of one source adjacent, ordered
 * by their shift, so a pair identifies its image by `image_ordinal` among the
 * `image_count` records sharing its ranges; free-space pairs use 0 and 1.
 */
struct StaticP2PLeafPair {
    int target_begin{0};    ///< First sorted target of the target leaf.
    int target_count{0};    ///< Number of targets in the target leaf.
    int source_begin{0};    ///< First sorted source of the source leaf.
    int source_count{0};    ///< Number of sources in the source leaf.
    int image_ordinal{0};   ///< Position of this image among the records with the same ranges.
    int image_count{1};     ///< Number of images sharing these ranges.
};

/**
 * @brief Metadata for one dense target/source tensor block.
 *
 * `skip_for_identity` carries the canonical identity policy of the block's
 * interactions: non-zero means a target whose identity map names one of the
 * block's sources omits that (singular point-dipole) pair, zero means every
 * stored tensor is applied, including a physical finite-body self tensor.
 * Executors obey this flag; they never infer identity handling from geometry.
 */
struct StaticP2PLeafBlock {
    int source_begin{0};          ///< First sorted source of the block.
    int source_count{0};          ///< Number of sources in the block.
    std::size_t tensor_offset{0}; ///< Index of the block's first tensor in each component array.
    int skip_for_identity{0};     ///< Non-zero when an identity match omits the pair.
};

/**
 * @brief Dense leaf-pair packing derived from canonical FP64 target rows.
 *
 * Within a block the tensors are target-major: the `source_count` tensors of
 * one local target are contiguous at
 * `tensor_offset + local_target * source_count`.  The occupancy statistics
 * describe the target leaves and feed the execution policy.
 */
struct StaticP2PLeafPlan {
    int source_count{0};                        ///< Number of sorted sources.
    int target_count{0};                        ///< Number of sorted targets.
    std::vector<int> target_begins{};           ///< First sorted target of each target leaf.
    std::vector<int> target_counts{};           ///< Number of targets in each target leaf.
    std::vector<int> leaf_row_offsets{};        ///< Range of `blocks` per target leaf (CSR offsets).
    std::vector<StaticP2PLeafBlock> blocks{};   ///< Dense source-leaf blocks in canonical order.
    std::array<std::vector<double>, 6> tensors{};  ///< One array per component xx, xy, xz, yy, yz, zz.
    int minimum_occupancy{0};                   ///< Fewest targets in any target leaf.
    int maximum_occupancy{0};                   ///< Most targets in any target leaf.
    double mean_occupancy{0.0};                 ///< Mean targets per target leaf.
    int unique_occupancies{0};                  ///< Number of distinct target-leaf occupancies.
    bool uniform_occupancy{false};              ///< Whether every target leaf holds the same number of targets.

    /// Retained storage split by category.
    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/** @brief Dense leaf-pair packing at FP32 execution precision. */
struct FloatStaticP2PLeafPlan {
    int source_count{0};                        ///< Number of sorted sources.
    int target_count{0};                        ///< Number of sorted targets.
    std::vector<int> target_begins{};           ///< First sorted target of each target leaf.
    std::vector<int> target_counts{};           ///< Number of targets in each target leaf.
    std::vector<int> leaf_row_offsets{};        ///< Range of `blocks` per target leaf (CSR offsets).
    std::vector<StaticP2PLeafBlock> blocks{};   ///< Dense source-leaf blocks in canonical order.
    std::array<std::vector<float>, 6> tensors{};   ///< One array per component xx, xy, xz, yy, yz, zz.
    int minimum_occupancy{0};                   ///< Fewest targets in any target leaf.
    int maximum_occupancy{0};                   ///< Most targets in any target leaf.
    double mean_occupancy{0.0};                 ///< Mean targets per target leaf.
    int unique_occupancies{0};                  ///< Number of distinct target-leaf occupancies.
    bool uniform_occupancy{false};              ///< Whether every target leaf holds the same number of targets.

    /// Retained storage split by category.
    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/**
 * @brief Deterministically packs dense leaf rectangles from canonical rows.
 *
 * The supplied rectangles must cover every canonical block exactly once and
 * must preserve the canonical interaction order. Periodic image records are
 * ordinary rectangles carrying their image ordinal; the packing itself does
 * not distinguish a primary-cell block from an image block.
 */
[[nodiscard]] StaticP2PLeafPlan build_static_p2p_leaf_plan(
    const StaticP2POperator& operator_map,
    std::span<const StaticP2PLeafPair> leaf_pairs);

} // namespace cdfmm
