// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "cdfmm/plan/p2p/leaf.hpp"

namespace cdfmm {

/**
 * @brief Source-major signed Tensor6 dictionary for FP64 CPU execution.
 *
 * The exact near field as dense leaf blocks whose tensors are replaced by
 * tokens into a dictionary of distinct, already-signed six-component
 * variants.  Within a block the tokens are source-major: the `target_count`
 * tokens of one source are contiguous at
 * `tensor_offset + local_source * target_count`.  Exactly one of the three
 * token vectors is populated, selected by `token_width_bytes` (the narrowest
 * width that indexes every variant).  A fixed point self pair is encoded as
 * `zero_variant_id`, so the executors need no identity map.
 */
struct StaticP2PSignedTensorDictionaryPlan {
    int source_count{0};                        ///< Number of sorted sources.
    int target_count{0};                        ///< Number of sorted targets.
    std::vector<int> target_begins{};           ///< First sorted target of each target leaf.
    std::vector<int> target_counts{};           ///< Number of targets in each target leaf.
    std::vector<int> leaf_row_offsets{};        ///< Range of `blocks` per target leaf (CSR offsets).
    std::vector<StaticP2PLeafBlock> blocks{};   ///< Dense source-leaf blocks in canonical order.
    std::vector<int> tile_leaf_indices{};       ///< Target leaf of each execution tile.
    std::vector<int> tile_target_offsets{};     ///< First local target of each execution tile.
    std::array<std::vector<double>, 6> tensors{};  ///< Signed variants, one array per component xx, xy, xz, yy, yz, zz.
    std::vector<std::uint8_t> tokens8{};        ///< Variant ids when `token_width_bytes == 1`.
    std::vector<std::uint16_t> tokens16{};      ///< Variant ids when `token_width_bytes == 2`.
    std::vector<std::uint32_t> tokens32{};      ///< Variant ids when `token_width_bytes == 4`.
    std::uint8_t token_width_bytes{1};          ///< Width of the populated token vector: 1, 2 or 4.
    std::uint32_t zero_variant_id{0};           ///< Variant that encodes an omitted point self pair.
    int target_tile_size{32};                   ///< Targets per execution tile, 1..128.

    /// Number of distinct signed tensors in the dictionary.
    [[nodiscard]] std::size_t variant_count() const noexcept
    {
        return tensors[0].size();
    }

    /// Number of stored tokens, one per (target, source) pair of every block.
    [[nodiscard]] std::size_t token_count() const noexcept
    {
        return tokens8.size() + tokens16.size() + tokens32.size();
    }

    /// Retained storage split by category.
    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/**
 * @brief Source-major signed Tensor6 dictionary at FP32 precision.
 *
 * Same layout and semantics as StaticP2PSignedTensorDictionaryPlan with
 * single-precision variants.
 */
struct FloatStaticP2PSignedTensorDictionaryPlan {
    int source_count{0};                        ///< Number of sorted sources.
    int target_count{0};                        ///< Number of sorted targets.
    std::vector<int> target_begins{};           ///< First sorted target of each target leaf.
    std::vector<int> target_counts{};           ///< Number of targets in each target leaf.
    std::vector<int> leaf_row_offsets{};        ///< Range of `blocks` per target leaf (CSR offsets).
    std::vector<StaticP2PLeafBlock> blocks{};   ///< Dense source-leaf blocks in canonical order.
    std::vector<int> tile_leaf_indices{};       ///< Target leaf of each execution tile.
    std::vector<int> tile_target_offsets{};     ///< First local target of each execution tile.
    std::array<std::vector<float>, 6> tensors{};   ///< Signed variants, one array per component xx, xy, xz, yy, yz, zz.
    std::vector<std::uint8_t> tokens8{};        ///< Variant ids when `token_width_bytes == 1`.
    std::vector<std::uint16_t> tokens16{};      ///< Variant ids when `token_width_bytes == 2`.
    std::vector<std::uint32_t> tokens32{};      ///< Variant ids when `token_width_bytes == 4`.
    std::uint8_t token_width_bytes{1};          ///< Width of the populated token vector: 1, 2 or 4.
    std::uint32_t zero_variant_id{0};           ///< Variant that encodes an omitted point self pair.
    int target_tile_size{32};                   ///< Targets per execution tile, 1..128.

    /// Number of distinct signed tensors in the dictionary.
    [[nodiscard]] std::size_t variant_count() const noexcept
    {
        return tensors[0].size();
    }

    /// Number of stored tokens, one per (target, source) pair of every block.
    [[nodiscard]] std::size_t token_count() const noexcept
    {
        return tokens8.size() + tokens16.size() + tokens32.size();
    }

    /// Retained storage split by category.
    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/**
 * @brief Derives the signed dictionary from the canonical rows.
 *
 * `leaf_pairs` must cover every canonical block exactly once in canonical
 * order.  With a fixed identity map (`target_source_indices`), each point
 * self pair is encoded as the zero variant; without one, point sources are
 * rejected because the executors carry no identity handling.
 */
[[nodiscard]] StaticP2PSignedTensorDictionaryPlan
build_static_p2p_signed_tensor_dictionary_plan(
    const StaticP2POperator& operator_map,
    std::span<const StaticP2PLeafPair> leaf_pairs,
    std::span<const int> target_source_indices = {},
    int target_tile_size = 32);

} // namespace cdfmm
