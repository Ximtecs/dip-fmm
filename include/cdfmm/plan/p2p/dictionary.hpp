// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "cdfmm/plan/p2p/leaf.hpp"

namespace cdfmm {

/**
 * @brief Exact-bit magnitude/sign Tensor6 dictionary derived from FP64 rows.
 *
 * Identity handling is carried per dense block (`StaticP2PLeafBlock::
 * skip_for_identity`); `skip_for_identity` here only records whether any block
 * omits identity pairs.
 *
 * Tokens are target-major within a block (`tensor_offset + local_target *
 * source_count`) and pack a dictionary id with a six-bit sign mask
 * (`pack_tensor6_token`); the dictionary stores component magnitudes only.
 */
struct StaticP2PTensorDictionaryPlan {
    int source_count{0};                        ///< Number of sorted sources.
    int target_count{0};                        ///< Number of sorted targets.
    std::vector<int> target_begins{};           ///< First sorted target of each target leaf.
    std::vector<int> target_counts{};           ///< Number of targets in each target leaf.
    std::vector<int> leaf_row_offsets{};        ///< Range of `blocks` per target leaf (CSR offsets).
    std::vector<StaticP2PLeafBlock> blocks{};   ///< Dense source-leaf blocks in canonical order.
    std::array<std::vector<double>, 6> tensors{};  ///< Component magnitudes of each distinct tensor, xx, xy, xz, yy, yz, zz.
    std::vector<std::uint32_t> tokens{};        ///< Packed (id, sign mask) token per pair.
    bool skip_for_identity{false};              ///< Whether any block omits identity pairs.

    /// Retained storage split by category.
    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/** @brief Exact-bit magnitude/sign Tensor6 dictionary at FP32 precision. */
struct FloatStaticP2PTensorDictionaryPlan {
    int source_count{0};                        ///< Number of sorted sources.
    int target_count{0};                        ///< Number of sorted targets.
    std::vector<int> target_begins{};           ///< First sorted target of each target leaf.
    std::vector<int> target_counts{};           ///< Number of targets in each target leaf.
    std::vector<int> leaf_row_offsets{};        ///< Range of `blocks` per target leaf (CSR offsets).
    std::vector<StaticP2PLeafBlock> blocks{};   ///< Dense source-leaf blocks in canonical order.
    std::array<std::vector<float>, 6> tensors{};   ///< Component magnitudes of each distinct tensor, xx, xy, xz, yy, yz, zz.
    std::vector<std::uint32_t> tokens{};        ///< Packed (id, sign mask) token per pair.
    bool skip_for_identity{false};              ///< Whether any block omits identity pairs.

    /// Retained storage split by category.
    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/**
 * @brief Derives the magnitude/sign dictionary from the canonical rows.
 *
 * `leaf_pairs` must cover every canonical block exactly once in canonical
 * order.
 */
[[nodiscard]] StaticP2PTensorDictionaryPlan
build_static_p2p_tensor_dictionary_plan(
    const StaticP2POperator& operator_map,
    std::span<const StaticP2PLeafPair> leaf_pairs);

} // namespace cdfmm
