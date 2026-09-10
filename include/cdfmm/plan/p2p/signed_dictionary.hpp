// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "cdfmm/plan/p2p/leaf.hpp"

namespace cdfmm {

/** @brief Source-major signed Tensor6 dictionary for FP64 CPU execution. */
struct StaticP2PSignedTensorDictionaryPlan {
    int source_count{0};
    int target_count{0};
    std::vector<int> target_begins{};
    std::vector<int> target_counts{};
    std::vector<int> leaf_row_offsets{};
    std::vector<StaticP2PLeafBlock> blocks{};
    std::vector<int> tile_leaf_indices{};
    std::vector<int> tile_target_offsets{};
    std::array<std::vector<double>, 6> tensors{};
    std::vector<std::uint8_t> tokens8{};
    std::vector<std::uint16_t> tokens16{};
    std::vector<std::uint32_t> tokens32{};
    std::uint8_t token_width_bytes{1};
    std::uint32_t zero_variant_id{0};
    int target_tile_size{32};

    [[nodiscard]] std::size_t variant_count() const noexcept
    {
        return tensors[0].size();
    }

    [[nodiscard]] std::size_t token_count() const noexcept
    {
        return tokens8.size() + tokens16.size() + tokens32.size();
    }

    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/** @brief Source-major signed Tensor6 dictionary at FP32 precision. */
struct FloatStaticP2PSignedTensorDictionaryPlan {
    int source_count{0};
    int target_count{0};
    std::vector<int> target_begins{};
    std::vector<int> target_counts{};
    std::vector<int> leaf_row_offsets{};
    std::vector<StaticP2PLeafBlock> blocks{};
    std::vector<int> tile_leaf_indices{};
    std::vector<int> tile_target_offsets{};
    std::array<std::vector<float>, 6> tensors{};
    std::vector<std::uint8_t> tokens8{};
    std::vector<std::uint16_t> tokens16{};
    std::vector<std::uint32_t> tokens32{};
    std::uint8_t token_width_bytes{1};
    std::uint32_t zero_variant_id{0};
    int target_tile_size{32};

    [[nodiscard]] std::size_t variant_count() const noexcept
    {
        return tensors[0].size();
    }

    [[nodiscard]] std::size_t token_count() const noexcept
    {
        return tokens8.size() + tokens16.size() + tokens32.size();
    }

    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

[[nodiscard]] StaticP2PSignedTensorDictionaryPlan
build_static_p2p_signed_tensor_dictionary_plan(
    const StaticP2POperator& operator_map,
    std::span<const StaticP2PLeafPair> leaf_pairs,
    std::span<const int> target_source_indices = {},
    int target_tile_size = 32);

} // namespace cdfmm
