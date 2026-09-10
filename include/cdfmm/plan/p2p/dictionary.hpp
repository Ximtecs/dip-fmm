// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "cdfmm/plan/p2p/leaf.hpp"

namespace cdfmm {

/** @brief Exact-bit magnitude/sign Tensor6 dictionary derived from FP64 rows. */
struct StaticP2PTensorDictionaryPlan {
    int source_count{0};
    int target_count{0};
    std::vector<int> target_begins{};
    std::vector<int> target_counts{};
    std::vector<int> leaf_row_offsets{};
    std::vector<StaticP2PLeafBlock> blocks{};
    std::array<std::vector<double>, 6> tensors{};
    std::vector<std::uint32_t> tokens{};
    bool skip_for_identity{false};

    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/** @brief Exact-bit magnitude/sign Tensor6 dictionary at FP32 precision. */
struct FloatStaticP2PTensorDictionaryPlan {
    int source_count{0};
    int target_count{0};
    std::vector<int> target_begins{};
    std::vector<int> target_counts{};
    std::vector<int> leaf_row_offsets{};
    std::vector<StaticP2PLeafBlock> blocks{};
    std::array<std::vector<float>, 6> tensors{};
    std::vector<std::uint32_t> tokens{};
    bool skip_for_identity{false};

    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

[[nodiscard]] StaticP2PTensorDictionaryPlan
build_static_p2p_tensor_dictionary_plan(
    const StaticP2POperator& operator_map,
    std::span<const StaticP2PLeafPair> leaf_pairs);

} // namespace cdfmm
