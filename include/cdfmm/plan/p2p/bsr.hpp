// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <vector>

#include "cdfmm/plan/p2p/canonical.hpp"
#include "cdfmm/plan/p2p/memory.hpp"

namespace cdfmm {

/**
 * @brief Backend-neutral FP64 BSR(3) packing with fixed identity policy.
 *
 * The plan contains only ordinary block-row data; it deliberately owns no
 * cuSPARSE, CUDA, or other vendor resource.
 */
struct StaticP2PBsrPlan {
    int source_count{0};
    int target_count{0};
    std::vector<int> row_offsets{};
    std::vector<int> source_indices{};
    std::vector<double> values{};
    std::vector<int> target_source_indices{};

    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/** @brief Backend-neutral BSR(3) packing at FP32 execution precision. */
struct FloatStaticP2PBsrPlan {
    int source_count{0};
    int target_count{0};
    std::vector<int> row_offsets{};
    std::vector<int> source_indices{};
    std::vector<float> values{};
    std::vector<int> target_source_indices{};

    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

[[nodiscard]] StaticP2PBsrPlan build_static_p2p_bsr_plan(
    const StaticP2POperator& operator_map,
    std::span<const int> target_source_indices = {});

[[nodiscard]] FloatStaticP2PBsrPlan build_static_p2p_bsr_plan(
    const FloatStaticP2POperator& operator_map,
    std::span<const int> target_source_indices = {});

} // namespace cdfmm
