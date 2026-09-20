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
 * cuSPARSE, CUDA, or other vendor resource.  Each pair is one full row-major
 * 3x3 block, and the identity map it was built for is applied at
 * construction: omitted self pairs are simply absent, so executors take no
 * identity array and callers must pass the same map on every evaluation.
 */
struct StaticP2PBsrPlan {
    int source_count{0};                        ///< Number of sorted sources (block columns).
    int target_count{0};                        ///< Number of sorted targets (block rows).
    std::vector<int> row_offsets{};             ///< Range of blocks per target (CSR offsets, `target_count + 1` entries).
    std::vector<int> source_indices{};          ///< Sorted source index (block column) of each block.
    std::vector<double> values{};               ///< Nine row-major values per block.
    std::vector<int> target_source_indices{};   ///< The sorted identity map the plan was built for, one entry per target (-1 for none).

    /// Retained storage split by category.
    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/** @brief Backend-neutral BSR(3) packing at FP32 execution precision. */
struct FloatStaticP2PBsrPlan {
    int source_count{0};                        ///< Number of sorted sources (block columns).
    int target_count{0};                        ///< Number of sorted targets (block rows).
    std::vector<int> row_offsets{};             ///< Range of blocks per target (CSR offsets, `target_count + 1` entries).
    std::vector<int> source_indices{};          ///< Sorted source index (block column) of each block.
    std::vector<float> values{};                ///< Nine row-major values per block.
    std::vector<int> target_source_indices{};   ///< The sorted identity map the plan was built for, one entry per target (-1 for none).

    /// Retained storage split by category.
    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/** @brief Derives the FP64 BSR(3) packing for one fixed identity map. */
[[nodiscard]] StaticP2PBsrPlan build_static_p2p_bsr_plan(
    const StaticP2POperator& operator_map,
    std::span<const int> target_source_indices = {});

/** @brief Derives the FP32 BSR(3) packing for one fixed identity map. */
[[nodiscard]] FloatStaticP2PBsrPlan build_static_p2p_bsr_plan(
    const FloatStaticP2POperator& operator_map,
    std::span<const int> target_source_indices = {});

} // namespace cdfmm
