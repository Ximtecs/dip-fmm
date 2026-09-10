// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

namespace cdfmm {

/**
 * @brief Canonical immutable execution plan for normalised static M2L.
 *
 * Matrices are column-major and level-independent. Scaling is laid out as
 * `[level][coefficient]`; parallel interaction arrays retain source and target
 * levels explicitly. The plan contains no mutable coefficients or backend
 * resources.
 */
struct StaticM2LPlan {
    int coefficient_count{0};
    int matrix_count{0};
    int level_count{0};
    std::vector<double> matrices{};
    std::vector<double> multipole_scaling{};
    std::vector<double> local_scaling{};
    std::vector<int> target_row_offsets{};
    std::vector<int> source_nodes{};
    std::vector<int> matrix_ids{};
    std::vector<int> source_levels{};
    std::vector<int> target_levels{};
    std::vector<int> interaction_levels{};
    std::vector<int> level_target_begin{};
    std::vector<int> level_target_end{};
    std::vector<int> target_level_offsets{};
    std::vector<int> target_nodes_by_level{};
    std::vector<int> node_levels{};
};

/** @brief FP32 M2L data with the same ordering and schedule as the FP64 plan. */
struct FloatStaticM2LPlan {
    int coefficient_count{0};
    int matrix_count{0};
    int level_count{0};
    std::vector<float> matrices{};
    std::vector<float> multipole_scaling{};
    std::vector<float> local_scaling{};
    std::vector<int> target_row_offsets{};
    std::vector<int> source_nodes{};
    std::vector<int> matrix_ids{};
    std::vector<int> source_levels{};
    std::vector<int> target_levels{};
    std::vector<int> interaction_levels{};
    std::vector<int> level_target_begin{};
    std::vector<int> level_target_end{};
    std::vector<int> target_level_offsets{};
    std::vector<int> target_nodes_by_level{};
    std::vector<int> node_levels{};
};

} // namespace cdfmm
