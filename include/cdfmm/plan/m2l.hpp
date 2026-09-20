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
 *
 * One interaction `i` in target `t`'s row applies
 * `L_t += S_L[level_t] * M_{matrix_ids[i]} * (S_M[level_s] * M_{source_nodes[i]})`,
 * with `S_M` and `S_L` the per-level scaling tables that restore the physical
 * box width to the dimensionless matrices.
 */
struct StaticM2LPlan {
    int coefficient_count{0};                ///< Coefficients per expansion, C.
    int matrix_count{0};                     ///< Number of distinct transfer-class matrices (plus one periodic root matrix when present).
    int level_count{0};                      ///< Number of tree levels, `maximum_level + 1`.
    std::vector<double> matrices{};          ///< `matrix_count` column-major C x C matrices, class-major.
    std::vector<double> multipole_scaling{}; ///< Source scaling `[level][coefficient]`.
    std::vector<double> local_scaling{};     ///< Target scaling `[level][coefficient]`.
    std::vector<int> target_row_offsets{};   ///< Range of interactions per target node (CSR offsets, one entry per node plus one).
    std::vector<int> source_nodes{};         ///< Source node of each interaction.
    std::vector<int> matrix_ids{};           ///< Transfer-class matrix of each interaction.
    std::vector<int> source_levels{};        ///< Level of each interaction's source node.
    std::vector<int> target_levels{};        ///< Level of each interaction's target node.
    std::vector<int> interaction_levels{};   ///< Level the interaction is scheduled at (the target level).
    std::vector<int> level_target_begin{};   ///< Legacy per-level first target index; superseded by `target_level_offsets`.
    std::vector<int> level_target_end{};     ///< Legacy per-level end target index; superseded by `target_level_offsets`.
    std::vector<int> target_level_offsets{}; ///< Range of `target_nodes_by_level` per level (CSR offsets).
    std::vector<int> target_nodes_by_level{};///< Node ids grouped by level, the executors' level schedule.
    std::vector<int> node_levels{};          ///< Level of every node, indexed by node id.
};

/** @brief FP32 M2L data with the same ordering and schedule as the FP64 plan. */
struct FloatStaticM2LPlan {
    int coefficient_count{0};                ///< Coefficients per expansion, C.
    int matrix_count{0};                     ///< Number of distinct transfer-class matrices (plus one periodic root matrix when present).
    int level_count{0};                      ///< Number of tree levels, `maximum_level + 1`.
    std::vector<float> matrices{};           ///< `matrix_count` column-major C x C matrices, class-major.
    std::vector<float> multipole_scaling{};  ///< Source scaling `[level][coefficient]`.
    std::vector<float> local_scaling{};      ///< Target scaling `[level][coefficient]`.
    std::vector<int> target_row_offsets{};   ///< Range of interactions per target node (CSR offsets, one entry per node plus one).
    std::vector<int> source_nodes{};         ///< Source node of each interaction.
    std::vector<int> matrix_ids{};           ///< Transfer-class matrix of each interaction.
    std::vector<int> source_levels{};        ///< Level of each interaction's source node.
    std::vector<int> target_levels{};        ///< Level of each interaction's target node.
    std::vector<int> interaction_levels{};   ///< Level the interaction is scheduled at (the target level).
    std::vector<int> level_target_begin{};   ///< Legacy per-level first target index; superseded by `target_level_offsets`.
    std::vector<int> level_target_end{};     ///< Legacy per-level end target index; superseded by `target_level_offsets`.
    std::vector<int> target_level_offsets{}; ///< Range of `target_nodes_by_level` per level (CSR offsets).
    std::vector<int> target_nodes_by_level{};///< Node ids grouped by level, the executors' level schedule.
    std::vector<int> node_levels{};          ///< Level of every node, indexed by node id.
};

} // namespace cdfmm
