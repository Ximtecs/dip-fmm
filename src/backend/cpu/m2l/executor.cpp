// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/backend/cpu/m2l.hpp"

#include <cstddef>

namespace cdfmm {

void apply_static_m2l_plan(const StaticM2LPlan &plan, const int level,
    const std::span<const double> multipoles,
                           const std::span<double> locals) {
    const int n = plan.coefficient_count;
    const bool explicit_targets = !plan.target_level_offsets.empty();
    const int target_begin = explicit_targets
        ? plan.target_level_offsets[static_cast<std::size_t>(level)]
        : plan.level_target_begin[static_cast<std::size_t>(level)];
    const int target_end = explicit_targets
        ? plan.target_level_offsets[static_cast<std::size_t>(level + 1)]
        : plan.level_target_end[static_cast<std::size_t>(level)];

    // The target schedule owns one output row at a time.  Iterating only this
    // target level avoids revisiting other rows; one iteration owns one target
    // coefficient and therefore needs no atomic accumulation.
    const std::ptrdiff_t output_count =
        static_cast<std::ptrdiff_t>(target_end - target_begin) * n;
#pragma omp parallel for schedule(static) if (output_count >= 256)
    for (std::ptrdiff_t output = 0; output < output_count; ++output) {
        const int target = explicit_targets
            ? plan.target_nodes_by_level[static_cast<std::size_t>(
                target_begin + static_cast<int>(output / n))]
            : target_begin + static_cast<int>(output / n);
        const int beta = static_cast<int>(output % n);
        double value = 0.0;
        const int row_begin = plan.target_row_offsets[target];
        const int row_end = plan.target_row_offsets[target + 1];
        for (int interaction = row_begin; interaction < row_end; ++interaction) {
            const int source = plan.source_nodes[interaction];
            const int matrix_id = plan.matrix_ids[interaction];
            const int target_level = plan.target_levels.empty()
                ? level : plan.target_levels[interaction];
            if (target_level != level) {
                continue;
            }
            const int source_level = plan.source_levels.empty()
                ? level : plan.source_levels[interaction];
            const double *source_scale = plan.multipole_scaling.data() +
                static_cast<std::size_t>(source_level) * n;
            const double *matrix_column =
                plan.matrices.data() +
                static_cast<std::size_t>(matrix_id) * n * n + beta;
            const double *source_M =
                multipoles.data() + static_cast<std::size_t>(source) * n;
            for (int alpha = 0; alpha < n; ++alpha) {
                value += matrix_column[static_cast<std::size_t>(alpha) * n] *
                    source_scale[alpha] * source_M[alpha];
            }
        }
        locals[static_cast<std::size_t>(target) * n + beta] +=
            plan.local_scaling[static_cast<std::size_t>(level) * n + beta] * value;
    }
}

void apply_static_m2l_plan(const FloatStaticM2LPlan &plan, const int level,
    const std::span<const float> multipoles,
                           const std::span<float> locals) {
    const int n = plan.coefficient_count;
    const bool explicit_targets = !plan.target_level_offsets.empty();
    const int target_begin = explicit_targets
        ? plan.target_level_offsets[static_cast<std::size_t>(level)]
        : plan.level_target_begin[static_cast<std::size_t>(level)];
    const int target_end = explicit_targets
        ? plan.target_level_offsets[static_cast<std::size_t>(level + 1)]
        : plan.level_target_end[static_cast<std::size_t>(level)];
    const std::ptrdiff_t output_count =
        static_cast<std::ptrdiff_t>(target_end - target_begin) * n;
#pragma omp parallel for schedule(static) if (output_count >= 256)
    for (std::ptrdiff_t output = 0; output < output_count; ++output) {
        const int target = explicit_targets
            ? plan.target_nodes_by_level[static_cast<std::size_t>(
                target_begin + static_cast<int>(output / n))]
            : target_begin + static_cast<int>(output / n);
        const int beta = static_cast<int>(output % n);
        float value = 0.0F;
        const int row_begin = plan.target_row_offsets[target];
        const int row_end = plan.target_row_offsets[target + 1];
        for (int interaction = row_begin; interaction < row_end;
             ++interaction) {
            const int source = plan.source_nodes[interaction];
            const int matrix_id = plan.matrix_ids[interaction];
            const int target_level = plan.target_levels.empty()
                ? level : plan.target_levels[interaction];
            if (target_level != level) {
                continue;
            }
            const int source_level = plan.source_levels.empty()
                ? level : plan.source_levels[interaction];
            const float* matrix_column = plan.matrices.data() +
                static_cast<std::size_t>(matrix_id) * n * n + beta;
            const float* source_M = multipoles.data() +
                static_cast<std::size_t>(source) * n;
            for (int alpha = 0; alpha < n; ++alpha) {
                value += matrix_column[static_cast<std::size_t>(alpha) * n] *
                    plan.multipole_scaling[static_cast<std::size_t>(source_level) * n + alpha] *
                    source_M[alpha];
            }
        }
        locals[static_cast<std::size_t>(target) * n + beta] +=
            plan.local_scaling[static_cast<std::size_t>(level) * n + beta] * value;
    }
}

void apply_static_m2l_plan(
    const StaticM2LPlan &plan, const int level,
    const std::span<const std::vector<double>> multipoles,
    const std::span<std::vector<double>> locals) {
    const int n = plan.coefficient_count;
    const std::ptrdiff_t output_count =
        static_cast<std::ptrdiff_t>(locals.size()) * n;
#pragma omp parallel for schedule(static) if (output_count >= 256)
    for (std::ptrdiff_t output = 0; output < output_count; ++output) {
        const int target = static_cast<int>(output / n);
        const int beta = static_cast<int>(output % n);
        double value = 0.0;
        for (int interaction = plan.target_row_offsets[target];
         interaction < plan.target_row_offsets[target + 1]; ++interaction) {
            const int target_level = plan.target_levels.empty()
                ? (plan.interaction_levels.empty() ? level
                   : plan.interaction_levels[interaction])
                : plan.target_levels[interaction];
            if (target_level != level) {
                continue;
            }
            const int source = plan.source_nodes[interaction];
            const int matrix_id = plan.matrix_ids[interaction];
            const int source_level = plan.source_levels.empty()
                ? level : plan.source_levels[interaction];
            const double local_scale = plan.local_scaling[
                static_cast<std::size_t>(target_level) * n + beta];
            for (int alpha = 0; alpha < n; ++alpha) {
                const std::size_t matrix_index =
                    (static_cast<std::size_t>(matrix_id) * n + alpha) * n + beta;
                value += local_scale * plan.matrices[matrix_index] *
                 plan.multipole_scaling[static_cast<std::size_t>(source_level) * n +
                                        alpha] *
                    multipoles[source][alpha];
            }
        }
        locals[target][beta] += value;
    }
}

} // namespace cdfmm
