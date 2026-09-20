// SPDX-License-Identifier: Apache-2.0
//
// Portable execution of one level of the prepared static M2L directly from
// the canonical target rows.  For a target t at this level:
//
//   L_t += S_L(level) * sum_{(s, class) in row(t)} M_class * (S_M(level_s) * M_s)
//
// where M_class is the normalised C x C matrix of the interaction's transfer
// class and S_M, S_L are the per-level scaling tables that make the bank
// depth independent.  The block-scheduled executor in schedule.cpp applies
// the same terms in a cache-friendlier order and is preferred when it fits.

#include "cdfmm/backend/cpu/m2l.hpp"

#include <cstddef>

namespace cdfmm {

namespace {

// Largest coefficient count served by the register/stack path below; larger
// plans (spherical order above 21) fall back to the per-output kernel.
constexpr int max_stack_coefficients = 512;

// Per-target M2L: one target's local row is accumulated in a stack buffer
// over its canonical interaction row.  Each interaction scales the source
// multipole once, then streams its C x C matrix exactly once as C unit-stride
// axpy updates over beta (the matrices are column-major with beta
// contiguous), so the inner loop vectorises and the matrix traffic per
// interaction is C^2 values instead of C^3 strided reads.  Interactions are
// visited in canonical row order and the local scaling is applied once per
// target, so the summation order over interactions is unchanged; only the
// product association within one term differs from the per-output kernel.
template <typename Plan, typename Scalar>
void apply_m2l_target_rows(const Plan &plan, const int level,
                           const std::span<const Scalar> multipoles,
                           const std::span<Scalar> locals) {
    const int n = plan.coefficient_count;
    const bool explicit_targets = !plan.target_level_offsets.empty();
    const int target_begin = explicit_targets
        ? plan.target_level_offsets[static_cast<std::size_t>(level)]
        : plan.level_target_begin[static_cast<std::size_t>(level)];
    const int target_end = explicit_targets
        ? plan.target_level_offsets[static_cast<std::size_t>(level + 1)]
        : plan.level_target_end[static_cast<std::size_t>(level)];
    const Scalar *local_scale =
        plan.local_scaling.data() + static_cast<std::size_t>(level) * n;
    const int target_count = target_end - target_begin;
#pragma omp parallel for schedule(dynamic, 8) if (target_count >= 16)
    for (int slot = 0; slot < target_count; ++slot) {
        const int target = explicit_targets
            ? plan.target_nodes_by_level[
                  static_cast<std::size_t>(target_begin + slot)]
            : target_begin + slot;
        alignas(64) Scalar accumulator[max_stack_coefficients];
        alignas(64) Scalar scaled_source[max_stack_coefficients];
        for (int beta = 0; beta < n; ++beta) {
            accumulator[beta] = Scalar{0};
        }
        const int row_begin = plan.target_row_offsets[target];
        const int row_end = plan.target_row_offsets[target + 1];
        for (int interaction = row_begin; interaction < row_end;
             ++interaction) {
            const std::size_t slot_index = static_cast<std::size_t>(interaction);
            const int target_level = plan.target_levels.empty()
                ? level : plan.target_levels[slot_index];
            if (target_level != level) {
                continue;
            }
            const int source_level = plan.source_levels.empty()
                ? level : plan.source_levels[slot_index];
            const Scalar *source_scale = plan.multipole_scaling.data() +
                static_cast<std::size_t>(source_level) * n;
            const Scalar *source_M = multipoles.data() +
                static_cast<std::size_t>(plan.source_nodes[slot_index]) * n;
            for (int alpha = 0; alpha < n; ++alpha) {
                scaled_source[alpha] = source_scale[alpha] * source_M[alpha];
            }
            const Scalar *matrix = plan.matrices.data() +
                static_cast<std::size_t>(plan.matrix_ids[slot_index]) * n * n;
            for (int alpha = 0; alpha < n; ++alpha) {
                const Scalar x = scaled_source[alpha];
                const Scalar *column =
                    matrix + static_cast<std::size_t>(alpha) * n;
#pragma omp simd
                for (int beta = 0; beta < n; ++beta) {
                    accumulator[beta] += column[beta] * x;
                }
            }
        }
        Scalar *L = locals.data() + static_cast<std::size_t>(target) * n;
        for (int beta = 0; beta < n; ++beta) {
            L[beta] += local_scale[beta] * accumulator[beta];
        }
    }
}

// Reference per-output kernel: one iteration owns one (target, beta) output
// and walks the target's interaction row with stride-n matrix reads.  Kept as
// the fallback for coefficient counts above the stack limit.
template <typename Plan, typename Scalar>
void apply_m2l_outputs(const Plan &plan, const int level,
                       const std::span<const Scalar> multipoles,
                       const std::span<Scalar> locals) {
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
        Scalar value = Scalar{0};
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
            const Scalar *source_scale = plan.multipole_scaling.data() +
                static_cast<std::size_t>(source_level) * n;
            const Scalar *matrix_column =
                plan.matrices.data() +
                static_cast<std::size_t>(matrix_id) * n * n + beta;
            const Scalar *source_M =
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

template <typename Plan, typename Scalar>
void apply_m2l_level(const Plan &plan, const int level,
                     const std::span<const Scalar> multipoles,
                     const std::span<Scalar> locals) {
    if (plan.coefficient_count <= max_stack_coefficients) {
        apply_m2l_target_rows(plan, level, multipoles, locals);
    } else {
        apply_m2l_outputs(plan, level, multipoles, locals);
    }
}

} // namespace

void apply_static_m2l_plan(const StaticM2LPlan &plan, const int level,
                           const std::span<const double> multipoles,
                           const std::span<double> locals) {
    apply_m2l_level(plan, level, multipoles, locals);
}

void apply_static_m2l_plan(const FloatStaticM2LPlan &plan, const int level,
                           const std::span<const float> multipoles,
                           const std::span<float> locals) {
    apply_m2l_level(plan, level, multipoles, locals);
}

// Compatibility overload over per-node coefficient vectors, retained for the
// public flat API and its tests; production uses the flat node-major arrays.
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
