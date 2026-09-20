// SPDX-License-Identifier: Apache-2.0
//
// Transfer-class sorted block schedule for the portable static M2L.  The
// canonical plan stores M2L interactions as target rows; applying them row by
// row re-reads a different C x C matrix for almost every interaction.  This
// schedule regroups each level's targets into blocks whose accumulators fit
// one thread's stack, and within a block sorts the interactions by transfer
// class so that consecutive interactions reuse the same matrix from L1.  It
// is a pure reordering of the same terms: every target still receives every
// interaction of its canonical row exactly once, with the per-level scaling
// applied once at the end.  The schedule is built once at construction and
// holds only indices.

#include "schedule.hpp"

#include <algorithm>
#include <numeric>

#ifdef CDFMM_USE_OPENMP
#include <omp.h>
#endif

namespace cdfmm::detail::cpu {

namespace {

int omp_max_threads() {
#ifdef CDFMM_USE_OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

// Block accumulators live in one thread's stack: at most this many values.
constexpr int max_block_values = 4096;
constexpr int max_block_targets = 128;
constexpr int max_stack_coefficients = 512;

// Build the schedule for both plan precisions.  An empty schedule (returned
// when the coefficient count exceeds the stack budget) tells the caller to use
// the per-target-row executor instead.  Layout of the result, all CSR-style
// offsets: level -> blocks, block -> targets and runs, run -> interactions,
// with `run_matrix` naming the shared transfer class of each run.
template <typename Plan>
M2LBlockSchedule build_schedule(const Plan& plan) {
  M2LBlockSchedule schedule;
  const int n = plan.coefficient_count;
  schedule.coefficient_count = n;
  if (n <= 0 || n > max_stack_coefficients) {
    return schedule;
  }
  schedule.block_targets =
      std::clamp(max_block_values / n, 1, max_block_targets);
  // Every level should offer several blocks per thread, otherwise small
  // levels (and small plans) leave most of the team idle.
  const int minimum_blocks = 4 * std::max(1, omp_max_threads());
  const bool explicit_targets = !plan.target_level_offsets.empty();
  const int level_count = explicit_targets
      ? static_cast<int>(plan.target_level_offsets.size()) - 1
      : static_cast<int>(plan.level_target_begin.size());
  schedule.level_block_offsets.push_back(0);
  schedule.block_target_offsets.push_back(0);
  schedule.block_run_offsets.push_back(0);
  schedule.run_offsets.push_back(0);

  std::vector<int> order;
  for (int level = 0; level < level_count; ++level) {
    const int target_begin = explicit_targets
        ? plan.target_level_offsets[static_cast<std::size_t>(level)]
        : plan.level_target_begin[static_cast<std::size_t>(level)];
    const int target_end = explicit_targets
        ? plan.target_level_offsets[static_cast<std::size_t>(level + 1)]
        : plan.level_target_end[static_cast<std::size_t>(level)];
    const int level_targets = target_end - target_begin;
    const int level_block_targets = std::clamp(
        (level_targets + minimum_blocks - 1) / minimum_blocks, 1,
        schedule.block_targets);
    for (int block_begin = target_begin; block_begin < target_end;
         block_begin += level_block_targets) {
      const int block_end =
          std::min(block_begin + level_block_targets, target_end);
      const std::size_t first_interaction = schedule.interactions.size();
      for (int slot = block_begin; slot < block_end; ++slot) {
        const int target = explicit_targets
            ? plan.target_nodes_by_level[static_cast<std::size_t>(slot)]
            : slot;
        schedule.targets.push_back(target);
        for (int interaction = plan.target_row_offsets[target];
             interaction < plan.target_row_offsets[target + 1];
             ++interaction) {
          const auto index = static_cast<std::size_t>(interaction);
          const int target_level = plan.target_levels.empty()
              ? level : plan.target_levels[index];
          if (target_level != level) {
            continue;
          }
          schedule.interactions.push_back(
              {plan.source_nodes[index], slot - block_begin,
               plan.source_levels.empty() ? level : plan.source_levels[index]});
          order.push_back(interaction);
        }
      }
      // Stable sort of this block's interactions by transfer class; ties keep
      // canonical (target row) order.
      const std::size_t count = schedule.interactions.size() - first_interaction;
      std::vector<std::size_t> permutation(count);
      std::iota(permutation.begin(), permutation.end(), std::size_t{0});
      std::stable_sort(permutation.begin(), permutation.end(),
                       [&](const std::size_t a, const std::size_t b) {
                         return plan.matrix_ids[static_cast<std::size_t>(
                                    order[a])] <
                                plan.matrix_ids[static_cast<std::size_t>(
                                    order[b])];
                       });
      std::vector<M2LBlockSchedule::Interaction> sorted(count);
      for (std::size_t position = 0; position < count; ++position) {
        sorted[position] =
            schedule.interactions[first_interaction + permutation[position]];
      }
      std::copy(sorted.begin(), sorted.end(),
                schedule.interactions.begin() +
                    static_cast<std::ptrdiff_t>(first_interaction));
      int previous_matrix = -1;
      for (std::size_t position = 0; position < count; ++position) {
        const int matrix =
            plan.matrix_ids[static_cast<std::size_t>(order[permutation[position]])];
        if (matrix != previous_matrix) {
          if (previous_matrix >= 0) {
            schedule.run_offsets.push_back(
                static_cast<int>(first_interaction + position));
          }
          schedule.run_matrix.push_back(matrix);
          previous_matrix = matrix;
        }
      }
      if (count > 0) {
        schedule.run_offsets.push_back(
            static_cast<int>(schedule.interactions.size()));
      }
      order.clear();
      schedule.block_target_offsets.push_back(
          static_cast<int>(schedule.targets.size()));
      schedule.block_run_offsets.push_back(
          static_cast<int>(schedule.run_matrix.size()));
    }
    schedule.level_block_offsets.push_back(
        static_cast<int>(schedule.block_target_offsets.size()) - 1);
  }
  return schedule;
}

// Outputs kept in registers per chunk: 16 doubles or 32 floats (four AVX2
// vectors) per accumulation pass over alpha, then half and quarter chunks so
// that at most a vector's worth of outputs takes the generic remainder path.
template <typename Scalar>
constexpr int register_chunk = static_cast<int>(64 / sizeof(Scalar)) * 2;

template <typename Scalar, int Chunk>
inline void accumulate_chunk(const Scalar* matrix_chunk, const int n,
                             const Scalar* scaled_source,
                             Scalar* target_accumulator) {
  Scalar acc[Chunk];
  for (int j = 0; j < Chunk; ++j) {
    acc[j] = Scalar{0};
  }
  for (int alpha = 0; alpha < n; ++alpha) {
    const Scalar x = scaled_source[alpha];
    const Scalar* column = matrix_chunk + static_cast<std::size_t>(alpha) * n;
#pragma omp simd
    for (int j = 0; j < Chunk; ++j) {
      acc[j] += column[j] * x;
    }
  }
  for (int j = 0; j < Chunk; ++j) {
    target_accumulator[j] += acc[j];
  }
}

// One block: accumulate every target of the block over the class-sorted
// runs, then add the level-scaled result to the locals.  Within a run the
// matrix stays in L1 while each interaction performs the C x C update as
// unit-stride axpy columns into its target's accumulator row.
template <typename Plan, typename Scalar>
void apply_block(const Plan& plan, const M2LBlockSchedule& schedule,
                 const int block, const Scalar* local_scale,
                 const std::span<const Scalar> multipoles,
                 const std::span<Scalar> locals) {
  const int n = plan.coefficient_count;
  const auto block_index = static_cast<std::size_t>(block);
  const int target_first = schedule.block_target_offsets[block_index];
  const int target_count =
      schedule.block_target_offsets[block_index + 1] - target_first;
  alignas(64) Scalar accumulator[max_block_values];
  alignas(64) Scalar scaled_source[max_stack_coefficients];
  for (int value = 0; value < target_count * n; ++value) {
    accumulator[value] = Scalar{0};
  }
  for (int run = schedule.block_run_offsets[block_index];
       run < schedule.block_run_offsets[block_index + 1]; ++run) {
    const auto run_index = static_cast<std::size_t>(run);
    const Scalar* matrix = plan.matrices.data() +
        static_cast<std::size_t>(schedule.run_matrix[run_index]) * n * n;
    for (int slot = schedule.run_offsets[run_index];
         slot < schedule.run_offsets[run_index + 1]; ++slot) {
      const M2LBlockSchedule::Interaction& interaction =
          schedule.interactions[static_cast<std::size_t>(slot)];
      const Scalar* source_scale = plan.multipole_scaling.data() +
          static_cast<std::size_t>(interaction.source_level) * n;
      const Scalar* source_M = multipoles.data() +
          static_cast<std::size_t>(interaction.source) * n;
      for (int alpha = 0; alpha < n; ++alpha) {
        scaled_source[alpha] = source_scale[alpha] * source_M[alpha];
      }
      Scalar* target_accumulator =
          accumulator + static_cast<std::size_t>(interaction.target_slot) * n;
      // Measured split (agent_docs/performance_optimization.md, 3B): FP32
      // rows are fastest as plain unit-stride axpy updates of the
      // accumulator row (the 49-196 byte row stays in registers/L1), while
      // FP64 gains from register-blocking a chunk of outputs across all
      // alpha so that each matrix value is loaded once per interaction.
      int beta = 0;
      if constexpr (sizeof(Scalar) == 4) {
        for (int alpha = 0; alpha < n; ++alpha) {
          const Scalar x = scaled_source[alpha];
          const Scalar* column = matrix + static_cast<std::size_t>(alpha) * n;
#pragma omp simd
          for (int b = 0; b < n; ++b) {
            target_accumulator[b] += column[b] * x;
          }
        }
        continue;
      }
      for (; beta + register_chunk<Scalar> <= n; beta += register_chunk<Scalar>) {
        accumulate_chunk<Scalar, register_chunk<Scalar>>(
            matrix + beta, n, scaled_source, target_accumulator + beta);
      }
      if (beta + register_chunk<Scalar> / 2 <= n) {
        accumulate_chunk<Scalar, register_chunk<Scalar> / 2>(
            matrix + beta, n, scaled_source, target_accumulator + beta);
        beta += register_chunk<Scalar> / 2;
      }
      if (beta + register_chunk<Scalar> / 4 <= n) {
        accumulate_chunk<Scalar, register_chunk<Scalar> / 4>(
            matrix + beta, n, scaled_source, target_accumulator + beta);
        beta += register_chunk<Scalar> / 4;
      }
      if (beta < n) {
        const int width = n - beta;
        for (int alpha = 0; alpha < n; ++alpha) {
          const Scalar x = scaled_source[alpha];
          const Scalar* column =
              matrix + static_cast<std::size_t>(alpha) * n + beta;
#pragma omp simd
          for (int j = 0; j < width; ++j) {
            target_accumulator[beta + j] += column[j] * x;
          }
        }
      }
    }
  }
  for (int local_slot = 0; local_slot < target_count; ++local_slot) {
    Scalar* L = locals.data() +
        static_cast<std::size_t>(
            schedule.targets[static_cast<std::size_t>(target_first + local_slot)]) *
            n;
    const Scalar* target_accumulator =
        accumulator + static_cast<std::size_t>(local_slot) * n;
    for (int beta = 0; beta < n; ++beta) {
      L[beta] += local_scale[beta] * target_accumulator[beta];
    }
  }
}

// Blocks of one level own disjoint targets, so they run in parallel without
// synchronisation; dynamic scheduling absorbs the uneven interaction counts.
template <typename Plan, typename Scalar>
void apply_level(const Plan& plan, const M2LBlockSchedule& schedule,
                 const int level, const std::span<const Scalar> multipoles,
                 const std::span<Scalar> locals) {
  const auto level_index = static_cast<std::size_t>(level);
  if (level_index + 1 >= schedule.level_block_offsets.size()) {
    return;
  }
  const int block_begin = schedule.level_block_offsets[level_index];
  const int block_end = schedule.level_block_offsets[level_index + 1];
  const Scalar* local_scale = plan.local_scaling.data() +
      static_cast<std::size_t>(level) * plan.coefficient_count;
#pragma omp parallel for schedule(dynamic, 1) if (block_end - block_begin >= 4)
  for (int block = block_begin; block < block_end; ++block) {
    apply_block(plan, schedule, block, local_scale, multipoles, locals);
  }
}

} // namespace

M2LBlockSchedule build_m2l_block_schedule(const StaticM2LPlan& plan) {
  return build_schedule(plan);
}

M2LBlockSchedule build_m2l_block_schedule(const FloatStaticM2LPlan& plan) {
  return build_schedule(plan);
}

void apply_static_m2l_plan(const StaticM2LPlan& plan,
                           const M2LBlockSchedule& schedule, const int level,
                           const std::span<const double> multipoles,
                           const std::span<double> locals) {
  apply_level(plan, schedule, level, multipoles, locals);
}

void apply_static_m2l_plan(const FloatStaticM2LPlan& plan,
                           const M2LBlockSchedule& schedule, const int level,
                           const std::span<const float> multipoles,
                           const std::span<float> locals) {
  apply_level(plan, schedule, level, multipoles, locals);
}

} // namespace cdfmm::detail::cpu
