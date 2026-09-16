// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "cdfmm/plan/m2l.hpp"

namespace cdfmm::detail::cpu {

/**
 * @brief Transfer-class-sorted block schedule of the canonical M2L plan.
 *
 * Targets of one level are cut into blocks of consecutive (Morton-ordered)
 * targets.  Inside a block every interaction is stably sorted by transfer
 * class (matrix id), so one `C x C` matrix is applied to a run of
 * interactions while it stays in the first-level cache instead of being
 * streamed from the last-level cache once per interaction.  The schedule
 * only reorders the interactions of a block; every interaction keeps its
 * target, source and source level, and one thread owns a whole block, so
 * the accumulation into a target is deterministic (transfer-class order
 * instead of canonical row order).
 */
struct M2LBlockSchedule {
  struct Interaction {
    int source{0};
    int target_slot{0};
    int source_level{0};
  };

  int coefficient_count{0};
  int block_targets{0};
  /// Blocks of level `l` are `[level_block_offsets[l], level_block_offsets[l + 1])`.
  std::vector<int> level_block_offsets{};
  /// Targets of block `b` are `targets[block_target_offsets[b] .. [b + 1])`.
  std::vector<int> block_target_offsets{};
  std::vector<int> targets{};
  /// Runs of block `b` are `[block_run_offsets[b], block_run_offsets[b + 1])`.
  std::vector<int> block_run_offsets{};
  /// Interactions of run `r` are `[run_offsets[r], run_offsets[r + 1])`.
  std::vector<int> run_offsets{};
  std::vector<int> run_matrix{};
  std::vector<Interaction> interactions{};

  [[nodiscard]] bool empty() const noexcept { return interactions.empty(); }

  [[nodiscard]] std::size_t memory_bytes() const noexcept {
    return (level_block_offsets.size() + block_target_offsets.size() +
            targets.size() + block_run_offsets.size() + run_offsets.size() +
            run_matrix.size()) *
               sizeof(int) +
           interactions.size() * sizeof(Interaction);
  }
};

/** @brief Builds the block schedule for every level of the plan. */
M2LBlockSchedule build_m2l_block_schedule(const StaticM2LPlan& plan);
M2LBlockSchedule build_m2l_block_schedule(const FloatStaticM2LPlan& plan);

/**
 * @brief Applies one level of the plan through the block schedule.
 *
 * Deterministic: block ownership and the in-block order do not depend on the
 * thread count.
 */
void apply_static_m2l_plan(const StaticM2LPlan& plan,
                           const M2LBlockSchedule& schedule, int level,
                           std::span<const double> multipoles,
                           std::span<double> locals);
void apply_static_m2l_plan(const FloatStaticM2LPlan& plan,
                           const M2LBlockSchedule& schedule, int level,
                           std::span<const float> multipoles,
                           std::span<float> locals);

} // namespace cdfmm::detail::cpu
