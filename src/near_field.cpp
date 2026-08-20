// SPDX-License-Identifier: Apache-2.0

#include "near_field.hpp"

namespace cdfmm::detail {

//------------------------------------------------------------------------------
// Near-field evaluation
//------------------------------------------------------------------------------
// This file owns direct neighbours only:
//
//   current moments -> list1 P2P -> near field
//
// Multipole and local expansions do not belong here.  UniformFmm combines this
// result with the independent expansion branch as H = H_far + H_near.  Both
// executors consume a packing derived once from the canonical P2P plan; no
// backend-specific interaction list is constructed during evaluation.

void evaluate_static_near_field(
    const StaticP2PCompactPlan &p2p_plan,
    const std::span<const Vec3> sorted_dipole_moments,
    const std::span<Vec3> near_fields,
    const std::span<const int> sorted_self_indices) {
  // Each target row owns its accumulation.  The packed ordering is therefore
  // deterministic and race free without atomics or per-thread temporaries.
  apply_static_p2p_compact_plan(p2p_plan, sorted_dipole_moments, near_fields,
                                sorted_self_indices);
}

void evaluate_reference_near_field(
    const UniformTree &tree, const std::span<const Vec3> sorted_dipole_moments,
    const std::span<const int> sorted_self_indices, const OutputFlags output,
    const std::span<PotentialField> sorted_results) {
  const auto nodes = tree.nodes();
  const auto targets = tree.sorted_target_positions();
  const auto sources = tree.sorted_source_positions();
  const auto occupied_leaves = tree.occupied_target_leaves();

#pragma omp parallel for schedule(static) if (occupied_leaves.size() >= 8)
  for (std::ptrdiff_t occupied_index = 0;
       occupied_index < static_cast<std::ptrdiff_t>(occupied_leaves.size());
       ++occupied_index) {
    const int leaf_index =
        occupied_leaves[static_cast<std::size_t>(occupied_index)];
    const TreeNode &leaf = nodes[static_cast<std::size_t>(leaf_index)];

    // A target is written by exactly one occupied leaf.  Neighbours are kept in
    // canonical list1 order so this parallel loop changes neither summation
    // order within a target nor the explicit self-interaction policy.
    for (std::size_t target_index = leaf.target_begin;
         target_index < leaf.target_end; ++target_index) {
      const int self_sorted_index = sorted_self_indices[target_index];
      PotentialField &result = sorted_results[target_index];

      for (const int neighbour_index : leaf.list1) {
        const TreeNode &neighbour =
            nodes[static_cast<std::size_t>(neighbour_index)];
        if (neighbour.source_count() == 0) {
          continue;
        }

        int local_self_index = -1;
        if (self_sorted_index >= static_cast<int>(neighbour.source_begin) &&
            self_sorted_index < static_cast<int>(neighbour.source_end)) {
          local_self_index =
              self_sorted_index - static_cast<int>(neighbour.source_begin);
        }

        const PotentialField near = p2p_dipole_sum(
            targets[target_index],
            sources.subspan(neighbour.source_begin, neighbour.source_count()),
            sorted_dipole_moments.subspan(neighbour.source_begin,
                                          neighbour.source_count()),
            output, local_self_index);
        result.phi += near.phi;
        result.H += near.H;
      }
    }
  }
}

} // namespace cdfmm::detail
