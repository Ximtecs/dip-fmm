// SPDX-License-Identifier: Apache-2.0

#include "near_field.hpp"

namespace cdfmm::detail {

//------------------------------------------------------------------------------
// Near-field evaluation
//------------------------------------------------------------------------------
// This file owns direct neighbours only:
//
//   current moments -> explicit P2P leaf rows -> near field
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
    const StaticFmmTopology &topology, const std::span<const Vec3> sorted_dipole_moments,
    const std::span<const int> sorted_self_indices, const OutputFlags output,
    const std::span<PotentialField> sorted_results) {
  const auto &nodes = topology.nodes;
  const auto &targets = topology.sorted_target_positions;
  const std::span<const Vec3> sources = topology.sorted_source_positions;
  const auto &occupied_leaves = topology.target_leaves;

#pragma omp parallel for schedule(static) if (occupied_leaves.size() >= 8)
  for (std::ptrdiff_t occupied_index = 0;
       occupied_index < static_cast<std::ptrdiff_t>(occupied_leaves.size());
       ++occupied_index) {
    const StaticLeafRange &leaf_range =
        occupied_leaves[static_cast<std::size_t>(occupied_index)];
    const int leaf_index = leaf_range.node;
    const StaticFmmTopology::Node &leaf = nodes[static_cast<std::size_t>(leaf_index)];

    // A target is written by exactly one occupied leaf.  Neighbours are kept in
    // canonical list1 order so this parallel loop changes neither summation
    // order within a target nor the explicit self-interaction policy.
    for (std::size_t target_index = leaf_range.begin;
         target_index < leaf_range.begin + leaf_range.count; ++target_index) {
      const int self_sorted_index = sorted_self_indices[target_index];
      PotentialField &result = sorted_results[target_index];

      const int row_begin = topology.p2p_target_leaf_offsets[
          static_cast<std::size_t>(occupied_index)];
      const int row_end = topology.p2p_target_leaf_offsets[
          static_cast<std::size_t>(occupied_index + 1)];
      for (int row = row_begin; row < row_end; ++row) {
        const StaticP2PLeafRecord &record = topology.p2p_leaf_records[
            static_cast<std::size_t>(row)];
        if (record.source_count == 0) {
          continue;
        }

        int local_self_index = -1;
        if (record.skip_for_identity &&
            self_sorted_index >= static_cast<int>(record.source_begin) &&
            self_sorted_index <
                static_cast<int>(record.source_begin + record.source_count)) {
          local_self_index =
              self_sorted_index - static_cast<int>(record.source_begin);
        }

        const PotentialField near = p2p_dipole_sum(
            targets[target_index],
            sources.subspan(record.source_begin, record.source_count),
            sorted_dipole_moments.subspan(
                record.source_begin, record.source_count),
            output, local_self_index);
        result.phi += near.phi;
        result.H += near.H;
      }
    }
  }
}

} // namespace cdfmm::detail
