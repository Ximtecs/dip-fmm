// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/tree/uniform_topology.hpp"

#include <numeric>

namespace cdfmm {

StaticFmmTopology build_uniform_fmm_topology(
    const UniformTree& tree, const PeriodicCellOptions& periodic) {
  StaticFmmTopology result;
  const auto source_positions = tree.sorted_source_positions();
  const auto target_positions = tree.sorted_target_positions();
  result.sorted_source_positions.assign(source_positions.begin(), source_positions.end());
  result.sorted_target_positions.assign(target_positions.begin(), target_positions.end());
  result.source_permutation.assign(tree.source_permutation().begin(),
                                   tree.source_permutation().end());
  result.source_inverse_permutation.assign(
      tree.source_inverse_permutation().begin(),
      tree.source_inverse_permutation().end());
  result.target_permutation.assign(tree.target_permutation().begin(),
                                   tree.target_permutation().end());
  result.target_inverse_permutation.assign(
      tree.target_inverse_permutation().begin(),
      tree.target_inverse_permutation().end());
  const auto nodes = tree.nodes();
  result.nodes.reserve(nodes.size());
  for (const TreeNode& node : nodes) {
    result.nodes.push_back({node.index, node.level, node.parent, node.children,
                            node.centre, node.half_width, node.source_begin,
                            node.source_end, node.target_begin, node.target_end});
  }
  result.maximum_level = tree.max_level();
  for (const int leaf_index : tree.occupied_source_leaves()) {
    const TreeNode& leaf = nodes[static_cast<std::size_t>(leaf_index)];
    result.source_leaves.push_back({leaf.index, leaf.source_begin,
                                    leaf.source_count()});
  }
  for (const int leaf_index : tree.occupied_target_leaves()) {
    const TreeNode& leaf = nodes[static_cast<std::size_t>(leaf_index)];
    result.target_leaves.push_back({leaf.index, leaf.target_begin,
                                    leaf.target_count()});
  }
  // Keep the reference M2L interaction order exactly as UniformTree::list2.
  const auto append_m2l = [&](const TreeNode& source, const TreeNode& target,
                              const std::array<int, 3> image_shift,
                              const Vec3 source_shift) {
    const int boxes_per_axis = 1 << target.level;
    const std::array<int, 3> transfer_class{
        target.ix - source.ix - image_shift[0] * boxes_per_axis,
        target.iy - source.iy - image_shift[1] * boxes_per_axis,
        target.iz - source.iz - image_shift[2] * boxes_per_axis};
    result.m2l_interactions.push_back(
        {source.index, target.index, source.level, target.level,
         image_shift, transfer_class,
         target.centre - source.centre - source_shift,
         source_shift,
         target.half_width == 0.0 ? 0.0 : source.half_width / target.half_width});
  };
  for (const TreeNode& target : nodes) {
    if (target.level == 0 || target.target_count() == 0) {
      continue;
    }
    if (periodic.enabled) {
      for (const PeriodicBoxIdentity& identity :
           build_periodic_list2(target.level, {target.ix, target.iy, target.iz})) {
        const TreeNode& source = nodes[static_cast<std::size_t>(identity.node)];
        if (source.source_count() == 0) continue;
        append_m2l(source, target, identity.image_shift,
                   {periodic.lengths.x * identity.image_shift[0],
                    periodic.lengths.y * identity.image_shift[1],
                    periodic.lengths.z * identity.image_shift[2]});
      }
    } else {
      for (const int source_index : target.list2) {
        const TreeNode& source = nodes[static_cast<std::size_t>(source_index)];
        if (source.source_count() == 0) continue;
        append_m2l(source, target, {}, {});
      }
    }
  }
  result.m2l_target_row_offsets.assign(result.nodes.size() + 1, 0);
  for (const StaticM2LInteraction& interaction : result.m2l_interactions) {
    ++result.m2l_target_row_offsets[
        static_cast<std::size_t>(interaction.target_node) + 1];
  }
  std::partial_sum(result.m2l_target_row_offsets.begin(),
                   result.m2l_target_row_offsets.end(),
                   result.m2l_target_row_offsets.begin());
  for (int level = 1; level <= tree.max_level(); ++level) {
    for (int index = level_offset(level); index < level_offset(level + 1); ++index) {
      const TreeNode& child = nodes[static_cast<std::size_t>(index)];
      result.m2m_edges.push_back({child.index, child.parent,
                                  (child.ix & 1) | ((child.iy & 1) << 1) |
                                      ((child.iz & 1) << 2),
                                  child.level});
    }
  }
  // Downward order is parent level ascending, with the historical child order.
  for (int level = 1; level <= tree.max_level(); ++level) {
    for (int index = level_offset(level); index < level_offset(level + 1); ++index) {
      const TreeNode& child = nodes[static_cast<std::size_t>(index)];
      result.l2l_edges.push_back({child.parent, child.index,
                                  (child.ix & 1) | ((child.iy & 1) << 1) |
                                      ((child.iz & 1) << 2),
                                  child.level});
    }
  }
  // The offsets are schedule offsets, not node offsets.
  result.m2m_level_offsets.assign(static_cast<std::size_t>(tree.max_level()) + 2, 0);
  int cursor = 0;
  for (int level = 1; level <= tree.max_level(); ++level) {
    result.m2m_level_offsets[static_cast<std::size_t>(level)] = cursor;
    while (cursor < static_cast<int>(result.m2m_edges.size()) &&
           result.m2m_edges[static_cast<std::size_t>(cursor)].child_level == level) {
      ++cursor;
    }
  }
  result.m2m_level_offsets.back() = cursor;
  result.m2m_parent_level_offsets.assign(
      static_cast<std::size_t>(tree.max_level()) + 2, 0);
  result.m2m_parent_edge_offsets.clear();
  result.m2m_parent_edge_offsets.push_back(0);
  for (int level = 1; level <= tree.max_level(); ++level) {
    const int edge_begin = result.m2m_level_offsets[static_cast<std::size_t>(level)];
    const int edge_end = result.m2m_level_offsets[static_cast<std::size_t>(level + 1)];
    int edge = edge_begin;
    while (edge < edge_end) {
      const int parent = result.m2m_edges[static_cast<std::size_t>(edge)].target_node;
      result.m2m_parent_nodes.push_back(parent);
      while (edge < edge_end &&
             result.m2m_edges[static_cast<std::size_t>(edge)].target_node ==
                 parent) {
        ++edge;
      }
      result.m2m_parent_edge_offsets.push_back(edge);
    }
    result.m2m_parent_level_offsets[static_cast<std::size_t>(level + 1)] =
        static_cast<int>(result.m2m_parent_nodes.size());
  }
  result.l2l_level_offsets.assign(static_cast<std::size_t>(tree.max_level()) + 2, 0);
  cursor = 0;
  for (int level = 1; level <= tree.max_level(); ++level) {
    result.l2l_level_offsets[static_cast<std::size_t>(level)] = cursor;
    while (cursor < static_cast<int>(result.l2l_edges.size()) &&
           result.l2l_edges[static_cast<std::size_t>(cursor)].child_level == level) {
      ++cursor;
    }
  }
  result.l2l_level_offsets.back() = cursor;

  result.p2p_target_leaves.reserve(result.target_leaves.size());
  result.p2p_target_leaf_offsets.push_back(0);
  for (const StaticLeafRange& target_leaf : result.target_leaves) {
    result.p2p_target_leaves.push_back(target_leaf.node);
    const TreeNode& target = nodes[static_cast<std::size_t>(target_leaf.node)];
    const auto identities = periodic.enabled
        ? build_periodic_list1(target.level, {target.ix, target.iy, target.iz})
        : std::vector<PeriodicBoxIdentity>{};
    if (periodic.enabled) {
      for (const auto& identity : identities) {
        const TreeNode& source = nodes[static_cast<std::size_t>(identity.node)];
        if (source.source_count() == 0) continue;
        result.p2p_leaf_records.push_back({
            target.index, source.index, target_leaf.begin, target_leaf.count,
            source.source_begin, source.source_count(),
            {periodic.lengths.x * identity.image_shift[0],
             periodic.lengths.y * identity.image_shift[1],
             periodic.lengths.z * identity.image_shift[2]},
            identity.image_shift,
            identity.image_shift == std::array<int, 3>{0, 0, 0}});
      }
    } else {
      for (const int source_index : target.list1) {
        const TreeNode& source = nodes[static_cast<std::size_t>(source_index)];
        if (source.source_count() == 0) continue;
        result.p2p_leaf_records.push_back({
            target.index, source.index, target_leaf.begin, target_leaf.count,
            source.source_begin, source.source_count(), {}, {}, true});
      }
    }
    result.p2p_target_leaf_offsets.push_back(
        static_cast<int>(result.p2p_leaf_records.size()));
  }
  result.validate();
  return result;
}

} // namespace cdfmm
