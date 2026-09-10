// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/static_topology.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace cdfmm {

void StaticFmmTopology::validate() const {
  if (root < 0 || root >= static_cast<int>(nodes.size())) {
    throw std::invalid_argument("static topology root is invalid");
  }
  if (maximum_level < 0) {
    throw std::invalid_argument("static topology maximum level is invalid");
  }
  const auto finite_vec3 = [](const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z);
  };
  if (!finite_vec3(coordinate_origin) || !std::isfinite(coordinate_scale) ||
      coordinate_scale <= 0.0) {
    throw std::invalid_argument(
        "static topology coordinate transform is invalid");
  }
  for (const Vec3& position : sorted_source_positions) {
    if (!finite_vec3(position)) {
      throw std::invalid_argument("static topology source position is invalid");
    }
  }
  for (const Vec3& position : sorted_target_positions) {
    if (!finite_vec3(position)) {
      throw std::invalid_argument("static topology target position is invalid");
    }
  }
  const auto check_permutation = [](
      const std::vector<int>& permutation,
      const std::vector<int>& inverse_permutation, const std::size_t count,
      const char* label) {
    if (permutation.size() != count || inverse_permutation.size() != count) {
      throw std::invalid_argument(std::string("static topology ") + label +
                                  " permutation size is invalid");
    }
    std::vector<bool> seen(count, false);
    for (std::size_t sorted = 0; sorted < count; ++sorted) {
      const int original = permutation[sorted];
      if (original < 0 || original >= static_cast<int>(count) ||
          seen[static_cast<std::size_t>(original)] ||
          inverse_permutation[static_cast<std::size_t>(original)] !=
              static_cast<int>(sorted)) {
        throw std::invalid_argument(std::string("static topology ") + label +
                                    " permutation is invalid");
      }
      seen[static_cast<std::size_t>(original)] = true;
    }
  };
  check_permutation(source_permutation, source_inverse_permutation,
                    sorted_source_positions.size(), "source");
  check_permutation(target_permutation, target_inverse_permutation,
                    sorted_target_positions.size(), "target");
  auto check_range = [](std::size_t begin, std::size_t end,
                        std::size_t size, const char* label) {
    if (begin > end || end > size) {
      throw std::invalid_argument(std::string("static topology ") + label +
                                  " range is invalid");
    }
  };
  for (std::size_t node_slot = 0; node_slot < nodes.size(); ++node_slot) {
    const Node& node = nodes[node_slot];
    if (node.index != static_cast<int>(node_slot)) {
      throw std::invalid_argument("static topology node ID is invalid");
    }
    if (node.parent < -1 || node.parent >= static_cast<int>(nodes.size())) {
      throw std::invalid_argument("static topology parent ID is invalid");
    }
    if (node.parent >= 0 &&
        nodes[static_cast<std::size_t>(node.parent)].level != node.level - 1) {
      throw std::invalid_argument("static topology parent level is invalid");
    }
    if (node.level < 0 || node.level > maximum_level) {
      throw std::invalid_argument("static topology node level is invalid");
    }
    if (!finite_vec3(node.centre) || !std::isfinite(node.half_width) ||
        node.half_width <= 0.0) {
      throw std::invalid_argument("static topology node geometry is invalid");
    }
    check_range(node.source_begin, node.source_end,
                sorted_source_positions.size(), "source");
    check_range(node.target_begin, node.target_end,
                sorted_target_positions.size(), "target");
    for (const int child : node.children) {
      if (child < -1 || child >= static_cast<int>(nodes.size())) {
        throw std::invalid_argument("static topology child ID is invalid");
      }
      if (child >= 0 &&
          nodes[static_cast<std::size_t>(child)].parent != node.index) {
        throw std::invalid_argument("static topology child link is invalid");
      }
    }
  }
  if (nodes[static_cast<std::size_t>(root)].parent != -1) {
    throw std::invalid_argument("static topology root parent is invalid");
  }
  for (std::size_t node_slot = 0; node_slot < nodes.size(); ++node_slot) {
    if (static_cast<int>(node_slot) != root &&
        nodes[node_slot].parent < 0) {
      throw std::invalid_argument("static topology non-root parent is invalid");
    }
  }
  auto check_edges = [this](const std::vector<StaticTranslationEdge>& edges,
                            const bool m2m) {
    for (const StaticTranslationEdge& edge : edges) {
      if (edge.source_node < 0 || edge.target_node < 0 ||
          edge.source_node >= static_cast<int>(nodes.size()) ||
          edge.target_node >= static_cast<int>(nodes.size()) ||
          edge.child_class < 0 || edge.child_class >= 8 || edge.child_level < 1) {
        throw std::invalid_argument("static topology translation edge is invalid");
      }
      const int source_level = nodes[static_cast<std::size_t>(edge.source_node)].level;
      const int target_level = nodes[static_cast<std::size_t>(edge.target_node)].level;
      if ((m2m && (source_level != edge.child_level ||
                   target_level != edge.child_level - 1)) ||
          (!m2m && (target_level != edge.child_level ||
                    source_level != edge.child_level - 1))) {
        throw std::invalid_argument("static topology translation levels are invalid");
      }
    }
  };
  check_edges(m2m_edges, true);
  check_edges(l2l_edges, false);
  if (m2l_target_row_offsets.size() != nodes.size() + 1 ||
      m2l_target_row_offsets.empty() || m2l_target_row_offsets.front() != 0 ||
      m2l_target_row_offsets.back() !=
          static_cast<int>(m2l_interactions.size()) ||
      !std::is_sorted(m2l_target_row_offsets.begin(),
                      m2l_target_row_offsets.end())) {
    throw std::invalid_argument("static topology M2L row offsets are invalid");
  }
  for (std::size_t target = 0; target < nodes.size(); ++target) {
    const int begin = m2l_target_row_offsets[target];
    const int end = m2l_target_row_offsets[target + 1];
    for (int interaction = begin; interaction < end; ++interaction) {
      if (m2l_interactions[static_cast<std::size_t>(interaction)].target_node !=
          static_cast<int>(target)) {
        throw std::invalid_argument("static topology M2L row grouping is invalid");
      }
    }
  }
  for (const StaticM2LInteraction& interaction : m2l_interactions) {
    if (interaction.source_node < 0 || interaction.target_node < 0 ||
        interaction.source_node >= static_cast<int>(nodes.size()) ||
        interaction.target_node >= static_cast<int>(nodes.size())) {
      throw std::invalid_argument("static topology M2L interaction is invalid");
    }
    const Node& source = nodes[static_cast<std::size_t>(interaction.source_node)];
    const Node& target = nodes[static_cast<std::size_t>(interaction.target_node)];
    if (interaction.source_level != source.level ||
        interaction.target_level != target.level ||
        !std::isfinite(interaction.source_to_target_width) ||
        interaction.source_to_target_width < 0.0 ||
        !std::isfinite(interaction.displacement.x) ||
        !std::isfinite(interaction.displacement.y) ||
        !std::isfinite(interaction.displacement.z) ||
        !std::isfinite(interaction.source_shift.x) ||
        !std::isfinite(interaction.source_shift.y) ||
        !std::isfinite(interaction.source_shift.z)) {
      throw std::invalid_argument("static topology M2L levels are invalid");
    }
  }
  auto check_offsets = [](const std::vector<int>& offsets, std::size_t size,
                          const char* label) {
    if (offsets.empty() || offsets.front() != 0 ||
        offsets.back() != static_cast<int>(size) ||
        !std::is_sorted(offsets.begin(), offsets.end())) {
      throw std::invalid_argument(std::string("static topology ") + label +
                                  " offsets are invalid");
    }
  };
  if (m2m_level_offsets.size() != static_cast<std::size_t>(maximum_level) + 2 ||
      l2l_level_offsets.size() != static_cast<std::size_t>(maximum_level) + 2) {
    throw std::invalid_argument("static topology translation level offsets are invalid");
  }
  check_offsets(m2m_level_offsets, m2m_edges.size(), "M2M");
  check_offsets(l2l_level_offsets, l2l_edges.size(), "L2L");
  if (m2m_parent_level_offsets.size() !=
          static_cast<std::size_t>(maximum_level) + 2 ||
      m2m_parent_level_offsets.empty() ||
      m2m_parent_level_offsets.front() != 0 ||
      m2m_parent_level_offsets.back() !=
          static_cast<int>(m2m_parent_nodes.size())) {
    throw std::invalid_argument("static topology M2M parent offsets are invalid");
  }
  if (m2m_parent_edge_offsets.size() != m2m_parent_nodes.size() + 1 ||
      m2m_parent_edge_offsets.empty() ||
      m2m_parent_edge_offsets.front() != 0 ||
      m2m_parent_edge_offsets.back() != static_cast<int>(m2m_edges.size())) {
    throw std::invalid_argument("static topology M2M parent edge offsets are invalid");
  }
  if (!std::is_sorted(m2m_parent_level_offsets.begin(),
                      m2m_parent_level_offsets.end()) ||
      !std::is_sorted(m2m_parent_edge_offsets.begin(),
                      m2m_parent_edge_offsets.end())) {
    throw std::invalid_argument("static topology M2M parent offsets are invalid");
  }
  const auto find_source_leaf = [this](const int node) {
    return std::find_if(source_leaves.begin(), source_leaves.end(),
                        [node](const StaticLeafRange& leaf) {
                          return leaf.node == node;
                        });
  };
  const auto find_target_leaf = [this](const int node) {
    return std::find_if(target_leaves.begin(), target_leaves.end(),
                        [node](const StaticLeafRange& leaf) {
                          return leaf.node == node;
                        });
  };
  for (const int parent : m2m_parent_nodes) {
    if (parent < 0 || parent >= static_cast<int>(nodes.size())) {
      throw std::invalid_argument("static topology M2M parent node is invalid");
    }
  }
  for (std::size_t parent_slot = 0;
       parent_slot < m2m_parent_nodes.size(); ++parent_slot) {
    const int begin = m2m_parent_edge_offsets[parent_slot];
    const int end = m2m_parent_edge_offsets[parent_slot + 1];
    for (int edge = begin; edge < end; ++edge) {
      if (m2m_edges[static_cast<std::size_t>(edge)].target_node !=
          m2m_parent_nodes[parent_slot]) {
        throw std::invalid_argument("static topology M2M parent grouping is invalid");
      }
    }
  }
  if (p2p_target_leaf_offsets.size() != p2p_target_leaves.size() + 1 ||
      p2p_target_leaf_offsets.empty() ||
      p2p_target_leaf_offsets.back() !=
          static_cast<int>(p2p_leaf_records.size())) {
    throw std::invalid_argument("static topology P2P row offsets are invalid");
  }
  for (std::size_t leaf_slot = 0; leaf_slot < p2p_target_leaves.size();
       ++leaf_slot) {
    const int target_leaf = p2p_target_leaves[leaf_slot];
    if (target_leaf < 0 || target_leaf >= static_cast<int>(nodes.size())) {
      throw std::invalid_argument("static topology P2P target leaf is invalid");
    }
    const auto target_range = find_target_leaf(target_leaf);
    if (target_range == target_leaves.end()) {
      throw std::invalid_argument("static topology P2P target leaf is not occupied");
    }
    const int begin = p2p_target_leaf_offsets[leaf_slot];
    const int end = p2p_target_leaf_offsets[leaf_slot + 1];
    for (int record = begin; record < end; ++record) {
      const StaticP2PLeafRecord& pair =
          p2p_leaf_records[static_cast<std::size_t>(record)];
      if (pair.target_leaf != target_leaf ||
          pair.target_begin != target_range->begin ||
          pair.target_count != target_range->count) {
        throw std::invalid_argument("static topology P2P target row is invalid");
      }
    }
  }
  for (const StaticLeafRange& leaf : source_leaves) {
    if (leaf.node < 0 || leaf.node >= static_cast<int>(nodes.size())) {
      throw std::invalid_argument("static topology source leaf is invalid");
    }
    check_range(leaf.begin, leaf.begin + leaf.count,
                sorted_source_positions.size(), "source leaf");
    const Node& node = nodes[static_cast<std::size_t>(leaf.node)];
    if (leaf.begin != node.source_begin ||
        leaf.count != node.source_count()) {
      throw std::invalid_argument(
          "static topology source leaf does not match its node range");
    }
  }
  for (const StaticLeafRange& leaf : target_leaves) {
    if (leaf.node < 0 || leaf.node >= static_cast<int>(nodes.size())) {
      throw std::invalid_argument("static topology target leaf is invalid");
    }
    check_range(leaf.begin, leaf.begin + leaf.count,
                sorted_target_positions.size(), "target leaf");
    const Node& node = nodes[static_cast<std::size_t>(leaf.node)];
    if (leaf.begin != node.target_begin ||
        leaf.count != node.target_count()) {
      throw std::invalid_argument(
          "static topology target leaf does not match its node range");
    }
  }
  for (const StaticP2PLeafRecord& record : p2p_leaf_records) {
    if (record.target_leaf < 0 || record.source_leaf < 0) {
      throw std::invalid_argument("static topology P2P leaf record is invalid");
    }
    if (record.target_leaf >= static_cast<int>(nodes.size()) ||
        record.source_leaf >= static_cast<int>(nodes.size()) ||
        record.target_begin > sorted_target_positions.size() ||
        record.source_begin > sorted_source_positions.size() ||
        (record.target_begin <= sorted_target_positions.size() &&
         record.target_count >
             sorted_target_positions.size() - record.target_begin) ||
        (record.source_begin <= sorted_source_positions.size() &&
         record.source_count >
             sorted_source_positions.size() - record.source_begin)) {
      throw std::invalid_argument("static topology P2P leaf range is invalid");
    }
    const auto source_range = find_source_leaf(record.source_leaf);
    if (source_range == source_leaves.end() ||
        record.source_begin != source_range->begin ||
        record.source_count != source_range->count) {
      throw std::invalid_argument("static topology P2P source row is invalid");
    }
  }
}

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
