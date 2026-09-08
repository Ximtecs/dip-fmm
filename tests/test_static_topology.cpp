// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>

#include "cdfmm/static_topology.hpp"

using namespace cdfmm;

TEST_CASE("uniform topology adapter preserves leaf and translation order",
          "[static_topology]") {
  const std::vector<Vec3> sources{{-0.8, -0.7, -0.6}, {0.7, 0.6, 0.5},
                                  {0.1, -0.2, 0.3}};
  const std::vector<Vec3> targets{{-0.4, 0.2, -0.1}, {0.6, -0.5, 0.4}};
  const UniformTree tree(sources, targets, UniformTreeOptions{.max_level = 2});
  const StaticFmmTopology topology = build_uniform_fmm_topology(tree);

  REQUIRE(topology.nodes.size() == tree.nodes().size());
  REQUIRE(topology.sorted_source_positions.size() == sources.size());
  REQUIRE(topology.sorted_target_positions.size() == targets.size());
  REQUIRE(std::equal(topology.source_permutation.begin(),
                     topology.source_permutation.end(),
                     tree.source_permutation().begin()));
  REQUIRE(std::equal(topology.target_permutation.begin(),
                     topology.target_permutation.end(),
                     tree.target_permutation().begin()));
  REQUIRE(topology.m2m_edges.size() == tree.nodes().size() - 1);
  REQUIRE(topology.l2l_edges.size() == tree.nodes().size() - 1);
  REQUIRE(topology.m2l_target_row_offsets.size() == topology.nodes.size() + 1);
  REQUIRE(topology.p2p_target_leaf_offsets.size() ==
          topology.target_leaves.size() + 1);

  std::size_t edge_slot = 0;
  for (int level = 1; level <= tree.max_level(); ++level) {
    for (int index = level_offset(level); index < level_offset(level + 1);
         ++index) {
      const TreeNode& child = tree.nodes()[static_cast<std::size_t>(index)];
      const int child_class = (child.ix & 1) | ((child.iy & 1) << 1) |
          ((child.iz & 1) << 2);
      REQUIRE(topology.m2m_edges[edge_slot++] ==
              StaticTranslationEdge{child.index, child.parent, child_class,
                                     child.level});
    }
  }
  REQUIRE(edge_slot == topology.m2m_edges.size());
  edge_slot = 0;
  for (int level = 1; level <= tree.max_level(); ++level) {
    for (int index = level_offset(level); index < level_offset(level + 1);
         ++index) {
      const TreeNode& child = tree.nodes()[static_cast<std::size_t>(index)];
      const int child_class = (child.ix & 1) | ((child.iy & 1) << 1) |
          ((child.iz & 1) << 2);
      REQUIRE(topology.l2l_edges[edge_slot++] ==
              StaticTranslationEdge{child.parent, child.index, child_class,
                                     child.level});
    }
  }
  REQUIRE(edge_slot == topology.l2l_edges.size());

  for (const StaticLeafRange& leaf : topology.source_leaves) {
    const TreeNode& expected = tree.nodes()[static_cast<std::size_t>(leaf.node)];
    REQUIRE(leaf.begin == expected.source_begin);
    REQUIRE(leaf.count == expected.source_count());
  }
  for (const StaticLeafRange& leaf : topology.target_leaves) {
    const TreeNode& expected = tree.nodes()[static_cast<std::size_t>(leaf.node)];
    REQUIRE(leaf.begin == expected.target_begin);
    REQUIRE(leaf.count == expected.target_count());
  }
  std::size_t interaction_slot = 0;
  for (const TreeNode& target : tree.nodes()) {
    if (target.level == 0 || target.target_count() == 0) {
      continue;
    }
    for (const int source_index : target.list2) {
      const TreeNode& source =
          tree.nodes()[static_cast<std::size_t>(source_index)];
      if (source.source_count() == 0) {
        continue;
      }
      REQUIRE(interaction_slot < topology.m2l_interactions.size());
      const StaticM2LInteraction& interaction =
          topology.m2l_interactions[interaction_slot++];
      REQUIRE(interaction.source_node == source.index);
      REQUIRE(interaction.target_node == target.index);
      REQUIRE(interaction.source_level == source.level);
      REQUIRE(interaction.target_level == target.level);
      REQUIRE(interaction.source_level == interaction.target_level);
      REQUIRE(interaction.source_to_target_width == Catch::Approx(1.0));
      const Vec3 displacement = target.centre - source.centre -
          interaction.source_shift;
      REQUIRE(interaction.displacement.x == Catch::Approx(displacement.x));
      REQUIRE(interaction.displacement.y == Catch::Approx(displacement.y));
      REQUIRE(interaction.displacement.z == Catch::Approx(displacement.z));
    }
  }
  REQUIRE(interaction_slot == topology.m2l_interactions.size());
}

TEST_CASE("uniform topology adapter handles empty depth-zero geometry",
          "[static_topology]") {
  const UniformTree tree(std::vector<Vec3>{}, std::vector<Vec3>{},
                         UniformTreeOptions{.max_level = 0});
  const StaticFmmTopology topology = build_uniform_fmm_topology(tree);
  REQUIRE(topology.nodes.size() == 1);
  REQUIRE(topology.source_leaves.empty());
  REQUIRE(topology.target_leaves.empty());
  REQUIRE(topology.m2m_edges.empty());
  REQUIRE(topology.l2l_edges.empty());
  REQUIRE(topology.m2l_interactions.empty());
  REQUIRE(topology.p2p_leaf_records.empty());
  REQUIRE_NOTHROW(topology.validate());
}

TEST_CASE("static topology validates compact non-analytical node IDs",
          "[static_topology]") {
  StaticFmmTopology topology;
  topology.nodes.resize(3);
  topology.nodes[0].index = 0;
  topology.nodes[0].half_width = 1.0;
  topology.nodes[0].children[0] = 2;
  topology.nodes[1].index = 1;
  topology.nodes[1].level = 2;
  topology.nodes[1].half_width = 0.25;
  topology.nodes[1].parent = 2;
  topology.nodes[2].index = 2;
  topology.nodes[2].level = 1;
  topology.nodes[2].half_width = 0.5;
  topology.nodes[2].parent = 0;
  topology.nodes[2].children[0] = 1;
  topology.m2m_edges = {{2, 0, 0, 1}, {1, 2, 0, 2}};
  topology.l2l_edges = {{0, 2, 0, 1}, {2, 1, 0, 2}};
  topology.m2l_target_row_offsets = {0, 0, 0, 0};
  topology.m2m_level_offsets = {0, 0, 1, 2};
  topology.l2l_level_offsets = {0, 0, 1, 2};
  topology.m2m_parent_nodes = {0, 2};
  topology.m2m_parent_level_offsets = {0, 0, 1, 2};
  topology.m2m_parent_edge_offsets = {0, 1, 2};
  topology.p2p_target_leaf_offsets = {0};
  topology.root = 0;
  topology.maximum_level = 2;
  REQUIRE_NOTHROW(topology.validate());
}

TEST_CASE("compact topology schedules leaf operators and unequal P2P rectangles",
          "[static_topology]") {
  StaticFmmTopology topology;
  topology.sorted_source_positions = {{-0.2, 0.0, 0.1},
                                      {0.1, -0.1, 0.2},
                                      {0.3, 0.2, -0.2}};
  topology.sorted_target_positions = {{-0.1, 0.1, 0.0},
                                      {0.2, 0.0, 0.1},
                                      {0.4, -0.2, 0.2},
                                      {0.5, 0.3, -0.1}};
  topology.nodes.resize(3);
  topology.nodes[0].index = 0;
  topology.nodes[0].half_width = 1.0;
  topology.nodes[0].children = {1, 2, -1, -1, -1, -1, -1, -1};
  topology.nodes[0].source_begin = 0;
  topology.nodes[0].source_end = 3;
  topology.nodes[0].target_begin = 0;
  topology.nodes[0].target_end = 4;
  topology.nodes[1].index = 1;
  topology.nodes[1].level = 1;
  topology.nodes[1].parent = 0;
  topology.nodes[1].half_width = 0.5;
  topology.nodes[1].centre = {-0.25, 0.0, 0.0};
  topology.nodes[1].source_begin = 0;
  topology.nodes[1].source_end = 2;
  topology.nodes[1].target_begin = 0;
  topology.nodes[1].target_end = 1;
  topology.nodes[2].index = 2;
  topology.nodes[2].level = 1;
  topology.nodes[2].parent = 0;
  topology.nodes[2].half_width = 0.5;
  topology.nodes[2].centre = {0.25, 0.0, 0.0};
  topology.nodes[2].source_begin = 2;
  topology.nodes[2].source_end = 3;
  topology.nodes[2].target_begin = 1;
  topology.nodes[2].target_end = 4;
  topology.source_leaves = {{1, 0, 2}, {2, 2, 1}};
  topology.target_leaves = {{1, 0, 1}, {2, 1, 3}};
  topology.m2m_edges = {{1, 0, 0, 1}, {2, 0, 1, 1}};
  topology.l2l_edges = {{0, 1, 0, 1}, {0, 2, 1, 1}};
  topology.m2l_target_row_offsets = {0, 0, 0, 0};
  topology.m2m_level_offsets = {0, 0, 2};
  topology.l2l_level_offsets = {0, 0, 2};
  topology.m2m_parent_nodes = {0};
  topology.m2m_parent_level_offsets = {0, 0, 1};
  topology.m2m_parent_edge_offsets = {0, 2};
  topology.p2p_target_leaves = {1, 2};
  topology.p2p_target_leaf_offsets = {0, 1, 2};
  topology.p2p_leaf_records = {
      {{0, 1, 0, 2}, 1, 1, {}, {}, true},
      {{1, 3, 2, 1}, 2, 2, {}, {}, true}};
  topology.maximum_level = 1;
  REQUIRE_NOTHROW(topology.validate());

  const MultiIndexSet basis(1);
  const StaticCoefficientOperator p2m = build_static_p2m_operator(
      basis, topology.nodes[1].centre,
      std::span<const Vec3>(topology.sorted_source_positions).subspan(0, 2));
  REQUIRE(p2m.input_size == 6);
  REQUIRE(p2m.output_size == basis.size());
  std::vector<double> p2m_output(static_cast<std::size_t>(basis.size()), 0.0);
  apply_static_operator(p2m, std::vector<double>(6, 1.0), p2m_output);
  REQUIRE(std::any_of(p2m_output.begin(), p2m_output.end(),
                      [](const double value) { return value != 0.0; }));

  const StaticL2PEvaluator l2p = build_static_l2p_evaluator(
      basis, topology.nodes[2].centre, topology.sorted_target_positions[1]);
  REQUIRE(l2p.potential.size() == static_cast<std::size_t>(basis.size()));
  REQUIRE(l2p.field[0].size() == static_cast<std::size_t>(basis.size()));

  const std::array<std::array<int, 2>, 5> interactions{{
      {{0, 0}}, {{0, 1}}, {{1, 2}}, {{2, 2}}, {{3, 2}}}};
  const StaticP2POperator p2p = build_static_p2p_operator(
      std::span<const Vec3>(topology.sorted_target_positions),
      std::span<const Vec3>(topology.sorted_source_positions), interactions);
  const std::array<StaticP2PLeafPair, 2> pairs{{
      {0, 1, 0, 2}, {1, 3, 2, 1}}};
  const StaticP2PLeafPlan leaf_plan = build_static_p2p_leaf_plan(p2p, pairs);
  REQUIRE(leaf_plan.blocks.size() == 2);
  REQUIRE(leaf_plan.blocks[0].source_count == 2);
  REQUIRE(leaf_plan.blocks[1].source_count == 1);
  REQUIRE(leaf_plan.target_counts == std::vector<int>{1, 3});
}

TEST_CASE("periodic topology retains image order and identity metadata",
          "[static_topology]") {
  const std::vector<Vec3> positions{{-0.49, -0.49, -0.49},
                                    {0.49, 0.49, 0.49}};
  UniformTreeOptions options;
  options.max_level = 1;
  options.root_centre = Vec3{};
  options.root_half_width = 0.5;
  const UniformTree tree(positions, positions, options);
  PeriodicCellOptions periodic;
  periodic.enabled = true;
  const StaticFmmTopology topology =
      build_uniform_fmm_topology(tree, periodic);

  std::size_t record = 0;
  for (std::size_t target_leaf = 0;
       target_leaf < topology.target_leaves.size(); ++target_leaf) {
    const auto& leaf = topology.target_leaves[target_leaf];
    const auto& node = tree.nodes()[static_cast<std::size_t>(leaf.node)];
    const auto identities =
        build_periodic_list1(node.level, {node.ix, node.iy, node.iz});
    for (const auto& identity : identities) {
      const auto& source = tree.nodes()[static_cast<std::size_t>(identity.node)];
      if (source.source_count() == 0) {
        continue;
      }
      REQUIRE(record < topology.p2p_leaf_records.size());
      const auto& actual = topology.p2p_leaf_records[record++];
      REQUIRE(actual.target_leaf == leaf.node);
      REQUIRE(actual.source_leaf == identity.node);
      REQUIRE(actual.skip_for_identity ==
              (identity.image_shift == std::array<int, 3>{0, 0, 0}));
      REQUIRE(actual.image_shift == identity.image_shift);
      REQUIRE(actual.source_shift.x ==
              Catch::Approx(identity.image_shift[0] * periodic.lengths.x));
      REQUIRE(actual.source_shift.y ==
              Catch::Approx(identity.image_shift[1] * periodic.lengths.y));
      REQUIRE(actual.source_shift.z ==
              Catch::Approx(identity.image_shift[2] * periodic.lengths.z));
    }
  }
  REQUIRE(record == topology.p2p_leaf_records.size());
}
