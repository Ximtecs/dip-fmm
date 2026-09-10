// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "cdfmm/periodic.hpp"
#include "cdfmm/uniform_tree.hpp"

namespace cdfmm {

/** @brief Compact source or target range owned by one static-tree leaf. */
struct StaticLeafRange {
  int node{-1};
  std::size_t begin{0};
  std::size_t count{0};
};

/** @brief One explicit parent/child translation edge. */
struct StaticTranslationEdge {
  int source_node{-1};
  int target_node{-1};
  int child_class{0};
  int child_level{0};

  [[nodiscard]] bool operator==(const StaticTranslationEdge&) const = default;
};

/** @brief One explicit far-field multipole-to-local interaction. */
struct StaticM2LInteraction {
  int source_node{-1};
  int target_node{-1};
  int source_level{0};
  int target_level{0};
  /// Integer periodic image identity; zero denotes the central image.
  std::array<int, 3> image_shift{{0, 0, 0}};
  /// Integer transfer class used by the current uniform normalisation.
  /// Future builders can pair it with the explicit displacement and ratio.
  std::array<int, 3> transfer_class{{0, 0, 0}};
  /// Target centre minus source centre and image shift.  Adaptive producers
  /// can use this together with the endpoint levels for normalized transfers.
  Vec3 displacement{};
  /// Physical source-image displacement, in normalised coordinates.
  Vec3 source_shift{};
  /// Source box width divided by target box width.  Uniform interactions use 1.
  double source_to_target_width{1.0};
};

/**
 * @brief One topology-native occupied source/target leaf interaction.
 *
 * This record describes the occupied ranges and periodic identity metadata
 * owned by topology.  Derived P2P execution packings convert it to their own
 * representation at the plan boundary.
 */
struct StaticP2PLeafRecord {
  int target_leaf{-1};
  int source_leaf{-1};
  std::size_t target_begin{0};
  std::size_t target_count{0};
  std::size_t source_begin{0};
  std::size_t source_count{0};
  Vec3 source_shift{};
  /// Integer periodic image identity; zero denotes the central image.
  std::array<int, 3> image_shift{{0, 0, 0}};
  bool skip_for_identity{true};
};

/**
 * @brief Canonical immutable topology consumed by static FMM operators.
 *
 * Node IDs are opaque compact IDs. The uniform builder currently chooses the
 * historical flat IDs, but no executor relies on level_offset or complete
 * levels. All arrays preserve the deterministic order of the existing
 * UniformTree traversal.
 */
struct StaticFmmTopology {
  struct Node {
    int index{-1};
    int level{0};
    int parent{-1};
    std::array<int, 8> children{{-1, -1, -1, -1, -1, -1, -1, -1}};
    Vec3 centre{};
    double half_width{0.0};
    std::size_t source_begin{0};
    std::size_t source_end{0};
    std::size_t target_begin{0};
    std::size_t target_end{0};

    [[nodiscard]] std::size_t source_count() const noexcept {
      return source_end - source_begin;
    }
    [[nodiscard]] std::size_t target_count() const noexcept {
      return target_end - target_begin;
    }
  };

  std::vector<Node> nodes{};
  std::vector<Vec3> sorted_source_positions{};
  std::vector<Vec3> sorted_target_positions{};
  std::vector<int> source_permutation{};
  std::vector<int> source_inverse_permutation{};
  std::vector<int> target_permutation{};
  std::vector<int> target_inverse_permutation{};
  std::vector<StaticLeafRange> source_leaves{};
  std::vector<StaticLeafRange> target_leaves{};
  std::vector<StaticTranslationEdge> m2m_edges{};
  std::vector<int> m2m_parent_nodes{};
  std::vector<int> m2m_parent_level_offsets{};
  std::vector<int> m2m_parent_edge_offsets{};
  std::vector<StaticTranslationEdge> l2l_edges{};
  std::vector<StaticM2LInteraction> m2l_interactions{};
  std::vector<int> m2l_target_row_offsets{};
  /// Offsets indexed by child level; entry zero is always zero.
  std::vector<int> m2m_level_offsets{};
  /// Offsets indexed by child level; entry zero is always zero.
  std::vector<int> l2l_level_offsets{};
  std::vector<StaticP2PLeafRecord> p2p_leaf_records{};
  std::vector<int> p2p_target_leaf_offsets{};
  std::vector<int> p2p_target_leaves{};

  /// Physical position = coordinate_origin + coordinate_scale * stored position.
  Vec3 coordinate_origin{};
  double coordinate_scale{1.0};
  int root{0};
  int maximum_level{0};

  [[nodiscard]] std::span<const StaticLeafRange> occupied_source_leaves() const noexcept {
    return source_leaves;
  }
  [[nodiscard]] std::span<const StaticLeafRange> occupied_target_leaves() const noexcept {
    return target_leaves;
  }

  /** @brief Returns retained canonical topology and geometry storage. */
  [[nodiscard]] std::size_t memory_bytes() const noexcept {
    return nodes.capacity() * sizeof(Node) +
        (sorted_source_positions.capacity() + sorted_target_positions.capacity()) *
            sizeof(Vec3) +
        (source_permutation.capacity() + source_inverse_permutation.capacity() +
         target_permutation.capacity() + target_inverse_permutation.capacity()) *
            sizeof(int) +
        (source_leaves.capacity() + target_leaves.capacity()) *
            sizeof(StaticLeafRange) +
        (m2m_edges.capacity() + l2l_edges.capacity()) *
            sizeof(StaticTranslationEdge) +
        m2l_interactions.capacity() * sizeof(StaticM2LInteraction) +
        (m2m_parent_nodes.capacity() + m2m_parent_level_offsets.capacity() +
         m2m_parent_edge_offsets.capacity() + m2m_level_offsets.capacity() +
         l2l_level_offsets.capacity() + m2l_target_row_offsets.capacity() +
         p2p_target_leaf_offsets.capacity() + p2p_target_leaves.capacity()) *
            sizeof(int) +
        p2p_leaf_records.capacity() * sizeof(StaticP2PLeafRecord);
  }

  /** @brief Checks all IDs, ranges, edge endpoints and row offsets. */
  void validate() const;
};

/** @brief Extracts canonical topology and geometry from a uniform tree. */
[[nodiscard]] StaticFmmTopology build_uniform_fmm_topology(
    const UniformTree& tree,
    const PeriodicCellOptions& periodic = {});

} // namespace cdfmm
