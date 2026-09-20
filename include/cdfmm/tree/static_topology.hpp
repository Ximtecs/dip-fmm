// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "cdfmm/math/vec3.hpp"

namespace cdfmm {

/** @brief Compact source or target range owned by one static-tree leaf. */
struct StaticLeafRange {
  int node{-1};           ///< Compact node id of the leaf.
  std::size_t begin{0};   ///< First sorted particle index of the range.
  std::size_t count{0};   ///< Number of sorted particles in the range.
};

/** @brief One explicit parent/child translation edge. */
struct StaticTranslationEdge {
  int source_node{-1};   ///< Node whose coefficients are read (child for M2M, parent for L2L).
  int target_node{-1};   ///< Node whose coefficients are accumulated.
  int child_class{0};    ///< Octant of the child within its parent, 0..7, bit i set for +axis i.
  int child_level{0};    ///< Tree level of the child endpoint.

  /// Two edges are equal when every endpoint and class agrees.
  [[nodiscard]] bool operator==(const StaticTranslationEdge&) const = default;
};

/** @brief One explicit far-field multipole-to-local interaction. */
struct StaticM2LInteraction {
  int source_node{-1};   ///< Node whose multipole is translated.
  int target_node{-1};   ///< Node whose local receives the translation.
  int source_level{0};   ///< Tree level of the source node.
  int target_level{0};   ///< Tree level of the target node.
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
  int target_leaf{-1};          ///< Compact node id of the target leaf.
  int source_leaf{-1};          ///< Compact node id of the source leaf.
  std::size_t target_begin{0};  ///< First sorted target of the target leaf.
  std::size_t target_count{0};  ///< Number of targets in the target leaf.
  std::size_t source_begin{0};  ///< First sorted source of the source leaf.
  std::size_t source_count{0};  ///< Number of sources in the source leaf.
  Vec3 source_shift{};          ///< Periodic image shift added to every source position; zero in free space.
  /// Integer periodic image identity; zero denotes the central image.
  std::array<int, 3> image_shift{{0, 0, 0}};
  /// Whether a target whose identity map names one of these sources omits
  /// that pair (a singular point self pair).  False for finite sources, whose
  /// self field is physical, and for every non-central image.
  bool skip_for_identity{true};
};

/**
 * @brief Canonical immutable topology consumed by static FMM operators.
 *
 * Node IDs are opaque compact IDs. The uniform adapter currently chooses the
 * historical flat IDs, but no executor relies on level_offset or complete
 * levels. All arrays preserve the deterministic order of the existing
 * complete-tree traversal.
 */
struct StaticFmmTopology {
  /** @brief One tree node with its box and sorted particle ranges. */
  struct Node {
    int index{-1};                  ///< Compact node id; equals the position in `nodes`.
    int level{0};                   ///< Depth below the root, root at 0.
    int parent{-1};                 ///< Parent node id, -1 for the root.
    std::array<int, 8> children{{-1, -1, -1, -1, -1, -1, -1, -1}};  ///< Child ids by octant, -1 when absent.
    Vec3 centre{};                  ///< Box centre in normalised coordinates.
    double half_width{0.0};         ///< Half the box side length in normalised coordinates.
    std::size_t source_begin{0};    ///< First sorted source inside the box.
    std::size_t source_end{0};      ///< One past the last sorted source inside the box.
    std::size_t target_begin{0};    ///< First sorted target inside the box.
    std::size_t target_end{0};      ///< One past the last sorted target inside the box.

    /// Number of sources inside the box.
    [[nodiscard]] std::size_t source_count() const noexcept {
      return source_end - source_begin;
    }
    /// Number of targets inside the box.
    [[nodiscard]] std::size_t target_count() const noexcept {
      return target_end - target_begin;
    }
  };

  std::vector<Node> nodes{};                       ///< Every node, indexed by compact id.
  std::vector<Vec3> sorted_source_positions{};    ///< Source positions in leaf (Morton) order, normalised.
  std::vector<Vec3> sorted_target_positions{};    ///< Target positions in leaf (Morton) order, normalised.
  std::vector<int> source_permutation{};           ///< Sorted source index -> user source index.
  std::vector<int> source_inverse_permutation{};   ///< User source index -> sorted source index.
  std::vector<int> target_permutation{};           ///< Sorted target index -> user target index.
  std::vector<int> target_inverse_permutation{};   ///< User target index -> sorted target index.
  std::vector<StaticLeafRange> source_leaves{};    ///< Occupied source leaves with their sorted ranges.
  std::vector<StaticLeafRange> target_leaves{};    ///< Occupied target leaves with their sorted ranges.
  std::vector<StaticTranslationEdge> m2m_edges{};  ///< Child -> parent edges, grouped by parent (see the offsets below).
  std::vector<int> m2m_parent_nodes{};             ///< Parents with at least one M2M edge, ordered by level.
  std::vector<int> m2m_parent_level_offsets{};     ///< Range of `m2m_parent_nodes` per parent level (CSR offsets).
  std::vector<int> m2m_parent_edge_offsets{};      ///< Range of `m2m_edges` per entry of `m2m_parent_nodes` (CSR offsets).
  std::vector<StaticTranslationEdge> l2l_edges{};  ///< Parent -> child edges, grouped by child level.
  std::vector<StaticM2LInteraction> m2l_interactions{};  ///< Every list-2 interaction, grouped by target node.
  std::vector<int> m2l_target_row_offsets{};       ///< Range of `m2l_interactions` per target node (CSR offsets).
  /// Offsets indexed by child level; entry zero is always zero.
  std::vector<int> m2m_level_offsets{};
  /// Offsets indexed by child level; entry zero is always zero.
  std::vector<int> l2l_level_offsets{};
  std::vector<StaticP2PLeafRecord> p2p_leaf_records{};  ///< Every list-1 leaf pair (one per periodic image), grouped by target leaf.
  std::vector<int> p2p_target_leaf_offsets{};      ///< Range of `p2p_leaf_records` per entry of `p2p_target_leaves` (CSR offsets).
  std::vector<int> p2p_target_leaves{};            ///< Target leaves with at least one list-1 record.

  /// Physical position = coordinate_origin + coordinate_scale * stored position.
  Vec3 coordinate_origin{};
  double coordinate_scale{1.0};   ///< Physical length of one normalised unit (the root side length).
  int root{0};                    ///< Compact id of the root node.
  int maximum_level{0};           ///< Deepest level present (the leaf level of a complete tree).

  /// The occupied source leaves; an alias of `source_leaves`.
  [[nodiscard]] std::span<const StaticLeafRange> occupied_source_leaves() const noexcept {
    return source_leaves;
  }
  /// The occupied target leaves; an alias of `target_leaves`.
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

} // namespace cdfmm
