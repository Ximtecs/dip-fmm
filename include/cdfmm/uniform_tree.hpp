// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <optional>
#include <span>
#include <vector>

#include "cdfmm/tree_node.hpp"
#include "cdfmm/timings.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Public types
//------------------------------------------------------------------------------

/**
 * @brief Options controlling complete uniform tree construction.
 *
 * The current implementation always materialises every box through
 * `max_level`; `include_empty_nodes` and `cubic_root_box` are reserved for
 * future policies and do not alter construction yet.  Unless explicitly
 * supplied, the root
 * cube is the smallest axis-aligned cube centred on the combined source and
 * target bounding box that contains every point.
 */
struct UniformTreeOptions {
    /// @brief Deepest level, with the root at level zero.
    int max_level{1};
    /// @brief Requests storage of empty boxes (currently always done).
    bool include_empty_nodes{true};
    /// @brief Requests a cubic root box (currently the only representation).
    bool cubic_root_box{true};
    /// @brief Optional fixed root centre; otherwise inferred from all points.
    std::optional<Vec3> root_centre{};
    /// @brief Optional positive root half-width; otherwise inferred.
    std::optional<double> root_half_width{};
};

/** @brief Retained storage owned by one immutable uniform tree. */
struct TreeMemoryStatistics {
    std::size_t node_bytes{0};
    std::size_t position_bytes{0};
    std::size_t index_bytes{0};
    std::size_t interaction_bytes{0};

    [[nodiscard]] std::size_t total_bytes() const noexcept {
        return node_bytes + position_bytes + index_bytes + interaction_bytes;
    }
};

//------------------------------------------------------------------------------
// Public functions
//------------------------------------------------------------------------------

/**
 * @brief Interleaves 21 bits from each non-negative box coordinate.
 *
 * Bits are stored in x, y, z order from least to most significant.  The
 * result is the box's zero-based Morton index within one tree level.
 */
std::uint64_t morton_encode(int ix, int iy, int iz);

/**
 * @brief Inverts `morton_encode` into `(ix, iy, iz)` box coordinates.
 */
std::array<int, 3> morton_decode(std::uint64_t morton);

/**
 * @brief Returns level offset `(8^level - 1) / 7` for complete octree storage.
 */
int level_offset(int level);

/**
 * @brief Returns the flat node index for a level and integer box coordinate.
 *
 * Nodes are stored level by level, and boxes within a level are in Morton
 * order: `level_offset(level) + morton_encode(ix, iy, iz)`.
 */
int node_index(int level, int ix, int iy, int iz);

/**
 * @brief Complete uniform octree with level-wise Morton ordering.
 *
 * Every level contains all `8^level` boxes, including empty boxes.  Source
 * and target positions are independently sorted by leaf Morton index.  Node
 * ranges are half-open intervals into those sorted arrays, so all descendants
 * of a node occupy one contiguous range.
 *
 * This class currently supplies geometry organisation and interaction lists;
 * it does not execute an upward or downward FMM pass.
 */
class UniformTree {
  public:
    /** @brief Builds a source-only tree using @p options. */
    UniformTree(
        const std::vector<Vec3>& source_positions,
        const UniformTreeOptions& options
    );

    /** @brief Builds a tree around independent source and target positions. */
    UniformTree(
        const std::vector<Vec3>& source_positions,
        const std::vector<Vec3>& target_positions,
        const UniformTreeOptions& options
    );

    /// @brief Returns the deepest materialised level.
    [[nodiscard]] int max_level() const;
    /// @brief Returns the number of levels, including the root.
    [[nodiscard]] int n_levels() const;
    /// @brief Returns the level containing the leaves.
    [[nodiscard]] int leaf_level() const;
    /// @brief Returns the geometric centre of the cubic root box.
    [[nodiscard]] const Vec3& root_centre() const;
    /// @brief Returns half the root cube side length.
    [[nodiscard]] double root_half_width() const;
    /// @brief Returns all nodes in flat level-wise Morton order.
    [[nodiscard]] std::span<const TreeNode> nodes() const;
    /// @brief Returns sources sorted by leaf Morton index.
    [[nodiscard]] std::span<const Vec3> sorted_source_positions() const;
    /// @brief Returns targets sorted by leaf Morton index.
    [[nodiscard]] std::span<const Vec3> sorted_target_positions() const;
    /// @brief Maps sorted source indices to original user indices.
    [[nodiscard]] std::span<const int> source_permutation() const;
    /// @brief Maps original source indices to sorted indices.
    [[nodiscard]] std::span<const int> source_inverse_permutation() const;
    /// @brief Maps sorted target indices to original user indices.
    [[nodiscard]] std::span<const int> target_permutation() const;
    /// @brief Maps original target indices to sorted indices.
    [[nodiscard]] std::span<const int> target_inverse_permutation() const;
    /// @brief Returns the flat indices of all leaf boxes in Morton order.
    [[nodiscard]] std::span<const int> leaf_indices() const;
    /// @brief Returns occupied source leaves cached during geometry setup.
    [[nodiscard]] std::span<const int> occupied_source_leaves() const;
    /// @brief Returns occupied target leaves cached during geometry setup.
    [[nodiscard]] std::span<const int> occupied_target_leaves() const;
    /// @brief Returns the wall-clock breakdown recorded during construction.
    [[nodiscard]] const TreeBuildTimings& build_timings() const;
    /// @brief Returns retained tree storage, including vector capacities.
    [[nodiscard]] TreeMemoryStatistics memory_statistics() const noexcept;
    /// @brief Returns the leaf containing a source at a sorted-array index.
    [[nodiscard]] int leaf_index_for_source(std::size_t sorted_source_index) const;
    /// @brief Returns the leaf containing a target at a sorted-array index.
    [[nodiscard]] int leaf_index_for_target(std::size_t sorted_target_index) const;

  private:
    std::vector<TreeNode> nodes_{};
    std::vector<Vec3> source_positions_sorted_{};
    std::vector<int> source_permutation_{};
    std::vector<int> source_inverse_permutation_{};
    std::vector<int> source_leaf_indices_{};
    std::vector<Vec3> target_positions_sorted_{};
    std::vector<int> target_permutation_{};
    std::vector<int> target_inverse_permutation_{};
    std::vector<int> target_leaf_indices_{};
    std::vector<int> leaf_indices_{};
    std::vector<int> occupied_source_leaves_{};
    std::vector<int> occupied_target_leaves_{};
    TreeBuildTimings build_timings_{};
    Vec3 root_centre_{};
    double root_half_width_{0.0};
    int max_level_{0};

    void build(
        const std::vector<Vec3>& source_positions,
        const std::vector<Vec3>& target_positions,
        const UniformTreeOptions& options
    );
};

} // namespace cdfmm
