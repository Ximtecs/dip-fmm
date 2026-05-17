// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <optional>
#include <span>
#include <vector>

#include "cdfmm/tree_node.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Public types
//------------------------------------------------------------------------------

/**
 * @brief Options controlling complete uniform tree construction.
 */
struct UniformTreeOptions {
    int max_level{1};
    bool include_empty_nodes{true};
    bool cubic_root_box{true};
    std::optional<Vec3> root_centre{};
    std::optional<double> root_half_width{};
};

//------------------------------------------------------------------------------
// Public functions
//------------------------------------------------------------------------------

/**
 * @brief Morton bit interleave for non-negative integer coordinates.
 */
std::uint64_t morton_encode(int ix, int iy, int iz);

/**
 * @brief Inverts `morton_encode` into integer coordinates.
 */
std::array<int, 3> morton_decode(std::uint64_t morton);

/**
 * @brief Returns level offset `(8^level - 1) / 7` for complete octree storage.
 */
int level_offset(int level);

/**
 * @brief Returns node index for a level and integer box coordinate.
 */
int node_index(int level, int ix, int iy, int iz);

/**
 * @brief Complete uniform octree with level-wise Morton ordering.
 */
class UniformTree {
  public:
    UniformTree(
        const std::vector<Vec3>& source_positions,
        const UniformTreeOptions& options
    );

    UniformTree(
        const std::vector<Vec3>& source_positions,
        const std::vector<Vec3>& target_positions,
        const UniformTreeOptions& options
    );

    [[nodiscard]] int max_level() const;
    [[nodiscard]] int n_levels() const;
    [[nodiscard]] int leaf_level() const;
    [[nodiscard]] const Vec3& root_centre() const;
    [[nodiscard]] double root_half_width() const;
    [[nodiscard]] std::span<const TreeNode> nodes() const;
    [[nodiscard]] std::span<const Vec3> sorted_source_positions() const;
    [[nodiscard]] std::span<const Vec3> sorted_target_positions() const;
    [[nodiscard]] std::span<const int> source_permutation() const;
    [[nodiscard]] std::span<const int> source_inverse_permutation() const;
    [[nodiscard]] std::span<const int> target_permutation() const;
    [[nodiscard]] std::span<const int> target_inverse_permutation() const;
    [[nodiscard]] std::vector<int> leaf_indices() const;
    [[nodiscard]] int leaf_index_for_source(std::size_t sorted_source_index) const;
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
