// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "cdfmm/vec3.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Public types
//------------------------------------------------------------------------------

/**
 * @brief Node metadata for a complete uniform octree layer in Morton order.
 *
 * Each node stores its topological links (`parent` and `children`), integer box
 * coordinates (`ix`, `iy`, `iz`), and a Morton index within the level. The
 * source and target ranges refer to half-open intervals in Morton-sorted point
 * arrays owned by `UniformTree`.
 *
 * `list1` contains same-level near neighbours (including self) in the
 * 3x3x3 neighbourhood where valid. `list2` contains same-level interaction
 * neighbours obtained from children of the parent `list1`, excluding `list1`.
 */
struct TreeNode {
    /// @brief Flat level-wise node index used by all topology lists.
    int index{-1};
    /// @brief Tree level, where zero denotes the root.
    int level{0};
    /// @brief Flat parent index, or -1 for the root.
    int parent{-1};
    /// @brief Flat child indices in local Morton order, or -1 at a leaf.
    std::array<int, 8> children{{-1, -1, -1, -1, -1, -1, -1, -1}};
    /// @brief Integer box coordinate along x within this level.
    int ix{0};
    /// @brief Integer box coordinate along y within this level.
    int iy{0};
    /// @brief Integer box coordinate along z within this level.
    int iz{0};
    /// @brief Morton index relative to the start of this level.
    std::uint64_t morton_index{0};
    /// @brief Physical box centre.
    Vec3 centre{};
    /// @brief Half of the cubic box side length.
    double half_width{0.0};
    /// @brief Start of the half-open source range in sorted storage.
    std::size_t source_begin{0};
    /// @brief End of the half-open source range in sorted storage.
    std::size_t source_end{0};
    /// @brief Start of the half-open target range in sorted storage.
    std::size_t target_begin{0};
    /// @brief End of the half-open target range in sorted storage.
    std::size_t target_end{0};
    /// @brief Same-level touching boxes, including this box.
    std::vector<int> list1{};
    /// @brief Same-level well-separated children of parent neighbours.
    std::vector<int> list2{};

    /// @brief Returns true when this node has no valid children.
    [[nodiscard]] bool is_leaf() const;

    /// @brief Number of Morton-sorted source points in this node.
    [[nodiscard]] std::size_t source_count() const;

    /// @brief Number of Morton-sorted target points in this node.
    [[nodiscard]] std::size_t target_count() const;
};

} // namespace cdfmm
