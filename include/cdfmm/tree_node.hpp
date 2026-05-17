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
    int index{-1};
    int level{0};
    int parent{-1};
    std::array<int, 8> children{{-1, -1, -1, -1, -1, -1, -1, -1}};
    int ix{0};
    int iy{0};
    int iz{0};
    std::uint64_t morton_index{0};
    Vec3 centre{};
    double half_width{0.0};
    std::size_t source_begin{0};
    std::size_t source_end{0};
    std::size_t target_begin{0};
    std::size_t target_end{0};
    std::vector<int> list1{};
    std::vector<int> list2{};

    /// @brief Returns true when this node has no valid children.
    [[nodiscard]] bool is_leaf() const;

    /// @brief Number of Morton-sorted source points in this node.
    [[nodiscard]] std::size_t source_count() const;

    /// @brief Number of Morton-sorted target points in this node.
    [[nodiscard]] std::size_t target_count() const;
};

} // namespace cdfmm
