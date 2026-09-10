// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/tree/uniform_tree.hpp"

#include "../common/root_box.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace cdfmm {

//------------------------------------------------------------------------------
// Tree-construction helpers
//------------------------------------------------------------------------------

namespace {

int boxes_per_dim(const int level)
{
    return 1 << level;
}

std::size_t total_nodes(const int max_level)
{
    std::size_t total = 0;
    std::size_t at_level = 1;
    for (int level = 0; level <= max_level; ++level) {
        total += at_level;
        at_level *= 8;
    }
    return total;
}

} // namespace

UniformTree::UniformTree(const std::vector<Vec3>& source_positions, const UniformTreeOptions& options)
{
    build(source_positions, {}, options);
}

UniformTree::UniformTree(const std::vector<Vec3>& source_positions, const std::vector<Vec3>& target_positions, const UniformTreeOptions& options)
{
    build(source_positions, target_positions, options);
}

//------------------------------------------------------------------------------
// Complete uniform-tree construction
//------------------------------------------------------------------------------

void UniformTree::build(const std::vector<Vec3>& source_positions, const std::vector<Vec3>& target_positions, const UniformTreeOptions& options)
{
    using Clock = std::chrono::steady_clock;
    const auto total_start = Clock::now();
    if (options.max_level < 0) {
        throw std::invalid_argument("UniformTreeOptions.max_level must be >= 0");
    }

    max_level_ = options.max_level;
    const auto bounds_start = Clock::now();

    const RootBox root_box = resolve_root_box(
        source_positions, target_positions,
        options.root_centre, options.root_half_width);
    root_centre_ = root_box.centre;
    root_half_width_ = root_box.half_width;
    build_timings_.root_bounds.add(
        std::chrono::duration<double>(Clock::now() - bounds_start).count()
    );

    const auto assign_leaf = [&](const Vec3& point) {
        if (max_level_ == 0) {
            return std::array<int, 3>{0, 0, 0};
        }
        const int n = boxes_per_dim(max_level_);
        const double cell_width = (2.0 * root_half_width_) / static_cast<double>(n);
        const Vec3 root_min = root_centre_ -
            Vec3{root_half_width_, root_half_width_, root_half_width_};
        const auto index_for = [&](double value, double lo) {
            // floor assigns internal boundaries to the box on their positive
            // side.  Clamp the closed upper root face to index n-1.
            int index = static_cast<int>(std::floor((value - lo) / cell_width));
            if (index < 0) {
                index = 0;
            }
            if (index >= n) {
                index = n - 1;
            }
            return index;
        };
        return std::array<int, 3>{index_for(point.x, root_min.x), index_for(point.y, root_min.y), index_for(point.z, root_min.z)};
    };

    // Materialise every box, including empty boxes.  Consequently a flat node
    // index is computable directly, with no sparse lookup table.
    const auto nodes_start = Clock::now();
    nodes_.assign(total_nodes(max_level_), TreeNode{});
    for (int level = 0; level <= max_level_; ++level) {
        const int n = boxes_per_dim(level);
        #pragma omp parallel for collapse(3) schedule(static) if(n >= 8)
        for (int iz = 0; iz < n; ++iz) {
            for (int iy = 0; iy < n; ++iy) {
                for (int ix = 0; ix < n; ++ix) {
                    const int idx = node_index(level, ix, iy, iz);
                    TreeNode& node = nodes_.at(idx);
                    node.index = idx;
                    node.level = level;
                    node.ix = ix;
                    node.iy = iy;
                    node.iz = iz;
                    node.morton_index = morton_encode(ix, iy, iz);
                    // Integer division removes the lowest coordinate bit and
                    // therefore gives the parent in a complete octree.
                    node.parent = (level == 0)
                        ? -1
                        : node_index(level - 1, ix / 2, iy / 2, iz / 2);
                    const double child_width = root_half_width_ / static_cast<double>(1 << level);
                    node.half_width = child_width;
                    const Vec3 offset{
                        ((static_cast<double>(ix) + 0.5) / static_cast<double>(n) - 0.5) * 2.0 * root_half_width_,
                        ((static_cast<double>(iy) + 0.5) / static_cast<double>(n) - 0.5) * 2.0 * root_half_width_,
                        ((static_cast<double>(iz) + 0.5) / static_cast<double>(n) - 0.5) * 2.0 * root_half_width_};
                    node.centre = root_centre_ + offset;
                }
            }
        }
    }
    build_timings_.node_construction.add(
        std::chrono::duration<double>(Clock::now() - nodes_start).count()
    );

    // Child slot bits encode local (dx,dy,dz), matching Morton bit order.
    const auto topology_start = Clock::now();
    #pragma omp parallel for schedule(static) if(nodes_.size() >= 512)
    for (std::ptrdiff_t node_index_value = 0;
         node_index_value < static_cast<std::ptrdiff_t>(nodes_.size());
         ++node_index_value) {
        TreeNode& node = nodes_[static_cast<std::size_t>(node_index_value)];
        if (node.level == max_level_) {
            node.children.fill(-1);
        } else {
            for (int child = 0; child < 8; ++child) {
                const int dx = child & 1;
                const int dy = (child >> 1) & 1;
                const int dz = (child >> 2) & 1;
                node.children[child] = node_index(
                    node.level + 1,
                    2 * node.ix + dx,
                    2 * node.iy + dy,
                    2 * node.iz + dz
                );
            }
        }
    }
    build_timings_.topology.add(
        std::chrono::duration<double>(Clock::now() - topology_start).count()
    );

    // Sort each population independently by leaf Morton index.  Stable sorting
    // preserves user order for points sharing a leaf, making the convention
    // deterministic.  permutation[sorted] = original, while
    // inverse[original] = sorted.
    auto sort_points = [&](
        const std::vector<Vec3>& original,
        std::vector<Vec3>& sorted,
        std::vector<int>& permutation,
        std::vector<int>& inverse,
        std::vector<int>& leaf_indices,
        PhaseTiming& morton_timing,
        PhaseTiming& sorting_timing
    ) {
        const auto morton_phase_start = Clock::now();
        std::vector<std::tuple<std::uint64_t, int, int>> keyed;
        keyed.reserve(original.size());
        for (std::size_t i = 0; i < original.size(); ++i) {
            const auto ijk = assign_leaf(original[i]);
            const auto morton = morton_encode(ijk[0], ijk[1], ijk[2]);
            const int leaf_index = node_index(max_level_, ijk[0], ijk[1], ijk[2]);
            keyed.emplace_back(morton, static_cast<int>(i), leaf_index);
        }
        morton_timing.add(
            std::chrono::duration<double>(Clock::now() - morton_phase_start).count()
        );
        const auto sorting_phase_start = Clock::now();
        std::stable_sort(
            keyed.begin(),
            keyed.end(),
            [](const auto& a, const auto& b) {
                return std::get<0>(a) < std::get<0>(b);
            }
        );
        sorting_timing.add(
            std::chrono::duration<double>(Clock::now() - sorting_phase_start).count()
        );

        sorted.resize(original.size());
        permutation.resize(original.size());
        inverse.resize(original.size());
        leaf_indices.resize(original.size());
        for (std::size_t sorted_index = 0; sorted_index < keyed.size(); ++sorted_index) {
            const int original_index = std::get<1>(keyed[sorted_index]);
            sorted[sorted_index] = original[original_index];
            permutation[sorted_index] = original_index;
            inverse[original_index] = static_cast<int>(sorted_index);
            leaf_indices[sorted_index] = std::get<2>(keyed[sorted_index]);
        }
    };

    sort_points(
        source_positions,
        source_positions_sorted_,
        source_permutation_,
        source_inverse_permutation_,
        source_leaf_indices_,
        build_timings_.source_morton,
        build_timings_.source_sorting
    );
    sort_points(
        target_positions,
        target_positions_sorted_,
        target_permutation_,
        target_inverse_permutation_,
        target_leaf_indices_,
        build_timings_.target_morton,
        build_timings_.target_sorting
    );

    // Empty ranges use [population_size, population_size).  Non-empty leaf
    // ranges overwrite these sentinels before ranges are propagated upwards.
    const auto ranges_start = Clock::now();
    #pragma omp parallel for schedule(static) if(nodes_.size() >= 512)
    for (std::ptrdiff_t node_index_value = 0;
         node_index_value < static_cast<std::ptrdiff_t>(nodes_.size());
         ++node_index_value) {
        TreeNode& node = nodes_[static_cast<std::size_t>(node_index_value)];
        node.source_begin = source_positions_sorted_.size();
        node.source_end = source_positions_sorted_.size();
        node.target_begin = target_positions_sorted_.size();
        node.target_end = target_positions_sorted_.size();
    }

    auto assign_ranges = [&](const std::vector<int>& leaf_for_sorted, bool is_source) {
        // Equal Morton keys are contiguous, so each occupied leaf receives one
        // half-open interval in the appropriate sorted point array.
        std::size_t cursor = 0;
        while (cursor < leaf_for_sorted.size()) {
            const int leaf = leaf_for_sorted[cursor];
            std::size_t next = cursor + 1;
            while (next < leaf_for_sorted.size() && leaf_for_sorted[next] == leaf) {
                ++next;
            }
            TreeNode& node = nodes_[leaf];
            if (is_source) {
                node.source_begin = cursor;
                node.source_end = next;
            } else {
                node.target_begin = cursor;
                node.target_end = next;
            }
            cursor = next;
        }

        // A depth-first Morton ordering makes all points below a parent
        // contiguous.  Taking the minimum occupied-child begin and maximum
        // occupied-child end therefore forms the exact parent range.  Empty
        // children use the population-size sentinel and must be ignored;
        // otherwise that sentinel would incorrectly extend a populated
        // parent's range to the end of the complete sorted array.
        for (int level = max_level_ - 1; level >= 0; --level) {
            const int begin = level_offset(level);
            const int end = level_offset(level + 1);
            #pragma omp parallel for schedule(static) if(end - begin >= 64)
            for (int idx = begin; idx < end; ++idx) {
                TreeNode& node = nodes_[idx];
                std::size_t min_begin = is_source ? source_positions_sorted_.size() : target_positions_sorted_.size();
                std::size_t max_end = 0;
                bool has_occupied_child = false;
                for (const int child : node.children) {
                    const TreeNode& child_node = nodes_[child];
                    const std::size_t child_begin = is_source ? child_node.source_begin : child_node.target_begin;
                    const std::size_t child_end = is_source ? child_node.source_end : child_node.target_end;
                    if (child_begin == child_end) {
                        continue;
                    }
                    has_occupied_child = true;
                    min_begin = std::min(min_begin, child_begin);
                    max_end = std::max(max_end, child_end);
                }
                if (!has_occupied_child) {
                    max_end = min_begin;
                }
                if (is_source) {
                    node.source_begin = min_begin;
                    node.source_end = max_end;
                } else {
                    node.target_begin = min_begin;
                    node.target_end = max_end;
                }
            }
        }
    };

    assign_ranges(source_leaf_indices_, true);
    assign_ranges(target_leaf_indices_, false);
    build_timings_.ranges.add(
        std::chrono::duration<double>(Clock::now() - ranges_start).count()
    );

    const int leaf_begin = level_offset(max_level_);
    const int leaf_end = static_cast<int>(nodes_.size());
    leaf_indices_.resize(static_cast<std::size_t>(leaf_end - leaf_begin));
    occupied_source_leaves_.reserve(leaf_indices_.size());
    occupied_target_leaves_.reserve(leaf_indices_.size());
    for (int index = leaf_begin; index < leaf_end; ++index) {
        leaf_indices_[static_cast<std::size_t>(index - leaf_begin)] = index;
        if (nodes_[static_cast<std::size_t>(index)].source_count() > 0) {
            occupied_source_leaves_.push_back(index);
        }
        if (nodes_[static_cast<std::size_t>(index)].target_count() > 0) {
            occupied_target_leaves_.push_back(index);
        }
    }

    // list1 is the clipped 3x3x3 same-level neighbourhood.  It includes the
    // node itself and has fewer than 27 entries at physical root boundaries.
    // These boxes define the direct near field in a uniform FMM.
    const auto lists_start = Clock::now();
    for (int level = 0; level <= max_level_; ++level) {
        const int begin = level_offset(level);
        const int end = level_offset(level + 1);
#pragma omp parallel for schedule(static) if (end - begin >= 64)
        for (int index = begin; index < end; ++index) {
            TreeNode& node = nodes_[static_cast<std::size_t>(index)];
            const int n = boxes_per_dim(level);
            node.list1.clear();
            node.list1.reserve(27);
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = node.ix + dx;
                        const int ny = node.iy + dy;
                        const int nz = node.iz + dz;
                        if (nx >= 0 && nx < n && ny >= 0 && ny < n &&
                            nz >= 0 && nz < n) {
                            node.list1.push_back(
                                node_index(node.level, nx, ny, nz)
                            );
                        }
                    }
                }
            }
            // Flat node indices define the historical deterministic list order.
            std::sort(node.list1.begin(), node.list1.end());

            if (node.level == 0) {
                node.list2.clear();
                continue;
            }

        // list2 contains children of the parent's list1 boxes minus this
        // node's own list1.  Thus every entry is well separated at this level
        // but its parent is adjacent to this node's parent, which is the
        // classical uniform-tree M2L interaction criterion.
            std::vector<int> candidates;
            candidates.reserve(27 * 8);
            const TreeNode& parent = nodes_.at(node.parent);
            for (const int parent_neighbour : parent.list1) {
                const TreeNode& parent_node = nodes_.at(parent_neighbour);
                for (const int child : parent_node.children) {
                    candidates.push_back(child);
                }
            }
            // Parent neighbours and their children are unique in a uniform tree.
            // Sorting a bounded contiguous buffer preserves historical order
            // without balanced-tree node allocations or pointer chasing.
            std::sort(candidates.begin(), candidates.end());
            node.list2.clear();
            node.list2.reserve(189);
            std::set_difference(
                candidates.begin(), candidates.end(),
                node.list1.begin(), node.list1.end(),
                std::back_inserter(node.list2)
            );
        }
    }
    build_timings_.interaction_lists.add(
        std::chrono::duration<double>(Clock::now() - lists_start).count()
    );
    build_timings_.total.add(
        std::chrono::duration<double>(Clock::now() - total_start).count()
    );
}

//------------------------------------------------------------------------------
// Public tree queries
//------------------------------------------------------------------------------

int UniformTree::max_level() const
{
    return max_level_;
}

int UniformTree::n_levels() const
{
    return max_level_ + 1;
}

int UniformTree::leaf_level() const
{
    return max_level_;
}

const Vec3& UniformTree::root_centre() const
{
    return root_centre_;
}

double UniformTree::root_half_width() const
{
    return root_half_width_;
}

std::span<const TreeNode> UniformTree::nodes() const
{
    return nodes_;
}

std::span<const Vec3> UniformTree::sorted_source_positions() const
{
    return source_positions_sorted_;
}

std::span<const Vec3> UniformTree::sorted_target_positions() const
{
    return target_positions_sorted_;
}

std::span<const int> UniformTree::source_permutation() const
{
    return source_permutation_;
}

std::span<const int> UniformTree::source_inverse_permutation() const
{
    return source_inverse_permutation_;
}

std::span<const int> UniformTree::target_permutation() const
{
    return target_permutation_;
}

std::span<const int> UniformTree::target_inverse_permutation() const
{
    return target_inverse_permutation_;
}

std::span<const int> UniformTree::leaf_indices() const
{
    return leaf_indices_;
}

std::span<const int> UniformTree::occupied_source_leaves() const
{
    return occupied_source_leaves_;
}

std::span<const int> UniformTree::occupied_target_leaves() const
{
    return occupied_target_leaves_;
}

const TreeBuildTimings& UniformTree::build_timings() const
{
    return build_timings_;
}

TreeMemoryStatistics UniformTree::memory_statistics() const noexcept
{
    TreeMemoryStatistics result;
    result.node_bytes = nodes_.capacity() * sizeof(TreeNode);
    result.position_bytes =
        (source_positions_sorted_.capacity() +
         target_positions_sorted_.capacity()) * sizeof(Vec3);
    result.index_bytes =
        (source_permutation_.capacity() +
         source_inverse_permutation_.capacity() +
         source_leaf_indices_.capacity() + target_permutation_.capacity() +
         target_inverse_permutation_.capacity() +
         target_leaf_indices_.capacity() + leaf_indices_.capacity() +
         occupied_source_leaves_.capacity() +
         occupied_target_leaves_.capacity()) * sizeof(int);
    for (const TreeNode& node : nodes_) {
        result.interaction_bytes +=
            (node.list1.capacity() + node.list2.capacity()) * sizeof(int);
    }
    return result;
}

int UniformTree::leaf_index_for_source(
    const std::size_t sorted_source_index
) const
{
    return source_leaf_indices_.at(sorted_source_index);
}

int UniformTree::leaf_index_for_target(
    const std::size_t sorted_target_index
) const
{
    return target_leaf_indices_.at(sorted_target_index);
}

} // namespace cdfmm
