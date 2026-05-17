// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/uniform_tree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>

namespace cdfmm {

namespace {

constexpr double kBoundaryTolerance = 1.0e-12;

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

bool TreeNode::is_leaf() const { return std::all_of(children.begin(), children.end(), [](int c) { return c < 0; }); }
std::size_t TreeNode::source_count() const { return source_end - source_begin; }
std::size_t TreeNode::target_count() const { return target_end - target_begin; }

std::uint64_t morton_encode(const int ix, const int iy, const int iz)
{
    std::uint64_t morton = 0;
    for (int bit = 0; bit < 21; ++bit) {
        const std::uint64_t xb = (static_cast<std::uint64_t>(ix) >> bit) & 1ULL;
        const std::uint64_t yb = (static_cast<std::uint64_t>(iy) >> bit) & 1ULL;
        const std::uint64_t zb = (static_cast<std::uint64_t>(iz) >> bit) & 1ULL;
        morton |= (xb << (3 * bit));
        morton |= (yb << (3 * bit + 1));
        morton |= (zb << (3 * bit + 2));
    }
    return morton;
}

std::array<int, 3> morton_decode(const std::uint64_t morton)
{
    int ix = 0;
    int iy = 0;
    int iz = 0;
    for (int bit = 0; bit < 21; ++bit) {
        ix |= static_cast<int>((morton >> (3 * bit)) & 1ULL) << bit;
        iy |= static_cast<int>((morton >> (3 * bit + 1)) & 1ULL) << bit;
        iz |= static_cast<int>((morton >> (3 * bit + 2)) & 1ULL) << bit;
    }
    return {ix, iy, iz};
}

int level_offset(const int level)
{
    std::size_t numerator = 1;
    for (int i = 0; i < level; ++i) {
        numerator *= 8;
    }
    return static_cast<int>((numerator - 1) / 7);
}

int node_index(const int level, const int ix, const int iy, const int iz)
{
    return level_offset(level) + static_cast<int>(morton_encode(ix, iy, iz));
}

UniformTree::UniformTree(const std::vector<Vec3>& source_positions, const UniformTreeOptions& options)
{
    build(source_positions, {}, options);
}

UniformTree::UniformTree(const std::vector<Vec3>& source_positions, const std::vector<Vec3>& target_positions, const UniformTreeOptions& options)
{
    build(source_positions, target_positions, options);
}

void UniformTree::build(const std::vector<Vec3>& source_positions, const std::vector<Vec3>& target_positions, const UniformTreeOptions& options)
{
    if (options.max_level < 0) {
        throw std::invalid_argument("UniformTreeOptions.max_level must be >= 0");
    }
    if (options.root_half_width.has_value() && options.root_half_width.value() <= 0.0) {
        throw std::invalid_argument("UniformTreeOptions.root_half_width must be positive");
    }

    max_level_ = options.max_level;
    const std::vector<Vec3>* combined_ptrs[2] = {&source_positions, &target_positions};
    Vec3 minimum{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    Vec3 maximum{-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    bool has_points = false;
    for (const auto* pts : combined_ptrs) {
        for (const Vec3& point : *pts) {
            has_points = true;
            minimum.x = std::min(minimum.x, point.x);
            minimum.y = std::min(minimum.y, point.y);
            minimum.z = std::min(minimum.z, point.z);
            maximum.x = std::max(maximum.x, point.x);
            maximum.y = std::max(maximum.y, point.y);
            maximum.z = std::max(maximum.z, point.z);
        }
    }
    if (!has_points) {
        minimum = {0.0, 0.0, 0.0};
        maximum = {0.0, 0.0, 0.0};
    }

    if (options.root_centre.has_value()) {
        root_centre_ = options.root_centre.value();
    } else {
        root_centre_ = (minimum + maximum) * 0.5;
    }

    if (options.root_half_width.has_value()) {
        root_half_width_ = options.root_half_width.value();
    } else {
        const Vec3 delta_max = maximum - root_centre_;
        const Vec3 delta_min = root_centre_ - minimum;
        root_half_width_ = std::max({delta_max.x, delta_max.y, delta_max.z, delta_min.x, delta_min.y, delta_min.z});
        if (!has_points) {
            root_half_width_ = 1.0;
        }
    }

    const Vec3 root_min = root_centre_ - Vec3{root_half_width_, root_half_width_, root_half_width_};
    const Vec3 root_max = root_centre_ + Vec3{root_half_width_, root_half_width_, root_half_width_};

    const auto assign_leaf = [&](const Vec3& point) {
        const auto in_range = [&](double value, double lo, double hi) {
            return value >= lo - kBoundaryTolerance && value <= hi + kBoundaryTolerance;
        };
        if (!in_range(point.x, root_min.x, root_max.x) || !in_range(point.y, root_min.y, root_max.y) || !in_range(point.z, root_min.z, root_max.z)) {
            throw std::invalid_argument("Point lies outside the requested root box");
        }
        if (max_level_ == 0) {
            return std::array<int, 3>{0, 0, 0};
        }
        const int n = boxes_per_dim(max_level_);
        const double cell_width = (2.0 * root_half_width_) / static_cast<double>(n);
        const auto index_for = [&](double value, double lo) {
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

    nodes_.assign(total_nodes(max_level_), TreeNode{});
    for (int level = 0; level <= max_level_; ++level) {
        const int n = boxes_per_dim(level);
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
                    node.parent = (level == 0) ? -1 : node_index(level - 1, ix / 2, iy / 2, iz / 2);
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

    for (TreeNode& node : nodes_) {
        if (node.level == max_level_) {
            node.children.fill(-1);
        } else {
            for (int child = 0; child < 8; ++child) {
                const int dx = child & 1;
                const int dy = (child >> 1) & 1;
                const int dz = (child >> 2) & 1;
                node.children[child] = node_index(node.level + 1, 2 * node.ix + dx, 2 * node.iy + dy, 2 * node.iz + dz);
            }
        }
    }

    auto sort_points = [&](const std::vector<Vec3>& original, std::vector<Vec3>& sorted, std::vector<int>& permutation, std::vector<int>& inverse, std::vector<int>& leaf_indices) {
        std::vector<std::tuple<std::uint64_t, int, int>> keyed;
        keyed.reserve(original.size());
        for (std::size_t i = 0; i < original.size(); ++i) {
            const auto ijk = assign_leaf(original[i]);
            const auto morton = morton_encode(ijk[0], ijk[1], ijk[2]);
            const int leaf_index = node_index(max_level_, ijk[0], ijk[1], ijk[2]);
            keyed.emplace_back(morton, static_cast<int>(i), leaf_index);
        }
        std::stable_sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

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

    sort_points(source_positions, source_positions_sorted_, source_permutation_, source_inverse_permutation_, source_leaf_indices_);
    sort_points(target_positions, target_positions_sorted_, target_permutation_, target_inverse_permutation_, target_leaf_indices_);

    for (TreeNode& node : nodes_) {
        node.source_begin = source_positions_sorted_.size();
        node.source_end = source_positions_sorted_.size();
        node.target_begin = target_positions_sorted_.size();
        node.target_end = target_positions_sorted_.size();
    }

    auto assign_ranges = [&](const std::vector<int>& leaf_for_sorted, bool is_source) {
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

        for (int level = max_level_ - 1; level >= 0; --level) {
            const int begin = level_offset(level);
            const int end = level_offset(level + 1);
            for (int idx = begin; idx < end; ++idx) {
                TreeNode& node = nodes_[idx];
                std::size_t min_begin = is_source ? source_positions_sorted_.size() : target_positions_sorted_.size();
                std::size_t max_end = min_begin;
                for (const int child : node.children) {
                    const TreeNode& child_node = nodes_[child];
                    const std::size_t child_begin = is_source ? child_node.source_begin : child_node.target_begin;
                    const std::size_t child_end = is_source ? child_node.source_end : child_node.target_end;
                    min_begin = std::min(min_begin, child_begin);
                    max_end = std::max(max_end, child_end);
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

    for (TreeNode& node : nodes_) {
        const int n = boxes_per_dim(node.level);
        std::set<int> l1;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = node.ix + dx;
                    const int ny = node.iy + dy;
                    const int nz = node.iz + dz;
                    if (nx >= 0 && nx < n && ny >= 0 && ny < n && nz >= 0 && nz < n) {
                        l1.insert(node_index(node.level, nx, ny, nz));
                    }
                }
            }
        }
        node.list1.assign(l1.begin(), l1.end());

        if (node.level == 0) {
            node.list2.clear();
            continue;
        }

        std::set<int> l2;
        const TreeNode& parent = nodes_.at(node.parent);
        for (const int parent_neighbour : parent.list1) {
            const TreeNode& parent_node = nodes_.at(parent_neighbour);
            for (const int child : parent_node.children) {
                l2.insert(child);
            }
        }
        for (const int near_node : node.list1) {
            l2.erase(near_node);
        }
        l2.erase(node.index);
        node.list2.assign(l2.begin(), l2.end());
    }
}

int UniformTree::max_level() const { return max_level_; }
int UniformTree::n_levels() const { return max_level_ + 1; }
int UniformTree::leaf_level() const { return max_level_; }
const Vec3& UniformTree::root_centre() const { return root_centre_; }
double UniformTree::root_half_width() const { return root_half_width_; }
std::span<const TreeNode> UniformTree::nodes() const { return nodes_; }
std::span<const Vec3> UniformTree::sorted_source_positions() const { return source_positions_sorted_; }
std::span<const Vec3> UniformTree::sorted_target_positions() const { return target_positions_sorted_; }
std::span<const int> UniformTree::source_permutation() const { return source_permutation_; }
std::span<const int> UniformTree::source_inverse_permutation() const { return source_inverse_permutation_; }
std::span<const int> UniformTree::target_permutation() const { return target_permutation_; }
std::span<const int> UniformTree::target_inverse_permutation() const { return target_inverse_permutation_; }

std::vector<int> UniformTree::leaf_indices() const
{
    std::vector<int> indices;
    const int begin = level_offset(max_level_);
    const int end = begin + boxes_per_dim(max_level_) * boxes_per_dim(max_level_) * boxes_per_dim(max_level_);
    for (int idx = begin; idx < end; ++idx) {
        indices.push_back(idx);
    }
    return indices;
}

int UniformTree::leaf_index_for_source(const std::size_t sorted_source_index) const { return source_leaf_indices_.at(sorted_source_index); }
int UniformTree::leaf_index_for_target(const std::size_t sorted_target_index) const { return target_leaf_indices_.at(sorted_target_index); }

} // namespace cdfmm
