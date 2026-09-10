// SPDX-License-Identifier: Apache-2.0
#include "cdfmm/tree/adaptive_tree.hpp"

#include "../common/root_box.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace cdfmm {
namespace {
using Clock = std::chrono::steady_clock;
bool is_leaf(const StaticFmmTopology::Node& node) {
    return std::all_of(node.children.begin(), node.children.end(),
                       [](int child) { return child < 0; });
}
int octant(const Vec3& point, const Vec3& centre) {
    return (point.x >= centre.x) | ((point.y >= centre.y) << 1) |
           ((point.z >= centre.z) << 2);
}
} // namespace

AdaptiveTree::AdaptiveTree(const std::vector<Vec3>& sources,
                           const AdaptiveTreeOptions& options)
    : AdaptiveTree(sources, sources, options) {}

AdaptiveTree::AdaptiveTree(const std::vector<Vec3>& sources,
                           const std::vector<Vec3>& targets,
                           const AdaptiveTreeOptions& options)
    : options_(options) {
    const auto start = Clock::now();
    if (options.max_depth < 0 || options.max_depth > 8 ||
        options.max_particles_per_leaf == 0) {
        throw std::invalid_argument("adaptive depth must be in [0,8] and capacity positive");
    }
    const auto finite = [](const Vec3& p) {
        return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
    };
    for (const auto* population : {&sources, &targets}) {
        for (const auto& point : *population) {
            if (!finite(point)) throw std::invalid_argument("adaptive positions must be finite");
        }
    }
    if ((options.root_centre && !finite(*options.root_centre)) ||
        (options.root_half_width && !std::isfinite(*options.root_half_width))) {
        throw std::invalid_argument("adaptive root geometry must be finite");
    }
    const RootBox root_box = resolve_root_box(
        sources, targets, options.root_centre, options.root_half_width);
    const double root_half_width = root_box.half_width > 0.0
        ? root_box.half_width : 1.0;
    auto topology = std::make_shared<StaticFmmTopology>();
    auto& t = *topology;
    t.source_permutation.resize(sources.size());
    t.target_permutation.resize(targets.size());
    std::iota(t.source_permutation.begin(), t.source_permutation.end(), 0);
    std::iota(t.target_permutation.begin(), t.target_permutation.end(), 0);
    const auto partition = [](auto& permutation, const auto& positions,
                              std::size_t begin, std::size_t end, Vec3 centre) {
        std::stable_sort(permutation.begin() + begin, permutation.begin() + end,
                         [&](int a, int b) {
            return octant(positions[a], centre) < octant(positions[b], centre);
        });
        std::array<std::size_t, 9> offsets{};
        std::size_t cursor = begin;
        for (int child = 0; child < 8; ++child) {
            offsets[child] = cursor;
            while (cursor < end && octant(positions[permutation[cursor]], centre) == child) {
                ++cursor;
            }
        }
        offsets[8] = end;
        return offsets;
    };
    std::function<int(int, int, Vec3, double, std::size_t, std::size_t,
                      std::size_t, std::size_t)> build;
    build = [&](int parent, int level, Vec3 centre, double half_width,
                std::size_t sb, std::size_t se, std::size_t tb, std::size_t te) {
        const int id = static_cast<int>(t.nodes.size());
        StaticFmmTopology::Node node;
        node.index = id;
        node.parent = parent;
        node.level = level;
        node.centre = centre;
        node.half_width = half_width;
        node.source_begin = sb;
        node.source_end = se;
        node.target_begin = tb;
        node.target_end = te;
        t.nodes.push_back(node);
        t.maximum_level = std::max(t.maximum_level, level);
        if (level == options.max_depth ||
            std::max(se - sb, te - tb) <= options.max_particles_per_leaf) {
            if (se > sb) t.source_leaves.push_back({id, sb, se - sb});
            if (te > tb) t.target_leaves.push_back({id, tb, te - tb});
            return id;
        }
        const auto so = partition(t.source_permutation, sources, sb, se, centre);
        const auto to = partition(t.target_permutation, targets, tb, te, centre);
        const double h = half_width * 0.5;
        for (int child = 0; child < 8; ++child) {
            if (so[child] == so[child + 1] && to[child] == to[child + 1]) continue;
            const Vec3 child_centre = centre + Vec3{
                (child & 1) ? h : -h, (child & 2) ? h : -h, (child & 4) ? h : -h};
            const int child_id = build(id, level + 1, child_centre, h,
                                      so[child], so[child + 1], to[child], to[child + 1]);
            t.nodes[id].children[child] = child_id;
        }
        return id;
    };
    build(-1, 0, root_box.centre, root_half_width,
          0, sources.size(), 0, targets.size());
    const auto reorder = [](const auto& positions, const auto& permutation,
                            auto& sorted, auto& inverse) {
        inverse.resize(permutation.size());
        for (std::size_t i = 0; i < permutation.size(); ++i) {
            sorted.push_back(positions[permutation[i]]);
            inverse[permutation[i]] = static_cast<int>(i);
        }
    };
    reorder(sources, t.source_permutation, t.sorted_source_positions, t.source_inverse_permutation);
    reorder(targets, t.target_permutation, t.sorted_target_positions, t.target_inverse_permutation);
    t.m2m_level_offsets.assign(t.maximum_level + 2, 0);
    t.l2l_level_offsets.assign(t.maximum_level + 2, 0);
    t.m2m_parent_level_offsets.assign(t.maximum_level + 2, 0);
    t.m2m_parent_edge_offsets.push_back(0);
    for (int level = 1; level <= t.maximum_level; ++level) {
        t.m2m_level_offsets[level] = static_cast<int>(t.m2m_edges.size());
        t.l2l_level_offsets[level] = static_cast<int>(t.l2l_edges.size());
        t.m2m_parent_level_offsets[level] = static_cast<int>(t.m2m_parent_nodes.size());
        for (const auto& parent : t.nodes) {
            if (parent.level != level - 1 || is_leaf(parent)) continue;
            t.m2m_parent_nodes.push_back(parent.index);
            for (int child_class = 0; child_class < 8; ++child_class) {
                const int child = parent.children[child_class];
                if (child < 0) continue;
                t.m2m_edges.push_back({child, parent.index, child_class, level});
                t.l2l_edges.push_back({parent.index, child, child_class, level});
            }
            t.m2m_parent_edge_offsets.push_back(static_cast<int>(t.m2m_edges.size()));
        }
    }
    t.m2m_level_offsets.back() = static_cast<int>(t.m2m_edges.size());
    t.l2l_level_offsets.back() = static_cast<int>(t.l2l_edges.size());
    t.m2m_parent_level_offsets.back() = static_cast<int>(t.m2m_parent_nodes.size());
    const auto interaction_start = Clock::now();
    tree_seconds_ = std::chrono::duration<double>(interaction_start - start).count();
    std::function<void(int, int)> visit;
    visit = [&](int source_id, int target_id) {
        const auto& source = t.nodes[source_id];
        const auto& target = t.nodes[target_id];
        if (source.source_count() == 0 || target.target_count() == 0) return;
        const Vec3 d = target.centre - source.centre;
        const double separation = std::max({std::abs(d.x), std::abs(d.y), std::abs(d.z)});
        const double distance = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
        const double radius_sum = std::sqrt(3.0) * (source.half_width + target.half_width);
        // Equal levels retain the classical stencil. Unequal boxes use a
        // convergent enclosing-sphere test plus explicit non-touching boxes.
        const bool separated = source.level == target.level
            ? separation >= 4.0 * source.half_width
            : separation > source.half_width + target.half_width &&
              radius_sum <= 0.75 * distance;
        if (separated) {
            StaticM2LInteraction interaction;
            interaction.source_node = source_id;
            interaction.target_node = target_id;
            interaction.source_level = source.level;
            interaction.target_level = target.level;
            interaction.displacement = d;
            interaction.source_to_target_width = source.half_width / target.half_width;
            const double width = 2.0 * target.half_width;
            interaction.transfer_class = {static_cast<int>(std::lround(d.x / width)),
                static_cast<int>(std::lround(d.y / width)), static_cast<int>(std::lround(d.z / width))};
            t.m2l_interactions.push_back(interaction);
            return;
        }
        const bool sl = is_leaf(source), tl = is_leaf(target);
        if (sl && tl) {
            t.p2p_leaf_records.push_back({
                target_id, source_id, target.target_begin, target.target_count(),
                source.source_begin, source.source_count(), {}, {}, true});
        } else if (!sl && !tl && source.level == target.level) {
            for (int target_child : target.children) {
                if (target_child < 0) continue;
                for (int source_child : source.children) {
                    if (source_child >= 0) visit(source_child, target_child);
                }
            }
        } else if (!sl && (tl || source.half_width > target.half_width)) {
            for (int child : source.children) {
                if (child >= 0) visit(child, target_id);
            }
        } else {
            for (int child : target.children) {
                if (child >= 0) visit(source_id, child);
            }
        }
    };
    visit(t.root, t.root);
    std::stable_sort(t.m2l_interactions.begin(), t.m2l_interactions.end(),
                     [](const auto& a, const auto& b) { return a.target_node < b.target_node; });
    t.m2l_target_row_offsets.assign(t.nodes.size() + 1, 0);
    for (const auto& interaction : t.m2l_interactions) ++t.m2l_target_row_offsets[interaction.target_node + 1];
    std::partial_sum(t.m2l_target_row_offsets.begin(), t.m2l_target_row_offsets.end(), t.m2l_target_row_offsets.begin());
    std::stable_sort(t.p2p_leaf_records.begin(), t.p2p_leaf_records.end(),
                     [](const auto& a, const auto& b) { return a.target_leaf < b.target_leaf; });
    t.p2p_target_leaf_offsets.push_back(0);
    std::size_t cursor = 0;
    for (const auto& leaf : t.target_leaves) {
        t.p2p_target_leaves.push_back(leaf.node);
        while (cursor < t.p2p_leaf_records.size() && t.p2p_leaf_records[cursor].target_leaf == leaf.node) ++cursor;
        t.p2p_target_leaf_offsets.push_back(static_cast<int>(cursor));
    }
    t.coordinate_origin = root_box.centre;
    t.coordinate_scale = 2.0 * root_half_width;
    for (auto& node : t.nodes) {
        node.centre = (node.centre - t.coordinate_origin) * (1.0 / t.coordinate_scale);
        node.half_width /= t.coordinate_scale;
    }
    for (auto& point : t.sorted_source_positions) point = (point - t.coordinate_origin) * (1.0 / t.coordinate_scale);
    for (auto& point : t.sorted_target_positions) point = (point - t.coordinate_origin) * (1.0 / t.coordinate_scale);
    for (auto& interaction : t.m2l_interactions) interaction.displacement = interaction.displacement * (1.0 / t.coordinate_scale);
    t.validate();
    interaction_seconds_ = std::chrono::duration<double>(Clock::now() - interaction_start).count();
    topology_ = std::move(topology);
}
} // namespace cdfmm
