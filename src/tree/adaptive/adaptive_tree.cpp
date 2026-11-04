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
    : AdaptiveTree(sources, targets, {}, {}, options) {}

AdaptiveTree::AdaptiveTree(const std::vector<Vec3>& sources,
                           const std::vector<Vec3>& targets,
                           const std::vector<double>& source_radii,
                           const std::vector<double>& target_radii,
                           const AdaptiveTreeOptions& options)
    : options_(options) {
    if ((!source_radii.empty() && source_radii.size() != sources.size()) ||
        (!target_radii.empty() && target_radii.size() != targets.size())) {
        throw std::invalid_argument("adaptive support radii must match the number of bodies");
    }
    for (const auto* radii : {&source_radii, &target_radii}) {
        for (const double r : *radii) {
            if (!(r >= 0.0) || !std::isfinite(r)) throw std::invalid_argument("adaptive support radii must be finite and non-negative");
        }
    }
    build(sources, targets, BodySupport{&source_radii, nullptr, 0}, BodySupport{&target_radii, nullptr, 0});
}

AdaptiveTree::AdaptiveTree(const std::vector<Vec3>& sources,
                           const std::vector<Vec3>& targets,
                           const std::vector<Vec3>& source_points,
                           std::size_t points_per_source,
                           const std::vector<Vec3>& target_points,
                           std::size_t points_per_target,
                           const AdaptiveTreeOptions& options)
    : options_(options) {
    const auto check = [](const std::vector<Vec3>& points, std::size_t per_body, std::size_t bodies) {
        if (points.empty()) return;
        if (per_body == 0 || points.size() != bodies * per_body) {
            throw std::invalid_argument("adaptive support points must be the number of bodies times the points per body");
        }
        for (const Vec3& p : points) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                throw std::invalid_argument("adaptive support points must be finite");
            }
        }
    };
    check(source_points, points_per_source, sources.size());
    check(target_points, points_per_target, targets.size());
    build(sources, targets, BodySupport{nullptr, &source_points, points_per_source},
          BodySupport{nullptr, &target_points, points_per_target});
}

void AdaptiveTree::build(const std::vector<Vec3>& sources,
                         const std::vector<Vec3>& targets,
                         const BodySupport& source_body_support,
                         const BodySupport& target_body_support) {
    const AdaptiveTreeOptions& options = options_;
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
        if (options.split_rule == AdaptiveSplitRule::Target) {
            // Subdivide only if the octants, weighted by their counts, are closer to the target
            // occupancy than this box: sum_k n_k |ln(n_k / T)| / sum_k n_k < |ln(n / T)|.
            const double target = static_cast<double>(options.max_particles_per_leaf);
            double weighted = 0.0, total = 0.0;
            for (int child = 0; child < 8; ++child) {
                const double n = static_cast<double>(std::max(so[child + 1] - so[child], to[child + 1] - to[child]));
                if (n == 0.0) continue;
                weighted += n * std::abs(std::log(n / target));
                total += n;
            }
            const double here = static_cast<double>(std::max(se - sb, te - tb));
            if (weighted >= total * std::abs(std::log(here / target))) {
                if (se > sb) t.source_leaves.push_back({id, sb, se - sb});
                if (te > tb) t.target_leaves.push_back({id, tb, te - tb});
                return id;
            }
        }
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
    // Finite bodies: support radius of every box about its centre (physical coordinates), the bound a of the
    // finite-support acceptance criterion a_s + a_t <= eta |c_t - c_s|. With support radii, a body extends at
    // most |x - c| + r from the centre; with support points (the body is their convex hull) its extent is exactly
    // the largest |x + p - c|.
    const bool finite_bodies = source_body_support.finite() || target_body_support.finite();
    std::vector<double> source_support, target_support;
    if (finite_bodies) {
        const auto norm = [](const Vec3& d) { return std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z); };
        const auto support = [&](const auto& sorted, const auto& permutation, const BodySupport& body,
                                 std::vector<double>& out, bool targets_side) {
            const bool radii = body.radii != nullptr && !body.radii->empty();
            const bool points = body.points != nullptr && !body.points->empty();
            out.assign(t.nodes.size(), 0.0);
            for (const auto& node : t.nodes) {
                const std::size_t begin = targets_side ? node.target_begin : node.source_begin;
                const std::size_t end = targets_side ? node.target_end : node.source_end;
                double a = 0.0;
                for (std::size_t i = begin; i < end; ++i) {
                    const Vec3 d = sorted[i] - node.centre;
                    const std::size_t b = static_cast<std::size_t>(permutation[i]);
                    if (points) {
                        for (std::size_t k = 0; k < body.points_per_body; ++k) {
                            a = std::max(a, norm(d + (*body.points)[b * body.points_per_body + k]));
                        }
                    } else {
                        a = std::max(a, norm(d) + (radii ? (*body.radii)[b] : 0.0));
                    }
                }
                out[node.index] = a;
            }
        };
        support(t.sorted_source_positions, t.source_permutation, source_body_support, source_support, false);
        support(t.sorted_target_positions, t.target_permutation, target_body_support, target_support, true);
    }
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
        // NOTE(cdfmm): box centres are accumulated root centre +- half widths in physical coordinates; for a root
        // that is not a binary fraction (e.g. 2^k cells of a lattice whose spacing is 240/37) they carry rounding,
        // and boxes exactly two widths apart would fail an exact >= and fall into the near field. The relative
        // tolerance (as in the finite-support test) keeps exact roots unchanged and treats the others alike.
        constexpr double tolerance = 1.0e-12;
        bool separated = source.level == target.level
            ? separation >= 4.0 * source.half_width * (1.0 - tolerance)
            : separation > (source.half_width + target.half_width) * (1.0 + tolerance) &&
              radius_sum <= 0.75 * distance * (1.0 + tolerance);
        // Finite bodies may protrude from their boxes: the expansions converge only if their supports satisfy
        // the acceptance criterion too, with the bound the box test gives for points (equal boxes: separation
        // >= 4h, so 2 sqrt(3) h <= sqrt(3)/2 |d|; across levels 0.75). Otherwise refine, as for neighbours.
        if (separated && finite_bodies) {
            const double eta = source.level == target.level ? 0.5 * std::sqrt(3.0) : 0.75;
            if (source_support[source_id] + target_support[target_id] > eta * distance * (1.0 + 1.0e-12)) {
                separated = false;
                ++support_refinements_;
            }
        }
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
