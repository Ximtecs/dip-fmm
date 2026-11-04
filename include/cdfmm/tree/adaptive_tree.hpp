// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "cdfmm/math/vec3.hpp"
#include "cdfmm/tree/static_topology.hpp"

namespace cdfmm {

/**
 * @brief How a box holding more than `max_particles_per_leaf` points decides whether to subdivide.
 *
 * `Capacity` subdivides every such box, so `max_particles_per_leaf` is an upper bound and leaves of
 * evenly spread points hold about an eighth to all of it (a quarter on average). `Target` treats
 * `max_particles_per_leaf` as the occupancy to aim for: a box subdivides only if its non-empty
 * octants are, weighted by their point counts, closer to the target than the box itself, measured
 * as |ln(n / target)|. Leaves then scatter around the target, from below to a few times above it.
 */
enum class AdaptiveSplitRule {
    Capacity,  ///< Subdivide every box above the capacity (the original rule, the default).
    Target,    ///< Subdivide only when the octants are closer to the target occupancy than the box.
};

/** @brief Geometry-only adaptive octree subdivision controls. */
struct AdaptiveTreeOptions {
    std::size_t max_particles_per_leaf{32};   ///< Capacity, or the target occupancy under `AdaptiveSplitRule::Target`.
    int max_depth{5};                         ///< Deepest level allowed (at most 8); coincident points stop here.
    std::optional<Vec3> root_centre{};        ///< Optional fixed root centre; otherwise inferred from all points.
    std::optional<double> root_half_width{};  ///< Optional positive root half-width; otherwise inferred.
    AdaptiveSplitRule split_rule{AdaptiveSplitRule::Capacity};  ///< Subdivision rule above `max_particles_per_leaf`.
};

/**
 * @brief Compact octree and directed near/far partition, without operators.
 *
 * Coordinates are physical. A static evaluator shares this immutable topology.
 * Empty octants are omitted; coincident points terminate at the depth cap.
 *
 * The public result is `StaticFmmTopology`, the same tree-owned
 * interaction-topology type `build_uniform_fmm_topology` produces from a
 * `UniformTree`. It holds only spatial-tree and interaction-topology facts
 * (nodes, permutations, M2M/L2L edges, M2L interactions, P2P leaf records);
 * it owns no operator coefficients, execution packing, or backend state, so
 * returning it here is not a tree-to-plan coupling. `fmm` and `plan` consume
 * it by reference through `shared_topology()` and never rebuild it.
 */
class AdaptiveTree {
public:
    /// @brief Builds the tree and its interaction topology for distinct source and target sets.
    AdaptiveTree(const std::vector<Vec3>& sources,
                 const std::vector<Vec3>& targets,
                 const AdaptiveTreeOptions& options = {});
    /// @brief Builds the tree with the sources as targets.
    explicit AdaptiveTree(const std::vector<Vec3>& sources,
                          const AdaptiveTreeOptions& options = {});
    /**
     * @brief Builds the tree for finite bodies of the given support radii.
     *
     * `source_radii[i]` (`target_radii[i]`) bounds the distance of every point of body i from its position,
     * e.g. the half diagonal of a prism or the farthest vertex of a tetrahedron; either vector may be empty
     * (points). Bodies are still assigned to leaves by their positions, but a pair of boxes counts as separated
     * only if, besides the box test, the support radii about the box centres satisfy the finite-support
     * acceptance criterion a_s + a_t <= eta |c_t - c_s|, with eta = sqrt(3)/2 for boxes of equal size and 0.75
     * across levels, the bounds the box tests already guarantee for points inside their boxes. A pair that
     * fails is refined like any non-separated pair, and two leaves then interact in the near field. Bodies that
     * lie inside their leaves therefore never change the topology.
     */
    AdaptiveTree(const std::vector<Vec3>& sources,
                 const std::vector<Vec3>& targets,
                 const std::vector<double>& source_radii,
                 const std::vector<double>& target_radii,
                 const AdaptiveTreeOptions& options = {});
    /**
     * @brief Builds the tree for finite convex bodies given by support points.
     *
     * Body i of the sources is the convex hull of `source_points[i * points_per_source + k]`,
     * k = 0..points_per_source-1, given as offsets from its position (e.g. the four vertices of a tetrahedron);
     * likewise for the targets. Either vector may be empty (points), otherwise its size must be the body count
     * times the points per body. The support radius of a box is then the exact a = max |x_i + p - c| over the
     * points of its bodies, the smallest sphere about the box centre that encloses them, instead of the bound
     * max (|x_i - c| + r_i) of the radius constructor, which can exceed it by about half (a tetrahedron of a cube
     * lattice inside its leaf: 1.29 against 0.87 times the cube edge). The acceptance criterion, refinement and
     * leaf assignment are those of the radius constructor, so bodies inside their leaves never change the
     * topology, and protruding bodies are refined only where their actual extent requires it.
     */
    AdaptiveTree(const std::vector<Vec3>& sources,
                 const std::vector<Vec3>& targets,
                 const std::vector<Vec3>& source_points,
                 std::size_t points_per_source,
                 const std::vector<Vec3>& target_points,
                 std::size_t points_per_target,
                 const AdaptiveTreeOptions& options = {});
    /// @brief The immutable topology, in normalised coordinates.
    [[nodiscard]] const StaticFmmTopology& topology() const { return *topology_; }
    /// @brief Shared ownership of the topology, for `UniformFmm`.
    [[nodiscard]] std::shared_ptr<const StaticFmmTopology> shared_topology() const {
        return topology_;
    }
    /// @brief Seconds spent building the spatial tree.
    [[nodiscard]] double tree_seconds() const { return tree_seconds_; }
    /// @brief Seconds spent deriving the near/far interaction lists.
    [[nodiscard]] double interaction_seconds() const { return interaction_seconds_; }
    /// @brief The options the tree was built with.
    [[nodiscard]] const AdaptiveTreeOptions& options() const { return options_; }
    /// @brief Box pairs that passed the box test but failed the finite-support criterion and were refined.
    [[nodiscard]] std::size_t support_refinements() const { return support_refinements_; }
private:
    // Extent of the finite bodies of one population: support radii, support points, or neither (points).
    struct BodySupport {
        const std::vector<double>* radii{nullptr};
        const std::vector<Vec3>* points{nullptr};
        std::size_t points_per_body{0};
        [[nodiscard]] bool finite() const {
            return (radii != nullptr && !radii->empty()) || (points != nullptr && !points->empty());
        }
    };
    void build(const std::vector<Vec3>& sources,
               const std::vector<Vec3>& targets,
               const BodySupport& source_support,
               const BodySupport& target_support);
    std::shared_ptr<const StaticFmmTopology> topology_;
    AdaptiveTreeOptions options_;
    double tree_seconds_{0};
    double interaction_seconds_{0};
    std::size_t support_refinements_{0};
};
} // namespace cdfmm
