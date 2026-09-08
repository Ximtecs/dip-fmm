// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <vector>
#include <memory>

#include "cdfmm/static_topology.hpp"

namespace cdfmm {

/** @brief Geometry-only adaptive octree subdivision controls. */
struct AdaptiveTreeOptions {
    std::size_t max_particles_per_leaf{10};
    int max_depth{3};
    std::optional<Vec3> root_centre{};
    std::optional<double> root_half_width{};
};

/**
 * @brief Compact octree and directed near/far partition, without operators.
 *
 * Coordinates are physical. A static evaluator shares this immutable topology.
 * Empty octants are omitted; coincident points terminate at the depth cap.
 */
class AdaptiveTree {
public:
    AdaptiveTree(const std::vector<Vec3>& sources,
                 const std::vector<Vec3>& targets,
                 const AdaptiveTreeOptions& options = {});
    explicit AdaptiveTree(const std::vector<Vec3>& sources,
                          const AdaptiveTreeOptions& options = {});
    [[nodiscard]] const StaticFmmTopology& topology() const { return *topology_; }
    [[nodiscard]] std::shared_ptr<const StaticFmmTopology> shared_topology() const {
        return topology_;
    }
    [[nodiscard]] double tree_seconds() const { return tree_seconds_; }
    [[nodiscard]] double interaction_seconds() const { return interaction_seconds_; }
    [[nodiscard]] const AdaptiveTreeOptions& options() const { return options_; }
private:
    std::shared_ptr<const StaticFmmTopology> topology_;
    AdaptiveTreeOptions options_;
    double tree_seconds_{0};
    double interaction_seconds_{0};
};
} // namespace cdfmm
