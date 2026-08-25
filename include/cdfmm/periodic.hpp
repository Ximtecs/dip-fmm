// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <vector>

#include "cdfmm/vec3.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Public types
//------------------------------------------------------------------------------

/** @brief Macroscopic convention used to define a 3D dipolar lattice sum. */
enum class PeriodicConvention {
    /// Reciprocal k=0 is omitted and no macroscopic surface term is added.
    ZeroK0
};

/** @brief Explicit periodic-cell configuration for a reusable FMM plan. */
struct PeriodicCellOptions {
    /// @brief Enables periodic topology. Disabled plans retain free-space behaviour.
    bool enabled{false};
    /// @brief Periodic axes, reserved for future partially periodic plans.
    std::array<bool, 3> axes{true, true, true};
    /// @brief Geometric centre of the fundamental cell.
    Vec3 centre{};
    /// @brief Fundamental-cell side lengths.
    Vec3 lengths{1.0, 1.0, 1.0};
    /// @brief Conditional lattice-sum convention.
    PeriodicConvention convention{PeriodicConvention::ZeroK0};
    /// @brief Absolute setup tolerance used to truncate Ewald construction.
    double setup_tolerance{1.0e-12};
};

/** @brief Identity of a periodically imaged box without tree replication. */
struct PeriodicBoxIdentity {
    int node{0};
    std::array<int, 3> image_shift{};

    [[nodiscard]] bool operator==(const PeriodicBoxIdentity&) const = default;
};

/** @brief Wrapped coordinate and cell shift associated with one integer axis. */
struct WrappedBoxCoordinate {
    int coordinate{0};
    int image_shift{0};
};

//------------------------------------------------------------------------------
// Public functions
//------------------------------------------------------------------------------

/** @brief Validates the currently supported fully periodic cubic mode. */
void validate_periodic_cell(const PeriodicCellOptions& options);

/** @brief Wraps an unbounded box coordinate using mathematical floor division. */
[[nodiscard]] WrappedBoxCoordinate wrap_periodic_box_coordinate(
    int unwrapped,
    int boxes_per_axis
);

/** @brief Builds the periodic 3x3x3 same-level neighbourhood of a box. */
[[nodiscard]] std::vector<PeriodicBoxIdentity> build_periodic_list1(
    int level,
    const std::array<int, 3>& target_coordinate
);

/**
 * @brief Builds children(parent list1) minus node list1 with image identity.
 *
 * Image shifts participate in equality, which preserves distinct images in
 * shallow trees where several unwrapped boxes map to the same central node.
 */
[[nodiscard]] std::vector<PeriodicBoxIdentity> build_periodic_list2(
    int level,
    const std::array<int, 3>& target_coordinate
);

} // namespace cdfmm
