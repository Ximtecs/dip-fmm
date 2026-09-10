// SPDX-License-Identifier: Apache-2.0

#include "root_box.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace cdfmm {
namespace {

// Permit round-off at a user-specified root boundary.  Coordinates accepted
// through this tolerance are clamped into the first or final leaf below.
constexpr double kBoundaryTolerance = 1.0e-12;

} // namespace

RootBox resolve_root_box(
    const std::span<const Vec3> source_positions,
    const std::span<const Vec3> target_positions,
    const std::optional<Vec3> root_centre,
    const std::optional<double> root_half_width
) {
    if (root_half_width.has_value() && root_half_width.value() <= 0.0) {
        throw std::invalid_argument("UniformTreeOptions.root_half_width must be positive");
    }

    const std::span<const Vec3> populations[] = {source_positions, target_positions};
    Vec3 minimum{std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::infinity()};
    Vec3 maximum{-std::numeric_limits<double>::infinity(),
                 -std::numeric_limits<double>::infinity(),
                 -std::numeric_limits<double>::infinity()};
    bool has_points = false;
    for (const std::span<const Vec3> population : populations) {
        for (const Vec3& point : population) {
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

    RootBox result;
    result.centre = root_centre.value_or((minimum + maximum) * 0.5);
    if (root_half_width.has_value()) {
        result.half_width = root_half_width.value();
    } else {
        const Vec3 delta_max = maximum - result.centre;
        const Vec3 delta_min = result.centre - minimum;
        // A single half-width makes the root cubic even for an anisotropic
        // point cloud.  A requested centre means both sides must be inspected.
        result.half_width = std::max({
            delta_max.x,
            delta_max.y,
            delta_max.z,
            delta_min.x,
            delta_min.y,
            delta_min.z
        });
        if (!has_points) {
            result.half_width = 1.0;
        }
    }

    const Vec3 root_min = result.centre -
        Vec3{result.half_width, result.half_width, result.half_width};
    const Vec3 root_max = result.centre +
        Vec3{result.half_width, result.half_width, result.half_width};
    const auto in_range = [](const double value, const double lo, const double hi) {
        return value >= lo - kBoundaryTolerance && value <= hi + kBoundaryTolerance;
    };
    for (const std::span<const Vec3> population : populations) {
        for (const Vec3& point : population) {
            if (!in_range(point.x, root_min.x, root_max.x) ||
                !in_range(point.y, root_min.y, root_max.y) ||
                !in_range(point.z, root_min.z, root_max.z)) {
                throw std::invalid_argument("Point lies outside the requested root box");
            }
        }
    }
    return result;
}

} // namespace cdfmm
