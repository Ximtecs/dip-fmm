// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/periodic.hpp"

#include <algorithm>
#include <stdexcept>

#include "cdfmm/uniform_tree.hpp"

namespace cdfmm {

namespace {

int floor_divide(const int numerator, const int denominator)
{
    int quotient = numerator / denominator;
    const int remainder = numerator % denominator;
    if (remainder < 0) {
        --quotient;
    }
    return quotient;
}

PeriodicBoxIdentity make_identity(
    const int level,
    const std::array<int, 3>& unwrapped
)
{
    const int boxes_per_axis = 1 << level;
    std::array<int, 3> wrapped{};
    std::array<int, 3> shift{};
    for (int axis = 0; axis < 3; ++axis) {
        const auto value = wrap_periodic_box_coordinate(
            unwrapped[static_cast<std::size_t>(axis)],
            boxes_per_axis
        );
        wrapped[static_cast<std::size_t>(axis)] = value.coordinate;
        shift[static_cast<std::size_t>(axis)] = value.image_shift;
    }
    return {
        node_index(level, wrapped[0], wrapped[1], wrapped[2]),
        shift
    };
}

} // namespace

void validate_periodic_cell(const PeriodicCellOptions& options)
{
    if (!options.enabled) {
        return;
    }
    if (!options.axes[0] || !options.axes[1] || !options.axes[2]) {
        throw std::invalid_argument(
            "periodic UniformFmm currently supports all three axes only"
        );
    }
    if (options.lengths.x <= 0.0 || options.lengths.y <= 0.0 ||
        options.lengths.z <= 0.0) {
        throw std::invalid_argument("periodic cell lengths must be positive");
    }
    if (options.lengths.x != options.lengths.y ||
        options.lengths.x != options.lengths.z) {
        throw std::invalid_argument(
            "periodic UniformFmm currently requires a cubic cell"
        );
    }
    if (options.convention != PeriodicConvention::ZeroK0) {
        throw std::invalid_argument("unsupported periodic convention");
    }
    if (options.setup_tolerance <= 0.0 || options.setup_tolerance >= 1.0) {
        throw std::invalid_argument(
            "periodic setup tolerance must lie strictly between zero and one"
        );
    }
}

WrappedBoxCoordinate wrap_periodic_box_coordinate(
    const int unwrapped,
    const int boxes_per_axis
)
{
    if (boxes_per_axis <= 0) {
        throw std::invalid_argument("boxes_per_axis must be positive");
    }
    const int image_shift = floor_divide(unwrapped, boxes_per_axis);
    return {
        unwrapped - image_shift * boxes_per_axis,
        image_shift
    };
}

std::vector<PeriodicBoxIdentity> build_periodic_list1(
    const int level,
    const std::array<int, 3>& target_coordinate
)
{
    if (level < 0) {
        throw std::invalid_argument("periodic list level must be non-negative");
    }
    std::vector<PeriodicBoxIdentity> result;
    result.reserve(27);
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                result.push_back(make_identity(
                    level,
                    {target_coordinate[0] + dx,
                     target_coordinate[1] + dy,
                     target_coordinate[2] + dz}
                ));
            }
        }
    }
    return result;
}

std::vector<PeriodicBoxIdentity> build_periodic_list2(
    const int level,
    const std::array<int, 3>& target_coordinate
)
{
    if (level < 1) {
        return {};
    }
    const std::array<int, 3> parent{
        target_coordinate[0] / 2,
        target_coordinate[1] / 2,
        target_coordinate[2] / 2
    };
    const auto list1 = build_periodic_list1(level, target_coordinate);
    std::vector<PeriodicBoxIdentity> result;
    result.reserve(189);
    for (int pz = -1; pz <= 1; ++pz) {
        for (int py = -1; py <= 1; ++py) {
            for (int px = -1; px <= 1; ++px) {
                const std::array<int, 3> source_parent{
                    parent[0] + px,
                    parent[1] + py,
                    parent[2] + pz
                };
                for (int child = 0; child < 8; ++child) {
                    const std::array<int, 3> source_child{
                        2 * source_parent[0] + ((child & 1) != 0),
                        2 * source_parent[1] + ((child & 2) != 0),
                        2 * source_parent[2] + ((child & 4) != 0)
                    };
                    const auto identity = make_identity(level, source_child);
                    if (std::find(list1.begin(), list1.end(), identity) ==
                        list1.end()) {
                        result.push_back(identity);
                    }
                }
            }
        }
    }
    return result;
}

} // namespace cdfmm
