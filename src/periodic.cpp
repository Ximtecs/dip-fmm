// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/periodic.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include "cdfmm/uniform_tree.hpp"

namespace cdfmm {

namespace {

constexpr double k_inverse_four_pi =
    1.0 / (4.0 * std::numbers::pi);

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

double integer_power(double value, int exponent)
{
    double result = 1.0;
    for (int power = 0; power < exponent; ++power) {
        result *= value;
    }
    return result;
}

std::vector<double> screened_radial_coefficients(
    const double radius_squared,
    const double ewald_alpha,
    const int order
)
{
    std::vector<double> radius(static_cast<std::size_t>(order + 1), 0.0);
    std::vector<double> inverse_radius(
        static_cast<std::size_t>(order + 1), 0.0);
    std::vector<double> gaussian(static_cast<std::size_t>(order + 1), 0.0);
    std::vector<double> erfc_radius(static_cast<std::size_t>(order + 1), 0.0);
    std::vector<double> result(static_cast<std::size_t>(order + 1), 0.0);

    radius[0] = std::sqrt(radius_squared);
    inverse_radius[0] = 1.0 / radius[0];
    gaussian[0] = std::exp(
        -ewald_alpha * ewald_alpha * radius_squared);
    erfc_radius[0] = std::erfc(ewald_alpha * radius[0]);

    for (int degree = 1; degree <= order; ++degree) {
        double radius_convolution = 0.0;
        for (int index = 1; index < degree; ++index) {
            radius_convolution +=
                radius[static_cast<std::size_t>(index)] *
                radius[static_cast<std::size_t>(degree - index)];
        }
        const double input = degree == 1 ? 1.0 : 0.0;
        radius[static_cast<std::size_t>(degree)] =
            (input - radius_convolution) / (2.0 * radius[0]);

        double inverse_convolution = 0.0;
        for (int index = 1; index <= degree; ++index) {
            inverse_convolution +=
                radius[static_cast<std::size_t>(index)] *
                inverse_radius[static_cast<std::size_t>(degree - index)];
        }
        inverse_radius[static_cast<std::size_t>(degree)] =
            -inverse_convolution / radius[0];

        gaussian[static_cast<std::size_t>(degree)] =
            gaussian[static_cast<std::size_t>(degree - 1)] *
            (-ewald_alpha * ewald_alpha) / static_cast<double>(degree);
    }

    const double derivative_factor =
        -ewald_alpha / std::sqrt(std::numbers::pi);
    for (int degree = 0; degree < order; ++degree) {
        double derivative_coefficient = 0.0;
        for (int index = 0; index <= degree; ++index) {
            derivative_coefficient +=
                gaussian[static_cast<std::size_t>(index)] *
                inverse_radius[static_cast<std::size_t>(degree - index)];
        }
        erfc_radius[static_cast<std::size_t>(degree + 1)] =
            derivative_factor * derivative_coefficient /
            static_cast<double>(degree + 1);
    }

    for (int degree = 0; degree <= order; ++degree) {
        for (int index = 0; index <= degree; ++index) {
            result[static_cast<std::size_t>(degree)] +=
                erfc_radius[static_cast<std::size_t>(index)] *
                inverse_radius[static_cast<std::size_t>(degree - index)] *
                k_inverse_four_pi;
        }
    }
    return result;
}

std::vector<double> self_radial_coefficients(
    const double ewald_alpha,
    const int order
)
{
    std::vector<double> result(static_cast<std::size_t>(order + 1), 0.0);
    result[0] = -ewald_alpha /
        (2.0 * std::numbers::pi * std::sqrt(std::numbers::pi));
    for (int degree = 0; degree < order; ++degree) {
        result[static_cast<std::size_t>(degree + 1)] =
            result[static_cast<std::size_t>(degree)] *
            (-ewald_alpha * ewald_alpha) *
            static_cast<double>(2 * degree + 1) /
            (static_cast<double>(degree + 1) *
             static_cast<double>(2 * degree + 3));
    }
    return result;
}

std::vector<double> laplace_radial_coefficients(
    const double radius_squared,
    const int order
)
{
    std::vector<double> result(static_cast<std::size_t>(order + 1), 0.0);
    result[0] = k_inverse_four_pi / std::sqrt(radius_squared);
    for (int degree = 0; degree < order; ++degree) {
        result[static_cast<std::size_t>(degree + 1)] =
            -result[static_cast<std::size_t>(degree)] *
            static_cast<double>(2 * degree + 1) /
            (2.0 * static_cast<double>(degree + 1) * radius_squared);
    }
    return result;
}

void add_radial_taylor(
    const MultiIndexSet& basis,
    const Vec3& centre,
    const std::vector<double>& radial_coefficients,
    const double scale,
    std::vector<double>& coefficients
)
{
    struct Term {
        MultiIndex power{};
        double coefficient{0.0};
    };
    const std::array<Term, 6> quadratic_terms{{
        {{1, 0, 0}, 2.0 * centre.x},
        {{0, 1, 0}, 2.0 * centre.y},
        {{0, 0, 1}, 2.0 * centre.z},
        {{2, 0, 0}, 1.0},
        {{0, 2, 0}, 1.0},
        {{0, 0, 2}, 1.0},
    }};

    std::vector<double> power(static_cast<std::size_t>(basis.size()), 0.0);
    std::vector<double> next(static_cast<std::size_t>(basis.size()), 0.0);
    power[static_cast<std::size_t>(basis.index({0, 0, 0}))] = 1.0;
    for (int radial_degree = 0; radial_degree <= basis.order();
         ++radial_degree) {
        const double radial =
            scale * radial_coefficients[static_cast<std::size_t>(radial_degree)];
        for (int index = 0; index < basis.size(); ++index) {
            coefficients[static_cast<std::size_t>(index)] +=
                radial * power[static_cast<std::size_t>(index)];
        }
        if (radial_degree == basis.order()) {
            break;
        }

        std::fill(next.begin(), next.end(), 0.0);
        for (int index = 0; index < basis.size(); ++index) {
            const MultiIndex alpha = basis[index];
            for (const Term& term : quadratic_terms) {
                if (term.coefficient == 0.0 || !leq(term.power, alpha)) {
                    continue;
                }
                next[static_cast<std::size_t>(index)] +=
                    term.coefficient * power[static_cast<std::size_t>(
                        basis.index(sub(alpha, term.power)))];
            }
        }
        power.swap(next);
    }
}

bool is_central_image(const int ix, const int iy, const int iz)
{
    return ix == 0 && iy == 0 && iz == 0;
}

int ewald_image_cutoff(const double tolerance, const int derivative_order)
{
    int cutoff = 2;
    const double log_tolerance = std::log(tolerance);
    while (true) {
        const double omitted_radius = static_cast<double>(cutoff + 1);
        // Derivatives introduce powers of reciprocal wave number. Include a
        // conservative shell-count factor so high-order matrices do not use
        // the scalar-Green-function cut-off unchanged.
        const double log_tail_bound =
            -std::numbers::pi * omitted_radius * omitted_radius +
            static_cast<double>(derivative_order) *
                std::log(2.0 * std::numbers::pi * omitted_radius) +
            3.0 * std::log(2.0 * omitted_radius + 1.0);
        if (log_tail_bound <= log_tolerance - 2.0) {
            return cutoff;
        }
        ++cutoff;
    }
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
    if (!std::isfinite(options.centre.x) ||
        !std::isfinite(options.centre.y) ||
        !std::isfinite(options.centre.z)) {
        throw std::invalid_argument("periodic cell centre must be finite");
    }
    if (!std::isfinite(options.lengths.x) ||
        !std::isfinite(options.lengths.y) ||
        !std::isfinite(options.lengths.z) || options.lengths.x <= 0.0 ||
        options.lengths.y <= 0.0 || options.lengths.z <= 0.0) {
        throw std::invalid_argument(
            "periodic cell lengths must be finite and positive");
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
    if (!std::isfinite(options.setup_tolerance) ||
        options.setup_tolerance <= 0.0 || options.setup_tolerance >= 1.0) {
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

Vec3 wrap_periodic_position(
    const Vec3& position,
    const PeriodicCellOptions& options
)
{
    validate_periodic_cell(options);
    if (!options.enabled) {
        return position;
    }
    const auto wrap_axis = [](const double value, const double centre,
                              const double length) {
        const double lower = centre - 0.5 * length;
        const double image = std::floor((value - lower) / length);
        double wrapped = value - image * length;
        const double upper = lower + length;
        if (wrapped >= upper) {
            wrapped = lower;
        }
        if (wrapped < lower) {
            wrapped = lower;
        }
        return wrapped;
    };
    return {
        wrap_axis(position.x, options.centre.x, options.lengths.x),
        wrap_axis(position.y, options.centre.y, options.lengths.y),
        wrap_axis(position.z, options.centre.z, options.lengths.z),
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

std::vector<double> periodic_laplace_derivatives_raw(
    const MultiIndexSet& basis,
    const PeriodicCellOptions& options,
    const int explicit_image_radius
)
{
    validate_periodic_cell(options);
    if (!options.enabled) {
        throw std::invalid_argument(
            "periodic derivatives require an enabled periodic cell");
    }
    if (explicit_image_radius < 0) {
        throw std::invalid_argument(
            "explicit periodic image radius must be non-negative");
    }

    const double cell_length = options.lengths.x;
    const double volume = cell_length * cell_length * cell_length;
    const double ewald_alpha = std::sqrt(std::numbers::pi) / cell_length;
    const int cutoff =
        ewald_image_cutoff(options.setup_tolerance, basis.order());

    // Coefficients use D_alpha/alpha! until the final conversion to the raw
    // derivative convention consumed by the M2L builders.
    std::vector<double> coefficients(
        static_cast<std::size_t>(basis.size()), 0.0);

    for (int iz = -cutoff; iz <= cutoff; ++iz) {
        for (int iy = -cutoff; iy <= cutoff; ++iy) {
            for (int ix = -cutoff; ix <= cutoff; ++ix) {
                if (is_central_image(ix, iy, iz)) {
                    add_radial_taylor(
                        basis, Vec3{},
                        self_radial_coefficients(ewald_alpha, basis.order()),
                        1.0, coefficients);
                    continue;
                }
                const Vec3 image{
                    static_cast<double>(ix) * cell_length,
                    static_cast<double>(iy) * cell_length,
                    static_cast<double>(iz) * cell_length,
                };
                const double radius_squared = dot(image, image);
                add_radial_taylor(
                    basis, image,
                    screened_radial_coefficients(
                        radius_squared, ewald_alpha, basis.order()),
                    1.0, coefficients);
            }
        }
    }

    const double reciprocal_unit =
        2.0 * std::numbers::pi / cell_length;
    for (int iz = -cutoff; iz <= cutoff; ++iz) {
        for (int iy = -cutoff; iy <= cutoff; ++iy) {
            for (int ix = -cutoff; ix <= cutoff; ++ix) {
                if (is_central_image(ix, iy, iz)) {
                    continue;
                }
                const Vec3 wave{
                    reciprocal_unit * static_cast<double>(ix),
                    reciprocal_unit * static_cast<double>(iy),
                    reciprocal_unit * static_cast<double>(iz),
                };
                const double wave_squared = dot(wave, wave);
                const double weight = std::exp(
                    -wave_squared /
                    (4.0 * ewald_alpha * ewald_alpha)) /
                    (volume * wave_squared);
                for (int index = 0; index < basis.size(); ++index) {
                    const MultiIndex alpha = basis[index];
                    const int degree = alpha.ax + alpha.ay + alpha.az;
                    if ((degree & 1) != 0) {
                        continue;
                    }
                    const double phase = degree % 4 == 0 ? 1.0 : -1.0;
                    const double derivative =
                        phase * integer_power(wave.x, alpha.ax) *
                        integer_power(wave.y, alpha.ay) *
                        integer_power(wave.z, alpha.az);
                    coefficients[static_cast<std::size_t>(index)] +=
                        weight * derivative /
                        MultiIndexSet::multi_factorial(alpha);
                }
            }
        }
    }

    coefficients[static_cast<std::size_t>(basis.index({0, 0, 0}))] -=
        1.0 / (4.0 * ewald_alpha * ewald_alpha * volume);

    // Wrapped list1/list2 traversal explicitly covers the neighbouring root
    // images. Remove those free-space translations from the Ewald-periodised
    // regular Green function so every image is represented exactly once.
    for (int iz = -explicit_image_radius; iz <= explicit_image_radius; ++iz) {
        for (int iy = -explicit_image_radius; iy <= explicit_image_radius;
             ++iy) {
            for (int ix = -explicit_image_radius; ix <= explicit_image_radius;
                 ++ix) {
                if (is_central_image(ix, iy, iz)) {
                    continue;
                }
                const Vec3 image{
                    static_cast<double>(ix) * cell_length,
                    static_cast<double>(iy) * cell_length,
                    static_cast<double>(iz) * cell_length,
                };
                add_radial_taylor(
                    basis, image,
                    laplace_radial_coefficients(dot(image, image),
                                                 basis.order()),
                    -1.0, coefficients);
            }
        }
    }

    std::vector<double> result(static_cast<std::size_t>(basis.size()), 0.0);
    for (int index = 0; index < basis.size(); ++index) {
        const MultiIndex alpha = basis[index];
        const int degree = alpha.ax + alpha.ay + alpha.az;
        if ((degree & 1) == 0) {
            result[static_cast<std::size_t>(index)] =
                coefficients[static_cast<std::size_t>(index)] *
                MultiIndexSet::multi_factorial(alpha);
        }
    }
    return result;
}

} // namespace cdfmm
