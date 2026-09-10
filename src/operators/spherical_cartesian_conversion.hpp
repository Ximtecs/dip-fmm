// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

#include "cdfmm/math/multi_index.hpp"
#include "cdfmm/math/spherical_harmonics.hpp"
#include "cdfmm/plan/static_coefficient.hpp"

namespace cdfmm::operators::detail {

inline double odd_double_factorial(const int l)
{
    double result = 1.0;
    for (int value = 1; value <= 2 * l - 1; value += 2) {
        result *= static_cast<double>(value);
    }
    return result;
}

struct SphericalCartesianMaps {
    int cartesian_count{0};
    int spherical_count{0};
    std::vector<double> multipole_embedding{};
    std::vector<double> multipole_projection{};
    std::vector<double> local_embedding{};
    std::vector<double> local_projection{};
};

inline SphericalCartesianMaps make_spherical_cartesian_maps(
    const SphericalHarmonicBasis& spherical,
    const MultiIndexSet& cartesian)
{
    SphericalCartesianMaps maps;
    maps.cartesian_count = cartesian.size();
    maps.spherical_count = spherical.size();
    const std::size_t values =
        static_cast<std::size_t>(cartesian.size()) * spherical.size();
    maps.multipole_embedding.assign(values, 0.0);
    maps.multipole_projection.assign(values, 0.0);
    maps.local_embedding.assign(values, 0.0);
    maps.local_projection.assign(values, 0.0);
    for (int mode = 0; mode < spherical.size(); ++mode) {
        const int l = spherical[mode].l;
        const double sign = l % 2 == 0 ? 1.0 : -1.0;
        const double double_factorial = odd_double_factorial(l);
        for (const SolidHarmonicTerm& term : spherical.polynomial(mode)) {
            const int alpha = cartesian.index(term.power);
            const double alpha_factorial =
                MultiIndexSet::multi_factorial(term.power);
            const std::size_t embedding_index =
                static_cast<std::size_t>(alpha) * spherical.size() + mode;
            const std::size_t projection_index =
                static_cast<std::size_t>(mode) * cartesian.size() + alpha;
            maps.multipole_embedding[embedding_index] =
                4.0 * std::numbers::pi * sign * term.coefficient /
                double_factorial;
            maps.multipole_projection[projection_index] =
                sign * term.coefficient * alpha_factorial /
                (4.0 * std::numbers::pi);
            maps.local_embedding[embedding_index] =
                alpha_factorial * term.coefficient;
            maps.local_projection[projection_index] =
                term.coefficient / double_factorial;
        }
    }
    return maps;
}

inline std::vector<double> compose_spherical_translation(
    const StaticCoefficientOperator& cartesian_operator,
    const std::span<const double> input_embedding,
    const std::span<const double> output_projection,
    const int cartesian_count,
    const int spherical_count)
{
    std::vector<double> intermediate(
        static_cast<std::size_t>(cartesian_count) * spherical_count, 0.0);
    for (const StaticOperatorEntry& entry : cartesian_operator.entries) {
        for (int input_mode = 0; input_mode < spherical_count; ++input_mode) {
            intermediate[static_cast<std::size_t>(entry.output) *
                             spherical_count + input_mode] +=
                entry.value * input_embedding[
                    static_cast<std::size_t>(entry.input) * spherical_count +
                    input_mode];
        }
    }
    std::vector<double> result(
        static_cast<std::size_t>(spherical_count) * spherical_count, 0.0);
    for (int output_mode = 0; output_mode < spherical_count; ++output_mode) {
        for (int cartesian = 0; cartesian < cartesian_count; ++cartesian) {
            const double projection = output_projection[
                static_cast<std::size_t>(output_mode) * cartesian_count +
                cartesian];
            if (projection == 0.0) {
                continue;
            }
            for (int input_mode = 0; input_mode < spherical_count;
                 ++input_mode) {
                result[static_cast<std::size_t>(output_mode) +
                       static_cast<std::size_t>(spherical_count) *
                           input_mode] +=
                    projection * intermediate[
                        static_cast<std::size_t>(cartesian) * spherical_count +
                        input_mode];
            }
        }
    }
    return result;
}

inline StaticCoefficientOperator pack_spherical_translation(
    const SphericalHarmonicBasis& basis,
    const std::span<const double> matrix,
    const bool multipole_translation)
{
    StaticCoefficientOperator result;
    result.input_size = basis.size();
    result.output_size = basis.size();
    for (int input = 0; input < basis.size(); ++input) {
        for (int output = 0; output < basis.size(); ++output) {
            const bool structurally_valid = multipole_translation
                ? basis[output].l >= basis[input].l
                : basis[output].l <= basis[input].l;
            if (!structurally_valid) {
                continue;
            }
            result.entries.push_back({
                output, input,
                matrix[static_cast<std::size_t>(output) +
                       static_cast<std::size_t>(basis.size()) * input]});
        }
    }
    return result;
}

} // namespace cdfmm::operators::detail
