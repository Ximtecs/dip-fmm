// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/operators/p2m.hpp"

#include <cstddef>
#include <numbers>
#include <stdexcept>

namespace cdfmm {

StaticCoefficientOperator build_static_p2m_operator(
    const MultiIndexSet& basis,
    const Vec3& centre,
    const std::span<const Vec3> source_positions)
{
    StaticCoefficientOperator result;
    result.input_size = static_cast<int>(3 * source_positions.size());
    result.output_size = basis.size();
    for (std::size_t source = 0; source < source_positions.size(); ++source) {
        const Vec3 dx = source_positions[source] - centre;
        for (int alpha_index = 0; alpha_index < basis.size(); ++alpha_index) {
            const MultiIndex alpha = basis[alpha_index];
            const double sign = alpha.degree() % 2 == 0 ? 1.0 : -1.0;
            const MultiIndex shifted[3] = {
                {alpha.ax - 1, alpha.ay, alpha.az},
                {alpha.ax, alpha.ay - 1, alpha.az},
                {alpha.ax, alpha.ay, alpha.az - 1}};
            const int components[3] = {alpha.ax, alpha.ay, alpha.az};
            for (int component = 0; component < 3; ++component) {
                if (components[component] == 0) {
                    continue;
                }
                result.entries.push_back({
                    alpha_index, static_cast<int>(3 * source) + component,
                    sign * MultiIndexSet::monomial_over_factorial(
                        dx, shifted[component])});
            }
        }
    }
    return result;
}

StaticCoefficientOperator build_static_p2m_operator(
    const SphericalHarmonicBasis& basis,
    const Vec3& centre,
    const std::span<const Vec3> source_positions)
{
    StaticCoefficientOperator result;
    result.input_size = static_cast<int>(3 * source_positions.size());
    result.output_size = basis.size();
    const double green_factor = 1.0 / (4.0 * std::numbers::pi);
    for (std::size_t source = 0; source < source_positions.size(); ++source) {
        const SolidHarmonicValues regular = regular_solid_harmonics(
            basis, source_positions[source] - centre);
        for (int mode = 0; mode < basis.size(); ++mode) {
            const Vec3 gradient =
                regular.gradients[static_cast<std::size_t>(mode)];
            for (int component = 0; component < 3; ++component) {
                const double value = green_factor * gradient[component];
                if (value != 0.0) {
                    result.entries.push_back({
                        mode, static_cast<int>(3 * source) + component,
                        value});
                }
            }
        }
    }
    return result;
}

StaticCoefficientOperator build_static_cuboid_p2m_operator(
    const MultiIndexSet& basis,
    const Vec3& centre,
    const std::span<const Vec3> source_positions,
    const std::span<const CuboidSize> source_sizes)
{
    if (source_sizes.size() != 1 &&
        source_sizes.size() != source_positions.size()) {
        throw std::invalid_argument(
            "cuboid P2M sizes must be common or per source");
    }
    StaticCoefficientOperator result;
    result.input_size = static_cast<int>(3 * source_positions.size());
    result.output_size = basis.size();
    for (std::size_t source = 0; source < source_positions.size(); ++source) {
        const Vec3 displacement = source_positions[source] - centre;
        const CuboidSize size =
            source_sizes[source_sizes.size() == 1 ? 0 : source];
        for (int alpha_index = 0; alpha_index < basis.size(); ++alpha_index) {
            const MultiIndex alpha = basis[alpha_index];
            const double sign = alpha.degree() % 2 == 0 ? 1.0 : -1.0;
            const MultiIndex shifted[3] = {
                {alpha.ax - 1, alpha.ay, alpha.az},
                {alpha.ax, alpha.ay - 1, alpha.az},
                {alpha.ax, alpha.ay, alpha.az - 1}};
            const int components[3] = {alpha.ax, alpha.ay, alpha.az};
            for (int component = 0; component < 3; ++component) {
                if (components[component] > 0) {
                    result.entries.push_back({
                        alpha_index,
                        static_cast<int>(3 * source) + component,
                        sign * cuboid_averaged_monomial(
                            shifted[component], displacement, size)});
                }
            }
        }
    }
    return result;
}

StaticCoefficientOperator build_static_cuboid_p2m_operator(
    const SphericalHarmonicBasis& basis,
    const Vec3& centre,
    const std::span<const Vec3> source_positions,
    const std::span<const CuboidSize> source_sizes)
{
    if (source_sizes.size() != 1 &&
        source_sizes.size() != source_positions.size()) {
        throw std::invalid_argument(
            "spherical cuboid P2M sizes must be common or per source");
    }
    StaticCoefficientOperator result;
    result.input_size = static_cast<int>(3 * source_positions.size());
    result.output_size = basis.size();
    const double green_factor = 1.0 / (4.0 * std::numbers::pi);
    for (std::size_t source = 0; source < source_positions.size(); ++source) {
        const Vec3 displacement = source_positions[source] - centre;
        const CuboidSize size =
            source_sizes[source_sizes.size() == 1 ? 0 : source];
        for (int mode = 0; mode < basis.size(); ++mode) {
            double averaged_gradient[3]{0.0, 0.0, 0.0};
            for (const SolidHarmonicTerm& term : basis.polynomial(mode)) {
                const int powers[3] = {
                    term.power.ax, term.power.ay, term.power.az};
                for (int component = 0; component < 3; ++component) {
                    if (powers[component] == 0) {
                        continue;
                    }
                    MultiIndex derivative = term.power;
                    if (component == 0) {
                        --derivative.ax;
                    } else if (component == 1) {
                        --derivative.ay;
                    } else {
                        --derivative.az;
                    }
                    averaged_gradient[component] +=
                        term.coefficient * powers[component] *
                        MultiIndexSet::multi_factorial(derivative) *
                        cuboid_averaged_monomial(
                            derivative, displacement, size);
                }
            }
            for (int component = 0; component < 3; ++component) {
                const double value =
                    green_factor * averaged_gradient[component];
                if (value != 0.0) {
                    result.entries.push_back({
                        mode, static_cast<int>(3 * source) + component,
                        value});
                }
            }
        }
    }
    return result;
}

StaticCoefficientOperator build_static_tetrahedron_p2m_operator(
    const MultiIndexSet& basis,
    const Vec3& centre,
    const std::span<const Vec3> source_positions,
    const std::span<const Tetrahedron> source_tetrahedra)
{
    if (source_tetrahedra.size() != 1 &&
        source_tetrahedra.size() != source_positions.size()) {
        throw std::invalid_argument(
            "tetrahedron P2M geometries must be common or per source");
    }
    for (const Tetrahedron& tetrahedron : source_tetrahedra) {
        static_cast<void>(tetrahedron_volume(tetrahedron));
    }

    StaticCoefficientOperator result;
    result.input_size = static_cast<int>(3 * source_positions.size());
    result.output_size = basis.size();
    for (std::size_t source = 0; source < source_positions.size(); ++source) {
        const Vec3 displacement = source_positions[source] - centre;
        const Tetrahedron& tetrahedron = source_tetrahedra[
            source_tetrahedra.size() == 1 ? 0 : source];
        for (int alpha_index = 0; alpha_index < basis.size(); ++alpha_index) {
            const MultiIndex alpha = basis[alpha_index];
            const double sign = alpha.degree() % 2 == 0 ? 1.0 : -1.0;
            const MultiIndex shifted[3] = {
                {alpha.ax - 1, alpha.ay, alpha.az},
                {alpha.ax, alpha.ay - 1, alpha.az},
                {alpha.ax, alpha.ay, alpha.az - 1}};
            const int components[3] = {alpha.ax, alpha.ay, alpha.az};
            for (int component = 0; component < 3; ++component) {
                if (components[component] == 0) {
                    continue;
                }
                const double value = sign * tetrahedron_averaged_monomial(
                    shifted[component], displacement, tetrahedron);
                if (value != 0.0) {
                    result.entries.push_back({
                        alpha_index,
                        static_cast<int>(3 * source) + component, value});
                }
            }
        }
    }
    return result;
}

StaticCoefficientOperator build_static_tetrahedron_p2m_operator(
    const SphericalHarmonicBasis& basis,
    const Vec3& centre,
    const std::span<const Vec3> source_positions,
    const std::span<const Tetrahedron> source_tetrahedra)
{
    if (source_tetrahedra.size() != 1 &&
        source_tetrahedra.size() != source_positions.size()) {
        throw std::invalid_argument(
            "spherical tetrahedron P2M geometries must be common or per source");
    }
    for (const Tetrahedron& tetrahedron : source_tetrahedra) {
        static_cast<void>(tetrahedron_volume(tetrahedron));
    }

    StaticCoefficientOperator result;
    result.input_size = static_cast<int>(3 * source_positions.size());
    result.output_size = basis.size();
    const double green_factor = 1.0 / (4.0 * std::numbers::pi);
    for (std::size_t source = 0; source < source_positions.size(); ++source) {
        const Vec3 displacement = source_positions[source] - centre;
        const Tetrahedron& tetrahedron = source_tetrahedra[
            source_tetrahedra.size() == 1 ? 0 : source];
        for (int mode = 0; mode < basis.size(); ++mode) {
            double averaged_gradient[3]{0.0, 0.0, 0.0};
            for (const SolidHarmonicTerm& term : basis.polynomial(mode)) {
                const int powers[3] = {
                    term.power.ax, term.power.ay, term.power.az};
                for (int component = 0; component < 3; ++component) {
                    if (powers[component] == 0) {
                        continue;
                    }
                    MultiIndex derivative = term.power;
                    if (component == 0) {
                        --derivative.ax;
                    } else if (component == 1) {
                        --derivative.ay;
                    } else {
                        --derivative.az;
                    }
                    averaged_gradient[component] +=
                        term.coefficient * powers[component] *
                        MultiIndexSet::multi_factorial(derivative) *
                        tetrahedron_averaged_monomial(
                            derivative, displacement, tetrahedron);
                }
            }
            for (int component = 0; component < 3; ++component) {
                const double value =
                    green_factor * averaged_gradient[component];
                if (value != 0.0) {
                    result.entries.push_back({
                        mode, static_cast<int>(3 * source) + component,
                        value});
                }
            }
        }
    }
    return result;
}

} // namespace cdfmm

namespace cdfmm::operators::p2m {

StaticCoefficientOperator build(
    const MultiIndexSet& basis, const Vec3& centre,
    const std::span<const Vec3> source_positions)
{
    return build_static_p2m_operator(basis, centre, source_positions);
}

StaticCoefficientOperator build(
    const SphericalHarmonicBasis& basis, const Vec3& centre,
    const std::span<const Vec3> source_positions)
{
    return build_static_p2m_operator(basis, centre, source_positions);
}

StaticCoefficientOperator build_cuboid(
    const MultiIndexSet& basis, const Vec3& centre,
    const std::span<const Vec3> source_positions,
    const std::span<const CuboidSize> source_sizes)
{
    return build_static_cuboid_p2m_operator(
        basis, centre, source_positions, source_sizes);
}

StaticCoefficientOperator build_cuboid(
    const SphericalHarmonicBasis& basis, const Vec3& centre,
    const std::span<const Vec3> source_positions,
    const std::span<const CuboidSize> source_sizes)
{
    return build_static_cuboid_p2m_operator(
        basis, centre, source_positions, source_sizes);
}

StaticCoefficientOperator build_tetrahedron(
    const MultiIndexSet& basis, const Vec3& centre,
    const std::span<const Vec3> source_positions,
    const std::span<const Tetrahedron> source_tetrahedra)
{
    return build_static_tetrahedron_p2m_operator(
        basis, centre, source_positions, source_tetrahedra);
}

StaticCoefficientOperator build_tetrahedron(
    const SphericalHarmonicBasis& basis, const Vec3& centre,
    const std::span<const Vec3> source_positions,
    const std::span<const Tetrahedron> source_tetrahedra)
{
    return build_static_tetrahedron_p2m_operator(
        basis, centre, source_positions, source_tetrahedra);
}

CoeffVector evaluate(
    const MultiIndexSet& basis, const Vec3& centre,
    const std::span<const Vec3> source_positions,
    const std::span<const Vec3> dipole_moments)
{
    CoeffVector M(basis.size(), 0.0);

    // Dipole sources contribute via first derivatives, hence alpha-e_k terms.
    // For alpha=(0,0,0) all components are excluded, so M_0 is zero by
    // construction for pure dipole input.
    for (size_t j = 0; j < source_positions.size(); ++j) {
        const Vec3 dx = source_positions[j] - centre;

        for (int i = 0; i < basis.size(); ++i) {
            const MultiIndex alpha = basis[i];
            double value = 0.0;

            if (alpha.ax > 0) {
                value += dipole_moments[j].x * MultiIndexSet::monomial_over_factorial(
                    dx, {alpha.ax - 1, alpha.ay, alpha.az});
            }
            if (alpha.ay > 0) {
                value += dipole_moments[j].y * MultiIndexSet::monomial_over_factorial(
                    dx, {alpha.ax, alpha.ay - 1, alpha.az});
            }
            if (alpha.az > 0) {
                value += dipole_moments[j].z * MultiIndexSet::monomial_over_factorial(
                    dx, {alpha.ax, alpha.ay, alpha.az - 1});
            }

            // (-1)^|alpha| matches the repository dipole-potential convention.
            const double sign = (alpha.degree() % 2 == 0) ? 1.0 : -1.0;
            M[i] += sign * value;
        }
    }

    return M;
}

} // namespace cdfmm::operators::p2m
