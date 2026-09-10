// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/operators/m2l.hpp"

#include <cstddef>
#include <numbers>
#include <vector>

#include "cdfmm/laplace_derivatives.hpp"
#include "spherical_cartesian_conversion.hpp"

namespace cdfmm {

std::vector<double> build_static_m2l_matrix(
    const MultiIndexSet& basis,
    const Vec3& displacement)
{
    const int coefficient_count = basis.size();
    const MultiIndexSet derivative_basis(2 * basis.order());
    const CoeffVector derivatives =
        laplace_derivatives_raw(derivative_basis, displacement);
    std::vector<double> matrix(
        static_cast<std::size_t>(coefficient_count) * coefficient_count);
    for (int alpha_index = 0; alpha_index < coefficient_count; ++alpha_index) {
        for (int beta_index = 0; beta_index < coefficient_count;
             ++beta_index) {
            const MultiIndex gamma =
                add(basis[alpha_index], basis[beta_index]);
            matrix[static_cast<std::size_t>(beta_index) +
                   static_cast<std::size_t>(coefficient_count) * alpha_index] =
                derivatives[static_cast<std::size_t>(
                    derivative_basis.index(gamma))];
        }
    }
    return matrix;
}

std::vector<double> build_static_m2l_matrix(
    const SphericalHarmonicBasis& basis,
    const Vec3& displacement)
{
    const MultiIndexSet derivative_basis(2 * basis.order());
    const CoeffVector derivatives =
        laplace_derivatives_raw(derivative_basis, displacement);
    std::vector<double> matrix(
        static_cast<std::size_t>(basis.size()) * basis.size(), 0.0);
    for (int input = 0; input < basis.size(); ++input) {
        const int input_degree = basis[input].l;
        const double input_factor =
            4.0 * std::numbers::pi *
            (input_degree % 2 == 0 ? 1.0 : -1.0) /
            operators::detail::odd_double_factorial(input_degree);
        for (int output = 0; output < basis.size(); ++output) {
            const double factor = input_factor /
                operators::detail::odd_double_factorial(basis[output].l);
            double value = 0.0;
            for (const SolidHarmonicTerm& source_term :
                 basis.polynomial(input)) {
                for (const SolidHarmonicTerm& target_term :
                     basis.polynomial(output)) {
                    const MultiIndex derivative =
                        add(source_term.power, target_term.power);
                    value += source_term.coefficient * target_term.coefficient *
                        derivatives[static_cast<std::size_t>(
                            derivative_basis.index(derivative))];
                }
            }
            matrix[static_cast<std::size_t>(output) +
                   static_cast<std::size_t>(basis.size()) * input] =
                factor * value;
        }
    }
    return matrix;
}

std::vector<double> build_static_periodic_m2l_matrix(
    const MultiIndexSet& basis,
    const PeriodicCellOptions& options)
{
    PeriodicCellOptions normalised = options;
    normalised.centre = {};
    normalised.lengths = {1.0, 1.0, 1.0};
    const MultiIndexSet derivative_basis(2 * basis.order());
    const std::vector<double> derivatives =
        periodic_laplace_derivatives_raw(derivative_basis, normalised);
    const int coefficient_count = basis.size();
    std::vector<double> matrix(
        static_cast<std::size_t>(coefficient_count) * coefficient_count, 0.0);
    for (int alpha_index = 0; alpha_index < coefficient_count; ++alpha_index) {
        for (int beta_index = 0; beta_index < coefficient_count;
             ++beta_index) {
            const MultiIndex gamma =
                add(basis[alpha_index], basis[beta_index]);
            matrix[static_cast<std::size_t>(beta_index) +
                   static_cast<std::size_t>(coefficient_count) * alpha_index] =
                derivatives[static_cast<std::size_t>(
                    derivative_basis.index(gamma))];
        }
    }
    return matrix;
}

std::vector<double> build_static_periodic_m2l_matrix(
    const SphericalHarmonicBasis& basis,
    const PeriodicCellOptions& options)
{
    PeriodicCellOptions normalised = options;
    normalised.centre = {};
    normalised.lengths = {1.0, 1.0, 1.0};
    const MultiIndexSet derivative_basis(2 * basis.order());
    const std::vector<double> derivatives =
        periodic_laplace_derivatives_raw(derivative_basis, normalised);
    std::vector<double> matrix(
        static_cast<std::size_t>(basis.size()) * basis.size(), 0.0);
    for (int input = 0; input < basis.size(); ++input) {
        const int input_degree = basis[input].l;
        const double input_factor =
            4.0 * std::numbers::pi *
            (input_degree % 2 == 0 ? 1.0 : -1.0) /
            operators::detail::odd_double_factorial(input_degree);
        for (int output = 0; output < basis.size(); ++output) {
            const double factor = input_factor /
                operators::detail::odd_double_factorial(basis[output].l);
            double value = 0.0;
            for (const SolidHarmonicTerm& source_term :
                 basis.polynomial(input)) {
                for (const SolidHarmonicTerm& target_term :
                     basis.polynomial(output)) {
                    const MultiIndex derivative =
                        add(source_term.power, target_term.power);
                    value += source_term.coefficient * target_term.coefficient *
                        derivatives[static_cast<std::size_t>(
                            derivative_basis.index(derivative))];
                }
            }
            matrix[static_cast<std::size_t>(output) +
                   static_cast<std::size_t>(basis.size()) * input] =
                factor * value;
        }
    }
    return matrix;
}

} // namespace cdfmm

namespace cdfmm::operators::m2l {

std::vector<double> build_matrix(const MultiIndexSet& basis,
                                 const Vec3& displacement)
{
    return build_static_m2l_matrix(basis, displacement);
}

std::vector<double> build_matrix(const SphericalHarmonicBasis& basis,
                                 const Vec3& displacement)
{
    return build_static_m2l_matrix(basis, displacement);
}

std::vector<double> build_periodic_matrix(
    const MultiIndexSet& basis,
    const PeriodicCellOptions& options)
{
    return build_static_periodic_m2l_matrix(basis, options);
}

std::vector<double> build_periodic_matrix(
    const SphericalHarmonicBasis& basis,
    const PeriodicCellOptions& options)
{
    return build_static_periodic_m2l_matrix(basis, options);
}

void apply(const MultiIndexSet& basis,
           const Vec3& displacement,
           const std::span<const double> multipole,
           const std::span<double> local)
{
    // Convert source multipole coefficients M to target local coefficients L
    // with R = c_target - c_source.
    // For order p, alpha+beta reaches total degree 2p.
    MultiIndexSet deriv_basis(2 * basis.order());
    const auto D = laplace_derivatives_raw(deriv_basis, displacement);

    for (int ib = 0; ib < basis.size(); ++ib) {
        const MultiIndex beta = basis[ib];

        for (int ia = 0; ia < basis.size(); ++ia) {
            const MultiIndex alpha = basis[ia];
            local[ib] += multipole[ia] * D[deriv_basis.index(add(alpha, beta))];
        }
    }
}

} // namespace cdfmm::operators::m2l
