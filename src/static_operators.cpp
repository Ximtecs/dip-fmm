// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/static_operators.hpp"

#include <stdexcept>

#include "cdfmm/laplace_derivatives.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Canonical static mathematical operators
//------------------------------------------------------------------------------

std::vector<double> build_static_m2l_matrix(
    const MultiIndexSet& basis,
    const Vec3& R)
{
    const int coefficient_count = basis.size();
    const MultiIndexSet derivative_basis(2 * basis.order());
    const CoeffVector derivatives = laplace_derivatives_raw(
        derivative_basis, R
    );
    std::vector<double> matrix(
        static_cast<std::size_t>(coefficient_count) * coefficient_count
    );
    for (int alpha_index = 0; alpha_index < coefficient_count; ++alpha_index) {
        for (int beta_index = 0; beta_index < coefficient_count; ++beta_index) {
            const MultiIndex gamma = add(
                basis[alpha_index], basis[beta_index]
            );
            matrix[static_cast<std::size_t>(beta_index) +
                   static_cast<std::size_t>(coefficient_count) * alpha_index] =
                derivatives[static_cast<std::size_t>(
                    derivative_basis.index(gamma)
                )];
        }
    }
    return matrix;
}

void apply_static_coefficient_matrix(
    const std::span<const double> matrix,
    const std::span<const double> input,
    const std::span<double> output)
{
    const std::size_t coefficient_count = input.size();
    if (output.size() != coefficient_count ||
        matrix.size() != coefficient_count * coefficient_count) {
        throw std::invalid_argument(
            "static coefficient matrix dimensions are inconsistent"
        );
    }
    for (std::size_t alpha = 0; alpha < coefficient_count; ++alpha) {
        for (std::size_t beta = 0; beta < coefficient_count; ++beta) {
            output[beta] += matrix[beta + coefficient_count * alpha] *
                input[alpha];
        }
    }
}

} // namespace cdfmm
