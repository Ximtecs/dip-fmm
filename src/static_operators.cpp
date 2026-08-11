// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/static_operators.hpp"

#include <stdexcept>

#include "cdfmm/laplace_derivatives.hpp"

namespace cdfmm {

namespace {

void validate_operator_dimensions(
    const StaticCoefficientOperator& operator_map,
    const std::span<const double> input,
    const std::span<double> output)
{
    if (input.size() != static_cast<std::size_t>(operator_map.input_size) ||
        output.size() != static_cast<std::size_t>(operator_map.output_size)) {
        throw std::invalid_argument("static operator dimensions are inconsistent");
    }
}

} // namespace

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
                {alpha.ax, alpha.ay, alpha.az - 1}
            };
            const int components[3] = {alpha.ax, alpha.ay, alpha.az};
            for (int component = 0; component < 3; ++component) {
                if (components[component] == 0) {
                    continue;
                }
                result.entries.push_back({
                    alpha_index,
                    static_cast<int>(3 * source) + component,
                    sign * MultiIndexSet::monomial_over_factorial(
                        dx, shifted[component]
                    )
                });
            }
        }
    }
    return result;
}

StaticCoefficientOperator build_static_m2m_operator(
    const MultiIndexSet& basis,
    const Vec3& d)
{
    StaticCoefficientOperator result{basis.size(), basis.size(), {}};
    for (int alpha_index = 0; alpha_index < basis.size(); ++alpha_index) {
        const MultiIndex alpha = basis[alpha_index];
        for (int gamma_index = 0; gamma_index < basis.size(); ++gamma_index) {
            const MultiIndex gamma = basis[gamma_index];
            if (leq(gamma, alpha)) {
                result.entries.push_back({
                    alpha_index,
                    basis.index(sub(alpha, gamma)),
                    MultiIndexSet::monomial_over_factorial(d, gamma)
                });
            }
        }
    }
    return result;
}

StaticCoefficientOperator build_static_l2l_operator(
    const MultiIndexSet& basis,
    const Vec3& d)
{
    StaticCoefficientOperator result{basis.size(), basis.size(), {}};
    for (int beta_index = 0; beta_index < basis.size(); ++beta_index) {
        const MultiIndex beta = basis[beta_index];
        for (int gamma_index = 0; gamma_index < basis.size(); ++gamma_index) {
            const MultiIndex gamma = basis[gamma_index];
            const MultiIndex sum = add(beta, gamma);
            if (sum.degree() <= basis.order()) {
                result.entries.push_back({
                    beta_index,
                    basis.index(sum),
                    MultiIndexSet::monomial_over_factorial(d, gamma)
                });
            }
        }
    }
    return result;
}

StaticL2PEvaluator build_static_l2p_evaluator(
    const MultiIndexSet& basis,
    const Vec3& centre,
    const Vec3& target)
{
    StaticL2PEvaluator result;
    result.potential.resize(static_cast<std::size_t>(basis.size()));
    for (auto& row : result.field) {
        row.resize(static_cast<std::size_t>(basis.size()));
    }
    const Vec3 dx = target - centre;
    for (int beta_index = 0; beta_index < basis.size(); ++beta_index) {
        const MultiIndex beta = basis[beta_index];
        result.potential[static_cast<std::size_t>(beta_index)] =
            MultiIndexSet::monomial_over_factorial(dx, beta);
        if (beta.ax > 0) {
            result.field[0][static_cast<std::size_t>(beta_index)] =
                -MultiIndexSet::monomial_over_factorial(
                    dx, {beta.ax - 1, beta.ay, beta.az}
                );
        }
        if (beta.ay > 0) {
            result.field[1][static_cast<std::size_t>(beta_index)] =
                -MultiIndexSet::monomial_over_factorial(
                    dx, {beta.ax, beta.ay - 1, beta.az}
                );
        }
        if (beta.az > 0) {
            result.field[2][static_cast<std::size_t>(beta_index)] =
                -MultiIndexSet::monomial_over_factorial(
                    dx, {beta.ax, beta.ay, beta.az - 1}
                );
        }
    }
    return result;
}

void apply_static_operator(
    const StaticCoefficientOperator& operator_map,
    const std::span<const double> input,
    const std::span<double> output)
{
    validate_operator_dimensions(operator_map, input, output);
    for (const StaticOperatorEntry& entry : operator_map.entries) {
        output[static_cast<std::size_t>(entry.output)] += entry.value *
            input[static_cast<std::size_t>(entry.input)];
    }
}

PotentialField apply_static_l2p_evaluator(
    const StaticL2PEvaluator& evaluator,
    const std::span<const double> L,
    const OutputFlags output)
{
    if (evaluator.potential.size() != L.size()) {
        throw std::invalid_argument("static L2P dimensions are inconsistent");
    }
    PotentialField result;
    for (std::size_t index = 0; index < L.size(); ++index) {
        if (has_flag(output, OutputFlags::Potential)) {
            result.phi += evaluator.potential[index] * L[index];
        }
        if (has_flag(output, OutputFlags::Field)) {
            result.H.x += evaluator.field[0][index] * L[index];
            result.H.y += evaluator.field[1][index] * L[index];
            result.H.z += evaluator.field[2][index] * L[index];
        }
    }
    return result;
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
