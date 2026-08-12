// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/static_operators.hpp"

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <limits>

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

std::size_t StaticP2POperator::memory_bytes() const noexcept
{
    return row_offsets.size() * sizeof(int) +
        blocks.size() * sizeof(StaticDipoleBlock);
}

StaticP2POperator build_static_p2p_operator(
    const std::span<const Vec3> target_positions,
    const std::span<const Vec3> source_positions,
    const std::span<const std::array<int, 2>> interactions)
{
    StaticP2POperator result;
    result.source_count = static_cast<int>(source_positions.size());
    result.target_count = static_cast<int>(target_positions.size());
    result.row_offsets.assign(target_positions.size() + 1, 0);

    std::vector<std::array<int, 2>> sorted(interactions.begin(), interactions.end());
    std::sort(sorted.begin(), sorted.end());
    result.blocks.reserve(sorted.size());
    constexpr double prefactor = 1.0 / (4.0 * std::numbers::pi);
    for (const auto pair : sorted) {
        const int target = pair[0];
        const int source = pair[1];
        if (target < 0 || target >= result.target_count || source < 0 ||
            source >= result.source_count) {
            throw std::invalid_argument("static P2P interaction index is invalid");
        }
        const Vec3 r = target_positions[static_cast<std::size_t>(target)] -
            source_positions[static_cast<std::size_t>(source)];
        const double r2 = dot(r, r);
        if (r2 == 0.0) {
            // Preserve the block so an explicit identity map can skip it. If
            // it is not a self pair, NaNs deliberately expose the same
            // undefined point-dipole singularity as the reference operator.
            const double undefined = std::numeric_limits<double>::quiet_NaN();
            result.blocks.push_back({target, source, undefined, undefined,
                                     undefined, undefined, undefined, undefined});
            ++result.row_offsets[static_cast<std::size_t>(target) + 1];
            continue;
        }
        const double inverse_r = 1.0 / std::sqrt(r2);
        const double inverse_r3 = inverse_r / r2;
        const double common = 3.0 * prefactor * inverse_r3 / r2;
        const double diagonal = prefactor * inverse_r3;
        result.blocks.push_back({
            target, source,
            common * r.x * r.x - diagonal,
            common * r.x * r.y,
            common * r.x * r.z,
            common * r.y * r.y - diagonal,
            common * r.y * r.z,
            common * r.z * r.z - diagonal
        });
        ++result.row_offsets[static_cast<std::size_t>(target) + 1];
    }
    for (std::size_t row = 1; row < result.row_offsets.size(); ++row) {
        result.row_offsets[row] += result.row_offsets[row - 1];
    }
    return result;
}

void apply_static_p2p_operator(
    const StaticP2POperator& operator_map,
    const std::span<const Vec3> dipole_moments,
    const std::span<Vec3> H,
    const std::span<const int> target_source_indices)
{
    if (dipole_moments.size() != static_cast<std::size_t>(operator_map.source_count) ||
        H.size() != static_cast<std::size_t>(operator_map.target_count) ||
        (!target_source_indices.empty() && target_source_indices.size() != H.size())) {
        throw std::invalid_argument("static P2P dimensions are inconsistent");
    }
    #pragma omp parallel for schedule(static) if(operator_map.target_count >= 64)
    for (int target = 0; target < operator_map.target_count; ++target) {
        Vec3 field{};
        const int self = target_source_indices.empty()
            ? -1 : target_source_indices[static_cast<std::size_t>(target)];
        for (int entry = operator_map.row_offsets[static_cast<std::size_t>(target)];
             entry < operator_map.row_offsets[static_cast<std::size_t>(target) + 1];
             ++entry) {
            const StaticDipoleBlock& block = operator_map.blocks[
                static_cast<std::size_t>(entry)];
            if (block.source == self) {
                continue;
            }
            const Vec3 m = dipole_moments[static_cast<std::size_t>(block.source)];
            field.x += block.xx * m.x + block.xy * m.y + block.xz * m.z;
            field.y += block.xy * m.x + block.yy * m.y + block.yz * m.z;
            field.z += block.xz * m.x + block.yz * m.y + block.zz * m.z;
        }
        H[static_cast<std::size_t>(target)] += field;
    }
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
