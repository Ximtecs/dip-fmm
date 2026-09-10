// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/operators/m2m.hpp"

#include "spherical_cartesian_conversion.hpp"

namespace cdfmm {

StaticCoefficientOperator build_static_m2m_operator(
    const MultiIndexSet& basis,
    const Vec3& displacement)
{
    StaticCoefficientOperator result{basis.size(), basis.size(), {}};
    for (int alpha_index = 0; alpha_index < basis.size(); ++alpha_index) {
        const MultiIndex alpha = basis[alpha_index];
        for (int gamma_index = 0; gamma_index < basis.size(); ++gamma_index) {
            const MultiIndex gamma = basis[gamma_index];
            if (leq(gamma, alpha)) {
                result.entries.push_back({
                    alpha_index, basis.index(sub(alpha, gamma)),
                    MultiIndexSet::monomial_over_factorial(
                        displacement, gamma)});
            }
        }
    }
    return result;
}

StaticCoefficientOperator build_static_m2m_operator(
    const SphericalHarmonicBasis& basis,
    const Vec3& displacement)
{
    const MultiIndexSet cartesian(basis.order());
    const operators::detail::SphericalCartesianMaps maps =
        operators::detail::make_spherical_cartesian_maps(basis, cartesian);
    const StaticCoefficientOperator cartesian_operator =
        build_static_m2m_operator(cartesian, displacement);
    const std::vector<double> matrix =
        operators::detail::compose_spherical_translation(
            cartesian_operator, maps.multipole_embedding,
            maps.multipole_projection, maps.cartesian_count,
            maps.spherical_count);
    return operators::detail::pack_spherical_translation(
        basis, matrix, true);
}

} // namespace cdfmm

namespace cdfmm::operators::m2m {

StaticCoefficientOperator build(const MultiIndexSet& basis,
                                const Vec3& displacement)
{
    return build_static_m2m_operator(basis, displacement);
}

StaticCoefficientOperator build(const SphericalHarmonicBasis& basis,
                                const Vec3& displacement)
{
    return build_static_m2m_operator(basis, displacement);
}

void apply(const MultiIndexSet& basis,
           const Vec3& displacement,
           const std::span<const double> child,
           const std::span<double> parent)
{
    m2m_add(basis, displacement, child, parent);
}

} // namespace cdfmm::operators::m2m
