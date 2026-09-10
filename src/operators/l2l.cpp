// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/operators/l2l.hpp"

#include "spherical_cartesian_conversion.hpp"

namespace cdfmm {

StaticCoefficientOperator build_static_l2l_operator(
    const MultiIndexSet& basis,
    const Vec3& displacement)
{
    StaticCoefficientOperator result{basis.size(), basis.size(), {}};
    for (int beta_index = 0; beta_index < basis.size(); ++beta_index) {
        const MultiIndex beta = basis[beta_index];
        for (int gamma_index = 0; gamma_index < basis.size(); ++gamma_index) {
            const MultiIndex gamma = basis[gamma_index];
            const MultiIndex sum = add(beta, gamma);
            if (sum.degree() <= basis.order()) {
                result.entries.push_back({
                    beta_index, basis.index(sum),
                    MultiIndexSet::monomial_over_factorial(
                        displacement, gamma)});
            }
        }
    }
    return result;
}

StaticCoefficientOperator build_static_l2l_operator(
    const SphericalHarmonicBasis& basis,
    const Vec3& displacement)
{
    const MultiIndexSet cartesian(basis.order());
    const operators::detail::SphericalCartesianMaps maps =
        operators::detail::make_spherical_cartesian_maps(basis, cartesian);
    const StaticCoefficientOperator cartesian_operator =
        build_static_l2l_operator(cartesian, displacement);
    const std::vector<double> matrix =
        operators::detail::compose_spherical_translation(
            cartesian_operator, maps.local_embedding, maps.local_projection,
            maps.cartesian_count, maps.spherical_count);
    return operators::detail::pack_spherical_translation(
        basis, matrix, false);
}

} // namespace cdfmm

namespace cdfmm::operators::l2l {

StaticCoefficientOperator build(const MultiIndexSet& basis,
                                const Vec3& displacement)
{
    return build_static_l2l_operator(basis, displacement);
}

StaticCoefficientOperator build(const SphericalHarmonicBasis& basis,
                                const Vec3& displacement)
{
    return build_static_l2l_operator(basis, displacement);
}

void apply(const MultiIndexSet& basis,
           const Vec3& displacement,
           const std::span<const double> parent,
           const std::span<double> child)
{
    // Shift local coefficients from parent target box to child target box using
    // d = c_child - c_parent.
    for (int ib = 0; ib < basis.size(); ++ib) {
        const MultiIndex beta = basis[ib];

        for (int ig = 0; ig < basis.size(); ++ig) {
            const MultiIndex gamma = basis[ig];
            const MultiIndex sum = add(beta, gamma);
            if (sum.degree() > basis.order()) {
                continue;
            }

            child[ib] += MultiIndexSet::monomial_over_factorial(displacement, gamma) *
                         parent[basis.index(sum)];
        }
    }
}

} // namespace cdfmm::operators::l2l
