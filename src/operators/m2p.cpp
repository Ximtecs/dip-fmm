// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/operators/m2p.hpp"

#include "cdfmm/laplace_derivatives.hpp"

namespace cdfmm::operators::m2p {

PotentialField evaluate(
    const MultiIndexSet& basis,
    const CoeffVector& multipole,
    const Vec3& source_centre,
    const Vec3& target,
    const OutputFlags output)
{
    PotentialField result;
    const Vec3 R = target - source_centre;
    // This direct far-field evaluation is mainly used to validate P2M/M2M
    // independently of M2L/L2L/L2P.

    // Field evaluation needs D_(alpha+e_k), so request one additional order.
    MultiIndexSet deriv_basis(basis.order() + 1);
    const auto D = laplace_derivatives_raw(deriv_basis, R);

    for (int ia = 0; ia < basis.size(); ++ia) {
        const MultiIndex alpha = basis[ia];
        const double M_alpha = multipole[ia];

        if (has_flag(output, OutputFlags::Potential)) {
            result.phi += M_alpha * D[deriv_basis.index(alpha)];
        }

        if (has_flag(output, OutputFlags::Field)) {
            result.H.x -= M_alpha * D[deriv_basis.index(add(alpha, {1, 0, 0}))];
            result.H.y -= M_alpha * D[deriv_basis.index(add(alpha, {0, 1, 0}))];
            result.H.z -= M_alpha * D[deriv_basis.index(add(alpha, {0, 0, 1}))];
        }
    }

    return result;
}

} // namespace cdfmm::operators::m2p
