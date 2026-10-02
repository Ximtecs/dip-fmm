// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/operators/m2p.hpp"

#include <cmath>

#include "cdfmm/math/laplace_derivatives.hpp"

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

    // NOTE(cdfmm): explicit fma, so each output is rounded the same way
    // whichever outputs are requested. Left to -ffp-contract, GCC 13 contracted
    // the loop of its flag-specialised clones differently and phi under
    // OutputFlags::Both differed from OutputFlags::Potential in the last bit.
    for (int ia = 0; ia < basis.size(); ++ia) {
        const MultiIndex alpha = basis[ia];
        const double M_alpha = multipole[ia];

        if (has_flag(output, OutputFlags::Potential)) {
            result.phi = std::fma(M_alpha, D[deriv_basis.index(alpha)], result.phi);
        }

        if (has_flag(output, OutputFlags::Field)) {
            result.H.x = std::fma(
                -M_alpha, D[deriv_basis.index(add(alpha, {1, 0, 0}))], result.H.x);
            result.H.y = std::fma(
                -M_alpha, D[deriv_basis.index(add(alpha, {0, 1, 0}))], result.H.y);
            result.H.z = std::fma(
                -M_alpha, D[deriv_basis.index(add(alpha, {0, 0, 1}))], result.H.z);
        }
    }

    return result;
}

} // namespace cdfmm::operators::m2p
