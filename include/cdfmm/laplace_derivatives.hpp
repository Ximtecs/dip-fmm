// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

#include "cdfmm/multi_index.hpp"

namespace cdfmm {

/**
 * @brief Computes raw Cartesian derivatives D_alpha G(r) for Laplace Green's function.
 *
 * The implementation uses truncated Cartesian Taylor algebra (TaylorJet)
 * rather than finite differences to generate deterministic high-order
 * derivatives needed by M2L and M2P field evaluation.
 *
 * Returned entries are unnormalised partial derivatives:
 *
 *   out[alpha] = D_alpha G(r),   G(r)=1/(4*pi*|r|)
 *
 * not the normalised Taylor coefficients D_alpha G(r)/alpha!.
 */
std::vector<double> laplace_derivatives_raw(const MultiIndexSet &basis,
                                            const Vec3 &r);

} // namespace cdfmm
