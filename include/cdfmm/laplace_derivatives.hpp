// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

#include "cdfmm/multi_index.hpp"

namespace cdfmm {

/**
 * @brief Computes raw Cartesian derivatives D_alpha G(r) for Laplace Green's function.
 *
 * This returns unnormalised derivatives, i.e. direct partial derivatives of
 * G(r)=1/(4*pi*|r|), not Taylor coefficients divided by alpha!.
 */
std::vector<double> laplace_derivatives_raw(const MultiIndexSet &basis,
                                            const Vec3 &r);

} // namespace cdfmm
