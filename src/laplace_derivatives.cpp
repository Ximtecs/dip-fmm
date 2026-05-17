// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/laplace_derivatives.hpp"

#include <numbers>

#include "cdfmm/taylor_jet.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

static constexpr double k_inv_four_pi = 1.0 / (4.0 * std::numbers::pi);

//------------------------------------------------------------------------------
// Public interface
//------------------------------------------------------------------------------

std::vector<double> laplace_derivatives_raw(const MultiIndexSet &basis,
                                            const Vec3 &r) {
  const TaylorJet x = TaylorJet::coordinate(basis, 0, r.x);
  const TaylorJet y = TaylorJet::coordinate(basis, 1, r.y);
  const TaylorJet z = TaylorJet::coordinate(basis, 2, r.z);

  const TaylorJet rho2 = x.mul(x).add(y.mul(y)).add(z.mul(z));
  const TaylorJet inv_r = rho2.invsqrt();
  const TaylorJet G = inv_r.mul(TaylorJet::constant(basis, k_inv_four_pi));

  std::vector<double> out(basis.size());
  for (int i = 0; i < basis.size(); ++i) {
    // TaylorJet stores normalised coefficients c_alpha = D_alpha G / alpha!.
    // Convert back to raw derivatives required by multipole operators.
    out[i] = G.at(basis[i]) * MultiIndexSet::multi_factorial(basis[i]);
  }
  return out;
}

} // namespace cdfmm
