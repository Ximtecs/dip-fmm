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
  // Build coordinate jets around the evaluation position r:
  // x = r_x + h_x, y = r_y + h_y, z = r_z + h_z.
  const TaylorJet x = TaylorJet::coordinate(basis, 0, r.x);
  const TaylorJet y = TaylorJet::coordinate(basis, 1, r.y);
  const TaylorJet z = TaylorJet::coordinate(basis, 2, r.z);

  // Compose G(h) = 1/(4*pi*sqrt(x*x + y*y + z*z)).
  const TaylorJet rho2 = x.mul(x).add(y.mul(y)).add(z.mul(z));
  const TaylorJet inv_r = rho2.invsqrt();
  const TaylorJet G = inv_r.mul(TaylorJet::constant(basis, k_inv_four_pi));

  std::vector<double> out(basis.size());
  for (int i = 0; i < basis.size(); ++i) {
    // TaylorJet coefficient: c_alpha = D_alpha G / alpha!
    // Returned value here:   D_alpha G
    // M2L and M2P consume raw derivatives in this repository.
    out[i] = G.at(basis[i]) * MultiIndexSet::multi_factorial(basis[i]);
  }
  return out;
}

} // namespace cdfmm
