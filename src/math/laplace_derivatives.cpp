// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/math/laplace_derivatives.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

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
  // Taylor coefficients c_k = D^k g(r) / k! of g(r + h) = 1 / |r + h|. From
  // |r + h|^2 grad g = -(r + h) g, contracted with h and matched term by term
  // in h^k, they obey the exact recurrence
  //
  //   |r|^2 |k| c_k = -(2|k| - 1) sum_i r_i c_{k - e_i} - (|k| - 1) sum_i c_{k - 2 e_i},
  //
  // with c_0 = 1 / |r| and terms of a negative index absent. Each coefficient
  // needs only lower degrees, and the basis is stored by increasing degree, so
  // one pass costs O(1) per coefficient. It replaces composing Taylor jets
  // (products and an inverse square root on the whole order-2p basis), whose
  // cost dominated the construction of the universal M2L bank; the values
  // agree to rounding (tests/test_laplace_derivatives.cpp).
  const double r2 = r.x * r.x + r.y * r.y + r.z * r.z;
  const double components[3] = {r.x, r.y, r.z};
  std::vector<double> c(basis.size());
  c[0] = 1.0 / std::sqrt(r2);
  for (int i = 1; i < basis.size(); ++i) {
    const MultiIndex k = basis[i];
    const int powers[3] = {k.ax, k.ay, k.az};
    const int n = k.ax + k.ay + k.az;
    double first = 0.0;
    double second = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      if (powers[axis] >= 1) {
        MultiIndex down = k;
        (axis == 0 ? down.ax : axis == 1 ? down.ay : down.az) -= 1;
        first += components[axis] * c[static_cast<std::size_t>(basis.index(down))];
      }
      if (powers[axis] >= 2) {
        MultiIndex down = k;
        (axis == 0 ? down.ax : axis == 1 ? down.ay : down.az) -= 2;
        second += c[static_cast<std::size_t>(basis.index(down))];
      }
    }
    c[static_cast<std::size_t>(i)] =
        -((2.0 * n - 1.0) * first + (n - 1.0) * second) / (r2 * n);
  }

  std::vector<double> out(basis.size());
  for (int i = 0; i < basis.size(); ++i) {
    // Coefficient c_k = D^k g / k!; returned value here: D^k G, G = g / (4 pi).
    // M2L and M2P consume raw derivatives in this repository.
    out[static_cast<std::size_t>(i)] = c[static_cast<std::size_t>(i)] *
        MultiIndexSet::multi_factorial(basis[i]) * k_inv_four_pi;
  }
  return out;
}

} // namespace cdfmm
