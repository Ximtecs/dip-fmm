// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/laplace_derivatives.hpp"

#include <cmath>
#include <numbers>

namespace cdfmm {

static double G(const Vec3 &r) {
  return 1.0 / (4.0 * std::numbers::pi * std::sqrt(dot(r, r)));
}

static double deriv(const Vec3 &r, const MultiIndex &a, double h = 1e-5) {
  if (a.ax + a.ay + a.az == 0) {
    return G(r);
  }

  if (a.ax > 0) {
    auto b = a;
    b.ax--;
    Vec3 rp = r;
    rp.x += h;
    Vec3 rm = r;
    rm.x -= h;
    return (deriv(rp, b, h) - deriv(rm, b, h)) / (2.0 * h);
  }

  if (a.ay > 0) {
    auto b = a;
    b.ay--;
    Vec3 rp = r;
    rp.y += h;
    Vec3 rm = r;
    rm.y -= h;
    return (deriv(rp, b, h) - deriv(rm, b, h)) / (2.0 * h);
  }

  auto b = a;
  b.az--;
  Vec3 rp = r;
  rp.z += h;
  Vec3 rm = r;
  rm.z -= h;
  return (deriv(rp, b, h) - deriv(rm, b, h)) / (2.0 * h);
}

std::vector<double> laplace_derivatives_raw(const MultiIndexSet &basis,
                                            const Vec3 &r) {
  std::vector<double> out(basis.size());
  for (int i = 0; i < basis.size(); ++i) {
    out[i] = deriv(r, basis[i]);
  }
  return out;
}

} // namespace cdfmm
