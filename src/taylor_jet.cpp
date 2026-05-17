// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/taylor_jet.hpp"

#include <cmath>

#include <stdexcept>

namespace cdfmm {

TaylorJet::TaylorJet(const MultiIndexSet &basis)
    : basis_(&basis), c_(basis.size(), 0.0) {}

double &TaylorJet::at(const MultiIndex &a) { return c_[basis_->index(a)]; }

double TaylorJet::at(const MultiIndex &a) const { return c_[basis_->index(a)]; }

TaylorJet TaylorJet::constant(const MultiIndexSet &basis, double c) {
  TaylorJet t(basis);
  t.at({0, 0, 0}) = c;
  return t;
}

TaylorJet TaylorJet::coordinate(const MultiIndexSet &basis, int axis,
                                double c) {
  TaylorJet t = constant(basis, c);
  if (axis == 0) {
    t.at({1, 0, 0}) = 1.0;
  } else if (axis == 1) {
    t.at({0, 1, 0}) = 1.0;
  } else {
    t.at({0, 0, 1}) = 1.0;
  }
  return t;
}

TaylorJet TaylorJet::add(const TaylorJet &b) const {
  TaylorJet o(*basis_);
  for (int i = 0; i < basis_->size(); ++i) {
    o.c_[i] = c_[i] + b.c_[i];
  }
  return o;
}

TaylorJet TaylorJet::mul(const TaylorJet &b) const {
  TaylorJet o(*basis_);
  for (int i = 0; i < basis_->size(); ++i) {
    const MultiIndex alpha = (*basis_)[i];
    double sum = 0.0;
    for (int j = 0; j < basis_->size(); ++j) {
      const MultiIndex gamma = (*basis_)[j];
      if (!leq(gamma, alpha)) {
        continue;
      }

      const MultiIndex eta = sub(alpha, gamma);
      sum += c_[j] * b.c_[basis_->index(eta)];
    }
    o.c_[i] = sum;
  }
  return o;
}

TaylorJet TaylorJet::invsqrt(int) const {
  TaylorJet y(*basis_);

  const double z0 = at({0, 0, 0});
  if (z0 <= 0.0) {
    throw std::invalid_argument(
        "TaylorJet::invsqrt requires positive constant coefficient");
  }

  // Solve y^2 * z = 1 coefficient-by-coefficient in increasing total degree.
  // The Taylor basis is stored by total degree, so lower-order coefficients
  // required by the recursion are always available when needed.
  y.at({0, 0, 0}) = 1.0 / std::sqrt(z0);

  for (int i = 1; i < basis_->size(); ++i) {
    const MultiIndex alpha = (*basis_)[i];

    // Compute the coefficient of y*y*z at alpha excluding the two linear
    // terms 2*y_0*z_0*y_alpha. The remaining part is denoted known_terms.
    double known_terms = 0.0;
    for (int j = 0; j < basis_->size(); ++j) {
      const MultiIndex gamma = (*basis_)[j];
      if (!leq(gamma, alpha)) {
        continue;
      }

      const MultiIndex rem_after_gamma = sub(alpha, gamma);
      for (int k = 0; k < basis_->size(); ++k) {
        const MultiIndex eta = (*basis_)[k];
        if (!leq(eta, rem_after_gamma)) {
          continue;
        }

        const MultiIndex delta = sub(rem_after_gamma, eta);

        const bool gamma_is_alpha = gamma.ax == alpha.ax &&
                                    gamma.ay == alpha.ay &&
                                    gamma.az == alpha.az;
        const bool eta_is_zero = eta.ax == 0 && eta.ay == 0 && eta.az == 0;
        const bool eta_is_alpha = eta.ax == alpha.ax && eta.ay == alpha.ay &&
                                  eta.az == alpha.az;
        const bool gamma_is_zero = gamma.ax == 0 && gamma.ay == 0 &&
                                   gamma.az == 0;
        const bool delta_is_zero = delta.ax == 0 && delta.ay == 0 &&
                                   delta.az == 0;

        // Exclude the two terms containing y_alpha linearly:
        // (gamma=alpha, eta=0, delta=0) and (gamma=0, eta=alpha, delta=0).
        if ((gamma_is_alpha && eta_is_zero && delta_is_zero) ||
            (gamma_is_zero && eta_is_alpha && delta_is_zero)) {
          continue;
        }

        known_terms += c_[j] * y.c_[basis_->index(eta)] *
                       y.c_[basis_->index(delta)];
      }
    }

    const double denom = 2.0 * y.at({0, 0, 0}) * z0;
    y.c_[i] = -known_terms / denom;
  }

  return y;
}

} // namespace cdfmm
