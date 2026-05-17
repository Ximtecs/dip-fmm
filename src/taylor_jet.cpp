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

    // Step 1: solve for w_alpha where w = y*y from w*z = 1.
    // Coefficient equation at alpha>0:
    //   w_alpha*z_0 + sum_{gamma<alpha} w_gamma*z_{alpha-gamma} = 0
    double sum_lower = 0.0;
    for (int j = 0; j < basis_->size(); ++j) {
      const MultiIndex gamma = (*basis_)[j];
      if (!leq(gamma, alpha)) {
        continue;
      }

      if (gamma.ax == alpha.ax && gamma.ay == alpha.ay &&
          gamma.az == alpha.az) {
        continue;
      }

      const MultiIndex delta = sub(alpha, gamma);

      double w_gamma = 0.0;
      for (int k = 0; k < basis_->size(); ++k) {
        const MultiIndex eta = (*basis_)[k];
        if (!leq(eta, gamma)) {
          continue;
        }

        const MultiIndex theta = sub(gamma, eta);
        w_gamma += y.c_[basis_->index(eta)] * y.c_[basis_->index(theta)];
      }

      sum_lower += w_gamma * c_[basis_->index(delta)];
    }
    const double w_alpha = -sum_lower / z0;

    // Step 2: recover y_alpha from w_alpha = (y*y)_alpha.
    //   w_alpha = 2*y_0*y_alpha + sum_{0<eta<alpha} y_eta*y_{alpha-eta}
    double nonlinear = 0.0;
    for (int j = 0; j < basis_->size(); ++j) {
      const MultiIndex eta = (*basis_)[j];
      if (!leq(eta, alpha)) {
        continue;
      }

      const MultiIndex theta = sub(alpha, eta);
      const bool eta_is_zero = eta.ax == 0 && eta.ay == 0 && eta.az == 0;
      const bool theta_is_zero = theta.ax == 0 && theta.ay == 0 && theta.az == 0;
      if (eta_is_zero || theta_is_zero) {
        continue;
      }

      nonlinear += y.c_[basis_->index(eta)] * y.c_[basis_->index(theta)];
    }

    y.c_[i] = (w_alpha - nonlinear) / (2.0 * y.at({0, 0, 0}));
  }

  return y;
}

} // namespace cdfmm
