// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/taylor_jet.hpp"

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
    const MultiIndex a = (*basis_)[i];
    double s = 0.0;
    for (int j = 0; j < basis_->size(); ++j) {
      const MultiIndex g = (*basis_)[j];
      if (!leq(g, a)) {
        continue;
      }
      const MultiIndex h = sub(a, g);
      s += c_[j] * b.c_[basis_->index(h)];
    }
    o.c_[i] = s;
  }
  return o;
}

TaylorJet TaylorJet::invsqrt(int) const {
  TaylorJet o(*basis_);
  o.c_ = c_;
  return o;
}

} // namespace cdfmm
