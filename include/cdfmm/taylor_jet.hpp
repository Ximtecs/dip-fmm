// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

#include "cdfmm/multi_index.hpp"

namespace cdfmm {

/**
 * @brief Truncated Cartesian Taylor jet represented on a MultiIndexSet basis.
 *
 * Coefficients are stored as normalised coefficients (term divided by alpha!).
 * This differs from raw derivatives D_alpha f, which require an additional
 * division by alpha! to become Taylor coefficients.
 */
class TaylorJet {
public:
  explicit TaylorJet(const MultiIndexSet &basis);

  double &at(const MultiIndex &a);
  double at(const MultiIndex &a) const;

  static TaylorJet constant(const MultiIndexSet &basis, double c);
  static TaylorJet coordinate(const MultiIndexSet &basis, int axis, double c);

  TaylorJet add(const TaylorJet &b) const;
  TaylorJet mul(const TaylorJet &b) const;
  TaylorJet invsqrt(int iters = 8) const;

  const MultiIndexSet &basis() const { return *basis_; }

private:
  const MultiIndexSet *basis_;
  std::vector<double> c_;
};

} // namespace cdfmm
