// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

#include "cdfmm/multi_index.hpp"

namespace cdfmm {

/**
 * @brief Truncated Cartesian Taylor jet represented on a MultiIndexSet basis.
 *
 * A TaylorJet stores coefficients c_alpha of the truncated expansion
 *
 *   f(r0+h) = sum_alpha c_alpha h^alpha
 *
 * with normalisation
 *
 *   c_alpha = D_alpha f(r0)/alpha!
 *
 * The jet therefore acts as a small deterministic automatic-differentiation
 * engine in Cartesian multi-index form. In this repository it is used to
 * generate high-order derivatives of G(r)=1/(4*pi*|r|) for FMM translations,
 * avoiding hand-coded derivative tables and finite-difference noise.
 *
 * See docs/math.md for the operator formulas that consume these derivatives.
 */
class TaylorJet {
public:
  /// @brief Constructs a zero jet on the supplied multi-index basis.
  explicit TaylorJet(const MultiIndexSet &basis);

  /// @brief Returns mutable coefficient c_alpha.
  double &at(const MultiIndex &a);
  /// @brief Returns coefficient c_alpha.
  double at(const MultiIndex &a) const;

  /// @brief Returns a constant jet f(h)=c.
  static TaylorJet constant(const MultiIndexSet &basis, double c);
  /// @brief Returns coordinate jet f(h)=c+h_axis.
  static TaylorJet coordinate(const MultiIndexSet &basis, int axis, double c);

  /// @brief Adds jets coefficient-wise.
  TaylorJet add(const TaylorJet &b) const;
  /**
   * @brief Multiplies jets using the multi-index Cauchy product.
   *
   * For normalised coefficients,
   *
   *   (a*b)_alpha = sum_{gamma<=alpha} a_gamma b_{alpha-gamma}
   *
   * with no additional binomial factors in this storage convention.
   */
  TaylorJet mul(const TaylorJet &b) const;
  /**
   * @brief Computes z^(-1/2) for the current jet z.
   *
   * This is used for invsqrt(x*x+y*y+z*z), the non-constant part of the
   * Laplace Green's function.
   */
  TaylorJet invsqrt(int iters = 8) const;

  const MultiIndexSet &basis() const { return *basis_; }

private:
  const MultiIndexSet *basis_;
  std::vector<double> c_;
};

} // namespace cdfmm
