// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace cdfmm {

/**
 * @brief Six independent components of a symmetric Cartesian pair tensor.
 *
 * `T` maps a source's total dipole moment to the field at a target,
 * `H = T m`, with `T_xy = T_yx` and so on.
 */
struct PairTensor {
  double xx{0.0};   ///< T_xx.
  double xy{0.0};   ///< T_xy = T_yx.
  double xz{0.0};   ///< T_xz = T_zx.
  double yy{0.0};   ///< T_yy.
  double yz{0.0};   ///< T_yz = T_zy.
  double zz{0.0};   ///< T_zz.
};

} // namespace cdfmm
