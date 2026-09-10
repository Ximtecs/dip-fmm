// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace cdfmm {

/** @brief Six independent components of a symmetric Cartesian pair tensor. */
struct PairTensor {
  double xx{0.0};
  double xy{0.0};
  double xz{0.0};
  double yy{0.0};
  double yz{0.0};
  double zz{0.0};
};

} // namespace cdfmm
