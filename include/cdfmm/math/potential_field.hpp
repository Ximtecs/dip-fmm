// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cdfmm/math/vec3.hpp"

namespace cdfmm {

/** @brief Optional scalar-potential and magnetic-field result. */
struct PotentialField {
  /// @brief Scalar potential contribution (optional output).
  double phi{0.0};
  /// @brief Magnetic field H = -grad(phi) contribution (optional output).
  Vec3 H{};
};

/** @brief Single-precision potential and field result for FP32 FMM. */
struct FloatPotentialField {
  float phi{0.0F};
  FloatVec3 H{};
};

} // namespace cdfmm
