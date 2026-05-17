// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "cdfmm/vec3.hpp"
namespace cdfmm {
/**
 * @brief Output selection for potential and magnetic field evaluation.
 *
 * Field-only is the default in operator APIs because H is the main quantity
 * of interest. Potential and field can be requested independently so callers
 * can skip unused work.
 */
enum class OutputFlags : unsigned {
  None = 0u,
  Potential = 1u,
  Field = 2u,
  Both = 3u
};
inline OutputFlags operator|(OutputFlags a, OutputFlags b) {
  return static_cast<OutputFlags>(static_cast<unsigned>(a) |
                                  static_cast<unsigned>(b));
}
inline bool has_flag(OutputFlags flags, OutputFlags test) {
  return (static_cast<unsigned>(flags) & static_cast<unsigned>(test)) != 0u;
}
struct PotentialField {
  /// @brief Scalar potential contribution (optional output).
  double phi{0.0};
  /// @brief Magnetic field H = -grad(phi) contribution (optional output).
  Vec3 H{};
};
} // namespace cdfmm
