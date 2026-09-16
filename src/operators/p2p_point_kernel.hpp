// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <numbers>

namespace cdfmm::operators::p2p {

/**
 * @brief Point-dipole near-field term of one source at one target.
 *
 * Adds `H = 1/(4 pi) * [3 r (m . r) / |r|^5 - m / |r|^3]` for `r = x_t - x_s`
 * to the accumulators.  This is the single authoritative point formula:
 * `evaluate_pair` and the CPU position-based P2P executor both call it, so
 * they cannot drift apart.  The caller excludes the singular self pair.
 */
template <typename Scalar>
inline void accumulate_point_dipole_field(const Scalar rx, const Scalar ry,
                                          const Scalar rz, const Scalar mx,
                                          const Scalar my, const Scalar mz,
                                          Scalar& Hx, Scalar& Hy, Scalar& Hz) {
  constexpr Scalar c = static_cast<Scalar>(1.0 / (4.0 * std::numbers::pi));
  const Scalar r2 = rx * rx + ry * ry + rz * rz;
  const Scalar rinv = Scalar{1} / std::sqrt(r2);
  const Scalar rinv3 = rinv * rinv * rinv;
  const Scalar m_dot_r = mx * rx + my * ry + mz * rz;
  const Scalar rinv5 = rinv3 * rinv * rinv;
  const Scalar radial = Scalar{3} * m_dot_r * rinv5;
  Hx += (rx * radial - mx * rinv3) * c;
  Hy += (ry * radial - my * rinv3) * c;
  Hz += (rz * radial - mz * rinv3) * c;
}

/** @brief Point-dipole potential `1/(4 pi) * (m . r) / |r|^3` of one source. */
template <typename Scalar>
inline Scalar point_dipole_potential(const Scalar rx, const Scalar ry,
                                     const Scalar rz, const Scalar mx,
                                     const Scalar my, const Scalar mz) {
  constexpr Scalar c = static_cast<Scalar>(1.0 / (4.0 * std::numbers::pi));
  const Scalar r2 = rx * rx + ry * ry + rz * rz;
  const Scalar rinv = Scalar{1} / std::sqrt(r2);
  const Scalar rinv3 = rinv * rinv * rinv;
  return c * (mx * rx + my * ry + mz * rz) * rinv3;
}

} // namespace cdfmm::operators::p2p
