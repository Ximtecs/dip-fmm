// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <numbers>

// The pair formula is shared by host code and the CUDA position-based P2P
// kernel, so it is compiled for both targets when a CUDA compiler sees it.
#if defined(__CUDACC__)
#define CDFMM_P2P_KERNEL_HOST_DEVICE __host__ __device__
#else
#define CDFMM_P2P_KERNEL_HOST_DEVICE
#endif

namespace cdfmm::operators::p2p {

/**
 * @brief Point-dipole near-field term of one source at one target for a
 *        given inverse separation.
 *
 * Adds `H = 1/(4 pi) * [3 r (m . r) / |r|^5 - m / |r|^3]` for `r = x_t - x_s`
 * to the accumulators, with `rinv = 1 / |r|` supplied by the caller.  The
 * split lets an executor choose how it forms the inverse square root (an
 * IEEE division on the CPU, the hardware reciprocal square root on a GPU)
 * without restating the formula; every executor still applies this one
 * expression.  The caller excludes the singular self pair.
 */
template <typename Scalar>
CDFMM_P2P_KERNEL_HOST_DEVICE inline void accumulate_point_dipole_field(
    const Scalar rx, const Scalar ry, const Scalar rz, const Scalar rinv,
    const Scalar mx, const Scalar my, const Scalar mz, Scalar& Hx,
    Scalar& Hy, Scalar& Hz) {
  constexpr Scalar c = static_cast<Scalar>(1.0 / (4.0 * std::numbers::pi));
  const Scalar rinv3 = rinv * rinv * rinv;
  const Scalar m_dot_r = mx * rx + my * ry + mz * rz;
  const Scalar rinv5 = rinv3 * rinv * rinv;
  const Scalar radial = Scalar{3} * m_dot_r * rinv5;
  Hx += (rx * radial - mx * rinv3) * c;
  Hy += (ry * radial - my * rinv3) * c;
  Hz += (rz * radial - mz * rinv3) * c;
}

/**
 * @brief Point-dipole near-field term of one source at one target.
 *
 * This is the single authoritative point formula: `evaluate_pair`, the CPU
 * position-based P2P executor and the CUDA position-based kernel all call it
 * (directly or through the inverse-separation overload above), so they
 * cannot drift apart.  The caller excludes the singular self pair.
 */
template <typename Scalar>
CDFMM_P2P_KERNEL_HOST_DEVICE inline void accumulate_point_dipole_field(
    const Scalar rx, const Scalar ry, const Scalar rz, const Scalar mx,
    const Scalar my, const Scalar mz, Scalar& Hx, Scalar& Hy, Scalar& Hz) {
  const Scalar r2 = rx * rx + ry * ry + rz * rz;
  const Scalar rinv = Scalar{1} / std::sqrt(r2);
  accumulate_point_dipole_field(rx, ry, rz, rinv, mx, my, mz, Hx, Hy, Hz);
}

/** @brief Point-dipole potential `1/(4 pi) * (m . r) / |r|^3` of one source. */
template <typename Scalar>
CDFMM_P2P_KERNEL_HOST_DEVICE inline Scalar point_dipole_potential(
    const Scalar rx, const Scalar ry, const Scalar rz, const Scalar mx,
    const Scalar my, const Scalar mz) {
  constexpr Scalar c = static_cast<Scalar>(1.0 / (4.0 * std::numbers::pi));
  const Scalar r2 = rx * rx + ry * ry + rz * rz;
  const Scalar rinv = Scalar{1} / std::sqrt(r2);
  const Scalar rinv3 = rinv * rinv * rinv;
  return c * (mx * rx + my * ry + mz * rz) * rinv3;
}

} // namespace cdfmm::operators::p2p
