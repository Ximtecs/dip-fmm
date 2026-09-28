// SPDX-License-Identifier: Apache-2.0
#pragma once

// Procedural point-source P2M and point-target L2P in the spherical basis.
//
// The stored operators of `operators/p2m.cpp` and `operators/l2p.cpp` are
//
//     M_lm  = 1/(4 pi) sum_j m_j . grad R_lm(x_j - c),       (point P2M)
//     H(x)  = -sum_lm L_lm grad R_lm(x - c),   phi = sum_lm L_lm R_lm(x - c),
//
// with the real regular solid harmonics R_lm of docs/math/conventions.md.  These kernels
// evaluate the same maps during every evaluation from the positions instead
// of streaming precomputed rows.  They run the recurrence of
// `math/solid_harmonic_recurrence.hpp` in its factorial normalisation and
// leave the per-mode factor f_{l,m} (times the operator's own constant) to a
// small table applied once per leaf, so the hot loop is multiply-add only.
// The single authoritative definition of the operators remains the stored
// construction; the tables below are the only place the constants appear.
//
// `Lane` is the arithmetic type of one point (float, double or a SIMD pack of
// several points); `Coefficient` is the scalar type of the leaf expansion.

#include <cstddef>
#include <numbers>
#include <type_traits>
#include <utility>
#include <vector>

#include "math/solid_harmonic_recurrence.hpp"

namespace cdfmm::operators::point_expansion {

#ifndef CDFMM_PROCEDURAL_MAX_ORDER
#error "CDFMM_PROCEDURAL_MAX_ORDER is defined by CMakeLists.txt"
#endif

/// Largest order the procedural executors are compiled for (both targets):
/// the build option CDFMM_PROCEDURAL_MAX_ORDER, 10 by default and at most 20.
/// The automatic policy is measured only to order 10
/// (`UniformFmm::resolve_point_expansion_execution`); higher compiled orders
/// are procedural on explicit request (17 is FMM3D's order at eps = 1e-4).
inline constexpr int max_procedural_order = CDFMM_PROCEDURAL_MAX_ORDER;
static_assert(max_procedural_order >= 1 &&
                  max_procedural_order <= solid_harmonics::max_recurrence_order,
              "CDFMM_PROCEDURAL_MAX_ORDER exceeds the solid-harmonic recurrence");

/// Calls `f(std::integral_constant<int, P>{})` for the compiled order
/// `P == order` and returns true, or returns false when `order` is outside
/// 1..max_procedural_order. The one place the compiled orders are enumerated,
/// so the CPU executor and the CUDA launchers instantiate the same set.
template <typename F>
bool dispatch_procedural_order(const int order, F &&f) {
  return [&]<int... I>(std::integer_sequence<int, I...>) {
    return ((order == I + 1 ? (f(std::integral_constant<int, I + 1>{}), true)
                            : false) ||
            ...);
  }(std::make_integer_sequence<int, max_procedural_order>{});
}

/**
 * @brief Adds `m . grad Q_index(d)` to `acc[index]` for every real mode.
 *
 * `d` is the source position relative to the leaf centre. Multiplying a
 * leaf's accumulated values by `p2m_mode_factors` gives the canonical
 * point-source multipole coefficients.
 */
template <int P, typename Lane>
CDFMM_SOLID_HARMONIC_HOST_DEVICE inline void
accumulate_point_p2m(const Lane dx, const Lane dy, const Lane dz,
                     const Lane mx, const Lane my, const Lane mz, Lane* acc) {
  solid_harmonics::visit_regular_solid_harmonics<P, Lane>(
      dx, dy, dz,
      [&](const int index, const Lane, const Lane gx, const Lane gy,
          const Lane gz) { acc[index] += mx * gx + my * gy + mz * gz; });
}

/**
 * @brief Adds `sum_index scaled_L[index] grad Q_index(d)` to the field.
 *
 * With `scaled_L[index] = l2p_field_mode_factors[index] * L[index]` the
 * result is the canonical far field `H = -sum L_lm grad R_lm(d)`.
 */
template <int P, typename Lane, typename Coefficient>
CDFMM_SOLID_HARMONIC_HOST_DEVICE inline void
accumulate_point_l2p_field(const Lane dx, const Lane dy, const Lane dz,
                           const Coefficient* scaled_L, Lane& Hx, Lane& Hy,
                           Lane& Hz) {
  solid_harmonics::visit_regular_solid_harmonics<P, Lane>(
      dx, dy, dz,
      [&](const int index, const Lane, const Lane gx, const Lane gy,
          const Lane gz) {
        const Coefficient c = scaled_L[index];
        Hx += gx * c;
        Hy += gy * c;
        Hz += gz * c;
      });
}

/**
 * @brief Adds `sum_index scaled_L[index] Q_index(d)` to the potential.
 *
 * With `scaled_L[index] = l2p_potential_mode_factors[index] * L[index]` the
 * result is the canonical far-field potential `phi = sum L_lm R_lm(d)`.
 */
template <int P, typename Lane, typename Coefficient>
CDFMM_SOLID_HARMONIC_HOST_DEVICE inline void
accumulate_point_l2p_potential(const Lane dx, const Lane dy, const Lane dz,
                               const Coefficient* scaled_L, Lane& phi) {
  solid_harmonics::visit_regular_solid_harmonics<P, Lane>(
      dx, dy, dz,
      [&](const int index, const Lane value, const Lane, const Lane,
          const Lane) { phi += value * scaled_L[index]; });
}

/// f_{l,m} / (4 pi): turns accumulated `m . grad Q` sums into `M_lm`.
inline std::vector<double> p2m_mode_factors(const int p) {
  std::vector<double> factors =
      solid_harmonics::regular_solid_harmonic_mode_factors(p);
  const double green_factor = 1.0 / (4.0 * std::numbers::pi);
  for (double& factor : factors) {
    factor *= green_factor;
  }
  return factors;
}

/// -f_{l,m}: scales the leaf locals for the field, `H = -grad phi`.
inline std::vector<double> l2p_field_mode_factors(const int p) {
  std::vector<double> factors =
      solid_harmonics::regular_solid_harmonic_mode_factors(p);
  for (double& factor : factors) {
    factor = -factor;
  }
  return factors;
}

/// f_{l,m}: scales the leaf locals for the potential.
inline std::vector<double> l2p_potential_mode_factors(const int p) {
  return solid_harmonics::regular_solid_harmonic_mode_factors(p);
}

} // namespace cdfmm::operators::point_expansion
