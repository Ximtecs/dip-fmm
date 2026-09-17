// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

namespace cdfmm::detail::cpu {

/**
 * @brief Fixed-width pack of `W` scalars evaluated lane by lane.
 *
 * The procedural point P2M/L2P executor feeds one pack of points through the
 * scalar solid-harmonic recurrence: every arithmetic operator below is an
 * elementwise loop of constant length that the vectoriser turns into one
 * SIMD instruction, so a pack behaves like a scalar in the recurrence while
 * processing `W` points at once. Only the operations the recurrence and the
 * operator kernels use are provided.
 */
template <typename Scalar, int W>
struct alignas(sizeof(Scalar) * W) Lanes {
  Scalar v[W];

  constexpr Lanes() : v{} {}
  explicit constexpr Lanes(const Scalar value) : v{} {
    for (int k = 0; k < W; ++k) {
      v[k] = value;
    }
  }

  Lanes& operator+=(const Lanes& other) {
#pragma omp simd
    for (int k = 0; k < W; ++k) {
      v[k] += other.v[k];
    }
    return *this;
  }
  Lanes& operator-=(const Lanes& other) {
#pragma omp simd
    for (int k = 0; k < W; ++k) {
      v[k] -= other.v[k];
    }
    return *this;
  }
  Lanes& operator*=(const Lanes& other) {
#pragma omp simd
    for (int k = 0; k < W; ++k) {
      v[k] *= other.v[k];
    }
    return *this;
  }
  Lanes& operator/=(const Lanes& other) {
#pragma omp simd
    for (int k = 0; k < W; ++k) {
      v[k] /= other.v[k];
    }
    return *this;
  }
  Lanes& operator*=(const Scalar other) {
#pragma omp simd
    for (int k = 0; k < W; ++k) {
      v[k] *= other;
    }
    return *this;
  }

  friend Lanes operator+(Lanes a, const Lanes& b) { return a += b; }
  friend Lanes operator-(Lanes a, const Lanes& b) { return a -= b; }
  friend Lanes operator*(Lanes a, const Lanes& b) { return a *= b; }
  friend Lanes operator/(Lanes a, const Lanes& b) { return a /= b; }
  friend Lanes operator*(Lanes a, const Scalar b) { return a *= b; }
  friend Lanes operator*(const Scalar b, Lanes a) { return a *= b; }
  friend Lanes operator-(Lanes a) {
#pragma omp simd
    for (int k = 0; k < W; ++k) {
      a.v[k] = -a.v[k];
    }
    return a;
  }

  /// Horizontal sum of the lanes.
  [[nodiscard]] Scalar sum() const {
    Scalar total = Scalar{0};
    for (int k = 0; k < W; ++k) {
      total += v[k];
    }
    return total;
  }
};

} // namespace cdfmm::detail::cpu
