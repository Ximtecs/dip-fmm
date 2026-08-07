// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include "cdfmm/vec3.hpp"

namespace cdfmm {

/**
 * @brief Cartesian multi-index alpha=(alpha_x, alpha_y, alpha_z).
 *
 * Multi-indices encode derivative orders and monomial powers in Cartesian
 * Taylor expansions.
 */
struct MultiIndex {
  int ax{0};
  int ay{0};
  int az{0};

  /// @brief Returns |alpha| = alpha_x + alpha_y + alpha_z.
  int degree() const { return ax + ay + az; }
  /// @brief Returns the component for axis 0 (x), 1 (y), or 2 (z).
  int operator[](int i) const { return i == 0 ? ax : (i == 1 ? ay : az); }
};

/// @brief Tests the component-wise partial order `a_k <= b_k`.
inline bool leq(const MultiIndex &a, const MultiIndex &b) {
  // Component-wise partial order: a <= b iff a_k <= b_k for each axis k.
  return a.ax <= b.ax && a.ay <= b.ay && a.az <= b.az;
}

/// @brief Adds two Cartesian multi-indices component-wise.
inline MultiIndex add(const MultiIndex &a, const MultiIndex &b) {
  return {a.ax + b.ax, a.ay + b.ay, a.az + b.az};
}

/// @brief Subtracts two Cartesian multi-indices component-wise.
inline MultiIndex sub(const MultiIndex &a, const MultiIndex &b) {
  return {a.ax - b.ax, a.ay - b.ay, a.az - b.az};
}

/**
 * @brief Dense total-degree multi-index basis up to order p.
 *
 * Storage order is grouped by total degree and then lexicographically in
 * (alpha_x, alpha_y) at fixed degree, with alpha_z implied by degree balance.
 *
 * Total-degree ordering is convenient for truncated FMM expansions because
 * translations couple nearby degrees and recursion dependencies are naturally
 * satisfied from lower to higher degree.
 * The basis contains `(p+1)(p+2)(p+3)/6` entries in three dimensions.
 */
class MultiIndexSet {
public:
  /** @brief Constructs the dense basis containing all degrees through @p p. */
  explicit MultiIndexSet(int p) : p_(p) {
    for (int d = 0; d <= p_; ++d) {
      for (int ax = 0; ax <= d; ++ax) {
        for (int ay = 0; ay <= d - ax; ++ay) {
          const int az = d - ax - ay;
          indices_.push_back({ax, ay, az});
        }
      }
    }
  }

  /// @brief Returns the maximum total degree p.
  int order() const { return p_; }
  /// @brief Returns the number of basis entries.
  int size() const { return static_cast<int>(indices_.size()); }
  /// @brief Returns the multi-index at a linear storage position.
  const MultiIndex &operator[](int i) const { return indices_.at(i); }

  /// @brief Maps a multi-index to its linear storage position.
  int index(const MultiIndex &a) const {
    for (int i = 0; i < size(); ++i) {
      if (indices_[i].ax == a.ax && indices_[i].ay == a.ay &&
          indices_[i].az == a.az) {
        return i;
      }
    }

    throw std::out_of_range("multi-index not found");
  }

  /// @brief Returns `n!` as a floating-point value.
  static double factorial(int n) {
    double v = 1.0;
    for (int i = 2; i <= n; ++i) {
      v *= i;
    }
    return v;
  }

  /// @brief Returns `alpha! = alpha_x! alpha_y! alpha_z!`.
  static double multi_factorial(const MultiIndex &a) {
    return factorial(a.ax) * factorial(a.ay) * factorial(a.az);
  }

  /**
   * @brief Returns r^alpha / alpha! for Cartesian Taylor expansions.
   *
   * This normalised monomial appears throughout multipole/local translations
   * because coefficients are stored in the same alpha!-normalised convention.
   */
  static double monomial_over_factorial(const Vec3 &r, const MultiIndex &a) {
    return std::pow(r.x, a.ax) * std::pow(r.y, a.ay) * std::pow(r.z, a.az) /
           multi_factorial(a);
  }

  /// @brief Returns r^alpha/alpha! for all basis entries in basis order.
  std::vector<double> all_monomials_over_factorial(const Vec3 &r) const {
    std::vector<double> out(size());
    for (int i = 0; i < size(); ++i) {
      out[i] = monomial_over_factorial(r, indices_[i]);
    }
    return out;
  }

private:
  int p_;
  std::vector<MultiIndex> indices_;
};

} // namespace cdfmm
