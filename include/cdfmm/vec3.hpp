// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace cdfmm {

/** @brief Lightweight Cartesian three-vector used by all public operators. */
struct Vec3 {
  /// @brief Cartesian x component.
  double x{0.0};
  /// @brief Cartesian y component.
  double y{0.0};
  /// @brief Cartesian z component.
  double z{0.0};

  /// @brief Constructs the zero vector.
  Vec3() = default;
  /// @brief Constructs a vector from its Cartesian components.
  Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

  /// @brief Returns a mutable component by axis index.
  double &operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
  /// @brief Returns a component by axis index.
  double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }

  /// @brief Adds two vectors component-wise.
  Vec3 operator+(const Vec3 &b) const { return {x + b.x, y + b.y, z + b.z}; }
  /// @brief Subtracts two vectors component-wise.
  Vec3 operator-(const Vec3 &b) const { return {x - b.x, y - b.y, z - b.z}; }
  /// @brief Multiplies every component by a scalar.
  Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
  /// @brief Adds another vector in place.
  Vec3 &operator+=(const Vec3 &b) {
    x += b.x;
    y += b.y;
    z += b.z;
    return *this;
  }
};

/// @brief Multiplies every vector component by a scalar.
inline Vec3 operator*(double s, const Vec3 &v) { return v * s; }
/// @brief Computes the Cartesian dot product.
inline double dot(const Vec3 &a, const Vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

} // namespace cdfmm
