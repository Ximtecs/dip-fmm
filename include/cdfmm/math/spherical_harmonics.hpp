// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <vector>

#include "cdfmm/math/multi_index.hpp"
#include "cdfmm/math/vec3.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Public types
//------------------------------------------------------------------------------

/** @brief Degree and azimuthal order of one real spherical-harmonic mode. */
struct SphericalHarmonicMode {
  int l{0};   ///< Degree, 0..p.
  int m{0};   ///< Real azimuthal order, -l..l (negative for the sine-type modes).
};

/** @brief One Cartesian monomial in a real regular solid harmonic. */
struct SolidHarmonicTerm {
  MultiIndex power{};       ///< Exponents of x, y, z.
  double coefficient{0.0};  ///< Coefficient of `x^ax y^ay z^az` in R_lm.
};

/**
 * @brief Complete real tesseral solid-harmonic basis through order p.
 *
 * Modes are ordered by increasing degree and then by `m=-l,...,+l`, giving
 * `(p+1)^2` real coefficients. The normalisation is documented in
 * `docs/math/spherical-expansions.md`.
 */
class SphericalHarmonicBasis {
public:
  /// @brief Builds the modes and monomial expansions of every R_lm, l <= p.
  explicit SphericalHarmonicBasis(int p);

  /// @brief The maximum degree p.
  [[nodiscard]] int order() const noexcept { return p_; }
  /// @brief The number of modes, `(p+1)^2`.
  [[nodiscard]] int size() const noexcept {
    return static_cast<int>(modes_.size());
  }
  /// @brief The (l, m) of the mode at a flat index; throws when out of range.
  [[nodiscard]] const SphericalHarmonicMode& operator[](int index) const {
    return modes_.at(static_cast<std::size_t>(index));
  }
  /// @brief The flat index of mode (l, m), `l^2 + l + m`.
  [[nodiscard]] int index(int l, int m) const;
  /// @brief The Cartesian monomial expansion of the regular solid harmonic at a flat index.
  [[nodiscard]] std::span<const SolidHarmonicTerm>
  polynomial(int index) const;

private:
  int p_{0};
  std::vector<SphericalHarmonicMode> modes_{};
  std::vector<std::vector<SolidHarmonicTerm>> polynomials_{};
};

/** @brief Values and Cartesian gradients of a solid-harmonic basis. */
struct SolidHarmonicValues {
  std::vector<double> values{};   ///< One value per mode in basis order.
  std::vector<Vec3> gradients{};  ///< One Cartesian gradient per mode in basis order.
};

//------------------------------------------------------------------------------
// Public functions
//------------------------------------------------------------------------------

/** @brief Evaluates all real regular solid harmonics and their gradients. */
[[nodiscard]] SolidHarmonicValues regular_solid_harmonics(
    const SphericalHarmonicBasis& basis, const Vec3& r);

/** @brief Evaluates all real irregular solid harmonics and their gradients. */
[[nodiscard]] SolidHarmonicValues irregular_solid_harmonics(
    const SphericalHarmonicBasis& basis, const Vec3& r);

} // namespace cdfmm
