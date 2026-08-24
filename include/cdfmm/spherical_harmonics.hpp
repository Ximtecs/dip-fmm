// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <vector>

#include "cdfmm/multi_index.hpp"
#include "cdfmm/vec3.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Public types
//------------------------------------------------------------------------------

/** @brief Degree and azimuthal order of one real spherical-harmonic mode. */
struct SphericalHarmonicMode {
  int l{0};
  int m{0};
};

/** @brief One Cartesian monomial in a real regular solid harmonic. */
struct SolidHarmonicTerm {
  MultiIndex power{};
  double coefficient{0.0};
};

/**
 * @brief Complete real tesseral solid-harmonic basis through order p.
 *
 * Modes are ordered by increasing degree and then by `m=-l,...,+l`, giving
 * `(p+1)^2` real coefficients. The normalisation is documented in
 * `docs/spherical-expansions.md`.
 */
class SphericalHarmonicBasis {
public:
  explicit SphericalHarmonicBasis(int p);

  [[nodiscard]] int order() const noexcept { return p_; }
  [[nodiscard]] int size() const noexcept {
    return static_cast<int>(modes_.size());
  }
  [[nodiscard]] const SphericalHarmonicMode& operator[](int index) const {
    return modes_.at(static_cast<std::size_t>(index));
  }
  [[nodiscard]] int index(int l, int m) const;
  [[nodiscard]] std::span<const SolidHarmonicTerm>
  polynomial(int index) const;

private:
  int p_{0};
  std::vector<SphericalHarmonicMode> modes_{};
  std::vector<std::vector<SolidHarmonicTerm>> polynomials_{};
};

/** @brief Values and Cartesian gradients of a solid-harmonic basis. */
struct SolidHarmonicValues {
  std::vector<double> values{};
  std::vector<Vec3> gradients{};
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
