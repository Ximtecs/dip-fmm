// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>

#include "cdfmm/plan/static_plan.hpp"
#include "cdfmm/math/multi_index.hpp"
#include "cdfmm/math/spherical_harmonics.hpp"

namespace cdfmm::detail::cpu {

/** @brief Non-owning degree lookup shared by Cartesian and spherical plans. */
struct CoefficientDegreeView {
  const MultiIndexSet *cartesian{nullptr};
  const SphericalHarmonicBasis *spherical{nullptr};

  [[nodiscard]] int operator()(const int coefficient) const noexcept {
    return spherical != nullptr ? (*spherical)[coefficient].l
                                : (*cartesian)[coefficient].degree();
  }
};

CoefficientDegreeView coefficient_degree_view(
    bool spherical, const MultiIndexSet &cartesian,
    const SphericalHarmonicBasis &spherical_basis) noexcept;

void apply_static_p2m(const StaticCoefficientOperator& operator_map,
                      std::span<const Vec3> moments,
                      std::span<double> multipole);
void apply_static_p2m(const FloatStaticCoefficientOperator& operator_map,
                      std::span<const FloatVec3> moments,
                      std::span<float> multipole);

void apply_static_translation(const StaticCoefficientOperator& operator_map,
                              std::span<double> input,
                              std::span<double> output, int level,
                              CoefficientDegreeView degree);
void apply_static_translation(
    const FloatStaticCoefficientOperator& operator_map,
    std::span<float> input, std::span<float> output, int level,
    CoefficientDegreeView degree);

} // namespace cdfmm::detail::cpu
