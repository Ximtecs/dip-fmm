// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>

#include "cdfmm/core/output_flags.hpp"
#include "cdfmm/math/coefficients.hpp"
#include "cdfmm/math/multi_index.hpp"
#include "cdfmm/math/potential_field.hpp"
#include "cdfmm/plan/static_coefficient.hpp"
#include "cdfmm/math/spherical_harmonics.hpp"

namespace cdfmm::operators::m2m {

/** @brief Builds the Cartesian child-to-parent translation map. */
[[nodiscard]] StaticCoefficientOperator build(const MultiIndexSet& basis,
                                              const Vec3& displacement);

/** @brief Builds the real-spherical child-to-parent translation map. */
[[nodiscard]] StaticCoefficientOperator build(
    const SphericalHarmonicBasis& basis,
    const Vec3& displacement);

/**
 * @brief Adds a child expansion translated to a parent centre.
 *
 * `displacement` is parent centre minus child centre.  The destination is
 * accumulated, which permits several children to share one parent.
 */
void apply(const MultiIndexSet& basis,
           const Vec3& displacement,
           std::span<const double> child,
           std::span<double> parent);

} // namespace cdfmm::operators::m2m

namespace cdfmm {

[[nodiscard]] StaticCoefficientOperator build_static_m2m_operator(
    const MultiIndexSet& basis, const Vec3& displacement);
[[nodiscard]] StaticCoefficientOperator build_static_m2m_operator(
    const SphericalHarmonicBasis& basis, const Vec3& displacement);

} // namespace cdfmm
