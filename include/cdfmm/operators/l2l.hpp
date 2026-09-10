// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>

#include "cdfmm/operators.hpp"
#include "cdfmm/plan/static_coefficient.hpp"
#include "cdfmm/spherical_harmonics.hpp"

namespace cdfmm::operators::l2l {

/** @brief Builds the Cartesian parent-to-child local translation map. */
[[nodiscard]] StaticCoefficientOperator build(const MultiIndexSet& basis,
                                              const Vec3& displacement);

/** @brief Builds the real-spherical parent-to-child local translation map. */
[[nodiscard]] StaticCoefficientOperator build(
    const SphericalHarmonicBasis& basis,
    const Vec3& displacement);

/**
 * @brief Adds a parent local expansion shifted to a child centre.
 *
 * `displacement` is child centre minus parent centre.  The destination is
 * accumulated so that independent parent contributions can be combined.
 */
void apply(const MultiIndexSet& basis,
           const Vec3& displacement,
           std::span<const double> parent,
           std::span<double> child);

} // namespace cdfmm::operators::l2l

namespace cdfmm {

[[nodiscard]] StaticCoefficientOperator build_static_l2l_operator(
    const MultiIndexSet& basis, const Vec3& displacement);
[[nodiscard]] StaticCoefficientOperator build_static_l2l_operator(
    const SphericalHarmonicBasis& basis, const Vec3& displacement);

} // namespace cdfmm
