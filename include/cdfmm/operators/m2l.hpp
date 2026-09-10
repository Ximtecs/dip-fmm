// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <vector>

#include "cdfmm/operators.hpp"
#include "cdfmm/periodic.hpp"
#include "cdfmm/spherical_harmonics.hpp"

namespace cdfmm::operators::m2l {

/** @brief Builds a column-major Cartesian M2L matrix for one displacement. */
[[nodiscard]] std::vector<double> build_matrix(const MultiIndexSet& basis,
                                               const Vec3& displacement);

/** @brief Builds a column-major real-spherical M2L matrix. */
[[nodiscard]] std::vector<double> build_matrix(
    const SphericalHarmonicBasis& basis,
    const Vec3& displacement);

/** @brief Builds a normalised Cartesian zero-k0 periodic root matrix. */
[[nodiscard]] std::vector<double> build_periodic_matrix(
    const MultiIndexSet& basis,
    const PeriodicCellOptions& options);

/** @brief Builds a normalised real-spherical zero-k0 periodic root matrix. */
[[nodiscard]] std::vector<double> build_periodic_matrix(
    const SphericalHarmonicBasis& basis,
    const PeriodicCellOptions& options);

/** @brief Adds a dynamic Cartesian M2L translation to a local expansion. */
void apply(const MultiIndexSet& basis,
           const Vec3& displacement,
           std::span<const double> multipole,
           std::span<double> local);

} // namespace cdfmm::operators::m2l

namespace cdfmm {

[[nodiscard]] std::vector<double> build_static_m2l_matrix(
    const MultiIndexSet& basis, const Vec3& displacement);
[[nodiscard]] std::vector<double> build_static_m2l_matrix(
    const SphericalHarmonicBasis& basis, const Vec3& displacement);
[[nodiscard]] std::vector<double> build_static_periodic_m2l_matrix(
    const MultiIndexSet& basis, const PeriodicCellOptions& options);
[[nodiscard]] std::vector<double> build_static_periodic_m2l_matrix(
    const SphericalHarmonicBasis& basis, const PeriodicCellOptions& options);

} // namespace cdfmm
