// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>

#include "cdfmm/cuboid.hpp"
#include "cdfmm/operators.hpp"
#include "cdfmm/plan/static_coefficient.hpp"
#include "cdfmm/spherical_harmonics.hpp"
#include "cdfmm/tetrahedron.hpp"

namespace cdfmm::operators::p2m {

/**
 * @brief Builds the static Cartesian point-dipole P2M map.
 *
 * The returned map has three input entries per source, in x/y/z order, and
 * uses the established Cartesian coefficient order.  Geometry preparation is
 * performed once; changing dipole moments are supplied to `evaluate`.
 */
[[nodiscard]] StaticCoefficientOperator build(
    const MultiIndexSet& basis,
    const Vec3& centre,
    std::span<const Vec3> source_positions);

/** @brief Builds the static real-spherical point-dipole P2M map. */
[[nodiscard]] StaticCoefficientOperator build(
    const SphericalHarmonicBasis& basis,
    const Vec3& centre,
    std::span<const Vec3> source_positions);

/** @brief Builds a cuboid-averaged Cartesian P2M map. */
[[nodiscard]] StaticCoefficientOperator build_cuboid(
    const MultiIndexSet& basis,
    const Vec3& centre,
    std::span<const Vec3> source_positions,
    std::span<const CuboidSize> source_sizes);

/** @brief Builds a cuboid-averaged real-spherical P2M map. */
[[nodiscard]] StaticCoefficientOperator build_cuboid(
    const SphericalHarmonicBasis& basis,
    const Vec3& centre,
    std::span<const Vec3> source_positions,
    std::span<const CuboidSize> source_sizes);

/** @brief Builds an exact tetrahedron-averaged Cartesian P2M map. */
[[nodiscard]] StaticCoefficientOperator build_tetrahedron(
    const MultiIndexSet& basis,
    const Vec3& centre,
    std::span<const Vec3> source_positions,
    std::span<const Tetrahedron> source_tetrahedra);

/** @brief Builds an exact tetrahedron-averaged real-spherical P2M map. */
[[nodiscard]] StaticCoefficientOperator build_tetrahedron(
    const SphericalHarmonicBasis& basis,
    const Vec3& centre,
    std::span<const Vec3> source_positions,
    std::span<const Tetrahedron> source_tetrahedra);

/** @brief Applies the dynamic point-dipole P2M map for one moment state. */
[[nodiscard]] CoeffVector evaluate(
    const MultiIndexSet& basis,
    const Vec3& centre,
    std::span<const Vec3> source_positions,
    std::span<const Vec3> dipole_moments);

} // namespace cdfmm::operators::p2m

namespace cdfmm {

[[nodiscard]] StaticCoefficientOperator build_static_p2m_operator(
    const MultiIndexSet& basis, const Vec3& centre,
    std::span<const Vec3> source_positions);
[[nodiscard]] StaticCoefficientOperator build_static_p2m_operator(
    const SphericalHarmonicBasis& basis, const Vec3& centre,
    std::span<const Vec3> source_positions);
[[nodiscard]] StaticCoefficientOperator build_static_cuboid_p2m_operator(
    const MultiIndexSet& basis, const Vec3& centre,
    std::span<const Vec3> source_positions,
    std::span<const CuboidSize> source_sizes);
[[nodiscard]] StaticCoefficientOperator build_static_cuboid_p2m_operator(
    const SphericalHarmonicBasis& basis, const Vec3& centre,
    std::span<const Vec3> source_positions,
    std::span<const CuboidSize> source_sizes);
[[nodiscard]] StaticCoefficientOperator build_static_tetrahedron_p2m_operator(
    const MultiIndexSet& basis, const Vec3& centre,
    std::span<const Vec3> source_positions,
    std::span<const Tetrahedron> source_tetrahedra);
[[nodiscard]] StaticCoefficientOperator build_static_tetrahedron_p2m_operator(
    const SphericalHarmonicBasis& basis, const Vec3& centre,
    std::span<const Vec3> source_positions,
    std::span<const Tetrahedron> source_tetrahedra);

} // namespace cdfmm
