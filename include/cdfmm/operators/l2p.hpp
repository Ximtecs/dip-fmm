// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>

#include "cdfmm/cuboid.hpp"
#include "cdfmm/operators.hpp"
#include "cdfmm/plan/l2p.hpp"
#include "cdfmm/spherical_harmonics.hpp"
#include "cdfmm/tetrahedron.hpp"

namespace cdfmm::operators::l2p {

/** @brief Builds Cartesian point-target local evaluation rows. */
[[nodiscard]] StaticL2PEvaluator build(const MultiIndexSet& basis,
                                       const Vec3& centre,
                                       const Vec3& target);

/** @brief Builds real-spherical point-target local evaluation rows. */
[[nodiscard]] StaticL2PEvaluator build(const SphericalHarmonicBasis& basis,
                                       const Vec3& centre,
                                       const Vec3& target);

/** @brief Builds Cartesian cuboid-volume target evaluation rows. */
[[nodiscard]] StaticL2PEvaluator build_cuboid(
    const MultiIndexSet& basis,
    const Vec3& centre,
    const Vec3& target,
    const CuboidSize& target_size);

/** @brief Builds real-spherical cuboid-volume target evaluation rows. */
[[nodiscard]] StaticL2PEvaluator build_cuboid(
    const SphericalHarmonicBasis& basis,
    const Vec3& centre,
    const Vec3& target,
    const CuboidSize& target_size);

/** @brief Builds Cartesian tetrahedron-volume target evaluation rows. */
[[nodiscard]] StaticL2PEvaluator build_tetrahedron(
    const MultiIndexSet& basis,
    const Vec3& centre,
    const Vec3& target,
    const Tetrahedron& target_tetrahedron);

/** @brief Builds real-spherical tetrahedron-volume target evaluation rows. */
[[nodiscard]] StaticL2PEvaluator build_tetrahedron(
    const SphericalHarmonicBasis& basis,
    const Vec3& centre,
    const Vec3& target,
    const Tetrahedron& target_tetrahedron);

/** @brief Evaluates a local expansion directly at one point target. */
[[nodiscard]] PotentialField evaluate(const MultiIndexSet& basis,
                                      const Vec3& centre,
                                      const Vec3& target,
                                      std::span<const double> local,
                                      OutputFlags output = OutputFlags::Field);

} // namespace cdfmm::operators::l2p

namespace cdfmm {

[[nodiscard]] StaticL2PEvaluator build_static_l2p_evaluator(
    const MultiIndexSet& basis, const Vec3& centre, const Vec3& target);
[[nodiscard]] StaticL2PEvaluator build_static_l2p_evaluator(
    const SphericalHarmonicBasis& basis, const Vec3& centre,
    const Vec3& target);
[[nodiscard]] StaticL2PEvaluator build_static_cuboid_l2p_evaluator(
    const MultiIndexSet& basis, const Vec3& centre, const Vec3& target,
    const CuboidSize& target_size);
[[nodiscard]] StaticL2PEvaluator build_static_cuboid_l2p_evaluator(
    const SphericalHarmonicBasis& basis, const Vec3& centre,
    const Vec3& target, const CuboidSize& target_size);
[[nodiscard]] StaticL2PEvaluator build_static_tetrahedron_l2p_evaluator(
    const MultiIndexSet& basis, const Vec3& centre, const Vec3& target,
    const Tetrahedron& target_tetrahedron);
[[nodiscard]] StaticL2PEvaluator build_static_tetrahedron_l2p_evaluator(
    const SphericalHarmonicBasis& basis, const Vec3& centre,
    const Vec3& target, const Tetrahedron& target_tetrahedron);

} // namespace cdfmm
