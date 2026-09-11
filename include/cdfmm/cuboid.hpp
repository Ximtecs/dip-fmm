// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cdfmm/plan/direct/dense.hpp"
#include "cdfmm/math/coefficients.hpp"
#include "cdfmm/geometry.hpp"
#include "cdfmm/multi_index.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Geometry and pair tensors
//------------------------------------------------------------------------------

/**
 * @brief Evaluates the factorial-normalised monomial averaged over a prism.
 *
 * This is J_beta(d,h) = V^-1 integral_V (d+u)^beta/beta! dV and is evaluated
 * by its finite even-power sum, without numerical quadrature.
 */
[[nodiscard]] double cuboid_averaged_monomial(
    const MultiIndex& beta,
    const Vec3& d,
    const CuboidSize& h
);

/** @brief Generic rectangular-prism spelling of cuboid_averaged_monomial. */
[[nodiscard]] double rectangular_prism_averaged_monomial(
    const MultiIndex& beta,
    const Vec3& d,
    const RectangularPrism& prism
);

/**
 * @brief Constructs the exact moment-to-field tensor for one geometry pair.
 *
 * Runtime inputs are total moments m=V*M. Cuboid source normalisation is
 * consequently included in the returned tensor. Point-point coincidence is
 * singular unless @p omit_singular_point_pair is true.
 */
[[nodiscard]] PairTensor build_pair_tensor(
    const Vec3& target_position,
    const Vec3& source_position,
    SourceGeometry source_geometry = SourceGeometry::PointDipole,
    TargetGeometry target_geometry = TargetGeometry::Point,
    const CuboidSize& source_size = {},
    const CuboidSize& target_size = {},
    bool omit_singular_point_pair = false
);

} // namespace cdfmm
