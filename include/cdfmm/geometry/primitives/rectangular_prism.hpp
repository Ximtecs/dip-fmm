// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cdfmm/math/multi_index.hpp"
#include "cdfmm/math/pair_tensor.hpp"
#include "cdfmm/geometry/models.hpp"
#include "cdfmm/math/vec3.hpp"

namespace cdfmm {

/** @brief Full side lengths of an axis-aligned rectangular prism. */
struct RectangularPrism {
    double hx{0.0};   ///< Full side length along x.
    double hy{0.0};   ///< Full side length along y.
    double hz{0.0};   ///< Full side length along z.

    /// @brief Returns the prism volume.
    [[nodiscard]] double volume() const noexcept { return hx * hy * hz; }
};

/** @brief Compatibility name for the pre-generalisation prism record. */
using CuboidSize = RectangularPrism;

/**
 * @brief Evaluates the factorial-normalised monomial averaged over a prism.
 *
 * This is J_beta(d,h) = V^-1 integral_V (d+u)^beta/beta! dV and is evaluated
 * by its finite even-power sum, without numerical quadrature.  The prism is
 * centred on `d = target - representative` and carries full side lengths.
 */
[[nodiscard]] double rectangular_prism_averaged_monomial(
    const MultiIndex& beta,
    const Vec3& d,
    const RectangularPrism& prism
);

/** @brief Legacy cuboid spelling of rectangular_prism_averaged_monomial. */
[[nodiscard]] double cuboid_averaged_monomial(
    const MultiIndex& beta,
    const Vec3& d,
    const CuboidSize& h
);

/**
 * @brief Exact field tensor of a uniformly magnetised rectangular prism at a
 * point.
 *
 * `source` contains full side lengths and is centred on the source
 * representative.  The displacement is `target - source_representative`.
 * The returned tensor maps the source's total magnetic moment (rather than
 * its magnetisation) to the signed field H, hence it includes `1/source.V`.
 */
[[nodiscard]] PairTensor rectangular_prism_point_tensor(
    const Vec3& target_minus_source_representative,
    const RectangularPrism& source
);

/**
 * @brief Exact field tensor averaged over a rectangular-prism target.
 *
 * Both records contain full side lengths and are centred on their respective
 * representatives.  The returned tensor maps source total moment to target
 * volume-averaged H, including `1/(source.V*target.V)`.
 */
[[nodiscard]] PairTensor rectangular_prism_rectangular_prism_tensor(
    const Vec3& target_minus_source_representative,
    const RectangularPrism& source,
    const RectangularPrism& target
);

} // namespace cdfmm
