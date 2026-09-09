// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cdfmm/geometry.hpp"

namespace cdfmm {

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
