// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

#include "cdfmm/geometry/primitives/rectangular_prism.hpp"

namespace cdfmm::detail {

// All average monomials J_alpha(d,h) in basis order. Each coordinate's
// one-dimensional average is computed once, then the separable factors are
// combined in the same x/y/z order as the scalar reference.
std::vector<double> rectangular_prism_averaged_monomials(
    const MultiIndexSet& basis, const Vec3& displacement,
    const RectangularPrism& prism);

} // namespace cdfmm::detail
