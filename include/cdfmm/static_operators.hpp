// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <vector>

#include "cdfmm/operators.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Canonical static mathematical operators
//------------------------------------------------------------------------------

/**
 * @brief Builds the exact dense Cartesian M2L matrix for one displacement.
 *
 * The column-major matrix satisfies L += T(R) M and is shared by execution
 * packings rather than re-derived by individual CPU or CUDA backends.
 */
[[nodiscard]] std::vector<double> build_static_m2l_matrix(
    const MultiIndexSet& basis,
    const Vec3& R
);

/** @brief Applies a column-major square static coefficient operator. */
void apply_static_coefficient_matrix(
    std::span<const double> matrix,
    std::span<const double> input,
    std::span<double> output
);

} // namespace cdfmm
