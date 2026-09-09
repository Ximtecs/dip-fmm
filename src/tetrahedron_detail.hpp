// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>

#include "cdfmm/vec3.hpp"

namespace cdfmm::detail {

/**
 * @brief Evaluates the unnormalised constant-density 1/R triangle pair
 * integral.
 *
 * This header is intentionally private to analytical-kernel tests.  The
 * production geometry API exposes only the resulting tetrahedron tensor.
 */
[[nodiscard]] double triangle_triangle_laplace_integral(
    const std::array<Vec3, 3>& first,
    const std::array<Vec3, 3>& second);

} // namespace cdfmm::detail
