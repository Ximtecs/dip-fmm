// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <vector>

namespace cdfmm {

/**
 * @brief Fixed FP64 potential and field rows for one target offset.
 *
 * With local coefficients `L`, `phi = sum_c potential[c] * L[c]` and
 * `H_k = sum_c field[k][c] * L[c]`; the field rows already carry the minus
 * sign of `H = -grad(phi)`.
 */
struct StaticL2PEvaluator {
    std::vector<double> potential{};              ///< Potential row, one entry per coefficient.
    std::array<std::vector<double>, 3> field{};   ///< Field rows for H_x, H_y, H_z.
};

/** @brief Fixed FP32 potential and field rows for one target offset. */
struct FloatStaticL2PEvaluator {
    std::vector<float> potential{};               ///< Potential row, one entry per coefficient.
    std::array<std::vector<float>, 3> field{};    ///< Field rows for H_x, H_y, H_z.
};

} // namespace cdfmm
