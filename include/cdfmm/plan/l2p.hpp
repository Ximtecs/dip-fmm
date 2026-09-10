// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <vector>

namespace cdfmm {

/** @brief Fixed FP64 potential and field rows for one target offset. */
struct StaticL2PEvaluator {
    std::vector<double> potential{};
    std::array<std::vector<double>, 3> field{};
};

/** @brief Fixed FP32 potential and field rows for one target offset. */
struct FloatStaticL2PEvaluator {
    std::vector<float> potential{};
    std::array<std::vector<float>, 3> field{};
};

} // namespace cdfmm
