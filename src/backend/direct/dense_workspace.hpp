// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <vector>

namespace cdfmm::detail::dense_direct {

// Evaluation staging is mutable execution state, not part of the immutable
// pair-tensor plan.  Keeping it here lets the CPU backends reuse allocations
// without exposing implementation details through the public plan header.
struct DenseDirectWorkspace {
    std::array<std::vector<float>, 3> float_moments;
    std::array<std::vector<float>, 3> float_fields;
    std::array<std::vector<double>, 3> double_moments;
    std::array<std::vector<double>, 3> double_fields;
};

} // namespace cdfmm::detail::dense_direct
