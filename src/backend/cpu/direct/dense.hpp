// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "cdfmm/math/vec3.hpp"
#include "backend/direct/dense_workspace.hpp"

namespace cdfmm::detail::dense_direct::cpu {

void apply(
    const std::array<std::vector<float>, 6>& matrices,
    std::size_t target_count,
    std::size_t source_count,
    std::span<const cdfmm::Vec3> total_moments,
    DenseDirectWorkspace& workspace,
    std::vector<cdfmm::Vec3>& result);

void apply(
    const std::array<std::vector<double>, 6>& matrices,
    std::size_t target_count,
    std::size_t source_count,
    std::span<const cdfmm::Vec3> total_moments,
    DenseDirectWorkspace& workspace,
    std::vector<cdfmm::Vec3>& result);

} // namespace cdfmm::detail::dense_direct::cpu
