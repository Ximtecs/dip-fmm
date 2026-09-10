// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cdfmm/periodic.hpp"
#include "cdfmm/tree/static_topology.hpp"
#include "cdfmm/tree/uniform_tree.hpp"

namespace cdfmm {

/** @brief Extracts canonical topology and geometry from a uniform tree. */
[[nodiscard]] StaticFmmTopology build_uniform_fmm_topology(
    const UniformTree& tree,
    const PeriodicCellOptions& periodic = {});

} // namespace cdfmm
