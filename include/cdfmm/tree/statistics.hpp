// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cdfmm/core/timing.hpp"

namespace cdfmm {

/** @brief Wall-clock breakdown for fixed uniform-tree geometry construction. */
struct TreeBuildTimings {
    /// @brief Time spent determining or validating the root bounds.
    PhaseTiming root_bounds{};
    /// @brief Time spent constructing the complete node array.
    PhaseTiming node_construction{};
    /// @brief Time spent building parent, child, and level topology.
    PhaseTiming topology{};
    /// @brief Time spent assigning Morton keys to source positions.
    PhaseTiming source_morton{};
    /// @brief Time spent sorting sources by Morton key.
    PhaseTiming source_sorting{};
    /// @brief Time spent assigning Morton keys to target positions.
    PhaseTiming target_morton{};
    /// @brief Time spent sorting targets by Morton key.
    PhaseTiming target_sorting{};
    /// @brief Time spent assigning source and target ranges to nodes.
    PhaseTiming ranges{};
    /// @brief Time spent constructing near- and far-field interaction lists.
    PhaseTiming interaction_lists{};
    /// @brief Total tree construction time.
    PhaseTiming total{};
};

} // namespace cdfmm
