// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace cdfmm {

//------------------------------------------------------------------------------
// Public timing types
//------------------------------------------------------------------------------

/** @brief Accumulated monotonic wall time and invocation count for one phase. */
struct PhaseTiming {
    double total_seconds{0.0};
    std::uint64_t calls{0};

    /// @brief Adds one elapsed wall-time observation.
    void add(double seconds)
    {
        total_seconds += seconds;
        ++calls;
    }
};

/** @brief Wall-clock breakdown for fixed uniform-tree geometry construction. */
struct TreeBuildTimings {
    PhaseTiming root_bounds{};
    PhaseTiming node_construction{};
    PhaseTiming topology{};
    PhaseTiming source_morton{};
    PhaseTiming source_sorting{};
    PhaseTiming target_morton{};
    PhaseTiming target_sorting{};
    PhaseTiming ranges{};
    PhaseTiming interaction_lists{};
    PhaseTiming total{};
};

/** @brief Wall-clock breakdown for one or more dipole evaluations. */
struct EvaluationTimings {
    PhaseTiming moment_permutation{};
    PhaseTiming multipole_reset{};
    PhaseTiming p2m{};
    PhaseTiming m2m{};
    PhaseTiming local_reset{};
    PhaseTiming l2l{};
    PhaseTiming m2l{};
    PhaseTiming l2p{};
    PhaseTiming p2p{};
    PhaseTiming result_unpermutation{};
    PhaseTiming total{};
    std::uint64_t evaluations{0};
};

} // namespace cdfmm
