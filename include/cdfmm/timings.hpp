// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace cdfmm {

//------------------------------------------------------------------------------
// Public timing types
//------------------------------------------------------------------------------

/** @brief Accumulated monotonic wall time and invocation count for one phase. */
struct PhaseTiming {
    /// @brief Accumulated wall time in seconds.
    double total_seconds{0.0};
    /// @brief Number of observations included in the accumulated time.
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

/** @brief Wall-clock breakdown for one or more dipole evaluations. */
struct EvaluationTimings {
    /// @brief Time spent permuting moments into Morton order.
    PhaseTiming moment_permutation{};
    /// @brief Time spent clearing multipole coefficients.
    PhaseTiming multipole_reset{};
    /// @brief Time spent accumulating leaf multipoles.
    PhaseTiming p2m{};
    /// @brief Time spent translating child multipoles to their parents.
    PhaseTiming m2m{};
    /// @brief Time spent clearing local coefficients.
    PhaseTiming local_reset{};
    /// @brief Time spent translating parent locals to their children.
    PhaseTiming l2l{};
    /// @brief Time spent translating well-separated multipoles to locals.
    PhaseTiming m2l{};
    /// @brief Time spent evaluating local expansions at targets.
    PhaseTiming l2p{};
    /// @brief Time spent evaluating direct near-field interactions.
    PhaseTiming p2p{};
    /// @brief Time spent restoring results to user target order.
    PhaseTiming result_unpermutation{};
    /// @brief Total complete-evaluation time.
    PhaseTiming total{};
    /// @brief Number of complete evaluations represented by these timings.
    std::uint64_t evaluations{0};
};

} // namespace cdfmm
