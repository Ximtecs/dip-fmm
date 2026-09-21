// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace cdfmm {

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

/**
 * @brief How much internal timing a solver collects.
 *
 * Timing is opt-in.  A region whose level is above the selected one is not
 * entered by any clock: no host clock is read, no CUDA timing event is
 * recorded and no elapsed time is queried, so its `PhaseTiming` stays at its
 * default of zero seconds and zero calls.  The level is recorded beside
 * every timing record (`EvaluationTimings::timing_level`,
 * `StaticPlanStatistics::timing_level`) so an uncollected zero can be told
 * from a measured one.  It never changes numerical results, the resolved
 * backend or packing, the cache key or the persisted cache contents.
 *
 * - `Off`: the production path.  Only functional synchronisation runs.
 * - `Coarse`: complete host wall times: the whole evaluation, the CPU
 *   far-field hierarchy and the CPU near field, and at construction the whole
 *   setup and the static-plan build.
 * - `Detailed`: every phase, including the CUDA device-stream lanes, the
 *   oneMKL gather/multiply/scatter split and the construction subphases,
 *   among them the build breakdown of the trees the plan constructs.
 *
 * The level governs everything a plan owns.  For `UniformFmm` that is
 * normalisation, the two uniform trees it builds, topology, every operator
 * and static-plan build, cache access, backend setup and every evaluation;
 * the dense direct plans take the same level as a constructor argument.  A
 * `UniformTree` built standalone collects its own `build_timings()` unless
 * `UniformTreeOptions::collect_build_timings` is false, and `AdaptiveTree`
 * always records its two coarse construction times.
 *
 * NVTX ranges (`CDFMM_ENABLE_PROFILING`) are a separate, compile-time
 * mechanism for external profilers and are unaffected by this level.
 */
enum class TimingLevel : std::uint8_t {
    Off = 0,
    Coarse = 1,
    Detailed = 2
};

} // namespace cdfmm
