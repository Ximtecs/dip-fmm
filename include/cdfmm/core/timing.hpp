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

} // namespace cdfmm
