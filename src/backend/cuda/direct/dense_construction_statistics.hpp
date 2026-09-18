// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

#include "cdfmm/core/timing.hpp"

namespace cdfmm::detail::cuda_dense_direct {

/**
 * @brief Per-phase cost and device storage of one CUDA dense plan setup.
 *
 * NOTE(cdfmm): internal diagnostics, as for the host record in
 * `src/plan/direct/construction_statistics.hpp`.  `CudaDenseDirectPlan`
 * delegates its exact tensors to a host `DenseDirectPlan`, so the host record
 * describes the construction and this one describes only what the device
 * setup adds: stream and handle creation, device allocation, and the upload.
 */
struct CudaConstructionStatistics {
    /// @brief Time spent building the exact tensors on the host.
    PhaseTiming host_construction{};
    /// @brief Time spent creating the stream and the cuBLAS handle.
    PhaseTiming context_creation{};
    /// @brief Time spent allocating device and pinned host memory.
    PhaseTiming allocation{};
    /// @brief Time spent issuing and completing the host-to-device upload.
    PhaseTiming upload{};
    /// @brief Complete constructor time, i.e. time until ready to evaluate.
    PhaseTiming total{};

    /// @brief Bytes transferred host to device during setup.
    std::size_t upload_bytes{0};
    /// @brief Persistent device bytes retained by the plan.
    std::size_t persistent_device_bytes{0};
    /// @brief Pinned host bytes retained for repeated evaluation staging.
    std::size_t pinned_host_bytes{0};
};

/// @brief Returns the calling thread's record of its most recent CUDA setup.
[[nodiscard]] CudaConstructionStatistics& cuda_construction_statistics() noexcept;

} // namespace cdfmm::detail::cuda_dense_direct
