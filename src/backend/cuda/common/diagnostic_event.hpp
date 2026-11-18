// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

#include "backend/cuda/common/error.hpp"

namespace cdfmm::cuda_detail {

// Diagnostic timing events are separate from functional synchronisation.  A
// functional event (a cross-stream dependency or the host completion point)
// is created with cudaEventDisableTiming and recorded on every evaluation; a
// diagnostic event is timing-capable and exists only to bound a phase for the
// device lanes of `CudaEvaluationTimings`.  It is recorded only when the plan
// collects detailed timings, and no elapsed time is queried below that level.

/// @brief Records `event` on `stream` only when `detailed` is set.
inline void record_diagnostic(const bool detailed, const cudaEvent_t event,
                              const cudaStream_t stream, const char *what) {
  if (detailed) {
    check_cuda(cudaEventRecord(event, stream), what);
  }
}

/// @brief Seconds between two completed diagnostic events.
inline double diagnostic_elapsed_seconds(const cudaEvent_t first,
                                         const cudaEvent_t second,
                                         const char *what) {
  // The host waits on the functional completion event, not on these. A
  // diagnostic event recorded around that point may not have completed yet,
  // and cudaEventElapsedTime then fails with cudaErrorNotReady ("device not
  // ready"; seen intermittently on Linux and, where the driver batches stream
  // commands under WDDM, regularly on Windows). Waiting on both returns at
  // once where they have already completed.
  check_cuda(cudaEventSynchronize(first), what);
  check_cuda(cudaEventSynchronize(second), what);
  float milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&milliseconds, first, second), what);
  return static_cast<double>(milliseconds) * 1.0e-3;
}

} // namespace cdfmm::cuda_detail
