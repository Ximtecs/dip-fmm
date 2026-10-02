// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include "error.hpp"

namespace cdfmm::cuda_detail {

/**
 * @brief Copies `bytes` of pageable host memory to the device and returns only
 *        once they are in device memory.
 *
 * WARNING(cdfmm): a synchronous `cudaMemcpy` from pageable host memory may
 * return as soon as the source has been staged, before the transfer reaches
 * device memory, and it runs on the legacy default stream. Every device plan
 * executes on `cudaStreamNonBlocking` streams, which are not ordered after the
 * legacy stream, so a plan's first evaluations could read its static data
 * while that transfer was still in flight. That window is short on an idle
 * GPU and wide when other processes share it: CudaPartial plans returned the
 * previous plan's near field in 4-6 of 12 runs of four concurrent test
 * processes. Construction therefore waits for the device before it returns.
 * The copy is setup-time only; evaluation-time uploads are stream-ordered
 * `cudaMemcpyAsync` calls on the stream that consumes them.
 */
inline void upload_to_device(void *destination, const void *source,
                             const std::size_t bytes, const char *operation) {
  if (bytes == 0) {
    return;
  }
  check_cuda(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice),
             operation);
  check_cuda(cudaDeviceSynchronize(), operation);
}

} // namespace cdfmm::cuda_detail
