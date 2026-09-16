// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

#include "error.hpp"

namespace cdfmm::cuda_detail {

/**
 * @brief Creates a non-blocking stream at the device's greatest priority.
 *
 * The far-field hierarchy (M2L in the hybrid backend, the whole far field in
 * the device-resident backend) runs concurrently with the list-1 P2P kernel,
 * which saturates every SM for milliseconds at high leaf occupancy. On an
 * equal-priority stream the far-field launches then wait behind it and the
 * host's dependent L2L/L2P work (hybrid) or the final combination (full)
 * starts late; a higher-priority stream lets the block scheduler feed the
 * short far-field kernels first, while the P2P kernel absorbs the delay it
 * has to absorb anyway. Measured in the Phase-3 P2P unification study.
 */
inline void create_priority_stream(cudaStream_t &stream,
                                   const char *operation) {
  int least_priority = 0;
  int greatest_priority = 0;
  check_cuda(cudaDeviceGetStreamPriorityRange(&least_priority,
                                              &greatest_priority),
             operation);
  check_cuda(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking,
                                          greatest_priority),
             operation);
}

} // namespace cdfmm::cuda_detail
