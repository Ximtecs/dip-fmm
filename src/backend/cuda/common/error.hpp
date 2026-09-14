// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace cdfmm::cuda_detail {

inline void check_cuda(const cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

} // namespace cdfmm::cuda_detail
