// SPDX-License-Identifier: Apache-2.0

#include "runtime.hpp"

#include "error.hpp"

#include <cuda_runtime.h>

#include <string>

namespace cdfmm {

bool cuda_runtime_available() noexcept {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

bool cuda_compiled() noexcept { return true; }

std::string cuda_runtime_description() {
  int device = 0;
  cudaDeviceProp properties{};
  cuda_detail::check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  cuda_detail::check_cuda(cudaGetDeviceProperties(&properties, device),
                          "cudaGetDeviceProperties");
  return std::string(properties.name) + " (compute capability " +
         std::to_string(properties.major) + "." +
         std::to_string(properties.minor) + ")";
}

} // namespace cdfmm
