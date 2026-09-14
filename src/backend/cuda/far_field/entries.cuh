// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace cdfmm::cuda_far_field_detail {

namespace {

template <typename Entry, typename Scalar>
__global__ void apply_entries_kernel(const Entry *entries,
                                     const std::size_t count,
                                     const Scalar *input, Scalar *output) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    const Entry entry = entries[index];
    atomicAdd(output + entry.output, entry.value * input[entry.input]);
  }
}

} // namespace

} // namespace cdfmm::cuda_far_field_detail
