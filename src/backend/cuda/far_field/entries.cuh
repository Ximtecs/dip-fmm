// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace cdfmm::cuda_far_field_detail {

namespace {

// A group of `lanes_per_row` consecutive lanes owns one output row of a sparse
// coefficient map stored as CSR by output. The lanes stride through the row,
// so a long row still keeps many loads in flight, and the group reduces with
// shuffles; P2M and L2P therefore need no atomics and stay deterministic. Every
// lane of a warp takes part in the shuffles, including lanes past the last
// row, so the group is always complete. Empty rows leave the output untouched,
// which is why callers still clear it once per pass.
template <typename Scalar, int lanes_per_row>
__global__ void apply_csr_rows_kernel(const int *__restrict__ row_offsets,
                                      const int *__restrict__ inputs,
                                      const Scalar *__restrict__ values,
                                      const int row_count,
                                      const Scalar *__restrict__ input,
                                      Scalar *__restrict__ output) {
  static_assert(lanes_per_row > 0 && lanes_per_row <= 32 &&
                    (lanes_per_row & (lanes_per_row - 1)) == 0,
                "lane groups must be powers of two within a warp");
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = item / lanes_per_row;
  const int lane = item - row * lanes_per_row;
  const bool valid = row < row_count;
  const int begin = valid ? row_offsets[row] : 0;
  const int end = valid ? row_offsets[row + 1] : 0;
  Scalar sum = Scalar{0};
  for (int entry = begin + lane; entry < end; entry += lanes_per_row) {
    sum += values[entry] * input[inputs[entry]];
  }
#pragma unroll
  for (int offset = lanes_per_row / 2; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(0xffffffffU, sum, offset);
  }
  if (valid && lane == 0 && begin != end) {
    output[row] += sum;
  }
}

} // namespace

} // namespace cdfmm::cuda_far_field_detail
