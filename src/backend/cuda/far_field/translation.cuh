// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace cdfmm::cuda_far_field_detail {

namespace {

// One level of M2M or L2L. A group of `lanes_per_output` lanes owns one
// (target node, output coefficient) pair and reduces over every translation
// into that node, so a level needs no atomics: within one launch sources and
// targets are distinct tree levels and the in-place coefficient buffer is
// safe. The eight child-class matrices are stored as CSR by output with the
// level scaling already folded in, and the matrix pointers supplied by the
// host already point at this level's block. All lanes join the shuffles.
template <typename Scalar, int lanes_per_output>
__global__ void translate_targets_kernel(
    const int *__restrict__ targets,
    const int *__restrict__ target_interaction_offsets,
    const int *__restrict__ sources, const int *__restrict__ classes,
    const int *__restrict__ matrix_row_offsets,
    const int *__restrict__ matrix_inputs,
    const Scalar *__restrict__ matrix_values, const int target_count,
    const int coefficient_count, const Scalar *__restrict__ input,
    Scalar *__restrict__ output) {
  static_assert(lanes_per_output > 0 && lanes_per_output <= 32 &&
                    (lanes_per_output & (lanes_per_output - 1)) == 0,
                "lane groups must be powers of two within a warp");
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  const int output_index = item / lanes_per_output;
  const int lane = item - output_index * lanes_per_output;
  const bool valid = output_index < target_count * coefficient_count;
  const int target_index = valid ? output_index / coefficient_count : 0;
  const int beta = output_index - target_index * coefficient_count;
  Scalar sum = Scalar{0};
  if (valid) {
    for (int interaction = target_interaction_offsets[target_index];
         interaction < target_interaction_offsets[target_index + 1];
         ++interaction) {
      const Scalar *source_coefficients =
          input + static_cast<std::size_t>(sources[interaction]) *
                      coefficient_count;
      const int row = classes[interaction] * (coefficient_count + 1) + beta;
      const int end = matrix_row_offsets[row + 1];
      for (int entry = matrix_row_offsets[row] + lane; entry < end;
           entry += lanes_per_output) {
        sum += matrix_values[entry] * source_coefficients[matrix_inputs[entry]];
      }
    }
  }
#pragma unroll
  for (int offset = lanes_per_output / 2; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(0xffffffffU, sum, offset);
  }
  if (valid && lane == 0) {
    output[static_cast<std::size_t>(targets[target_index]) * coefficient_count +
           beta] += sum;
  }
}

} // namespace

} // namespace cdfmm::cuda_far_field_detail
