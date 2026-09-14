// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace cdfmm::cuda_far_field_detail {

namespace {

template <typename Entry, typename Scalar>
__global__ void apply_shared_translation_kernel(
    const Entry *matrices, const CudaTranslationInteraction *interactions,
    const std::size_t interaction_count, const int entries_per_matrix,
    const int coefficient_count, const int *coefficient_degrees,
    const int level, const Scalar *input, Scalar *output) {
  const std::size_t item =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t item_count = interaction_count * entries_per_matrix;
  if (item >= item_count) {
    return;
  }
  const std::size_t interaction_index = item / entries_per_matrix;
  const CudaTranslationInteraction interaction =
      interactions[interaction_index];
  if (interaction.level != level) {
    return;
  }
  const int matrix_entry = static_cast<int>(item % entries_per_matrix);
  const Entry entry = matrices[static_cast<std::size_t>(interaction.matrix_id) *
                                   entries_per_matrix +
                               matrix_entry];
  const int degree_difference =
      coefficient_degrees[entry.output] - coefficient_degrees[entry.input];
  const int power =
      degree_difference < 0 ? -degree_difference : degree_difference;
  const Scalar scaled_value =
      ldexp(static_cast<Scalar>(entry.value), -(level - 1) * power);
  atomicAdd(output +
                static_cast<std::size_t>(interaction.target_node) *
                    coefficient_count +
                entry.output,
            scaled_value *
                input[static_cast<std::size_t>(interaction.source_node) *
                          coefficient_count +
                      entry.input]);
}

} // namespace

} // namespace cdfmm::cuda_far_field_detail
