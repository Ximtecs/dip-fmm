// SPDX-License-Identifier: Apache-2.0

#include "backend/cuda/common/error.hpp"
#include "backend/cuda/far_field/internal.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace cdfmm::cuda_far_field_detail {

namespace {

using cuda_detail::check_cuda;

constexpr int far_field_threads = 256;

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

template <typename Scalar, typename Entry>
struct CudaFarFieldExecutionPlan<Scalar, Entry>::Implementation {
  Entry *entries{nullptr};
  Entry *m2m_matrices{nullptr};
  Entry *l2l_matrices{nullptr};
  CudaTranslationInteraction *m2m_interactions{nullptr};
  CudaTranslationInteraction *l2l_interactions{nullptr};
  int *coefficient_degrees{nullptr};

  std::vector<std::size_t> offsets{};
  std::vector<std::size_t> counts{};
  int coefficient_count{0};
  int maximum_level{0};
  int m2m_entries_per_matrix{0};
  int l2l_entries_per_matrix{0};
  std::size_t m2m_interaction_count{0};
  std::size_t l2l_interaction_count{0};
  CudaPlanStatistics statistics{};

  ~Implementation() {
    cudaFree(entries);
    cudaFree(m2m_matrices);
    cudaFree(l2l_matrices);
    cudaFree(m2m_interactions);
    cudaFree(l2l_interactions);
    cudaFree(coefficient_degrees);
  }
};

namespace {

template <typename T>
void allocate(T **pointer, const std::size_t bytes, const char *description) {
  check_cuda(cudaMalloc(reinterpret_cast<void **>(pointer),
                        std::max(bytes, std::size_t{1})),
             description);
}

template <typename T>
void upload(T *destination, std::span<const T> values, cudaStream_t stream,
            const char *description) {
  if (!values.empty()) {
    check_cuda(cudaMemcpyAsync(destination, values.data(), values.size_bytes(),
                               cudaMemcpyHostToDevice, stream),
               description);
  }
}

} // namespace

template <typename Scalar, typename Entry>
CudaFarFieldExecutionPlan<Scalar, Entry>::CudaFarFieldExecutionPlan(
    const CudaFarFieldStaticData<Entry> &data, cudaStream_t stream)
    : implementation_(new Implementation{}) {
  try {
    auto &plan = *implementation_;
    if (data.coefficient_count < 0 || data.maximum_level < -1 ||
        data.m2m_entries_per_matrix < 0 || data.l2l_entries_per_matrix < 0 ||
        data.m2m_matrix_count < 0 || data.l2l_matrix_count < 0) {
      throw std::invalid_argument("CUDA far-field dimensions are invalid");
    }
    if (data.coefficient_degrees.size() !=
        static_cast<std::size_t>(data.coefficient_count)) {
      throw std::invalid_argument("CUDA far-field degree metadata is invalid");
    }
    if (data.m2m_entries_per_matrix != 0 &&
        data.m2m_matrices.size() %
                static_cast<std::size_t>(data.m2m_entries_per_matrix) !=
            0) {
      throw std::invalid_argument("CUDA far-field M2M matrix data is invalid");
    }
    if (data.l2l_entries_per_matrix != 0 &&
        data.l2l_matrices.size() %
                static_cast<std::size_t>(data.l2l_entries_per_matrix) !=
            0) {
      throw std::invalid_argument("CUDA far-field L2L matrix data is invalid");
    }
    if (data.m2m_entries_per_matrix == 0 && !data.m2m_interactions.empty()) {
      throw std::invalid_argument(
          "CUDA far-field M2M interactions are invalid");
    }
    if (data.l2l_entries_per_matrix == 0 && !data.l2l_interactions.empty()) {
      throw std::invalid_argument(
          "CUDA far-field L2L interactions are invalid");
    }

    plan.coefficient_count = data.coefficient_count;
    plan.maximum_level = data.maximum_level;
    plan.m2m_entries_per_matrix = data.m2m_entries_per_matrix;
    plan.l2l_entries_per_matrix = data.l2l_entries_per_matrix;
    plan.m2m_interaction_count = data.m2m_interactions.size();
    plan.l2l_interaction_count = data.l2l_interactions.size();

    plan.offsets.reserve(2);
    plan.counts.reserve(2);
    plan.offsets.push_back(0);
    plan.counts.push_back(data.p2m.size());
    plan.offsets.push_back(data.p2m.size());
    plan.counts.push_back(data.l2p.size());
    std::vector<Entry> entries;
    entries.reserve(data.p2m.size() + data.l2p.size());
    entries.insert(entries.end(), data.p2m.begin(), data.p2m.end());
    entries.insert(entries.end(), data.l2p.begin(), data.l2p.end());

    allocate(&plan.entries, entries.size() * sizeof(Entry),
             "allocate CUDA far-field entries");
    allocate(&plan.coefficient_degrees,
             data.coefficient_degrees.size() * sizeof(int),
             "allocate CUDA far-field coefficient degrees");
    allocate(&plan.m2m_matrices, data.m2m_matrices.size() * sizeof(Entry),
             "allocate CUDA far-field M2M matrices");
    allocate(&plan.m2m_interactions,
             data.m2m_interactions.size() * sizeof(CudaTranslationInteraction),
             "allocate CUDA far-field M2M interactions");
    allocate(&plan.l2l_matrices, data.l2l_matrices.size() * sizeof(Entry),
             "allocate CUDA far-field L2L matrices");
    allocate(&plan.l2l_interactions,
             data.l2l_interactions.size() * sizeof(CudaTranslationInteraction),
             "allocate CUDA far-field L2L interactions");

    upload(plan.entries, std::span<const Entry>(entries), stream,
           "upload CUDA far-field entries");
    upload(plan.coefficient_degrees, data.coefficient_degrees, stream,
           "upload CUDA far-field coefficient degrees");
    upload(plan.m2m_matrices, data.m2m_matrices, stream,
           "upload CUDA far-field M2M matrices");
    upload(plan.m2m_interactions, data.m2m_interactions, stream,
           "upload CUDA far-field M2M interactions");
    upload(plan.l2l_matrices, data.l2l_matrices, stream,
           "upload CUDA far-field L2L matrices");
    upload(plan.l2l_interactions, data.l2l_interactions, stream,
           "upload CUDA far-field L2L interactions");

    plan.statistics.scalar_bytes = sizeof(Scalar);
    plan.statistics.m2m_unique_matrix_count = data.m2m_matrix_count;
    plan.statistics.m2m_matrix_bytes = data.m2m_matrices.size() * sizeof(Entry);
    plan.statistics.l2l_unique_matrix_count = data.l2l_matrix_count;
    plan.statistics.l2l_matrix_bytes = data.l2l_matrices.size() * sizeof(Entry);
    plan.statistics.setup_h2d_bytes =
        entries.size() * sizeof(Entry) +
        data.coefficient_degrees.size() * sizeof(int) +
        data.m2m_matrices.size() * sizeof(Entry) +
        data.m2m_interactions.size() * sizeof(CudaTranslationInteraction) +
        data.l2l_matrices.size() * sizeof(Entry) +
        data.l2l_interactions.size() * sizeof(CudaTranslationInteraction);
    plan.statistics.persistent_device_bytes = plan.statistics.setup_h2d_bytes;
  } catch (...) {
    delete implementation_;
    implementation_ = nullptr;
    throw;
  }
}

template <typename Scalar, typename Entry>
CudaFarFieldExecutionPlan<Scalar, Entry>::~CudaFarFieldExecutionPlan() {
  if (implementation_ == nullptr) {
    return;
  }
  delete implementation_;
}

template <typename Scalar, typename Entry>
void CudaFarFieldExecutionPlan<Scalar, Entry>::enqueue_p2m(
    const Scalar *input, Scalar *output, cudaStream_t stream) const {
  const auto &plan = *implementation_;
  const std::size_t count = plan.counts[0];
  if (count != 0) {
    apply_entries_kernel<<<(count + far_field_threads - 1) / far_field_threads,
                           far_field_threads, 0, stream>>>(
        plan.entries + plan.offsets[0], count, input, output);
    check_cuda(cudaGetLastError(), "launch CUDA far-field P2M kernel");
  }
}

template <typename Scalar, typename Entry>
void CudaFarFieldExecutionPlan<Scalar, Entry>::enqueue_m2m(
    const Scalar *input, Scalar *output, cudaStream_t stream) const {
  const auto &plan = *implementation_;
  const std::size_t items =
      plan.m2m_interaction_count * plan.m2m_entries_per_matrix;
  if (items == 0) {
    return;
  }
  for (int level = plan.maximum_level; level >= 1; --level) {
    apply_shared_translation_kernel<<<(items + far_field_threads - 1) /
                                          far_field_threads,
                                      far_field_threads, 0, stream>>>(
        plan.m2m_matrices, plan.m2m_interactions, plan.m2m_interaction_count,
        plan.m2m_entries_per_matrix, plan.coefficient_count,
        plan.coefficient_degrees, level, input, output);
    check_cuda(cudaGetLastError(), "launch CUDA far-field M2M kernel");
  }
}

template <typename Scalar, typename Entry>
void CudaFarFieldExecutionPlan<Scalar, Entry>::enqueue_l2l(
    const Scalar *input, Scalar *output, cudaStream_t stream) const {
  const auto &plan = *implementation_;
  const std::size_t items =
      plan.l2l_interaction_count * plan.l2l_entries_per_matrix;
  if (items == 0) {
    return;
  }
  for (int level = 1; level <= plan.maximum_level; ++level) {
    apply_shared_translation_kernel<<<(items + far_field_threads - 1) /
                                          far_field_threads,
                                      far_field_threads, 0, stream>>>(
        plan.l2l_matrices, plan.l2l_interactions, plan.l2l_interaction_count,
        plan.l2l_entries_per_matrix, plan.coefficient_count,
        plan.coefficient_degrees, level, input, output);
    check_cuda(cudaGetLastError(), "launch CUDA far-field L2L kernel");
  }
}

template <typename Scalar, typename Entry>
void CudaFarFieldExecutionPlan<Scalar, Entry>::enqueue_l2p(
    const Scalar *input, Scalar *output, cudaStream_t stream) const {
  const auto &plan = *implementation_;
  const std::size_t count = plan.counts[1];
  if (count != 0) {
    apply_entries_kernel<<<(count + far_field_threads - 1) / far_field_threads,
                           far_field_threads, 0, stream>>>(
        plan.entries + plan.offsets[1], count, input, output);
    check_cuda(cudaGetLastError(), "launch CUDA far-field L2P kernel");
  }
}

template <typename Scalar, typename Entry>
const CudaPlanStatistics &
CudaFarFieldExecutionPlan<Scalar, Entry>::statistics() const noexcept {
  return implementation_->statistics;
}

template class CudaFarFieldExecutionPlan<double, StaticOperatorEntry>;
template class CudaFarFieldExecutionPlan<float, FloatStaticOperatorEntry>;

} // namespace cdfmm::cuda_far_field_detail
