// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <span>
#include <vector>

#include "cdfmm/plan/static_coefficient.hpp"
#include "cdfmm/timings.hpp"
#include "cuda_fmm_plan.hpp"

namespace cdfmm::cuda_far_field_detail {

/** @brief Host view of immutable CUDA P2M/M2M/L2L/L2P execution data. */
template <typename Entry> struct CudaFarFieldStaticData {
  int coefficient_count{0};
  int maximum_level{0};
  std::span<const int> coefficient_degrees{};
  std::span<const Entry> p2m{};
  std::span<const Entry> l2p{};
  std::span<const Entry> m2m_matrices{};
  std::span<const CudaTranslationInteraction> m2m_interactions{};
  int m2m_entries_per_matrix{0};
  int m2m_matrix_count{0};
  std::span<const Entry> l2l_matrices{};
  std::span<const CudaTranslationInteraction> l2l_interactions{};
  int l2l_entries_per_matrix{0};
  int l2l_matrix_count{0};
};

/**
 * @brief Persistent CUDA executor for the non-M2L far-field passes.
 *
 * Immutable operator entries, shared translation tables, and degree metadata
 * are uploaded once. Evaluation buffers remain owned by the complete FMM
 * plan and are supplied to each enqueue operation.
 */
template <typename Scalar, typename Entry> class CudaFarFieldExecutionPlan {
public:
  CudaFarFieldExecutionPlan(const CudaFarFieldStaticData<Entry> &data,
                            cudaStream_t stream);
  ~CudaFarFieldExecutionPlan();

  CudaFarFieldExecutionPlan(const CudaFarFieldExecutionPlan &) = delete;
  CudaFarFieldExecutionPlan &
  operator=(const CudaFarFieldExecutionPlan &) = delete;

  void enqueue_p2m(const Scalar *input, Scalar *output,
                   cudaStream_t stream) const;
  void enqueue_m2m(const Scalar *input, Scalar *output,
                   cudaStream_t stream) const;
  void enqueue_l2l(const Scalar *input, Scalar *output,
                   cudaStream_t stream) const;
  void enqueue_l2p(const Scalar *input, Scalar *output,
                   cudaStream_t stream) const;

  [[nodiscard]] const CudaPlanStatistics &statistics() const noexcept;

private:
  struct Implementation;
  Implementation *implementation_{nullptr};
};

extern template class CudaFarFieldExecutionPlan<double, StaticOperatorEntry>;
extern template class CudaFarFieldExecutionPlan<float,
                                                FloatStaticOperatorEntry>;

} // namespace cdfmm::cuda_far_field_detail
