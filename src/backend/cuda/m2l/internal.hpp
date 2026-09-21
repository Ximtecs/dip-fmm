// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include "cdfmm/plan/m2l.hpp"
#include "cdfmm/timings.hpp"

namespace cdfmm::cuda_m2l_detail {

/** @brief One non-empty canonical M2L target row on the device. */
struct CudaM2LActiveRow {
  int target{0};
  int level{0};
  int interaction_begin{0};
  int interaction_end{0};
};

/**
 * @brief Reusable device representation and executor for a static M2L plan.
 *
 * The object owns only persistent M2L resources. Its enqueue operation accepts
 * caller-owned coefficient buffers and stream/event resources, allowing the
 * standalone and complete-FMM CUDA plans to share the exact same kernels and
 * execution policy.
 */
template <typename Scalar, typename Plan>
class CudaM2LExecutionPlan {
public:
  CudaM2LExecutionPlan(const Plan &data, cudaStream_t stream);
  ~CudaM2LExecutionPlan();

  CudaM2LExecutionPlan(const CudaM2LExecutionPlan &) = delete;
  CudaM2LExecutionPlan &operator=(const CudaM2LExecutionPlan &) = delete;

  /// @brief Launches the complete M2L on `stream`.  `scale_complete` is a
  /// diagnostic timing event recorded between the scaling and multiply
  /// phases; pass `nullptr` to record nothing (the phases are stream-ordered
  /// regardless).
  void enqueue(const Scalar *multipoles, Scalar *locals, cudaStream_t stream,
               cudaEvent_t scale_complete) const;

  [[nodiscard]] const CudaPlanStatistics &statistics() const noexcept;

private:
  struct Implementation;
  Implementation *implementation_{nullptr};
};

extern template class CudaM2LExecutionPlan<double, StaticM2LPlan>;
extern template class CudaM2LExecutionPlan<float, FloatStaticM2LPlan>;

} // namespace cdfmm::cuda_m2l_detail
