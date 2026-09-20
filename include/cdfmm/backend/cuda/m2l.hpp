// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>

#include "cdfmm/plan/m2l.hpp"
#include "cdfmm/timings.hpp"

namespace cdfmm {

/**
 * @brief Persistent CUDA executor for a canonical target-row M2L plan.
 *
 * Construction uploads the immutable transfer matrices and interaction
 * metadata. Repeated evaluation transfers the changing multipole coefficients
 * and resulting local coefficients while preserving the precision selected by
 * the supplied static plan.
 */
class CudaM2LPlan {
public:
  /// @brief Uploads an FP64 plan; throws when no CUDA device is available.
  explicit CudaM2LPlan(const StaticM2LPlan &plan);
  /// @brief Uploads an FP32 plan; throws when no CUDA device is available.
  explicit CudaM2LPlan(const FloatStaticM2LPlan &plan);
  ~CudaM2LPlan();

  CudaM2LPlan(const CudaM2LPlan &) = delete;
  CudaM2LPlan &operator=(const CudaM2LPlan &) = delete;

  /// @brief Applies every level of the FP64 M2L: all node multipoles in, all node locals out (raw, before L2L).
  void evaluate(std::span<const double> multipoles,
                std::span<double> raw_locals);
  /// @brief Applies every level of the FP32 M2L: all node multipoles in, all node locals out (raw, before L2L).
  void evaluate(std::span<const float> multipoles,
                std::span<float> raw_locals);

  /// @brief Persistent storage and transfer diagnostics.
  [[nodiscard]] const CudaPlanStatistics &statistics() const noexcept;
  /// @brief Device-stream timings of the latest evaluation.
  [[nodiscard]] const CudaEvaluationTimings &timings() const noexcept;

private:
  struct Implementation;
  Implementation *implementation_{nullptr};
};

} // namespace cdfmm
