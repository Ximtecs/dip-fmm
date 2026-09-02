// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>

#include "cdfmm/static_operators.hpp"
#include "cdfmm/timings.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Public types
//------------------------------------------------------------------------------

/**
 * @brief Persistent CUDA executor for one static near-field packing.
 *
 * Construction uploads immutable geometry-dependent data. Repeated evaluation
 * transfers changing moments and result fields. A self-identity map supplied
 * at construction is uploaded once; omitting it retains the dynamic identity
 * behaviour required by the existing hybrid FMM path. The constructors select
 * canonical AoS rows, source-only SoA rows, compact leaf blocks, or cuSPARSE
 * BSR(3), respectively.
 */
class CudaP2PPlan {
public:
  /** @brief Builds the canonical one-thread-per-target CUDA baseline. */
  explicit CudaP2PPlan(const StaticP2POperator &operator_map,
                       std::span<const int> fixed_self_indices = {});

  /** @brief Builds the source-only SoA one-thread-per-target CUDA plan. */
  explicit CudaP2PPlan(const StaticP2PCompactPlan &plan,
                       std::span<const int> fixed_self_indices = {});

  /** @brief Builds the shared-memory one-block-per-target-leaf CUDA plan. */
  explicit CudaP2PPlan(const StaticP2PLeafPlan &plan,
                       std::span<const int> fixed_self_indices = {});

  /** @brief Builds a persistent cuSPARSE BSR(3) CUDA plan. */
  explicit CudaP2PPlan(const StaticP2PBsrPlan &plan);

  /** @brief Builds the canonical FP32 CUDA baseline. */
  explicit CudaP2PPlan(const FloatStaticP2POperator &operator_map,
                       std::span<const int> fixed_self_indices = {});
  /** @brief Builds the source-only FP32 SoA CUDA plan. */
  explicit CudaP2PPlan(const FloatStaticP2PCompactPlan &plan,
                       std::span<const int> fixed_self_indices = {});
  /** @brief Builds the shared-memory FP32 leaf CUDA plan. */
  explicit CudaP2PPlan(const FloatStaticP2PLeafPlan &plan,
                       std::span<const int> fixed_self_indices = {});
  /** @brief Builds a persistent FP32 cuSPARSE BSR(3) plan. */
  explicit CudaP2PPlan(const FloatStaticP2PBsrPlan &plan);

  ~CudaP2PPlan();
  CudaP2PPlan(const CudaP2PPlan &) = delete;
  CudaP2PPlan &operator=(const CudaP2PPlan &) = delete;

  /** @brief Starts asynchronous upload, execution, and result download. */
  void begin_evaluate(std::span<const Vec3> moments,
                      std::span<const int> target_source_indices);

  /** @brief Waits for a pending evaluation and copies its result. */
  void finish_evaluate(std::span<Vec3> fields);

  /** @brief Cancels the pending host-side state after stream completion. */
  void cancel_evaluate() noexcept;

  /** @brief Applies the static plan synchronously to changing moments. */
  void evaluate(std::span<const Vec3> moments,
                std::span<const int> target_source_indices,
                std::span<Vec3> fields);

  void begin_evaluate(std::span<const FloatVec3> moments,
                      std::span<const int> target_source_indices);
  void finish_evaluate(std::span<FloatVec3> fields);
  void evaluate(std::span<const FloatVec3> moments,
                std::span<const int> target_source_indices,
                std::span<FloatVec3> fields);

  /// @brief Returns persistent storage and transfer diagnostics.
  [[nodiscard]] const CudaPlanStatistics &statistics() const noexcept;

  /// @brief Returns device-stream timings for the latest evaluation.
  [[nodiscard]] const CudaEvaluationTimings &timings() const noexcept;

private:
  struct Implementation;

  CudaP2PPlan(int source_count, int target_count,
              std::span<const int> fixed_self_indices,
              bool device_self_indices);
  CudaP2PPlan(int source_count, int target_count,
              std::span<const int> fixed_self_indices,
              bool device_self_indices, StaticPrecision precision);

  Implementation *implementation_{nullptr};
};

} // namespace cdfmm
