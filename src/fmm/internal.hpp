// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cdfmm/uniform_fmm.hpp"
#include "cdfmm/backend/cuda/m2l.hpp"
#include "cdfmm/backend/cuda/p2p.hpp"

#include "backend/cuda/execution_policy.hpp"
#include "backend/cuda/fmm/internal.hpp"
#include "backend/cpu/far_field/packing.hpp"
#include "backend/cpu/m2l/schedule.hpp"
#include "backend/mkl/m2l.hpp"

namespace cdfmm {

class UniformFmm::MklM2LPlanOwner {
public:
  explicit MklM2LPlanOwner(const StaticM2LPlan& plan);
  explicit MklM2LPlanOwner(const FloatStaticM2LPlan& plan);

  [[nodiscard]] detail::mkl::M2LApplyTimings apply(
      const StaticM2LPlan& plan, const int level,
      const std::span<const double> multipoles,
      const std::span<double> locals);

  [[nodiscard]] detail::mkl::M2LApplyTimings apply(
      const FloatStaticM2LPlan& plan, const int level,
      const std::span<const float> multipoles,
      const std::span<float> locals);

  [[nodiscard]] detail::mkl::M2LStorageStatistics statistics() const noexcept;

private:
  detail::mkl::M2LExecutor executor_;
};

// These owners keep CUDA implementation types out of the public header while
// preserving one persistent plan and its allocations for the evaluator's life.
class UniformFmm::CudaM2LPlanOwner {
public:
  explicit CudaM2LPlanOwner(std::unique_ptr<CudaM2LPlan> value);
  std::unique_ptr<CudaM2LPlan> plan;
};

class UniformFmm::CudaP2PPlanOwner {
public:
  explicit CudaP2PPlanOwner(std::unique_ptr<CudaP2PPlan> value);
  std::unique_ptr<CudaP2PPlan> plan;
};

class UniformFmm::CudaFullPlanOwner {
public:
  explicit CudaFullPlanOwner(std::unique_ptr<CudaFullPlan> value);
  std::unique_ptr<CudaFullPlan> plan;
};

/** @brief CPU far-field execution packing in the plan's precision. */
class UniformFmm::CpuFarFieldPackingOwner {
public:
  detail::cpu::FarFieldPacking<double> fp64{};
  detail::cpu::FarFieldPacking<float> fp32{};
  /// Transfer-class-sorted M2L block schedule (portable M2L executor only).
  detail::cpu::M2LBlockSchedule m2l_schedule{};
};

/** @brief Resolved CUDA execution policy of one plan (see execution_policy.hpp). */
class UniformFmm::CudaExecutionPolicyOwner {
public:
  cuda_policy::CudaExecutionPolicyInputs inputs{};
  cuda_policy::CudaExecutionPolicy policy{};
};

} // namespace cdfmm
