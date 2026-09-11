// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cdfmm/uniform_fmm.hpp"
#include "cuda_fmm_plan.hpp"
#include "cuda_m2l_plan.hpp"
#include "cuda_p2p_plan.hpp"

namespace cdfmm {

// These owners keep CUDA implementation types out of the public header while
// preserving one persistent plan and its allocations for the evaluator's life.
class UniformFmm::CudaM2LPlanOwner {
public:
  explicit CudaM2LPlanOwner(std::unique_ptr<CudaM2LPlan> value)
      : plan(std::move(value)) {}
  std::unique_ptr<CudaM2LPlan> plan;
};

class UniformFmm::CudaP2PPlanOwner {
public:
  explicit CudaP2PPlanOwner(std::unique_ptr<CudaP2PPlan> value)
      : plan(std::move(value)) {}
  std::unique_ptr<CudaP2PPlan> plan;
};

class UniformFmm::CudaFullPlanOwner {
public:
  explicit CudaFullPlanOwner(std::unique_ptr<CudaFullPlan> value)
      : plan(std::move(value)) {}
  std::unique_ptr<CudaFullPlan> plan;
};

} // namespace cdfmm
