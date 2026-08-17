// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <vector>

#include "cdfmm/timings.hpp"
#include "cdfmm/static_operators.hpp"

namespace cdfmm {

/** @brief Persistent CUDA executor for the canonical target-row M2L plan. */
class CudaM2LPlan {
public:
  explicit CudaM2LPlan(const StaticM2LPlan& plan);
  ~CudaM2LPlan();

  CudaM2LPlan(const CudaM2LPlan &) = delete;
    CudaM2LPlan& operator=(const CudaM2LPlan&) = delete;

    void evaluate(std::span<const double> multipoles,
                  std::span<double> raw_locals);

    [[nodiscard]] const CudaPlanStatistics& statistics() const noexcept;
    [[nodiscard]] const CudaEvaluationTimings& timings() const noexcept;

private:
    struct Implementation;
    Implementation* implementation_{nullptr};
};

} // namespace cdfmm
