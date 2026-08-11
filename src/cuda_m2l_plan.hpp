// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <vector>

#include "cdfmm/timings.hpp"

namespace cdfmm {

/** @brief Immutable description of one static M2L transfer class. */
struct CudaM2LGroupView {
    std::span<const double> matrix;
    std::span<const int> sources;
    std::span<const int> targets;
};

/** @brief Persistent CUDA executor for grouped static M2L matrices. */
class CudaM2LPlan {
public:
    CudaM2LPlan(int coefficient_count,
                std::span<const CudaM2LGroupView> groups);
    ~CudaM2LPlan();

    CudaM2LPlan(const CudaM2LPlan&) = delete;
    CudaM2LPlan& operator=(const CudaM2LPlan&) = delete;

    void evaluate(std::span<const std::vector<double>> multipoles,
                  std::span<std::vector<double>> raw_locals);

    [[nodiscard]] const CudaPlanStatistics& statistics() const noexcept;
    [[nodiscard]] const CudaEvaluationTimings& timings() const noexcept;

private:
    struct Implementation;
    Implementation* implementation_{nullptr};
};

} // namespace cdfmm
