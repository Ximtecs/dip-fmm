// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <string>
#include <vector>

#include "cdfmm/output_flags.hpp"
#include "cdfmm/timings.hpp"
#include "cdfmm/uniform_fmm.hpp"

namespace cdfmm {

/** @brief Persistent device-resident plan used by CUDA execution backends. */
class CudaFmmPlan {
public:
    CudaFmmPlan(std::span<const Vec3> source_positions,
                std::span<const Vec3> target_positions,
                ExecutionBackend backend);
    ~CudaFmmPlan();

    CudaFmmPlan(const CudaFmmPlan&) = delete;
    CudaFmmPlan& operator=(const CudaFmmPlan&) = delete;

    void evaluate(std::span<const Vec3> moments,
                  std::span<PotentialField> results,
                  OutputFlags output,
                  std::span<const int> target_source_indices);

    [[nodiscard]] const CudaPlanStatistics& statistics() const noexcept;
    [[nodiscard]] const CudaEvaluationTimings& evaluation_timings() const noexcept;

private:
    struct Implementation;
    Implementation* implementation_{nullptr};
};

[[nodiscard]] bool cuda_runtime_available() noexcept;
[[nodiscard]] std::string cuda_runtime_description();

} // namespace cdfmm
