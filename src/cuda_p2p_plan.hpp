// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <span>

#include "cdfmm/static_operators.hpp"
#include "cdfmm/timings.hpp"

namespace cdfmm {

/** @brief Persistent CUDA executor for a compact static near-field tensor. */
class CudaP2PPlan {
public:
    explicit CudaP2PPlan(const StaticP2POperator& operator_map);
    ~CudaP2PPlan();
    CudaP2PPlan(const CudaP2PPlan&) = delete;
    CudaP2PPlan& operator=(const CudaP2PPlan&) = delete;

    void begin_evaluate(std::span<const Vec3> moments,
                        std::span<const int> target_source_indices);
    void finish_evaluate(std::span<Vec3> fields);
    void cancel_evaluate() noexcept;
    void evaluate(std::span<const Vec3> moments,
                  std::span<const int> target_source_indices,
                  std::span<Vec3> fields);
    [[nodiscard]] const CudaPlanStatistics& statistics() const noexcept;
    [[nodiscard]] const CudaEvaluationTimings& timings() const noexcept;

private:
    struct Implementation;
    Implementation* implementation_{nullptr};
};

} // namespace cdfmm
