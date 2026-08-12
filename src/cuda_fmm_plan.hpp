// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <string>
#include <vector>

#include "cdfmm/output_flags.hpp"
#include "cdfmm/timings.hpp"
#include "cdfmm/uniform_fmm.hpp"
#include "cdfmm/static_operators.hpp"

namespace cdfmm {

/** @brief Persistent device-resident O(N^2) direct-reference plan. */
class CudaFmmPlan {
public:
    CudaFmmPlan(std::span<const Vec3> source_positions,
                std::span<const Vec3> target_positions,
                ExecutionBackend backend = ExecutionBackend::CpuStatic);
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

//------------------------------------------------------------------------------
// Full static FMM plan
//------------------------------------------------------------------------------

/** @brief One level of dependency-ordered static coefficient applications. */
struct CudaStaticLevel {
    std::vector<StaticOperatorEntry> entries{};
};

/** @brief Immutable CPU-built operators consumed by the full CUDA plan. */
struct CudaFullPlanData {
    int coefficient_count{0};
    int node_count{0};
    int source_count{0};
    int target_count{0};
    std::vector<int> source_permutation{};
    std::vector<int> target_permutation{};
    std::vector<StaticOperatorEntry> p2m{};
    std::vector<CudaStaticLevel> m2m_levels{};
    std::vector<StaticOperatorEntry> m2l{};
    std::vector<CudaStaticLevel> l2l_levels{};
    std::vector<StaticOperatorEntry> l2p{};
    StaticP2POperator p2p{};
};

/**
 * @brief Owns a complete device-resident static CUDA FMM evaluation plan.
 *
 * All geometry and operator data are uploaded during construction. The first
 * identity map fixes self interaction metadata; changing it requires rebuilding
 * the plan. A normal field evaluation transfers only moments and final fields.
 */
class CudaFullPlan {
public:
    explicit CudaFullPlan(const CudaFullPlanData& data);
    ~CudaFullPlan();
    CudaFullPlan(const CudaFullPlan&) = delete;
    CudaFullPlan& operator=(const CudaFullPlan&) = delete;

    void evaluate(std::span<const Vec3> moments,
                  std::span<Vec3> fields,
                  std::span<const int> sorted_self_indices);
    [[nodiscard]] const CudaPlanStatistics& statistics() const noexcept;
    [[nodiscard]] const CudaEvaluationTimings& timings() const noexcept;

private:
    struct Implementation;
    Implementation* implementation_{nullptr};
};

[[nodiscard]] bool cuda_runtime_available() noexcept;
[[nodiscard]] std::string cuda_runtime_description();

} // namespace cdfmm
