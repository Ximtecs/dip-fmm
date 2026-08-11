// SPDX-License-Identifier: Apache-2.0

#include "cuda_fmm_plan.hpp"
#include "cdfmm/cuda_direct.hpp"

#include <stdexcept>

namespace cdfmm {

bool cuda_compiled() noexcept
{
    return false;
}

bool cuda_runtime_available() noexcept
{
    return false;
}

bool cuda_direct_available() noexcept
{
    return false;
}

bool cuda_farfield_available() noexcept
{
    return false;
}

bool cuda_full_available() noexcept
{
    return false;
}

std::string cuda_runtime_description()
{
    return "CUDA support is not enabled in this build";
}

CudaFmmPlan::CudaFmmPlan(std::span<const Vec3>, std::span<const Vec3>,
                         ExecutionBackend)
{
    throw std::runtime_error(
        "CUDA backend requested, but CDFMM_ENABLE_CUDA is OFF"
    );
}

CudaFmmPlan::~CudaFmmPlan() = default;

void CudaFmmPlan::evaluate(std::span<const Vec3>,
                           std::span<PotentialField>, OutputFlags,
                           std::span<const int>)
{
    throw std::runtime_error("CUDA backend is unavailable");
}

const CudaPlanStatistics& CudaFmmPlan::statistics() const noexcept
{
    static const CudaPlanStatistics empty{};
    return empty;
}

const CudaEvaluationTimings& CudaFmmPlan::evaluation_timings() const noexcept
{
    static const CudaEvaluationTimings empty{};
    return empty;
}

std::vector<PotentialField> cuda_direct_p2p_reference(
    std::span<const Vec3>, std::span<const Vec3>, std::span<const Vec3>,
    OutputFlags, std::span<const int>)
{
    throw std::runtime_error(
        "CUDA direct P2P requested, but CDFMM_ENABLE_CUDA is OFF"
    );
}

} // namespace cdfmm
