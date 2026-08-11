// SPDX-License-Identifier: Apache-2.0

#include "cuda_fmm_plan.hpp"

#include <stdexcept>

namespace cdfmm {

bool cuda_runtime_available() noexcept
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

} // namespace cdfmm
