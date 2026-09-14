// SPDX-License-Identifier: Apache-2.0

#include "backend/cuda/fmm/internal.hpp"

#include <stdexcept>

namespace cdfmm {

bool cuda_compiled() noexcept { return false; }

bool cuda_runtime_available() noexcept { return false; }

bool cuda_m2l_available() noexcept { return cuda_m2l_p2p_available(); }

bool cuda_m2l_p2p_available() noexcept { return false; }

bool cuda_full_available() noexcept { return false; }

std::string cuda_runtime_description() {
  return "CUDA support is not enabled in this build";
}

CudaFullPlan::CudaFullPlan(const CudaFullPlanData &) {
  throw std::runtime_error("full CUDA FMM is unavailable in this build");
}

CudaFullPlan::CudaFullPlan(const FloatCudaFullPlanData &) {
  throw std::runtime_error("full CUDA FMM is unavailable in this build");
}

CudaFullPlan::~CudaFullPlan() = default;

void CudaFullPlan::evaluate(std::span<const Vec3>, std::span<Vec3>,
                            std::span<const int>) {
  throw std::runtime_error("full CUDA FMM is unavailable in this build");
}

void CudaFullPlan::evaluate(std::span<const FloatVec3>,
                            std::span<FloatVec3>, std::span<const int>) {
  throw std::runtime_error("full CUDA FMM is unavailable in this build");
}

const CudaPlanStatistics &CudaFullPlan::statistics() const noexcept {
  static const CudaPlanStatistics empty{};
  return empty;
}

const CudaEvaluationTimings &CudaFullPlan::timings() const noexcept {
  static const CudaEvaluationTimings empty{};
  return empty;
}

void CudaFullPlan::copy_far_fields(std::span<Vec3>) const {
  throw std::runtime_error("CUDA backend is unavailable");
}

} // namespace cdfmm
