// SPDX-License-Identifier: Apache-2.0

#include "cuda_fmm_plan.hpp"
#include "cuda_m2l_plan.hpp"

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

CudaM2LPlan::CudaM2LPlan(const StaticM2LPlan&) {
  throw std::runtime_error("CUDA M2L requested, but CDFMM_ENABLE_CUDA is OFF");
}

CudaM2LPlan::CudaM2LPlan(const FloatStaticM2LPlan&) {
  throw std::runtime_error("CUDA M2L requested, but CDFMM_ENABLE_CUDA is OFF");
}

CudaM2LPlan::~CudaM2LPlan() = default;

void CudaM2LPlan::evaluate(std::span<const double>, std::span<double>) {
  throw std::runtime_error("CUDA M2L backend is unavailable");
}

void CudaM2LPlan::evaluate(std::span<const float>, std::span<float>) {
  throw std::runtime_error("CUDA M2L backend is unavailable");
}

const CudaPlanStatistics &CudaM2LPlan::statistics() const noexcept {
  static const CudaPlanStatistics empty{};
  return empty;
}

const CudaEvaluationTimings &CudaM2LPlan::timings() const noexcept {
  static const CudaEvaluationTimings empty{};
  return empty;
}

void CudaFullPlan::copy_far_fields(std::span<Vec3>) const {
  throw std::runtime_error("CUDA backend is unavailable");
}

} // namespace cdfmm
