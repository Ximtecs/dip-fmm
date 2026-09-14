// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/backend/cuda/m2l.hpp"

#include <stdexcept>

namespace cdfmm {

CudaM2LPlan::CudaM2LPlan(const StaticM2LPlan &) {
  throw std::runtime_error("CUDA M2L requested, but CDFMM_ENABLE_CUDA is OFF");
}

CudaM2LPlan::CudaM2LPlan(const FloatStaticM2LPlan &) {
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

} // namespace cdfmm
