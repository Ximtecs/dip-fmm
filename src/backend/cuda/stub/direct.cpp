// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/backend/cuda/direct.hpp"
#include "cdfmm/backend/cuda/dense_direct.hpp"

#include <stdexcept>

namespace cdfmm {

bool cuda_direct_available() noexcept { return false; }

bool cuda_dense_direct_available() noexcept { return false; }

CudaDirectPlan::CudaDirectPlan(std::span<const Vec3>, std::span<const Vec3>,
                               std::span<const int>) {
  throw std::runtime_error(
      "CUDA backend requested, but CDFMM_ENABLE_CUDA is OFF");
}

CudaDirectPlan::~CudaDirectPlan() = default;

void CudaDirectPlan::evaluate(std::span<const Vec3>,
                              std::span<PotentialField>, OutputFlags) {
  throw std::runtime_error("CUDA backend is unavailable");
}

std::size_t CudaDirectPlan::source_count() const noexcept { return 0; }

std::size_t CudaDirectPlan::target_count() const noexcept { return 0; }

const CudaPlanStatistics &CudaDirectPlan::statistics() const noexcept {
  static const CudaPlanStatistics empty{};
  return empty;
}

const CudaEvaluationTimings &
CudaDirectPlan::evaluation_timings() const noexcept {
  static const CudaEvaluationTimings empty{};
  return empty;
}

CudaDenseDirectPlan::CudaDenseDirectPlan(
    std::span<const Vec3>, std::span<const Vec3>, SourceGeometry,
    TargetGeometry, std::span<const CuboidSize>,
    std::span<const CuboidSize>, std::span<const int>, StaticPrecision,
    std::span<const Tetrahedron>, std::span<const Tetrahedron>, SourceModel,
    TargetModel) {
  throw std::runtime_error(
      "CUDA dense direct backend requested, but CDFMM_ENABLE_CUDA is OFF");
}

CudaDenseDirectPlan::~CudaDenseDirectPlan() = default;

std::vector<Vec3> CudaDenseDirectPlan::evaluate(std::span<const Vec3>) {
  throw std::runtime_error("CUDA dense direct backend is unavailable");
}

std::size_t CudaDenseDirectPlan::source_count() const noexcept { return 0; }

std::size_t CudaDenseDirectPlan::target_count() const noexcept { return 0; }

std::size_t CudaDenseDirectPlan::tensor_memory_bytes() const noexcept {
  return 0;
}

std::size_t CudaDenseDirectPlan::persistent_device_bytes() const noexcept {
  return 0;
}

StaticPrecision CudaDenseDirectPlan::static_precision() const noexcept {
  return StaticPrecision::Float64;
}

std::vector<PotentialField> cuda_direct_p2p_reference(
    std::span<const Vec3>, std::span<const Vec3>, std::span<const Vec3>,
    OutputFlags, std::span<const int>) {
  throw std::runtime_error(
      "CUDA direct P2P requested, but CDFMM_ENABLE_CUDA is OFF");
}

} // namespace cdfmm
