// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/cuda_direct.hpp"
#include "cdfmm/cuda_cuboid.hpp"
#include "cuda_fmm_plan.hpp"
#include "cuda_m2l_plan.hpp"
#include "cuda_p2p_plan.hpp"

#include <stdexcept>

namespace cdfmm {

bool cuda_compiled() noexcept { return false; }

bool cuda_runtime_available() noexcept { return false; }

bool cuda_direct_available() noexcept { return false; }

bool cuda_dense_direct_available() noexcept { return false; }

bool cuda_m2l_available() noexcept { return cuda_m2l_p2p_available(); }

bool cuda_m2l_p2p_available() noexcept { return false; }

bool cuda_full_available() noexcept { return false; }

std::string cuda_runtime_description() {
  return "CUDA support is not enabled in this build";
}

CudaDirectPlan::CudaDirectPlan(std::span<const Vec3>, std::span<const Vec3>,
                               std::span<const int>) {
  throw std::runtime_error(
      "CUDA backend requested, but CDFMM_ENABLE_CUDA is OFF");
}

CudaDirectPlan::~CudaDirectPlan() = default;

CudaDenseDirectPlan::CudaDenseDirectPlan(
    std::span<const Vec3>, std::span<const Vec3>, SourceGeometry,
    TargetGeometry, std::span<const CuboidSize>,
    std::span<const CuboidSize>, std::span<const int>)
{
  throw std::runtime_error(
      "CUDA dense direct backend requested, but CDFMM_ENABLE_CUDA is OFF");
}

CudaDenseDirectPlan::~CudaDenseDirectPlan() = default;

std::vector<Vec3> CudaDenseDirectPlan::evaluate(std::span<const Vec3>)
{
  throw std::runtime_error("CUDA dense direct backend is unavailable");
}

std::size_t CudaDenseDirectPlan::source_count() const noexcept { return 0; }

std::size_t CudaDenseDirectPlan::target_count() const noexcept { return 0; }

std::size_t CudaDenseDirectPlan::tensor_memory_bytes() const noexcept
{
  return 0;
}

std::size_t CudaDenseDirectPlan::persistent_device_bytes() const noexcept
{
  return 0;
}

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

CudaFullPlan::CudaFullPlan(const CudaFullPlanData &) {
  throw std::runtime_error("full CUDA FMM is unavailable in this build");
}

CudaFullPlan::~CudaFullPlan() = default;

void CudaFullPlan::evaluate(std::span<const Vec3>, std::span<Vec3>,
                            std::span<const int>) {
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

CudaM2LPlan::~CudaM2LPlan() = default;

void CudaM2LPlan::evaluate(std::span<const double>, std::span<double>) {
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

CudaP2PPlan::CudaP2PPlan(const StaticP2POperator &, std::span<const int>) {
  throw std::runtime_error("CUDA static P2P is unavailable in this build");
}

CudaP2PPlan::CudaP2PPlan(const StaticP2PCompactPlan &, std::span<const int>) {
  throw std::runtime_error("CUDA compact P2P is unavailable in this build");
}

CudaP2PPlan::CudaP2PPlan(const StaticP2PLeafPlan &, std::span<const int>) {
  throw std::runtime_error("CUDA leaf P2P is unavailable in this build");
}

CudaP2PPlan::CudaP2PPlan(const StaticP2PBsrPlan &) {
  throw std::runtime_error("CUDA BSR P2P is unavailable in this build");
}

CudaP2PPlan::CudaP2PPlan(int, int, std::span<const int>, bool) {
  throw std::runtime_error("CUDA static P2P is unavailable in this build");
}

CudaP2PPlan::~CudaP2PPlan() = default;

void CudaP2PPlan::begin_evaluate(std::span<const Vec3>, std::span<const int>) {
  throw std::runtime_error("CUDA static P2P is unavailable in this build");
}

void CudaP2PPlan::finish_evaluate(std::span<Vec3>) {
  throw std::runtime_error("CUDA static P2P is unavailable in this build");
}

void CudaP2PPlan::cancel_evaluate() noexcept {}

void CudaP2PPlan::evaluate(std::span<const Vec3>, std::span<const int>,
                           std::span<Vec3>) {
  throw std::runtime_error("CUDA static P2P is unavailable in this build");
}

const CudaPlanStatistics &CudaP2PPlan::statistics() const noexcept {
  static const CudaPlanStatistics empty{};
  return empty;
}

const CudaEvaluationTimings &CudaP2PPlan::timings() const noexcept {
  static const CudaEvaluationTimings empty{};
  return empty;
}

std::vector<PotentialField> cuda_direct_p2p_reference(std::span<const Vec3>,
                                                      std::span<const Vec3>,
                                                      std::span<const Vec3>,
                                                      OutputFlags,
                                                      std::span<const int>) {
  throw std::runtime_error(
      "CUDA direct P2P requested, but CDFMM_ENABLE_CUDA is OFF");
}

} // namespace cdfmm
