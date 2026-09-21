// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/backend/cuda/p2p.hpp"

#include <stdexcept>

namespace cdfmm {

CudaP2PPlan::CudaP2PPlan(const StaticP2POperator &, std::span<const int>) {
  throw std::runtime_error("CUDA static P2P is unavailable in this build");
}

CudaP2PPlan::CudaP2PPlan(const StaticP2PCompactPlan &, std::span<const int>) {
  throw std::runtime_error("CUDA compact P2P is unavailable in this build");
}

CudaP2PPlan::CudaP2PPlan(const StaticP2PLeafPlan &, std::span<const int>) {
  throw std::runtime_error("CUDA leaf P2P is unavailable in this build");
}

CudaP2PPlan::CudaP2PPlan(
    const StaticP2PSignedTensorDictionaryPlan &,
    bool,
    bool) {
  throw std::runtime_error(
      "CUDA signed tensor-dictionary P2P is unavailable in this build");
}

CudaP2PPlan::CudaP2PPlan(const StaticP2PBsrPlan &) {
  throw std::runtime_error("CUDA BSR P2P is unavailable in this build");
}

CudaP2PPlan::CudaP2PPlan(const StaticFmmTopology &, StaticPrecision,
                         std::span<const int>) {
  throw std::runtime_error(
      "CUDA position-based point P2P is unavailable in this build");
}

CudaP2PPlan::CudaP2PPlan(const FloatStaticP2POperator &,
                         std::span<const int>) {
  throw std::runtime_error("CUDA static P2P is unavailable in this build");
}

CudaP2PPlan::CudaP2PPlan(const FloatStaticP2PCompactPlan &,
                         std::span<const int>) {
  throw std::runtime_error("CUDA compact P2P is unavailable in this build");
}

CudaP2PPlan::CudaP2PPlan(const FloatStaticP2PLeafPlan &,
                         std::span<const int>) {
  throw std::runtime_error("CUDA leaf P2P is unavailable in this build");
}

CudaP2PPlan::CudaP2PPlan(
    const FloatStaticP2PSignedTensorDictionaryPlan &,
    bool,
    bool) {
  throw std::runtime_error(
      "CUDA signed tensor-dictionary P2P is unavailable in this build");
}

CudaP2PPlan::CudaP2PPlan(const FloatStaticP2PBsrPlan &) {
  throw std::runtime_error("CUDA BSR P2P is unavailable in this build");
}

CudaP2PPlan::CudaP2PPlan(int, int, std::span<const int>, bool) {
  throw std::runtime_error("CUDA static P2P is unavailable in this build");
}

CudaP2PPlan::CudaP2PPlan(int, int, std::span<const int>, bool,
                         StaticPrecision) {
  throw std::runtime_error("CUDA static P2P is unavailable in this build");
}

CudaP2PPlan::~CudaP2PPlan() = default;

void CudaP2PPlan::begin_evaluate(std::span<const Vec3>, std::span<const int>) {
  throw std::runtime_error("CUDA static P2P is unavailable in this build");
}

void CudaP2PPlan::finish_evaluate(std::span<Vec3>) {
  throw std::runtime_error("CUDA static P2P is unavailable in this build");
}

void CudaP2PPlan::begin_evaluate(
    std::span<const FloatVec3>, std::span<const int>) {
  throw std::runtime_error("CUDA static P2P is unavailable in this build");
}

void CudaP2PPlan::finish_evaluate(std::span<FloatVec3>) {
  throw std::runtime_error("CUDA static P2P is unavailable in this build");
}

void CudaP2PPlan::cancel_evaluate() noexcept {}

void CudaP2PPlan::evaluate(std::span<const Vec3>, std::span<const int>,
                           std::span<Vec3>) {
  throw std::runtime_error("CUDA static P2P is unavailable in this build");
}

void CudaP2PPlan::evaluate(std::span<const FloatVec3>,
                           std::span<const int>, std::span<FloatVec3>) {
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

TimingLevel CudaP2PPlan::timing_level() const noexcept {
  return TimingLevel::Off;
}

void CudaP2PPlan::set_timing_level(TimingLevel) noexcept {}

} // namespace cdfmm
