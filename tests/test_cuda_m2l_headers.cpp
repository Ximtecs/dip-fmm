// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include <span>
#include <type_traits>

#include "cdfmm/backend/cuda/m2l.hpp"

using CudaM2LPlan = cdfmm::CudaM2LPlan;

static_assert(std::is_class_v<CudaM2LPlan>);
static_assert(!std::is_copy_constructible_v<CudaM2LPlan>);
static_assert(!std::is_copy_assignable_v<CudaM2LPlan>);

static_assert(std::is_same_v<
              decltype(static_cast<void (CudaM2LPlan::*)(
                  std::span<const double>, std::span<double>)>(
                  &CudaM2LPlan::evaluate)),
              void (CudaM2LPlan::*)(std::span<const double>,
                                     std::span<double>)>);
static_assert(std::is_same_v<
              decltype(static_cast<void (CudaM2LPlan::*)(
                  std::span<const float>, std::span<float>)>(
                  &CudaM2LPlan::evaluate)),
              void (CudaM2LPlan::*)(std::span<const float>,
                                     std::span<float>)>);

static_assert(std::is_same_v<
              decltype(&CudaM2LPlan::statistics),
              const cdfmm::CudaPlanStatistics &(CudaM2LPlan::*)() const
                  noexcept>);
static_assert(std::is_same_v<
              decltype(&CudaM2LPlan::timings),
              const cdfmm::CudaEvaluationTimings &(CudaM2LPlan::*)() const
                  noexcept>);

TEST_CASE("Canonical CUDA M2L header is self-contained", "[headers]") {
  SUCCEED();
}
