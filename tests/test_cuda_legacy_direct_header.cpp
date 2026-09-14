// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include <type_traits>

#include "cdfmm/cuda_direct.hpp"

static_assert(std::is_class_v<cdfmm::CudaDirectPlan>);
static_assert(std::is_same_v<
              decltype(&cdfmm::cuda_direct_p2p_reference),
              std::vector<cdfmm::PotentialField> (*)(
                  std::span<const cdfmm::Vec3>,
                  std::span<const cdfmm::Vec3>,
                  std::span<const cdfmm::Vec3>,
                  cdfmm::OutputFlags,
                  std::span<const int>)>);

TEST_CASE("Legacy CUDA direct header remains compatible", "[headers]")
{
    SUCCEED();
}
