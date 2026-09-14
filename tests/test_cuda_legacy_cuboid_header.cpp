// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include <type_traits>

#include "cdfmm/cuda_cuboid.hpp"

static_assert(std::is_class_v<cdfmm::CudaDenseDirectPlan>);
static_assert(std::is_same_v<
              decltype(&cdfmm::cuda_dense_direct_available),
              bool (*)() noexcept>);

TEST_CASE("Legacy CUDA cuboid header remains compatible", "[headers]")
{
    SUCCEED();
}
