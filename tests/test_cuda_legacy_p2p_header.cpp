// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include <type_traits>

#include "cdfmm/cuda_p2p.hpp"

static_assert(std::is_class_v<cdfmm::CudaP2PPlan>);
static_assert(!std::is_copy_constructible_v<cdfmm::CudaP2PPlan>);
static_assert(!std::is_copy_assignable_v<cdfmm::CudaP2PPlan>);

TEST_CASE("Legacy CUDA P2P header remains compatible", "[headers]")
{
    SUCCEED();
}
