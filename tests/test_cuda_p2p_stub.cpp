// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include "cdfmm/cuda_p2p.hpp"
#include "cdfmm/uniform_fmm.hpp"

using namespace cdfmm;

TEST_CASE("CUDA P2P stubs preserve disabled-build behavior", "[cuda]")
{
    if (cuda_compiled()) {
        SUCCEED("CUDA is compiled in; disabled-build stubs are not active");
        return;
    }

    REQUIRE_THROWS_WITH(
        CudaP2PPlan(StaticP2POperator{}),
        "CUDA static P2P is unavailable in this build");
    REQUIRE_THROWS_WITH(
        CudaP2PPlan(StaticP2PCompactPlan{}),
        "CUDA compact P2P is unavailable in this build");
    REQUIRE_THROWS_WITH(
        CudaP2PPlan(StaticP2PLeafPlan{}),
        "CUDA leaf P2P is unavailable in this build");
    REQUIRE_THROWS_WITH(
        CudaP2PPlan(StaticP2PSignedTensorDictionaryPlan{}),
        "CUDA signed tensor-dictionary P2P is unavailable in this build");
    REQUIRE_THROWS_WITH(
        CudaP2PPlan(StaticP2PBsrPlan{}),
        "CUDA BSR P2P is unavailable in this build");

    REQUIRE_THROWS_WITH(
        CudaP2PPlan(FloatStaticP2POperator{}),
        "CUDA static P2P is unavailable in this build");
    REQUIRE_THROWS_WITH(
        CudaP2PPlan(FloatStaticP2PCompactPlan{}),
        "CUDA compact P2P is unavailable in this build");
    REQUIRE_THROWS_WITH(
        CudaP2PPlan(FloatStaticP2PLeafPlan{}),
        "CUDA leaf P2P is unavailable in this build");
    REQUIRE_THROWS_WITH(
        CudaP2PPlan(FloatStaticP2PSignedTensorDictionaryPlan{}),
        "CUDA signed tensor-dictionary P2P is unavailable in this build");
    REQUIRE_THROWS_WITH(
        CudaP2PPlan(FloatStaticP2PBsrPlan{}),
        "CUDA BSR P2P is unavailable in this build");
}
