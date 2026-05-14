#include <catch2/catch_test_macros.hpp>

#include "cdfmm/multi_index.hpp"

using namespace cdfmm;

TEST_CASE("MultiIndexSet sizes") {
    REQUIRE(MultiIndexSet(0).size() == 1);
    REQUIRE(MultiIndexSet(1).size() == 4);
    REQUIRE(MultiIndexSet(2).size() == 10);
    REQUIRE(MultiIndexSet(3).size() == 20);
    REQUIRE(MultiIndexSet(5).size() == 56);
}
