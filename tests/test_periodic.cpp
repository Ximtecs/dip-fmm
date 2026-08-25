// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include "cdfmm/periodic.hpp"
#include "cdfmm/uniform_fmm.hpp"

using namespace cdfmm;

TEST_CASE("periodic coordinate wrapping uses floor division")
{
    CHECK(wrap_periodic_box_coordinate(-1, 4).coordinate == 3);
    CHECK(wrap_periodic_box_coordinate(-1, 4).image_shift == -1);
    CHECK(wrap_periodic_box_coordinate(4, 4).coordinate == 0);
    CHECK(wrap_periodic_box_coordinate(4, 4).image_shift == 1);
}

TEST_CASE("periodic list1 preserves shallow image identities")
{
    const auto list1 = build_periodic_list1(0, {0, 0, 0});
    REQUIRE(list1.size() == 27);
    for (const auto& identity : list1) {
        CHECK(identity.node == 0);
    }
    CHECK(list1.front().image_shift == std::array<int, 3>{-1, -1, -1});
    CHECK(list1.back().image_shift == std::array<int, 3>{1, 1, 1});
}

TEST_CASE("periodic list2 is the ordinary 189 child interaction set")
{
    const auto list2 = build_periodic_list2(2, {0, 0, 0});
    CHECK(list2.size() == 189);
}

TEST_CASE("periodic cell defines the FMM root")
{
    UniformFmmOptions options;
    options.periodic.enabled = true;
    options.periodic.centre = {2.0, -1.0, 0.5};
    options.periodic.lengths = {8.0, 8.0, 8.0};
    const UniformFmm plan({Vec3{2.0, -1.0, 0.5}}, options);
    CHECK(plan.tree().root_centre().x == 2.0);
    CHECK(plan.tree().root_centre().y == -1.0);
    CHECK(plan.tree().root_centre().z == 0.5);
    CHECK(plan.tree().root_half_width() == 4.0);
}

TEST_CASE("unsupported periodic cells are rejected")
{
    PeriodicCellOptions options;
    options.enabled = true;
    options.axes = {true, true, false};
    CHECK_THROWS_AS(validate_periodic_cell(options), std::invalid_argument);
    options.axes = {true, true, true};
    options.lengths = {1.0, 2.0, 1.0};
    CHECK_THROWS_AS(validate_periodic_cell(options), std::invalid_argument);
}
