// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <numbers>
#include <vector>

#include "cdfmm/periodic.hpp"
#include "cdfmm/uniform_fmm.hpp"

using namespace cdfmm;
using Catch::Approx;

TEST_CASE("periodic coordinate wrapping uses floor division")
{
    CHECK(wrap_periodic_box_coordinate(-1, 4).coordinate == 3);
    CHECK(wrap_periodic_box_coordinate(-1, 4).image_shift == -1);
    CHECK(wrap_periodic_box_coordinate(4, 4).coordinate == 0);
    CHECK(wrap_periodic_box_coordinate(4, 4).image_shift == 1);
}

TEST_CASE("periodic positions wrap into the half-open fundamental cell")
{
    PeriodicCellOptions options;
    options.enabled = true;
    options.centre = {2.0, -1.0, 0.5};
    options.lengths = {4.0, 4.0, 4.0};

    const Vec3 wrapped = wrap_periodic_position({4.5, -3.5, 4.5}, options);
    CHECK(wrapped.x == Approx(0.5));
    CHECK(wrapped.y == Approx(0.5));
    CHECK(wrapped.z == Approx(0.5));
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

TEST_CASE("zero-k0 Ewald derivatives have the cubic background Hessian")
{
    PeriodicCellOptions options;
    options.enabled = true;
    options.setup_tolerance = 1.0e-12;
    const MultiIndexSet derivatives(2);
    const auto values = periodic_laplace_derivatives_raw(derivatives, options);

    CHECK(values[derivatives.index({2, 0, 0})] ==
          Approx(1.0 / 3.0).margin(1.0e-10));
    CHECK(values[derivatives.index({0, 2, 0})] ==
          Approx(1.0 / 3.0).margin(1.0e-10));
    CHECK(values[derivatives.index({0, 0, 2})] ==
          Approx(1.0 / 3.0).margin(1.0e-10));
    CHECK(std::abs(values[derivatives.index({1, 1, 0})]) < 1.0e-12);
    CHECK(std::abs(values[derivatives.index({1, 0, 1})]) < 1.0e-12);
    CHECK(std::abs(values[derivatives.index({0, 1, 1})]) < 1.0e-12);
}

TEST_CASE("periodic point self excludes only the central image")
{
    UniformFmmOptions options;
    options.precision = StaticPrecision::Float64;
    options.expansion_order = 2;
    options.tree.max_level = 0;
    options.periodic.enabled = true;
    options.fixed_target_source_indices = std::vector<int>{0};
    const std::vector<Vec3> moments{{0.0, 0.0, 1.0}};

    for (const ExpansionBasis basis :
         {ExpansionBasis::Cartesian, ExpansionBasis::Spherical}) {
        options.expansion_basis = basis;
        UniformFmm plan({Vec3{}}, {Vec3{}}, options);
        const auto result = plan.evaluate(moments);
        CHECK(std::abs(result[0].H.x) < 1.0e-10);
        CHECK(std::abs(result[0].H.y) < 1.0e-10);
        CHECK(result[0].H.z == Approx(1.0 / 3.0).margin(1.0e-9));
    }
}

TEST_CASE("image-aware P2P retains a non-zero self image")
{
    const std::vector<Vec3> positions{Vec3{}};
    const std::vector<StaticP2PInteraction> interactions{
        {0, 0, {}, true},
        {0, 0, {0.0, 0.0, 1.0}, false},
    };
    const auto operator_map = build_static_p2p_operator(
        positions, positions, interactions);
    const std::vector<Vec3> moments{{0.0, 0.0, 1.0}};
    const std::vector<int> identities{0};
    std::vector<Vec3> field(1);
    apply_static_p2p_operator(
        operator_map, moments, field, identities);

    CHECK(std::abs(field[0].x) < 1.0e-14);
    CHECK(std::abs(field[0].y) < 1.0e-14);
    CHECK(field[0].z == Approx(1.0 / (2.0 * std::numbers::pi)));
}

TEST_CASE("periodic spherical cuboids preserve whole-cell translations")
{
    UniformFmmOptions options;
    options.precision = StaticPrecision::Float64;
    options.expansion_order = 4;
    options.expansion_basis = ExpansionBasis::Spherical;
    options.tree.max_level = 1;
    options.periodic.enabled = true;
    options.source_geometry = SourceGeometry::UniformCuboid;
    options.target_geometry = TargetGeometry::VolumeAveragedCuboid;
    options.source_sizes = {CuboidSize{0.05, 0.05, 0.05}};
    options.target_sizes = {CuboidSize{0.05, 0.05, 0.05}};

    UniformFmm wrapped({Vec3{0.45, 0.1, -0.2}},
                       {Vec3{-0.45, -0.1, 0.2}}, options);
    UniformFmm shifted({Vec3{1.45, 0.1, -0.2}},
                       {Vec3{-0.45, -0.1, 0.2}}, options);
    const std::vector<Vec3> moments{{0.2, -0.3, 0.7}};
    const auto wrapped_result = wrapped.evaluate(moments);
    const auto shifted_result = shifted.evaluate(moments);

    // Finite-cuboid tensors contain cancellation-sensitive logarithmic terms;
    // periodic wrapping can therefore expose round-off from equivalent inputs.
    CHECK(shifted_result[0].H.x ==
          Approx(wrapped_result[0].H.x).epsilon(1.0e-7));
    CHECK(shifted_result[0].H.y ==
          Approx(wrapped_result[0].H.y).epsilon(1.0e-7));
    CHECK(shifted_result[0].H.z ==
          Approx(wrapped_result[0].H.z).epsilon(1.0e-7));
}

TEST_CASE("periodic evaluation is invariant under whole-cell translation")
{
    UniformFmmOptions options;
    options.precision = StaticPrecision::Float64;
    options.expansion_order = 4;
    options.expansion_basis = ExpansionBasis::Cartesian;
    options.tree.max_level = 2;
    options.periodic.enabled = true;

    UniformFmm wrapped({Vec3{0.45, 0.1, -0.2}},
                       {Vec3{-0.45, -0.1, 0.2}}, options);
    UniformFmm shifted({Vec3{1.45, 0.1, -0.2}},
                       {Vec3{-0.45, -0.1, 0.2}}, options);
    const std::vector<Vec3> moments{{0.2, -0.3, 0.7}};
    const auto wrapped_result = wrapped.evaluate(moments);
    const auto shifted_result = shifted.evaluate(moments);
    CHECK(shifted_result[0].H.x ==
          Approx(wrapped_result[0].H.x).epsilon(1.0e-12));
    CHECK(shifted_result[0].H.y ==
          Approx(wrapped_result[0].H.y).epsilon(1.0e-12));
    CHECK(shifted_result[0].H.z ==
          Approx(wrapped_result[0].H.z).epsilon(1.0e-12));
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
