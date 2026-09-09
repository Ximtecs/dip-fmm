// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "cdfmm/operators.hpp"
#include "cdfmm/adaptive_tree.hpp"
#include "cdfmm/rectangular_prism.hpp"
#include "cdfmm/tetrahedron.hpp"
#include "cdfmm/uniform_fmm.hpp"
#include "cdfmm/validation.hpp"

#ifdef CDFMM_USE_OPENMP
#include <omp.h>
#endif

using namespace cdfmm;

namespace {

Tetrahedron test_tetrahedron()
{
    return Tetrahedron{
        std::array<Vec3, 4>{
            Vec3{-0.18, -0.12, -0.10},
            Vec3{0.22, -0.10, -0.08},
            Vec3{-0.06, 0.24, -0.09},
            Vec3{0.02, -0.02, 0.27}}};
}

Vec3 apply_pair_tensor(const PairTensor& tensor, const Vec3& moment)
{
    return {tensor.xx * moment.x + tensor.xy * moment.y + tensor.xz * moment.z,
            tensor.xy * moment.x + tensor.yy * moment.y + tensor.yz * moment.z,
            tensor.xz * moment.x + tensor.yz * moment.y + tensor.zz * moment.z};
}

} // namespace

TEST_CASE("UniformFmm evaluates an exact tetrahedron source", "[uniform_fmm][tetrahedron]")
{
    const Tetrahedron tetrahedron = test_tetrahedron();
    const Vec3 source{0.0, 0.0, 0.0};
    const Vec3 target{0.7, -0.2, 0.3};
    const Vec3 moment{0.31, -0.27, 0.44};

    UniformFmmOptions options;
    options.backend = ExecutionBackend::CpuStatic;
    options.precision = StaticPrecision::Float64;
    options.expansion_basis = ExpansionBasis::Cartesian;
    options.expansion_order = 4;
    options.tree.max_level = 0;
    options.source_geometry = SourceGeometry::Tetrahedron;
    options.source_tetrahedra = {tetrahedron};

    UniformFmm fmm(std::vector<Vec3>{source}, std::vector<Vec3>{target},
                   options);
    const auto result = fmm.evaluate(std::vector<Vec3>{moment},
                                     OutputFlags::Field);
    const PairTensor tensor = tetrahedron_point_tensor(target - source,
                                                        tetrahedron);
    const Vec3 expected{
        tensor.xx * moment.x + tensor.xy * moment.y + tensor.xz * moment.z,
        tensor.xy * moment.x + tensor.yy * moment.y + tensor.yz * moment.z,
        tensor.xz * moment.x + tensor.yz * moment.y + tensor.zz * moment.z};
    REQUIRE(result[0].H.x == Catch::Approx(expected.x).margin(2.0e-12));
    REQUIRE(result[0].H.y == Catch::Approx(expected.y).margin(2.0e-12));
    REQUIRE(result[0].H.z == Catch::Approx(expected.z).margin(2.0e-12));
}

TEST_CASE("UniformFmm retains exact finite source self blocks with identities",
          "[uniform_fmm][p2p][self][geometry]")
{
    const std::vector<Vec3> positions{{0.0, 0.0, 0.0}};
    const std::vector<int> identities{0};

    SECTION("rectangular prism source")
    {
        const RectangularPrism prism{0.2, 0.2, 0.2};
        const Vec3 moment{prism.volume(), -2.0 * prism.volume(),
                          0.5 * prism.volume()};
        UniformFmmOptions options;
        options.backend = ExecutionBackend::CpuStatic;
        options.precision = StaticPrecision::Float64;
        options.expansion_basis = ExpansionBasis::Cartesian;
        options.expansion_order = 4;
        options.tree.max_level = 0;
        options.source_geometry = SourceGeometry::RectangularPrism;
        options.source_sizes = {prism};
        options.target_geometry = TargetGeometry::Point;
        options.near_field_source_model = SourceModel::ExactGeometry;
        options.near_field_target_model = TargetModel::Point;
        options.far_field_source_model = SourceModel::PointDipole;
        options.far_field_target_model = TargetModel::Point;
        options.fixed_target_source_indices = identities;

        UniformFmm fmm(positions, positions, options);
        const auto result = fmm.evaluate(std::vector<Vec3>{moment},
                                         OutputFlags::Field, identities);
        const Vec3 expected = apply_pair_tensor(
            rectangular_prism_point_tensor({}, prism), moment);
        REQUIRE(std::isfinite(result[0].H.x));
        REQUIRE(std::isfinite(result[0].H.y));
        REQUIRE(std::isfinite(result[0].H.z));
        REQUIRE(std::abs(result[0].H.x) + std::abs(result[0].H.y) +
                    std::abs(result[0].H.z) >
                1.0e-12);
        REQUIRE(result[0].H.x == Catch::Approx(expected.x).margin(3.0e-12));
        REQUIRE(result[0].H.y == Catch::Approx(expected.y).margin(3.0e-12));
        REQUIRE(result[0].H.z == Catch::Approx(expected.z).margin(3.0e-12));

        options.fixed_target_source_indices.reset();
        UniformFmm dynamic_map_fmm(positions, positions, options);
        const auto dynamic_result = dynamic_map_fmm.evaluate(
            std::vector<Vec3>{moment}, OutputFlags::Field, identities);
        REQUIRE(dynamic_result[0].H.x ==
                Catch::Approx(expected.x).margin(3.0e-12));
        REQUIRE(dynamic_result[0].H.y ==
                Catch::Approx(expected.y).margin(3.0e-12));
        REQUIRE(dynamic_result[0].H.z ==
                Catch::Approx(expected.z).margin(3.0e-12));
    }

    SECTION("tetrahedron source")
    {
        const Tetrahedron tetrahedron = test_tetrahedron();
        const Vec3 moment{0.31, -0.27, 0.44};
        UniformFmmOptions options;
        options.backend = ExecutionBackend::CpuStatic;
        options.precision = StaticPrecision::Float64;
        options.expansion_basis = ExpansionBasis::Cartesian;
        options.expansion_order = 4;
        options.tree.max_level = 0;
        options.source_geometry = SourceGeometry::Tetrahedron;
        options.source_tetrahedra = {tetrahedron};
        options.target_geometry = TargetGeometry::Point;
        options.near_field_source_model = SourceModel::ExactGeometry;
        options.near_field_target_model = TargetModel::Point;
        options.far_field_source_model = SourceModel::PointDipole;
        options.far_field_target_model = TargetModel::Point;
        options.fixed_target_source_indices = identities;

        UniformFmm fmm(positions, positions, options);
        const auto result = fmm.evaluate(std::vector<Vec3>{moment},
                                         OutputFlags::Field, identities);
        const Vec3 expected = apply_pair_tensor(
            tetrahedron_point_tensor({}, tetrahedron), moment);
        REQUIRE(std::isfinite(result[0].H.x));
        REQUIRE(std::isfinite(result[0].H.y));
        REQUIRE(std::isfinite(result[0].H.z));
        REQUIRE(std::abs(result[0].H.x) + std::abs(result[0].H.y) +
                    std::abs(result[0].H.z) >
                1.0e-12);
        REQUIRE(result[0].H.x == Catch::Approx(expected.x).margin(3.0e-12));
        REQUIRE(result[0].H.y == Catch::Approx(expected.y).margin(3.0e-12));
        REQUIRE(result[0].H.z == Catch::Approx(expected.z).margin(3.0e-12));

        options.fixed_target_source_indices.reset();
        UniformFmm dynamic_map_fmm(positions, positions, options);
        const auto dynamic_result = dynamic_map_fmm.evaluate(
            std::vector<Vec3>{moment}, OutputFlags::Field, identities);
        REQUIRE(dynamic_result[0].H.x ==
                Catch::Approx(expected.x).margin(3.0e-12));
        REQUIRE(dynamic_result[0].H.y ==
                Catch::Approx(expected.y).margin(3.0e-12));
        REQUIRE(dynamic_result[0].H.z ==
                Catch::Approx(expected.z).margin(3.0e-12));
    }
}

TEST_CASE("Point identity still omits singular point self interaction",
          "[uniform_fmm][p2p][self]")
{
    const std::vector<Vec3> positions{{0.0, 0.0, 0.0}};
    const std::vector<Vec3> moments{{0.31, -0.27, 0.44}};
    const std::vector<int> identities{0};
    UniformFmmOptions options;
    options.backend = ExecutionBackend::CpuStatic;
    options.precision = StaticPrecision::Float64;
    options.expansion_basis = ExpansionBasis::Cartesian;
    options.expansion_order = 4;
    options.tree.max_level = 0;
    options.fixed_target_source_indices = identities;

    UniformFmm fmm(positions, positions, options);
    const auto result = fmm.evaluate(moments, OutputFlags::Field, identities);
    REQUIRE(result[0].H.x == 0.0);
    REQUIRE(result[0].H.y == 0.0);
    REQUIRE(result[0].H.z == 0.0);
}

TEST_CASE("Point-source identities omit self for exact finite targets",
          "[uniform_fmm][p2p][self][geometry]")
{
    const std::vector<Vec3> positions{{0.0, 0.0, 0.0}};
    const std::vector<Vec3> moments{{0.31, -0.27, 0.44}};
    const std::vector<int> identities{0};
    const RectangularPrism target_prism{0.2, 0.2, 0.2};

    UniformFmmOptions options;
    options.backend = ExecutionBackend::CpuStatic;
    options.precision = StaticPrecision::Float64;
    options.expansion_basis = ExpansionBasis::Cartesian;
    options.expansion_order = 4;
    options.tree.max_level = 0;
    options.tree.root_centre = Vec3{};
    options.tree.root_half_width = 0.5;
    options.source_geometry = SourceGeometry::PointDipole;
    options.near_field_source_model = SourceModel::PointDipole;
    options.near_field_target_model = TargetModel::ExactGeometry;
    options.far_field_source_model = SourceModel::PointDipole;
    options.far_field_target_model = TargetModel::Point;
    options.target_geometry = TargetGeometry::RectangularPrism;
    options.target_sizes = {target_prism};
    options.fixed_target_source_indices = identities;

    UniformFmm fixed_map_fmm(positions, positions, options);
    const auto fixed_result =
        fixed_map_fmm.evaluate(moments, OutputFlags::Field, identities);
    REQUIRE(fixed_result[0].H.x == 0.0);
    REQUIRE(fixed_result[0].H.y == 0.0);
    REQUIRE(fixed_result[0].H.z == 0.0);

    options.fixed_target_source_indices.reset();
    UniformFmm dynamic_map_fmm(positions, positions, options);
    const auto dynamic_result =
        dynamic_map_fmm.evaluate(moments, OutputFlags::Field, identities);
    REQUIRE(dynamic_result[0].H.x == 0.0);
    REQUIRE(dynamic_result[0].H.y == 0.0);
    REQUIRE(dynamic_result[0].H.z == 0.0);
}

TEST_CASE("Point target model ignores physical tetrahedron target record",
          "[uniform_fmm][p2p][tetrahedron]")
{
    const std::vector<Vec3> sources{{0.0, 0.0, 0.0}};
    const std::vector<Vec3> targets{{0.7, -0.2, 0.3}};
    const std::vector<Vec3> moments{{0.31, -0.27, 0.44}};
    const Tetrahedron tetrahedron = test_tetrahedron();

    UniformFmmOptions point_options;
    point_options.backend = ExecutionBackend::CpuStatic;
    point_options.precision = StaticPrecision::Float64;
    point_options.expansion_basis = ExpansionBasis::Cartesian;
    point_options.expansion_order = 4;
    point_options.tree.max_level = 0;
    point_options.tree.root_centre = Vec3{0.35, 0.0, 0.15};
    point_options.tree.root_half_width = 1.0;
    point_options.source_geometry = SourceGeometry::Tetrahedron;
    point_options.source_tetrahedra = {tetrahedron};
    point_options.target_geometry = TargetGeometry::Point;
    point_options.near_field_source_model = SourceModel::ExactGeometry;
    point_options.near_field_target_model = TargetModel::Point;
    point_options.far_field_source_model = SourceModel::PointDipole;
    point_options.far_field_target_model = TargetModel::Point;

    UniformFmmOptions tetra_target_options = point_options;
    tetra_target_options.target_geometry = TargetGeometry::Tetrahedron;
    tetra_target_options.target_tetrahedra = {tetrahedron};

    UniformFmm point_plan(sources, targets, point_options);
    UniformFmm tetra_target_plan(sources, targets, tetra_target_options);
    const auto point_result = point_plan.evaluate(moments, OutputFlags::Field);
    const auto tetra_target_result =
        tetra_target_plan.evaluate(moments, OutputFlags::Field);
    REQUIRE(tetra_target_result[0].H.x ==
            Catch::Approx(point_result[0].H.x).margin(3.0e-12));
    REQUIRE(tetra_target_result[0].H.y ==
            Catch::Approx(point_result[0].H.y).margin(3.0e-12));
    REQUIRE(tetra_target_result[0].H.z ==
            Catch::Approx(point_result[0].H.z).margin(3.0e-12));
}

TEST_CASE("UniformFmm builds exact tetrahedron L2P rows", "[uniform_fmm][tetrahedron]")
{
    const Tetrahedron tetrahedron = test_tetrahedron();
    UniformFmmOptions options;
    options.backend = ExecutionBackend::CpuStatic;
    options.precision = StaticPrecision::Float64;
    options.expansion_basis = ExpansionBasis::Cartesian;
    options.expansion_order = 5;
    options.tree.max_level = 2;
    options.tree.root_centre = {0.0, 0.0, 0.0};
    // The target representative is at x=0.8 and this tetrahedron extends to
    // z=0.27, so a unit half-width root would clip its complete geometry.
    options.tree.root_half_width = 1.1;
    options.target_geometry = TargetGeometry::Tetrahedron;
    options.target_tetrahedra = {tetrahedron};

    UniformFmm fmm(std::vector<Vec3>{{-0.8, 0.0, 0.0}},
                   std::vector<Vec3>{{0.8, 0.0, 0.0}}, options);
    const auto result = fmm.evaluate(std::vector<Vec3>{{0.2, -0.1, 0.3}},
                                     OutputFlags::Field);
    REQUIRE(result.size() == 1);
    REQUIRE(std::isfinite(result[0].H.x));
    REQUIRE(std::isfinite(result[0].H.y));
    REQUIRE(std::isfinite(result[0].H.z));
    REQUIRE(fmm.static_plan_statistics().l2p_operator_bytes > 0);
}

TEST_CASE("Adaptive shared topology accepts common and permuted tetrahedra",
          "[uniform_fmm][adaptive][tetrahedron]")
{
    const std::vector<Vec3> positions{{0.75, 0.75, 0.75},
                                      {-0.75, -0.75, -0.75}};
    const Tetrahedron tetrahedron = test_tetrahedron();
    const AdaptiveTree tree(
        positions,
        AdaptiveTreeOptions{.max_particles_per_leaf = 1,
                            .max_depth = 1,
                            .root_centre = Vec3{},
                            // The positive source reaches z=1.02 including
                            // its tetrahedron offset.
                            .root_half_width = 1.1});
    REQUIRE(tree.topology().source_permutation != std::vector<int>{0, 1});

    UniformFmmOptions common;
    common.backend = ExecutionBackend::CpuStatic;
    common.precision = StaticPrecision::Float64;
    common.expansion_basis = ExpansionBasis::Cartesian;
    common.expansion_order = 3;
    common.source_geometry = SourceGeometry::Tetrahedron;
    common.source_tetrahedra = {tetrahedron};
    common.target_geometry = TargetGeometry::Point;
    common.enable_cache = false;

    UniformFmmOptions per_object = common;
    per_object.source_tetrahedra = {tetrahedron, tetrahedron};
    UniformFmm common_plan(tree.shared_topology(), common);
    UniformFmm per_object_plan(tree.shared_topology(), per_object);
    const std::vector<Vec3> moments{{0.3, -0.2, 0.4}, {-0.1, 0.5, 0.2}};
    const auto common_result = common_plan.evaluate(moments, OutputFlags::Field);
    const auto per_object_result =
        per_object_plan.evaluate(moments, OutputFlags::Field);
    REQUIRE(common_result.size() == per_object_result.size());
    for (std::size_t index = 0; index < common_result.size(); ++index) {
        REQUIRE(per_object_result[index].H.x ==
                Catch::Approx(common_result[index].H.x).margin(2.0e-12));
        REQUIRE(per_object_result[index].H.y ==
                Catch::Approx(common_result[index].H.y).margin(2.0e-12));
        REQUIRE(per_object_result[index].H.z ==
                Catch::Approx(common_result[index].H.z).margin(2.0e-12));
    }
}

TEST_CASE("Exact near and point far prism models are independently selectable",
          "[uniform_fmm][models]")
{
    const std::vector<Vec3> positions{{-0.75, -0.75, -0.75},
                                      {0.75, 0.75, 0.75}};
    UniformFmmOptions options;
    options.backend = ExecutionBackend::CpuStatic;
    options.precision = StaticPrecision::Float64;
    options.expansion_basis = ExpansionBasis::Cartesian;
    options.expansion_order = 4;
    options.tree.max_level = 2;
    options.tree.root_centre = {0.0, 0.0, 0.0};
    options.tree.root_half_width = 1.0;
    options.source_geometry = SourceGeometry::RectangularPrism;
    options.target_geometry = TargetGeometry::RectangularPrism;
    options.source_sizes = {{0.12, 0.10, 0.08}};
    options.target_sizes = {{0.09, 0.11, 0.07}};
    options.near_field_source_model = SourceModel::ExactGeometry;
    options.near_field_target_model = TargetModel::ExactGeometry;
    options.far_field_source_model = SourceModel::PointDipole;
    options.far_field_target_model = TargetModel::Point;

    std::ostringstream output;
    std::streambuf* previous = std::cout.rdbuf(output.rdbuf());
    const UniformFmm fmm(positions, positions, options);
    std::cout.rdbuf(previous);
    REQUIRE(output.str().find("near_field_source_model: exact_geometry") !=
            std::string::npos);
    REQUIRE(output.str().find("near_field_target_model: exact_geometry") !=
            std::string::npos);
    REQUIRE(output.str().find("far_field_source_model: point_dipole") !=
            std::string::npos);
    REQUIRE(output.str().find("far_field_target_model: point") !=
            std::string::npos);
}

TEST_CASE("out-of-bounds tetrahedron is rejected by an explicit root",
          "[uniform_fmm][tetrahedron]")
{
    Tetrahedron tetrahedron = test_tetrahedron();
    tetrahedron.vertices[0].x = -0.8;
    UniformFmmOptions options;
    options.source_geometry = SourceGeometry::Tetrahedron;
    options.source_tetrahedra = {tetrahedron};
    options.tree.root_centre = {};
    options.tree.root_half_width = 0.5;
    REQUIRE_THROWS_AS(UniformFmm(std::vector<Vec3>{{0.0, 0.0, 0.0}}, options),
                    std::invalid_argument);
}

TEST_CASE("FMM initialisation reports requested and resolved options")
{
    const std::vector<Vec3> positions{{-0.25, 0.0, 0.0},
                                      {0.25, 0.0, 0.0}};
    UniformFmmOptions options;
    options.expansion_order = 5;
    options.tree.max_level = 2;
    options.backend = ExecutionBackend::Auto;

    std::ostringstream output;
    std::streambuf* previous = std::cout.rdbuf(output.rdbuf());
    {
        UniformFmm fmm(positions, positions, options);
    }
    std::cout.rdbuf(previous);

    REQUIRE(output.str().find("[cdfmm] UniformFmm initialisation") !=
            std::string::npos);
    REQUIRE(output.str().find("expansion_order: 5") != std::string::npos);
    REQUIRE(output.str().find("backend.requested: auto") !=
            std::string::npos);
    REQUIRE(output.str().find("backend.resolved: cpu_static") !=
            std::string::npos);
    REQUIRE(output.str().find("p2p_packing: particle_row_soa") !=
            std::string::npos);
}

TEST_CASE("automatic FMM execution resolves to a truthful CPU backend")
{
    const std::vector<Vec3> positions{{-0.25, 0.0, 0.0}, {0.25, 0.0, 0.0}};
    UniformFmm fmm(positions);
    REQUIRE(fmm.backend() == ExecutionBackend::CpuStatic);
    REQUIRE(fmm.precision() == StaticPrecision::Float32);
    REQUIRE(fmm.p2p_execution_packing() == P2PExecutionPacking::ParticleRowSoa);
}

TEST_CASE("cuboid FMM includes finite centre self field", "[uniform_fmm][cuboid]")
{
    const std::vector<Vec3> positions{{0.0, 0.0, 0.0}};
    const CuboidSize cube{0.2, 0.2, 0.2};
    const double volume = cube.volume();
    const std::vector<Vec3> moments{{volume, -2.0 * volume, 0.5 * volume}};
    UniformFmmOptions options;
    options.expansion_basis = ExpansionBasis::Cartesian;
    options.precision = StaticPrecision::Float64;
    options.expansion_order = 5;
    options.tree.max_level = 0;
    options.source_geometry = SourceGeometry::RectangularPrism;
    options.source_sizes = {cube};
    options.fixed_target_source_indices = std::vector<int>{0};
    UniformFmm fmm(positions, positions, options);
    const std::vector<int> identities{0};
    const auto result = fmm.evaluate(moments, OutputFlags::Field, identities);
    REQUIRE(result[0].H.x == Catch::Approx(-1.0 / 3.0).margin(2.0e-13));
    REQUIRE(result[0].H.y == Catch::Approx(2.0 / 3.0).margin(2.0e-13));
    REQUIRE(result[0].H.z == Catch::Approx(-1.0 / 6.0).margin(2.0e-13));

    // Disabling cuboid P2M must not change the exact cuboid near field.
    options.far_field_source_model = SourceModel::PointDipole;
    UniformFmm point_p2m_fmm(positions, positions, options);
    const auto point_p2m_result =
        point_p2m_fmm.evaluate(moments, OutputFlags::Field, identities);
    REQUIRE(point_p2m_result[0].H.x ==
            Catch::Approx(-1.0 / 3.0).margin(2.0e-13));
    REQUIRE(point_p2m_result[0].H.y ==
            Catch::Approx(2.0 / 3.0).margin(2.0e-13));
    REQUIRE(point_p2m_result[0].H.z ==
            Catch::Approx(-1.0 / 6.0).margin(2.0e-13));
}

TEST_CASE("cuboid FMM converges to exact dense direct", "[uniform_fmm][cuboid]")
{
    std::vector<Vec3> positions;
    std::vector<Vec3> moments;
    const CuboidSize cube{0.08, 0.08, 0.08};
    for (int iz = 0; iz < 3; ++iz) {
        for (int iy = 0; iy < 3; ++iy) {
            for (int ix = 0; ix < 3; ++ix) {
                positions.push_back({0.2 * ix, 0.2 * iy, 0.2 * iz});
                moments.push_back({cube.volume() * (0.3 + 0.07 * ix),
                                   cube.volume() * (-0.2 + 0.05 * iy),
                                   cube.volume() * (0.4 - 0.03 * iz)});
            }
        }
    }
    const DenseDirectPlan direct(
        positions, positions, SourceGeometry::RectangularPrism,
        TargetGeometry::Point, std::span<const CuboidSize>(&cube, 1));
    const auto reference = direct.evaluate(moments, DenseDirectBackend::Portable);
    const auto error_at_order = [&](const int order,
                           const bool use_exact_source_model) {
        UniformFmmOptions options;
        options.expansion_basis = ExpansionBasis::Cartesian;
        options.expansion_order = order;
        options.tree.max_level = 2;
        options.tree.root_centre = {0.2, 0.2, 0.2};
        // The explicit root encloses the complete cuboids, not only centres.
        options.tree.root_half_width = 0.24;
        options.source_geometry = SourceGeometry::RectangularPrism;
        options.source_sizes = {cube};
        options.far_field_source_model = use_exact_source_model
            ? SourceModel::ExactGeometry : SourceModel::PointDipole;
        UniformFmm fmm(positions, positions, options);
        const auto approximate = fmm.evaluate(moments);
        double difference_squared = 0.0;
        double reference_squared = 0.0;
        for (std::size_t i = 0; i < reference.size(); ++i) {
            const Vec3 difference = approximate[i].H - reference[i];
            difference_squared += dot(difference, difference);
            reference_squared += dot(reference[i], reference[i]);
        }
        return std::sqrt(difference_squared / reference_squared);
    };
    const double low_order_error = error_at_order(2, true);
    const double high_order_error = error_at_order(6, true);
    const double point_p2m_error = error_at_order(6, false);
    REQUIRE(high_order_error < 0.5 * low_order_error);
    REQUIRE(high_order_error < 1.0e-2);
    REQUIRE(std::abs(high_order_error - point_p2m_error) > 1.0e-5);
}

TEST_CASE("tetrahedron FMM converges to exact dense direct",
          "[uniform_fmm][tetrahedron]")
{
    // Two compact clusters make the diagonal cluster pairs list1 interactions
    // while the opposite clusters traverse the far-field operators.
    const std::vector<Vec3> positions{
        {-0.85, 0.00, 0.00},
        {-0.75, 0.06, 0.04},
        {0.75, -0.05, 0.08},
        {0.85, 0.01, -0.02}
    };
    const Tetrahedron tetrahedron = test_tetrahedron();
    const std::vector<Tetrahedron> source_tetrahedra(positions.size(),
                                                      tetrahedron);
    const std::vector<Tetrahedron> target_tetrahedra(positions.size(),
                                                      tetrahedron);
    const std::vector<Vec3> moments{
        {0.31, -0.27, 0.44},
        {-0.18, 0.36, 0.21},
        {0.42, 0.11, -0.29},
        {-0.37, 0.22, 0.17}
    };
    const std::vector<int> identities{0, 1, 2, 3};

    const DenseDirectPlan direct(
        positions, positions, SourceGeometry::Tetrahedron,
        TargetGeometry::Tetrahedron, {}, {}, identities,
        StaticPrecision::Float64,
        source_tetrahedra, target_tetrahedra, SourceModel::ExactGeometry,
        TargetModel::ExactGeometry);
    const std::vector<Vec3> reference = direct.evaluate(
        moments, DenseDirectBackend::Portable);
    const PairTensor self_tensor = tetrahedron_tetrahedron_tensor(
        {}, tetrahedron, tetrahedron);
    REQUIRE(std::isfinite(self_tensor.xx));
    REQUIRE(std::isfinite(self_tensor.yy));
    REQUIRE(std::isfinite(self_tensor.zz));

    const auto relative_rms_error = [&](const int order) {
        UniformFmmOptions options;
        options.backend = ExecutionBackend::CpuStatic;
        options.precision = StaticPrecision::Float64;
        options.expansion_basis = ExpansionBasis::Cartesian;
        options.expansion_order = order;
        options.tree.max_level = 2;
        options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
        options.tree.root_half_width = 1.2;
        options.source_geometry = SourceGeometry::Tetrahedron;
        options.source_tetrahedra = source_tetrahedra;
        options.target_geometry = TargetGeometry::Tetrahedron;
        options.target_tetrahedra = target_tetrahedra;
        options.near_field_source_model = SourceModel::ExactGeometry;
        options.near_field_target_model = TargetModel::ExactGeometry;
        options.far_field_source_model = SourceModel::ExactGeometry;
        options.far_field_target_model = TargetModel::ExactGeometry;
        options.fixed_target_source_indices = identities;

        UniformFmm fmm(positions, positions, options);
        if (order == 6) {
            const StaticPlanStatistics& statistics =
                fmm.static_plan_statistics();
            REQUIRE(statistics.p2p_interactions > 0);
            REQUIRE(statistics.p2p_interactions <
                    positions.size() * positions.size());
            REQUIRE(statistics.interactions > 0);
        }
        const auto approximate = fmm.evaluate(
            moments, OutputFlags::Field, identities);
        double difference_squared = 0.0;
        double reference_squared = 0.0;
        for (std::size_t index = 0; index < reference.size(); ++index) {
            const Vec3 difference = approximate[index].H - reference[index];
            difference_squared += dot(difference, difference);
            reference_squared += dot(reference[index], reference[index]);
        }
        return std::sqrt(difference_squared / reference_squared);
    };

    const double low_order_error = relative_rms_error(2);
    const double high_order_error = relative_rms_error(6);
    REQUIRE(high_order_error < low_order_error);
    REQUIRE(high_order_error < 1.0e-2);
}

TEST_CASE("static cuboid P2P reuses exact pair tensors", "[cuboid][p2p]")
{
    const std::array<Vec3, 1> positions{{{0.0, 0.0, 0.0}}};
    const std::array<CuboidSize, 1> sizes{{{0.1, 0.1, 0.1}}};
    const std::array<std::array<int, 2>, 1> interactions{{{0, 0}}};
    const auto sparse = build_static_p2p_operator(
        positions, positions, interactions, SourceGeometry::RectangularPrism, sizes);
    const PairTensor exact = build_pair_tensor(
        {}, {}, SourceGeometry::RectangularPrism, TargetGeometry::Point, sizes[0]);
    REQUIRE(sparse.blocks[0].xx == exact.xx);
    REQUIRE(sparse.blocks[0].yy == exact.yy);
    REQUIRE(sparse.blocks[0].zz == exact.zz);
}

TEST_CASE("reduced-symmetry P2P supports cuboid near fields",
          "[uniform_fmm][cuboid][p2p]") {
  std::vector<Vec3> positions;
  std::vector<Vec3> moments;
  for (int z = 0; z < 2; ++z) {
    for (int y = 0; y < 2; ++y) {
      for (int x = 0; x < 2; ++x) {
        positions.push_back({0.3 * x, 0.3 * y, 0.3 * z});
        moments.push_back({0.01 * (x + 1), -0.02 * (y + 1),
                           0.03 * (z + 1)});
      }
    }
  }
  const CuboidSize cube{0.1, 0.1, 0.1};

  const auto compare = [&](const TargetGeometry target_geometry,
                           const std::vector<CuboidSize> &target_sizes) {
    UniformFmmOptions options;
    options.precision = StaticPrecision::Float64;
    options.expansion_basis = ExpansionBasis::Cartesian;
    options.expansion_order = 5;
    options.tree.max_level = 1;
    options.tree.root_centre = {0.15, 0.15, 0.15};
    options.tree.root_half_width = 0.25;
    options.source_geometry = SourceGeometry::RectangularPrism;
    options.source_sizes = {cube};
    options.target_geometry = target_geometry;
    options.target_sizes = target_sizes;

    UniformFmm canonical(positions, positions, options);
    options.use_reduced_symmetry_p2p = true;
    UniformFmm reduced(positions, positions, options);
    REQUIRE(canonical.p2p_execution_packing() ==
            P2PExecutionPacking::ParticleRowSoa);
    REQUIRE(reduced.p2p_execution_packing() ==
            P2PExecutionPacking::TensorDictionary);

    const auto reference = canonical.evaluate(moments);
    const auto candidate = reduced.evaluate(moments);
    REQUIRE(reduced.static_plan_statistics().p2p_unique_tensors > 0);
    REQUIRE(reduced.static_plan_statistics().p2p_unique_tensors <=
            reduced.static_plan_statistics().p2p_interactions);
    for (std::size_t target = 0; target < positions.size(); ++target) {
      REQUIRE(candidate[target].H.x ==
              Catch::Approx(reference[target].H.x).margin(2.0e-12));
      REQUIRE(candidate[target].H.y ==
              Catch::Approx(reference[target].H.y).margin(2.0e-12));
      REQUIRE(candidate[target].H.z ==
              Catch::Approx(reference[target].H.z).margin(2.0e-12));
    }
  };

  compare(TargetGeometry::Point, {});
  compare(TargetGeometry::RectangularPrism, {cube});
}

TEST_CASE("prebuilt adaptive topology preserves physical cuboid geometry",
          "[uniform_fmm][adaptive][cuboid][p2p]") {
  const Vec3 root_centre{2.0, -1.0, 0.5};
  const std::vector<Vec3> positions{
      root_centre + Vec3{0.8, 0.8, -0.8},
      root_centre + Vec3{-0.8, -0.8, 0.8},
      root_centre + Vec3{0.8, -0.8, 0.8},
      root_centre + Vec3{-0.8, 0.8, -0.8},
      root_centre + Vec3{-0.8, -0.8, -0.8},
      root_centre + Vec3{0.8, 0.8, 0.8},
      root_centre + Vec3{-0.8, 0.8, 0.8},
      root_centre + Vec3{0.8, -0.8, -0.8}};
  std::vector<CuboidSize> sizes;
  std::vector<Vec3> moments;
  for (std::size_t index = 0; index < positions.size(); ++index) {
    const double side = 0.08 + 0.005 * static_cast<double>(index);
    sizes.push_back({side, 0.9 * side, 0.8 * side});
    const double volume = sizes.back().volume();
    moments.push_back({volume * (0.2 + 0.03 * index),
                       volume * (-0.4 + 0.02 * index),
                       volume * (0.1 - 0.01 * index)});
  }

  AdaptiveTreeOptions tree_options;
  tree_options.max_particles_per_leaf = 1;
  tree_options.max_depth = 1;
  tree_options.root_centre = root_centre;
  tree_options.root_half_width = 2.0;
  const AdaptiveTree tree(positions, tree_options);
  REQUIRE(tree.topology().source_permutation !=
          std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7});

  for (const TargetGeometry target_geometry :
       {TargetGeometry::Point, TargetGeometry::RectangularPrism}) {
    CAPTURE(target_geometry);
    const std::vector<CuboidSize> target_sizes =
        target_geometry == TargetGeometry::RectangularPrism
            ? sizes
            : std::vector<CuboidSize>{};
    const DenseDirectPlan direct(
        positions, positions, SourceGeometry::RectangularPrism, target_geometry,
        sizes, target_sizes, {}, StaticPrecision::Float64);
    const std::vector<Vec3> reference = direct.evaluate(moments);

    std::vector<PotentialField> ordinary;
    for (const bool reduced : {false, true}) {
      UniformFmmOptions options;
      options.backend = ExecutionBackend::CpuStatic;
      options.precision = StaticPrecision::Float64;
      options.expansion_basis = ExpansionBasis::Spherical;
      options.expansion_order = 4;
      options.source_geometry = SourceGeometry::RectangularPrism;
      options.source_sizes = sizes;
      options.target_geometry = target_geometry;
      options.target_sizes = target_sizes;
      options.fixed_target_source_indices =
          std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
      options.use_reduced_symmetry_p2p = reduced;
      options.enable_cache = false;
      UniformFmm plan(tree.shared_topology(), options);
      REQUIRE(plan.p2p_execution_packing() ==
              (reduced ? P2PExecutionPacking::TensorDictionary
                       : P2PExecutionPacking::ParticleRowSoa));
      const auto actual = plan.evaluate(moments, OutputFlags::Field);
      for (std::size_t index = 0; index < actual.size(); ++index) {
        REQUIRE(actual[index].H.x ==
                Catch::Approx(reference[index].x).margin(3.0e-12));
        REQUIRE(actual[index].H.y ==
                Catch::Approx(reference[index].y).margin(3.0e-12));
        REQUIRE(actual[index].H.z ==
                Catch::Approx(reference[index].z).margin(3.0e-12));
      }
      if (!reduced) {
        ordinary = actual;
      } else {
        REQUIRE(plan.static_plan_statistics().p2p_dictionary_tokens ==
                plan.static_plan_statistics().p2p_interactions);
        REQUIRE(plan.static_plan_statistics()
                    .p2p_dictionary_token_width_bytes > 0);
        for (std::size_t index = 0; index < actual.size(); ++index) {
          REQUIRE(actual[index].H.x ==
                  Catch::Approx(ordinary[index].H.x).margin(3.0e-12));
          REQUIRE(actual[index].H.y ==
                  Catch::Approx(ordinary[index].H.y).margin(3.0e-12));
          REQUIRE(actual[index].H.z ==
                  Catch::Approx(ordinary[index].H.z).margin(3.0e-12));
        }
      }
    }
  }

  UniformFmmOptions invalid;
  invalid.backend = ExecutionBackend::CpuStatic;
  invalid.source_geometry = SourceGeometry::RectangularPrism;
  invalid.source_sizes = sizes;
  invalid.source_sizes[0].hx = 5.0;
  REQUIRE_THROWS_AS(UniformFmm(tree.shared_topology(), invalid),
                    std::invalid_argument);
}

TEST_CASE("fixed P2P identities are optional and immutable",
          "[uniform_fmm][p2p]") {
  const std::vector<Vec3> positions{
      {-0.4, 0.0, 0.0}, {0.1, 0.2, 0.0}, {0.35, -0.15, 0.1}};
  const std::vector<Vec3> moments{
      {0.2, -0.3, 0.5}, {-0.1, 0.7, 0.4}, {0.6, 0.2, -0.5}};
  const std::vector<int> identities{0, 1, 2};

  UniformFmmOptions options;
  options.expansion_basis = ExpansionBasis::Cartesian;
  options.tree.max_level = 0;
  options.fixed_target_source_indices = identities;
  UniformFmm fmm(positions, positions, options);

  const auto implicit = fmm.evaluate(moments, OutputFlags::Field);
  const auto explicit_map =
      fmm.evaluate(moments, OutputFlags::Field, identities);
  REQUIRE(implicit.size() == explicit_map.size());
  for (std::size_t index = 0; index < implicit.size(); ++index) {
    REQUIRE(implicit[index].H.x ==
            Catch::Approx(explicit_map[index].H.x).margin(1.0e-14));
    REQUIRE(implicit[index].H.y ==
            Catch::Approx(explicit_map[index].H.y).margin(1.0e-14));
    REQUIRE(implicit[index].H.z ==
            Catch::Approx(explicit_map[index].H.z).margin(1.0e-14));
  }

  REQUIRE_THROWS_AS(
      fmm.evaluate(moments, OutputFlags::Field, std::vector<int>{-1, 1, 2}),
      std::invalid_argument);
}

namespace {

void require_coefficients_equal(std::span<const double> actual,
                                std::span<const double> expected,
                                const double scale = 1.0) {
  REQUIRE(actual.size() == expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    REQUIRE(actual[i] == Catch::Approx(expected[i]).margin(2.0e-13 * scale));
  }
}

CoeffVector direct_root_multipole(const UniformFmm &fmm,
                                  const std::vector<Vec3> &source_positions,
                                  const std::vector<Vec3> &dipole_moments) {
  return p2m_dipole(fmm.basis(), fmm.tree().root_centre(), source_positions,
                    dipole_moments);
}

const std::vector<Vec3> distributed_positions{
    {-0.83, -0.71, -0.64}, {0.76, -0.58, -0.42}, {-0.61, 0.69, -0.37},
    {0.57, 0.73, 0.66},    {-0.14, 0.22, 0.51},  {0.31, -0.19, 0.12},
    {-0.42, 0.08, -0.11},  {0.08, 0.49, -0.72}};

const std::vector<Vec3> distributed_moments{
    {0.7, -0.2, 0.1}, {-0.4, 0.8, 0.3}, {0.2, 0.1, -0.6}, {-0.3, -0.5, 0.9},
    {0.6, 0.4, -0.2}, {-0.1, 0.3, 0.5}, {0.9, -0.7, 0.2}, {-0.5, 0.2, -0.4}};

} // namespace

TEST_CASE("Uniform upward root equals direct P2M at several orders",
          "[uniform_fmm]") {
  for (const int order : {1, 2, 4, 6}) {
    UniformFmmOptions options;
    options.expansion_basis = ExpansionBasis::Cartesian;
    options.precision = StaticPrecision::Float64;
    options.expansion_order = order;
    options.tree.max_level = 3;
    options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
    options.tree.root_half_width = 1.0;

    UniformFmm fmm(distributed_positions, options);
    fmm.upward_pass(distributed_moments);

    const CoeffVector direct =
        direct_root_multipole(fmm, distributed_positions, distributed_moments);
    require_coefficients_equal(fmm.root_multipole(), direct);
  }
}

TEST_CASE("Depth-zero upward pass is direct leaf P2M", "[uniform_fmm]") {
  UniformFmmOptions options;
  options.expansion_basis = ExpansionBasis::Cartesian;
  options.precision = StaticPrecision::Float64;
  options.expansion_order = 5;
  options.tree.max_level = 0;
  UniformFmm fmm(distributed_positions, options);

  fmm.upward_pass(distributed_moments);
  const CoeffVector direct =
      direct_root_multipole(fmm, distributed_positions, distributed_moments);
  require_coefficients_equal(fmm.root_multipole(), direct);
}

TEST_CASE("One populated leaf translates through multiple levels",
          "[uniform_fmm]") {
  const std::vector<Vec3> positions{
      {-0.88, -0.84, -0.91}, {-0.79, -0.93, -0.82}, {-0.86, -0.77, -0.89}};
  const std::vector<Vec3> moments{
      {0.4, -0.2, 0.7}, {-0.5, 0.9, 0.1}, {0.3, 0.2, -0.6}};
  UniformFmmOptions options;
  options.expansion_basis = ExpansionBasis::Cartesian;
  options.precision = StaticPrecision::Float64;
  options.expansion_order = 5;
  options.tree.max_level = 3;
  options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
  options.tree.root_half_width = 1.0;
  UniformFmm fmm(positions, options);

  fmm.upward_pass(moments);
  const CoeffVector direct = direct_root_multipole(fmm, positions, moments);
  require_coefficients_equal(fmm.root_multipole(), direct);

  const auto nodes = fmm.tree().nodes();
  const auto populated_leaves =
      std::count_if(nodes.begin(), nodes.end(), [](const TreeNode &node) {
        return node.is_leaf() && node.source_count() > 0;
      });
  REQUIRE(populated_leaves == 1);
}

TEST_CASE("Every populated node agrees with direct subtree P2M",
          "[uniform_fmm]") {
  UniformFmmOptions options;
  options.expansion_basis = ExpansionBasis::Cartesian;
  options.precision = StaticPrecision::Float64;
  options.expansion_order = 4;
  options.tree.max_level = 3;
  options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
  options.tree.root_half_width = 1.0;
  UniformFmm fmm(distributed_positions, options);
  fmm.upward_pass(distributed_moments);

  std::vector<Vec3> sorted_moments(distributed_moments.size());
  for (std::size_t sorted_index = 0; sorted_index < sorted_moments.size();
       ++sorted_index) {
    sorted_moments[sorted_index] = distributed_moments[static_cast<std::size_t>(
        fmm.tree().source_permutation()[sorted_index])];
  }

  const auto sorted_positions = fmm.tree().sorted_source_positions();
  for (const TreeNode &node : fmm.tree().nodes()) {
    if (node.source_count() == 0) {
      for (const double coefficient : fmm.multipole(node.index)) {
        REQUIRE(coefficient == 0.0);
      }
      continue;
    }

    const CoeffVector direct = p2m_dipole(
        fmm.basis(), node.centre,
        sorted_positions.subspan(node.source_begin, node.source_count()),
        std::span<const Vec3>(sorted_moments)
            .subspan(node.source_begin, node.source_count()));
    require_coefficients_equal(fmm.multipole(node.index), direct);
  }
}

TEST_CASE("Upward pass is independent of source input ordering",
          "[uniform_fmm]") {
  std::vector<std::size_t> order(distributed_positions.size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 generator(741);
  std::shuffle(order.begin(), order.end(), generator);

  std::vector<Vec3> shuffled_positions;
  std::vector<Vec3> shuffled_moments;
  for (const std::size_t index : order) {
    shuffled_positions.push_back(distributed_positions[index]);
    shuffled_moments.push_back(distributed_moments[index]);
  }

  UniformFmmOptions options;
  options.expansion_basis = ExpansionBasis::Cartesian;
  options.precision = StaticPrecision::Float64;
  options.expansion_order = 4;
  options.tree.max_level = 2;
  options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
  options.tree.root_half_width = 1.0;
  UniformFmm original(distributed_positions, options);
  UniformFmm shuffled(shuffled_positions, options);
  original.upward_pass(distributed_moments);
  shuffled.upward_pass(shuffled_moments);

  require_coefficients_equal(shuffled.root_multipole(),
                             original.root_multipole());
}

TEST_CASE("Repeated upward passes replace all magnetic state",
          "[uniform_fmm]") {
  UniformFmmOptions options;
  options.expansion_basis = ExpansionBasis::Cartesian;
  options.precision = StaticPrecision::Float64;
  options.expansion_order = 4;
  options.tree.max_level = 2;
  UniformFmm fmm(distributed_positions, options);
  fmm.upward_pass(distributed_moments);

  std::vector<Vec3> zero_moments(distributed_moments.size());
  fmm.upward_pass(zero_moments);
  for (const TreeNode &node : fmm.tree().nodes()) {
    for (const double coefficient : fmm.multipole(node.index)) {
      REQUIRE(coefficient == 0.0);
    }
  }

  std::vector<Vec3> second_moments = distributed_moments;
  for (Vec3 &moment : second_moments) {
    moment = moment * -0.37;
  }
  fmm.upward_pass(second_moments);
  const CoeffVector direct =
      direct_root_multipole(fmm, distributed_positions, second_moments);
  require_coefficients_equal(fmm.root_multipole(), direct);
}

TEST_CASE("Uniform upward pass validates public input", "[uniform_fmm]") {
  UniformFmmOptions invalid_options;
  invalid_options.expansion_order = -1;
  REQUIRE_THROWS_AS(UniformFmm(distributed_positions, invalid_options),
                    std::invalid_argument);

  UniformFmm fmm(distributed_positions, UniformFmmOptions{});
  REQUIRE_THROWS_AS(
      fmm.upward_pass(std::vector<Vec3>(distributed_positions.size() - 1)),
      std::invalid_argument);
}

TEST_CASE("Hierarchical root and direct root give the same distant field",
          "[uniform_fmm]") {
  UniformFmmOptions options;
  options.expansion_basis = ExpansionBasis::Cartesian;
  options.precision = StaticPrecision::Float64;
  options.expansion_order = 6;
  options.tree.max_level = 3;
  options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
  options.tree.root_half_width = 1.0;
  UniformFmm fmm(distributed_positions, options);
  fmm.upward_pass(distributed_moments);

  const CoeffVector direct =
      direct_root_multipole(fmm, distributed_positions, distributed_moments);
  const Vec3 target{8.0, -7.0, 9.0};
  const PotentialField hierarchical_field = m2p_eval(
      fmm.basis(),
      CoeffVector(fmm.root_multipole().begin(), fmm.root_multipole().end()),
      fmm.tree().root_centre(), target, OutputFlags::Both);
  const PotentialField direct_root_field = m2p_eval(
      fmm.basis(), direct, fmm.tree().root_centre(), target, OutputFlags::Both);

  REQUIRE(hierarchical_field.phi ==
          Catch::Approx(direct_root_field.phi).margin(1.0e-15));
  REQUIRE(hierarchical_field.H.x ==
          Catch::Approx(direct_root_field.H.x).margin(1.0e-15));
  REQUIRE(hierarchical_field.H.y ==
          Catch::Approx(direct_root_field.H.y).margin(1.0e-15));
  REQUIRE(hierarchical_field.H.z ==
          Catch::Approx(direct_root_field.H.z).margin(1.0e-15));
}

TEST_CASE("Complete uniform FMM converges towards direct P2P",
          "[uniform_fmm]") {
  std::mt19937 generator(9281);
  std::uniform_real_distribution<double> distribution(-0.95, 0.95);
  std::vector<Vec3> sources(20);
  std::vector<Vec3> targets(15);
  std::vector<Vec3> moments(20);
  for (Vec3 &position : sources) {
    position = {distribution(generator), distribution(generator),
                distribution(generator)};
  }
  for (Vec3 &position : targets) {
    position = {distribution(generator), distribution(generator),
                distribution(generator)};
  }
  for (Vec3 &moment : moments) {
    moment = {distribution(generator), distribution(generator),
              distribution(generator)};
  }

  const auto direct = direct_p2p_reference(targets, sources, moments);
  std::vector<Vec3> direct_fields;
  for (const PotentialField &value : direct) {
    direct_fields.push_back(value.H);
  }

  double previous_rms = 1.0;
  for (const int order : {2, 3, 4}) {
    UniformFmmOptions options;
    options.expansion_basis = ExpansionBasis::Cartesian;
    options.expansion_order = order;
    options.backend = ExecutionBackend::CpuStatic;
    options.tree.max_level = 2;
    options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
    options.tree.root_half_width = 1.0;
    UniformFmm fmm(sources, targets, options);
    const auto approximate = fmm.evaluate(moments);
    std::vector<Vec3> approximate_fields;
    for (const PotentialField &value : approximate) {
      approximate_fields.push_back(value.H);
    }

    const ErrorMetrics metrics =
        compute_error_metrics(approximate_fields, direct_fields);
    REQUIRE(metrics.rms_relative_error < previous_rms);
    previous_rms = metrics.rms_relative_error;
  }
  REQUIRE(previous_rms < 2.0e-2);
}

TEST_CASE("Depth-zero complete evaluation is direct in every output mode",
          "[uniform_fmm]") {
  const std::vector<Vec3> targets{{0.12, -0.27, 0.34}, {-0.45, 0.16, -0.08}};
  UniformFmmOptions options;
  options.expansion_basis = ExpansionBasis::Cartesian;
  options.precision = StaticPrecision::Float64;
  options.expansion_order = 3;
  options.backend = ExecutionBackend::CpuStatic;
  options.tree.max_level = 0;
  UniformFmm fmm(distributed_positions, targets, options);

  for (const OutputFlags output :
       {OutputFlags::Field, OutputFlags::Potential, OutputFlags::Both}) {
    const auto actual = fmm.evaluate(distributed_moments, output);
    const auto direct = direct_p2p_reference(targets, distributed_positions,
                                             distributed_moments, output);
    for (std::size_t i = 0; i < targets.size(); ++i) {
      REQUIRE(actual[i].phi == Catch::Approx(direct[i].phi).margin(1.0e-14));
      REQUIRE(actual[i].H.x == Catch::Approx(direct[i].H.x).margin(1.0e-14));
      REQUIRE(actual[i].H.y == Catch::Approx(direct[i].H.y).margin(1.0e-14));
      REQUIRE(actual[i].H.z == Catch::Approx(direct[i].H.z).margin(1.0e-14));
    }
  }
}

TEST_CASE("Target permutation and repeated evaluation replace downward state",
          "[uniform_fmm]") {
  std::vector<Vec3> targets{{0.72, 0.68, 0.61},
                            {-0.77, -0.66, -0.59},
                            {0.15, -0.31, 0.47},
                            {-0.52, 0.63, -0.44}};
  UniformFmmOptions options;
  options.expansion_basis = ExpansionBasis::Cartesian;
  options.expansion_order = 4;
  options.backend = ExecutionBackend::CpuStatic;
  options.tree.max_level = 2;
  options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
  options.tree.root_half_width = 1.0;
  UniformFmm fmm(distributed_positions, targets, options);

  const auto first = fmm.evaluate(distributed_moments);
  const auto direct =
      direct_p2p_reference(targets, distributed_positions, distributed_moments);
  for (std::size_t i = 0; i < targets.size(); ++i) {
    REQUIRE(relative_error(first[i].H, direct[i].H) < 3.0e-2);
  }

  std::vector<Vec3> zero_moments(distributed_moments.size());
  const auto second = fmm.evaluate(zero_moments);
  for (const PotentialField &value : second) {
    REQUIRE(value.H.x == 0.0);
    REQUIRE(value.H.y == 0.0);
    REQUIRE(value.H.z == 0.0);
  }
  for (const TreeNode &node : fmm.tree().nodes()) {
    for (const double coefficient : fmm.local(node.index)) {
      REQUIRE(coefficient == 0.0);
    }
  }
}

TEST_CASE("Explicit source identities exclude only singular self pairs",
          "[uniform_fmm]") {
  const std::vector<Vec3> positions{
      {-0.2, 0.1, 0.3}, {0.1, 0.2, -0.3}, {0.5, -0.4, 0.2}};
  const std::vector<Vec3> moments{
      {0.4, 0.1, -0.2}, {-0.3, 0.7, 0.5}, {0.2, -0.6, 0.8}};
  UniformFmmOptions options;
  options.expansion_basis = ExpansionBasis::Cartesian;
  options.expansion_order = 3;
  options.backend = ExecutionBackend::CpuStatic;
  options.tree.max_level = 0;
  UniformFmm fmm(positions, positions, options);
  const std::vector<int> identities{0, 1, 2};

  const auto actual = fmm.evaluate(moments, OutputFlags::Both, identities);
  for (std::size_t target = 0; target < positions.size(); ++target) {
    const PotentialField expected =
        p2p_dipole_sum(positions[target], positions, moments, OutputFlags::Both,
                       static_cast<int>(target));
    REQUIRE(actual[target].phi == Catch::Approx(expected.phi).margin(1.0e-14));
    REQUIRE(actual[target].H.x == Catch::Approx(expected.H.x).margin(1.0e-14));
    REQUIRE(actual[target].H.y == Catch::Approx(expected.H.y).margin(1.0e-14));
    REQUIRE(actual[target].H.z == Catch::Approx(expected.H.z).margin(1.0e-14));
  }

  REQUIRE_THROWS_AS(
      fmm.evaluate(moments, OutputFlags::Field, std::vector<int>{0, 1}),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      fmm.evaluate(moments, OutputFlags::Field, std::vector<int>{0, 1, 4}),
      std::invalid_argument);
}

#ifdef CDFMM_USE_OPENMP
TEST_CASE("OpenMP thread counts preserve complete FMM results",
          "[uniform_fmm][openmp]") {
  UniformFmmOptions options;
  options.expansion_basis = ExpansionBasis::Cartesian;
  options.expansion_order = 4;
  options.backend = ExecutionBackend::CpuStatic;
  options.tree.max_level = 3;
  options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
  options.tree.root_half_width = 1.0;
  UniformFmm fmm(distributed_positions, distributed_positions, options);
  std::vector<int> source_identities(distributed_positions.size());
  std::iota(source_identities.begin(), source_identities.end(), 0);

  omp_set_num_threads(1);
  const auto serial = fmm.evaluate(distributed_moments, OutputFlags::Field,
                                   source_identities);
  omp_set_num_threads(std::min(4, omp_get_num_procs()));
  const auto threaded = fmm.evaluate(distributed_moments, OutputFlags::Field,
                                     source_identities);

  REQUIRE(threaded.size() == serial.size());
  for (std::size_t index = 0; index < serial.size(); ++index) {
    // Explicit source identities exclude the coincident self pair so this
    // comparison checks finite physical fields rather than matching NaNs.
    REQUIRE(std::isfinite(serial[index].H.x));
    REQUIRE(std::isfinite(serial[index].H.y));
    REQUIRE(std::isfinite(serial[index].H.z));
    REQUIRE(threaded[index].H.x ==
            Catch::Approx(serial[index].H.x).margin(1.0e-14));
    REQUIRE(threaded[index].H.y ==
            Catch::Approx(serial[index].H.y).margin(1.0e-14));
    REQUIRE(threaded[index].H.z ==
            Catch::Approx(serial[index].H.z).margin(1.0e-14));
  }
}
#endif

TEST_CASE("Evaluation timings aggregate and reset", "[uniform_fmm][timing]") {
  UniformFmmOptions options;
  options.expansion_basis = ExpansionBasis::Cartesian;
  options.backend = ExecutionBackend::CpuStatic;
  options.tree.max_level = 2;
  UniformFmm fmm(distributed_positions, distributed_positions, options);
  std::vector<PotentialField> results(distributed_positions.size());

  fmm.evaluate_into(distributed_moments, results);
  fmm.evaluate_into(distributed_moments, results);
  REQUIRE(fmm.last_timings().evaluations == 1);
  REQUIRE(fmm.last_timings().total.total_seconds > 0.0);
  REQUIRE(fmm.aggregate_timings().evaluations == 2);
  REQUIRE(fmm.tree().build_timings().total.total_seconds > 0.0);

  fmm.reset_timings();
  REQUIRE(fmm.aggregate_timings().evaluations == 0);
}
