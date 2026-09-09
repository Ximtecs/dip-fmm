// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cdfmm/static_operators.hpp"

namespace cdfmm {
namespace {

Tetrahedron centred_reference_tetrahedron()
{
    // Unit simplex translated by its centroid.  The representative point is
    // therefore the centroid and all four stored vertices are offsets.
    return Tetrahedron{{
        Vec3{-0.25, -0.25, -0.25},
        Vec3{0.75, -0.25, -0.25},
        Vec3{-0.25, 0.75, -0.25},
        Vec3{-0.25, -0.25, 0.75}
    }};
}

TEST_CASE("tetrahedron P2M uses exact simplex moments")
{
    const MultiIndexSet basis(3);
    const Tetrahedron tetrahedron = centred_reference_tetrahedron();
    const std::array<Vec3, 1> source_positions{{{0.7, 0.0, 0.0}}};
    const std::array<Tetrahedron, 1> source_geometry{{tetrahedron}};
    const StaticCoefficientOperator map = build_static_tetrahedron_p2m_operator(
        basis, Vec3{}, source_positions, source_geometry);
    std::vector<double> moments{1.0, 0.0, 0.0};
    std::vector<double> output(static_cast<std::size_t>(basis.size()), 0.0);

    apply_static_operator(map, moments, output);

    REQUIRE(output[static_cast<std::size_t>(basis.index({1, 0, 0}))] ==
            Catch::Approx(-1.0));
    // Var[x] for the centred unit simplex is 3/80.  The P2M coefficient is
    // -E[(0.7+x)^2]/2 for a third-degree x derivative.
    const double expected_xxx = -(0.7 * 0.7 + 3.0 / 80.0) / 2.0;
    REQUIRE(output[static_cast<std::size_t>(basis.index({3, 0, 0}))] ==
            Catch::Approx(expected_xxx));
}

TEST_CASE("tetrahedron L2P rows use exact simplex moments")
{
    const MultiIndexSet basis(3);
    const Tetrahedron tetrahedron = centred_reference_tetrahedron();
    const StaticL2PEvaluator evaluator = build_static_tetrahedron_l2p_evaluator(
        basis, Vec3{}, Vec3{0.7, 0.0, 0.0}, tetrahedron);
    const std::size_t xxx = static_cast<std::size_t>(basis.index({3, 0, 0}));

    REQUIRE(evaluator.potential[xxx] == Catch::Approx(
        (0.7 * 0.7 * 0.7 + 9.0 * 0.7 / 80.0 + 1.0 / 160.0) / 6.0));
    REQUIRE(evaluator.field[0][xxx] == Catch::Approx(
        -(0.7 * 0.7 + 3.0 / 80.0) / 2.0));
}

TEST_CASE("tetrahedron spherical static operators retain finite moments")
{
    const SphericalHarmonicBasis basis(3);
    const Tetrahedron tetrahedron = centred_reference_tetrahedron();
    const std::array<Vec3, 1> source_positions{{{0.7, 0.0, 0.0}}};
    const std::array<Tetrahedron, 1> source_geometry{{tetrahedron}};
    const StaticCoefficientOperator p2m = build_static_tetrahedron_p2m_operator(
        basis, Vec3{}, source_positions, source_geometry);
    const StaticCoefficientOperator point_p2m = build_static_p2m_operator(
        basis, Vec3{}, source_positions);
    std::vector<double> moments{1.0, 0.0, 0.0};
    std::vector<double> finite_output(static_cast<std::size_t>(basis.size()), 0.0);
    std::vector<double> point_output(static_cast<std::size_t>(basis.size()), 0.0);
    apply_static_operator(p2m, moments, finite_output);
    apply_static_operator(point_p2m, moments, point_output);

    double difference = 0.0;
    for (std::size_t mode = 0; mode < finite_output.size(); ++mode) {
        difference += std::abs(finite_output[mode] - point_output[mode]);
    }
    REQUIRE(difference > 1.0e-12);

    const StaticL2PEvaluator finite_l2p =
        build_static_tetrahedron_l2p_evaluator(
            basis, Vec3{}, Vec3{0.7, 0.0, 0.0}, tetrahedron);
    const StaticL2PEvaluator point_l2p = build_static_l2p_evaluator(
        basis, Vec3{}, Vec3{0.7, 0.0, 0.0});
    difference = 0.0;
    for (std::size_t mode = 0; mode < finite_l2p.potential.size(); ++mode) {
        difference += std::abs(finite_l2p.potential[mode] -
                               point_l2p.potential[mode]);
    }
    REQUIRE(difference > 1.0e-12);
}

TEST_CASE("static P2P dispatches tetrahedron point and point tetrahedron")
{
    const Tetrahedron tetrahedron = centred_reference_tetrahedron();
    const std::array<Tetrahedron, 1> source_geometry{{tetrahedron}};
    const std::array<Tetrahedron, 1> target_geometry{{tetrahedron}};
    const std::array<Vec3, 1> sources{{{0.0, 0.0, 0.0}}};
    const std::array<Vec3, 1> targets{{{1.7, 0.2, -0.1}}};
    const std::array<std::array<int, 2>, 1> interactions{{{{0, 0}}}};

    const StaticP2POperator tetra_point = build_static_p2p_operator(
        targets, sources, interactions, SourceGeometry::Tetrahedron,
        std::span<const RectangularPrism>{}, source_geometry,
        TargetGeometry::Point, std::span<const RectangularPrism>{},
        std::span<const Tetrahedron>{}, SourceModel::ExactGeometry,
        TargetModel::Point);
    const PairTensor expected_tetra_point = tetrahedron_point_tensor(
        targets[0] - sources[0], tetrahedron);
    REQUIRE(tetra_point.blocks[0].xx == Catch::Approx(expected_tetra_point.xx));
    REQUIRE(tetra_point.blocks[0].xy == Catch::Approx(expected_tetra_point.xy));

    const StaticP2POperator point_tetra = build_static_p2p_operator(
        targets, sources, interactions, SourceGeometry::PointDipole,
        std::span<const RectangularPrism>{}, std::span<const Tetrahedron>{},
        TargetGeometry::Tetrahedron, std::span<const RectangularPrism>{},
        target_geometry, SourceModel::PointDipole,
        TargetModel::ExactGeometry);
    const PairTensor expected_point_tetra = point_tetrahedron_tensor(
        targets[0] - sources[0], tetrahedron);
    REQUIRE(point_tetra.blocks[0].xx == Catch::Approx(expected_point_tetra.xx));
    REQUIRE(point_tetra.blocks[0].zz == Catch::Approx(expected_point_tetra.zz));
}

TEST_CASE("unsupported exact prism tetrahedron P2P fails at construction")
{
    const Tetrahedron tetrahedron = centred_reference_tetrahedron();
    const std::array<Tetrahedron, 1> tetrahedra{{tetrahedron}};
    const std::array<RectangularPrism, 1> prisms{{{0.4, 0.3, 0.2}}};
    const std::array<Vec3, 1> positions{{{0.0, 0.0, 0.0}}};
    const std::array<std::array<int, 2>, 1> interactions{{{{0, 0}}}};

    REQUIRE_THROWS_AS(build_static_p2p_operator(
                          positions, positions, interactions,
                          SourceGeometry::Tetrahedron,
                          std::span<const RectangularPrism>{}, tetrahedra,
                          TargetGeometry::RectangularPrism, prisms,
                          std::span<const Tetrahedron>{}),
                      std::invalid_argument);
}

TEST_CASE("dense pair helper rejects tetrahedral geometry explicitly")
{
    REQUIRE_THROWS_AS(build_pair_tensor(
                          Vec3{1.0, 0.0, 0.0}, Vec3{},
                          SourceGeometry::Tetrahedron, TargetGeometry::Point),
                      std::invalid_argument);
}

Vec3 apply_pair_tensor(const PairTensor& tensor, const Vec3& moment)
{
    return {
        tensor.xx * moment.x + tensor.xy * moment.y + tensor.xz * moment.z,
        tensor.xy * moment.x + tensor.yy * moment.y + tensor.yz * moment.z,
        tensor.xz * moment.x + tensor.yz * moment.y + tensor.zz * moment.z};
}

TEST_CASE("dense direct caches exact tetrahedron pairs including self",
          "[dense_direct][tetrahedron]")
{
    const Tetrahedron tetrahedron = centred_reference_tetrahedron();
    const std::array<Vec3, 2> positions{{
        {0.0, 0.0, 0.0},
        {1.8, -0.3, 0.4}
    }};
    const std::array<Tetrahedron, 2> source_tetrahedra{{
        tetrahedron,
        tetrahedron
    }};
    const std::array<Tetrahedron, 2> target_tetrahedra{{
        tetrahedron,
        tetrahedron
    }};
    const DenseDirectPlan plan(
        positions, positions, SourceGeometry::Tetrahedron,
        TargetGeometry::Tetrahedron, {}, {}, {}, StaticPrecision::Float64,
        source_tetrahedra, target_tetrahedra, SourceModel::ExactGeometry,
        TargetModel::ExactGeometry);
    const std::array<Vec3, 2> moments{{
        {0.7, -0.2, 0.4},
        {-0.3, 0.6, 0.1}
    }};
    const std::vector<Vec3> fields = plan.evaluate(
        moments, DenseDirectBackend::Portable);

    REQUIRE(plan.tensor_memory_bytes() ==
            6 * positions.size() * positions.size() * sizeof(double));
    for (std::size_t target = 0; target < positions.size(); ++target) {
        Vec3 expected{};
        for (std::size_t source = 0; source < positions.size(); ++source) {
            const PairTensor tensor = tetrahedron_tetrahedron_tensor(
                positions[target] - positions[source],
                source_tetrahedra[source], target_tetrahedra[target]);
            const Vec3 contribution = apply_pair_tensor(tensor, moments[source]);
            expected.x += contribution.x;
            expected.y += contribution.y;
            expected.z += contribution.z;
        }
        REQUIRE(std::isfinite(fields[target].x));
        REQUIRE(std::isfinite(fields[target].y));
        REQUIRE(std::isfinite(fields[target].z));
        REQUIRE(fields[target].x == Catch::Approx(expected.x).margin(2.0e-12));
        REQUIRE(fields[target].y == Catch::Approx(expected.y).margin(2.0e-12));
        REQUIRE(fields[target].z == Catch::Approx(expected.z).margin(2.0e-12));
    }

    const PairTensor self = tetrahedron_tetrahedron_tensor(
        {}, tetrahedron, tetrahedron);
    REQUIRE(self.xx + self.yy + self.zz ==
            Catch::Approx(-1.0 / tetrahedron_volume(tetrahedron))
                .margin(2.0e-10));
}

TEST_CASE("static P2P caches exact tetrahedron dispatch and retains finite self",
          "[static_p2p][tetrahedron]")
{
    const Tetrahedron tetrahedron = centred_reference_tetrahedron();
    const std::array<Vec3, 2> positions{{
        {0.0, 0.0, 0.0},
        {1.8, -0.3, 0.4}
    }};
    const std::array<Tetrahedron, 2> source_tetrahedra{{
        tetrahedron,
        tetrahedron
    }};
    const std::array<Tetrahedron, 2> target_tetrahedra{{
        tetrahedron,
        tetrahedron
    }};
    const std::array<StaticP2PInteraction, 4> interactions{{
        StaticP2PInteraction{0, 0, {}, true},
        StaticP2PInteraction{0, 1, {}, false},
        StaticP2PInteraction{1, 0, {}, false},
        StaticP2PInteraction{1, 1, {}, true}
    }};
    const StaticP2POperator operator_map = build_static_p2p_operator(
        positions, positions, interactions, SourceGeometry::Tetrahedron, {},
        source_tetrahedra, TargetGeometry::Tetrahedron, {},
        target_tetrahedra, SourceModel::ExactGeometry,
        TargetModel::ExactGeometry);

    REQUIRE(operator_map.blocks.size() == interactions.size());
    for (const StaticDipoleBlock& block : operator_map.blocks) {
        const PairTensor expected = tetrahedron_tetrahedron_tensor(
            positions[static_cast<std::size_t>(block.target)] -
                positions[static_cast<std::size_t>(block.source)],
            source_tetrahedra[static_cast<std::size_t>(block.source)],
            target_tetrahedra[static_cast<std::size_t>(block.target)]);
        REQUIRE(block.xx == Catch::Approx(expected.xx).margin(2.0e-12));
        REQUIRE(block.xy == Catch::Approx(expected.xy).margin(2.0e-12));
        REQUIRE(block.xz == Catch::Approx(expected.xz).margin(2.0e-12));
        REQUIRE(block.yy == Catch::Approx(expected.yy).margin(2.0e-12));
        REQUIRE(block.yz == Catch::Approx(expected.yz).margin(2.0e-12));
        REQUIRE(block.zz == Catch::Approx(expected.zz).margin(2.0e-12));
        REQUIRE(block.skip_for_identity == 0);
    }

    const std::array<Vec3, 2> moments{{
        {0.7, -0.2, 0.4},
        {-0.3, 0.6, 0.1}
    }};
    const std::array<int, 2> identities{{0, 1}};
    std::array<Vec3, 2> fields{};
    apply_static_p2p_operator(operator_map, moments, fields, identities);
    for (std::size_t target = 0; target < positions.size(); ++target) {
        Vec3 expected{};
        for (std::size_t source = 0; source < positions.size(); ++source) {
            const PairTensor tensor = tetrahedron_tetrahedron_tensor(
                positions[target] - positions[source],
                source_tetrahedra[source], target_tetrahedra[target]);
            const Vec3 contribution = apply_pair_tensor(tensor, moments[source]);
            expected.x += contribution.x;
            expected.y += contribution.y;
            expected.z += contribution.z;
        }
        REQUIRE(fields[target].x == Catch::Approx(expected.x).margin(2.0e-12));
        REQUIRE(fields[target].y == Catch::Approx(expected.y).margin(2.0e-12));
        REQUIRE(fields[target].z == Catch::Approx(expected.z).margin(2.0e-12));
    }
}

TEST_CASE("static P2P identity handling distinguishes finite and point sources",
          "[static_p2p][self]")
{
    const std::array<Vec3, 1> positions{{{0.0, 0.0, 0.0}}};
    const std::array<StaticP2PInteraction, 1> interactions{{
        StaticP2PInteraction{0, 0, {}, true}
    }};
    const std::array<int, 1> identities{{0}};
    const std::array<Vec3, 1> moments{{{0.7, -0.2, 0.4}}};

    const Tetrahedron tetrahedron = centred_reference_tetrahedron();
    const std::array<Tetrahedron, 1> tetrahedra{{tetrahedron}};
    const StaticP2POperator tetrahedron_to_point = build_static_p2p_operator(
        positions, positions, interactions, SourceGeometry::Tetrahedron, {},
        tetrahedra, TargetGeometry::Point, {}, std::span<const Tetrahedron>{},
        SourceModel::ExactGeometry, TargetModel::Point);
    REQUIRE(tetrahedron_to_point.blocks[0].skip_for_identity == 0);
    std::array<Vec3, 1> tetrahedron_field{};
    apply_static_p2p_operator(
        tetrahedron_to_point, moments, tetrahedron_field, identities);
    REQUIRE(std::isfinite(tetrahedron_field[0].x));
    REQUIRE(std::isfinite(tetrahedron_field[0].y));
    REQUIRE(std::isfinite(tetrahedron_field[0].z));
    REQUIRE(std::abs(tetrahedron_field[0].x) +
                std::abs(tetrahedron_field[0].y) +
                std::abs(tetrahedron_field[0].z) >
            1.0e-12);

    const std::array<RectangularPrism, 1> prisms{{{0.2, 0.2, 0.2}}};
    const StaticP2POperator prism_to_point = build_static_p2p_operator(
        positions, positions, interactions, SourceGeometry::RectangularPrism,
        prisms, std::span<const Tetrahedron>{}, TargetGeometry::Point, {},
        std::span<const Tetrahedron>{}, SourceModel::ExactGeometry,
        TargetModel::Point);
    REQUIRE(prism_to_point.blocks[0].skip_for_identity == 0);
    std::array<Vec3, 1> prism_field{};
    apply_static_p2p_operator(prism_to_point, moments, prism_field, identities);
    REQUIRE(std::isfinite(prism_field[0].x));
    REQUIRE(std::isfinite(prism_field[0].y));
    REQUIRE(std::isfinite(prism_field[0].z));
    REQUIRE(std::abs(prism_field[0].x) + std::abs(prism_field[0].y) +
                std::abs(prism_field[0].z) >
            1.0e-12);

    const StaticP2POperator point_to_tetrahedron = build_static_p2p_operator(
        positions, positions, interactions, SourceGeometry::PointDipole, {},
        std::span<const Tetrahedron>{}, TargetGeometry::Tetrahedron, {},
        tetrahedra, SourceModel::PointDipole, TargetModel::ExactGeometry);
    REQUIRE(point_to_tetrahedron.blocks[0].skip_for_identity == 1);
    std::array<Vec3, 1> point_field{};
    apply_static_p2p_operator(
        point_to_tetrahedron, moments, point_field, identities);
    REQUIRE(point_field[0].x == 0.0);
    REQUIRE(point_field[0].y == 0.0);
    REQUIRE(point_field[0].z == 0.0);
}

TEST_CASE("dense direct supports exact tetrahedron source and point self")
{
    const Tetrahedron tetrahedron = centred_reference_tetrahedron();
    const std::array<Vec3, 1> positions{{{0.0, 0.0, 0.0}}};
    const std::array<Tetrahedron, 1> tetrahedra{{tetrahedron}};
    const std::array<int, 1> identities{{0}};
    const Vec3 moment{1.2, -0.4, 0.7};
    const DenseDirectPlan plan(
        positions, positions, SourceGeometry::Tetrahedron,
        TargetGeometry::Point, std::span<const CuboidSize>{},
        std::span<const CuboidSize>{}, identities, StaticPrecision::Float64,
        tetrahedra, std::span<const Tetrahedron>{}, SourceModel::ExactGeometry,
        TargetModel::Point);

    const std::vector<Vec3> field = plan.evaluate(
        std::array<Vec3, 1>{{moment}}, DenseDirectBackend::Portable);
    const PairTensor expected = tetrahedron_point_tensor({}, tetrahedron);
    const Vec3 expected_field = apply_pair_tensor(expected, moment);
    REQUIRE(field[0].x == Catch::Approx(expected_field.x));
    REQUIRE(field[0].y == Catch::Approx(expected_field.y));
    REQUIRE(field[0].z == Catch::Approx(expected_field.z));
    REQUIRE(std::isfinite(field[0].x));
    REQUIRE(std::abs(field[0].x) + std::abs(field[0].y) +
            std::abs(field[0].z) > 1.0e-12);
}

TEST_CASE("dense direct omits identity only for effective point sources")
{
    const Tetrahedron tetrahedron = centred_reference_tetrahedron();
    const std::array<Vec3, 1> positions{{{0.0, 0.0, 0.0}}};
    const std::array<int, 1> identities{{0}};
    const std::array<Vec3, 1> moments{{{1.0, 0.0, 0.0}}};

    // A physical tetrahedron retains its finite source self contribution.
    const std::array<Tetrahedron, 1> source_tetrahedra{{tetrahedron}};
    const DenseDirectPlan finite_source(
        positions, positions, SourceGeometry::Tetrahedron,
        TargetGeometry::Point, {}, {}, identities, StaticPrecision::Float64,
        source_tetrahedra);
    const std::vector<Vec3> finite_field = finite_source.evaluate(
        moments, DenseDirectBackend::Portable);
    REQUIRE(std::isfinite(finite_field[0].x));
    REQUIRE(std::abs(finite_field[0].x) + std::abs(finite_field[0].y) +
            std::abs(finite_field[0].z) > 1.0e-12);

    // A point source has no defined self field, even when the target is an
    // exact finite tetrahedron.
    const std::array<Tetrahedron, 1> target_tetrahedra{{tetrahedron}};
    const DenseDirectPlan point_source(
        positions, positions, SourceGeometry::PointDipole,
        TargetGeometry::Tetrahedron, {}, {}, identities,
        StaticPrecision::Float64, {}, target_tetrahedra,
        SourceModel::PointDipole, TargetModel::ExactGeometry);
    const std::vector<Vec3> point_field = point_source.evaluate(
        moments, DenseDirectBackend::Portable);
    REQUIRE(point_field[0].x == 0.0);
    REQUIRE(point_field[0].y == 0.0);
    REQUIRE(point_field[0].z == 0.0);
}

} // namespace
} // namespace cdfmm
