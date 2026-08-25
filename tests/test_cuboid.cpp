// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cdfmm/cuboid.hpp"
#include "cdfmm/static_operators.hpp"

using namespace cdfmm;

TEST_CASE("cuboid averaged monomials have analytical moments") {
    const CuboidSize h{2.0, 4.0, 6.0};
    const Vec3 d{1.0, -2.0, 3.0};
    REQUIRE(cuboid_averaged_monomial({0, 0, 0}, d, h) == 1.0);
    REQUIRE(cuboid_averaged_monomial({1, 0, 0}, d, h) == d.x);
    REQUIRE(cuboid_averaged_monomial({2, 0, 0}, d, h) ==
            Catch::Approx((d.x * d.x + h.hx * h.hx / 12.0) / 2.0));
    REQUIRE(cuboid_averaged_monomial({1, 1, 0}, d, h) == d.x * d.y);
}

TEST_CASE("cube self tensor includes finite demagnetising field") {
    const CuboidSize cube{1.0, 1.0, 1.0};
  const PairTensor tensor =
      build_pair_tensor({}, {}, SourceGeometry::UniformCuboid,
        TargetGeometry::VolumeAveragedCuboid, cube, cube);
    REQUIRE(tensor.xx == Catch::Approx(-1.0 / 3.0).margin(2.0e-14));
    REQUIRE(tensor.yy == Catch::Approx(-1.0 / 3.0).margin(2.0e-14));
    REQUIRE(tensor.zz == Catch::Approx(-1.0 / 3.0).margin(2.0e-14));
    REQUIRE(tensor.xy == Catch::Approx(0.0).margin(2.0e-14));
    REQUIRE(tensor.xz == Catch::Approx(0.0).margin(2.0e-14));
    REQUIRE(tensor.yz == Catch::Approx(0.0).margin(2.0e-14));
}

TEST_CASE("dense direct stores six rectangular matrices and reuses them")
{
    const std::array<Vec3, 2> sources{{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}};
    const std::array<Vec3, 3> targets{{{0.0, 0.0, 2.0}, {0.0, 1.0, 2.0},
                                      {2.0, 1.0, 3.0}}};
    const DenseDirectPlan plan(
        sources, targets, SourceGeometry::PointDipole, TargetGeometry::Point,
        {}, {}, {}, StaticPrecision::Float64);
    REQUIRE(plan.tensor_component_count() == 6);
    REQUIRE(plan.tensor_memory_bytes() == 6 * 2 * 3 * sizeof(double));
    for (const auto& matrix : plan.matrices()) {
        REQUIRE(matrix.size() == 6);
    }
    const std::array<Vec3, 2> moments{{{1.0, 2.0, 3.0}, {-2.0, 1.0, 0.5}}};
    const auto first = plan.evaluate(moments);
    const auto second = plan.evaluate(moments, DenseDirectBackend::Portable);
    REQUIRE(first[2].x == second[2].x);
    REQUIRE(first[2].y == second[2].y);
    REQUIRE(first[2].z == second[2].z);

    if (dense_direct_mkl_available()) {
        const auto mkl = plan.evaluate(moments, DenseDirectBackend::OneMkl);
        REQUIRE(mkl[2].x == Catch::Approx(first[2].x));
        REQUIRE(mkl[2].y == Catch::Approx(first[2].y));
        REQUIRE(mkl[2].z == Catch::Approx(first[2].z));
    } else {
    REQUIRE_THROWS_AS(plan.evaluate(moments, DenseDirectBackend::OneMkl),
            std::runtime_error);
    }
}

TEST_CASE("dense direct FP32 storage halves tensor memory") {
  const std::array<Vec3, 2> sources{{{0.0, 0.0, 0.0}, {1.0, -0.5, 0.25}}};
  const std::array<Vec3, 2> targets{{{0.0, 0.0, 2.0}, {2.0, 1.0, 3.0}}};
  const std::array<Vec3, 2> moments{{{1.0, 2.0, 3.0}, {-2.0, 1.0, 0.5}}};
  const DenseDirectPlan fp64(sources, targets, SourceGeometry::PointDipole,
                             TargetGeometry::Point, {}, {}, {},
                             StaticPrecision::Float64);
  const DenseDirectPlan fp32(sources, targets, SourceGeometry::PointDipole,
                             TargetGeometry::Point, {}, {}, {},
                             StaticPrecision::Float32);

    REQUIRE(fp32.static_precision() == StaticPrecision::Float32);
    REQUIRE(fp32.tensor_memory_bytes() * 2 == fp64.tensor_memory_bytes());

    const auto reference = fp64.evaluate(moments, DenseDirectBackend::Portable);
    const auto reduced = fp32.evaluate(moments, DenseDirectBackend::Portable);
    for (std::size_t target = 0; target < targets.size(); ++target) {
    REQUIRE(reduced[target].x ==
            Catch::Approx(reference[target].x).epsilon(2.0e-6));
    REQUIRE(reduced[target].y ==
            Catch::Approx(reference[target].y).epsilon(2.0e-6));
    REQUIRE(reduced[target].z ==
            Catch::Approx(reference[target].z).epsilon(2.0e-6));
    }
}

TEST_CASE("cuboid P2M and L2P use volume averaged monomials") {
    const MultiIndexSet basis(3);
    const CuboidSize h{2.0, 4.0, 6.0};
    const std::array<Vec3, 1> position{{{1.0, 2.0, 3.0}}};
    const std::array<CuboidSize, 1> sizes{{h}};
  const auto p2m = build_static_cuboid_p2m_operator(basis, {}, position, sizes);
    REQUIRE(p2m.input_size == 3);
  const auto l2p = build_static_cuboid_l2p_evaluator(basis, {}, position[0], h);
    REQUIRE(l2p.potential[basis.index({2, 0, 0})] ==
            Catch::Approx((1.0 + h.hx * h.hx / 12.0) / 2.0));
}

TEST_CASE("dense and static P2P store the same canonical cuboid tensor") {
  const std::array<Vec3, 1> sources{{{0.0, 0.0, 0.0}}};
  const std::array<Vec3, 1> targets{{{1.2, -0.7, 0.5}}};
  const std::array<CuboidSize, 1> source_sizes{{{0.8, 0.5, 1.1}}};
  const std::array<CuboidSize, 1> target_sizes{{{0.4, 0.9, 0.6}}};
  const std::array<std::array<int, 2>, 1> interactions{{{0, 0}}};
  const DenseDirectPlan dense(sources, targets, SourceGeometry::UniformCuboid,
                              TargetGeometry::VolumeAveragedCuboid,
                              source_sizes, target_sizes, {},
                              StaticPrecision::Float64);
  const StaticP2POperator sparse = build_static_p2p_operator(
      targets, sources, interactions, SourceGeometry::UniformCuboid,
      source_sizes, TargetGeometry::VolumeAveragedCuboid, target_sizes);
  const auto &matrices = dense.matrices();
  const auto &block = sparse.blocks.front();
  REQUIRE(block.xx == matrices[0][0]);
  REQUIRE(block.xy == matrices[1][0]);
  REQUIRE(block.xz == matrices[2][0]);
  REQUIRE(block.yy == matrices[3][0]);
  REQUIRE(block.yz == matrices[4][0]);
  REQUIRE(block.zz == matrices[5][0]);
}
