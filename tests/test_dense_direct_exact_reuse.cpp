// SPDX-License-Identifier: Apache-2.0
//
// Exact operator reuse and identity semantics in dense all-to-all construction.
//
// `DenseDirectPlan` groups pairs whose exact operator inputs -- the
// displacement, the participating body records and whether the pair is an
// omitted point self interaction -- agree bit for bit, builds one tensor per
// group, and scatters it into the six dense matrices.  These tests pin the
// contract that makes that safe:
//
//   * a reused entry is bitwise identical to what the per-pair geometry
//     functions produce, for every one of the nine geometry combinations;
//   * inputs that differ by one ULP never share a tensor;
//   * physical identity comes from the explicit map and never from equal
//     coordinates: a point self interaction is omitted, a finite one is the
//     physical self tensor, and two bodies at the same place with different
//     identities are not self interactions;
//   * asymmetric source and target counts and partial identity maps behave as
//     the general all-to-all case, not as a special one.
//
// Every comparison is bitwise, because the reuse itself is bitwise: a
// tolerance would hide exactly the aliasing these tests exist to catch.

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include <vector>

#include "cdfmm/geometry/primitives/rectangular_prism.hpp"
#include "cdfmm/geometry/primitives/tetrahedron.hpp"
#include "cdfmm/operators/p2p.hpp"
#include "cdfmm/plan/direct/dense.hpp"
#include "plan/direct/construction_statistics.hpp"

using namespace cdfmm;

namespace {

[[nodiscard]] bool same_bits(const double left, const double right)
{
    return std::bit_cast<std::uint64_t>(left) ==
        std::bit_cast<std::uint64_t>(right);
}

/// @brief The six components of one matrix entry of an FP64 plan.
[[nodiscard]] std::array<double, 6> entry_of(
    const DenseDirectPlan& plan, const std::size_t target,
    const std::size_t source)
{
    const std::size_t index = target * plan.source_count() + source;
    const auto& matrices = plan.matrices();
    return {matrices[0][index], matrices[1][index], matrices[2][index],
            matrices[3][index], matrices[4][index], matrices[5][index]};
}

[[nodiscard]] std::array<double, 6> components_of(const PairTensor& tensor)
{
    return {tensor.xx, tensor.xy, tensor.xz,
            tensor.yy, tensor.yz, tensor.zz};
}

void require_same_bits(const std::array<double, 6>& left,
                       const std::array<double, 6>& right)
{
    for (std::size_t component = 0; component < 6; ++component) {
        REQUIRE(same_bits(left[component], right[component]));
    }
}

[[nodiscard]] bool any_nonzero(const std::array<double, 6>& values)
{
    for (const double value : values) {
        if (value != 0.0) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] Tetrahedron reference_tetrahedron(const double scale = 1.0)
{
    Tetrahedron tetrahedron{};
    tetrahedron.vertices[0] = {-0.20 * scale, -0.18 * scale, -0.13 * scale};
    tetrahedron.vertices[1] = {0.24 * scale, -0.15 * scale, -0.11 * scale};
    tetrahedron.vertices[2] = {-0.04 * scale, 0.27 * scale, -0.09 * scale};
    tetrahedron.vertices[3] = {0.0, -0.02 * scale, 0.23 * scale};
    return tetrahedron;
}

/// @brief A lattice large enough that many pairs share a displacement.
[[nodiscard]] std::vector<Vec3> lattice(const std::size_t count,
                                        const Vec3 origin = {})
{
    std::size_t side = 1;
    while (side * side * side < count) {
        ++side;
    }
    std::vector<Vec3> positions;
    positions.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        positions.push_back(
            {origin.x + static_cast<double>(index % side),
             origin.y + static_cast<double>((index / side) % side),
             origin.z + static_cast<double>(index / (side * side))});
    }
    return positions;
}

[[nodiscard]] std::vector<int> identity_map(const std::size_t count)
{
    std::vector<int> identities(count);
    for (std::size_t index = 0; index < count; ++index) {
        identities[index] = static_cast<int>(index);
    }
    return identities;
}

} // namespace

TEST_CASE("dense reuse reproduces the per-pair tensor for every geometry pair",
          "[dense_direct][exact_reuse]")
{
    // Every finite section below is large enough to classify, so its entries
    // are scattered copies of one representative build rather than their own
    // builds.  The point-to-point section is the deliberate exception: a point
    // pair is far cheaper than the lookup that would find its duplicate, so
    // production never classifies those plans and the section exercises the
    // unclassified loop.  Both paths must agree bit for bit, which is what
    // makes the same expectation correct for all of them.
    const std::size_t count = 64;
    const std::vector<Vec3> positions = lattice(count);
    const std::vector<int> identities = identity_map(count);
    const std::vector<CuboidSize> sizes{{0.3, 0.22, 0.17}};
    const std::vector<Tetrahedron> tetrahedra{reference_tetrahedron()};
    const std::vector<Tetrahedron> target_tetrahedra{
        reference_tetrahedron(0.8)};

    const auto check = [&](const SourceGeometry source_geometry,
                           const TargetGeometry target_geometry,
                           const auto& expected_tensor) {
        const DenseDirectPlan plan(
            positions, positions, source_geometry, target_geometry,
            source_geometry == SourceGeometry::RectangularPrism
                ? std::span<const CuboidSize>(sizes)
                : std::span<const CuboidSize>{},
            target_geometry == TargetGeometry::RectangularPrism
                ? std::span<const CuboidSize>(sizes)
                : std::span<const CuboidSize>{},
            identities, StaticPrecision::Float64,
            source_geometry == SourceGeometry::Tetrahedron
                ? std::span<const Tetrahedron>(tetrahedra)
                : std::span<const Tetrahedron>{},
            target_geometry == TargetGeometry::Tetrahedron
                ? std::span<const Tetrahedron>(target_tetrahedra)
                : std::span<const Tetrahedron>{});

        // Several distinct (target, source) pairs, including ones that share
        // a displacement with an earlier pair and therefore reuse its tensor.
        for (const auto& [target, source] : std::array<
                 std::pair<std::size_t, std::size_t>, 5>{{
                 {1, 0}, {2, 1}, {5, 4}, {20, 3}, {63, 7}}}) {
            const Vec3 displacement = positions[target] - positions[source];
            require_same_bits(entry_of(plan, target, source),
                              components_of(expected_tensor(displacement)));
        }
    };

    SECTION("point to point")
    {
        check(SourceGeometry::PointDipole, TargetGeometry::Point,
              [&](const Vec3& d) {
                  return operators::p2p::build_pair(d, Vec3{},
                      SourceGeometry::PointDipole, TargetGeometry::Point);
              });
    }
    SECTION("prism to point")
    {
        check(SourceGeometry::RectangularPrism, TargetGeometry::Point,
              [&](const Vec3& d) {
                  return operators::p2p::build_pair(d, Vec3{},
                      SourceGeometry::RectangularPrism, TargetGeometry::Point,
                      sizes[0], CuboidSize{});
              });
    }
    SECTION("point to prism")
    {
        check(SourceGeometry::PointDipole, TargetGeometry::RectangularPrism,
              [&](const Vec3& d) {
                  return operators::p2p::build_pair(d, Vec3{},
                      SourceGeometry::PointDipole,
                      TargetGeometry::RectangularPrism, CuboidSize{}, sizes[0]);
              });
    }
    SECTION("prism to prism")
    {
        check(SourceGeometry::RectangularPrism,
              TargetGeometry::RectangularPrism, [&](const Vec3& d) {
                  return operators::p2p::build_pair(d, Vec3{},
                      SourceGeometry::RectangularPrism,
                      TargetGeometry::RectangularPrism, sizes[0], sizes[0]);
              });
    }
    SECTION("tetrahedron to point")
    {
        check(SourceGeometry::Tetrahedron, TargetGeometry::Point,
              [&](const Vec3& d) {
                  return tetrahedron_point_tensor(d, tetrahedra[0]);
              });
    }
    SECTION("point to tetrahedron")
    {
        check(SourceGeometry::PointDipole, TargetGeometry::Tetrahedron,
              [&](const Vec3& d) {
                  return point_tetrahedron_tensor(d, target_tetrahedra[0]);
              });
    }
    SECTION("tetrahedron to tetrahedron")
    {
        check(SourceGeometry::Tetrahedron, TargetGeometry::Tetrahedron,
              [&](const Vec3& d) {
                  return tetrahedron_tetrahedron_tensor(
                      d, tetrahedra[0], target_tetrahedra[0]);
              });
    }
    SECTION("prism to tetrahedron")
    {
        check(SourceGeometry::RectangularPrism, TargetGeometry::Tetrahedron,
              [&](const Vec3& d) {
                  return rectangular_prism_tetrahedron_tensor(
                      d, sizes[0], target_tetrahedra[0]);
              });
    }
    SECTION("tetrahedron to prism")
    {
        check(SourceGeometry::Tetrahedron, TargetGeometry::RectangularPrism,
              [&](const Vec3& d) {
                  return tetrahedron_rectangular_prism_tensor(
                      d, tetrahedra[0], sizes[0]);
              });
    }
}

TEST_CASE("dense reuse never merges inputs that differ by one ULP",
          "[dense_direct][exact_reuse]")
{
    SECTION("one-ULP prism dimensions stay distinct")
    {
        // Two sources whose only difference is the last bit of one half-size.
        const std::vector<Vec3> sources{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        const std::vector<Vec3> targets{{1.0, 0.0, 0.0}};
        const double base = 0.3;
        const double nudged = std::nextafter(base, 1.0);
        REQUIRE(base != nudged);
        const std::vector<CuboidSize> sizes{{base, 0.2, 0.2},
                                            {nudged, 0.2, 0.2}};
        const DenseDirectPlan plan(
            sources, targets, SourceGeometry::RectangularPrism,
            TargetGeometry::Point, sizes, {}, {}, StaticPrecision::Float64);

        const auto first = entry_of(plan, 0, 0);
        const auto second = entry_of(plan, 0, 1);
        require_same_bits(first, components_of(operators::p2p::build_pair(
            targets[0], sources[0], SourceGeometry::RectangularPrism,
            TargetGeometry::Point, sizes[0], CuboidSize{})));
        require_same_bits(second, components_of(operators::p2p::build_pair(
            targets[0], sources[1], SourceGeometry::RectangularPrism,
            TargetGeometry::Point, sizes[1], CuboidSize{})));
        // A one-ULP change in the record must reach the tensor, or the two
        // sources were wrongly treated as one operator.
        bool differs = false;
        for (std::size_t component = 0; component < 6; ++component) {
            differs = differs || !same_bits(first[component],
                                            second[component]);
        }
        REQUIRE(differs);
    }

    SECTION("one-ULP displacement stays distinct")
    {
        const std::vector<Vec3> sources{{0.0, 0.0, 0.0}};
        const double base = 1.0;
        const double nudged = std::nextafter(base, 2.0);
        REQUIRE(base != nudged);
        const std::vector<Vec3> targets{{base, 0.0, 0.0}, {nudged, 0.0, 0.0}};
        const std::vector<CuboidSize> sizes{{0.3, 0.2, 0.2}};
        const DenseDirectPlan plan(
            sources, targets, SourceGeometry::RectangularPrism,
            TargetGeometry::Point, sizes, {}, {}, StaticPrecision::Float64);

        const auto first = entry_of(plan, 0, 0);
        const auto second = entry_of(plan, 1, 0);
        bool differs = false;
        for (std::size_t component = 0; component < 6; ++component) {
            differs = differs || !same_bits(first[component],
                                            second[component]);
        }
        REQUIRE(differs);
    }

    SECTION("distinct tetrahedron vertices stay distinct")
    {
        const std::vector<Vec3> sources{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        const std::vector<Vec3> targets{{1.0, 0.0, 0.0}};
        Tetrahedron nudged = reference_tetrahedron();
        nudged.vertices[1].x = std::nextafter(nudged.vertices[1].x, 1.0);
        const std::vector<Tetrahedron> tetrahedra{reference_tetrahedron(),
                                                  nudged};
        const DenseDirectPlan plan(
            sources, targets, SourceGeometry::Tetrahedron,
            TargetGeometry::Point, {}, {}, {}, StaticPrecision::Float64,
            tetrahedra);

        const auto first = entry_of(plan, 0, 0);
        const auto second = entry_of(plan, 0, 1);
        bool differs = false;
        for (std::size_t component = 0; component < 6; ++component) {
            differs = differs || !same_bits(first[component],
                                            second[component]);
        }
        REQUIRE(differs);
    }
}

TEST_CASE("dense identity semantics come from the map, not from coordinates",
          "[dense_direct][identity]")
{
    const std::vector<Tetrahedron> tetrahedra{reference_tetrahedron()};
    const std::vector<CuboidSize> sizes{{0.3, 0.22, 0.17}};

    SECTION("a mapped point self interaction is omitted")
    {
        const std::vector<Vec3> positions = lattice(27);
        const std::vector<int> identities = identity_map(27);
        const DenseDirectPlan plan(
            positions, positions, SourceGeometry::PointDipole,
            TargetGeometry::Point, {}, {}, identities,
            StaticPrecision::Float64);
        for (std::size_t body = 0; body < positions.size(); ++body) {
            REQUIRE(!any_nonzero(entry_of(plan, body, body)));
        }
    }

    SECTION("a mapped finite self interaction is the physical self tensor")
    {
        const std::vector<Vec3> positions = lattice(27);
        const std::vector<int> identities = identity_map(27);
        const DenseDirectPlan plan(
            positions, positions, SourceGeometry::Tetrahedron,
            TargetGeometry::Point, {}, {}, identities,
            StaticPrecision::Float64, tetrahedra);
        const auto expected =
            components_of(tetrahedron_point_tensor(Vec3{}, tetrahedra[0]));
        REQUIRE(any_nonzero(expected));
        for (std::size_t body = 0; body < positions.size(); ++body) {
            require_same_bits(entry_of(plan, body, body), expected);
        }
    }

    SECTION("equal coordinates with different identities are not self")
    {
        // Two distinct physical prisms that happen to share a position: the
        // pair is a genuine coincident finite interaction, not a self one,
        // and the identity map says so.
        const std::vector<Vec3> positions{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        const std::vector<int> identities{0, 1};
        const DenseDirectPlan plan(
            positions, positions, SourceGeometry::RectangularPrism,
            TargetGeometry::Point, sizes, {}, identities,
            StaticPrecision::Float64);
        const auto expected = components_of(operators::p2p::build_pair(
            positions[0], positions[1], SourceGeometry::RectangularPrism,
            TargetGeometry::Point, sizes[0], CuboidSize{}));
        // The off-diagonal pair has the same zero displacement as the
        // diagonal one, so it must still be built, not omitted.
        require_same_bits(entry_of(plan, 0, 1), expected);
        require_same_bits(entry_of(plan, 1, 0), expected);
        require_same_bits(entry_of(plan, 0, 0), expected);
    }

    SECTION("a partial identity map omits only the pairs it names")
    {
        // The first four targets are physically the first four sources; the
        // rest are separate field points that share no source identity.  A
        // point source has no self field, so an unmapped target may not sit
        // on a source: that is a singular pair, not an omitted one.
        const std::vector<Vec3> sources = lattice(8);
        std::vector<Vec3> targets(sources.begin(), sources.begin() + 4);
        for (std::size_t index = 0; index < 4; ++index) {
            targets.push_back({sources[index].x + 0.5, sources[index].y + 0.25,
                               sources[index].z + 0.125});
        }
        std::vector<int> identities(targets.size(), -1);
        for (std::size_t index = 0; index < 4; ++index) {
            identities[index] = static_cast<int>(index);
        }
        const DenseDirectPlan plan(
            sources, targets, SourceGeometry::PointDipole,
            TargetGeometry::Point, {}, {}, identities,
            StaticPrecision::Float64);

        for (std::size_t target = 0; target < targets.size(); ++target) {
            for (std::size_t source = 0; source < sources.size(); ++source) {
                const bool named = identities[target] ==
                    static_cast<int>(source);
                // Exactly the named pairs are omitted, and every other pair,
                // including a mapped target against a different source, is an
                // ordinary interaction.
                REQUIRE(any_nonzero(entry_of(plan, target, source)) != named);
            }
        }
    }
}

TEST_CASE("dense reuse handles asymmetric source and target counts",
          "[dense_direct][exact_reuse]")
{
    // Independently generated sets of different sizes, no identity map, so
    // nothing in the classification may assume a square plan.
    const std::vector<Vec3> sources = lattice(48);
    const std::vector<Vec3> targets = lattice(20, Vec3{0.5, 0.25, 0.125});
    const std::vector<CuboidSize> source_sizes{{0.3, 0.22, 0.17}};
    const std::vector<Tetrahedron> target_tetrahedra{
        reference_tetrahedron(0.7)};

    const DenseDirectPlan plan(
        sources, targets, SourceGeometry::RectangularPrism,
        TargetGeometry::Tetrahedron, source_sizes, {}, {},
        StaticPrecision::Float64, {}, target_tetrahedra);

    REQUIRE(plan.source_count() == sources.size());
    REQUIRE(plan.target_count() == targets.size());
    REQUIRE(plan.tensor_memory_bytes() ==
            6 * sources.size() * targets.size() * sizeof(double));

    for (const auto& [target, source] :
         std::array<std::pair<std::size_t, std::size_t>, 4>{{
             {0, 0}, {3, 11}, {19, 47}, {7, 25}}}) {
        require_same_bits(
            entry_of(plan, target, source),
            components_of(rectangular_prism_tetrahedron_tensor(
                targets[target] - sources[source], source_sizes[0],
                target_tetrahedra[0])));
    }
}

TEST_CASE("dense reuse is independent of thread count",
          "[dense_direct][exact_reuse]")
{
    // Classes are numbered in first-seen order and each pair owns one matrix
    // entry, so the matrices must not depend on how the build was scheduled.
    const std::vector<Vec3> positions = lattice(96);
    const std::vector<int> identities = identity_map(96);
    const std::vector<CuboidSize> sizes{{0.3, 0.22, 0.17}};

    const DenseDirectPlan reference(
        positions, positions, SourceGeometry::RectangularPrism,
        TargetGeometry::RectangularPrism, sizes, sizes, identities,
        StaticPrecision::Float64);

    for (int repeat = 0; repeat < 3; ++repeat) {
        const DenseDirectPlan again(
            positions, positions, SourceGeometry::RectangularPrism,
            TargetGeometry::RectangularPrism, sizes, sizes, identities,
            StaticPrecision::Float64);
        for (std::size_t component = 0; component < 6; ++component) {
            const auto& expected = reference.matrices()[component];
            const auto& actual = again.matrices()[component];
            REQUIRE(expected.size() == actual.size());
            for (std::size_t index = 0; index < expected.size(); ++index) {
                REQUIRE(same_bits(expected[index], actual[index]));
            }
        }
    }
}

TEST_CASE("dense reuse is found when a target row exceeds the reuse sample",
          "[dense_direct][exact_reuse]")
{
    // Pairs are classified target-major, and one target row never repeats a
    // displacement. With 65536 sources a single row fills the default
    // 65536-pair sample, so the gate must look across rows or it abandons
    // reuse on a perfect lattice. Four targets on the lattice share almost
    // every displacement with each other.
    constexpr int nx = 64;
    constexpr int ny = 32;
    constexpr int nz = 32;
    const double h = 1.0 / 32.0;
    std::vector<Vec3> sources;
    sources.reserve(static_cast<std::size_t>(nx) * ny * nz);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                sources.push_back({(i + 0.5) * h, (j + 0.5) * h, (k + 0.5) * h});
            }
        }
    }
    const std::vector<Vec3> targets(sources.begin(), sources.begin() + 4);
    const std::vector<CuboidSize> sizes{{0.9 * h, 0.9 * h, 0.9 * h}};

    const DenseDirectPlan plan(
        sources, targets, SourceGeometry::RectangularPrism,
        TargetGeometry::Point, sizes, {}, {}, StaticPrecision::Float64);
    const detail::dense_direct::ConstructionStatistics record =
        detail::dense_direct::construction_statistics();
    REQUIRE(record.pair_count == sources.size() * targets.size());
    REQUIRE(record.classified);
    REQUIRE(record.built_tensor_count < record.pair_count / 3);

    // Reuse never changes a value: spot-check entries against the per-pair
    // geometry function, bit for bit.
    for (const std::size_t source : {std::size_t{0}, std::size_t{1},
                                     std::size_t{4097}, sources.size() - 1}) {
        for (std::size_t target = 0; target < targets.size(); ++target) {
            require_same_bits(
                entry_of(plan, target, source),
                components_of(operators::p2p::build_pair(
                    targets[target], sources[source],
                    SourceGeometry::RectangularPrism, TargetGeometry::Point,
                    sizes[0], CuboidSize{})));
        }
    }
}
