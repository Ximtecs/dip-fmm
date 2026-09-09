// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cdfmm/tetrahedron.hpp"

using namespace cdfmm;

namespace {

Tetrahedron reference_tetrahedron()
{
    // Unit right tetrahedron, shifted so that its representative is its
    // centroid.  This gives volume 1/6 and a particularly simple moment
    // reference.
    return {{{{-0.25, -0.25, -0.25},
              {0.75, -0.25, -0.25},
              {-0.25, 0.75, -0.25},
              {-0.25, -0.25, 0.75}}}};
}

PairTensor point_tensor(const Vec3& r)
{
    const double r2 = dot(r, r);
    const double inverse_r3 = 1.0 / (std::sqrt(r2) * r2);
    const double diagonal = inverse_r3 / (4.0 * std::numbers::pi);
    const double common = 3.0 * diagonal / r2;
    return {common * r.x * r.x - diagonal,
            common * r.x * r.y,
            common * r.x * r.z,
            common * r.y * r.y - diagonal,
            common * r.y * r.z,
            common * r.z * r.z - diagonal};
}

void require_close(const PairTensor& actual, const PairTensor& expected,
                   const double epsilon = 2.0e-11)
{
    const auto close = [epsilon](const double value, const double reference) {
        return value == Catch::Approx(reference)
            .epsilon(epsilon)
            .margin(2.0e-9);
    };
    REQUIRE(close(actual.xx, expected.xx));
    REQUIRE(close(actual.xy, expected.xy));
    REQUIRE(close(actual.xz, expected.xz));
    REQUIRE(close(actual.yy, expected.yy));
    REQUIRE(close(actual.yz, expected.yz));
    REQUIRE(close(actual.zz, expected.zz));
}

} // namespace

TEST_CASE("tetrahedron volume validation and centroided moments")
{
    const Tetrahedron tetrahedron = reference_tetrahedron();
    REQUIRE(tetrahedron_volume(tetrahedron) == Catch::Approx(1.0 / 6.0));
    REQUIRE(tetrahedron.centroid_offset().x == Catch::Approx(0.0));
    REQUIRE(tetrahedron.centroid_offset().y == Catch::Approx(0.0));
    REQUIRE(tetrahedron.centroid_offset().z == Catch::Approx(0.0));

    REQUIRE(tetrahedron_averaged_monomial({0, 0, 0}, {}, tetrahedron) ==
            Catch::Approx(1.0));
    REQUIRE(tetrahedron_averaged_monomial({1, 0, 0}, {2.0, -1.0, 0.5},
                                          tetrahedron) == Catch::Approx(2.0));
    // E[(x-E[x])^2]/2 = (3/80)/2 for the unit reference tetrahedron.
    REQUIRE(tetrahedron_averaged_monomial({2, 0, 0}, {}, tetrahedron) ==
            Catch::Approx(3.0 / 160.0));

    Tetrahedron degenerate = tetrahedron;
    degenerate.vertices[3] = degenerate.vertices[2];
    REQUIRE_THROWS_AS(tetrahedron_volume(degenerate), std::invalid_argument);
}

TEST_CASE("tetrahedron point tensor has far-field and symmetry limits")
{
    const Tetrahedron tetrahedron = reference_tetrahedron();
    const PairTensor expected = point_tensor({0.0, 0.0, 30.0});
    const PairTensor actual = tetrahedron_point_tensor({0.0, 0.0, 30.0},
                                                       tetrahedron);
    require_close(actual, expected, 2.0e-5);

}

TEST_CASE("tetrahedron face orientation is permutation independent")
{
    const Tetrahedron tetrahedron = reference_tetrahedron();
    const PairTensor expected = tetrahedron_point_tensor({2.0, -1.5, 3.0},
                                                         tetrahedron);
    const std::array<int, 4> permutations[3]{
        {1, 0, 2, 3}, {2, 3, 0, 1}, {3, 1, 0, 2}};
    for (const auto& permutation : permutations) {
        Tetrahedron reordered;
        for (int i = 0; i < 4; ++i) {
            reordered.vertices[static_cast<std::size_t>(i)] =
                tetrahedron.vertices[static_cast<std::size_t>(
                    permutation[static_cast<std::size_t>(i)])];
        }
        require_close(tetrahedron_point_tensor({2.0, -1.5, 3.0}, reordered),
                      expected, 2.0e-11);
    }
}

TEST_CASE("point to tetrahedron uses reciprocal exact tensor")
{
    const Tetrahedron tetrahedron = reference_tetrahedron();
    const Vec3 displacement{1.3, -0.9, 2.4};
    require_close(point_tetrahedron_tensor(displacement, tetrahedron),
                  tetrahedron_point_tensor(displacement * -1.0, tetrahedron),
                  1.0e-13);
}

TEST_CASE("finite tetrahedron-to-tetrahedron P2P is explicit")
{
    REQUIRE_THROWS_AS(tetrahedron_tetrahedron_tensor(
                          {}, reference_tetrahedron(), reference_tetrahedron()),
                      std::domain_error);
}
