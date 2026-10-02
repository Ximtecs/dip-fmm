// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include "geometry/primitives/rectangular_prism_moments.hpp"

using namespace cdfmm;

TEST_CASE("rectangular-prism moment tables reproduce scalar averages exactly",
          "[prism][moments]")
{
    const std::array<RectangularPrism, 3> prisms{{
        {2.0, 4.0, 6.0},
        {0.125, 2.5, 0.75},
        {3.0, 3.0, 3.0},
    }};
    const std::array<Vec3, 3> displacements{{
        {}, {0.125, -0.375, 0.0625}, {-0.75, 0.5, -0.2},
    }};

    for (const auto& prism : prisms) {
        for (const Vec3& displacement : displacements) {
            const MultiIndexSet basis(14);
            const auto table = detail::rectangular_prism_averaged_monomials(
                basis, displacement, prism);
            REQUIRE(table.size() == static_cast<std::size_t>(basis.size()));
            for (int index = 0; index < basis.size(); ++index) {
                const double scalar = rectangular_prism_averaged_monomial(
                    basis[index], displacement, prism);
                CAPTURE(index, basis[index].ax, basis[index].ay,
                        basis[index].az, displacement.x, displacement.y,
                        displacement.z);
                REQUIRE(std::bit_cast<std::uint64_t>(
                            table[static_cast<std::size_t>(index)]) ==
                        std::bit_cast<std::uint64_t>(scalar));
            }
        }
    }
}

TEST_CASE("rectangular-prism moment tables preserve scalar validation",
          "[prism][moments]")
{
    const MultiIndexSet basis(3);
    REQUIRE_THROWS_WITH(
        detail::rectangular_prism_averaged_monomials(
            basis, {}, RectangularPrism{0.0, 1.0, 1.0}),
        "cuboid dimensions must be finite and positive");

    // Match the scalar builder's no-work behaviour for an empty basis.
    const auto empty = detail::rectangular_prism_averaged_monomials(
        MultiIndexSet(-1), {}, RectangularPrism{});
    REQUIRE(empty.empty());
}
