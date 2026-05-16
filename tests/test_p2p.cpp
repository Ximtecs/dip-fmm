#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <numbers>

#include "cdfmm/operators.hpp"

using namespace cdfmm;

TEST_CASE("P2P axial") {
    const auto r =
        p2p_dipole_pair({1, 0, 0}, {0, 0, 0}, {1, 0, 0}, OutputFlags::Both);
    const double c = 1.0 / (4.0 * std::numbers::pi);

    REQUIRE(r.phi == Catch::Approx(c));
    REQUIRE(r.H.x == Catch::Approx(2 * c));
    REQUIRE(r.H.y == Catch::Approx(0));
    REQUIRE(r.H.z == Catch::Approx(0));
}

TEST_CASE("P2P transverse") {
    const auto r =
        p2p_dipole_pair({0, 1, 0}, {0, 0, 0}, {1, 0, 0}, OutputFlags::Both);
    const double c = 1.0 / (4.0 * std::numbers::pi);

    REQUIRE(r.phi == Catch::Approx(0));
    REQUIRE(r.H.x == Catch::Approx(-c));
    REQUIRE(r.H.y == Catch::Approx(0));
    REQUIRE(r.H.z == Catch::Approx(0));
}
