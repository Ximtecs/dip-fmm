#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <numbers>

#include "cdfmm/laplace_derivatives.hpp"

using namespace cdfmm;

TEST_CASE("Laplace derivative sanity") {
    MultiIndexSet b(1);
    const auto d = laplace_derivatives_raw(b, {1, 0, 0});
    const double c = 1.0 / (4.0 * std::numbers::pi);

    REQUIRE(d[b.index({0, 0, 0})] == Catch::Approx(c).epsilon(1e-6));
    REQUIRE(d[b.index({1, 0, 0})] == Catch::Approx(-c).epsilon(1e-3));
    REQUIRE(d[b.index({0, 1, 0})] == Catch::Approx(0.0).margin(1e-6));
    REQUIRE(d[b.index({0, 0, 1})] == Catch::Approx(0.0).margin(1e-6));
}
