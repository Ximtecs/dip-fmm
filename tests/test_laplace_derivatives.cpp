#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <numbers>

#include "cdfmm/laplace_derivatives.hpp"

using namespace cdfmm;

TEST_CASE("Laplace derivatives on axis at r=(1,0,0)") {
  const MultiIndexSet basis(2);
  const auto d = laplace_derivatives_raw(basis, {1.0, 0.0, 0.0});
  const double c = 1.0 / (4.0 * std::numbers::pi);

  REQUIRE(d[basis.index({0, 0, 0})] == Catch::Approx(c).epsilon(1e-12));
  REQUIRE(d[basis.index({1, 0, 0})] == Catch::Approx(-c).epsilon(1e-12));
  REQUIRE(d[basis.index({0, 1, 0})] == Catch::Approx(0.0).margin(1e-14));
  REQUIRE(d[basis.index({0, 0, 1})] == Catch::Approx(0.0).margin(1e-14));

  REQUIRE(d[basis.index({2, 0, 0})] == Catch::Approx(2.0 * c).epsilon(1e-12));
  REQUIRE(d[basis.index({0, 2, 0})] == Catch::Approx(-c).epsilon(1e-12));
  REQUIRE(d[basis.index({0, 0, 2})] == Catch::Approx(-c).epsilon(1e-12));
  REQUIRE(d[basis.index({1, 1, 0})] == Catch::Approx(0.0).margin(1e-14));
}

TEST_CASE("Laplace derivatives satisfy Laplace equation away from singularity") {
  const MultiIndexSet basis(2);

  const std::array<Vec3, 2> points{
      Vec3{1.3, -0.7, 0.4},
      Vec3{-0.8, 1.1, 0.9},
  };

  for (const Vec3 &r : points) {
    const auto d = laplace_derivatives_raw(basis, r);
    const double laplacian =
        d[basis.index({2, 0, 0})] + d[basis.index({0, 2, 0})] +
        d[basis.index({0, 0, 2})];
    REQUIRE(laplacian == Catch::Approx(0.0).margin(1e-11));
  }
}
