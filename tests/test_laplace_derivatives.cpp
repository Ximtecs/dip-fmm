#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numbers>
#include <stdexcept>

#include "cdfmm/laplace_derivatives.hpp"
#include "cdfmm/math/taylor_jet.hpp"

using namespace cdfmm;

TEST_CASE("Laplace derivatives from the recurrence match Taylor-jet composition") {
  // The recurrence of laplace_derivatives_raw against the composition it
  // replaced, G = 1/(4 pi sqrt(x^2 + y^2 + z^2)) in Taylor jets, through
  // order 30 (the order-2p set of a p = 15 M2L). Errors are scaled by the
  // largest derivative of the same total order: entries that vanish by
  // symmetry make a pointwise relative error meaningless.
  constexpr int order = 30;
  const MultiIndexSet basis(order);
  const std::array<Vec3, 4> displacements{
      Vec3{2.0, 1.0, 0.0}, Vec3{3.0, -2.0, 1.0}, Vec3{0.0, 0.0, 2.0}, Vec3{-3.0, 3.0, -3.0}};
  for (const Vec3& r : displacements) {
    const TaylorJet x = TaylorJet::coordinate(basis, 0, r.x);
    const TaylorJet y = TaylorJet::coordinate(basis, 1, r.y);
    const TaylorJet z = TaylorJet::coordinate(basis, 2, r.z);
    const TaylorJet G = x.mul(x).add(y.mul(y)).add(z.mul(z)).invsqrt().mul(
        TaylorJet::constant(basis, 1.0 / (4.0 * std::numbers::pi)));
    const auto recurrence = laplace_derivatives_raw(basis, r);
    std::map<int, double> scale;
    std::vector<double> reference(static_cast<std::size_t>(basis.size()));
    for (int i = 0; i < basis.size(); ++i) {
      reference[static_cast<std::size_t>(i)] =
          G.at(basis[i]) * MultiIndexSet::multi_factorial(basis[i]);
      const MultiIndex k = basis[i];
      const int degree = k.ax + k.ay + k.az;
      scale[degree] = std::max(scale[degree], std::abs(reference[static_cast<std::size_t>(i)]));
    }
    for (int i = 0; i < basis.size(); ++i) {
      const MultiIndex k = basis[i];
      const double difference = std::abs(recurrence[static_cast<std::size_t>(i)] -
                                         reference[static_cast<std::size_t>(i)]);
      REQUIRE(difference <= 1.0e-12 * scale[k.ax + k.ay + k.az]);
    }
  }
}

TEST_CASE("multi-index positions are the storage order, in closed form") {
  // index() computes the position instead of searching for it; it must agree
  // with the constructor's storage order at every entry, including the
  // order-2p derivative sets the M2L construction looks up.
  for (int order : {0, 1, 2, 6, 17, 34}) {
    const MultiIndexSet basis(order);
    for (int position = 0; position < basis.size(); ++position) {
      REQUIRE(basis.index(basis[position]) == position);
    }
    REQUIRE_THROWS_AS(basis.index({order + 1, 0, 0}), std::out_of_range);
    REQUIRE_THROWS_AS(basis.index({0, 0, order + 1}), std::out_of_range);
    REQUIRE_THROWS_AS(basis.index({-1, 1, 0}), std::out_of_range);
  }
}

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
