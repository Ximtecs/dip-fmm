#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cdfmm/taylor_jet.hpp"

using namespace cdfmm;

TEST_CASE("TaylorJet constant coefficients") {
  const MultiIndexSet basis(4);
  const TaylorJet c = TaylorJet::constant(basis, 3.25);

  REQUIRE(c.at({0, 0, 0}) == Catch::Approx(3.25));

  for (int i = 1; i < basis.size(); ++i) {
    REQUIRE(c.at(basis[i]) == Catch::Approx(0.0).margin(1e-15));
  }
}

TEST_CASE("TaylorJet coordinate coefficients") {
  const MultiIndexSet basis(3);
  const TaylorJet x = TaylorJet::coordinate(basis, 0, 2.0);

  REQUIRE(x.at({0, 0, 0}) == Catch::Approx(2.0));
  REQUIRE(x.at({1, 0, 0}) == Catch::Approx(1.0));
  REQUIRE(x.at({0, 1, 0}) == Catch::Approx(0.0).margin(1e-15));
  REQUIRE(x.at({0, 0, 1}) == Catch::Approx(0.0).margin(1e-15));
}

TEST_CASE("TaylorJet multiplication uses normalised coefficients") {
  const MultiIndexSet basis(3);
  const TaylorJet x = TaylorJet::coordinate(basis, 0, 2.0);
  const TaylorJet x2 = x.mul(x);

  REQUIRE(x2.at({0, 0, 0}) == Catch::Approx(4.0));
  REQUIRE(x2.at({1, 0, 0}) == Catch::Approx(4.0));
  REQUIRE(x2.at({2, 0, 0}) == Catch::Approx(1.0));
}

TEST_CASE("TaylorJet inverse square root") {
  const MultiIndexSet basis(4);

  SECTION("constant argument") {
    const TaylorJet z = TaylorJet::constant(basis, 4.0);
    const TaylorJet y = z.invsqrt();

    REQUIRE(y.at({0, 0, 0}) == Catch::Approx(0.5));
    for (int i = 1; i < basis.size(); ++i) {
      REQUIRE(y.at(basis[i]) == Catch::Approx(0.0).margin(1e-15));
    }
  }

  SECTION("one-dimensional polynomial") {
    const TaylorJet x = TaylorJet::coordinate(basis, 0, 2.0);
    const TaylorJet z = x.mul(x);
    const TaylorJet y = z.invsqrt();

    REQUIRE(y.at({0, 0, 0}) == Catch::Approx(0.5).epsilon(1e-12));
    REQUIRE(y.at({1, 0, 0}) == Catch::Approx(-0.25).epsilon(1e-12));
    REQUIRE(y.at({2, 0, 0}) == Catch::Approx(0.125).epsilon(1e-12));
    REQUIRE(y.at({3, 0, 0}) == Catch::Approx(-0.0625).epsilon(1e-12));
  }
}
