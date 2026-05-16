#include <catch2/catch_test_macros.hpp>

#include "cdfmm/operators.hpp"

using namespace cdfmm;

TEST_CASE("Output flags") {
  const auto field_only =
      p2p_dipole_pair({1, 0, 0}, {0, 0, 0}, {1, 0, 0}, OutputFlags::Field);
  REQUIRE(field_only.phi == 0.0);
  REQUIRE(field_only.H.x != 0.0);

  const auto potential_only =
      p2p_dipole_pair({1, 0, 0}, {0, 0, 0}, {1, 0, 0}, OutputFlags::Potential);
  REQUIRE(potential_only.phi != 0.0);
  REQUIRE(potential_only.H.x == 0.0);

  const auto both =
      p2p_dipole_pair({1, 0, 0}, {0, 0, 0}, {1, 0, 0}, OutputFlags::Both);
  REQUIRE(both.phi != 0.0);
  REQUIRE(both.H.x != 0.0);
}
