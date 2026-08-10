#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numbers>
#include <stdexcept>
#include <vector>

#include "cdfmm/operators.hpp"
#include "cdfmm/validation.hpp"

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

TEST_CASE("Direct all-to-all reference excludes singular self pairs") {
  const std::vector<Vec3> positions{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  const std::vector<Vec3> moments{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
  const std::vector<int> identities{0, 1};

  const auto result = direct_p2p_reference(
      positions, positions, moments, OutputFlags::Both, identities
  );

  for (std::size_t index = 0; index < positions.size(); ++index) {
    const auto expected = p2p_dipole_sum(
        positions[index], positions, moments, OutputFlags::Both,
        static_cast<int>(index)
    );
    REQUIRE(result[index].phi == Catch::Approx(expected.phi));
    REQUIRE(result[index].H.x == Catch::Approx(expected.H.x));
    REQUIRE(result[index].H.y == Catch::Approx(expected.H.y));
    REQUIRE(result[index].H.z == Catch::Approx(expected.H.z));
  }

  REQUIRE_THROWS_AS(
      direct_p2p_reference(
          positions, positions, moments, OutputFlags::Field,
          std::vector<int>{0}
      ),
      std::invalid_argument
  );
  REQUIRE_THROWS_AS(
      direct_p2p_reference(
          positions, positions, moments, OutputFlags::Field,
          std::vector<int>{0, 2}
      ),
      std::invalid_argument
  );
}
