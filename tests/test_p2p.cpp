#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numbers>
#include <stdexcept>
#include <vector>

#include "cdfmm/operators.hpp"
#include "cdfmm/static_operators.hpp"
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

  const auto result = direct_p2p_reference(positions, positions, moments,
                                           OutputFlags::Both, identities);

  for (std::size_t index = 0; index < positions.size(); ++index) {
    const auto expected =
        p2p_dipole_sum(positions[index], positions, moments, OutputFlags::Both,
                       static_cast<int>(index));
    REQUIRE(result[index].phi == Catch::Approx(expected.phi));
    REQUIRE(result[index].H.x == Catch::Approx(expected.H.x));
    REQUIRE(result[index].H.y == Catch::Approx(expected.H.y));
    REQUIRE(result[index].H.z == Catch::Approx(expected.H.z));
  }

  REQUIRE_THROWS_AS(direct_p2p_reference(positions, positions, moments,
                                         OutputFlags::Field,
                                         std::vector<int>{0}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(direct_p2p_reference(positions, positions, moments,
                                         OutputFlags::Field,
                                         std::vector<int>{0, 2}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(direct_p2p_reference(positions, positions,
                                         std::vector<Vec3>{moments.front()}),
                    std::invalid_argument);
}

TEST_CASE("Regular-grid P2P table and on-the-fly execution match canonical") {
  const std::vector<Vec3> positions{
      {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}};
  const std::vector<Vec3> moments{{1.0, 0.5, -0.25},
                                  {-0.5, 1.0, 0.25},
                                  {0.25, -0.75, 1.0},
                                  {0.5, 0.25, -1.0}};
  std::vector<std::array<int, 2>> interactions;
  for (int target = 0; target < 4; ++target) {
    for (int source = 0; source < 4; ++source) {
      interactions.push_back({target, source});
    }
  }
  const auto canonical =
      build_static_p2p_operator(positions, positions, interactions);
  const auto grid =
      build_static_p2p_grid_stencil_plan(canonical, positions, true);
  REQUIRE(grid.has_value());
  REQUIRE(grid->displacements.size() == 9);
  REQUIRE(grid->tensors[0].size() == grid->displacements.size());

  const std::vector<int> identities{0, 1, 2, 3};
  std::vector<Vec3> expected(4);
  std::vector<Vec3> table(4);
  std::vector<Vec3> on_the_fly(4);
  apply_static_p2p_operator(canonical, moments, expected, identities);
  apply_static_p2p_grid_stencil_plan(*grid, moments, table, identities);
  apply_static_p2p_grid_point_onthefly(*grid, moments, on_the_fly, identities);
  for (std::size_t index = 0; index < expected.size(); ++index) {
    REQUIRE(table[index].x == Catch::Approx(expected[index].x));
    REQUIRE(table[index].y == Catch::Approx(expected[index].y));
    REQUIRE(table[index].z == Catch::Approx(expected[index].z));
    REQUIRE(on_the_fly[index].x == Catch::Approx(expected[index].x));
    REQUIRE(on_the_fly[index].y == Catch::Approx(expected[index].y));
    REQUIRE(on_the_fly[index].z == Catch::Approx(expected[index].z));
  }
}

TEST_CASE("Irregular particle geometry rejects the grid stencil") {
  const std::vector<Vec3> positions{
      {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.2, 1.0, 0.0}};
  const std::vector<std::array<int, 2>> interactions{{0, 1}, {1, 0}};
  const auto canonical =
      build_static_p2p_operator(positions, positions, interactions);
  REQUIRE_FALSE(build_static_p2p_grid_stencil_plan(canonical, positions, true)
                    .has_value());
}
