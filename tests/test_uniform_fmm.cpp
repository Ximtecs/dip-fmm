// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "cdfmm/operators.hpp"
#include "cdfmm/uniform_fmm.hpp"

using namespace cdfmm;

namespace {

void require_coefficients_equal(std::span<const double> actual,
                                std::span<const double> expected,
                                const double scale = 1.0) {
  REQUIRE(actual.size() == expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    REQUIRE(actual[i] == Catch::Approx(expected[i]).margin(2.0e-13 * scale));
  }
}

CoeffVector direct_root_multipole(const UniformFmm &fmm,
                                  const std::vector<Vec3> &source_positions,
                                  const std::vector<Vec3> &dipole_moments) {
  return p2m_dipole(fmm.basis(), fmm.tree().root_centre(), source_positions,
                    dipole_moments);
}

const std::vector<Vec3> distributed_positions{
    {-0.83, -0.71, -0.64}, {0.76, -0.58, -0.42}, {-0.61, 0.69, -0.37},
    {0.57, 0.73, 0.66},    {-0.14, 0.22, 0.51},  {0.31, -0.19, 0.12},
    {-0.42, 0.08, -0.11},  {0.08, 0.49, -0.72}};

const std::vector<Vec3> distributed_moments{
    {0.7, -0.2, 0.1}, {-0.4, 0.8, 0.3}, {0.2, 0.1, -0.6}, {-0.3, -0.5, 0.9},
    {0.6, 0.4, -0.2}, {-0.1, 0.3, 0.5}, {0.9, -0.7, 0.2}, {-0.5, 0.2, -0.4}};

} // namespace

TEST_CASE("Uniform upward root equals direct P2M at several orders",
          "[uniform_fmm]") {
  for (const int order : {1, 2, 4, 6}) {
    UniformFmmOptions options;
    options.expansion_order = order;
    options.tree.max_level = 3;
    options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
    options.tree.root_half_width = 1.0;

    UniformFmm fmm(distributed_positions, options);
    fmm.upward_pass(distributed_moments);

    const CoeffVector direct =
        direct_root_multipole(fmm, distributed_positions, distributed_moments);
    require_coefficients_equal(fmm.root_multipole(), direct);
  }
}

TEST_CASE("Depth-zero upward pass is direct leaf P2M", "[uniform_fmm]") {
  UniformFmmOptions options;
  options.expansion_order = 5;
  options.tree.max_level = 0;
  UniformFmm fmm(distributed_positions, options);

  fmm.upward_pass(distributed_moments);
  const CoeffVector direct =
      direct_root_multipole(fmm, distributed_positions, distributed_moments);
  require_coefficients_equal(fmm.root_multipole(), direct);
}

TEST_CASE("One populated leaf translates through multiple levels",
          "[uniform_fmm]") {
  const std::vector<Vec3> positions{
      {-0.88, -0.84, -0.91}, {-0.79, -0.93, -0.82}, {-0.86, -0.77, -0.89}};
  const std::vector<Vec3> moments{
      {0.4, -0.2, 0.7}, {-0.5, 0.9, 0.1}, {0.3, 0.2, -0.6}};
  UniformFmmOptions options;
  options.expansion_order = 5;
  options.tree.max_level = 3;
  options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
  options.tree.root_half_width = 1.0;
  UniformFmm fmm(positions, options);

  fmm.upward_pass(moments);
  const CoeffVector direct = direct_root_multipole(fmm, positions, moments);
  require_coefficients_equal(fmm.root_multipole(), direct);

  const auto nodes = fmm.tree().nodes();
  const auto populated_leaves =
      std::count_if(nodes.begin(), nodes.end(), [](const TreeNode &node) {
        return node.is_leaf() && node.source_count() > 0;
      });
  REQUIRE(populated_leaves == 1);
}

TEST_CASE("Every populated node agrees with direct subtree P2M",
          "[uniform_fmm]") {
  UniformFmmOptions options;
  options.expansion_order = 4;
  options.tree.max_level = 3;
  options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
  options.tree.root_half_width = 1.0;
  UniformFmm fmm(distributed_positions, options);
  fmm.upward_pass(distributed_moments);

  std::vector<Vec3> sorted_moments(distributed_moments.size());
  for (std::size_t sorted_index = 0; sorted_index < sorted_moments.size();
       ++sorted_index) {
    sorted_moments[sorted_index] = distributed_moments[static_cast<std::size_t>(
        fmm.tree().source_permutation()[sorted_index])];
  }

  const auto sorted_positions = fmm.tree().sorted_source_positions();
  for (const TreeNode &node : fmm.tree().nodes()) {
    if (node.source_count() == 0) {
      for (const double coefficient : fmm.multipole(node.index)) {
        REQUIRE(coefficient == 0.0);
      }
      continue;
    }

    const CoeffVector direct = p2m_dipole(
        fmm.basis(), node.centre,
        sorted_positions.subspan(node.source_begin, node.source_count()),
        std::span<const Vec3>(sorted_moments)
            .subspan(node.source_begin, node.source_count()));
    require_coefficients_equal(fmm.multipole(node.index), direct);
  }
}

TEST_CASE("Upward pass is independent of source input ordering",
          "[uniform_fmm]") {
  std::vector<std::size_t> order(distributed_positions.size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 generator(741);
  std::shuffle(order.begin(), order.end(), generator);

  std::vector<Vec3> shuffled_positions;
  std::vector<Vec3> shuffled_moments;
  for (const std::size_t index : order) {
    shuffled_positions.push_back(distributed_positions[index]);
    shuffled_moments.push_back(distributed_moments[index]);
  }

  UniformFmmOptions options;
  options.expansion_order = 4;
  options.tree.max_level = 2;
  options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
  options.tree.root_half_width = 1.0;
  UniformFmm original(distributed_positions, options);
  UniformFmm shuffled(shuffled_positions, options);
  original.upward_pass(distributed_moments);
  shuffled.upward_pass(shuffled_moments);

  require_coefficients_equal(shuffled.root_multipole(),
                             original.root_multipole());
}

TEST_CASE("Repeated upward passes replace all magnetic state",
          "[uniform_fmm]") {
  UniformFmmOptions options;
  options.expansion_order = 4;
  options.tree.max_level = 2;
  UniformFmm fmm(distributed_positions, options);
  fmm.upward_pass(distributed_moments);

  std::vector<Vec3> zero_moments(distributed_moments.size());
  fmm.upward_pass(zero_moments);
  for (const TreeNode &node : fmm.tree().nodes()) {
    for (const double coefficient : fmm.multipole(node.index)) {
      REQUIRE(coefficient == 0.0);
    }
  }

  std::vector<Vec3> second_moments = distributed_moments;
  for (Vec3 &moment : second_moments) {
    moment = moment * -0.37;
  }
  fmm.upward_pass(second_moments);
  const CoeffVector direct =
      direct_root_multipole(fmm, distributed_positions, second_moments);
  require_coefficients_equal(fmm.root_multipole(), direct);
}

TEST_CASE("Uniform upward pass validates public input", "[uniform_fmm]") {
  UniformFmmOptions invalid_options;
  invalid_options.expansion_order = -1;
  REQUIRE_THROWS_AS(UniformFmm(distributed_positions, invalid_options),
                    std::invalid_argument);

  UniformFmm fmm(distributed_positions, UniformFmmOptions{});
  REQUIRE_THROWS_WITH(
      fmm.upward_pass(std::vector<Vec3>(distributed_positions.size() - 1)),
      "UniformFmm::upward_pass requires one dipole moment per source position");
}

TEST_CASE("Hierarchical root and direct root give the same distant field",
          "[uniform_fmm]") {
  UniformFmmOptions options;
  options.expansion_order = 6;
  options.tree.max_level = 3;
  options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
  options.tree.root_half_width = 1.0;
  UniformFmm fmm(distributed_positions, options);
  fmm.upward_pass(distributed_moments);

  const CoeffVector direct =
      direct_root_multipole(fmm, distributed_positions, distributed_moments);
  const Vec3 target{8.0, -7.0, 9.0};
  const PotentialField hierarchical_field = m2p_eval(
      fmm.basis(),
      CoeffVector(fmm.root_multipole().begin(), fmm.root_multipole().end()),
      fmm.tree().root_centre(), target, OutputFlags::Both);
  const PotentialField direct_root_field = m2p_eval(
      fmm.basis(), direct, fmm.tree().root_centre(), target, OutputFlags::Both);

  REQUIRE(hierarchical_field.phi ==
          Catch::Approx(direct_root_field.phi).margin(1.0e-15));
  REQUIRE(hierarchical_field.H.x ==
          Catch::Approx(direct_root_field.H.x).margin(1.0e-15));
  REQUIRE(hierarchical_field.H.y ==
          Catch::Approx(direct_root_field.H.y).margin(1.0e-15));
  REQUIRE(hierarchical_field.H.z ==
          Catch::Approx(direct_root_field.H.z).margin(1.0e-15));
}
