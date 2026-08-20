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
#include "cdfmm/validation.hpp"

#ifdef CDFMM_USE_OPENMP
#include <omp.h>
#endif

using namespace cdfmm;

TEST_CASE("automatic FMM execution resolves to a truthful CPU backend")
{
    const std::vector<Vec3> positions{{-0.25, 0.0, 0.0}, {0.25, 0.0, 0.0}};
    UniformFmm fmm(positions);
    REQUIRE(fmm.backend() == ExecutionBackend::CpuStatic);
    REQUIRE(fmm.p2p_execution_packing() == P2PExecutionPacking::ParticleRowSoa);
}

TEST_CASE("fixed P2P identities are optional and immutable",
          "[uniform_fmm][p2p]") {
  const std::vector<Vec3> positions{
      {-0.4, 0.0, 0.0}, {0.1, 0.2, 0.0}, {0.35, -0.15, 0.1}};
  const std::vector<Vec3> moments{
      {0.2, -0.3, 0.5}, {-0.1, 0.7, 0.4}, {0.6, 0.2, -0.5}};
  const std::vector<int> identities{0, 1, 2};

  UniformFmmOptions options;
  options.tree.max_level = 0;
  options.fixed_target_source_indices = identities;
  UniformFmm fmm(positions, positions, options);

  const auto implicit = fmm.evaluate(moments, OutputFlags::Field);
  const auto explicit_map =
      fmm.evaluate(moments, OutputFlags::Field, identities);
  REQUIRE(implicit.size() == explicit_map.size());
  for (std::size_t index = 0; index < implicit.size(); ++index) {
    REQUIRE(implicit[index].H.x ==
            Catch::Approx(explicit_map[index].H.x).margin(1.0e-14));
    REQUIRE(implicit[index].H.y ==
            Catch::Approx(explicit_map[index].H.y).margin(1.0e-14));
    REQUIRE(implicit[index].H.z ==
            Catch::Approx(explicit_map[index].H.z).margin(1.0e-14));
  }

  REQUIRE_THROWS_AS(
      fmm.evaluate(moments, OutputFlags::Field, std::vector<int>{-1, 1, 2}),
      std::invalid_argument);
}

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
  REQUIRE_THROWS_AS(
      fmm.upward_pass(std::vector<Vec3>(distributed_positions.size() - 1)),
      std::invalid_argument);
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

TEST_CASE("Complete uniform FMM converges towards direct P2P",
          "[uniform_fmm]") {
  std::mt19937 generator(9281);
  std::uniform_real_distribution<double> distribution(-0.95, 0.95);
  std::vector<Vec3> sources(20);
  std::vector<Vec3> targets(15);
  std::vector<Vec3> moments(20);
  for (Vec3 &position : sources) {
    position = {distribution(generator), distribution(generator),
                distribution(generator)};
  }
  for (Vec3 &position : targets) {
    position = {distribution(generator), distribution(generator),
                distribution(generator)};
  }
  for (Vec3 &moment : moments) {
    moment = {distribution(generator), distribution(generator),
              distribution(generator)};
  }

  const auto direct = direct_p2p_reference(targets, sources, moments);
  std::vector<Vec3> direct_fields;
  for (const PotentialField &value : direct) {
    direct_fields.push_back(value.H);
  }

  double previous_rms = 1.0;
  for (const int order : {2, 3, 4}) {
    UniformFmmOptions options;
    options.expansion_order = order;
    options.backend = ExecutionBackend::CpuStatic;
    options.tree.max_level = 2;
    options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
    options.tree.root_half_width = 1.0;
    UniformFmm fmm(sources, targets, options);
    const auto approximate = fmm.evaluate(moments);
    std::vector<Vec3> approximate_fields;
    for (const PotentialField &value : approximate) {
      approximate_fields.push_back(value.H);
    }

    const ErrorMetrics metrics =
        compute_error_metrics(approximate_fields, direct_fields);
    REQUIRE(metrics.rms_relative_error < previous_rms);
    previous_rms = metrics.rms_relative_error;
  }
  REQUIRE(previous_rms < 2.0e-2);
}

TEST_CASE("Depth-zero complete evaluation is direct in every output mode",
          "[uniform_fmm]") {
  const std::vector<Vec3> targets{{0.12, -0.27, 0.34}, {-0.45, 0.16, -0.08}};
  UniformFmmOptions options;
  options.expansion_order = 3;
  options.backend = ExecutionBackend::CpuStatic;
  options.tree.max_level = 0;
  UniformFmm fmm(distributed_positions, targets, options);

  for (const OutputFlags output :
       {OutputFlags::Field, OutputFlags::Potential, OutputFlags::Both}) {
    const auto actual = fmm.evaluate(distributed_moments, output);
    const auto direct = direct_p2p_reference(targets, distributed_positions,
                                             distributed_moments, output);
    for (std::size_t i = 0; i < targets.size(); ++i) {
      REQUIRE(actual[i].phi == Catch::Approx(direct[i].phi).margin(1.0e-14));
      REQUIRE(actual[i].H.x == Catch::Approx(direct[i].H.x).margin(1.0e-14));
      REQUIRE(actual[i].H.y == Catch::Approx(direct[i].H.y).margin(1.0e-14));
      REQUIRE(actual[i].H.z == Catch::Approx(direct[i].H.z).margin(1.0e-14));
    }
  }
}

TEST_CASE("Target permutation and repeated evaluation replace downward state",
          "[uniform_fmm]") {
  std::vector<Vec3> targets{{0.72, 0.68, 0.61},
                            {-0.77, -0.66, -0.59},
                            {0.15, -0.31, 0.47},
                            {-0.52, 0.63, -0.44}};
  UniformFmmOptions options;
  options.expansion_order = 4;
  options.backend = ExecutionBackend::CpuStatic;
  options.tree.max_level = 2;
  options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
  options.tree.root_half_width = 1.0;
  UniformFmm fmm(distributed_positions, targets, options);

  const auto first = fmm.evaluate(distributed_moments);
  const auto direct =
      direct_p2p_reference(targets, distributed_positions, distributed_moments);
  for (std::size_t i = 0; i < targets.size(); ++i) {
    REQUIRE(relative_error(first[i].H, direct[i].H) < 3.0e-2);
  }

  std::vector<Vec3> zero_moments(distributed_moments.size());
  const auto second = fmm.evaluate(zero_moments);
  for (const PotentialField &value : second) {
    REQUIRE(value.H.x == 0.0);
    REQUIRE(value.H.y == 0.0);
    REQUIRE(value.H.z == 0.0);
  }
  for (const TreeNode &node : fmm.tree().nodes()) {
    for (const double coefficient : fmm.local(node.index)) {
      REQUIRE(coefficient == 0.0);
    }
  }
}

TEST_CASE("Explicit source identities exclude only singular self pairs",
          "[uniform_fmm]") {
  const std::vector<Vec3> positions{
      {-0.2, 0.1, 0.3}, {0.1, 0.2, -0.3}, {0.5, -0.4, 0.2}};
  const std::vector<Vec3> moments{
      {0.4, 0.1, -0.2}, {-0.3, 0.7, 0.5}, {0.2, -0.6, 0.8}};
  UniformFmmOptions options;
  options.expansion_order = 3;
  options.backend = ExecutionBackend::CpuStatic;
  options.tree.max_level = 0;
  UniformFmm fmm(positions, positions, options);
  const std::vector<int> identities{0, 1, 2};

  const auto actual = fmm.evaluate(moments, OutputFlags::Both, identities);
  for (std::size_t target = 0; target < positions.size(); ++target) {
    const PotentialField expected =
        p2p_dipole_sum(positions[target], positions, moments, OutputFlags::Both,
                       static_cast<int>(target));
    REQUIRE(actual[target].phi == Catch::Approx(expected.phi).margin(1.0e-14));
    REQUIRE(actual[target].H.x == Catch::Approx(expected.H.x).margin(1.0e-14));
    REQUIRE(actual[target].H.y == Catch::Approx(expected.H.y).margin(1.0e-14));
    REQUIRE(actual[target].H.z == Catch::Approx(expected.H.z).margin(1.0e-14));
  }

  REQUIRE_THROWS_AS(
      fmm.evaluate(moments, OutputFlags::Field, std::vector<int>{0, 1}),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      fmm.evaluate(moments, OutputFlags::Field, std::vector<int>{0, 1, 4}),
      std::invalid_argument);
}

#ifdef CDFMM_USE_OPENMP
TEST_CASE("OpenMP thread counts preserve complete FMM results",
          "[uniform_fmm][openmp]") {
  UniformFmmOptions options;
  options.expansion_order = 4;
  options.backend = ExecutionBackend::CpuStatic;
  options.tree.max_level = 3;
  options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
  options.tree.root_half_width = 1.0;
  UniformFmm fmm(distributed_positions, distributed_positions, options);
  std::vector<int> source_identities(distributed_positions.size());
  std::iota(source_identities.begin(), source_identities.end(), 0);

  omp_set_num_threads(1);
  const auto serial = fmm.evaluate(distributed_moments, OutputFlags::Field,
                                   source_identities);
  omp_set_num_threads(std::min(4, omp_get_num_procs()));
  const auto threaded = fmm.evaluate(distributed_moments, OutputFlags::Field,
                                     source_identities);

  REQUIRE(threaded.size() == serial.size());
  for (std::size_t index = 0; index < serial.size(); ++index) {
    // Explicit source identities exclude the coincident self pair so this
    // comparison checks finite physical fields rather than matching NaNs.
    REQUIRE(std::isfinite(serial[index].H.x));
    REQUIRE(std::isfinite(serial[index].H.y));
    REQUIRE(std::isfinite(serial[index].H.z));
    REQUIRE(threaded[index].H.x ==
            Catch::Approx(serial[index].H.x).margin(1.0e-14));
    REQUIRE(threaded[index].H.y ==
            Catch::Approx(serial[index].H.y).margin(1.0e-14));
    REQUIRE(threaded[index].H.z ==
            Catch::Approx(serial[index].H.z).margin(1.0e-14));
  }
}
#endif

TEST_CASE("Evaluation timings aggregate and reset", "[uniform_fmm][timing]") {
  UniformFmmOptions options;
  options.backend = ExecutionBackend::CpuStatic;
  options.tree.max_level = 2;
  UniformFmm fmm(distributed_positions, distributed_positions, options);
  std::vector<PotentialField> results(distributed_positions.size());

  fmm.evaluate_into(distributed_moments, results);
  fmm.evaluate_into(distributed_moments, results);
  REQUIRE(fmm.last_timings().evaluations == 1);
  REQUIRE(fmm.last_timings().total.total_seconds > 0.0);
  REQUIRE(fmm.aggregate_timings().evaluations == 2);
  REQUIRE(fmm.tree().build_timings().total.total_seconds > 0.0);

  fmm.reset_timings();
  REQUIRE(fmm.aggregate_timings().evaluations == 0);
}
