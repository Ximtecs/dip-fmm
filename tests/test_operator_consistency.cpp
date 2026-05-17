#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "cdfmm/operators.hpp"
#include "cdfmm/validation.hpp"

using namespace cdfmm;

namespace {

void generate_random_sources(int n_sources, uint32_t seed,
                             std::vector<Vec3> &source_positions,
                             std::vector<Vec3> &dipole_moments)
{
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> position_distribution(-0.5, 0.5);
  std::uniform_real_distribution<double> moment_distribution(-1.0, 1.0);

  source_positions.resize(n_sources);
  dipole_moments.resize(n_sources);

  for (int i = 0; i < n_sources; ++i) {
    source_positions[i] = {
        position_distribution(generator),
        position_distribution(generator),
        position_distribution(generator)};
    dipole_moments[i] = {
        moment_distribution(generator),
        moment_distribution(generator),
        moment_distribution(generator)};
  }
}

std::vector<Vec3> generate_far_targets(int n_targets, uint32_t seed)
{
  std::mt19937 generator(seed);
  std::normal_distribution<double> direction_distribution(0.0, 1.0);
  std::uniform_real_distribution<double> radius_distribution(3.0, 8.0);

  std::vector<Vec3> target_positions;
  target_positions.reserve(n_targets);

  for (int i = 0; i < n_targets; ++i) {
    Vec3 direction{
        direction_distribution(generator),
        direction_distribution(generator),
        direction_distribution(generator)};
    direction = direction * (1.0 / norm(direction));

    const double radius = radius_distribution(generator);
    target_positions.push_back(direction * radius);
  }

  return target_positions;
}

std::vector<Vec3> evaluate_m2p_fields(const MultiIndexSet &basis,
                                      const CoeffVector &M,
                                      const Vec3 &source_centre,
                                      std::span<const Vec3> target_positions)
{
  std::vector<Vec3> fields;
  fields.reserve(target_positions.size());

  for (const Vec3 &target_position : target_positions) {
    const PotentialField far =
        m2p_eval(basis, M, source_centre, target_position, OutputFlags::Field);
    fields.push_back(far.H);
  }

  return fields;
}

std::vector<Vec3> evaluate_l2p_fields(const MultiIndexSet &basis,
                                      std::span<const double> L,
                                      const Vec3 &target_centre,
                                      std::span<const Vec3> target_positions)
{
  std::vector<Vec3> fields;
  fields.reserve(target_positions.size());

  for (const Vec3 &target_position : target_positions) {
    const PotentialField far =
        l2p_eval(basis, target_centre, target_position, L, OutputFlags::Field);
    fields.push_back(far.H);
  }

  return fields;
}

std::string metrics_string(const ErrorMetrics &metrics)
{
  return "mean_rel=" + std::to_string(metrics.mean_relative_error) +
         ", rms_rel=" + std::to_string(metrics.rms_relative_error) +
         ", max_rel=" + std::to_string(metrics.max_relative_error) +
         ", mean_abs=" + std::to_string(metrics.mean_absolute_error) +
         ", max_abs=" + std::to_string(metrics.max_absolute_error);
}

} // namespace

TEST_CASE("P2P pair equals one-source P2P sum")
{
  const Vec3 source_position{0.0, 0.0, 0.0};
  const Vec3 dipole_moment{0.75, -0.5, 0.25};
  const Vec3 target_position{0.6, -0.2, 0.9};

  const std::vector<Vec3> source_positions{source_position};
  const std::vector<Vec3> dipole_moments{dipole_moment};

  const PotentialField pair_field = p2p_dipole_pair(
      target_position, source_position, dipole_moment, OutputFlags::Field);
  const PotentialField sum_field = p2p_dipole_sum(
      target_position, source_positions, dipole_moments, OutputFlags::Field);

  REQUIRE(absolute_error(pair_field.H, sum_field.H) < 1e-14);

  const PotentialField pair_both = p2p_dipole_pair(
      target_position, source_position, dipole_moment, OutputFlags::Both);
  const PotentialField sum_both = p2p_dipole_sum(
      target_position, source_positions, dipole_moments, OutputFlags::Both);

  REQUIRE(std::abs(pair_both.phi - sum_both.phi) < 1e-14);
  REQUIRE(absolute_error(pair_both.H, sum_both.H) < 1e-14);
}

TEST_CASE("P2P sum self-index skips singular self interaction")
{
  const std::vector<Vec3> source_positions{{0.0, 0.0, 0.0},
                                           {0.2, -0.1, 0.0},
                                           {-0.3, 0.1, 0.25}};
  const std::vector<Vec3> dipole_moments{{1.0, 0.0, 0.0},
                                         {-0.5, 0.2, 0.1},
                                         {0.3, -0.4, 0.6}};

  const int self_index = 1;
  const Vec3 target_position = source_positions[self_index];

  const PotentialField with_skip = p2p_dipole_sum(
      target_position, source_positions, dipole_moments, OutputFlags::Both,
      self_index);

  PotentialField explicit_sum;
  for (size_t i = 0; i < source_positions.size(); ++i) {
    if (static_cast<int>(i) == self_index) {
      continue;
    }
    const PotentialField pair = p2p_dipole_pair(
        target_position, source_positions[i], dipole_moments[i],
        OutputFlags::Both);
    explicit_sum.phi += pair.phi;
    explicit_sum.H += pair.H;
  }

  REQUIRE(std::abs(with_skip.phi - explicit_sum.phi) < 1e-13);
  REQUIRE(absolute_error(with_skip.H, explicit_sum.H) < 1e-13);
}

TEST_CASE("M2P far-field accuracy improves with order")
{
  constexpr int n_sources = 400;
  constexpr int n_targets = 16;

  std::vector<Vec3> source_positions;
  std::vector<Vec3> dipole_moments;
  generate_random_sources(400, 1337u, source_positions, dipole_moments);

  const std::vector<Vec3> target_positions = generate_far_targets(16, 4242u);
  const std::vector<PotentialField> direct = direct_p2p_reference(
      target_positions, source_positions, dipole_moments, OutputFlags::Field);

  std::vector<Vec3> direct_fields;
  direct_fields.reserve(direct.size());
  for (const PotentialField &result : direct) {
    direct_fields.push_back(result.H);
  }

  std::vector<ErrorMetrics> metrics_by_order;

  for (int p = 2; p <= 6; ++p) {
    const MultiIndexSet basis(p);
    const Vec3 source_centre{0.0, 0.0, 0.0};
    const CoeffVector M = p2m_dipole(
        basis, source_centre, source_positions, dipole_moments);

    const std::vector<Vec3> fields = evaluate_m2p_fields(
        basis, M, source_centre, target_positions);
    const ErrorMetrics metrics = compute_error_metrics(fields, direct_fields);
    metrics_by_order.push_back(metrics);

    INFO("p=" << p << " " << metrics_string(metrics));
    REQUIRE(std::isfinite(metrics.mean_relative_error));
    REQUIRE(std::isfinite(metrics.rms_relative_error));
  }

  REQUIRE(metrics_by_order.back().rms_relative_error <
          metrics_by_order.front().rms_relative_error);
  REQUIRE(metrics_by_order.back().rms_relative_error < 5e-3);
}

TEST_CASE("M2M-combined expansion agrees with single-centre expansion")
{
  constexpr int n_sources = 400;

  std::vector<Vec3> source_positions;
  std::vector<Vec3> dipole_moments;
  generate_random_sources(n_sources, 1337u, source_positions, dipole_moments);
  const std::vector<Vec3> target_positions = generate_far_targets(16, 4242u);

  std::vector<Vec3> direct_fields;
  for (const PotentialField &result : direct_p2p_reference(
           target_positions, source_positions, dipole_moments,
           OutputFlags::Field)) {
    direct_fields.push_back(result.H);
  }

  std::vector<ErrorMetrics> combined_vs_direct;

  for (int p = 2; p <= 6; ++p) {
    const MultiIndexSet basis(p);
    const Vec3 parent_centre{0.0, 0.0, 0.0};
    CoeffVector parent_from_children(basis.size(), 0.0);

    std::array<std::vector<Vec3>, 8> octant_positions;
    std::array<std::vector<Vec3>, 8> octant_moments;

    for (size_t i = 0; i < source_positions.size(); ++i) {
      const Vec3 position = source_positions[i];
      const int sx = (position.x >= 0.0) ? 1 : 0;
      const int sy = (position.y >= 0.0) ? 1 : 0;
      const int sz = (position.z >= 0.0) ? 1 : 0;
      const int octant_index = sx + 2 * sy + 4 * sz;
      octant_positions[octant_index].push_back(position);
      octant_moments[octant_index].push_back(dipole_moments[i]);
    }

    for (int octant = 0; octant < 8; ++octant) {
      if (octant_positions[octant].empty()) {
        continue;
      }

      const Vec3 child_centre{
          (octant & 1) ? 0.25 : -0.25,
          (octant & 2) ? 0.25 : -0.25,
          (octant & 4) ? 0.25 : -0.25};

      const CoeffVector child_M = p2m_dipole(
          basis, child_centre, octant_positions[octant], octant_moments[octant]);
      m2m_add(basis, parent_centre - child_centre, child_M, parent_from_children);
    }

    const CoeffVector single_centre_M = p2m_dipole(
        basis, parent_centre, source_positions, dipole_moments);

    const std::vector<Vec3> combined_fields = evaluate_m2p_fields(
        basis, parent_from_children, parent_centre, target_positions);
    const std::vector<Vec3> single_fields = evaluate_m2p_fields(
        basis, single_centre_M, parent_centre, target_positions);

    const ErrorMetrics combined_direct_metrics =
        compute_error_metrics(combined_fields, direct_fields);
    const ErrorMetrics single_direct_metrics =
        compute_error_metrics(single_fields, direct_fields);
    const ErrorMetrics combined_single_metrics =
        compute_error_metrics(combined_fields, single_fields);

    combined_vs_direct.push_back(combined_direct_metrics);

    INFO("p=" << p << " combined vs direct: "
              << metrics_string(combined_direct_metrics));
    INFO("p=" << p << " single vs direct: "
              << metrics_string(single_direct_metrics));
    INFO("p=" << p << " combined vs single: "
              << metrics_string(combined_single_metrics));

    REQUIRE(combined_single_metrics.rms_relative_error < 5e-3);
    REQUIRE(combined_direct_metrics.rms_relative_error <
            1.8 * single_direct_metrics.rms_relative_error);
  }

  REQUIRE(combined_vs_direct.back().rms_relative_error <
          combined_vs_direct.front().rms_relative_error);
  REQUIRE(combined_vs_direct.back().rms_relative_error < 5e-3);
}

TEST_CASE("M2L plus L2P agrees with M2P for well-separated boxes")
{
  std::vector<Vec3> source_positions;
  std::vector<Vec3> dipole_moments;
  generate_random_sources(400, 1337u, source_positions, dipole_moments);

  const Vec3 source_centre{0.0, 0.0, 0.0};
  const Vec3 target_centre{4.0, 1.0, -2.0};

  std::vector<Vec3> target_positions;
  target_positions.reserve(16);

  std::mt19937 generator(7781u);
  std::uniform_real_distribution<double> near_distribution(-0.1, 0.1);
  for (int i = 0; i < 16; ++i) {
    target_positions.push_back(
        target_centre + Vec3{near_distribution(generator),
                             near_distribution(generator),
                             near_distribution(generator)});
  }

  std::vector<Vec3> direct_fields;
  for (const PotentialField &result : direct_p2p_reference(
           target_positions, source_positions, dipole_moments,
           OutputFlags::Field)) {
    direct_fields.push_back(result.H);
  }

  std::vector<ErrorMetrics> m2p_vs_direct;
  std::vector<ErrorMetrics> l2p_vs_direct;

  for (int p = 2; p <= 6; ++p) {
    const MultiIndexSet basis(p);
    const CoeffVector M = p2m_dipole(
        basis, source_centre, source_positions, dipole_moments);

    std::vector<double> L(basis.size(), 0.0);
    m2l_add(basis, target_centre - source_centre, M, L);

    const std::vector<Vec3> m2p_fields = evaluate_m2p_fields(
        basis, M, source_centre, target_positions);
    const std::vector<Vec3> l2p_fields = evaluate_l2p_fields(
        basis, L, target_centre, target_positions);

    const ErrorMetrics m2p_direct_metrics =
        compute_error_metrics(m2p_fields, direct_fields);
    const ErrorMetrics l2p_direct_metrics =
        compute_error_metrics(l2p_fields, direct_fields);
    const ErrorMetrics l2p_m2p_metrics =
        compute_error_metrics(l2p_fields, m2p_fields);

    m2p_vs_direct.push_back(m2p_direct_metrics);
    l2p_vs_direct.push_back(l2p_direct_metrics);

    INFO("p=" << p << " M2P vs direct: " << metrics_string(m2p_direct_metrics));
    INFO("p=" << p << " M2L+L2P vs direct: "
              << metrics_string(l2p_direct_metrics));
    INFO("p=" << p << " M2L+L2P vs M2P: " << metrics_string(l2p_m2p_metrics));

    REQUIRE(l2p_m2p_metrics.rms_relative_error < 4e-3);
  }

  REQUIRE(m2p_vs_direct.back().rms_relative_error <
          m2p_vs_direct.front().rms_relative_error);
  REQUIRE(l2p_vs_direct.back().rms_relative_error <
          l2p_vs_direct.front().rms_relative_error);
  REQUIRE(l2p_vs_direct.back().rms_relative_error < 5e-3);
}
