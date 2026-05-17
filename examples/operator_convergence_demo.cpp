// SPDX-License-Identifier: Apache-2.0

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "cdfmm/operators.hpp"
#include "cdfmm/validation.hpp"

int main()
{
  using namespace cdfmm;

  constexpr int n_sources = 400;
  constexpr int n_targets = 16;

  std::mt19937 source_generator(1337u);
  std::uniform_real_distribution<double> position_distribution(-0.5, 0.5);
  std::uniform_real_distribution<double> moment_distribution(-1.0, 1.0);

  std::vector<Vec3> source_positions(n_sources);
  std::vector<Vec3> dipole_moments(n_sources);

  for (int i = 0; i < n_sources; ++i) {
    source_positions[i] = {
        position_distribution(source_generator),
        position_distribution(source_generator),
        position_distribution(source_generator)};
    dipole_moments[i] = {
        moment_distribution(source_generator),
        moment_distribution(source_generator),
        moment_distribution(source_generator)};
  }

  std::mt19937 target_generator(4242u);
  std::normal_distribution<double> direction_distribution(0.0, 1.0);
  std::uniform_real_distribution<double> radius_distribution(3.0, 8.0);

  std::vector<Vec3> target_positions;
  target_positions.reserve(n_targets);

  for (int i = 0; i < n_targets; ++i) {
    Vec3 direction{
        direction_distribution(target_generator),
        direction_distribution(target_generator),
        direction_distribution(target_generator)};
    direction = direction * (1.0 / norm(direction));

    const double radius = radius_distribution(target_generator);
    target_positions.push_back(direction * radius);
  }

  const std::vector<PotentialField> direct = direct_p2p_reference(
      target_positions, source_positions, dipole_moments, OutputFlags::Field);

  std::vector<Vec3> direct_fields;
  direct_fields.reserve(direct.size());
  for (const PotentialField &value : direct) {
    direct_fields.push_back(value.H);
  }

  const Vec3 source_centre{0.0, 0.0, 0.0};

  std::cout << "Operator convergence diagnostics (P2M + M2P vs direct P2P)\n";
  std::cout << "p    mean_rel_error    rms_rel_error     max_rel_error\n";

  for (int p = 2; p <= 6; ++p) {
    const MultiIndexSet basis(p);
    const CoeffVector M = p2m_dipole(
        basis, source_centre, source_positions, dipole_moments);

    std::vector<Vec3> m2p_fields;
    m2p_fields.reserve(target_positions.size());

    for (const Vec3 &target_position : target_positions) {
      const PotentialField far = m2p_eval(
          basis, M, source_centre, target_position, OutputFlags::Field);
      m2p_fields.push_back(far.H);
    }

    const ErrorMetrics metrics = compute_error_metrics(m2p_fields, direct_fields);

    std::cout << std::setw(1) << p << "    " << std::scientific
              << std::setprecision(6) << metrics.mean_relative_error << "    "
              << metrics.rms_relative_error << "    "
              << metrics.max_relative_error << "\n";
  }

  return 0;
}
