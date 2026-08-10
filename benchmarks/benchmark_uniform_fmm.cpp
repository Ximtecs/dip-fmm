// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/uniform_fmm.hpp"
#include "cdfmm/validation.hpp"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

int main() {
  using namespace cdfmm;

  constexpr int particle_count = 200;
  constexpr int tree_depth = 2;
  constexpr int expansion_order = 4;
  std::mt19937 generator(314159u);
  std::uniform_real_distribution<double> distribution(-0.95, 0.95);
  std::vector<Vec3> source_positions(particle_count);
  std::vector<Vec3> target_positions(particle_count);
  std::vector<Vec3> dipole_moments(particle_count);

  for (int i = 0; i < particle_count; ++i) {
    source_positions[i] = {distribution(generator), distribution(generator),
                           distribution(generator)};
    target_positions[i] = {distribution(generator), distribution(generator),
                           distribution(generator)};
    dipole_moments[i] = {distribution(generator), distribution(generator),
                         distribution(generator)};
  }

  UniformFmmOptions options;
  options.expansion_order = expansion_order;
  options.tree.max_level = tree_depth;
  options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
  options.tree.root_half_width = 1.0;
  UniformFmm fmm(source_positions, target_positions, options);

  const auto fmm_start = std::chrono::steady_clock::now();
  const auto approximate = fmm.evaluate(dipole_moments);
  const auto fmm_stop = std::chrono::steady_clock::now();

  const auto direct_start = std::chrono::steady_clock::now();
  const auto reference =
      direct_p2p_reference(target_positions, source_positions, dipole_moments);
  const auto direct_stop = std::chrono::steady_clock::now();

  std::vector<Vec3> approximate_fields;
  std::vector<Vec3> reference_fields;
  for (int i = 0; i < particle_count; ++i) {
    approximate_fields.push_back(approximate[i].H);
    reference_fields.push_back(reference[i].H);
  }
  const ErrorMetrics metrics =
      compute_error_metrics(approximate_fields, reference_fields);

  std::cout
      << "sources/targets: " << particle_count << "/" << particle_count << "\n"
      << "tree depth: " << tree_depth << "\n"
      << "expansion order: " << expansion_order << "\n"
      << "FMM evaluation: "
      << std::chrono::duration<double, std::milli>(fmm_stop - fmm_start).count()
      << " ms\n"
      << "direct P2P: "
      << std::chrono::duration<double, std::milli>(direct_stop - direct_start)
             .count()
      << " ms\n"
      << "RMS relative field error: " << metrics.rms_relative_error << "\n";

  return 0;
}
