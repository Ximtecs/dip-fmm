// SPDX-License-Identifier: Apache-2.0
#include "cdfmm/operators.hpp"

#include <chrono>
#include <iostream>
#include <random>

int main()
{
  using namespace cdfmm;

  constexpr int n_sources = 10000;
  std::mt19937 generator(1u);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  std::vector<Vec3> source_positions(n_sources);
  std::vector<Vec3> dipole_moments(n_sources);

  // Use a deterministic cloud so smoke-benchmark runs remain comparable.
  for (int i = 0; i < n_sources; ++i) {
    source_positions[i] = {
        distribution(generator),
        distribution(generator),
        distribution(generator)
    };
    dipole_moments[i] = {
        distribution(generator),
        distribution(generator),
        distribution(generator)
    };
  }

  const Vec3 target_position{0.3, -0.2, 0.1};
  const auto start = std::chrono::high_resolution_clock::now();
  const PotentialField result = p2p_dipole_sum(
      target_position,
      source_positions,
      dipole_moments
  );
  const auto stop = std::chrono::high_resolution_clock::now();

  std::cout << result.H.x << " "
            << std::chrono::duration<double, std::milli>(stop - start).count()
            << " ms\n";

  return 0;
}
