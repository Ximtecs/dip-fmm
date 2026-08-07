// SPDX-License-Identifier: Apache-2.0
#include "cdfmm/operators.hpp"

#include <iostream>

int main()
{
  using namespace cdfmm;

  const MultiIndexSet basis(3);
  const std::vector<Vec3> source_positions{
      {0.1, 0.0, 0.0},
      {-0.1, 0.05, 0.0}
  };
  const std::vector<Vec3> dipole_moments{
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0}
  };
  const Vec3 source_centre{0.0, 0.0, 0.0};
  const Vec3 target_centre{2.0, 0.0, 0.0};
  const Vec3 target_position{2.02, 0.01, 0.0};

  // Compose a single well-separated P2M-M2L-L2P path and compare it with
  // direct summation at the same target.
  const CoeffVector M = p2m_dipole(
      basis,
      source_centre,
      source_positions,
      dipole_moments
  );
  CoeffVector L(basis.size(), 0.0);
  m2l_add(basis, target_centre - source_centre, M, L);

  const PotentialField far = l2p_eval(
      basis,
      target_centre,
      target_position,
      L,
      OutputFlags::Both
  );
  const PotentialField direct = p2p_dipole_sum(
      target_position,
      source_positions,
      dipole_moments,
      OutputFlags::Both
  );

  std::cout << far.phi << " " << far.H.x << "\n"
            << direct.phi << " " << direct.H.x << "\n";

  return 0;
}
