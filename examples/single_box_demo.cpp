// SPDX-License-Identifier: Apache-2.0
#include "cdfmm/operators.hpp"
#include <iostream>
int main() {
  using namespace cdfmm;
  MultiIndexSet b(3);
  std::vector<Vec3> xs{{0.1, 0, 0}, {-0.1, 0.05, 0}};
  std::vector<Vec3> ms{{1, 0, 0}, {0, 1, 0}};
  Vec3 cs{0, 0, 0}, ct{2, 0, 0}, target{2.02, 0.01, 0};
  auto M = p2m_dipole(b, cs, xs, ms);
  std::vector<double> L(b.size(), 0.0);
  m2l_add(b, ct - cs, M, L);
  auto far = l2p_eval(b, ct, target, L, OutputFlags::Both);
  auto direct = p2p_dipole_sum(target, xs, ms, OutputFlags::Both);
  std::cout << far.phi << " " << far.H.x << "\n"
            << direct.phi << " " << direct.H.x << "\n";
}
