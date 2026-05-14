// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <span>
#include "cdfmm/coefficients.hpp"
#include "cdfmm/output_flags.hpp"
namespace cdfmm {
CoeffVector p2m_dipole(const MultiIndexSet& basis, const Vec3& centre, std::span<const Vec3> source_positions, std::span<const Vec3> dipole_moments);
void m2m_add(const MultiIndexSet& basis, const Vec3& d, std::span<const double> child, std::span<double> parent);
void m2l_add(const MultiIndexSet& basis, const Vec3& R, std::span<const double> M, std::span<double> L);
void l2l_add(const MultiIndexSet& basis, const Vec3& d, std::span<const double> parent, std::span<double> child);
PotentialField l2p_eval(const MultiIndexSet& basis, const Vec3& centre, const Vec3& target, std::span<const double> L, OutputFlags output = OutputFlags::Field);
PotentialField p2p_dipole_pair(const Vec3& target, const Vec3& source, const Vec3& moment, OutputFlags output = OutputFlags::Field);
PotentialField p2p_dipole_sum(const Vec3& target, std::span<const Vec3> sources, std::span<const Vec3> moments, OutputFlags output = OutputFlags::Field, int self_index = -1);
}
