// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/operators.hpp"

#include "cdfmm/operators/l2l.hpp"
#include "cdfmm/operators/l2p.hpp"
#include "cdfmm/operators/m2l.hpp"
#include "cdfmm/operators/m2m.hpp"
#include "cdfmm/operators/m2p.hpp"
#include "cdfmm/operators/p2m.hpp"
#include "cdfmm/operators/p2p.hpp"

namespace cdfmm {

// These flat entry points are retained for source and Python compatibility.
// Mathematical ownership lives in the responsibility-specific namespaces.

CoeffVector p2m_dipole(const MultiIndexSet& basis, const Vec3& centre,
                       std::span<const Vec3> source_positions,
                       std::span<const Vec3> dipole_moments)
{
    return operators::p2m::evaluate(
        basis, centre, source_positions, dipole_moments);
}

void m2m_add(const MultiIndexSet& basis, const Vec3& displacement,
             std::span<const double> child, std::span<double> parent)
{
    operators::m2m::apply(basis, displacement, child, parent);
}

void m2l_add(const MultiIndexSet& basis, const Vec3& displacement,
             std::span<const double> multipole, std::span<double> local)
{
    operators::m2l::apply(basis, displacement, multipole, local);
}

void l2l_add(const MultiIndexSet& basis, const Vec3& displacement,
             std::span<const double> parent, std::span<double> child)
{
    operators::l2l::apply(basis, displacement, parent, child);
}

PotentialField l2p_eval(const MultiIndexSet& basis, const Vec3& centre,
                        const Vec3& target, std::span<const double> local,
                        OutputFlags output)
{
    return operators::l2p::evaluate(basis, centre, target, local, output);
}

PotentialField m2p_eval(const MultiIndexSet& basis, const CoeffVector& multipole,
                        const Vec3& source_centre, const Vec3& target,
                        OutputFlags output)
{
    return operators::m2p::evaluate(
        basis, multipole, source_centre, target, output);
}

PotentialField p2p_dipole_pair(const Vec3& target, const Vec3& source,
                               const Vec3& moment, OutputFlags output)
{
    return operators::p2p::evaluate_pair(target, source, moment, output);
}

PotentialField p2p_dipole_sum(const Vec3& target,
                              std::span<const Vec3> sources,
                              std::span<const Vec3> moments,
                              OutputFlags output, int self_index)
{
    return operators::p2p::evaluate_sum(
        target, sources, moments, output, self_index);
}

} // namespace cdfmm
