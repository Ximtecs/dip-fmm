// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <vector>

#include "cdfmm/geometry/primitives/tetrahedron.hpp"

namespace cdfmm::detail {

// MultiIndexSet's total-degree / (ax, ay) order, without its linear search.
// The caller supplies non-negative exponents belonging to its table.
inline std::size_t tetrahedron_monomial_index(const MultiIndex& alpha)
{
    const auto n = static_cast<std::size_t>(alpha.degree());
    const auto ax = static_cast<std::size_t>(alpha.ax);
    const auto ay = static_cast<std::size_t>(alpha.ay);
    return n * (n + 1) * (n + 2) / 6 + ax * (2 * n + 3 - ax) / 2 + ay;
}

// Uniform averages E[(d + t)^alpha] / alpha! for every basis entry.
// Geometry validation is identical to the scalar public trust anchor; the
// table is local construction scratch, never a persistent or global cache.
std::vector<double> tetrahedron_averaged_monomials(
    const MultiIndexSet& basis, const Vec3& d, const Tetrahedron& tetrahedron);

} // namespace cdfmm::detail
