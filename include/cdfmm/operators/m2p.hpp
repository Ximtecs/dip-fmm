// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cdfmm/operators.hpp"

namespace cdfmm::operators::m2p {

/**
 * @brief Evaluates a Cartesian multipole expansion directly at one target.
 *
 * This reference operator is useful for validating P2M and M2M independently
 * of local-expansion operators.  Field-only output is the default because
 * the magnetic field is the primary quantity of interest.
 */
[[nodiscard]] PotentialField evaluate(
    const MultiIndexSet& basis,
    const CoeffVector& multipole,
    const Vec3& source_centre,
    const Vec3& target,
    OutputFlags output = OutputFlags::Field);

} // namespace cdfmm::operators::m2p
