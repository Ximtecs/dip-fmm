// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <vector>

#include "cdfmm/operators.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// CUDA direct reference
//------------------------------------------------------------------------------

/**
 * @brief Evaluates the O(N^2) direct dipole sum on a CUDA device.
 *
 * This is a numerical and performance reference, not an FMM backend. Geometry
 * is uploaded when this convenience function is called; repeated workloads
 * should use a persistent direct plan in a future dedicated plan API.
 *
 * @param targets Target positions in user order.
 * @param sources Source positions in user order.
 * @param moments Dipole moments in source order.
 * @param output Requested potential and/or magnetic field components.
 * @param target_source_indices Optional source identity to omit per target.
 * @return Direct values in target order.
 */
[[nodiscard]] std::vector<PotentialField> cuda_direct_p2p_reference(
    std::span<const Vec3> targets,
    std::span<const Vec3> sources,
    std::span<const Vec3> moments,
    OutputFlags output = OutputFlags::Field,
    std::span<const int> target_source_indices = {}
);

} // namespace cdfmm
