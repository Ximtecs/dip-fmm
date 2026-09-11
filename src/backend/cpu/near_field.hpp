// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <vector>

#include "cdfmm/backend/cpu/static_plan_apply.hpp"
#include "cdfmm/operators.hpp"
#include "cdfmm/tree/static_topology.hpp"

namespace cdfmm::detail {

/** @brief Applies the default compact SoA list1 packing without allocating. */
void evaluate_static_near_field(const StaticP2PCompactPlan &p2p_plan,
                                std::span<const Vec3> sorted_dipole_moments,
                                std::span<Vec3> near_fields,
                                std::span<const int> sorted_self_indices);

/** @brief Evaluates list1 interactions with the mathematical reference P2P. */
void evaluate_reference_near_field(const StaticFmmTopology &topology,
                                   std::span<const Vec3> sorted_dipole_moments,
                                   std::span<const int> sorted_self_indices,
                                   OutputFlags output,
                                   std::span<PotentialField> sorted_results);

} // namespace cdfmm::detail
