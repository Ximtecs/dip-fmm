// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <vector>

#include "cdfmm/operators.hpp"
#include "cdfmm/static_operators.hpp"
#include "cdfmm/uniform_tree.hpp"

namespace cdfmm::detail {

/** @brief Applies the canonical packed list1 tensor without allocating. */
void evaluate_static_near_field(const StaticP2POperator &p2p_operator,
                                std::span<const Vec3> sorted_dipole_moments,
                                std::span<Vec3> near_fields,
                                std::span<const int> sorted_self_indices);

/** @brief Evaluates list1 interactions with the mathematical reference P2P. */
void evaluate_reference_near_field(const UniformTree &tree,
                                   std::span<const Vec3> sorted_dipole_moments,
                                   std::span<const int> sorted_self_indices,
                                   OutputFlags output,
                                   std::span<PotentialField> sorted_results);

} // namespace cdfmm::detail
