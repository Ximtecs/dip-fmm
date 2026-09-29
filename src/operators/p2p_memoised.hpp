// SPDX-License-Identifier: Apache-2.0
#pragma once

// The canonical P2P builder as the chunked near-field construction calls it:
// the pair list is handed over rather than copied, and the exact operators
// built for one chunk are remembered for the next through the memo.  The
// public `build_static_p2p_operator` overloads are this entry without a
// memo; the rows either produces are bitwise the same.

#include <span>
#include <vector>

#include "cdfmm/geometry/models.hpp"
#include "cdfmm/geometry/primitives/rectangular_prism.hpp"
#include "cdfmm/geometry/primitives/tetrahedron.hpp"
#include "cdfmm/math/vec3.hpp"
#include "cdfmm/operators/p2p.hpp"
#include "operators/exact_operator_reuse.hpp"

namespace cdfmm::detail {

/// @brief Exact pair tensors remembered across the chunks of one build.
using ExactPairTensorMemo = exact_reuse::ExactOperatorMemo<PairTensor>;

/// @brief `build_static_p2p_operator` over @p interactions, taking the list
///        by value (it becomes the builder's sorted copy; a list already in
///        canonical order is not sorted again) and reusing the exact
///        operators @p memo holds from earlier calls.
[[nodiscard]] StaticP2POperator build_static_p2p_operator_memoised(
    std::span<const Vec3> target_positions,
    std::span<const Vec3> source_positions,
    std::vector<StaticP2PInteraction> interactions,
    SourceGeometry source_geometry,
    std::span<const RectangularPrism> source_prisms,
    std::span<const Tetrahedron> source_tetrahedra,
    TargetGeometry target_geometry,
    std::span<const RectangularPrism> target_prisms,
    std::span<const Tetrahedron> target_tetrahedra,
    SourceModel source_model,
    TargetModel target_model,
    ExactPairTensorMemo& memo);

} // namespace cdfmm::detail
