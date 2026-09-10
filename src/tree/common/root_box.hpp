// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <span>

#include "cdfmm/math/vec3.hpp"

namespace cdfmm {

// Implementation-only result for the common physical root cube selected by a
// tree.  This header intentionally lives below src/ rather than include/ so
// root-box policy does not become part of the public tree API.
struct RootBox {
    Vec3 centre{};
    double half_width{0.0};
};

/**
 * @brief Resolve and validate the common cubic root for two populations.
 *
 * Both populations participate in inferred bounds.  A requested centre or
 * half-width replaces only its corresponding inferred value.  The point
 * containment check uses the historical boundary tolerance and is performed
 * before either tree materialises its topology.
 */
[[nodiscard]] RootBox resolve_root_box(
    std::span<const Vec3> source_positions,
    std::span<const Vec3> target_positions,
    std::optional<Vec3> root_centre,
    std::optional<double> root_half_width
);

} // namespace cdfmm
