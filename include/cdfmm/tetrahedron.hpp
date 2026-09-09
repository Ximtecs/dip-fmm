// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>

#include "cdfmm/geometry.hpp"
#include "cdfmm/multi_index.hpp"
#include "cdfmm/vec3.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Tetrahedral geometry
//------------------------------------------------------------------------------

/**
 * @brief An arbitrary non-degenerate tetrahedron in representative-relative
 * coordinates.
 *
 * `vertices` are offsets from the source or target representative point.  In
 * the usual convention the representative is the centroid, so their sum is
 * zero.  Keeping offsets here makes translations and topology permutations
 * explicit: the physical vertices are `representative + vertices[i]`.
 */
struct Tetrahedron {
    std::array<Vec3, 4> vertices{};

    /// @brief Returns the signed volume, with the supplied vertex ordering.
    [[nodiscard]] double signed_volume() const noexcept;
    /// @brief Returns the positive volume of the tetrahedron.
    [[nodiscard]] double volume() const noexcept;
    /// @brief Returns the mean of the four offsets (zero for centroided data).
    [[nodiscard]] Vec3 centroid_offset() const noexcept;
};

/**
 * @brief Validates a tetrahedron and returns its positive volume.
 *
 * A non-finite coordinate or a rank-deficient tetrahedron throws
 * `std::invalid_argument`.  The scale-aware rank test accepts valid geometry
 * over a wide range of physical scales while rejecting round-off-sized
 * volumes.
 */
[[nodiscard]] double tetrahedron_volume(const Tetrahedron& tetrahedron);

/**
 * @brief Exact uniformly magnetised tetrahedron-to-point pair tensor.
 *
 * The displacement is target representative minus source representative and
 * the result maps the *total* source moment to the field.  The implementation
 * sums the four oriented analytical triangular-face contributions adapted
 * from MagTense's `TileTriangle`/`TileNComponents` routines, then divides the
 * magnetisation tensor by the tetrahedron volume.
 *
 * Points on a face use the limiting value from the side selected by the
 * floating-point sign of the normal coordinate.  A point at an edge or vertex
 * is singular and may throw `std::domain_error` when the analytical limit is
 * not finite.
 */
[[nodiscard]] PairTensor tetrahedron_point_tensor(
    const Vec3& target_minus_source_representative,
    const Tetrahedron& source);

/**
 * @brief Exact point-to-tetrahedron tensor by reciprocity.
 *
 * A point source at `source_position` and a tetrahedral target at
 * `target_position` is equal to a tetrahedral source observed at the negated
 * displacement.  This avoids a second surface implementation and preserves
 * the total-moment normalisation.
 */
[[nodiscard]] PairTensor point_tetrahedron_tensor(
    const Vec3& target_minus_source_representative,
    const Tetrahedron& target);

/**
 * @brief Exact tetrahedral volume average of a factorial-normalised monomial.
 *
 * This returns
 * `J_beta(d) = V^-1 integral_T (d + u)^beta / beta! dV`, where `u` is the
 * representative-relative tetrahedron coordinate.  The expansion is a
 * finite barycentric polynomial sum; no numerical quadrature is used.
 */
[[nodiscard]] double tetrahedron_averaged_monomial(
    const MultiIndex& beta,
    const Vec3& d,
    const Tetrahedron& tetrahedron);

/**
 * @brief Deliberately unsupported finite tetrahedron-to-tetrahedron P2P pair.
 *
 * No quadrature fallback is provided.  Callers should use the exact target
 * L2P path or report this construction-time limitation to the user.
 */
[[nodiscard]] PairTensor tetrahedron_tetrahedron_tensor(
    const Vec3& target_minus_source_representative,
    const Tetrahedron& source,
    const Tetrahedron& target);

} // namespace cdfmm
