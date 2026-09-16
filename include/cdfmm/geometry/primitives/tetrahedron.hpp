// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>

#include "cdfmm/geometry/primitives/rectangular_prism.hpp"
#include "cdfmm/math/pair_tensor.hpp"
#include "cdfmm/math/multi_index.hpp"
#include "cdfmm/math/vec3.hpp"

namespace cdfmm {

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
 */
[[nodiscard]] PairTensor tetrahedron_point_tensor(
    const Vec3& target_minus_source_representative,
    const Tetrahedron& source
);

/** @brief Exact point-to-tetrahedron tensor by reciprocity. */
[[nodiscard]] PairTensor point_tetrahedron_tensor(
    const Vec3& target_minus_source_representative,
    const Tetrahedron& target
);

/**
 * @brief Exact tetrahedral volume average of a factorial-normalised monomial.
 */
[[nodiscard]] double tetrahedron_averaged_monomial(
    const MultiIndex& beta,
    const Vec3& d,
    const Tetrahedron& tetrahedron
);

/**
 * @brief Exact uniformly magnetised tetrahedron-to-tetrahedron pair tensor.
 */
[[nodiscard]] PairTensor tetrahedron_tetrahedron_tensor(
    const Vec3& target_minus_source_representative,
    const Tetrahedron& source,
    const Tetrahedron& target
);

/**
 * @brief Exact rectangular-prism-to-tetrahedron pair tensor.
 *
 * The prism is centred on the source representative and the tetrahedron on
 * the target representative.  Both bodies are closed polyhedra, so the
 * volume-averaged field of one uniformly magnetised body over the other is
 * the same double surface integral used by the tetrahedron pair:
 * `K = -1/(4 pi V_s V_t) sum_{faces} n_t n_s^T integral integral 1/R`.  The
 * prism contributes its twelve boundary triangles; no numerical quadrature is
 * involved.  The result maps the total source moment to the target-averaged
 * field.
 */
[[nodiscard]] PairTensor rectangular_prism_tetrahedron_tensor(
    const Vec3& target_minus_source_representative,
    const RectangularPrism& source,
    const Tetrahedron& target
);

/** @brief Exact tetrahedron-to-rectangular-prism pair tensor (see above). */
[[nodiscard]] PairTensor tetrahedron_rectangular_prism_tensor(
    const Vec3& target_minus_source_representative,
    const Tetrahedron& source,
    const RectangularPrism& target
);

} // namespace cdfmm
