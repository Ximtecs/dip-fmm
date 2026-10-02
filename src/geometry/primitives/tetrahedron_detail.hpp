// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <span>
#include <variant>
#include <vector>

#include "cdfmm/geometry/primitives/rectangular_prism.hpp"
#include "cdfmm/geometry/primitives/tetrahedron.hpp"
#include "cdfmm/math/vec3.hpp"

namespace cdfmm::detail {

/**
 * @brief Triangulated closed surface of a uniformly magnetised polyhedron.
 *
 * Every triangle is stored representative-relative together with its unit
 * outward normal.  This is the geometry-neutral input of the exact
 * polyhedron pair tensor: a tetrahedron contributes four triangles, an
 * axis-aligned rectangular prism twelve.
 */
struct PolyhedronSurface {
    double volume{0.0};
    /// Largest vertex distance from the representative point.
    double circumradius{0.0};
    std::vector<std::array<Vec3, 3>> faces{};
    std::vector<Vec3> outward_normals{};
};

/**
 * @brief A finite body prepared for exact pair-tensor construction.
 *
 * The surface feeds the analytical double surface integral, while the
 * geometric record supplies the body's exact point-field tensor for widely
 * separated pairs (see `polyhedron_pair_tensor`).
 */
struct PolyhedronBody {
    PolyhedronSurface surface{};
    std::variant<RectangularPrism, Tetrahedron> record{};
};

[[nodiscard]] PolyhedronBody prepare_polyhedron_body(
    const RectangularPrism& prism);
[[nodiscard]] PolyhedronBody prepare_polyhedron_body(
    const Tetrahedron& tetrahedron);

/**
 * @brief Separation, in units of the summed circumradii, from which the pair
 * tensor is averaged by Gauss quadrature of the exact source point tensor
 * over the target instead of formed from the analytical surface integrals.
 *
 * The analytical reduction is ill-conditioned for separated irregular pairs:
 * on random irregular tetrahedra its error grows from a median of 3e-11 at
 * 1.5 circumradii to 2e-8 at eight, with outliers up to 1.5e-5 where edges
 * are nearly parallel, and it loses all digits by a hundred body sizes. Below
 * 1.5 the source's singularity is too close for a fixed rule and the
 * analytical integral is the more accurate one.
 */
inline constexpr double polyhedron_near_quadrature_factor = 1.5;

/**
 * @brief The quadrature ladder: from each separation (summed circumradii) the
 * one-dimensional collapsed Gauss rule of that many points is used.
 *
 * Each rule is the cheapest whose worst relative error over random irregular
 * tetrahedra in its range, against a 16-point rule, is at most about 2e-10:
 * seven points from 1.5 (1.8e-10), six from 2 (1.3e-10) and five from 4
 * (4.4e-11, 1.5e-13 beyond 8). That is 100 to 1000 times below the worst
 * error of the analytical reduction in the same ranges, at a cost below it.
 */
inline constexpr double polyhedron_quadrature_six_point_factor = 2.0;
inline constexpr double polyhedron_quadrature_five_point_factor = 4.0;

/**
 * @brief Exact pair tensor of two prepared bodies, choosing the analytical
 * surface integrals or the far-separation quadrature (see above).
 */
[[nodiscard]] PairTensor polyhedron_pair_tensor(
    const Vec3& target_minus_source_representative,
    const PolyhedronBody& source,
    const PolyhedronBody& target);

/** @brief Twelve oriented boundary triangles of a prism centred on its representative. */
[[nodiscard]] PolyhedronSurface prepare_rectangular_prism_surface(
    const RectangularPrism& prism);

/** @brief The four boundary triangles of a tetrahedron in the surface form. */
[[nodiscard]] PolyhedronSurface prepare_tetrahedron_surface(
    const Tetrahedron& tetrahedron);

/**
 * @brief Exact mutual tensor of two uniformly magnetised polyhedra.
 *
 * Applies the divergence theorem twice: the target-averaged field of the
 * source is `-1/(4 pi V_s V_t) sum_t sum_s n_t n_s^T I(f_t + d, f_s)` where
 * `I` is the constant-density triangle pair Laplace integral and `d` the
 * displacement between the representatives.  The face pairs may touch,
 * share edges or coincide; only degenerate triangles are rejected.
 */
[[nodiscard]] PairTensor polyhedron_polyhedron_tensor(
    const Vec3& target_minus_source_representative,
    std::span<const std::array<Vec3, 3>> source_faces,
    std::span<const Vec3> source_outward_normals,
    double source_volume,
    std::span<const std::array<Vec3, 3>> target_faces,
    std::span<const Vec3> target_outward_normals,
    double target_volume);

/**
 * @brief Local orthonormal frame and in-plane extents of one oriented face.
 *
 * The frame depends only on the face's vertices, so it is constant for a
 * given body while the evaluation point varies.
 */
struct FaceFrame {
    Vec3 e1{};
    Vec3 e2{};
    Vec3 normal{};
    Vec3 origin{};
    double x_first{0.0};
    double x_third{0.0};
    double height{0.0};
};

/**
 * @brief A tetrahedron's point-field geometry, derived once per record.
 *
 * `tetrahedron_magnetisation_tensor` re-derives the volume, the largest edge
 * length and all four face frames on every evaluation, although every one of
 * them is a pure function of the record.  That is wasteful wherever one body
 * is evaluated at many points -- most of all in the far-separation
 * quadrature, which evaluates a single source at 216 nodes.
 */
struct PreparedTetrahedronPointField {
    double volume{0.0};
    /// Largest edge length, which sets the singularity tolerance.
    double geometry_scale{0.0};
    std::array<Vec3, 4> vertices{};
    /// One frame per face, in the order the surface sum visits them.
    std::array<FaceFrame, 4> faces{};
};

[[nodiscard]] PreparedTetrahedronPointField prepare_tetrahedron_point_field(
    const Tetrahedron& tetrahedron);

/// @brief Exact point field of a prepared uniformly magnetised tetrahedron.
[[nodiscard]] PairTensor tetrahedron_point_tensor_prepared(
    const Vec3& target_minus_source_representative,
    const PreparedTetrahedronPointField& source);

struct PreparedTetrahedron {
    double volume{0.0};

    // Largest vertex distance from the representative point.
    double circumradius{0.0};

    // Representative-relative vertices.
    std::array<Vec3, 4> vertices{};

    // Representative-relative triangular faces.
    std::array<std::array<Vec3, 3>, 4> faces{};

    // Unit outward normal corresponding to each face.
    std::array<Vec3, 4> outward_normals{};
};

[[nodiscard]] PreparedTetrahedron prepare_tetrahedron(
    const Tetrahedron& tetrahedron);

[[nodiscard]] PairTensor tetrahedron_tetrahedron_tensor_prepared(
    const Vec3& target_minus_source_representative,
    const PreparedTetrahedron& source,
    const PreparedTetrahedron& target,
    bool coincident_same_geometry);

/**
 * @brief Evaluates the unnormalised constant-density 1/R triangle pair
 * integral.
 *
 * This header is intentionally private to analytical-kernel tests.  The
 * production geometry API exposes only the resulting tetrahedron tensor.
 */
[[nodiscard]] double triangle_triangle_laplace_integral(
    const std::array<Vec3, 3>& first,
    const std::array<Vec3, 3>& second);

} // namespace cdfmm::detail
