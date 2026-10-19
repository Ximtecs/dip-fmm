// SPDX-License-Identifier: Apache-2.0
//
// Exact analytical operators of a uniformly magnetised tetrahedron.  Three
// formulations live here, all expressed as symmetric pair tensors T with
// H = T m for the source's total moment m = V M (docs/math/finite-geometry.md):
//
//   * tetrahedron -> point: the surface-charge picture.  The field of a
//     uniformly magnetised body is the sum over its faces of (M . n) times
//     the field of a uniformly charged triangle, evaluated with the
//     analytical edge primitives adapted from MagTense's TileTriangle
//     (in-plane components) and the Van Oosterom-Strackee solid angle
//     (normal component);
//   * point -> tetrahedron: the same tensor by reciprocity;
//   * tetrahedron -> tetrahedron (and prism <-> tetrahedron): the Galerkin
//     double surface integral, where each face pair contributes the exact
//     integral of 1/|x - y| over two triangles (Gumerov, Kaneko and
//     Duraiswami, SIAM J. Sci. Comput. 46 (2024)) times the outer product
//     of the outward normals.  The reduction runs in quadruple precision;
//     nearly degenerate face pairs, where it is unreliable whatever the
//     precision, are integrated numerically instead (Sauter-Schwab for
//     touching triangles, split Gauss quadrature otherwise).  Separated
//     pairs (from 1.5 summed circumradii) switch to a converged Gauss rule
//     over the target of the exact source field, because the analytical
//     reduction loses accuracy through cancellation exactly where
//     quadrature becomes accurate.
//
// Vertices are representative-relative.  Degenerate faces, evaluation on an
// edge or vertex, and a lost rank in the dimensional reduction are reported
// as exceptions rather than patched with a limiting value.  Near-degenerate
// configurations are different: nearly parallel face pairs and ulp-level
// heights are resolved to the exactly degenerate case they approximate, and
// a nearly dependent expansion direction close to the rank tolerance sends
// the face pair to quadrature (docs/math/finite-geometry.md), because the
// reduction's general branches cancel catastrophically there.  The `Prepared*`
// records hoist everything that depends on the body alone (frames, normals,
// circumradius) so that a body used in many pairs derives it once.

#include "cdfmm/geometry/primitives/tetrahedron.hpp"

#include "tetrahedron_detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace cdfmm {
namespace {

constexpr long double four_pi = 4.0L * std::numbers::pi_v<long double>;

// Full 3x3 working tensor; results are symmetrised and reduced to the public
// six-component PairTensor at the end.
struct Matrix3 {
    double value[3][3]{};
};

// The face opposite vertex i, oriented so that the right-hand normal points
// outwards for a positively oriented tetrahedron.
constexpr std::array<std::array<int, 3>, 4> tetrahedron_face_vertices{{
    {{1, 2, 3}},
    {{0, 3, 2}},
    {{0, 1, 3}},
    {{0, 2, 1}},
}};

inline void add_scaled_outer_product(Matrix3& matrix,
                                     const double factor,
                                     const Vec3& left,
                                     const Vec3& right) noexcept
{
    matrix.value[0][0] += factor * left.x * right.x;
    matrix.value[0][1] += factor * left.x * right.y;
    matrix.value[0][2] += factor * left.x * right.z;

    matrix.value[1][0] += factor * left.y * right.x;
    matrix.value[1][1] += factor * left.y * right.y;
    matrix.value[1][2] += factor * left.y * right.z;

    matrix.value[2][0] += factor * left.z * right.x;
    matrix.value[2][1] += factor * left.z * right.y;
    matrix.value[2][2] += factor * left.z * right.z;
}

Vec3 cross(const Vec3& a, const Vec3& b)
{
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

double norm(const Vec3& value)
{
    return std::sqrt(dot(value, value));
}

Vec3 scale(const Vec3& value, const double factor)
{
    return {factor * value.x, factor * value.y, factor * value.z};
}

bool finite(const Vec3& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z);
}

// Unit outward normal of `face`, fixed by requiring the opposite vertex to
// lie behind it, so the result does not depend on the vertex ordering.
Vec3 tetrahedron_outward_normal(
    const std::array<Vec3, 4>& vertices,
    const std::array<int, 3>& face,
    const int opposite)
{
    const Vec3 first = vertices[static_cast<std::size_t>(face[0])];
    const Vec3 second = vertices[static_cast<std::size_t>(face[1])];
    const Vec3 third = vertices[static_cast<std::size_t>(face[2])];
    Vec3 normal = cross(second - first, third - first);
    if (dot(normal,
            vertices[static_cast<std::size_t>(opposite)] - first) > 0.0) {
        normal = -1.0 * normal;
    }
    const double length = norm(normal);
    if (!(length > 0.0) || !std::isfinite(length)) {
        throw std::invalid_argument("tetrahedron face is degenerate");
    }
    return scale(normal, 1.0 / length);
}

// The limiting value used by TileTriangle when a point lies exactly on a
// local coordinate plane.  It selects the positive side for an exact zero;
// callers can supply a signed value for a one-sided limit.
double nonzero_coordinate(const double value)
{
    if (value != 0.0) {
        return value;
    }
    constexpr double threshold = 64.0 * std::numeric_limits<double>::epsilon();
    return threshold;
}

// Difference atanh(a2/R2) - atanh(a1/R1) of the edge primitive between the
// two endpoints of one straight edge.  `a1`, `a2` are the signed positions of
// the endpoints along the edge line measured from the foot of the evaluation
// point, and `b2` is the squared perpendicular distance of the point from that
// line, so R_i = sqrt(a_i^2 + b2) are the endpoint distances.  Written as
// atanh(a/R) the two terms diverge like ln(1/b) whenever the point is
// collinear with the edge; their difference is finite when the foot lies
// outside the segment and the logs of b cancel exactly in the closed form
// below, which removes the catastrophic cancellation of the naive quotient.
long double edge_atanh_difference(const long double a1, const long double a2,
                                  const long double b2)
{
    const long double r1 = std::sqrt(a1 * a1 + b2);
    const long double r2 = std::sqrt(a2 * a2 + b2);
    if (a1 * a2 > 0.0L) {
        const long double sign = a1 > 0.0L ? 1.0L : -1.0L;
        return sign * std::log((r2 + std::abs(a2)) / (r1 + std::abs(a1)));
    }
    // The foot lies on the segment: each term is finite for b2 > 0 and the
    // edge itself (b2 == 0) is a genuine singularity, reported as infinite so
    // the caller's finiteness check rejects it.
    const auto term = [b2](const long double a, const long double r) {
        if (a == 0.0L) {
            return 0.0L;
        }
        if (!(b2 > 0.0L)) {
            return std::copysign(std::numeric_limits<long double>::infinity(),
                                 a);
        }
        return std::copysign(std::log((r + std::abs(a)) / std::sqrt(b2)), a);
    };
    return term(a2, r2) - term(a1, r1);
}

// The following three functions are double-precision adaptations of
// TileTriangle.f90's Nxz, Nyz and Nzz primitives.  Their arguments describe a
// right-triangle representation in the local face frame: the base endpoints
// are (x=0,l) and the apex has coordinates (0,h), with h>0.  The expressions
// are the analytical edge primitives, not numerical quadrature.  The atanh
// edge terms are evaluated through `edge_atanh_difference`, which is exact for
// evaluation points collinear with an edge outside the triangle.
double triangle_nxz(const Vec3& r, const double l, const double h)
{
    const long double rx = r.x;
    const long double ry = r.y;
    const long double rz = r.z;
    const long double L = l;
    const long double H = h;
    const long double diagonal = std::hypot(L, H);
    // Hypotenuse from (L, 0) to (0, H): signed endpoint positions along the
    // edge direction (-L, H)/|diagonal| relative to the point's foot.
    const long double hypotenuse_offset = (L * L - L * rx + H * ry) / diagonal;
    const long double hypotenuse_distance = (H * rx + L * ry - L * H) / diagonal;
    const long double hypotenuse_b2 =
        hypotenuse_distance * hypotenuse_distance + rz * rz;
    const long double f_difference = H / diagonal * edge_atanh_difference(
        hypotenuse_offset, hypotenuse_offset - diagonal, hypotenuse_b2);
    // Vertical edge from (0, 0) to (0, H).
    const long double g_difference = edge_atanh_difference(
        ry, ry - H, rx * rx + rz * rz);
    return static_cast<double>(-(f_difference - g_difference) / four_pi);
}

double triangle_nyz(const Vec3& r, const double l, const double h)
{
    const long double rx = r.x;
    const long double ry = r.y;
    const long double rz = r.z;
    const long double L = l;
    const long double H = h;
    const long double diagonal = std::hypot(L, H);
    // Hypotenuse from (0, H) to (L, 0): signed endpoint positions along the
    // edge direction (L, -H)/|diagonal| relative to the point's foot.
    const long double hypotenuse_offset = (H * H - H * ry + L * rx) / diagonal;
    const long double hypotenuse_distance = (H * rx + L * ry - L * H) / diagonal;
    const long double hypotenuse_b2 =
        hypotenuse_distance * hypotenuse_distance + rz * rz;
    const long double k_difference = L / diagonal * edge_atanh_difference(
        hypotenuse_offset, hypotenuse_offset - diagonal, hypotenuse_b2);
    // Horizontal edge from (0, 0) to (L, 0).
    const long double ell_difference = edge_atanh_difference(
        rx, rx - L, ry * ry + rz * rz);
    return static_cast<double>(-(k_difference - ell_difference) / four_pi);
}

// Normal component of the face integral: -Omega/(4 pi), where Omega is the
// signed solid angle of the local triangle (x_first, 0), (0, h), (x_third, 0)
// seen from r.  Omega = 2 atan2(a.(b x c), |a||b||c| + (a.b)|c| + (b.c)|a|
// + (c.a)|b|) with a, b, c the vertex vectors from the point.
double triangle_solid_angle_column(const Vec3& r, const double x_first,
                                   const double x_third, const double h)
{
    const long double ax = static_cast<long double>(x_first) - r.x;
    const long double ay = -static_cast<long double>(r.y);
    const long double az = -static_cast<long double>(r.z);
    const long double bx = -static_cast<long double>(r.x);
    const long double by = static_cast<long double>(h) - r.y;
    const long double bz = az;
    const long double cx = static_cast<long double>(x_third) - r.x;
    const long double cy = ay;
    const long double cz = az;
    const long double a_norm = std::sqrt(ax * ax + ay * ay + az * az);
    const long double b_norm = std::sqrt(bx * bx + by * by + bz * bz);
    const long double c_norm = std::sqrt(cx * cx + cy * cy + cz * cz);
    const long double triple = ax * (by * cz - bz * cy) +
        ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx);
    const long double denominator = a_norm * b_norm * c_norm +
        (ax * bx + ay * by + az * bz) * c_norm +
        (bx * cx + by * cy + bz * cz) * a_norm +
        (cx * ax + cy * ay + cz * az) * b_norm;
    const long double solid_angle = 2.0L * std::atan2(triple, denominator);
    return static_cast<double>(-solid_angle / four_pi);
}

using detail::FaceFrame;

// Local frame of one face for the TileTriangle primitives: e1 along the base
// edge, the normal outward, e2 in-plane, origin at the foot of the apex, so
// the face is the union of two right triangles with x-extents `x_first` (> 0)
// and `x_third` (< 0) and common height.  `face[3]` is the opposite vertex.
FaceFrame make_face_frame(std::array<Vec3, 4> face)
{
    const Vec3 first_edge = face[0] - face[1];
    const Vec3 second_edge = face[1] - face[2];
    const Vec3 third_edge = face[2] - face[0];
    const double first_length = norm(first_edge);
    const double second_length = norm(second_edge);
    const double third_length = norm(third_edge);
    if (!(first_length > 0.0 && second_length > 0.0 && third_length > 0.0)) {
        throw std::invalid_argument("tetrahedron face has coincident vertices");
    }

    // TileTriangle first places the largest angle at vertex 2.  In that
    // ordering the projection of the apex lies between the base endpoints,
    // making x_first strictly positive and keeping the edge primitives away
    // from their l=0 algebraic branch.
    const double angle0 = std::acos(std::clamp(
        dot(face[0] - face[1], face[0] - face[2]) /
            (norm(face[0] - face[1]) * norm(face[0] - face[2])), -1.0, 1.0));
    const double angle1 = std::acos(std::clamp(
        dot(face[1] - face[0], face[1] - face[2]) /
            (norm(face[1] - face[0]) * norm(face[1] - face[2])), -1.0, 1.0));
    const double angle2 = std::acos(std::clamp(
        dot(face[2] - face[0], face[2] - face[1]) /
            (norm(face[2] - face[0]) * norm(face[2] - face[1])), -1.0, 1.0));
    const std::array<double, 3> angles{{angle0, angle1, angle2}};
    std::array<int, 3> order{{0, 1, 2}};
    std::sort(order.begin(), order.end(), [&](const int a, const int b) {
        return angles[static_cast<std::size_t>(a)] <
            angles[static_cast<std::size_t>(b)];
    });
    const std::array<Vec3, 3> ordered{{
        face[static_cast<std::size_t>(order[0])],
        face[static_cast<std::size_t>(order[2])],
        face[static_cast<std::size_t>(order[1])]}};
    face[0] = ordered[0];
    face[1] = ordered[1];
    face[2] = ordered[2];

    auto recompute_frame = [&]() {
        const Vec3 edge13 = face[0] - face[2];
        const double edge13_length = norm(edge13);
        FaceFrame frame;
        frame.e1 = scale(edge13, 1.0 / edge13_length);
        frame.normal = cross(frame.e1, face[1] - face[2]);
        const double normal_length = norm(frame.normal);
        if (!(normal_length > 0.0)) {
            throw std::invalid_argument("tetrahedron face is degenerate");
        }
        frame.normal = scale(frame.normal, 1.0 / normal_length);
        frame.e2 = cross(frame.normal, frame.e1);
        frame.origin = face[2] + scale(
            frame.e1, dot(face[1] - face[2], frame.e1));
        frame.x_first = dot(face[0] - frame.origin, frame.e1);
        frame.x_third = dot(face[2] - frame.origin, frame.e1);
        frame.height = dot(face[1] - frame.origin, frame.e2);
        return frame;
    };

    FaceFrame frame = recompute_frame();
    // The opposite tetrahedron vertex is behind the outward normal.  If it is
    // in front, reverse the cyclic edge orientation and recompute all axes.
    if (dot(frame.normal, face[3] - face[0]) > 0.0) {
        std::swap(face[0], face[2]);
        frame = recompute_frame();
    }
    if (!(frame.x_first > 0.0 && frame.height > 0.0)) {
        throw std::invalid_argument("tetrahedron face has invalid orientation");
    }
    return frame;
}

/// @brief The face's contribution at one point, given its constant frame.
Matrix3 face_tensor_prepared(const FaceFrame& frame,
                             const Vec3& target_relative_position)
{
    const Vec3 local_position = target_relative_position - frame.origin;
    Vec3 r{
        dot(local_position, frame.e1),
        dot(local_position, frame.e2),
        dot(local_position, frame.normal)};
    r.x = nonzero_coordinate(r.x);
    r.y = nonzero_coordinate(r.y);
    r.z = nonzero_coordinate(r.z);

    // The local result has only its third column.  Transforming it as
    // P*N_local*P^T gives (integral R/R^3) tensor-product normal^T for this
    // face, exactly as in MagTense's getN_Triangle.  The in-plane entries are
    // the edge (log) primitives; the normal entry is the flux of R/R^3
    // through the face, i.e. the signed solid angle the triangle subtends at
    // the point, evaluated with the Van Oosterom-Strackee formula, which is
    // well conditioned everywhere except on the face itself (the nudged
    // normal coordinate keeps the one-sided convention for exact zeros).
    const double local_column[3]{
        triangle_nxz(r, frame.x_first, frame.height) -
            triangle_nxz(r, frame.x_third, frame.height),
        triangle_nyz(r, frame.x_first, frame.height) -
            triangle_nyz(r, frame.x_third, frame.height),
        triangle_solid_angle_column(r, frame.x_first, frame.x_third,
                                    frame.height)};
    const Vec3 global_column = frame.e1 * local_column[0] +
        frame.e2 * local_column[1] + frame.normal * local_column[2];
    Matrix3 result;
    const Vec3 basis[3]{frame.e1, frame.e2, frame.normal};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result.value[row][column] = global_column[row] * basis[2][column];
        }
    }
    return result;
}

// Sum of the four face tensors, i.e. the demagnetisation tensor N with
// H = -N M of the whole tetrahedron at a point, then divided by the volume so
// the returned tensor acts on the total moment.  Evaluation on a vertex or an
// edge is a genuine singularity and is rejected up front.
Matrix3 tetrahedron_magnetisation_tensor_prepared(
    const Vec3& target_relative_position,
    const detail::PreparedTetrahedronPointField& source)
{
    const std::array<Vec3, 4>& vertices = source.vertices;
    const double singular_tolerance =
        256.0 * std::numeric_limits<double>::epsilon() * source.geometry_scale;
    for (int i = 0; i < 4; ++i) {
        if (norm(target_relative_position -
                 vertices[static_cast<std::size_t>(i)]) <=
            singular_tolerance) {
            throw std::domain_error(
                "tetrahedron-to-point tensor is singular at a vertex");
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            const Vec3 edge = vertices[static_cast<std::size_t>(j)] -
                vertices[static_cast<std::size_t>(i)];
            const double edge_length_squared = dot(edge, edge);
            const double parameter = dot(
                target_relative_position -
                    vertices[static_cast<std::size_t>(i)], edge) /
                edge_length_squared;
            if (parameter >= 0.0 && parameter <= 1.0) {
                const Vec3 closest = vertices[static_cast<std::size_t>(i)] +
                    scale(edge, parameter);
                if (norm(target_relative_position - closest) <=
                    singular_tolerance) {
                    throw std::domain_error(
                        "tetrahedron-to-point tensor is singular at an edge");
                }
            }
        }
    }
    Matrix3 result;
    for (int missing_vertex = 0; missing_vertex < 4; ++missing_vertex) {
        const Matrix3 contribution = face_tensor_prepared(
            source.faces[static_cast<std::size_t>(missing_vertex)],
            target_relative_position);
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                result.value[row][column] += contribution.value[row][column];
            }
        }
    }

    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            if (!std::isfinite(result.value[row][column])) {
                throw std::domain_error(
                    "tetrahedron-to-point tensor is singular at an edge or vertex");
            }
        }
    }

    // The surface sum is symmetric analytically.  Symmetrising suppresses
    // the harmless last-bit asymmetry introduced by independently evaluated
    // face primitives and ensures the public six-component representation is
    // deterministic under vertex permutations.
    for (int row = 0; row < 3; ++row) {
        for (int column = row + 1; column < 3; ++column) {
            const double average = 0.5 * (result.value[row][column] +
                                          result.value[column][row]);
            result.value[row][column] = average;
            result.value[column][row] = average;
        }
    }
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result.value[row][column] /= source.volume;
        }
    }
    return result;
}

Matrix3 tetrahedron_magnetisation_tensor(
    const Vec3& target_relative_position, const Tetrahedron& source)
{
    return tetrahedron_magnetisation_tensor_prepared(
        target_relative_position,
        detail::prepare_tetrahedron_point_field(source));
}

PairTensor to_pair_tensor(const Matrix3& matrix)
{
    return {matrix.value[0][0], matrix.value[0][1], matrix.value[0][2],
            matrix.value[1][1], matrix.value[1][2], matrix.value[2][2]};
}

//------------------------------------------------------------------------------
// Analytical constant-density triangle Galerkin integral
//------------------------------------------------------------------------------

// This is a focused C++ port of the constant-density L path in
// GalerkinLaplaceTriGS.m and its I3/I2s/I2t/I1/I0 dependencies from
// Gumerov, Kaneko, and Duraiswami, SIAM J. Sci. Comput. 46 (2024),
// DOI 10.1137/23M1547688.  The upstream reference implementation is MIT
// licensed: https://github.com/pirl-lab/analytical-quadrature-laplace-galerkin.
//
// NOTE(cdfmm): accuracy.  The reduction is exact in exact arithmetic, but its
// terms carry coefficients that grow without bound when the geometry is
// nearly degenerate (nearly parallel planes, nearly collinear or coplanar
// edge vectors), and it drops or snaps those directions with fixed
// tolerances.  In double precision that lost every digit for parallel faces
// with non-aligned edges -- the top and bottom faces of an unstructured film
// mesh -- with errors up to 24 times the tensor against an independent
// Sauter-Schwab quadrature (440-pair validation of 19 Oct 2026, shared with
// MagTense's port of this reduction).  Therefore:
//   * the reduction runs in quadruple precision (`ReductionReal`), with
//     tolerances chosen for it rather than scaled from double precision;
//   * a Gram-Schmidt rank decision close to the rank tolerance, where
//     neither dropping nor keeping the direction is reliable, flags the face
//     pair, which is then integrated numerically: Sauter-Schwab quadrature
//     for touching triangles, split Gauss quadrature otherwise;
//   * triangles without a common vertex in nearly, but not exactly, parallel
//     planes are integrated by split Gauss quadrature directly, because the
//     reduction expands their offset in a nearly degenerate basis whatever
//     the precision.
// The quadratures run in double precision: their integrand is positive, so
// they do not cancel.

#if defined(__FLT128_MANT_DIG__)
// GCC/Clang quadruple precision with the glibc *f128 functions (libm).
using ReductionReal = _Float128;

// Square root from the double-precision root and one Newton step, which
// doubles its 53 bits; glibc's software sqrtf128 was 35 % of the construction
// time.  Arguments outside the double range use sqrtf128.
inline ReductionReal reduction_sqrt(const ReductionReal x)
{
    const double approximation = std::sqrt(static_cast<double>(x));
    if (!(approximation > 0.0) || !std::isfinite(approximation) ||
        approximation < 1.0e-150 || approximation > 1.0e150) {
        return sqrtf128(x);
    }
    const ReductionReal y = approximation;
    return (y + x / y) / 2;
}
inline ReductionReal reduction_log(const ReductionReal x) { return logf128(x); }
inline ReductionReal reduction_atan(const ReductionReal x) { return atanf128(x); }
inline ReductionReal reduction_asinh(const ReductionReal x) { return asinhf128(x); }
inline ReductionReal reduction_hypot(const ReductionReal x, const ReductionReal y)
{
    return hypotf128(x, y);
}
inline ReductionReal reduction_abs(const ReductionReal x) { return fabsf128(x); }
#else
// WARNING(cdfmm): without a quadruple-precision type the reduction runs in
// long double, where the tolerances below are too tight for the round-off of
// nearly degenerate pairs; the guard band then catches fewer of them.
using ReductionReal = long double;

inline ReductionReal reduction_sqrt(const ReductionReal x) { return std::sqrt(x); }
inline ReductionReal reduction_log(const ReductionReal x) { return std::log(x); }
inline ReductionReal reduction_atan(const ReductionReal x) { return std::atan(x); }
inline ReductionReal reduction_asinh(const ReductionReal x) { return std::asinh(x); }
inline ReductionReal reduction_hypot(const ReductionReal x, const ReductionReal y)
{
    return std::hypot(x, y);
}
inline ReductionReal reduction_abs(const ReductionReal x) { return std::abs(x); }
#endif

// Double-precision versions of the same functions, for the reduction of
// well-conditioned face pairs (see kScreenBand).
inline double reduction_sqrt(const double x) { return std::sqrt(x); }
inline double reduction_log(const double x) { return std::log(x); }
inline double reduction_atan(const double x) { return std::atan(x); }
inline double reduction_asinh(const double x) { return std::asinh(x); }
inline double reduction_hypot(const double x, const double y) { return std::hypot(x, y); }
inline double reduction_abs(const double x) { return std::abs(x); }
#if defined(__FLT128_MANT_DIG__)
// Long double versions, for I0 of the double-precision reduction (as before
// the quadruple-precision change, its logarithms and arctangents cancel).
inline long double reduction_sqrt(const long double x) { return std::sqrt(x); }
inline long double reduction_log(const long double x) { return std::log(x); }
inline long double reduction_atan(const long double x) { return std::atan(x); }
inline long double reduction_asinh(const long double x) { return std::asinh(x); }
inline long double reduction_hypot(const long double x, const long double y)
{
    return std::hypot(x, y);
}
inline long double reduction_abs(const long double x) { return std::abs(x); }
#endif

constexpr ReductionReal reduction_real(const long double value) noexcept
{
    return static_cast<ReductionReal>(value);
}

// Tolerances of the reduction, all relative to unit-scale geometry (the
// triangle pair is normalised before the core).
template <typename Real>
struct ReductionTolerances;

// Quadruple precision, for every face pair the screen (kScreenBand) does not
// clear.  The validation that fixed them is described above; a decade either
// way on the rank tolerance fails on some face pair of an unstructured film
// mesh.
template <>
struct ReductionTolerances<ReductionReal> {
    // A coefficient below this drops its term.  Dropping is what keeps
    // touching triangles finite; the former double-precision 1e-14 dropped
    // real terms of nearly degenerate pairs (errors up to 7e-8).
    static constexpr ReductionReal zero = reduction_real(1.0e-22L);
    // Squared relative test of the I0 case selection for vanishing heights.
    static constexpr ReductionReal i0_zero_squared = reduction_real(1.0e-24L);
    // Heights below this fraction of the vector scale are zero.  A nearly
    // shared vertex of double-precision input leaves heights of a few double
    // ulps, which must be snapped; the I0 forms lose about eps / h^2, which
    // this keeps below 1e-10 in quadruple precision.
    static constexpr ReductionReal relative_residual = reduction_real(1.0e-12L);
    // Gram-Schmidt rank tolerance, squared and relative to the largest basis
    // vector: directions down to 1e-23 must be kept, those below it dropped.
    static constexpr ReductionReal rank_squared = reduction_real(1.0e-23L);
    // A rank decision with a squared relative norm in [guard_low, guard_high)
    // is unreliable and sends the face pair to quadrature (about 2 % of the
    // touching face pairs of an unstructured film mesh).
    static constexpr ReductionReal guard_low = reduction_real(1.0e-26L);
    static constexpr ReductionReal guard_high = reduction_real(1.0e-20L);
    // Face pairs whose normals differ by less than this are projected onto a
    // common plane and evaluated by the parallel branch: the projection costs
    // about 0.1 x angle, the general branch's round-off grows like
    // eps / angle^3, both below 1e-9 here in quadruple precision.
    static constexpr ReductionReal near_parallel = reduction_real(1.0e-8L);
};

// Double precision, for face pairs the screen clears: they have no nearly
// degenerate direction, so these are the former double-precision values and
// the guard is off.
template <>
struct ReductionTolerances<double> {
    static constexpr double zero = 1.0e-14;
    static constexpr double i0_zero_squared = 1.0e-28;
    static constexpr double relative_residual = 1.0e-10;
    static constexpr double rank_squared = 256.0 * std::numeric_limits<double>::epsilon();
    static constexpr double guard_low = 0.0;
    static constexpr double guard_high = 0.0;
    static constexpr double near_parallel = 1.0e-3;
};

// The screen: a face pair whose smallest non-exact measure of near-degeneracy
// (angle between the normals, between an edge and the other plane, between
// two edges, height of a vertex over the other plane; measures below 1e-13
// are exact degeneracies) is at least this is evaluated in double precision.
// On the 440-pair validation set the double-precision reduction is accurate
// to 3.6e-11 on every face pair it clears (26 % of face pairs are not
// cleared), while 1e-3 would have let 2.8e-9 through.
constexpr double kScreenBand = 1.0e-2;
constexpr double kScreenExact = 1.0e-13;

// Separated triangles (see kSeparatedVertex) whose normals differ by more
// than round-off but less than this are integrated by split Gauss quadrature.
constexpr double kParallelQuadrature = 1.0e-2;
// Vertices closer than this fraction of the larger edge are common.  It is
// loose on purpose: the FMM's canonical normalisation moves the shared
// vertices of touching tetrahedra apart by up to ~1e-9 of the root, and the
// Sauter-Schwab maps tolerate such a mismatch.
constexpr double kSharedVertex = 1.0e-6;
// Triangles count as separated, and may be integrated by split Gauss
// quadrature, only when every vertex pair is further apart than this
// fraction of the larger edge: a nearly touching pair needs more splitting
// than kMaxSplitDepth and stays with the closed form, whose near-parallel
// projection and height snapping exist for exactly that case.
constexpr double kSeparatedVertex = 1.0e-3;

// Three-vector of the reduction's working precision.
template <typename Real>
struct ReductionVec {
    Real x{0};
    Real y{0};
    Real z{0};
};

template <typename Real>
ReductionVec<Real> operator+(const ReductionVec<Real>& a, const ReductionVec<Real>& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

template <typename Real>
ReductionVec<Real> operator-(const ReductionVec<Real>& a, const ReductionVec<Real>& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

template <typename Real>
ReductionVec<Real> operator*(const Real factor, const ReductionVec<Real>& a)
{
    return {factor * a.x, factor * a.y, factor * a.z};
}

template <typename Real>
Real reduction_dot(const ReductionVec<Real>& a, const ReductionVec<Real>& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename Real>
ReductionVec<Real> reduction_cross(const ReductionVec<Real>& a, const ReductionVec<Real>& b)
{
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

template <typename Real>
Real reduction_norm(const ReductionVec<Real>& a)
{
    return reduction_sqrt(reduction_dot(a, a));
}

// Outcome flags of one reduction: `guard` marks an unreliable rank decision,
// `prohibited` a reduction that reached the reference's prohibited I0 case.
struct ReductionStatus {
    bool guard{false};
    bool prohibited{false};
};

// Decomposition of an offset vector into the span of up to four edge vectors
// (the coefficients) plus the perpendicular remainder (`residual`, the
// "height" the reduced integrals see).  The I3/I2/I1/I0 recursion reduces the
// dimension of the integration domain one edge at a time using these.
template <typename Real>
struct ExpansionProjection {
    std::array<Real, 4> coefficient{};
    ReductionVec<Real> projected{};
    Real residual{0};
};

// This is the small Gram--Schmidt projection used by the reference
// expan_GrammSchmidt.m routine.  Keeping the triangular back substitution
// explicit is important: I3/I2s/I2t use the coefficients of the original
// edge vectors, rather than coefficients of an orthogonal basis.  A direction
// whose independent part is below `ReductionTolerances<Real>::rank_squared` is dependent; one near that
// tolerance sets `status.guard`.
template <typename Real>
ExpansionProjection<Real> expand_gram_schmidt(
    const std::array<ReductionVec<Real>, 4>& vectors, const int count,
    const ReductionVec<Real>& value, ReductionStatus& status)
{
    std::array<ReductionVec<Real>, 4> orthogonal{};
    std::array<Real, 4> orthogonal_norm_squared{};
    Real vector_scale = reduction_norm(value);
    Real basis_scale_squared = 0;
    for (int i = 0; i < count; ++i) {
        const Real length = reduction_norm(
            vectors[static_cast<std::size_t>(i)]);
        vector_scale = std::max(vector_scale, length);
        basis_scale_squared = std::max(basis_scale_squared, length * length);
    }
    basis_scale_squared = std::max(
        basis_scale_squared,
        static_cast<Real>(std::numeric_limits<double>::min()));
    const Real rank_tolerance = ReductionTolerances<Real>::rank_squared * basis_scale_squared;

    Real coefficient_matrix[4][4]{};
    for (int column = 0; column < count; ++column) {
        orthogonal[static_cast<std::size_t>(column)] =
            vectors[static_cast<std::size_t>(column)];
        for (int row = 0; row < column; ++row) {
            coefficient_matrix[row][column] =
                reduction_dot(orthogonal[static_cast<std::size_t>(row)],
                              vectors[static_cast<std::size_t>(column)]) /
                orthogonal_norm_squared[static_cast<std::size_t>(row)];
            orthogonal[static_cast<std::size_t>(column)] =
                orthogonal[static_cast<std::size_t>(column)] -
                coefficient_matrix[row][column] *
                    orthogonal[static_cast<std::size_t>(row)];
        }
        const Real norm_squared = reduction_dot(
            orthogonal[static_cast<std::size_t>(column)],
            orthogonal[static_cast<std::size_t>(column)]);
        const Real ratio = norm_squared / basis_scale_squared;
        if (ratio >= ReductionTolerances<Real>::guard_low && ratio < ReductionTolerances<Real>::guard_high) {
            status.guard = true;
        }
        if (!(norm_squared > rank_tolerance)) {
            orthogonal[static_cast<std::size_t>(column)] = {};
            orthogonal_norm_squared[static_cast<std::size_t>(column)] = 1;
            coefficient_matrix[column][column] = 0;
        } else {
            orthogonal_norm_squared[static_cast<std::size_t>(column)] =
                norm_squared;
            coefficient_matrix[column][column] = reduction_dot(
                orthogonal[static_cast<std::size_t>(column)],
                vectors[static_cast<std::size_t>(column)]) /
                norm_squared;
        }
    }

    ExpansionProjection<Real> result;
    std::array<Real, 4> residual_projection{};
    for (int index = count - 1; index >= 0; --index) {
        const Real diagonal = coefficient_matrix[index][index];
        if (diagonal == 0) {
            continue;
        }
        const Real numerator = reduction_dot(
            value, orthogonal[static_cast<std::size_t>(index)]) -
            residual_projection[static_cast<std::size_t>(index)];
        result.coefficient[static_cast<std::size_t>(index)] = numerator /
            (diagonal * orthogonal_norm_squared[
                static_cast<std::size_t>(index)]);
        for (int row = 0; row < index; ++row) {
            residual_projection[static_cast<std::size_t>(row)] +=
                result.coefficient[static_cast<std::size_t>(index)] *
                coefficient_matrix[row][index] *
                orthogonal_norm_squared[static_cast<std::size_t>(row)];
        }
    }

    for (int index = 0; index < count; ++index) {
        result.projected = result.projected +
            result.coefficient[static_cast<std::size_t>(index)] *
                vectors[static_cast<std::size_t>(index)];
    }
    result.residual = reduction_norm(value - result.projected);
    const Real residual_tolerance = ReductionTolerances<Real>::relative_residual *
        std::max(static_cast<Real>(1), vector_scale);
    if (result.residual <= residual_tolerance) {
        result.residual = 0;
    }
    return result;
}

// I0: the fully reduced one-dimensional primitive of the recursion, a closed
// form in the edge length `p` and the four heights, with the reference's
// case analysis of which heights vanish.
template <typename Real>
Real triangle_i0_work(const Real p,
                      const std::array<Real, 4>& input_heights,
                      const Real zero_squared,
                      ReductionStatus& status)
{
    std::array<Real, 4> heights = input_heights;
    std::array<int, 4> order{{0, 1, 2, 3}};
    std::stable_sort(order.begin(), order.end(), [&](const int first,
                                                     const int second) {
        return heights[static_cast<std::size_t>(first)] <
            heights[static_cast<std::size_t>(second)];
    });
    heights[static_cast<std::size_t>(order[0])] = 0;
    heights[static_cast<std::size_t>(order[1])] = 0;
    const Real h1 = heights[0];
    const Real h2 = heights[1];
    const Real h3 = heights[2];
    const Real h4 = heights[3];
    const Real P = reduction_abs(p);
    const Real hh = h1 * h1 + h2 * h2 + h3 * h3 + h4 * h4;
    if (P == 0 && hh == 0) {
        // This primitive is only reached with a zero edge in a degenerate
        // triangle.  Valid triangle reductions skip that term before I0.
        return 0;
    }

    const Real radius = reduction_sqrt(P * P + hh);
    if (hh < zero_squared * P * P) {
        // This is the complete zero-height branch in the reference I0
        // reduction, not merely an alternative way to compute Phi1.
        return reduction_log(P) / (6 * P);
    }

    Real phi1;
    if (P == 0) {
        phi1 = 1 / reduction_sqrt(hh);
    } else {
        // asinh(P/sqrt(hh))/P is equivalent to the logarithm in I0 and is
        // well behaved at P=0.
        phi1 = reduction_asinh(P / reduction_sqrt(hh)) / P;
    }

    const Real hh1 = hh - h1 * h1;
    const Real tiny = zero_squared;
    const auto phi2_for = [&](const Real h) {
        if (h1 * P == 0) {
            return 1 / (hh + h * radius);
        }
        return reduction_atan((h1 * P) / (hh + h * radius)) / (h1 * P);
    };
    Real answer;
    if (h1 * h1 + h2 * h2 + h4 * h4 < tiny * hh) {
        // Rank-deficient reductions can leave h3 as the only non-zero
        // height.  The reference case-4 expression is 0/0 at h1=0; this is
        // its analytical h1 -> 0 limit.
        const Real q = reduction_hypot(P, h3);
        const Real logarithm = reduction_asinh(P / h3);
        answer = (logarithm * (2 * q * q - 3 * h3 * h3) / (2 * P * P * P) +
                  (4 * h3 - 3 * q) / (2 * P * P)) /
            6;
    } else if (h1 * h1 + h2 * h2 + h3 * h3 < tiny * hh) {
        // Likewise, the reference case-6 expression needs its h1 -> 0
        // limit when h4 is the only non-zero height.
        const Real q = reduction_hypot(P, h4);
        const Real logarithm = reduction_asinh(P / h4);
        answer = (logarithm * (2 * q * q - 5 * h4 * h4) / (2 * P * P * P) +
                  3 * (2 * h4 - q) / (2 * P * P) -
                  (q + 2 * h4) / (3 * (q + h4) * (q + h4))) /
            6;
    } else if (hh1 < tiny * hh) {
        answer = phi1 / 6;
    } else if (h3 * h3 + h4 * h4 < tiny * hh) {
        answer = (phi1 - h2 * phi2_for(h2)) / 6;
    } else {
        const Real radius1 = reduction_sqrt(P * P + h1 * h1);
        if (h2 * h2 + h4 * h4 < tiny * hh) {
            const Real phi2 = phi2_for(h3);
            const Real phi3 = hh1 / (2 * radius1) *
                reduction_log((radius1 + radius) * (radius1 + radius) / hh1);
            answer = ((h1 * h1 - h3 * h3) * phi1 -
                      2 * h1 * h1 * h3 * phi2 + phi3) /
                (6 * h1 * h1);
        } else if (h2 * h2 + h3 * h3 < tiny * hh) {
            const Real phi2 = phi2_for(h4);
            const Real phi3 = 1 / (2 * radius1) *
                reduction_log((radius1 + radius) * (radius1 + radius) / hh1);
            answer = ((h1 * h1 - 3 * h4 * h4) * phi1 -
                      h4 * (3 * h1 * h1 - h4 * h4) * phi2 +
                      3 * h4 * h4 * phi3 -
                      h4 * h4 / (radius + h4)) /
                (6 * h1 * h1);
        } else if (h1 * h1 + h4 * h4 < tiny * hh) {
            const Real h = reduction_sqrt(hh);
            const Real radius2 = reduction_sqrt(P * P + h2 * h2);
            const Real phi4 = h3 * h3 / (h2 * P * P) *
                ((radius2 / h2) * reduction_log((radius2 + radius) / h3) -
                 reduction_log((h2 + h) / h3));
            answer = (hh / (h2 * h2) * phi1 - 1 / (radius + h) - phi4) / 6;
        } else if (h1 * h1 + h3 * h3 < tiny * hh) {
            const Real h = reduction_sqrt(hh);
            const Real radius2 = reduction_sqrt(P * P + h2 * h2);
            const Real phi2 = reduction_atan(h2 * P /
                (hh + h4 * radius)) / P;
            const Real phi4 = h4 * h4 / (h2 * P * P) *
                ((radius2 / h2) * reduction_log((radius2 + radius) / h4) -
                 reduction_log((h2 + h) / h4));
            answer = ((1 + 3 * h4 * h4 / (h2 * h2)) * phi1 -
                      2 * (h4 / h2) * (h4 / h2) * (h4 / h2) * phi2 -
                      3 * phi4 +
                      (2 * h4 * h4 - h2 * h2) /
                      (h2 * h2 * (radius + h))) / 6;
        } else {
            // This is the prohibited reference-I0 case.  Reaching it means
            // that the preceding dimensional reduction has lost its expected
            // rank; the caller reports it unless the face pair is guarded.
            status.prohibited = true;
            return 0;
        }
    }
    return answer;
}

// The working precision of I0: long double for the double-precision
// reduction, the reduction's own precision otherwise.
template <typename Real>
struct I0Precision {
    using type = Real;
};

#if defined(__FLT128_MANT_DIG__)
template <>
struct I0Precision<double> {
    using type = long double;
};
#endif

template <typename Real>
Real triangle_i0(const Real p, const std::array<Real, 4>& heights,
                 ReductionStatus& status)
{
    using Work = typename I0Precision<Real>::type;
    return static_cast<Real>(triangle_i0_work<Work>(
        static_cast<Work>(p),
        {{static_cast<Work>(heights[0]), static_cast<Work>(heights[1]),
          static_cast<Work>(heights[2]), static_cast<Work>(heights[3])}},
        static_cast<Work>(ReductionTolerances<Real>::i0_zero_squared), status));
}

// I1, I2s, I2t, I3: the reduction steps.  Each expands the offset in the
// remaining edge vectors, drops one edge and calls the next lower primitive
// at the two endpoints of that edge; a coefficient within `ReductionTolerances<Real>::zero`
// of zero contributes nothing and is skipped, exactly as in the reference.
template <typename Real>
Real triangle_i1(const ReductionVec<Real>& vector, const ReductionVec<Real>& offset,
                          const Real h2, const Real h3,
                          const Real h4, ReductionStatus& status)
{
    const Real length = reduction_norm(vector);
    if (!(length > 0)) {
        return 0;
    }
    std::array<ReductionVec<Real>, 4> basis{{vector, {}, {}, {}}};
    const ExpansionProjection<Real> expansion = expand_gram_schmidt(
        basis, 1, offset, status);
    Real result = 0;
    const Real coefficient = expansion.coefficient[0];
    if (reduction_abs(1 + coefficient) > ReductionTolerances<Real>::zero) {
        result += (1 + coefficient) * triangle_i0(
            reduction_abs(1 + coefficient) * length,
            {{expansion.residual, h2, h3, h4}}, status);
    }
    if (reduction_abs(coefficient) > ReductionTolerances<Real>::zero) {
        result -= coefficient * triangle_i0(
            reduction_abs(coefficient) * length,
            {{expansion.residual, h2, h3, h4}}, status);
    }
    return result;
}

template <typename Real>
Real triangle_i2s(const ReductionVec<Real>& first, const ReductionVec<Real>& second,
                           const ReductionVec<Real>& offset, const Real h3,
                           const Real h4, ReductionStatus& status)
{
    std::array<ReductionVec<Real>, 4> basis{{first, second, {}, {}}};
    const ExpansionProjection<Real> expansion = expand_gram_schmidt(
        basis, 2, offset, status);
    Real result = 0;
    const Real s0 = expansion.coefficient[0];
    const Real s1 = expansion.coefficient[1];
    if (reduction_abs(1 + s0) > ReductionTolerances<Real>::zero) {
        result += (1 + s0) * triangle_i1(
            second, expansion.projected + first, expansion.residual, h3, h4,
            status);
    }
    if (reduction_abs(s0) > ReductionTolerances<Real>::zero) {
        result -= s0 * triangle_i1(
            second, expansion.projected, expansion.residual, h3, h4, status);
    }
    if (reduction_abs(1 + s1) > ReductionTolerances<Real>::zero) {
        result += (1 + s1) * triangle_i1(
            first, expansion.projected + second, expansion.residual, h3, h4,
            status);
    }
    if (reduction_abs(s1) > ReductionTolerances<Real>::zero) {
        result -= s1 * triangle_i1(
            first, expansion.projected, expansion.residual, h3, h4, status);
    }
    return result;
}

template <typename Real>
Real triangle_i2t(const ReductionVec<Real>& first, const ReductionVec<Real>& second,
                           const ReductionVec<Real>& offset, const Real h3,
                           const Real h4, ReductionStatus& status)
{
    std::array<ReductionVec<Real>, 4> basis{{first, second, {}, {}}};
    const ExpansionProjection<Real> expansion = expand_gram_schmidt(
        basis, 2, offset, status);
    Real result = 0;
    const Real s0 = expansion.coefficient[0];
    const Real s1 = expansion.coefficient[1];
    if (reduction_abs(s0) > ReductionTolerances<Real>::zero) {
        result -= s0 * triangle_i1(
            second, expansion.projected, expansion.residual, h3, h4, status);
    }
    if (reduction_abs(s1) > ReductionTolerances<Real>::zero) {
        result -= s1 * triangle_i1(
            first, expansion.projected, expansion.residual, h3, h4, status);
    }
    if (reduction_abs(1 + s0 + s1) > ReductionTolerances<Real>::zero) {
        result += (1 + s0 + s1) * triangle_i1(
            first - second, expansion.projected + second,
            expansion.residual, h3, h4, status);
    }
    return result;
}

template <typename Real>
Real triangle_i3(const ReductionVec<Real>& first, const ReductionVec<Real>& second,
                          const ReductionVec<Real>& third, const ReductionVec<Real>& offset,
                          const Real h4, ReductionStatus& status)
{
    std::array<ReductionVec<Real>, 4> basis{{first, second, third, {}}};
    const ExpansionProjection<Real> expansion = expand_gram_schmidt(
        basis, 3, offset, status);
    Real result = 0;
    const Real s0 = expansion.coefficient[0];
    const Real s1 = expansion.coefficient[1];
    const Real s2 = expansion.coefficient[2];
    if (reduction_abs(1 + s0 + s1) > ReductionTolerances<Real>::zero) {
        result += (1 + s0 + s1) * triangle_i2s(
            first - second, third, expansion.projected + second,
            expansion.residual, h4, status);
    }
    if (reduction_abs(s0) > ReductionTolerances<Real>::zero) {
        result -= s0 * triangle_i2s(
            second, third, expansion.projected, expansion.residual, h4, status);
    }
    if (reduction_abs(s1) > ReductionTolerances<Real>::zero) {
        result -= s1 * triangle_i2s(
            first, third, expansion.projected, expansion.residual, h4, status);
    }
    if (reduction_abs(1 + s2) > ReductionTolerances<Real>::zero) {
        result += (1 + s2) * triangle_i2t(
            first, second, expansion.projected + third,
            expansion.residual, h4, status);
    }
    if (reduction_abs(s2) > ReductionTolerances<Real>::zero) {
        result -= s2 * triangle_i2t(
            first, second, expansion.projected, expansion.residual, h4, status);
    }
    return result;
}

// The Galerkin integral of 1/|x - y| over two triangles in normalised
// coordinates (largest triangle first, common origin, unit scale).  The two
// triangles' edge vectors a1..a4 and the vertex offset e4 are expanded
// together; coplanar (parallel) pairs take the reduced branch in which only
// the in-plane offset and the separation height enter.
template <typename Real>
Real triangle_triangle_integral_core(
    const std::array<ReductionVec<Real>, 3>& first,
    const std::array<ReductionVec<Real>, 3>& second,
    ReductionStatus& status)
{
    // Only the two edges meeting at each triangle's first vertex enter the
    // expansion basis and the area; the third edge of each triangle is
    // implied by them and is not formed.
    const ReductionVec<Real> a1 = first[1] - first[0];
    const ReductionVec<Real> a2 = first[2] - first[0];
    const ReductionVec<Real> a3 = second[0] - second[1];
    const ReductionVec<Real> a4 = second[0] - second[2];
    const ReductionVec<Real> e4 = first[0] - second[0];
    const ReductionVec<Real> first_cross = reduction_cross(a1, a2);
    const ReductionVec<Real> second_cross = reduction_cross(a3, a4);
    const Real first_area_twice = reduction_norm(first_cross);
    const Real second_area_twice = reduction_norm(second_cross);
    if (!(first_area_twice > 0 && second_area_twice > 0)) {
        throw std::invalid_argument("triangle pair contains a degenerate triangle");
    }

    std::array<ReductionVec<Real>, 4> all_vectors{{a1, a2, a3, a4}};
    const ExpansionProjection<Real> complete_expansion = expand_gram_schmidt(
        all_vectors, 4, e4, status);
    const Real s0 = complete_expansion.coefficient[0];
    const Real s1 = complete_expansion.coefficient[1];
    const Real s2 = complete_expansion.coefficient[2];
    const Real s3 = complete_expansion.coefficient[3];
    const ReductionVec<Real> e4x = complete_expansion.projected;
    const ReductionVec<Real> first_normal = (1 / first_area_twice) * first_cross;
    const ReductionVec<Real> second_normal = (1 / second_area_twice) * second_cross;
    const Real parallel_measure = std::min(
        reduction_norm(first_normal + second_normal),
        reduction_norm(first_normal - second_normal));
    // Nearly parallel faces (two coplanar faces of a conforming mesh after the
    // FMM rounds their coordinates, or genuinely almost parallel ones) are
    // projected onto planes normal to their mean normal, each through its own
    // centroid, and the parallel branch evaluates them (see ReductionTolerances<Real>::near_parallel).
    // The projection is symmetric in the two triangles.
    if (parallel_measure > ReductionTolerances<Real>::zero &&
        parallel_measure <= ReductionTolerances<Real>::near_parallel) {
        const Real sign =
            reduction_dot(first_normal, second_normal) >= 0 ? 1 : -1;
        const ReductionVec<Real> mean_normal = first_normal + sign * second_normal;
        const ReductionVec<Real> normal =
            (1 / reduction_norm(mean_normal)) * mean_normal;
        const auto project = [&normal](const std::array<ReductionVec<Real>, 3>& triangle) {
            const ReductionVec<Real> centroid = static_cast<Real>(1) /
                static_cast<Real>(3) *
                (triangle[0] + triangle[1] + triangle[2]);
            std::array<ReductionVec<Real>, 3> projected{};
            for (std::size_t vertex = 0; vertex < 3; ++vertex) {
                projected[vertex] = triangle[vertex] -
                    reduction_dot(triangle[vertex] - centroid, normal) * normal;
            }
            return projected;
        };
        return triangle_triangle_integral_core(project(first), project(second),
                                               status);
    }
    Real reduced = 0;
    if (parallel_measure > ReductionTolerances<Real>::zero) {
        const Real h4 = 0;
        const Real i31 = triangle_i3(a1, a2, a3, e4, h4, status);
        const Real i32 = triangle_i3(a1, a2, a4, e4, h4, status);
        const Real i33 = triangle_i3(a1, a2, a4 - a3, e4 + a3, h4,
                                              status);
        const Real i34 = triangle_i3(a3, a4, a1, e4, h4, status);
        const Real i35 = triangle_i3(a3, a4, a2, e4, h4, status);
        const Real i36 = triangle_i3(a3, a4, a2 - a1, e4 + a1, h4,
                                              status);
        reduced = (1 + s2 + s3) * i33 - s3 * i31 - s2 * i32 +
            (1 + s0 + s1) * i36 - s1 * i34 - s0 * i35;
    } else {
        // In the parallel case the projection of e4 onto the common plane
        // is the only offset required by the dimensional reduction.
        const Real h4 = complete_expansion.residual;
        const Real i33 = triangle_i3(a1, a2, a4 - a3, e4x + a3, h4,
                                              status);
        const Real i34 = triangle_i3(a3, a4, a1, e4x, h4, status);
        const Real i35 = triangle_i3(a3, a4, a2, e4x, h4, status);
        const Real i36 = triangle_i3(a3, a4, a2 - a1, e4x + a1, h4,
                                              status);
        reduced = i33 + (1 + s0 + s1) * i36 - s1 * i34 - s0 * i35;
    }
    return first_area_twice * second_area_twice * reduced;
}

//------------------------------------------------------------------------------
// Numerical triangle-pair quadratures (the fallbacks of the reduction)
//------------------------------------------------------------------------------

// Gauss-Legendre nodes and weights on [0, 1] (Newton iteration on the
// Legendre recurrence; the weights sum to one).
struct GaussRule {
    std::vector<double> node;
    std::vector<double> weight;
};

GaussRule make_gauss_legendre_rule(const int points)
{
    GaussRule rule;
    rule.node.resize(static_cast<std::size_t>(points));
    rule.weight.resize(static_cast<std::size_t>(points));
    for (int i = 0; i < points; ++i) {
        double z = std::cos(std::numbers::pi * (i + 0.75) / (points + 0.5));
        double derivative = 1.0;
        for (int iteration = 0; iteration < 100; ++iteration) {
            double p1 = 1.0;
            double p2 = 0.0;
            for (int j = 1; j <= points; ++j) {
                const double p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
            }
            derivative = points * (z * p1 - p2) / (z * z - 1.0);
            const double previous = z;
            z = previous - p1 / derivative;
            if (std::abs(z - previous) <= 4.0 * std::numeric_limits<double>::epsilon()) {
                break;
            }
        }
        rule.node[static_cast<std::size_t>(i)] = 0.5 * (1.0 - z);
        rule.weight[static_cast<std::size_t>(i)] =
            1.0 / ((1.0 - z * z) * derivative * derivative);
    }
    return rule;
}

// Points of the reduced regular quadrature (per direction, per triangle) and
// the Sauter-Schwab rule (per direction of the four-dimensional domain).
constexpr int kRegularGaussPoints = 10;
constexpr int kSauterSchwabGaussPoints = 24;
// The larger triangle of a separated pair is split into four until the gap
// between the two bounding spheres is at least this times the larger radius.
constexpr double kSeparationRatio = 1.0;
constexpr int kMaxSplitDepth = 12;

const GaussRule& regular_gauss_rule()
{
    static const GaussRule rule = make_gauss_legendre_rule(kRegularGaussPoints);
    return rule;
}

const GaussRule& sauter_schwab_gauss_rule()
{
    static const GaussRule rule = make_gauss_legendre_rule(kSauterSchwabGaussPoints);
    return rule;
}

double twice_triangle_area(const std::array<Vec3, 3>& triangle)
{
    return norm(cross(triangle[1] - triangle[0], triangle[2] - triangle[1]));
}

// chi(x1, x2) = P0 + x1 (P1 - P0) + x2 (P2 - P1) on the reference triangle
// {0 <= x2 <= x1 <= 1}, whose Jacobian is twice the triangle's area.
Vec3 reference_triangle_point(const std::array<Vec3, 3>& triangle,
                              const double x1, const double x2)
{
    return triangle[0] + (triangle[1] - triangle[0]) * x1 +
        (triangle[2] - triangle[1]) * x2;
}

// The four midpoint children of a triangle.
std::array<std::array<Vec3, 3>, 4> split_triangle(const std::array<Vec3, 3>& triangle)
{
    const Vec3 m01 = (triangle[0] + triangle[1]) * 0.5;
    const Vec3 m12 = (triangle[1] + triangle[2]) * 0.5;
    const Vec3 m20 = (triangle[2] + triangle[0]) * 0.5;
    return {{{{triangle[0], m01, m20}},
             {{m01, triangle[1], m12}},
             {{m20, m12, triangle[2]}},
             {{m01, m12, m20}}}};
}

// Gauss quadrature of the pair integral for triangles without a common
// point: collapsed Gauss rules on both triangles, the larger one split
// recursively until the two are well separated.
double triangle_pair_gauss(const std::array<Vec3, 3>& first,
                           const std::array<Vec3, 3>& second,
                           const int depth)
{
    const Vec3 first_centre = (first[0] + first[1] + first[2]) * (1.0 / 3.0);
    const Vec3 second_centre = (second[0] + second[1] + second[2]) * (1.0 / 3.0);
    double first_radius = 0.0;
    double second_radius = 0.0;
    for (std::size_t vertex = 0; vertex < 3; ++vertex) {
        first_radius = std::max(first_radius, norm(first[vertex] - first_centre));
        second_radius = std::max(second_radius, norm(second[vertex] - second_centre));
    }
    const double gap = norm(first_centre - second_centre) - first_radius - second_radius;
    if (gap < kSeparationRatio * std::max(first_radius, second_radius) &&
        depth < kMaxSplitDepth) {
        double sum = 0.0;
        if (first_radius >= second_radius) {
            for (const auto& child : split_triangle(first)) {
                sum += triangle_pair_gauss(child, second, depth + 1);
            }
        } else {
            for (const auto& child : split_triangle(second)) {
                sum += triangle_pair_gauss(first, child, depth + 1);
            }
        }
        return sum;
    }

    const GaussRule& rule = regular_gauss_rule();
    constexpr std::size_t count =
        static_cast<std::size_t>(kRegularGaussPoints * kRegularGaussPoints);
    std::array<Vec3, count> first_point{};
    std::array<Vec3, count> second_point{};
    std::array<double, count> weight{};
    std::size_t index = 0;
    for (std::size_t i = 0; i < rule.node.size(); ++i) {
        for (std::size_t j = 0; j < rule.node.size(); ++j) {
            // Collapsed map (s, s t) of the unit square onto the reference
            // triangle, with Jacobian s.
            const double s = rule.node[i];
            const double t = rule.node[j];
            first_point[index] = reference_triangle_point(first, s, s * t);
            second_point[index] = reference_triangle_point(second, s, s * t);
            weight[index] = rule.weight[i] * rule.weight[j] * s;
            ++index;
        }
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t j = 0; j < count; ++j) {
            sum += weight[i] * weight[j] / norm(first_point[i] - second_point[j]);
        }
    }
    return twice_triangle_area(first) * twice_triangle_area(second) * sum;
}

// Sauter-Schwab quadrature of the pair integral for triangles with a common
// vertex, edge or face (S. Sauter and C. Schwab, Boundary Element Methods,
// Springer 2011, Sec. 5.2).  The singularity is removed by transformations
// of the four-dimensional reference domain, after which the tensor Gauss rule
// converges exponentially (relative error 1e-9 to 1e-12 on mesh face pairs).
// `first` and `second` must list their common vertices first, in matching
// order; `shared` is their number.
double triangle_pair_sauter_schwab(const std::array<Vec3, 3>& first,
                                   const std::array<Vec3, 3>& second,
                                   const int shared)
{
    const GaussRule& rule = sauter_schwab_gauss_rule();
    struct ReferencePair {
        double a1, a2, b1, b2, jacobian;
    };
    const auto kernel = [&](const ReferencePair& pair) {
        return pair.jacobian / norm(
            reference_triangle_point(first, pair.a1, pair.a2) -
            reference_triangle_point(second, pair.b1, pair.b2));
    };
    double sum = 0.0;
    const std::size_t points = rule.node.size();
    for (std::size_t i1 = 0; i1 < points; ++i1) {
        const double xi = rule.node[i1];
        for (std::size_t i2 = 0; i2 < points; ++i2) {
            const double e1 = rule.node[i2];
            for (std::size_t i3 = 0; i3 < points; ++i3) {
                const double e2 = rule.node[i3];
                for (std::size_t i4 = 0; i4 < points; ++i4) {
                    const double e3 = rule.node[i4];
                    const double w = rule.weight[i1] * rule.weight[i2] *
                        rule.weight[i3] * rule.weight[i4];
                    double value = 0.0;
                    if (shared == 3) {
                        const double jacobian = xi * xi * xi * e1 * e1 * e2;
                        const double u1 = xi * (1.0 - e1 + e1 * e2);
                        const double v1 = xi * (1.0 - e1 * e2 * e3);
                        const double v2 = xi * (1.0 - e1);
                        const double u3 = xi * e1 * (1.0 - e2 + e2 * e3);
                        const double v3 = xi * (1.0 - e1 * e2);
                        const double w3 = xi * e1 * (1.0 - e2);
                        const double u5 = xi * (1.0 - e1 * e2 * e3);
                        const double v5 = xi * e1 * (1.0 - e2 * e3);
                        const double w5 = xi * e1 * (1.0 - e2);
                        value = kernel({xi, u1, v1, v2, jacobian}) +
                            kernel({v1, v2, xi, u1, jacobian}) +
                            kernel({xi, u3, v3, w3, jacobian}) +
                            kernel({v3, w3, xi, u3, jacobian}) +
                            kernel({u5, v5, xi, w5, jacobian}) +
                            kernel({xi, w5, u5, v5, jacobian});
                    } else if (shared == 2) {
                        const double j1 = xi * xi * xi * e1 * e1;
                        const double j2 = j1 * e2;
                        value = kernel({xi, xi * e1 * e3,
                                        xi * (1.0 - e1 * e2), xi * e1 * (1.0 - e2), j1}) +
                            kernel({xi, xi * e1,
                                    xi * (1.0 - e1 * e2 * e3), xi * e1 * e2 * (1.0 - e3), j2}) +
                            kernel({xi * (1.0 - e1 * e2), xi * e1 * (1.0 - e2),
                                    xi, xi * e1 * e2 * e3, j2}) +
                            kernel({xi * (1.0 - e1 * e2 * e3), xi * e1 * e2 * (1.0 - e3),
                                    xi, xi * e1, j2}) +
                            kernel({xi * (1.0 - e1 * e2 * e3), xi * e1 * (1.0 - e2 * e3),
                                    xi, xi * e1 * e2, j2});
                    } else {
                        const double jacobian = xi * xi * xi * e2;
                        value = kernel({xi, xi * e1, xi * e2, xi * e2 * e3, jacobian}) +
                            kernel({xi * e2, xi * e2 * e3, xi, xi * e1, jacobian});
                    }
                    sum += w * value;
                }
            }
        }
    }
    return twice_triangle_area(first) * twice_triangle_area(second) * sum;
}

// The common vertices of two triangles (to kSharedVertex of the larger
// edge) as (first, second) index pairs, in the order of the first triangle.
std::vector<std::pair<int, int>> common_vertices(const std::array<Vec3, 3>& first,
                                                 const std::array<Vec3, 3>& second,
                                                 const double edge)
{
    std::vector<std::pair<int, int>> common;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (norm(first[static_cast<std::size_t>(i)] -
                     second[static_cast<std::size_t>(j)]) <= kSharedVertex * edge &&
                common.size() < 3) {
                common.emplace_back(i, j);
            }
        }
    }
    return common;
}

// Sauter-Schwab quadrature of touching triangles, after reordering both so
// that their common vertices come first in matching order.
double touching_triangle_pair_quadrature(const std::array<Vec3, 3>& first,
                                         const std::array<Vec3, 3>& second,
                                         const std::vector<std::pair<int, int>>& common)
{
    std::array<int, 3> first_order{};
    std::array<int, 3> second_order{};
    if (common.size() == 3) {
        first_order = {{0, 1, 2}};
        second_order = {{common[0].second, common[1].second, common[2].second}};
    } else if (common.size() == 2) {
        first_order = {{common[0].first, common[1].first,
                        3 - common[0].first - common[1].first}};
        second_order = {{common[0].second, common[1].second,
                         3 - common[0].second - common[1].second}};
    } else {
        first_order = {{common[0].first, (common[0].first + 1) % 3,
                        (common[0].first + 2) % 3}};
        second_order = {{common[0].second, (common[0].second + 1) % 3,
                         (common[0].second + 2) % 3}};
    }
    std::array<Vec3, 3> reordered_first{};
    std::array<Vec3, 3> reordered_second{};
    for (std::size_t vertex = 0; vertex < 3; ++vertex) {
        reordered_first[vertex] = first[static_cast<std::size_t>(first_order[vertex])];
        reordered_second[vertex] = second[static_cast<std::size_t>(second_order[vertex])];
    }
    return triangle_pair_sauter_schwab(reordered_first, reordered_second,
                                       static_cast<int>(common.size()));
}

// The reduced integral of a pair already ordered larger-triangle-first, in
// the working precision Real: normalised to unit scale about the first vertex
// of the larger triangle (the integral is homogeneous of degree 3 in length,
// so the caller rescales by scale^3).
template <typename Real>
Real reduced_triangle_pair(const std::array<Vec3, 3>& ordered_first,
                           const std::array<Vec3, 3>& ordered_second,
                           const double scale_value,
                           ReductionStatus& status)
{
    const Real inverse_scale = 1 / static_cast<Real>(scale_value);
    const Vec3& origin = ordered_first[0];
    const auto to_reduction = [&](const Vec3& point) {
        return ReductionVec<Real>{
            (static_cast<Real>(point.x) - static_cast<Real>(origin.x)) * inverse_scale,
            (static_cast<Real>(point.y) - static_cast<Real>(origin.y)) * inverse_scale,
            (static_cast<Real>(point.z) - static_cast<Real>(origin.z)) * inverse_scale};
    };
    std::array<ReductionVec<Real>, 3> normalised_first{};
    std::array<ReductionVec<Real>, 3> normalised_second{};
    for (std::size_t index = 0; index < 3; ++index) {
        normalised_first[index] = to_reduction(ordered_first[index]);
        normalised_second[index] = to_reduction(ordered_second[index]);
    }
    return triangle_triangle_integral_core(normalised_first, normalised_second, status);
}

// The smallest measure of near-degeneracy of a triangle pair that is not an
// exact degeneracy (see kScreenBand); infinity if all are exact.
double smallest_degeneracy(const std::array<Vec3, 3>& first,
                           const std::array<Vec3, 3>& second,
                           const Vec3& first_normal,
                           const Vec3& second_normal,
                           const double parallel_measure,
                           const double edge)
{
    double smallest = std::numeric_limits<double>::infinity();
    const auto consider = [&smallest](const double measure) {
        if (measure >= kScreenExact) {
            smallest = std::min(smallest, measure);
        }
    };
    consider(parallel_measure);
    std::array<Vec3, 3> first_edges{};
    std::array<Vec3, 3> second_edges{};
    for (std::size_t i = 0; i < 3; ++i) {
        const Vec3 a = first[(i + 1) % 3] - first[i];
        const Vec3 b = second[(i + 1) % 3] - second[i];
        first_edges[i] = scale(a, 1.0 / norm(a));
        second_edges[i] = scale(b, 1.0 / norm(b));
    }
    for (std::size_t i = 0; i < 3; ++i) {
        // An edge nearly parallel to the other triangle's plane.
        consider(std::abs(dot(first_normal, second_edges[i])));
        consider(std::abs(dot(second_normal, first_edges[i])));
        // A vertex nearly in the other triangle's plane.
        consider(std::abs(dot(first_normal, second[i] - first[0])) / edge);
        consider(std::abs(dot(second_normal, first[i] - second[0])) / edge);
        for (std::size_t j = 0; j < 3; ++j) {
            // Two nearly parallel edges.
            consider(norm(cross(first_edges[i], second_edges[j])));
        }
    }
    return smallest;
}

// The pair integral: the closed form, with the numerical fallbacks described
// at the top of this section.  Face pairs the screen clears are reduced in
// double precision, all others in quadruple precision with the guard.
double triangle_triangle_integral_impl(
    std::array<Vec3, 3> first,
    std::array<Vec3, 3> second)
{
    const auto triangle_scale = [](const std::array<Vec3, 3>& triangle) {
        return std::max({
            norm(triangle[1] - triangle[0]),
            norm(triangle[2] - triangle[1]),
            norm(triangle[0] - triangle[2])});
    };
    const double first_scale = triangle_scale(first);
    const double second_scale = triangle_scale(second);
    if (!(first_scale > 0.0) || !(second_scale > 0.0) ||
        !std::isfinite(first_scale) || !std::isfinite(second_scale)) {
        throw std::invalid_argument("triangle pair has invalid scale");
    }
    const Vec3 first_cross = cross(first[1] - first[0], first[2] - first[0]);
    const Vec3 second_cross = cross(second[1] - second[0], second[2] - second[0]);
    if (!(norm(first_cross) > 0.0) || !(norm(second_cross) > 0.0)) {
        throw std::invalid_argument("triangle pair contains a degenerate triangle");
    }
    const Vec3 first_normal = scale(first_cross, 1.0 / norm(first_cross));
    const Vec3 second_normal = scale(second_cross, 1.0 / norm(second_cross));
    const double parallel_measure = std::min(norm(first_normal + second_normal),
                                             norm(first_normal - second_normal));
    const double edge = std::max(first_scale, second_scale);
    const std::vector<std::pair<int, int>> common = common_vertices(
        first, second, edge);
    double closest_vertices = std::numeric_limits<double>::infinity();
    for (const Vec3& a : first) {
        for (const Vec3& b : second) {
            closest_vertices = std::min(closest_vertices, norm(a - b));
        }
    }
    const bool separated = closest_vertices > kSeparatedVertex * edge;

    // Separated triangles in nearly parallel planes: quadrature directly.
    if (separated && parallel_measure > 1.0e-14 &&
        parallel_measure < kParallelQuadrature) {
        return triangle_pair_gauss(first, second, 0);
    }

    // The core is symmetric in its two triangles.  Keep the larger triangle
    // first for the same ordering used by the reference reduction, but do so
    // before choosing the common origin so normalization remains one-pass.
    std::array<Vec3, 3> ordered_first = first;
    std::array<Vec3, 3> ordered_second = second;
    if (first_scale < second_scale) {
        std::swap(ordered_first, ordered_second);
    }
    const double cube = std::pow(edge, 3.0);
    ReductionStatus status;
    if (smallest_degeneracy(first, second, first_normal, second_normal,
                            parallel_measure, edge) >= kScreenBand) {
        const double reduced = reduced_triangle_pair<double>(
            ordered_first, ordered_second, edge, status);
        if (status.prohibited) {
            throw std::domain_error("triangle I0 reached prohibited branch");
        }
        return cube * reduced;
    }
    const ReductionReal reduced = reduced_triangle_pair<ReductionReal>(
        ordered_first, ordered_second, edge, status);
    if (status.guard && !common.empty()) {
        // An unreliable rank decision: integrate numerically instead.
        return touching_triangle_pair_quadrature(first, second, common);
    }
    if (status.guard && separated) {
        return triangle_pair_gauss(first, second, 0);
    }
    // NOTE(cdfmm): a guarded pair that is neither touching nor separated
    // (nearly touching without a common vertex) keeps the closed form: no
    // fallback here resolves its gap.
    if (status.prohibited) {
        // Reaching the prohibited reference-I0 case means that the
        // dimensional reduction has lost its expected rank; assigning an
        // ad-hoc limiting value would hide that error.
        throw std::domain_error("triangle I0 reached prohibited branch");
    }
    return cube * static_cast<double>(reduced);
}

double triangle_triangle_integral(const std::array<Vec3, 3>& first,
                                  const std::array<Vec3, 3>& second)
{
    return triangle_triangle_integral_impl(first, second);
}

// Support for the tetrahedron-averaged monomials of the finite P2M/L2P
// operators: a monomial in one Cartesian coordinate of a point inside the
// tetrahedron, written as a polynomial in the four barycentric coordinates.
struct BarycentricPolynomial {
    std::array<int, 4> power{{0, 0, 0, 0}};
    double coefficient{0.0};
};

// Expand (offset + sum_v coordinate_v * lambda_v)^power / power! into
// barycentric monomials lambda^n (each with its own 1/n! factors), which the
// caller integrates exactly with the Dirichlet formula.
void expand_axis(const int power, const double offset,
                 const std::array<double, 4>& coordinates,
                 std::vector<BarycentricPolynomial>& terms)
{
    if (power < 0) {
        throw std::invalid_argument("monomial exponents must be non-negative");
    }
    // The loops below enumerate every 4-tuple of non-negative integers summing
    // to at most `power`, so the term count is exactly C(power + 4, 4).
    // Reserving it removes the repeated reallocation of a buffer whose size is
    // known before the first term is written.
    const std::size_t term_count =
        static_cast<std::size_t>(power + 1) *
        static_cast<std::size_t>(power + 2) *
        static_cast<std::size_t>(power + 3) *
        static_cast<std::size_t>(power + 4) / 24;
    terms.reserve(terms.size() + term_count);
    for (int n0 = 0; n0 <= power; ++n0) {
        for (int n1 = 0; n1 <= power - n0; ++n1) {
            for (int n2 = 0; n2 <= power - n0 - n1; ++n2) {
                for (int n3 = 0; n3 <= power - n0 - n1 - n2; ++n3) {
                    const int barycentric_power = n0 + n1 + n2 + n3;
                    const int offset_power = power - barycentric_power;
                    double coefficient = std::pow(offset, offset_power) /
                        MultiIndexSet::factorial(offset_power);
                    const int powers[4]{n0, n1, n2, n3};
                    for (int vertex = 0; vertex < 4; ++vertex) {
                        coefficient *= std::pow(coordinates[
                            static_cast<std::size_t>(vertex)],
                            powers[vertex]) /
                            MultiIndexSet::factorial(powers[vertex]);
                    }
                    terms.push_back({{n0, n1, n2, n3}, coefficient});
                }
            }
        }
    }
}

} // namespace

namespace detail {

double triangle_triangle_laplace_integral(
    const std::array<Vec3, 3>& first,
    const std::array<Vec3, 3>& second)
{
    return triangle_triangle_integral(first, second);
}

PreparedTetrahedronPointField prepare_tetrahedron_point_field(
    const Tetrahedron& tetrahedron)
{
    PreparedTetrahedronPointField prepared;
    prepared.volume = tetrahedron_volume(tetrahedron);
    prepared.vertices = tetrahedron.vertices;
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            prepared.geometry_scale = std::max(
                prepared.geometry_scale,
                norm(prepared.vertices[static_cast<std::size_t>(i)] -
                     prepared.vertices[static_cast<std::size_t>(j)]));
        }
    }
    // One frame per omitted vertex, in the order the surface sum visits the
    // faces, with the omitted vertex kept last for the orientation test.
    for (int missing_vertex = 0; missing_vertex < 4; ++missing_vertex) {
        std::array<Vec3, 4> face{};
        int output = 0;
        for (int vertex = 0; vertex < 4; ++vertex) {
            if (vertex != missing_vertex) {
                face[static_cast<std::size_t>(output++)] =
                    prepared.vertices[static_cast<std::size_t>(vertex)];
            }
        }
        face[3] = prepared.vertices[static_cast<std::size_t>(missing_vertex)];
        prepared.faces[static_cast<std::size_t>(missing_vertex)] =
            make_face_frame(face);
    }
    return prepared;
}

PairTensor tetrahedron_point_tensor_prepared(
    const Vec3& target_minus_source_representative,
    const PreparedTetrahedronPointField& source)
{
    return to_pair_tensor(tetrahedron_magnetisation_tensor_prepared(
        target_minus_source_representative, source));
}

PreparedTetrahedron prepare_tetrahedron(const Tetrahedron& tetrahedron)
{
    PreparedTetrahedron prepared;
    prepared.volume = tetrahedron_volume(tetrahedron);
    prepared.vertices = tetrahedron.vertices;
    for (const Vec3& vertex : prepared.vertices) {
        prepared.circumradius = std::max(prepared.circumradius, norm(vertex));
    }
    for (int face_index = 0; face_index < 4; ++face_index) {
        const auto& indices = tetrahedron_face_vertices[
            static_cast<std::size_t>(face_index)];
        for (int vertex = 0; vertex < 3; ++vertex) {
            prepared.faces[static_cast<std::size_t>(face_index)][
                static_cast<std::size_t>(vertex)] = prepared.vertices[
                    static_cast<std::size_t>(indices[vertex])];
        }
        prepared.outward_normals[static_cast<std::size_t>(face_index)] =
            tetrahedron_outward_normal(
                prepared.vertices, indices, face_index);
    }
    return prepared;
}

namespace {

// Applies the total-moment normalisation -1/(4 pi V_s V_t) to the accumulated
// face-pair sum and symmetrises the harmless last-bit asymmetry of the
// independently evaluated face integrals.
PairTensor normalise_polyhedron_pair_tensor(Matrix3 tensor,
                                            const double source_volume,
                                            const double target_volume)
{
    const long double normalisation = -1.0L /
        (four_pi * static_cast<long double>(source_volume) *
         static_cast<long double>(target_volume));
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            tensor.value[row][column] = static_cast<double>(
                normalisation * static_cast<long double>(
                    tensor.value[row][column]));
            if (!std::isfinite(tensor.value[row][column])) {
                throw std::domain_error(
                    "polyhedron pair tensor is not finite");
            }
        }
    }
    for (int row = 0; row < 3; ++row) {
        for (int column = row + 1; column < 3; ++column) {
            const double average = 0.5 * (tensor.value[row][column] +
                                          tensor.value[column][row]);
            tensor.value[row][column] = average;
            tensor.value[column][row] = average;
        }
    }
    return to_pair_tensor(tensor);
}

} // namespace

PolyhedronSurface prepare_tetrahedron_surface(const Tetrahedron& tetrahedron)
{
    const PreparedTetrahedron prepared = prepare_tetrahedron(tetrahedron);
    PolyhedronSurface surface;
    surface.volume = prepared.volume;
    surface.circumradius = prepared.circumradius;
    surface.faces.assign(prepared.faces.begin(), prepared.faces.end());
    surface.outward_normals.assign(prepared.outward_normals.begin(),
                                   prepared.outward_normals.end());
    return surface;
}

PolyhedronSurface prepare_rectangular_prism_surface(
    const RectangularPrism& prism)
{
    if (!(prism.hx > 0.0 && prism.hy > 0.0 && prism.hz > 0.0) ||
        !std::isfinite(prism.hx) || !std::isfinite(prism.hy) ||
        !std::isfinite(prism.hz)) {
        throw std::invalid_argument(
            "rectangular prism dimensions must be positive and finite");
    }
    const std::array<double, 3> half{{0.5 * prism.hx, 0.5 * prism.hy,
                                      0.5 * prism.hz}};
    PolyhedronSurface surface;
    surface.volume = prism.hx * prism.hy * prism.hz;
    surface.circumradius = std::sqrt(half[0] * half[0] + half[1] * half[1] +
                                     half[2] * half[2]);
    surface.faces.reserve(12);
    surface.outward_normals.reserve(12);
    // Each of the six faces has one fixed coordinate `axis = sign * half`
    // and is split along a diagonal into two triangles.  The two free axes
    // are visited in cyclic order so the four corners form a simple loop;
    // the triangle orientation itself does not matter because the outward
    // normal is stored explicitly.
    constexpr std::array<std::array<int, 2>, 4> corner_signs{{
        {{-1, -1}}, {{1, -1}}, {{1, 1}}, {{-1, 1}}}};
    for (int axis = 0; axis < 3; ++axis) {
        const int second = (axis + 1) % 3;
        const int third = (axis + 2) % 3;
        for (const int sign : {-1, 1}) {
            std::array<Vec3, 4> corners{};
            for (std::size_t corner = 0; corner < 4; ++corner) {
                std::array<double, 3> coordinate{};
                coordinate[static_cast<std::size_t>(axis)] =
                    sign * half[static_cast<std::size_t>(axis)];
                coordinate[static_cast<std::size_t>(second)] =
                    corner_signs[corner][0] *
                    half[static_cast<std::size_t>(second)];
                coordinate[static_cast<std::size_t>(third)] =
                    corner_signs[corner][1] *
                    half[static_cast<std::size_t>(third)];
                corners[corner] = {coordinate[0], coordinate[1], coordinate[2]};
            }
            std::array<double, 3> normal{};
            normal[static_cast<std::size_t>(axis)] = static_cast<double>(sign);
            const Vec3 outward{normal[0], normal[1], normal[2]};
            surface.faces.push_back({corners[0], corners[1], corners[2]});
            surface.outward_normals.push_back(outward);
            surface.faces.push_back({corners[0], corners[2], corners[3]});
            surface.outward_normals.push_back(outward);
        }
    }
    return surface;
}

// General Galerkin pair of two triangulated polyhedra: the target-averaged
// field of the source is sum over face pairs of (integral of 1/|x - y| over
// the two faces) times n_target n_source^T, normalised afterwards.  Both
// bodies enter through their faces and outward normals only, so a prism
// (twelve triangles) and a tetrahedron (four) combine freely.
PairTensor polyhedron_polyhedron_tensor(
    const Vec3& target_minus_source_representative,
    const std::span<const std::array<Vec3, 3>> source_faces,
    const std::span<const Vec3> source_outward_normals,
    const double source_volume,
    const std::span<const std::array<Vec3, 3>> target_faces,
    const std::span<const Vec3> target_outward_normals,
    const double target_volume)
{
    if (source_faces.size() != source_outward_normals.size() ||
        target_faces.size() != target_outward_normals.size() ||
        source_faces.empty() || target_faces.empty()) {
        throw std::invalid_argument(
            "polyhedron surfaces need one outward normal per face");
    }
    if (!(source_volume > 0.0) || !(target_volume > 0.0)) {
        throw std::invalid_argument("polyhedron volumes must be positive");
    }
    Matrix3 tensor{};
    for (std::size_t target_face = 0; target_face < target_faces.size();
         ++target_face) {
        std::array<Vec3, 3> translated_target{};
        for (std::size_t vertex = 0; vertex < 3; ++vertex) {
            translated_target[vertex] = target_faces[target_face][vertex] +
                target_minus_source_representative;
        }
        for (std::size_t source_face = 0; source_face < source_faces.size();
             ++source_face) {
            const double integral = triangle_triangle_laplace_integral(
                translated_target, source_faces[source_face]);
            if (!std::isfinite(integral)) {
                throw std::domain_error(
                    "polyhedron face-pair integral is not finite");
            }
            add_scaled_outer_product(
                tensor, integral, target_outward_normals[target_face],
                source_outward_normals[source_face]);
        }
    }
    return normalise_polyhedron_pair_tensor(tensor, source_volume,
                                            target_volume);
}

namespace {

// Six-point Gauss-Legendre rule on [0, 1], one step of the quadrature ladder.
constexpr std::array<double, 6> gauss6_node{{
    0.033765242898423986, 0.169395306766867743,
    0.380690406958401546, 0.619309593041598454,
    0.830604693233132257, 0.966234757101576014}};
constexpr std::array<double, 6> gauss6_weight{{
    0.085662246189585173, 0.180380786524069304,
    0.233956967286345524, 0.233956967286345524,
    0.180380786524069304, 0.085662246189585173}};

// Five- and seven-point Gauss-Legendre rules on [0, 1], the rest of the
// quadrature ladder (tetrahedron_detail.hpp).
constexpr std::array<double, 5> gauss5_node{{
    0.046910077030668018, 0.230765344947158446, 0.500000000000000000,
    0.769234655052841498, 0.953089922969331926}};
constexpr std::array<double, 5> gauss5_weight{{
    0.118463442528094640, 0.239314335249683152, 0.284444444444444333,
    0.239314335249683152, 0.118463442528094640}};
constexpr std::array<double, 7> gauss7_node{{
    0.025446043828620701, 0.129234407200302770, 0.297077424311301408,
    0.500000000000000000, 0.702922575688698537, 0.870765592799697230,
    0.974553956171379299}};
constexpr std::array<double, 7> gauss7_weight{{
    0.064742483084434865, 0.139852695744638433, 0.190915025252559351,
    0.208979591836734646, 0.190915025252559351, 0.139852695744638433,
    0.064742483084434865}};

void accumulate_scaled(PairTensor& sum, const PairTensor& value,
                       const double weight) noexcept
{
    sum.xx += weight * value.xx;
    sum.xy += weight * value.xy;
    sum.xz += weight * value.xz;
    sum.yy += weight * value.yy;
    sum.yz += weight * value.yz;
    sum.zz += weight * value.zz;
}

/**
 * @brief The source's exact point field, with its geometry derived once.
 *
 * The quadrature below evaluates one source at 216 nodes.  A tetrahedron
 * source's volume, extent and four face frames are constant across all of
 * them, so deriving them per node -- which the record-level entry point does
 * -- is 216 times more work than the mathematics needs.  A prism source has
 * no such derived geometry: its formula reads the half-sizes directly.
 */
class SourcePointField {
public:
    explicit SourcePointField(const PolyhedronBody& source)
        : prism_(std::get_if<RectangularPrism>(&source.record))
    {
        if (prism_ == nullptr) {
            tetrahedron_ = prepare_tetrahedron_point_field(
                std::get<Tetrahedron>(source.record));
        }
    }

    [[nodiscard]] PairTensor operator()(const Vec3& point) const
    {
        if (prism_ != nullptr) {
            return rectangular_prism_point_tensor(point, *prism_);
        }
        return tetrahedron_point_tensor_prepared(point, tetrahedron_);
    }

private:
    const RectangularPrism* prism_{nullptr};
    PreparedTetrahedronPointField tetrahedron_{};
};

// Volume average of the source point tensor over the target body, using a
// collapsed-cube (Duffy) map for a tetrahedron and a tensor-product rule for
// a prism, with the given one-dimensional Gauss rule on [0, 1].  Only used
// when the separation makes the integrand smooth.
template <std::size_t Points>
PairTensor average_source_tensor_over_target(
    const Vec3& target_minus_source_representative,
    const PolyhedronBody& source,
    const PolyhedronBody& target,
    const std::array<double, Points>& gauss_node,
    const std::array<double, Points>& gauss_weight)
{
    PairTensor result{};
    const Vec3& d = target_minus_source_representative;
    const SourcePointField source_field(source);
    if (const auto* prism = std::get_if<RectangularPrism>(&target.record)) {
        for (std::size_t i = 0; i < Points; ++i) {
            const double x = (gauss_node[i] - 0.5) * prism->hx;
            for (std::size_t j = 0; j < Points; ++j) {
                const double y = (gauss_node[j] - 0.5) * prism->hy;
                for (std::size_t k = 0; k < Points; ++k) {
                    const double z = (gauss_node[k] - 0.5) * prism->hz;
                    accumulate_scaled(
                        result,
                        source_field(d + Vec3{x, y, z}),
                        gauss_weight[i] * gauss_weight[j] * gauss_weight[k]);
                }
            }
        }
        return result;
    }
    const Tetrahedron& tetrahedron = std::get<Tetrahedron>(target.record);
    const Vec3 edge_b = tetrahedron.vertices[1] - tetrahedron.vertices[0];
    const Vec3 edge_c = tetrahedron.vertices[2] - tetrahedron.vertices[0];
    const Vec3 edge_d = tetrahedron.vertices[3] - tetrahedron.vertices[0];
    for (std::size_t i = 0; i < Points; ++i) {
        const double u = gauss_node[i];
        for (std::size_t j = 0; j < Points; ++j) {
            const double v = gauss_node[j];
            for (std::size_t k = 0; k < Points; ++k) {
                const double w = gauss_node[k];
                const Vec3 offset = tetrahedron.vertices[0] + edge_b * u +
                    edge_c * ((1.0 - u) * v) +
                    edge_d * ((1.0 - u) * (1.0 - v) * w);
                // The Jacobian of the collapsed cube is 6 V (1-u)^2 (1-v);
                // dividing by V gives the volume average directly.
                const double weight = 6.0 * gauss_weight[i] *
                    gauss_weight[j] * gauss_weight[k] * (1.0 - u) *
                    (1.0 - u) * (1.0 - v);
                accumulate_scaled(result, source_field(d + offset),
                                  weight);
            }
        }
    }
    return result;
}

// Which evaluation a pair takes, by its separation in summed circumradii:
// the analytical surface integral when close, otherwise the rule of the
// quadrature ladder (tetrahedron_detail.hpp) for that separation, which is
// the cheapest one converged there.
enum class PairEvaluation { Analytical, Gauss7, Gauss6, Gauss5 };

PairEvaluation pair_evaluation(const Vec3& target_minus_source_representative,
                               const double source_circumradius,
                               const double target_circumradius) noexcept
{
    const double separation = norm(target_minus_source_representative);
    const double radii = source_circumradius + target_circumradius;
    if (separation >= polyhedron_quadrature_five_point_factor * radii) {
        return PairEvaluation::Gauss5;
    }
    if (separation >= polyhedron_quadrature_six_point_factor * radii) {
        return PairEvaluation::Gauss6;
    }
    if (separation >= polyhedron_near_quadrature_factor * radii) {
        return PairEvaluation::Gauss7;
    }
    return PairEvaluation::Analytical;
}

PairTensor averaged_pair_tensor(const PairEvaluation evaluation,
                                const Vec3& target_minus_source_representative,
                                const PolyhedronBody& source,
                                const PolyhedronBody& target)
{
    switch (evaluation) {
    case PairEvaluation::Gauss5:
        return average_source_tensor_over_target(
            target_minus_source_representative, source, target, gauss5_node,
            gauss5_weight);
    case PairEvaluation::Gauss6:
        return average_source_tensor_over_target(
            target_minus_source_representative, source, target, gauss6_node,
            gauss6_weight);
    default:
        return average_source_tensor_over_target(
            target_minus_source_representative, source, target, gauss7_node,
            gauss7_weight);
    }
}

} // namespace

PolyhedronBody prepare_polyhedron_body(const RectangularPrism& prism)
{
    return {prepare_rectangular_prism_surface(prism), prism};
}

PolyhedronBody prepare_polyhedron_body(const Tetrahedron& tetrahedron)
{
    return {prepare_tetrahedron_surface(tetrahedron), tetrahedron};
}

// Entry point for prepared bodies: quadrature of the exact source field when
// the pair is far separated (in circumradii), the Galerkin surface integral
// otherwise.  The switch is a numerical-accuracy decision; both branches
// evaluate the same tensor.
PairTensor polyhedron_pair_tensor(
    const Vec3& target_minus_source_representative,
    const PolyhedronBody& source,
    const PolyhedronBody& target)
{
    const PairEvaluation evaluation = pair_evaluation(
        target_minus_source_representative, source.surface.circumradius,
        target.surface.circumradius);
    if (evaluation != PairEvaluation::Analytical) {
        return averaged_pair_tensor(evaluation,
                                    target_minus_source_representative,
                                    source, target);
    }
    return polyhedron_polyhedron_tensor(
        target_minus_source_representative, source.surface.faces,
        source.surface.outward_normals, source.surface.volume,
        target.surface.faces, target.surface.outward_normals,
        target.surface.volume);
}

PairTensor tetrahedron_tetrahedron_tensor_prepared(
    const Vec3& target_minus_source_representative,
    const PreparedTetrahedron& source,
    const PreparedTetrahedron& target,
    const bool coincident_same_geometry)
{
    const PairEvaluation evaluation = coincident_same_geometry
        ? PairEvaluation::Analytical
        : pair_evaluation(target_minus_source_representative,
                          source.circumradius, target.circumradius);
    if (evaluation != PairEvaluation::Analytical) {
        // Separated pairs average the exact source field over the target
        // instead of cancelling large face integrals.
        return averaged_pair_tensor(
            evaluation, target_minus_source_representative,
            PolyhedronBody{{}, Tetrahedron{source.vertices}},
            PolyhedronBody{{}, Tetrahedron{target.vertices}});
    }
    if (!coincident_same_geometry) {
        // A displaced or differently shaped pair is the general polyhedron
        // case; the coincident branch below only halves the face-pair work.
        return polyhedron_polyhedron_tensor(
            target_minus_source_representative, source.faces,
            source.outward_normals, source.volume, target.faces,
            target.outward_normals, target.volume);
    }
    // Coincident identical bodies (the self tensor): the face-pair integral
    // is symmetric in its two faces, so only the upper triangle of the 4x4
    // face matrix is evaluated and each off-diagonal integral is added with
    // both normal orderings.
    Matrix3 tensor{};
    {
        for (int target_face = 0; target_face < 4; ++target_face) {
            for (int source_face = target_face; source_face < 4;
                 ++source_face) {
                const double integral = triangle_triangle_laplace_integral(
                    target.faces[static_cast<std::size_t>(target_face)],
                    source.faces[static_cast<std::size_t>(source_face)]);
                if (!std::isfinite(integral)) {
                    throw std::domain_error(
                        "tetrahedron face-pair integral is not finite");
                }
                add_scaled_outer_product(
                    tensor, integral,
                    target.outward_normals[
                        static_cast<std::size_t>(target_face)],
                    source.outward_normals[
                        static_cast<std::size_t>(source_face)]);
                if (target_face != source_face) {
                    add_scaled_outer_product(
                        tensor, integral,
                        target.outward_normals[
                            static_cast<std::size_t>(source_face)],
                        source.outward_normals[
                            static_cast<std::size_t>(target_face)]);
                }
            }
        }
    }
    return normalise_polyhedron_pair_tensor(tensor, source.volume,
                                            target.volume);
}

} // namespace detail

double Tetrahedron::signed_volume() const noexcept
{
    const Vec3 a = vertices[1] - vertices[0];
    const Vec3 b = vertices[2] - vertices[0];
    const Vec3 c = vertices[3] - vertices[0];
    return dot(a, cross(b, c)) / 6.0;
}

double Tetrahedron::volume() const noexcept
{
    return std::abs(signed_volume());
}

Vec3 Tetrahedron::centroid_offset() const noexcept
{
    return {(vertices[0].x + vertices[1].x + vertices[2].x + vertices[3].x) /
                4.0,
            (vertices[0].y + vertices[1].y + vertices[2].y + vertices[3].y) /
                4.0,
            (vertices[0].z + vertices[1].z + vertices[2].z + vertices[3].z) /
                4.0};
}

// The validating volume: every finite-tetrahedron path calls this first, so a
// non-finite, zero-extent or degenerate (flat) record is rejected with one
// consistent message before any primitive sees it.
double tetrahedron_volume(const Tetrahedron& tetrahedron)
{
    for (const Vec3& vertex : tetrahedron.vertices) {
        if (!finite(vertex)) {
            throw std::invalid_argument("tetrahedron vertices must be finite");
        }
    }
    double scale_value = 0.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            scale_value = std::max(scale_value,
                                   norm(tetrahedron.vertices[i] -
                                        tetrahedron.vertices[j]));
        }
    }
    if (!(scale_value > 0.0)) {
        throw std::invalid_argument("tetrahedron has zero extent");
    }
    const double volume = tetrahedron.volume();
    const double tolerance = 128.0 * std::numeric_limits<double>::epsilon() *
        scale_value * scale_value * scale_value;
    if (!(volume > tolerance) || !std::isfinite(volume)) {
        throw std::invalid_argument("tetrahedron is degenerate");
    }
    return volume;
}

PairTensor tetrahedron_point_tensor(const Vec3& target_minus_source_representative,
                                    const Tetrahedron& source)
{
    return to_pair_tensor(tetrahedron_magnetisation_tensor(
        target_minus_source_representative, source));
}

// Point source -> tetrahedron target by reciprocity: the average over the
// target of the point field equals the field of the target-as-source at the
// point, with the displacement reversed.  The tensor is symmetric and both
// directions act on a total moment, so no volume factor is needed.
PairTensor point_tetrahedron_tensor(const Vec3& target_minus_source_representative,
                                    const Tetrahedron& target)
{
    return tetrahedron_point_tensor(
        scale(target_minus_source_representative, -1.0), target);
}

// Average over the tetrahedron of (d + t)^beta / beta!, t in the tetrahedron
// about its representative point.  Each Cartesian factor is expanded into
// barycentric monomials and the products are integrated exactly with the
// Dirichlet formula: the average of lambda^n over a tetrahedron is
// 3! * prod(n_i!) / (3 + sum n_i)!.
double tetrahedron_averaged_monomial(const MultiIndex& beta, const Vec3& d,
                                     const Tetrahedron& tetrahedron)
{
    const double volume = tetrahedron_volume(tetrahedron);
    (void)volume;
    if (beta.ax < 0 || beta.ay < 0 || beta.az < 0) {
        throw std::invalid_argument("monomial exponents must be non-negative");
    }

    const int powers[3]{beta.ax, beta.ay, beta.az};
    const double offsets[3]{d.x, d.y, d.z};
    std::vector<BarycentricPolynomial> polynomial{{{{0, 0, 0, 0}}, 1.0}};
    for (int axis = 0; axis < 3; ++axis) {
        std::array<double, 4> coordinate{};
        for (int vertex = 0; vertex < 4; ++vertex) {
            coordinate[static_cast<std::size_t>(vertex)] =
                tetrahedron.vertices[static_cast<std::size_t>(vertex)][axis];
        }
        std::vector<BarycentricPolynomial> axis_terms;
        expand_axis(powers[axis], offsets[axis], coordinate, axis_terms);
        std::vector<BarycentricPolynomial> product;
        product.reserve(polynomial.size() * axis_terms.size());
        for (const BarycentricPolynomial& left : polynomial) {
            for (const BarycentricPolynomial& right : axis_terms) {
                BarycentricPolynomial combined;
                combined.coefficient = left.coefficient * right.coefficient;
                for (int vertex = 0; vertex < 4; ++vertex) {
                    combined.power[static_cast<std::size_t>(vertex)] =
                        left.power[static_cast<std::size_t>(vertex)] +
                        right.power[static_cast<std::size_t>(vertex)];
                }
                product.push_back(combined);
            }
        }
        polynomial = std::move(product);
    }

    double result = 0.0;
    for (const BarycentricPolynomial& term : polynomial) {
        const int total_power = term.power[0] + term.power[1] +
            term.power[2] + term.power[3];
        double barycentric_average = 6.0 /
            MultiIndexSet::factorial(3 + total_power);
        for (const int power : term.power) {
            barycentric_average *= MultiIndexSet::factorial(power);
        }
        result += term.coefficient * barycentric_average;
    }
    return result;
}

PairTensor tetrahedron_tetrahedron_tensor(
    const Vec3& target_minus_source_representative,
    const Tetrahedron& source,
    const Tetrahedron& target)
{
    const auto prepared_source = detail::prepare_tetrahedron(source);
    const auto prepared_target = detail::prepare_tetrahedron(target);
    const bool zero_displacement =
        target_minus_source_representative.x == 0.0 &&
        target_minus_source_representative.y == 0.0 &&
        target_minus_source_representative.z == 0.0;
    const auto same_vertex = [](const Vec3& first, const Vec3& second) {
        return first.x == second.x && first.y == second.y &&
            first.z == second.z;
    };
    bool exact_same_geometry = true;
    for (int vertex = 0; vertex < 4; ++vertex) {
        exact_same_geometry = exact_same_geometry && same_vertex(
            source.vertices[static_cast<std::size_t>(vertex)],
            target.vertices[static_cast<std::size_t>(vertex)]);
    }
    return detail::tetrahedron_tetrahedron_tensor_prepared(
        target_minus_source_representative,
        prepared_source,
        prepared_target,
        zero_displacement && exact_same_geometry);
}

PairTensor rectangular_prism_tetrahedron_tensor(
    const Vec3& target_minus_source_representative,
    const RectangularPrism& source,
    const Tetrahedron& target)
{
    return detail::polyhedron_pair_tensor(
        target_minus_source_representative,
        detail::prepare_polyhedron_body(source),
        detail::prepare_polyhedron_body(target));
}

PairTensor tetrahedron_rectangular_prism_tensor(
    const Vec3& target_minus_source_representative,
    const Tetrahedron& source,
    const RectangularPrism& target)
{
    return detail::polyhedron_pair_tensor(
        target_minus_source_representative,
        detail::prepare_polyhedron_body(source),
        detail::prepare_polyhedron_body(target));
}

} // namespace cdfmm
