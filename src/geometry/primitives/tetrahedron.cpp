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
//     of the outward normals.  Far-separated pairs switch to a converged
//     Gauss rule over the target of the exact source field, because the
//     analytical reduction loses accuracy through cancellation exactly where
//     quadrature becomes trivially accurate.
//
// Vertices are representative-relative.  Degenerate faces, evaluation on an
// edge or vertex, and a lost rank in the dimensional reduction are reported
// as exceptions rather than patched with a limiting value.  The `Prepared*`
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

// Decomposition of an offset vector into the span of up to four edge vectors
// (the coefficients) plus the perpendicular remainder (`residual`, the
// "height" the reduced integrals see).  The I3/I2/I1/I0 recursion reduces the
// dimension of the integration domain one edge at a time using these.
struct ExpansionProjection {
    std::array<double, 4> coefficient{{0.0, 0.0, 0.0, 0.0}};
    Vec3 projected{};
    double residual{0.0};
};

// This is the small Gram--Schmidt projection used by the reference
// expan_GrammSchmidt.m routine.  Keeping the triangular back substitution
// explicit is important: I3/I2s/I2t use the coefficients of the original
// edge vectors, rather than coefficients of an orthogonal basis.
ExpansionProjection expand_gram_schmidt(
    const std::array<Vec3, 4>& vectors, const int count, const Vec3& value)
{
    std::array<Vec3, 4> orthogonal{};
    std::array<double, 4> orthogonal_norm_squared{{0.0, 0.0, 0.0, 0.0}};
    double vector_scale = norm(value);
    double basis_scale = 0.0;
    for (int i = 0; i < count; ++i) {
        const double length = norm(vectors[static_cast<std::size_t>(i)]);
        vector_scale = std::max(vector_scale, length);
        basis_scale = std::max(basis_scale, length);
    }
    const double rank_tolerance = 256.0 *
        std::numeric_limits<double>::epsilon() *
        std::max(std::numeric_limits<double>::min(), basis_scale * basis_scale);

    double coefficient_matrix[4][4]{};
    for (int column = 0; column < count; ++column) {
        orthogonal[static_cast<std::size_t>(column)] =
            vectors[static_cast<std::size_t>(column)];
        for (int row = 0; row < column; ++row) {
            const double denominator = orthogonal_norm_squared[
                static_cast<std::size_t>(row)];
            coefficient_matrix[row][column] =
                dot(orthogonal[static_cast<std::size_t>(row)],
                    vectors[static_cast<std::size_t>(column)]) /
                denominator;
            orthogonal[static_cast<std::size_t>(column)] =
                orthogonal[static_cast<std::size_t>(column)] -
                coefficient_matrix[row][column] *
                    orthogonal[static_cast<std::size_t>(row)];
        }
        const double norm_squared = dot(
            orthogonal[static_cast<std::size_t>(column)],
            orthogonal[static_cast<std::size_t>(column)]);
        if (!(norm_squared > rank_tolerance)) {
            orthogonal[static_cast<std::size_t>(column)] = {};
            orthogonal_norm_squared[static_cast<std::size_t>(column)] = 1.0;
            coefficient_matrix[column][column] = 0.0;
        } else {
            orthogonal_norm_squared[static_cast<std::size_t>(column)] =
                norm_squared;
            coefficient_matrix[column][column] = dot(
                orthogonal[static_cast<std::size_t>(column)],
                vectors[static_cast<std::size_t>(column)]) /
                norm_squared;
        }
    }

    ExpansionProjection result;
    std::array<double, 4> residual_projection{{0.0, 0.0, 0.0, 0.0}};
    for (int index = count - 1; index >= 0; --index) {
        const double diagonal = coefficient_matrix[index][index];
        if (diagonal == 0.0) {
            continue;
        }
        double correction = residual_projection[static_cast<std::size_t>(index)];
        const double numerator = dot(
            value, orthogonal[static_cast<std::size_t>(index)]) - correction;
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
        result.projected += result.coefficient[
            static_cast<std::size_t>(index)] * vectors[
                static_cast<std::size_t>(index)];
    }
    result.residual = norm(value - result.projected);
    const double residual_tolerance = 256.0 *
        std::numeric_limits<double>::epsilon() * std::max(1.0, vector_scale);
    if (result.residual <= residual_tolerance) {
        result.residual = 0.0;
    }
    return result;
}

// I0: the fully reduced one-dimensional primitive of the recursion, a closed
// form in the edge length `p` and the four heights, with the reference's
// case analysis of which heights vanish.  Evaluated in long double because
// the branches subtract nearly equal logarithms and arctangents.
long double triangle_i0(const double p,
                        const std::array<double, 4>& input_heights)
{
    std::array<double, 4> heights = input_heights;
    std::array<int, 4> order{{0, 1, 2, 3}};
    std::sort(order.begin(), order.end(), [&](const int first,
                                              const int second) {
        return heights[static_cast<std::size_t>(first)] <
            heights[static_cast<std::size_t>(second)];
    });
    heights[static_cast<std::size_t>(order[0])] = 0.0;
    heights[static_cast<std::size_t>(order[1])] = 0.0;
    const long double h1 = heights[0];
    const long double h2 = heights[1];
    const long double h3 = heights[2];
    const long double h4 = heights[3];
    const long double P = std::abs(static_cast<long double>(p));
    const long double hh = h1 * h1 + h2 * h2 + h3 * h3 + h4 * h4;
    if (P == 0.0L && hh == 0.0L) {
        // This primitive is only reached with a zero edge in a degenerate
        // triangle.  Valid triangle reductions skip that term before I0.
        return 0.0L;
    }

    const long double zero2 = 1.0e-28L;
    const long double radius = std::sqrt(P * P + hh);
    if (hh < zero2 * P * P) {
        // This is the complete zero-height branch in the reference I0
        // reduction, not merely an alternative way to compute Phi1.
        return std::log(P) / (6.0L * P);
    }

    long double phi1;
    if (P == 0.0L) {
        phi1 = 1.0L / std::sqrt(hh);
    } else {
        // asinh(P/sqrt(hh))/P is equivalent to the logarithm in I0 and is
        // well behaved at P=0.
        phi1 = std::asinh(P / std::sqrt(hh)) / P;
    }

    const long double hh1 = hh - h1 * h1;
    const long double tiny = zero2;
    const auto phi2_for = [&](const long double h) {
        if (h1 * P == 0.0L) {
            return 1.0L / (hh + h * radius);
        }
        return std::atan((h1 * P) / (hh + h * radius)) / (h1 * P);
    };
    long double answer;
    if (h1 * h1 + h2 * h2 + h4 * h4 < tiny * hh) {
        // Rank-deficient reductions can leave h3 as the only non-zero
        // height.  The reference case-4 expression is 0/0 at h1=0; this is
        // its analytical h1 -> 0 limit.
        const long double q = std::hypot(P, h3);
        const long double logarithm = std::asinh(P / h3);
        answer = (logarithm * (2.0L * q * q - 3.0L * h3 * h3) /
                      (2.0L * P * P * P) +
                  (4.0L * h3 - 3.0L * q) / (2.0L * P * P)) /
            6.0L;
    } else if (h1 * h1 + h2 * h2 + h3 * h3 < tiny * hh) {
        // Likewise, the reference case-6 expression needs its h1 -> 0
        // limit when h4 is the only non-zero height.
        const long double q = std::hypot(P, h4);
        const long double logarithm = std::asinh(P / h4);
        answer = (logarithm * (2.0L * q * q - 5.0L * h4 * h4) /
                      (2.0L * P * P * P) +
                  3.0L * (2.0L * h4 - q) / (2.0L * P * P) -
                  (q + 2.0L * h4) /
                      (3.0L * (q + h4) * (q + h4))) /
            6.0L;
    } else if (hh1 < tiny * hh) {
        answer = phi1 / 6.0L;
    } else if (h3 * h3 + h4 * h4 < tiny * hh) {
        answer = (phi1 - h2 * phi2_for(h2)) / 6.0L;
    } else {
        const long double radius1 = std::sqrt(P * P + h1 * h1);
        if (h2 * h2 + h4 * h4 < tiny * hh) {
            const long double phi2 = phi2_for(h3);
            const long double phi3 = 0.5L * hh1 / radius1 *
                std::log((radius1 + radius) * (radius1 + radius) / hh1);
            answer = ((h1 * h1 - h3 * h3) * phi1 -
                      2.0L * h1 * h1 * h3 * phi2 + phi3) /
                (6.0L * h1 * h1);
        } else if (h2 * h2 + h3 * h3 < tiny * hh) {
            const long double phi2 = phi2_for(h4);
            const long double phi3 = 0.5L / radius1 *
                std::log((radius1 + radius) * (radius1 + radius) / hh1);
            answer = ((h1 * h1 - 3.0L * h4 * h4) * phi1 -
                      h4 * (3.0L * h1 * h1 - h4 * h4) * phi2 +
                      3.0L * h4 * h4 * phi3 -
                      h4 * h4 / (radius + h4)) /
                (6.0L * h1 * h1);
        } else if (h1 * h1 + h4 * h4 < tiny * hh) {
            const long double h = std::sqrt(hh);
            const long double radius2 = std::sqrt(P * P + h2 * h2);
            const long double phi4 = h3 * h3 / (h2 * P * P) *
                ((radius2 / h2) * std::log((radius2 + radius) / h3) -
                 std::log((h2 + h) / h3));
            answer = (hh / (h2 * h2) * phi1 - 1.0L / (radius + h) -
                      phi4) / 6.0L;
        } else if (h1 * h1 + h3 * h3 < tiny * hh) {
            const long double h = std::sqrt(hh);
            const long double radius2 = std::sqrt(P * P + h2 * h2);
            const long double phi2 = std::atan(h2 * P /
                (hh + h4 * radius)) / P;
            const long double phi4 = h4 * h4 / (h2 * P * P) *
                ((radius2 / h2) * std::log((radius2 + radius) / h4) -
                 std::log((h2 + h) / h4));
            answer = ((1.0L + 3.0L * h4 * h4 / (h2 * h2)) * phi1 -
                      2.0L * (h4 / h2) * (h4 / h2) * (h4 / h2) * phi2 -
                      3.0L * phi4 +
                      (2.0L * h4 * h4 - h2 * h2) /
                      (h2 * h2 * (radius + h))) / 6.0L;
        } else {
            // This is the prohibited reference-I0 case.  Reaching it means
            // that the preceding dimensional reduction has lost its expected
            // rank; assigning an ad-hoc limiting value would hide that error.
            throw std::domain_error("triangle I0 reached prohibited branch");
        }
    }
    return answer;
}

// I1, I2s, I2t, I3: the reduction steps.  Each expands the offset in the
// remaining edge vectors, drops one edge and calls the next lower primitive
// at the two endpoints of that edge; a coefficient within `zero_tolerance`
// of zero contributes nothing and is skipped, exactly as in the reference.
double triangle_i1(const Vec3& vector, const Vec3& offset,
                   const double h2, const double h3, const double h4)
{
    const double length = norm(vector);
    if (!(length > 0.0)) {
        return 0.0;
    }
    std::array<Vec3, 4> basis{{vector, {}, {}, {}}};
    const ExpansionProjection expansion = expand_gram_schmidt(
        basis, 1, offset);
    constexpr double zero_tolerance = 1.0e-14;
    double result = 0.0;
    const double coefficient = expansion.coefficient[0];
    if (std::abs(1.0 + coefficient) > zero_tolerance) {
        result += (1.0 + coefficient) * static_cast<double>(triangle_i0(
            std::abs(1.0 + coefficient) * length,
            {{expansion.residual, h2, h3, h4}}));
    }
    if (std::abs(coefficient) > zero_tolerance) {
        result -= coefficient * static_cast<double>(triangle_i0(
            std::abs(coefficient) * length,
            {{expansion.residual, h2, h3, h4}}));
    }
    return result;
}

double triangle_i2s(const Vec3& first, const Vec3& second,
                    const Vec3& offset, const double h3, const double h4)
{
    std::array<Vec3, 4> basis{{first, second, {}, {}}};
    const ExpansionProjection expansion = expand_gram_schmidt(
        basis, 2, offset);
    constexpr double zero_tolerance = 1.0e-14;
    double result = 0.0;
    const double s0 = expansion.coefficient[0];
    const double s1 = expansion.coefficient[1];
    if (std::abs(1.0 + s0) > zero_tolerance) {
        result += (1.0 + s0) * triangle_i1(
            second, expansion.projected + first, expansion.residual, h3, h4);
    }
    if (std::abs(s0) > zero_tolerance) {
        result -= s0 * triangle_i1(
            second, expansion.projected, expansion.residual, h3, h4);
    }
    if (std::abs(1.0 + s1) > zero_tolerance) {
        result += (1.0 + s1) * triangle_i1(
            first, expansion.projected + second, expansion.residual, h3, h4);
    }
    if (std::abs(s1) > zero_tolerance) {
        result -= s1 * triangle_i1(
            first, expansion.projected, expansion.residual, h3, h4);
    }
    return result;
}

double triangle_i2t(const Vec3& first, const Vec3& second,
                    const Vec3& offset, const double h3, const double h4)
{
    std::array<Vec3, 4> basis{{first, second, {}, {}}};
    const ExpansionProjection expansion = expand_gram_schmidt(
        basis, 2, offset);
    constexpr double zero_tolerance = 1.0e-14;
    double result = 0.0;
    const double s0 = expansion.coefficient[0];
    const double s1 = expansion.coefficient[1];
    if (std::abs(s0) > zero_tolerance) {
        result -= s0 * triangle_i1(
            second, expansion.projected, expansion.residual, h3, h4);
    }
    if (std::abs(s1) > zero_tolerance) {
        result -= s1 * triangle_i1(
            first, expansion.projected, expansion.residual, h3, h4);
    }
    if (std::abs(1.0 + s0 + s1) > zero_tolerance) {
        result += (1.0 + s0 + s1) * triangle_i1(
            first - second, expansion.projected + second,
            expansion.residual, h3, h4);
    }
    return result;
}

double triangle_i3(const Vec3& first, const Vec3& second, const Vec3& third,
                   const Vec3& offset, const double h4)
{
    std::array<Vec3, 4> basis{{first, second, third, {}}};
    const ExpansionProjection expansion = expand_gram_schmidt(
        basis, 3, offset);
    constexpr double zero_tolerance = 1.0e-14;
    double result = 0.0;
    const double s0 = expansion.coefficient[0];
    const double s1 = expansion.coefficient[1];
    const double s2 = expansion.coefficient[2];
    if (std::abs(1.0 + s0 + s1) > zero_tolerance) {
        result += (1.0 + s0 + s1) * triangle_i2s(
            first - second, third, expansion.projected + second,
            expansion.residual, h4);
    }
    if (std::abs(s0) > zero_tolerance) {
        result -= s0 * triangle_i2s(
            second, third, expansion.projected, expansion.residual, h4);
    }
    if (std::abs(s1) > zero_tolerance) {
        result -= s1 * triangle_i2s(
            first, third, expansion.projected, expansion.residual, h4);
    }
    if (std::abs(1.0 + s2) > zero_tolerance) {
        result += (1.0 + s2) * triangle_i2t(
            first, second, expansion.projected + third,
            expansion.residual, h4);
    }
    if (std::abs(s2) > zero_tolerance) {
        result -= s2 * triangle_i2t(
            first, second, expansion.projected, expansion.residual, h4);
    }
    return result;
}

// The Galerkin integral of 1/|x - y| over two triangles in normalised
// coordinates (largest triangle first, common origin, unit scale).  The two
// triangles' edge vectors a1..a4 and the vertex offset e4 are expanded
// together; coplanar (parallel) pairs take the reduced branch in which only
// the in-plane offset and the separation height enter.
double triangle_triangle_integral_core(
    const std::array<Vec3, 3>& first,
    const std::array<Vec3, 3>& second)
{
    // Only the two edges meeting at each triangle's first vertex enter the
    // expansion basis and the area; the third edge of each triangle is
    // implied by them and is not formed.
    const Vec3 first_edge_1 = first[1] - first[0];
    const Vec3 first_edge_3 = first[0] - first[2];
    const Vec3 second_edge_1 = second[1] - second[0];
    const Vec3 second_edge_3 = second[0] - second[2];
    const Vec3 first_cross = cross(first_edge_1, -1.0 * first_edge_3);
    const Vec3 second_cross = cross(second_edge_1, -1.0 * second_edge_3);
    const double first_area_twice = norm(first_cross);
    const double second_area_twice = norm(second_cross);
    if (!(first_area_twice > 0.0 && second_area_twice > 0.0)) {
        throw std::invalid_argument("triangle pair contains a degenerate triangle");
    }

    const Vec3 a1 = first_edge_1;
    const Vec3 a2 = -1.0 * first_edge_3;
    const Vec3 a3 = -1.0 * second_edge_1;
    const Vec3 a4 = second_edge_3;
    const Vec3 e4 = first[0] - second[0];
    std::array<Vec3, 4> all_vectors{{a1, a2, a3, a4}};
    const ExpansionProjection complete_expansion = expand_gram_schmidt(
        all_vectors, 4, e4);
    const double s0 = complete_expansion.coefficient[0];
    const double s1 = complete_expansion.coefficient[1];
    const double s2 = complete_expansion.coefficient[2];
    const double s3 = complete_expansion.coefficient[3];
    const Vec3 e4x = complete_expansion.projected;
    const Vec3 first_normal = scale(first_cross, 1.0 / first_area_twice);
    const Vec3 second_normal = scale(second_cross, 1.0 / second_area_twice);
    const double parallel_measure = std::min(
        norm(first_normal + second_normal),
        norm(first_normal - second_normal));
    constexpr double zero_tolerance = 1.0e-14;
    double reduced = 0.0;
    if (parallel_measure > zero_tolerance) {
        const double h4 = 0.0;
        const double i31 = triangle_i3(a1, a2, a3, e4, h4);
        const double i32 = triangle_i3(a1, a2, a4, e4, h4);
        const double i33 = triangle_i3(a1, a2, a4 - a3, e4 + a3, h4);
        const double i34 = triangle_i3(a3, a4, a1, e4, h4);
        const double i35 = triangle_i3(a3, a4, a2, e4, h4);
        const double i36 = triangle_i3(a3, a4, a2 - a1, e4 + a1, h4);
        reduced = (1.0 + s2 + s3) * i33 - s3 * i31 - s2 * i32 +
            (1.0 + s0 + s1) * i36 - s1 * i34 - s0 * i35;
    } else {
        // In the parallel case the projection of e4 onto the common plane
        // is the only offset required by the dimensional reduction.
        const double h4 = complete_expansion.residual;
        const double i33 = triangle_i3(
            a1, a2, a4 - a3, e4x + a3, h4);
        const double i34 = triangle_i3(a3, a4, a1, e4x, h4);
        const double i35 = triangle_i3(a3, a4, a2, e4x, h4);
        const double i36 = triangle_i3(
            a3, a4, a2 - a1, e4x + a1, h4);
        reduced = i33 + (1.0 + s0 + s1) * i36 - s1 * i34 - s0 * i35;
    }
    return 4.0 * (0.5 * first_area_twice) *
        (0.5 * second_area_twice) * reduced;
}

// Normalise a triangle pair to unit scale about the first vertex of the
// larger triangle before the core; the integral is homogeneous of degree 3
// in length, so the result is rescaled by scale^3.
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
    // The core is symmetric in its two triangles.  Keep the larger triangle
    // first for the same ordering used by the reference reduction, but do so
    // before choosing the common origin so normalization remains one-pass.
    if (first_scale < second_scale) {
        std::swap(first, second);
    }

    const double scale_value = std::max(first_scale, second_scale);
    const Vec3 origin = first[0];
    const double inverse_scale = 1.0 / scale_value;
    std::array<Vec3, 3> normalised_first{};
    std::array<Vec3, 3> normalised_second{};
    for (int index = 0; index < 3; ++index) {
        normalised_first[static_cast<std::size_t>(index)] =
            (first[static_cast<std::size_t>(index)] - origin) * inverse_scale;
        normalised_second[static_cast<std::size_t>(index)] =
            (second[static_cast<std::size_t>(index)] - origin) * inverse_scale;
    }
    return std::pow(scale_value, 3.0) *
        triangle_triangle_integral_core(normalised_first,
                                        normalised_second);
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

// Six-point Gauss-Legendre rule on [0, 1].  With the far-separation factor
// above, the exact source field is analytic on a Bernstein ellipse of
// parameter rho > 10 around the target, so the rule is converged well below
// 1e-12 relative error.
constexpr std::array<double, 6> gauss6_node{{
    0.033765242898423986, 0.169395306766867743,
    0.380690406958401546, 0.619309593041598454,
    0.830604693233132257, 0.966234757101576014}};
constexpr std::array<double, 6> gauss6_weight{{
    0.085662246189585173, 0.180380786524069304,
    0.233956967286345524, 0.233956967286345524,
    0.180380786524069304, 0.085662246189585173}};

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
// a prism.  Only used when the separation makes the integrand smooth.
PairTensor average_source_tensor_over_target(
    const Vec3& target_minus_source_representative,
    const PolyhedronBody& source,
    const PolyhedronBody& target)
{
    PairTensor result{};
    const Vec3& d = target_minus_source_representative;
    const SourcePointField source_field(source);
    if (const auto* prism = std::get_if<RectangularPrism>(&target.record)) {
        for (std::size_t i = 0; i < 6; ++i) {
            const double x = (gauss6_node[i] - 0.5) * prism->hx;
            for (std::size_t j = 0; j < 6; ++j) {
                const double y = (gauss6_node[j] - 0.5) * prism->hy;
                for (std::size_t k = 0; k < 6; ++k) {
                    const double z = (gauss6_node[k] - 0.5) * prism->hz;
                    accumulate_scaled(
                        result,
                        source_field(d + Vec3{x, y, z}),
                        gauss6_weight[i] * gauss6_weight[j] * gauss6_weight[k]);
                }
            }
        }
        return result;
    }
    const Tetrahedron& tetrahedron = std::get<Tetrahedron>(target.record);
    const Vec3 edge_b = tetrahedron.vertices[1] - tetrahedron.vertices[0];
    const Vec3 edge_c = tetrahedron.vertices[2] - tetrahedron.vertices[0];
    const Vec3 edge_d = tetrahedron.vertices[3] - tetrahedron.vertices[0];
    for (std::size_t i = 0; i < 6; ++i) {
        const double u = gauss6_node[i];
        for (std::size_t j = 0; j < 6; ++j) {
            const double v = gauss6_node[j];
            for (std::size_t k = 0; k < 6; ++k) {
                const double w = gauss6_node[k];
                const Vec3 offset = tetrahedron.vertices[0] + edge_b * u +
                    edge_c * ((1.0 - u) * v) +
                    edge_d * ((1.0 - u) * (1.0 - v) * w);
                // The Jacobian of the collapsed cube is 6 V (1-u)^2 (1-v);
                // dividing by V gives the volume average directly.
                const double weight = 6.0 * gauss6_weight[i] *
                    gauss6_weight[j] * gauss6_weight[k] * (1.0 - u) *
                    (1.0 - u) * (1.0 - v);
                accumulate_scaled(result, source_field(d + offset),
                                  weight);
            }
        }
    }
    return result;
}

bool far_separated(const Vec3& target_minus_source_representative,
                   const double source_circumradius,
                   const double target_circumradius) noexcept
{
    return norm(target_minus_source_representative) >
        polyhedron_far_separation_factor *
        (source_circumradius + target_circumradius);
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
    if (far_separated(target_minus_source_representative,
                      source.surface.circumradius,
                      target.surface.circumradius)) {
        return average_source_tensor_over_target(
            target_minus_source_representative, source, target);
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
    if (!coincident_same_geometry &&
        far_separated(target_minus_source_representative, source.circumradius,
                      target.circumradius)) {
        // Widely separated pairs average the exact source field over the
        // target instead of cancelling large face integrals.
        return average_source_tensor_over_target(
            target_minus_source_representative,
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
