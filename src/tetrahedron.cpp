// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/tetrahedron.hpp"

#include "tetrahedron_detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cdfmm {
namespace {

constexpr long double four_pi = 4.0L * std::numbers::pi_v<long double>;

struct Matrix3 {
    double value[3][3]{};
};

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

// This is the principal atan quotient used by TileTriangle's P_Nzz/Q_Nzz
// primitives.  It is intentionally not atan2: those primitives use the sign
// of the normal coordinate as part of the one-sided limiting behaviour.
long double safe_atan_ratio(const long double numerator,
                            const long double denominator)
{
    if (denominator != 0.0) {
        return std::atan(numerator / denominator);
    }
    if (numerator == 0.0) {
        return 0.0;
    }
    return std::copysign(0.5 * std::numbers::pi, numerator);
}

// MagTense evaluates atanh ratios which are mathematically in (-1,1), but a
// rounded denominator can put a ratio outside that interval.  Clamping to a
// one-sided endpoint keeps the analytical cancellation finite in that case.
long double safe_atanh_ratio(const long double numerator,
                             const long double denominator)
{
    if (denominator == 0.0) {
        if (numerator == 0.0) {
            return 0.0;
        }
        return std::copysign(std::numeric_limits<long double>::infinity(),
                             numerator);
    }

    const long double ratio = numerator / denominator;
    if (!std::isfinite(ratio)) {
        return std::copysign(std::numeric_limits<long double>::infinity(),
                             ratio);
    }

    if (ratio >= 1.0) {
        // These ratios are bounded by one analytically.  Even a somewhat
        // larger excursion can occur after subtracting nearly equal squared
        // distances in the far field; treating it as the endpoint preserves
        // the finite cancellation between the two edge primitives.
        return std::atanh(1.0L - 32.0L *
                          std::numeric_limits<long double>::epsilon());
    }
    if (ratio <= -1.0) {
        return std::atanh(-1.0L + 32.0L *
                          std::numeric_limits<long double>::epsilon());
    }
    return std::atanh(ratio);
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

// The following three functions are direct double-precision adaptations of
// TileTriangle.f90's Nxz, Nyz and Nzz primitives.  Their arguments describe a
// right-triangle representation in the local face frame: the base endpoints
// are (x=0,l) and the apex has coordinates (0,h), with h>0.  The expressions
// are the analytical edge primitives, not numerical quadrature.
double triangle_nxz(const Vec3& r, const double l, const double h)
{
    const long double rx = r.x;
    const long double ry = r.y;
    const long double rz = r.z;
    const long double L = l;
    const long double H = h;
    const auto f = [&](const long double yp) {
        const long double root = std::hypot(
            std::hypot(L - rx - yp * L / H, ry - yp), rz);
        const long double diagonal = std::hypot(L, H);
        const long double numerator = L * L - L * rx + H * ry -
            H * yp * (1.0L + L * L / (H * H));
        return H / diagonal * safe_atanh_ratio(numerator, diagonal * root);
    };
    const auto g = [&](const long double yp) {
        const long double denominator =
            std::sqrt(rx * rx + (ry - yp) * (ry - yp) + rz * rz);
        return safe_atanh_ratio(r.y - yp, denominator);
    };
    return static_cast<double>(-(f(H) - f(0.0L) - (g(H) - g(0.0L))) /
                               four_pi);
}

double triangle_nyz(const Vec3& r, const double l, const double h)
{
    const long double rx = r.x;
    const long double ry = r.y;
    const long double rz = r.z;
    const long double L = l;
    const long double H = h;
    const auto k = [&](const long double xp) {
        const long double root = std::hypot(
            std::hypot(rx - xp, ry - H + xp * H / L), rz);
        const long double diagonal = std::hypot(L, H);
        const long double numerator = H * H - H * ry + L * rx -
            L * xp * (1.0L + H * H / (L * L));
        return l / diagonal * safe_atanh_ratio(numerator, diagonal * root);
    };
    const auto ell = [&](const long double xp) {
        const long double denominator =
            std::sqrt((rx - xp) * (rx - xp) + ry * ry + rz * rz);
        return safe_atanh_ratio(rx - xp, denominator);
    };
    return static_cast<double>(-(k(L) - k(0.0L) - (ell(L) - ell(0.0L))) /
                               four_pi);
}

double triangle_nzz(const Vec3& r, const double l, const double h)
{
    const long double rx = r.x;
    const long double ry = r.y;
    const long double rz = r.z;
    const long double L = l;
    const long double H = h;
    const auto p = [&](const long double xp) {
        const long double root = std::hypot(
            std::hypot(rx - xp, ry - H + xp * H / L), rz);
        const long double numerator = rx * (H - ry) -
            xp * (H * (1.0L - rx / L) - ry) -
            H * (rx * rx + rz * rz) / L;
        return safe_atan_ratio(numerator, rz * root);
    };
    const auto q = [&](const long double xp) {
        return -safe_atan_ratio((rx - xp) * ry,
                                rz * std::hypot(std::hypot(rx - xp, ry), rz));
    };
    return static_cast<double>(-(p(L) - p(0.0L) - (q(L) - q(0.0L))) /
                               four_pi);
}

struct FaceFrame {
    Vec3 e1{};
    Vec3 e2{};
    Vec3 normal{};
    Vec3 origin{};
    double x_first{0.0};
    double x_third{0.0};
    double height{0.0};
};

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

Matrix3 face_tensor(const std::array<Vec3, 4>& face,
                    const Vec3& target_relative_position)
{
    const FaceFrame frame = make_face_frame(face);
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
    // face, exactly as in MagTense's getN_Triangle.
    const double local_column[3]{
        triangle_nxz(r, frame.x_first, frame.height) -
            triangle_nxz(r, frame.x_third, frame.height),
        triangle_nyz(r, frame.x_first, frame.height) -
            triangle_nyz(r, frame.x_third, frame.height),
            triangle_nzz(r, frame.x_first, frame.height) -
            triangle_nzz(r, frame.x_third, frame.height)};
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

Matrix3 tetrahedron_magnetisation_tensor(
    const Vec3& target_relative_position, const Tetrahedron& source)
{
    const double volume = tetrahedron_volume(source);
    double geometry_scale = 0.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            geometry_scale = std::max(
                geometry_scale,
                norm(source.vertices[static_cast<std::size_t>(i)] -
                     source.vertices[static_cast<std::size_t>(j)]));
        }
    }
    const double singular_tolerance =
        256.0 * std::numeric_limits<double>::epsilon() * geometry_scale;
    for (int i = 0; i < 4; ++i) {
        if (norm(target_relative_position -
                 source.vertices[static_cast<std::size_t>(i)]) <=
            singular_tolerance) {
            throw std::domain_error(
                "tetrahedron-to-point tensor is singular at a vertex");
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            const Vec3 edge = source.vertices[static_cast<std::size_t>(j)] -
                source.vertices[static_cast<std::size_t>(i)];
            const double edge_length_squared = dot(edge, edge);
            const double parameter = dot(
                target_relative_position -
                    source.vertices[static_cast<std::size_t>(i)], edge) /
                edge_length_squared;
            if (parameter >= 0.0 && parameter <= 1.0) {
                const Vec3 closest = source.vertices[static_cast<std::size_t>(i)] +
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
        std::array<Vec3, 4> face{};
        int output = 0;
        for (int vertex = 0; vertex < 4; ++vertex) {
            if (vertex != missing_vertex) {
                face[static_cast<std::size_t>(output++)] =
                    source.vertices[static_cast<std::size_t>(vertex)];
            }
        }
        face[3] = source.vertices[static_cast<std::size_t>(missing_vertex)];
        const Matrix3 contribution = face_tensor(face, target_relative_position);
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
            result.value[row][column] /= volume;
        }
    }
    return result;
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

double triangle_triangle_integral_core(
    const std::array<Vec3, 3>& first,
    const std::array<Vec3, 3>& second)
{
    const Vec3 first_edge_1 = first[1] - first[0];
    const Vec3 first_edge_2 = first[2] - first[1];
    const Vec3 first_edge_3 = first[0] - first[2];
    const Vec3 second_edge_1 = second[1] - second[0];
    const Vec3 second_edge_2 = second[2] - second[1];
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

struct BarycentricPolynomial {
    std::array<int, 4> power{{0, 0, 0, 0}};
    double coefficient{0.0};
};

void expand_axis(const int power, const double offset,
                 const std::array<double, 4>& coordinates,
                 std::vector<BarycentricPolynomial>& terms)
{
    if (power < 0) {
        throw std::invalid_argument("monomial exponents must be non-negative");
    }
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

PairTensor point_tetrahedron_tensor(const Vec3& target_minus_source_representative,
                                    const Tetrahedron& target)
{
    return tetrahedron_point_tensor(
        scale(target_minus_source_representative, -1.0), target);
}

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
    const double source_volume = tetrahedron_volume(source);
    const double target_volume = tetrahedron_volume(target);

    // The face order is immaterial to the scalar Galerkin integral.  It is
    // made outward below using the vertex opposite each face, which keeps the
    // tensor sign independent of the caller's tetrahedron orientation.
    constexpr std::array<std::array<int, 3>, 4> face_vertices{{
        {{1, 2, 3}},
        {{0, 3, 2}},
        {{0, 1, 3}},
        {{0, 2, 1}},
    }};

    const auto outward_normal = [](const std::array<Vec3, 4>& vertices,
                                   const std::array<int, 3>& face,
                                   const int opposite) {
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
    };

    std::array<Vec3, 4> source_normals{};
    std::array<Vec3, 4> target_normals{};
    for (int face = 0; face < 4; ++face) {
        source_normals[static_cast<std::size_t>(face)] = outward_normal(
            source.vertices, face_vertices[static_cast<std::size_t>(face)],
            face);
        target_normals[static_cast<std::size_t>(face)] = outward_normal(
            target.vertices, face_vertices[static_cast<std::size_t>(face)],
            face);
    }

    std::array<Vec3, 4> target_vertices{};
    for (int vertex = 0; vertex < 4; ++vertex) {
        target_vertices[static_cast<std::size_t>(vertex)] =
            target_minus_source_representative + target.vertices[
                static_cast<std::size_t>(vertex)];
    }

    Matrix3 tensor;
    for (int target_face = 0; target_face < 4; ++target_face) {
        const auto& target_face_indices = face_vertices[
            static_cast<std::size_t>(target_face)];
        const std::array<Vec3, 3> target_triangle{{
            target_vertices[static_cast<std::size_t>(target_face_indices[0])],
            target_vertices[static_cast<std::size_t>(target_face_indices[1])],
            target_vertices[static_cast<std::size_t>(target_face_indices[2])],
        }};
        for (int source_face = 0; source_face < 4; ++source_face) {
            const auto& source_face_indices = face_vertices[
                static_cast<std::size_t>(source_face)];
            const std::array<Vec3, 3> source_triangle{{
                source.vertices[static_cast<std::size_t>(source_face_indices[0])],
                source.vertices[static_cast<std::size_t>(source_face_indices[1])],
                source.vertices[static_cast<std::size_t>(source_face_indices[2])],
            }};
            const double integral = detail::triangle_triangle_laplace_integral(
                target_triangle, source_triangle);
            if (!std::isfinite(integral)) {
                throw std::domain_error(
                    "tetrahedron face-pair integral is not finite (target face " +
                    std::to_string(target_face) + ", source face " +
                    std::to_string(source_face) + ")");
            }
            const Vec3& target_normal = target_normals[
                static_cast<std::size_t>(target_face)];
            const Vec3& source_normal = source_normals[
                static_cast<std::size_t>(source_face)];
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    tensor.value[row][column] += integral *
                        target_normal[row] * source_normal[column];
                }
            }
        }
    }

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
                    "tetrahedron-to-tetrahedron tensor is not finite");
            }
        }
    }
    // The Hessian of 1/R is symmetric.  Face-pair evaluation can leave only
    // last-bit antisymmetry, and PairTensor stores the six symmetric entries.
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

} // namespace cdfmm
