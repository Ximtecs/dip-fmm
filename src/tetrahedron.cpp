// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/tetrahedron.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
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
    const Vec3&, const Tetrahedron&, const Tetrahedron&)
{
    throw std::domain_error(
        "exact tetrahedron-to-tetrahedron P2P is not implemented analytically");
}

} // namespace cdfmm
