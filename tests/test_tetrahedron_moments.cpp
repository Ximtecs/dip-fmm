// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "geometry/primitives/tetrahedron_moments.hpp"

using namespace cdfmm;

namespace {

Tetrahedron unit_simplex()
{
    return {{{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
              {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}}};
}

// An absolute bound avoids relative comparisons of vanishing odd moments.
double moment_bound(const MultiIndex& alpha, const Vec3& d,
                    const Tetrahedron& tetrahedron)
{
    long double bound = 1.0L;
    for (int axis = 0; axis < 3; ++axis) {
        long double coordinate = 0.0L;
        for (const Vec3& vertex : tetrahedron.vertices) {
            coordinate = std::max(coordinate,
                std::abs(static_cast<long double>(d[axis]) + vertex[axis]));
        }
        for (int power = 1; power <= alpha[axis]; ++power) {
            bound *= coordinate / power;
        }
    }
    return static_cast<double>(bound);
}

// Independent volume quadrature through the Duffy map to the unit cube.
// A degree-p Cartesian polynomial becomes degree <=p+2 in u after including
// the Jacobian, so 12 Gauss points integrate every degree <=20 exactly.
std::vector<std::pair<long double, long double>> gauss_rule(int count)
{
    std::vector<std::pair<long double, long double>> rule;
    for (int root = 0; root < count; ++root) {
        long double x = std::cos(std::numbers::pi_v<long double> *
            (root + 0.75L) / (count + 0.5L));
        long double derivative = 0.0L;
        for (int iteration = 0; iteration < 30; ++iteration) {
            long double previous = 1.0L;
            long double value = x;
            for (int degree = 2; degree <= count; ++degree) {
                const long double next = ((2 * degree - 1) * x * value -
                    (degree - 1) * previous) / degree;
                previous = value;
                value = next;
            }
            derivative = count * (x * value - previous) / (x * x - 1.0L);
            const long double correction = value / derivative;
            x -= correction;
            if (std::abs(correction) < 8 *
                std::numeric_limits<long double>::epsilon()) {
                break;
            }
        }
        rule.emplace_back((x + 1.0L) / 2.0L,
                          1.0L / ((1.0L - x * x) * derivative * derivative));
    }
    return rule;
}

std::vector<long double> quadrature_moments(
    const MultiIndexSet& basis, const Vec3& d, const Tetrahedron& tetrahedron)
{
    const auto rule = gauss_rule(12);
    std::vector<long double> averages(static_cast<std::size_t>(basis.size()));
    std::array<std::vector<long double>, 3> powers;
    for (auto& axis : powers) {
        axis.resize(static_cast<std::size_t>(basis.order() + 1));
    }
    for (const auto& [u, wu] : rule) {
        for (const auto& [v, wv] : rule) {
            for (const auto& [w, ww] : rule) {
                const long double lambda[4]{
                    (1 - u) * (1 - v) * (1 - w), u,
                    (1 - u) * v, (1 - u) * (1 - v) * w};
                const long double weight =
                    6 * wu * wv * ww * (1 - u) * (1 - u) * (1 - v);
                for (int axis = 0; axis < 3; ++axis) {
                    long double coordinate = d[axis];
                    for (int vertex = 0; vertex < 4; ++vertex) {
                        coordinate += lambda[vertex] *
                            tetrahedron.vertices[static_cast<std::size_t>(vertex)][axis];
                    }
                    powers[axis][0] = 1;
                    for (int degree = 1; degree <= basis.order(); ++degree) {
                        powers[axis][static_cast<std::size_t>(degree)] =
                            powers[axis][static_cast<std::size_t>(degree - 1)] *
                            coordinate / degree;
                    }
                }
                for (int index = 0; index < basis.size(); ++index) {
                    const auto alpha = basis[index];
                    averages[static_cast<std::size_t>(index)] += weight *
                        powers[0][static_cast<std::size_t>(alpha.ax)] *
                        powers[1][static_cast<std::size_t>(alpha.ay)] *
                        powers[2][static_cast<std::size_t>(alpha.az)];
                }
            }
        }
    }
    return averages;
}

} // namespace

TEST_CASE("Tetrahedron moment table matches scalar integration", "[tetrahedron][moments]")
{
    const std::array<Tetrahedron, 3> geometries{
        unit_simplex(),
        Tetrahedron{{{{-0.25, -0.25, -0.25}, {0.75, -0.25, -0.25},
                       {-0.25, 0.75, -0.25}, {-0.25, -0.25, 0.75}}}},
        Tetrahedron{{{{-0.3, 0.2, -0.00001}, {0.8, -0.4, 0.00001},
                       {0.1, 0.9, -0.00001}, {-0.2, -0.1, 0.00002}}}}};
    const std::array<Vec3, 3> offsets{{{0.0, 0.0, 0.0},
                                      {0.125, -0.375, 0.0625},
                                      {-0.75, 0.5, -0.2}}};
    const MultiIndexSet basis(10);
    for (const Tetrahedron& tetrahedron : geometries) {
        for (const Vec3& d : offsets) {
            const auto averages = detail::tetrahedron_averaged_monomials(basis, d, tetrahedron);
            for (int index = 0; index < basis.size(); ++index) {
                const MultiIndex alpha = basis[index];
                REQUIRE(detail::tetrahedron_monomial_index(alpha) ==
                        static_cast<std::size_t>(index));
                // Cover every low-order monomial and mixed/axial high orders,
                // without retaining the legacy combinatorial cost in CI.
                if (alpha.degree() > 5 && alpha.ax != alpha.ay &&
                    alpha.ax != 0 && alpha.ay != 0 && alpha.az != 0) {
                    continue;
                }
                const double scalar = tetrahedron_averaged_monomial(alpha, d, tetrahedron);
                const double bound = moment_bound(alpha, d, tetrahedron);
                CAPTURE(alpha.ax, alpha.ay, alpha.az, scalar, bound);
                REQUIRE(std::abs(averages[static_cast<std::size_t>(index)] - scalar) <=
                        3e-13 * bound + 1e-300);
            }
        }
    }
}

TEST_CASE("Tetrahedron moment table integrates degree twenty exactly", "[tetrahedron][moments]")
{
    const MultiIndexSet basis(20);
    const auto simplex = unit_simplex();
    const auto analytic = detail::tetrahedron_averaged_monomials(basis, {}, simplex);
    for (int index = 0; index < basis.size(); ++index) {
        const double exact = 6.0 / MultiIndexSet::factorial(basis[index].degree() + 3);
        REQUIRE(std::abs(analytic[static_cast<std::size_t>(index)] - exact) <=
                3e-16 * exact);
    }

    // The representative need not be the centroid. Exactly representable
    // translations cancel before the recurrence, even at a large offset.
    auto translated = simplex;
    const Vec3 shift{1048576.0, -1048576.0, 524288.0};
    for (auto& vertex : translated.vertices) {
        vertex = vertex + shift;
    }
    REQUIRE(detail::tetrahedron_averaged_monomials(
        basis, shift * -1.0, translated) == analytic);

    const Tetrahedron mixed{{{{-0.8, 0.3, -0.00002}, {0.7, -0.5, 0.00001},
                              {0.2, 0.9, -0.00001}, {-0.1, -0.2, 0.00003}}}};
    for (const Vec3 d : {Vec3{}, Vec3{0.8, -0.3, 0.2}}) {
        const auto averages = detail::tetrahedron_averaged_monomials(basis, d, mixed);
        const auto quadrature = quadrature_moments(basis, d, mixed);
        for (int index = 0; index < basis.size(); ++index) {
            const auto alpha = basis[index];
            const double bound = moment_bound(alpha, d, mixed);
            CAPTURE(alpha.ax, alpha.ay, alpha.az, bound);
            REQUIRE(std::abs(averages[static_cast<std::size_t>(index)] -
                             quadrature[static_cast<std::size_t>(index)]) <=
                    5e-15L * bound + 1e-300L);
        }
    }
}

TEST_CASE("Tetrahedron moment table preserves geometry validation", "[tetrahedron][moments]")
{
    REQUIRE_THROWS_AS(detail::tetrahedron_averaged_monomials(
        MultiIndexSet(0), {}, Tetrahedron{}), std::invalid_argument);
    auto tetrahedron = unit_simplex();
    tetrahedron.vertices[0].x = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS_AS(detail::tetrahedron_averaged_monomials(
        MultiIndexSet(0), {}, tetrahedron), std::invalid_argument);
}
