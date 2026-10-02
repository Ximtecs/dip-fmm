// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "cdfmm/math/laplace_derivatives.hpp"
#include "cdfmm/math/taylor_jet.hpp"

using namespace cdfmm;

TEST_CASE("Laplace derivative recurrence matches Taylor-jet composition", "[laplace][recurrence]")
{
    // Keep the former implementation as an independent numerical reference.
    // One displacement reaches order 30, while symmetry and sign cases reach
    // order 20: the jet reference is much more costly than the recurrence.
    struct Sample {
        int order;
        Vec3 r;
    };
    const std::array<Sample, 4> samples{{
        {30, {3.0, -2.0, 1.0}}, {20, {2.0, 1.0, 0.0}},
        {20, {0.0, 0.0, 2.0}}, {20, {-3.0, 3.0, -3.0}}}};
    for (const auto& sample : samples) {
        const MultiIndexSet basis(sample.order);
        const Vec3& r = sample.r;
        const TaylorJet x = TaylorJet::coordinate(basis, 0, r.x);
        const TaylorJet y = TaylorJet::coordinate(basis, 1, r.y);
        const TaylorJet z = TaylorJet::coordinate(basis, 2, r.z);
        const TaylorJet G = x.mul(x).add(y.mul(y)).add(z.mul(z)).invsqrt().mul(
            TaylorJet::constant(basis, 1.0 / (4.0 * std::numbers::pi)));
        const auto recurrence = laplace_derivatives_raw(basis, r);
        std::vector<double> reference(static_cast<std::size_t>(basis.size()));
        std::vector<double> scales(static_cast<std::size_t>(sample.order + 1));
        for (int index = 0; index < basis.size(); ++index) {
            const MultiIndex alpha = basis[index];
            const double value = G.at(alpha) * MultiIndexSet::multi_factorial(alpha);
            reference[static_cast<std::size_t>(index)] = value;
            double& scale = scales[static_cast<std::size_t>(alpha.degree())];
            scale = std::max(scale, std::abs(value));
        }
        for (int index = 0; index < basis.size(); ++index) {
            const MultiIndex alpha = basis[index];
            const double difference = std::abs(recurrence[static_cast<std::size_t>(index)] -
                                               reference[static_cast<std::size_t>(index)]);
            // Scale by the whole degree, since symmetry can annihilate an
            // individual derivative and make its relative error meaningless.
            CAPTURE(sample.order, r.x, r.y, r.z, alpha.ax, alpha.ay, alpha.az);
            REQUIRE(difference <=
                    1e-12 * scales[static_cast<std::size_t>(alpha.degree())]);
        }
    }
}

TEST_CASE("Multi-index closed form matches exhaustive storage order", "[multi_index]")
{
    for (const int order : {0, 1, 2, 6, 17, 34}) {
        const MultiIndexSet basis(order);
        for (int index = 0; index < basis.size(); ++index) {
            REQUIRE(basis.index(basis[index]) == index);
        }
        for (const MultiIndex outside : {
            MultiIndex{order + 1, 0, 0}, MultiIndex{0, 0, order + 1},
            MultiIndex{-1, 1, 0}, MultiIndex{0, -1, 1}, MultiIndex{1, 0, -1},
            MultiIndex{std::numeric_limits<int>::max(), 1, 1},
            MultiIndex{std::numeric_limits<int>::max(),
                       std::numeric_limits<int>::max(),
                       std::numeric_limits<int>::max()}}) {
            REQUIRE_THROWS_AS(basis.index(outside), std::out_of_range);
        }
    }
    REQUIRE_THROWS_AS(MultiIndexSet(-1).index({0, 0, 0}), std::out_of_range);
}

TEST_CASE("Laplace derivative recurrence preserves exceptional inputs", "[laplace]")
{
    for (const int order : {-1, 0}) {
        REQUIRE_THROWS_AS(laplace_derivatives_raw(MultiIndexSet(order), {1.0, 0.0, 0.0}),
                          std::out_of_range);
    }
    REQUIRE_THROWS_AS(laplace_derivatives_raw(MultiIndexSet(2), {}), std::invalid_argument);
    // Squaring a subnormal displacement underflows in the former jet path.
    REQUIRE_THROWS_AS(laplace_derivatives_raw(MultiIndexSet(2),
        {std::numeric_limits<double>::denorm_min(), 0.0, 0.0}), std::invalid_argument);
}

TEST_CASE("Laplace derivatives match analytical gradient and Hessian", "[laplace]")
{
    const MultiIndexSet basis(2);
    const Vec3 r{1.3, -0.7, 0.4};
    const double radius = std::sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
    const double c = 1.0 / (4.0 * std::numbers::pi);
    const auto derivatives = laplace_derivatives_raw(basis, r);
    const MultiIndex axes[3]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    REQUIRE(derivatives[0] == Catch::Approx(c / radius).epsilon(1e-14));
    for (int first = 0; first < 3; ++first) {
        const double gradient = -c * r[first] / std::pow(radius, 3);
        REQUIRE(derivatives[static_cast<std::size_t>(basis.index(axes[first]))] ==
                Catch::Approx(gradient).epsilon(1e-14));
        for (int second = first; second < 3; ++second) {
            const MultiIndex alpha = add(axes[first], axes[second]);
            const double hessian = c * (3.0 * r[first] * r[second] /
                std::pow(radius, 5) - (first == second ? 1.0 / std::pow(radius, 3) : 0.0));
            REQUIRE(derivatives[static_cast<std::size_t>(basis.index(alpha))] ==
                    Catch::Approx(hessian).epsilon(1e-14));
        }
    }
}

TEST_CASE("Laplace derivatives on axis at r=(1,0,0)") {
  const MultiIndexSet basis(2);
  const auto d = laplace_derivatives_raw(basis, {1.0, 0.0, 0.0});
  const double c = 1.0 / (4.0 * std::numbers::pi);

  REQUIRE(d[basis.index({0, 0, 0})] == Catch::Approx(c).epsilon(1e-12));
  REQUIRE(d[basis.index({1, 0, 0})] == Catch::Approx(-c).epsilon(1e-12));
  REQUIRE(d[basis.index({0, 1, 0})] == Catch::Approx(0.0).margin(1e-14));
  REQUIRE(d[basis.index({0, 0, 1})] == Catch::Approx(0.0).margin(1e-14));

  REQUIRE(d[basis.index({2, 0, 0})] == Catch::Approx(2.0 * c).epsilon(1e-12));
  REQUIRE(d[basis.index({0, 2, 0})] == Catch::Approx(-c).epsilon(1e-12));
  REQUIRE(d[basis.index({0, 0, 2})] == Catch::Approx(-c).epsilon(1e-12));
  REQUIRE(d[basis.index({1, 1, 0})] == Catch::Approx(0.0).margin(1e-14));
}

TEST_CASE("Laplace derivatives satisfy Laplace equation away from singularity") {
  const MultiIndexSet basis(2);

  const std::array<Vec3, 2> points{
      Vec3{1.3, -0.7, 0.4},
      Vec3{-0.8, 1.1, 0.9},
  };

  for (const Vec3 &r : points) {
    const auto d = laplace_derivatives_raw(basis, r);
    const double laplacian =
        d[basis.index({2, 0, 0})] + d[basis.index({0, 2, 0})] +
        d[basis.index({0, 0, 2})];
    REQUIRE(laplacian == Catch::Approx(0.0).margin(1e-11));
  }
}
