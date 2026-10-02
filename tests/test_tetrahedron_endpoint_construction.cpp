// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numbers>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "cdfmm/operators/l2p.hpp"
#include "cdfmm/operators/p2m.hpp"

namespace cdfmm {
namespace {

using EntryKey = std::pair<int, int>;
using EntryMap = std::map<EntryKey, double>;

Tetrahedron kuhn_tetrahedron()
{
    // The Kuhn simplex is the chain 000 -> 100 -> 110 -> 111,
    // represented relative to its centroid (3/4, 1/2, 1/4).
    return {{{{-0.75, -0.5, -0.25}, {0.25, -0.5, -0.25},
              {0.25, 0.5, -0.25}, {0.25, 0.5, 0.75}}}};
}

Tetrahedron mixed_tetrahedron()
{
    return {{{{-0.3, 0.2, 0.1}, {0.8, -0.4, 0.3},
              {0.1, 0.9, -0.2}, {-0.2, -0.1, 0.7}}}};
}

EntryMap entries_by_key(const StaticCoefficientOperator& op)
{
    EntryMap result;
    for (const StaticOperatorEntry& entry : op.entries) {
        result[{entry.output, entry.input}] = entry.value;
    }
    return result;
}

double monomial_bound(const MultiIndex& power, const Vec3& displacement,
                      const Tetrahedron& tetrahedron);

// The scale is the sum of absolute contributions from the solid-harmonic
// polynomial after replacing each monomial by the independent scalar integral.
double term_scale(std::span<const SolidHarmonicTerm> terms,
                  const Vec3& displacement, const Tetrahedron& tetrahedron,
                  const bool derivative, const int component)
{
    double scale = 0.0;
    for (const SolidHarmonicTerm& term : terms) {
        MultiIndex power = term.power;
        double factor = term.coefficient;
        if (derivative) {
            const int exponent = power[component];
            if (exponent == 0) {
                continue;
            }
            factor *= exponent;
            if (component == 0) {
                --power.ax;
            } else if (component == 1) {
                --power.ay;
            } else {
                --power.az;
            }
            factor *= MultiIndexSet::multi_factorial(power);
        } else {
            factor *= MultiIndexSet::multi_factorial(power);
        }
        scale += std::abs(factor) * monomial_bound(
            power, displacement, tetrahedron);
    }
    return scale;
}

double tolerance(const double scale)
{
    return 5.0e-14 * scale + 2.0e-300;
}

double monomial_bound(const MultiIndex& power, const Vec3& displacement,
                      const Tetrahedron& tetrahedron)
{
    long double bound = 1.0L;
    for (int axis = 0; axis < 3; ++axis) {
        long double coordinate = 0.0L;
        for (const Vec3& vertex : tetrahedron.vertices) {
            coordinate = std::max(coordinate,
                std::abs(static_cast<long double>(displacement[axis]) +
                         vertex[axis]));
        }
        for (int exponent = 1; exponent <= power[axis]; ++exponent) {
            bound *= coordinate / exponent;
        }
    }
    return static_cast<double>(bound);
}

void compare_cartesian(const int order, const Vec3& centre,
                       const Vec3& position, const Tetrahedron& tetrahedron)
{
    const MultiIndexSet basis(order);
    const std::array<Vec3, 1> positions{position};
    const std::array<Tetrahedron, 1> geometries{tetrahedron};
    const Vec3 displacement = position - centre;
    const auto p2m = build_static_tetrahedron_p2m_operator(
        basis, centre, positions, geometries);
    const EntryMap actual_entries = entries_by_key(p2m);
    EntryMap expected_entries;
    for (int alpha_index = 0; alpha_index < basis.size(); ++alpha_index) {
        const MultiIndex alpha = basis[alpha_index];
        const int exponents[3]{alpha.ax, alpha.ay, alpha.az};
        const MultiIndex shifted[3]{{alpha.ax - 1, alpha.ay, alpha.az},
                                    {alpha.ax, alpha.ay - 1, alpha.az},
                                    {alpha.ax, alpha.ay, alpha.az - 1}};
        const double sign = alpha.degree() % 2 == 0 ? 1.0 : -1.0;
        for (int component = 0; component < 3; ++component) {
            if (exponents[component] == 0) {
                continue;
            }
            const double value = sign * tetrahedron_averaged_monomial(
                shifted[component], displacement, tetrahedron);
            if (value != 0.0) {
                expected_entries[{alpha_index, component}] = value;
            }
        }
    }
    std::size_t pattern_differences = 0;
    for (const auto& [key, expected] : expected_entries) {
        pattern_differences += actual_entries.contains(key) ? 0U : 1U;
    }
    for (const auto& [key, actual] : actual_entries) {
        pattern_differences += expected_entries.contains(key) ? 0U : 1U;
    }
    CAPTURE(order, actual_entries.size(), expected_entries.size(),
            pattern_differences);
    EntryMap all_keys = expected_entries;
    all_keys.insert(actual_entries.begin(), actual_entries.end());
    for (const auto& [key, unused] : all_keys) {
        (void)unused;
        const double expected = expected_entries.contains(key)
            ? expected_entries.at(key) : 0.0;
        const double actual = actual_entries.contains(key)
            ? actual_entries.at(key) : 0.0;
        CAPTURE(order, key.first, key.second, expected);
        const MultiIndex alpha = basis[key.first];
        MultiIndex derivative = alpha;
        if (key.second == 0) {
            --derivative.ax;
        } else if (key.second == 1) {
            --derivative.ay;
        } else {
            --derivative.az;
        }
        REQUIRE(std::abs(actual - expected) <=
                tolerance(monomial_bound(
                    derivative, displacement, tetrahedron)));
    }

    const StaticL2PEvaluator l2p = build_static_tetrahedron_l2p_evaluator(
        basis, centre, position, tetrahedron);
    for (int beta_index = 0; beta_index < basis.size(); ++beta_index) {
        const MultiIndex beta = basis[beta_index];
        const double expected = tetrahedron_averaged_monomial(
            beta, displacement, tetrahedron);
        CAPTURE(order, beta_index, beta.ax, beta.ay, beta.az, expected);
        REQUIRE(std::abs(l2p.potential[static_cast<std::size_t>(beta_index)] -
                         expected) <= tolerance(monomial_bound(
                             beta, displacement, tetrahedron)));
        const int powers[3]{beta.ax, beta.ay, beta.az};
        const MultiIndex derivative[3]{{beta.ax - 1, beta.ay, beta.az},
                                       {beta.ax, beta.ay - 1, beta.az},
                                       {beta.ax, beta.ay, beta.az - 1}};
        for (int component = 0; component < 3; ++component) {
            const double field_expected = powers[component] == 0 ? 0.0 :
                -tetrahedron_averaged_monomial(
                    derivative[component], displacement, tetrahedron);
            REQUIRE(std::abs(l2p.field[static_cast<std::size_t>(component)]
                                        [static_cast<std::size_t>(beta_index)] -
                             field_expected) <= tolerance(monomial_bound(
                                 derivative[component], displacement,
                                 tetrahedron)));
        }
    }
}

void compare_spherical(const int order, const Vec3& centre,
                       const Vec3& position, const Tetrahedron& tetrahedron)
{
    const SphericalHarmonicBasis basis(order);
    const std::array<Vec3, 1> positions{position};
    const std::array<Tetrahedron, 1> geometries{tetrahedron};
    const Vec3 displacement = position - centre;
    const double green = 1.0 / (4.0 * std::numbers::pi);
    const auto p2m = build_static_tetrahedron_p2m_operator(
        basis, centre, positions, geometries);
    const EntryMap actual_entries = entries_by_key(p2m);
    EntryMap expected_entries;
    for (int mode = 0; mode < basis.size(); ++mode) {
        for (const SolidHarmonicTerm& term : basis.polynomial(mode)) {
            const int powers[3]{term.power.ax, term.power.ay, term.power.az};
            for (int component = 0; component < 3; ++component) {
                if (powers[component] == 0) {
                    continue;
                }
                MultiIndex derivative = term.power;
                if (component == 0) {
                    --derivative.ax;
                } else if (component == 1) {
                    --derivative.ay;
                } else {
                    --derivative.az;
                }
                expected_entries[{mode, component}] += green *
                    term.coefficient * powers[component] *
                    MultiIndexSet::multi_factorial(derivative) *
                    tetrahedron_averaged_monomial(
                        derivative, displacement, tetrahedron);
            }
        }
    }
    for (auto iter = expected_entries.begin(); iter != expected_entries.end();) {
        if (iter->second == 0.0) {
            iter = expected_entries.erase(iter);
        } else {
            ++iter;
        }
    }
    std::size_t pattern_differences = 0;
    for (const auto& [key, expected] : expected_entries) {
        pattern_differences += actual_entries.contains(key) ? 0U : 1U;
    }
    for (const auto& [key, actual] : actual_entries) {
        pattern_differences += expected_entries.contains(key) ? 0U : 1U;
    }
    CAPTURE(order, actual_entries.size(), expected_entries.size(),
            pattern_differences);
    EntryMap all_keys = expected_entries;
    all_keys.insert(actual_entries.begin(), actual_entries.end());
    for (const auto& [key, unused] : all_keys) {
        (void)unused;
        const double expected = expected_entries.contains(key)
            ? expected_entries.at(key) : 0.0;
        const double actual = actual_entries.contains(key)
            ? actual_entries.at(key) : 0.0;
        CAPTURE(order, key.first, key.second, expected);
        const int mode = key.first;
        const int component = key.second;
        const double scale = green * term_scale(
            basis.polynomial(mode), displacement, tetrahedron, true, component);
        REQUIRE(std::abs(actual - expected) <= tolerance(scale));
    }

    const StaticL2PEvaluator l2p = build_static_tetrahedron_l2p_evaluator(
        basis, centre, position, tetrahedron);
    for (int mode = 0; mode < basis.size(); ++mode) {
        double potential_expected = 0.0;
        for (const SolidHarmonicTerm& term : basis.polynomial(mode)) {
            potential_expected += term.coefficient *
                MultiIndexSet::multi_factorial(term.power) *
                tetrahedron_averaged_monomial(
                    term.power, displacement, tetrahedron);
        }
        const double potential_scale = term_scale(
            basis.polynomial(mode), displacement, tetrahedron, false, 0);
        CAPTURE(order, mode, potential_expected);
        REQUIRE(std::abs(l2p.potential[static_cast<std::size_t>(mode)] -
                         potential_expected) <= tolerance(potential_scale));
        for (int component = 0; component < 3; ++component) {
            double field_expected = 0.0;
            for (const SolidHarmonicTerm& term : basis.polynomial(mode)) {
                const int exponent = term.power[component];
                if (exponent == 0) {
                    continue;
                }
                MultiIndex derivative = term.power;
                if (component == 0) {
                    --derivative.ax;
                } else if (component == 1) {
                    --derivative.ay;
                } else {
                    --derivative.az;
                }
                field_expected -= term.coefficient * exponent *
                    MultiIndexSet::multi_factorial(derivative) *
                    tetrahedron_averaged_monomial(
                        derivative, displacement, tetrahedron);
            }
            const double field_scale = term_scale(
                basis.polynomial(mode), displacement, tetrahedron, true,
                component);
            REQUIRE(std::abs(l2p.field[static_cast<std::size_t>(component)]
                                        [static_cast<std::size_t>(mode)] -
                             field_expected) <= tolerance(field_scale));
        }
    }
}

} // namespace

TEST_CASE("batched tetrahedron endpoint operators match scalar construction",
          "[tetrahedron][operators]")
{
    const std::array<int, 5> orders{0, 1, 4, 6, 10};
    const std::array<std::pair<Vec3, Tetrahedron>, 2> cases{{
        {Vec3{0.17, -0.23, 0.31}, kuhn_tetrahedron()},
        {Vec3{-0.61, 0.37, 0.19}, mixed_tetrahedron()}}};
    for (const int order : orders) {
        for (const auto& [position, tetrahedron] : cases) {
            compare_cartesian(order, Vec3{0.1, -0.2, 0.3}, position, tetrahedron);
            compare_spherical(order, Vec3{0.1, -0.2, 0.3}, position, tetrahedron);
        }
    }
}

TEST_CASE("tetrahedron P2M accepts common and per-source geometry records",
          "[tetrahedron][operators]")
{
    const MultiIndexSet basis(4);
    const SphericalHarmonicBasis spherical_basis(4);
    const Vec3 centre{0.1, -0.2, 0.3};
    const std::array<Vec3, 2> positions{{{0.17, -0.23, 0.31},
                                          {-0.61, 0.37, 0.19}}};
    const Tetrahedron first = kuhn_tetrahedron();
    const Tetrahedron second = mixed_tetrahedron();
    const std::array<Tetrahedron, 1> common{{first}};
    const std::array<Tetrahedron, 2> per_source{{first, second}};

    const auto common_cartesian = build_static_tetrahedron_p2m_operator(
        basis, centre, positions, common);
    REQUIRE(common_cartesian.input_size == 6);
    REQUIRE(common_cartesian.output_size == basis.size());
    const auto separate_cartesian = build_static_tetrahedron_p2m_operator(
        basis, centre, positions, per_source);
    const EntryMap common_entries = entries_by_key(common_cartesian);
    const EntryMap separate_entries = entries_by_key(separate_cartesian);
    for (const auto& [key, value] : common_entries) {
        if (key.second < 3) {
            REQUIRE(separate_entries.at(key) == value);
        }
    }
    REQUIRE(std::any_of(separate_entries.begin(), separate_entries.end(),
        [&](const auto& entry) {
            return entry.first.second >= 3 &&
                entry.second != common_entries.at(entry.first);
        }));

    const auto common_spherical = build_static_tetrahedron_p2m_operator(
        spherical_basis, centre, positions, common);
    const auto separate_spherical = build_static_tetrahedron_p2m_operator(
        spherical_basis, centre, positions, per_source);
    const EntryMap common_spherical_entries = entries_by_key(common_spherical);
    const EntryMap separate_spherical_entries = entries_by_key(separate_spherical);
    for (const auto& [key, value] : common_spherical_entries) {
        if (key.second < 3) {
            REQUIRE(separate_spherical_entries.at(key) == value);
        }
    }
}

TEST_CASE("tetrahedron endpoint builders preserve invalid-input checks",
          "[tetrahedron][operators]")
{
    const Tetrahedron degenerate{};
    const std::array<Vec3, 1> positions{{{0.0, 0.0, 0.0}}};
    const std::array<Tetrahedron, 1> geometries{{degenerate}};
    REQUIRE_THROWS_AS(build_static_tetrahedron_p2m_operator(
        MultiIndexSet(1), Vec3{}, positions, geometries), std::invalid_argument);
    REQUIRE_THROWS_AS(build_static_tetrahedron_p2m_operator(
        SphericalHarmonicBasis(1), Vec3{}, positions, geometries),
        std::invalid_argument);
    REQUIRE_THROWS_AS(build_static_tetrahedron_l2p_evaluator(
        MultiIndexSet(1), Vec3{}, Vec3{}, degenerate), std::invalid_argument);
    REQUIRE_THROWS_AS(build_static_tetrahedron_l2p_evaluator(
        SphericalHarmonicBasis(1), Vec3{}, Vec3{}, degenerate),
        std::invalid_argument);
    REQUIRE_THROWS_AS(build_static_tetrahedron_p2m_operator(
        MultiIndexSet(1), Vec3{}, positions,
        std::span<const Tetrahedron>{}), std::invalid_argument);
}

} // namespace cdfmm
