// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/spherical_harmonics.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <numbers>
#include <stdexcept>
#include <tuple>

#include "cdfmm/laplace_derivatives.hpp"

namespace cdfmm {

namespace {

long double factorial(const int n)
{
  long double result = 1.0L;
  for (int value = 2; value <= n; ++value) {
    result *= static_cast<long double>(value);
  }
  return result;
}

double odd_double_factorial(const int l)
{
  double result = 1.0;
  for (int value = 1; value <= 2 * l - 1; value += 2) {
    result *= static_cast<double>(value);
  }
  return result;
}

double integer_power(const double value, const int exponent)
{
  double result = 1.0;
  for (int count = 0; count < exponent; ++count) {
    result *= value;
  }
  return result;
}

using PowerKey = std::tuple<int, int, int>;

std::vector<SolidHarmonicTerm> make_polynomial(const int l, const int real_m)
{
  const int m = std::abs(real_m);
  std::map<PowerKey, std::complex<long double>> complex_terms;
  const long double normalisation =
      std::sqrt(factorial(l - m) / factorial(l + m));

  // Expand r^l P_l^m(z/r) exp(i*m*phi) as a homogeneous Cartesian
  // polynomial. The associated Legendre function includes the
  // Condon--Shortley phase.
  for (int k = 0; k <= (l - m) / 2; ++k) {
    const int z_power = l - 2 * k - m;
    const long double legendre =
        ((m + k) % 2 == 0 ? 1.0L : -1.0L) *
        factorial(2 * l - 2 * k) /
        (std::pow(2.0L, l) * factorial(k) * factorial(l - k) *
         factorial(z_power));

    for (int y_power_linear = 0; y_power_linear <= m; ++y_power_linear) {
      const int x_power_linear = m - y_power_linear;
      const long double binomial =
          factorial(m) /
          (factorial(y_power_linear) * factorial(x_power_linear));
      std::complex<long double> imaginary_power{1.0L, 0.0L};
      for (int count = 0; count < y_power_linear; ++count) {
        imaginary_power *= std::complex<long double>{0.0L, 1.0L};
      }

      // Multinomial expansion of (x^2+y^2+z^2)^k.
      for (int x_pairs = 0; x_pairs <= k; ++x_pairs) {
        for (int y_pairs = 0; y_pairs <= k - x_pairs; ++y_pairs) {
          const int z_pairs = k - x_pairs - y_pairs;
          const long double multinomial =
              factorial(k) /
              (factorial(x_pairs) * factorial(y_pairs) * factorial(z_pairs));
          const PowerKey key{
              x_power_linear + 2 * x_pairs,
              y_power_linear + 2 * y_pairs,
              z_power + 2 * z_pairs};
          complex_terms[key] += normalisation * legendre * binomial *
                                multinomial * imaginary_power;
        }
      }
    }
  }

  std::vector<SolidHarmonicTerm> result;
  result.reserve(complex_terms.size());
  const long double real_scale = m == 0
      ? 1.0L
      : std::sqrt(2.0L) * (m % 2 == 0 ? 1.0L : -1.0L);
  for (const auto& [key, value] : complex_terms) {
    const long double selected = real_m < 0 ? value.imag() : value.real();
    const double coefficient = static_cast<double>(real_scale * selected);
    if (coefficient == 0.0) {
      continue;
    }
    const auto [ax, ay, az] = key;
    result.push_back({{ax, ay, az}, coefficient});
  }
  return result;
}

} // namespace

//------------------------------------------------------------------------------
// Basis construction
//------------------------------------------------------------------------------

SphericalHarmonicBasis::SphericalHarmonicBasis(const int p) : p_(p)
{
  if (p < 0) {
    throw std::invalid_argument("spherical-harmonic order must be >= 0");
  }
  modes_.reserve(static_cast<std::size_t>((p + 1) * (p + 1)));
  polynomials_.reserve(modes_.capacity());
  for (int l = 0; l <= p; ++l) {
    for (int m = -l; m <= l; ++m) {
      modes_.push_back({l, m});
      polynomials_.push_back(make_polynomial(l, m));
    }
  }
}

int SphericalHarmonicBasis::index(const int l, const int m) const
{
  if (l < 0 || l > p_ || m < -l || m > l) {
    throw std::out_of_range("spherical-harmonic mode not found");
  }
  return l * l + (m + l);
}

std::span<const SolidHarmonicTerm>
SphericalHarmonicBasis::polynomial(const int index) const
{
  return polynomials_.at(static_cast<std::size_t>(index));
}

//------------------------------------------------------------------------------
// Solid-harmonic evaluation
//------------------------------------------------------------------------------

SolidHarmonicValues regular_solid_harmonics(
    const SphericalHarmonicBasis& basis, const Vec3& r)
{
  SolidHarmonicValues result;
  result.values.assign(static_cast<std::size_t>(basis.size()), 0.0);
  result.gradients.assign(static_cast<std::size_t>(basis.size()), Vec3{});
  for (int mode = 0; mode < basis.size(); ++mode) {
    for (const SolidHarmonicTerm& term : basis.polynomial(mode)) {
      const MultiIndex power = term.power;
      const double monomial = integer_power(r.x, power.ax) *
                              integer_power(r.y, power.ay) *
                              integer_power(r.z, power.az);
      result.values[static_cast<std::size_t>(mode)] +=
          term.coefficient * monomial;
      if (power.ax > 0) {
        result.gradients[static_cast<std::size_t>(mode)].x +=
            term.coefficient * power.ax * integer_power(r.x, power.ax - 1) *
            integer_power(r.y, power.ay) * integer_power(r.z, power.az);
      }
      if (power.ay > 0) {
        result.gradients[static_cast<std::size_t>(mode)].y +=
            term.coefficient * power.ay * integer_power(r.x, power.ax) *
            integer_power(r.y, power.ay - 1) * integer_power(r.z, power.az);
      }
      if (power.az > 0) {
        result.gradients[static_cast<std::size_t>(mode)].z +=
            term.coefficient * power.az * integer_power(r.x, power.ax) *
            integer_power(r.y, power.ay) * integer_power(r.z, power.az - 1);
      }
    }
  }
  return result;
}

SolidHarmonicValues irregular_solid_harmonics(
    const SphericalHarmonicBasis& basis, const Vec3& r)
{
  const MultiIndexSet derivative_basis(basis.order() + 1);
  const std::vector<double> derivatives =
      laplace_derivatives_raw(derivative_basis, r);
  SolidHarmonicValues result;
  result.values.assign(static_cast<std::size_t>(basis.size()), 0.0);
  result.gradients.assign(static_cast<std::size_t>(basis.size()), Vec3{});
  for (int mode = 0; mode < basis.size(); ++mode) {
    const int l = basis[mode].l;
    const double factor = 4.0 * std::numbers::pi *
                          (l % 2 == 0 ? 1.0 : -1.0) /
                          odd_double_factorial(l);
    for (const SolidHarmonicTerm& term : basis.polynomial(mode)) {
      const MultiIndex alpha = term.power;
      const double weight = factor * term.coefficient;
      result.values[static_cast<std::size_t>(mode)] +=
          weight * derivatives[static_cast<std::size_t>(
              derivative_basis.index(alpha))];
      result.gradients[static_cast<std::size_t>(mode)].x +=
          weight * derivatives[static_cast<std::size_t>(
              derivative_basis.index(add(alpha, {1, 0, 0})))];
      result.gradients[static_cast<std::size_t>(mode)].y +=
          weight * derivatives[static_cast<std::size_t>(
              derivative_basis.index(add(alpha, {0, 1, 0})))];
      result.gradients[static_cast<std::size_t>(mode)].z +=
          weight * derivatives[static_cast<std::size_t>(
              derivative_basis.index(add(alpha, {0, 0, 1})))];
    }
  }
  return result;
}

} // namespace cdfmm
