// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "math/double_double.hpp"

using cdfmm::math::DoubleDouble;

namespace {

// A double-double with a random low part, |a.hi| = 10^e for e uniform in
// [low, high).
DoubleDouble random_double_double(std::mt19937_64& generator, const double low, const double high,
                                  const bool sign)
{
    std::uniform_real_distribution<double> exponent(low, high);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    double x = std::pow(10.0, exponent(generator));
    if (sign && unit(generator) < 0.0) {
        x = -x;
    }
    return DoubleDouble(x) + DoubleDouble(x * 1.0e-17 * unit(generator));
}

} // namespace

TEST_CASE("double-double transcendental functions keep about 103 bits", "[double_double]")
{
#if defined(__FLT128_MANT_DIG__)
    // _Float128 represents each argument used here exactly; the reference
    // functions are correctly rounded to 113 bits.
    const auto wide = [](const DoubleDouble& a) {
        return static_cast<_Float128>(a.hi) + static_cast<_Float128>(a.lo);
    };
    const auto relative = [](const _Float128 value, const _Float128 reference) {
        return static_cast<double>((value - reference) / reference);
    };
    const double unit = std::ldexp(1.0, -106);
    double log_error = 0.0, log1p_error = 0.0, atan_error = 0.0, asinh_error = 0.0, exp_error = 0.0;
    std::mt19937_64 generator(2026);
    for (int i = 0; i < 20000; ++i) {
        const DoubleDouble a = random_double_double(generator, -20.0, 20.0, false);
        log_error = std::max(log_error, std::abs(relative(wide(cdfmm::math::log(a)), logf128(wide(a)))));
        const double small = random_double_double(generator, -30.0, -0.5, true).hi;
        const DoubleDouble one_plus = cdfmm::math::double_double_detail::two_sum(1.0, small);
        log1p_error = std::max(log1p_error, std::abs(relative(wide(cdfmm::math::log(one_plus)),
                                                              log1pf128(static_cast<_Float128>(small)))));
        const DoubleDouble b = random_double_double(generator, -20.0, 20.0, true);
        atan_error = std::max(atan_error, std::abs(relative(wide(cdfmm::math::atan(b)), atanf128(wide(b)))));
        asinh_error = std::max(asinh_error, std::abs(relative(wide(cdfmm::math::asinh(b)), asinhf128(wide(b)))));
        const DoubleDouble c = random_double_double(generator, -20.0, 0.0, true);
        exp_error = std::max(exp_error, std::abs(relative(wide(cdfmm::math::exp(c)), expf128(wide(c)))));
    }
    CHECK(log_error < 16.0 * unit);
    CHECK(log1p_error < 16.0 * unit);
    CHECK(atan_error < 16.0 * unit);
    CHECK(asinh_error < 16.0 * unit);
    CHECK(exp_error < 16.0 * unit);
#else
    SKIP("no _Float128 reference");
#endif
}

TEST_CASE("double-double transcendental functions handle special arguments", "[double_double]")
{
    using cdfmm::math::asinh;
    using cdfmm::math::atan;
    using cdfmm::math::log;
    CHECK(static_cast<double>(log(DoubleDouble(1.0))) == 0.0);
    CHECK(static_cast<double>(atan(DoubleDouble(0.0))) == 0.0);
    CHECK(static_cast<double>(asinh(DoubleDouble(0.0))) == 0.0);
    const double infinity = std::numeric_limits<double>::infinity();
    CHECK(static_cast<double>(atan(DoubleDouble(infinity))) == std::atan(infinity));
    CHECK(static_cast<double>(atan(DoubleDouble(-1.0e300))) == std::atan(-1.0e300));
    CHECK(static_cast<double>(asinh(DoubleDouble(1.0e40))) == std::asinh(1.0e40));
    CHECK(static_cast<double>(asinh(DoubleDouble(-1.0e-300))) == -1.0e-300);
    CHECK(std::isnan(static_cast<double>(log(DoubleDouble(-1.0)))));
}
