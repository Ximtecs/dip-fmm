// SPDX-License-Identifier: Apache-2.0
#pragma once

// Double-double arithmetic: a value hi + lo of two doubles with |lo| <= ulp(hi)/2,
// about 106 bits of mantissa.  The four basic operations and the square root
// use the error-free transformations TwoSum and TwoProd (by fused multiply-add)
// and cost a few tens of double operations; they replace software quadruple
// precision where that dominated the run time (the triangle-pair reduction of
// the tetrahedron pair tensor).  The transcendental functions are native too:
// exp and expm1 by argument reduction and a Taylor series, log and asinh by one
// Newton step from the double result, atan by half-angle reduction and its
// series from a table; their relative error is a few times 2^-104
// (tests/test_double_double).
//
// The operations are the standard "sloppy" double-double algorithms (Hida, Li
// and Bailey, QD library; Joldes, Muller and Popescu, ACM TOMS 44 (2018)):
// relative error a few times 2^-106 for +, -, *, / and sqrt.

#include <array>
#include <cmath>

namespace cdfmm::math {

struct DoubleDouble {
    double hi{0.0};
    double lo{0.0};

    constexpr DoubleDouble() = default;
    constexpr DoubleDouble(const double value) : hi(value) {}
    constexpr DoubleDouble(const int value) : hi(static_cast<double>(value)) {}
    constexpr DoubleDouble(const double high, const double low) : hi(high), lo(low) {}

    // The nearest double-double to a long double constant (exact for the
    // 64-bit long double mantissa).
    static constexpr DoubleDouble from_long_double(const long double value)
    {
        const double high = static_cast<double>(value);
        return {high, static_cast<double>(value - static_cast<long double>(high))};
    }

    explicit constexpr operator double() const { return hi + lo; }
};

namespace double_double_detail {

// s + e = a + b exactly.
inline DoubleDouble two_sum(const double a, const double b)
{
    const double s = a + b;
    const double v = s - a;
    const double e = (a - (s - v)) + (b - v);
    return {s, e};
}

// s + e = a + b exactly, given |a| >= |b|.
inline DoubleDouble quick_two_sum(const double a, const double b)
{
    const double s = a + b;
    return {s, b - (s - a)};
}

// p + e = a * b exactly.
inline DoubleDouble two_prod(const double a, const double b)
{
    const double p = a * b;
    return {p, std::fma(a, b, -p)};
}

} // namespace double_double_detail

inline DoubleDouble operator+(const DoubleDouble& a, const DoubleDouble& b)
{
    using namespace double_double_detail;
    const DoubleDouble s = two_sum(a.hi, b.hi);
    const DoubleDouble t = two_sum(a.lo, b.lo);
    DoubleDouble r = quick_two_sum(s.hi, s.lo + t.hi);
    return quick_two_sum(r.hi, r.lo + t.lo);
}

inline DoubleDouble operator-(const DoubleDouble& a)
{
    return {-a.hi, -a.lo};
}

inline DoubleDouble operator-(const DoubleDouble& a, const DoubleDouble& b)
{
    return a + (-b);
}

inline DoubleDouble operator*(const DoubleDouble& a, const DoubleDouble& b)
{
    using namespace double_double_detail;
    DoubleDouble p = two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return quick_two_sum(p.hi, p.lo);
}

inline DoubleDouble operator/(const DoubleDouble& a, const DoubleDouble& b)
{
    // Two quotient digits: q1 from the leading parts, q2 from the remainder.
    const double q1 = a.hi / b.hi;
    const DoubleDouble r = a - b * DoubleDouble(q1);
    const double q2 = r.hi / b.hi;
    const DoubleDouble r2 = r - b * DoubleDouble(q2);
    const double q3 = r2.hi / b.hi;
    return DoubleDouble(q1) + DoubleDouble(q2) + DoubleDouble(q3);
}

inline DoubleDouble& operator+=(DoubleDouble& a, const DoubleDouble& b)
{
    a = a + b;
    return a;
}

inline DoubleDouble& operator-=(DoubleDouble& a, const DoubleDouble& b)
{
    a = a - b;
    return a;
}

inline bool operator==(const DoubleDouble& a, const DoubleDouble& b)
{
    return a.hi == b.hi && a.lo == b.lo;
}

inline bool operator!=(const DoubleDouble& a, const DoubleDouble& b)
{
    return !(a == b);
}

inline bool operator<(const DoubleDouble& a, const DoubleDouble& b)
{
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}

inline bool operator>(const DoubleDouble& a, const DoubleDouble& b)
{
    return b < a;
}

inline bool operator<=(const DoubleDouble& a, const DoubleDouble& b)
{
    return !(b < a);
}

inline bool operator>=(const DoubleDouble& a, const DoubleDouble& b)
{
    return !(a < b);
}

inline DoubleDouble abs(const DoubleDouble& a)
{
    return a.hi < 0.0 ? -a : a;
}

inline bool isfinite(const DoubleDouble& a)
{
    return std::isfinite(a.hi);
}

// Newton correction of the double root: y + (a - y^2) / (2 y).
inline DoubleDouble sqrt(const DoubleDouble& a)
{
    if (!(a.hi > 0.0)) {
        return DoubleDouble(a.hi == 0.0 ? 0.0 : std::sqrt(a.hi));
    }
    const double y = std::sqrt(a.hi);
    const DoubleDouble y2 = double_double_detail::two_prod(y, y);
    const double correction = ((a - y2).hi) / (2.0 * y);
    return double_double_detail::quick_two_sum(y, correction);
}

inline DoubleDouble hypot(const DoubleDouble& a, const DoubleDouble& b)
{
    return sqrt(a * a + b * b);
}

namespace double_double_detail {

// ln 2 and pi / 2, rounded to double-double.
inline constexpr DoubleDouble kLn2{0.6931471805599453, 2.3190468138462996e-17};
inline constexpr DoubleDouble kHalfPi{1.5707963267948966, 6.123233995736766e-17};

inline DoubleDouble scale(const DoubleDouble& a, const int exponent)
{
    return {std::ldexp(a.hi, exponent), std::ldexp(a.lo, exponent)};
}

// a times a power of two that keeps it in range (exact).
inline DoubleDouble times_power_of_two(const DoubleDouble& a, const double power)
{
    return {a.hi * power, a.lo * power};
}

// 1 / n! for n = 0..10.
inline const std::array<DoubleDouble, 11>& inverse_factorials()
{
    static const std::array<DoubleDouble, 11> table = [] {
        std::array<DoubleDouble, 11> t{};
        double factorial = 1.0;
        for (int n = 0; n < 11; ++n) {
            factorial *= n > 0 ? n : 1;
            t[n] = DoubleDouble(1.0) / DoubleDouble(factorial);
        }
        return t;
    }();
    return table;
}

// 1 / (2 n + 1) for n = 0..10.
inline const std::array<DoubleDouble, 11>& inverse_odd()
{
    static const std::array<DoubleDouble, 11> table = [] {
        std::array<DoubleDouble, 11> t{};
        for (int n = 0; n < 11; ++n) {
            t[n] = DoubleDouble(1.0) / DoubleDouble(2 * n + 1);
        }
        return t;
    }();
    return table;
}

// exp(a) = 2^k (1 + s): returns s and sets k (k = 0 for |a| < ln2 / 2, where
// s is expm1(a) to full relative precision).  a - k ln2 is divided by 2^9,
// the Taylor series of expm1 summed to the r^10 term (relative truncation
// below 2^-115 for |r| <= ln2 / 1024), and the argument restored by nine
// doublings expm1(2x) = 2 expm1(x) + expm1(x)^2 (QD library).
inline DoubleDouble expm1_reduced(const DoubleDouble& a, int& k)
{
    k = static_cast<int>(std::nearbyint(a.hi / kLn2.hi));
    const DoubleDouble r = times_power_of_two(
        k == 0 ? a : a - kLn2 * DoubleDouble(static_cast<double>(k)), 0x1p-9);
    const auto& c = inverse_factorials();
    DoubleDouble p = c[10];
    for (int n = 9; n >= 2; --n) {
        p = c[n] + r * p;
    }
    DoubleDouble s = r + r * r * p;
    for (int i = 0; i < 9; ++i) {
        s = times_power_of_two(s, 2.0) + s * s;
    }
    return s;
}

} // namespace double_double_detail

inline DoubleDouble exp(const DoubleDouble& a)
{
    if (!std::isfinite(a.hi) || a.hi > 709.0 || a.hi < -745.0) {
        return DoubleDouble(std::exp(a.hi));
    }
    int k = 0;
    const DoubleDouble s = double_double_detail::expm1_reduced(a, k);
    return double_double_detail::scale(DoubleDouble(1.0) + s, k);
}

inline DoubleDouble expm1(const DoubleDouble& a)
{
    if (!std::isfinite(a.hi) || a.hi > 709.0 || a.hi < -745.0) {
        return DoubleDouble(std::expm1(a.hi));
    }
    int k = 0;
    const DoubleDouble s = double_double_detail::expm1_reduced(a, k);
    if (k == 0) {
        return s;
    }
    return double_double_detail::scale(DoubleDouble(1.0) + s, k) - DoubleDouble(1.0);
}

// y = y0 + log1p(d) from y0 = log(a.hi) and d = a exp(-y0) - 1, which is
// (a - 1) + a expm1(-y0) near a = 1, where y is small; |d| is about 2^-53
// |y| (or |a.lo| when y0 = 0), so log1p(d) = d - d^2 / 2 to 2^-106 |y|.
inline DoubleDouble log(const DoubleDouble& a)
{
    if (!(a.hi > 0.0) || !std::isfinite(a.hi)) {
        return DoubleDouble(std::log(a.hi));
    }
    const double y0 = std::log(a.hi);
    int k = 0;
    const DoubleDouble s = double_double_detail::expm1_reduced(DoubleDouble(-y0), k);
    const DoubleDouble residual = k == 0
        ? (a - DoubleDouble(1.0)) + a * s
        : a * double_double_detail::scale(DoubleDouble(1.0) + s, k) - DoubleDouble(1.0);
    return DoubleDouble(y0) + (residual - DoubleDouble(0.5 * residual.hi * residual.hi));
}

namespace double_double_detail {

// atan x for 0 <= x <= 1 by five half-angle reductions
// atan x = 2 atan(x / (1 + sqrt(1 + x^2))), to |x| <= tan(pi/128), and the
// series x (1 - x^2/3 + x^4/5 - ...) to the x^21 term (relative truncation
// below 2^-110).  It builds the table of atan_table().
inline DoubleDouble atan_by_halving(DoubleDouble x)
{
    for (int i = 0; i < 5; ++i) {
        x = x / (DoubleDouble(1.0) + sqrt(DoubleDouble(1.0) + x * x));
    }
    const DoubleDouble x2 = x * x;
    const auto& c = inverse_odd();
    DoubleDouble p = c[10];
    for (int n = 9; n >= 0; --n) {
        p = c[n] - x2 * p;
    }
    return times_power_of_two(x * p, 32.0);
}

// atan(k / 64) for k = 0..64.
inline const std::array<DoubleDouble, 65>& atan_table()
{
    static const std::array<DoubleDouble, 65> table = [] {
        std::array<DoubleDouble, 65> t{};
        for (int k = 0; k <= 64; ++k) {
            t[k] = atan_by_halving(DoubleDouble(k / 64.0));
        }
        return t;
    }();
    return table;
}

} // namespace double_double_detail

// atan x = atan c + atan((x - c) / (1 + x c)) for the nearest c = k / 64 after
// atan x = pi/2 - atan(1/x) for x > 1; the reduced argument is below 1/128
// and its series x (1 - x^2/3 + ...) is summed to the x^17 term (relative
// truncation below 2^-110).
inline DoubleDouble atan(const DoubleDouble& a)
{
    if (!(a.hi != 0.0) || std::isnan(a.hi)) {
        return a;
    }
    const bool negative = a.hi < 0.0;
    if (std::isinf(a.hi)) {
        return negative ? -double_double_detail::kHalfPi : double_double_detail::kHalfPi;
    }
    DoubleDouble x = negative ? -a : a;
    const bool invert = x.hi > 1.0;
    if (invert) {
        x = DoubleDouble(1.0) / x;
    }
    const int k = static_cast<int>(std::nearbyint(64.0 * x.hi));
    const double c = k / 64.0;
    const DoubleDouble u = k == 0 ? x : (x - DoubleDouble(c)) / (DoubleDouble(1.0) + x * DoubleDouble(c));
    const DoubleDouble u2 = u * u;
    const auto& odd = double_double_detail::inverse_odd();
    DoubleDouble p = odd[8];
    for (int n = 7; n >= 0; --n) {
        p = odd[n] - u2 * p;
    }
    DoubleDouble y = double_double_detail::atan_table()[static_cast<std::size_t>(k)] + u * p;
    if (invert) {
        y = double_double_detail::kHalfPi - y;
    }
    return negative ? -y : y;
}

// Second-order Newton step for sinh(y) = a from y0 = asinh(a.hi):
// y0 + c - tanh(y0) c^2 / 2 with c = (a - sinh y0) / cosh y0, and
// sinh y0 = (e + e / (1 + e)) / 2 for e = expm1(y0) when |y0| < ln2 / 2.  Beyond |a| = 1e30, asinh a = log(2|a|)
// to well below 2^-106.
inline DoubleDouble asinh(const DoubleDouble& a)
{
    if (!(a.hi != 0.0) || !std::isfinite(a.hi)) {
        return a;
    }
    if (std::abs(a.hi) > 1.0e30) {
        const DoubleDouble y = log(double_double_detail::times_power_of_two(abs(a), 2.0));
        return a.hi < 0.0 ? -y : y;
    }
    const double y0 = std::asinh(a.hi);
    int k = 0;
    const DoubleDouble s = double_double_detail::expm1_reduced(DoubleDouble(y0), k);
    DoubleDouble sinh_y0;
    if (k == 0) {
        sinh_y0 = double_double_detail::times_power_of_two(s + s / (DoubleDouble(1.0) + s), 0.5);
    } else {
        const DoubleDouble e = double_double_detail::scale(DoubleDouble(1.0) + s, k);
        sinh_y0 = double_double_detail::times_power_of_two(e - DoubleDouble(1.0) / e, 0.5);
    }
    const double c = (a - sinh_y0).hi / std::cosh(y0);
    return DoubleDouble(y0) + (DoubleDouble(c) - DoubleDouble(0.5 * std::tanh(y0) * c * c));
}

} // namespace cdfmm::math
