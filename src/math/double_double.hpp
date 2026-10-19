// SPDX-License-Identifier: Apache-2.0
#pragma once

// Double-double arithmetic: a value hi + lo of two doubles with |lo| <= ulp(hi)/2,
// about 106 bits of mantissa.  The four basic operations and the square root
// use the error-free transformations TwoSum and TwoProd (by fused multiply-add)
// and cost a few tens of double operations; they replace software quadruple
// precision where that dominated the run time (the triangle-pair reduction of
// the tetrahedron pair tensor).  The few transcendental functions go through
// _Float128, which represents every double-double exactly.
//
// The operations are the standard "sloppy" double-double algorithms (Hida, Li
// and Bailey, QD library; Joldes, Muller and Popescu, ACM TOMS 44 (2018)):
// relative error a few times 2^-106 for +, -, *, / and sqrt.

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

#if defined(__FLT128_MANT_DIG__)
inline _Float128 to_float128(const DoubleDouble& a)
{
    return static_cast<_Float128>(a.hi) + static_cast<_Float128>(a.lo);
}

inline DoubleDouble from_float128(const _Float128 value)
{
    const double high = static_cast<double>(value);
    return {high, static_cast<double>(value - static_cast<_Float128>(high))};
}

inline DoubleDouble log(const DoubleDouble& a) { return from_float128(logf128(to_float128(a))); }
inline DoubleDouble atan(const DoubleDouble& a) { return from_float128(atanf128(to_float128(a))); }
inline DoubleDouble asinh(const DoubleDouble& a) { return from_float128(asinhf128(to_float128(a))); }
inline DoubleDouble hypot(const DoubleDouble& a, const DoubleDouble& b)
{
    return sqrt(a * a + b * b);
}
#endif

} // namespace cdfmm::math
