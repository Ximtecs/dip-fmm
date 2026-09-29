// SPDX-License-Identifier: Apache-2.0
//
// Exact analytical operators of a uniformly magnetised, axis-aligned
// rectangular prism, adapted from MagTense (`getN_prism_3D` for the field at
// a point, `TileRectangularPrismAvgTensor.f90` for the field averaged over a
// second prism).  Conventions shared with docs/math/finite-geometry.md:
//
//   * `hx, hy, hz` are full side lengths and the representative point is the
//     prism centre;
//   * the displacement argument is target representative minus source
//     representative;
//   * a pair tensor T is the symmetric 3x3 map H = T m from the source's
//     *total* moment m = V M, stored as (xx, xy, xz, yy, yz, zz).
//
// The pair tensors are evaluated in `long double` because the closed forms
// combine logarithms, arctangents and square roots that cancel severely near
// faces, edges and corners; the result is rounded to double once.

#include "cdfmm/geometry/primitives/rectangular_prism.hpp"

#include "geometry/primitives/rectangular_prism_point_kernel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace cdfmm {
namespace {

using Real = long double;
constexpr Real four_pi = 4.0L * std::numbers::pi_v<Real>;

void validate_prism(const RectangularPrism& prism, const char* name)
{
    if (!(std::isfinite(prism.hx) && std::isfinite(prism.hy) &&
          std::isfinite(prism.hz) && prism.hx > 0.0 && prism.hy > 0.0 &&
          prism.hz > 0.0)) {
        throw std::invalid_argument(std::string(name) +
                                    " dimensions must be finite and positive");
    }
    const Real volume = static_cast<Real>(prism.hx) * prism.hy * prism.hz;
    if (!(volume > 0.0L) || !std::isfinite(volume)) {
        throw std::invalid_argument(std::string(name) + " volume is invalid");
    }
}

// NOTE(cdfmm): The averaged-monomial path keeps its own side-length check.
// It deliberately omits the additional volume test applied by the exact
// pair-tensor routines, preserving the pre-v0.2 validation behaviour, and it
// reports the historical "cuboid" wording for both public spellings.
void validate_monomial_prism(const RectangularPrism& h, const char* name)
{
    if (!(std::isfinite(h.hx) && std::isfinite(h.hy) &&
          std::isfinite(h.hz) && h.hx > 0.0 && h.hy > 0.0 && h.hz > 0.0)) {
        throw std::invalid_argument(std::string(name) +
                                    " dimensions must be finite and positive");
    }
}

void validate_displacement(const Vec3& displacement)
{
    if (!(std::isfinite(displacement.x) && std::isfinite(displacement.y) &&
          std::isfinite(displacement.z))) {
        throw std::invalid_argument(
            "rectangular-prism displacement must be finite");
    }
}

// The MagTense primitives replace an exactly-zero coordinate by a small
// epsilon before taking logarithms and ratios.  The epsilon is scaled to the
// magnitude of the arguments so that the replacement is a relative, not an
// absolute, perturbation.
Real scale_for_limits(const Real x, const Real y, const Real z,
                      const Real a, const Real b, const Real c)
{
    return std::max({std::abs(x), std::abs(y), std::abs(z), std::abs(a),
                     std::abs(b), std::abs(c),
                     std::numeric_limits<Real>::denorm_min()});
}

Real limiting_epsilon(const Real x, const Real y, const Real z,
                      const Real a, const Real b, const Real c)
{
    return 64.0L * std::numeric_limits<Real>::epsilon() *
        scale_for_limits(x, y, z, a, b, c);
}

// atan(n / d) with the d -> 0 limit taken explicitly (+-pi/2, or 0 when the
// numerator vanishes too) instead of dividing by zero.
Real safe_atan_ratio(const Real numerator, const Real denominator)
{
    if (denominator != 0.0L) {
        return std::atan(numerator / denominator);
    }
    if (numerator == 0.0L) {
        return 0.0L;
    }
    return std::copysign(0.5L * std::numbers::pi_v<Real>, numerator);
}

Real safe_log_abs(const Real value, const Real epsilon)
{
    return std::log(std::abs(value == 0.0L ? epsilon : value));
}

// MagTense TileRectangularPrismAvgTensor.f90, F1.  The small replacements
// follow its zero-coordinate limiting branch, with a scale-aware epsilon.
Real F1(const Real x, const Real y, const Real z,
        const Real a, const Real b, const Real c)
{
    const Real epsilon = limiting_epsilon(x, y, z, a, b, c);
    const Real distance = std::hypot(std::hypot(x, y), z);
    Real X = x;
    Real Y = y;
    Real Z = z;
    if (X == 0.0L) {
        X = epsilon;
    }
    if (Y == 0.0L) {
        Y = epsilon;
    }
    if (Z == 0.0L) {
        Z = epsilon;
    }
    if (distance - Y == 0.0L) {
        Y += epsilon;
    }
    if (distance - Z == 0.0L) {
        Z += epsilon;
    }
    return X * Y * Z * safe_atan_ratio(Y * Z, X * distance) +
        0.5L * Y * (Z * Z - X * X) * safe_log_abs(distance - Y, epsilon) +
        0.5L * Z * (Y * Y - X * X) * safe_log_abs(distance - Z, epsilon) +
        (Y * Y + Z * Z - 2.0L * X * X) * distance / 6.0L;
}

// MagTense TileRectangularPrismAvgTensor.f90, F2.
Real F2(const Real x, const Real y, const Real z,
        const Real a, const Real b, const Real c)
{
    const Real epsilon = limiting_epsilon(x, y, z, a, b, c);
    Real X = x;
    Real Y = y;
    Real Z = z;
    if (X == 0.0L) {
        X = epsilon;
    }
    if (Y == 0.0L) {
        Y = epsilon;
    }
    if (Z == 0.0L) {
        Z = epsilon;
    }
    Real distance = std::hypot(std::hypot(X, Y), Z);
    if (distance == 0.0L) {
        distance = epsilon;
    }
    Real A = distance + X;
    Real B = distance + Y;
    Real C = distance + Z;
    if (A == 0.0L) {
        A = epsilon;
    }
    if (B == 0.0L) {
        B = epsilon;
    }
    if (C == 0.0L) {
        C = epsilon;
    }
    return -X * Y * Z * safe_log_abs(C, epsilon) +
        Y * (Y * Y - 3.0L * Z * Z) * safe_log_abs(A, epsilon) / 6.0L +
        X * (X * X - 3.0L * Z * Z) * safe_log_abs(B, epsilon) / 6.0L +
        0.5L * X * X * Z * safe_atan_ratio(Y * Z, X * distance) +
        0.5L * Y * Y * Z * safe_atan_ratio(X * Z, Y * distance) +
        Z * Z * Z * safe_atan_ratio(X * Y, Z * distance) / 6.0L +
        X * Y * distance / 3.0L;
}

// The six primitives at one evaluation point.  F1 and its two cyclic
// permutations give the diagonal components, F2 and its permutations the
// off-diagonal ones.  Between them the six contain one distance, three
// arctangents and six logarithms, each used by several of the six, so those
// are evaluated once and the six polynomials are combined from them in the
// primitives' own term order.  A point on a coordinate plane, on an axis, at
// the origin, or where a rounded distance equals a coordinate takes the
// primitives' own limiting branches instead, unchanged.
struct PointPrimitives {
    Real xx;
    Real yy;
    Real zz;
    Real xy;
    Real yz;
    Real xz;
};

PointPrimitives primitives_at(const Real x, const Real y, const Real z,
                              const Real a, const Real b, const Real c)
{
    const Real d = std::hypot(std::hypot(x, y), z);
    const bool limiting = x == 0.0L || y == 0.0L || z == 0.0L || d == 0.0L ||
        d - x == 0.0L || d - y == 0.0L || d - z == 0.0L ||
        d + x == 0.0L || d + y == 0.0L || d + z == 0.0L;
    if (limiting) {
        return {F1(x, y, z, a, b, c), F1(y, z, x, a, b, c),
                F1(z, x, y, a, b, c), F2(x, y, z, a, b, c),
                F2(y, z, x, a, b, c), F2(z, x, y, a, b, c)};
    }
    const Real atan_x = std::atan(y * z / (x * d));
    const Real atan_y = std::atan(z * x / (y * d));
    const Real atan_z = std::atan(x * y / (z * d));
    const Real log_minus_x = std::log(std::abs(d - x));
    const Real log_minus_y = std::log(std::abs(d - y));
    const Real log_minus_z = std::log(std::abs(d - z));
    const Real log_plus_x = std::log(std::abs(d + x));
    const Real log_plus_y = std::log(std::abs(d + y));
    const Real log_plus_z = std::log(std::abs(d + z));
    PointPrimitives p;
    // F1(x, y, z), F1(y, z, x), F1(z, x, y).
    p.xx = x * y * z * atan_x +
        0.5L * y * (z * z - x * x) * log_minus_y +
        0.5L * z * (y * y - x * x) * log_minus_z +
        (y * y + z * z - 2.0L * x * x) * d / 6.0L;
    p.yy = y * z * x * atan_y +
        0.5L * z * (x * x - y * y) * log_minus_z +
        0.5L * x * (z * z - y * y) * log_minus_x +
        (z * z + x * x - 2.0L * y * y) * d / 6.0L;
    p.zz = z * x * y * atan_z +
        0.5L * x * (y * y - z * z) * log_minus_x +
        0.5L * y * (x * x - z * z) * log_minus_y +
        (x * x + y * y - 2.0L * z * z) * d / 6.0L;
    // F2(x, y, z), F2(y, z, x), F2(z, x, y).
    p.xy = -x * y * z * log_plus_z +
        y * (y * y - 3.0L * z * z) * log_plus_x / 6.0L +
        x * (x * x - 3.0L * z * z) * log_plus_y / 6.0L +
        0.5L * x * x * z * atan_x +
        0.5L * y * y * z * atan_y +
        z * z * z * atan_z / 6.0L +
        x * y * d / 3.0L;
    p.yz = -y * z * x * log_plus_x +
        z * (z * z - 3.0L * x * x) * log_plus_y / 6.0L +
        y * (y * y - 3.0L * x * x) * log_plus_z / 6.0L +
        0.5L * y * y * x * atan_y +
        0.5L * z * z * x * atan_z +
        x * x * x * atan_x / 6.0L +
        y * z * d / 3.0L;
    p.xz = -z * x * y * log_plus_y +
        x * (x * x - 3.0L * y * y) * log_plus_z / 6.0L +
        z * (z * z - 3.0L * y * y) * log_plus_x / 6.0L +
        0.5L * z * z * y * atan_z +
        0.5L * x * x * y * atan_x +
        y * y * y * atan_y / 6.0L +
        z * x * d / 3.0L;
    return p;
}

// One axis of the prism-to-prism sum.  The integral over the target box of
// the source's field is an inclusion-exclusion over the eight target corners
// (sign t per axis) of an inclusion-exclusion over the eight source corners
// (sign s per axis), so along one axis the primitive is taken at
// d + t h_t - s h_s with coefficient t (-s), and a point's coefficient is the
// product over the three axes (AvgN_prism_definite_integral and its caller in
// TileRectangularPrismAvgTensor.f90).  Equal half-widths put the (+,+) and
// (-,-) points both at d, so that axis has three points with coefficients
// (1, -2, 1) -- the second difference -- instead of four: 27 evaluations for
// two equal prisms instead of 64.
struct AxisTerm {
    Real point{0.0L};
    int coefficient{0};
};

struct AxisTerms {
    std::array<AxisTerm, 4> terms{};
    int count{0};
};

AxisTerms axis_terms(const Real d, const Real source_half,
                     const Real target_half)
{
    AxisTerms axis;
    if (source_half == target_half) {
        axis.terms = {{{(d - target_half) - source_half, 1},
                       {d, -2},
                       {(d + target_half) + source_half, 1},
                       {}}};
        axis.count = 3;
        return axis;
    }
    axis.terms = {{{(d + target_half) - source_half, -1},
                   {(d + target_half) + source_half, 1},
                   {(d - target_half) - source_half, 1},
                   {(d - target_half) + source_half, -1}}};
    axis.count = 4;
    return axis;
}

} // namespace

PairTensor rectangular_prism_point_tensor(
    const Vec3& target_minus_source_representative,
    const RectangularPrism& source)
{
    validate_displacement(target_minus_source_representative);
    validate_prism(source, "source prism");
    // The formulas live in the precision-generic kernel header; the
    // production tensor evaluates them in `long double`.
    const Vec3& r = target_minus_source_representative;
    Real tensor[6];
    detail::prism_kernel::rectangular_prism_point_tensor_components<Real>(
        static_cast<Real>(source.hx), static_cast<Real>(source.hy),
        static_cast<Real>(source.hz), static_cast<Real>(r.x),
        static_cast<Real>(r.y), static_cast<Real>(r.z), tensor);
    for (const Real component : tensor) {
        if (std::isnan(component)) {
            // A prism field is finite away from a degenerate prism.  This is
            // only reachable for an extreme floating-point corner of the
            // off-diagonal limit; keep the historical diagnostic.
            throw std::domain_error(
                "rectangular-prism off-diagonal limit is outside "
                "floating-point range");
        }
    }
    return {static_cast<double>(tensor[0]), static_cast<double>(tensor[1]),
            static_cast<double>(tensor[2]), static_cast<double>(tensor[3]),
            static_cast<double>(tensor[4]), static_cast<double>(tensor[5])};
}

PairTensor rectangular_prism_rectangular_prism_tensor(
    const Vec3& target_minus_source_representative,
    const RectangularPrism& source,
    const RectangularPrism& target)
{
    validate_displacement(target_minus_source_representative);
    validate_prism(source, "source prism");
    validate_prism(target, "target prism");
    const Real source_volume = static_cast<Real>(source.hx) * source.hy *
        source.hz;
    const Real target_volume = static_cast<Real>(target.hx) * target.hy *
        target.hz;
    // 1/V_source converts the total moment to magnetisation, 1/V_target turns
    // the integral over the target into an average, and the minus sign is
    // H = -grad(phi).  F1 gives a diagonal component and F2 an off-diagonal
    // one; the other components follow by cyclic permutation of the axes.
    const Real scale = -1.0L / (four_pi * source_volume * target_volume);
    const Vec3& r = target_minus_source_representative;
    const AxisTerms axes[3] = {
        axis_terms(static_cast<Real>(r.x), 0.5L * source.hx, 0.5L * target.hx),
        axis_terms(static_cast<Real>(r.y), 0.5L * source.hy, 0.5L * target.hy),
        axis_terms(static_cast<Real>(r.z), 0.5L * source.hz, 0.5L * target.hz)};
    const Real a = static_cast<Real>(source.hx);
    const Real b = static_cast<Real>(source.hy);
    const Real c = static_cast<Real>(source.hz);
    Real xx = 0.0L;
    Real yy = 0.0L;
    Real zz = 0.0L;
    Real xy = 0.0L;
    Real yz = 0.0L;
    Real xz = 0.0L;
    for (int i = 0; i < axes[0].count; ++i) {
        for (int j = 0; j < axes[1].count; ++j) {
            for (int k = 0; k < axes[2].count; ++k) {
                const AxisTerm& tx = axes[0].terms[static_cast<std::size_t>(i)];
                const AxisTerm& ty = axes[1].terms[static_cast<std::size_t>(j)];
                const AxisTerm& tz = axes[2].terms[static_cast<std::size_t>(k)];
                const Real weight = static_cast<Real>(
                    tx.coefficient * ty.coefficient * tz.coefficient);
                const PointPrimitives p =
                    primitives_at(tx.point, ty.point, tz.point, a, b, c);
                xx += weight * p.xx;
                yy += weight * p.yy;
                zz += weight * p.zz;
                xy += weight * p.xy;
                yz += weight * p.yz;
                xz += weight * p.xz;
            }
        }
    }
    xx *= scale;
    yy *= scale;
    zz *= scale;
    xy *= scale;
    yz *= scale;
    xz *= scale;
    return {static_cast<double>(xx), static_cast<double>(xy),
            static_cast<double>(xz), static_cast<double>(yy),
            static_cast<double>(yz), static_cast<double>(zz)};
}

// J_beta(d, h): the average over the prism of the monomial-over-factorial
// (d + t)^beta / beta! for t in the prism about its centre.  The average
// separates by axis; along one axis with power n the binomial expansion of
// (d + t)^n / n! averaged over [-h/2, h/2] keeps only even powers of t and
// gives sum_{gamma even} d^(n-gamma)/(n-gamma)! * h^gamma / (2^gamma (gamma+1)!).
// This is the building block of every finite-prism P2M and L2P row.
double rectangular_prism_averaged_monomial(const MultiIndex& beta,
                                           const Vec3& d,
                                           const RectangularPrism& prism)
{
    validate_monomial_prism(prism, "cuboid");
    const int powers[3] = {beta.ax, beta.ay, beta.az};
    const double offsets[3] = {d.x, d.y, d.z};
    const double lengths[3] = {prism.hx, prism.hy, prism.hz};
    double result = 1.0;
    for (int axis = 0; axis < 3; ++axis) {
        double axis_sum = 0.0;
        for (int gamma = 0; gamma <= powers[axis]; gamma += 2) {
            axis_sum += std::pow(offsets[axis], powers[axis] - gamma) /
                MultiIndexSet::factorial(powers[axis] - gamma) *
                std::pow(lengths[axis], gamma) /
                (std::pow(2.0, gamma) * MultiIndexSet::factorial(gamma + 1));
        }
        result *= axis_sum;
    }
    return result;
}

double cuboid_averaged_monomial(const MultiIndex& beta, const Vec3& d,
                                const CuboidSize& h)
{
    return rectangular_prism_averaged_monomial(beta, d, h);
}

} // namespace cdfmm
