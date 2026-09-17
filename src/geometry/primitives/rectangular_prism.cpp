// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/geometry/primitives/rectangular_prism.hpp"

#include "geometry/primitives/rectangular_prism_point_kernel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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

template <typename Primitive>
Real definite_integral(Primitive primitive, const Real x1, const Real x2,
                       const Real y1, const Real y2, const Real z1,
                       const Real z2, const Real a, const Real b,
                       const Real c)
{
    // AvgN_prism_definite_integral, TileRectangularPrismAvgTensor.f90:80-94.
    return primitive(x2, y2, z2, a, b, c) -
        primitive(x1, y2, z2, a, b, c) -
        primitive(x2, y1, z2, a, b, c) +
        primitive(x1, y1, z2, a, b, c) -
        primitive(x2, y2, z1, a, b, c) +
        primitive(x1, y2, z1, a, b, c) +
        primitive(x2, y1, z1, a, b, c) -
        primitive(x1, y1, z1, a, b, c);
}

template <typename Primitive>
Real averaged_prism_sum(Primitive primitive, const Vec3& displacement,
                        const RectangularPrism& source,
                        const RectangularPrism& target)
{
    const Real source_half[3] = {
        0.5L * source.hx, 0.5L * source.hy, 0.5L * source.hz};
    const Real target_half[3] = {
        0.5L * target.hx, 0.5L * target.hy, 0.5L * target.hz};
    Real sum = 0.0L;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                const Real source_corner[3] = {
                    (i == 0 ? -1.0L : 1.0L) * source_half[0],
                    (j == 0 ? -1.0L : 1.0L) * source_half[1],
                    (k == 0 ? -1.0L : 1.0L) * source_half[2]};
                const Real x1 = displacement.x - target_half[0] -
                    source_corner[0];
                const Real x2 = displacement.x + target_half[0] -
                    source_corner[0];
                const Real y1 = displacement.y - target_half[1] -
                    source_corner[1];
                const Real y2 = displacement.y + target_half[1] -
                    source_corner[1];
                const Real z1 = displacement.z - target_half[2] -
                    source_corner[2];
                const Real z2 = displacement.z + target_half[2] -
                    source_corner[2];
                const Real sign = ((i + j + k) & 1) == 0 ? 1.0L : -1.0L;
                sum += sign * definite_integral(
                    primitive, x1, x2, y1, y2, z1, z2,
                    static_cast<Real>(source.hx),
                    static_cast<Real>(source.hy),
                    static_cast<Real>(source.hz));
            }
        }
    }
    return sum;
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
    const Real scale = -1.0L / (four_pi * source_volume * target_volume);
    const Vec3& r = target_minus_source_representative;
    const Real xx = scale * averaged_prism_sum(F1, r, source, target);
    const Real yy = scale * averaged_prism_sum(
        [](const Real x, const Real y, const Real z,
           const Real a, const Real b, const Real c) {
            return F1(y, z, x, a, b, c);
        }, r, source, target);
    const Real zz = scale * averaged_prism_sum(
        [](const Real x, const Real y, const Real z,
           const Real a, const Real b, const Real c) {
            return F1(z, x, y, a, b, c);
        }, r, source, target);
    const Real xy = scale * averaged_prism_sum(F2, r, source, target);
    const Real yz = scale * averaged_prism_sum(
        [](const Real x, const Real y, const Real z,
           const Real a, const Real b, const Real c) {
            return F2(y, z, x, a, b, c);
        }, r, source, target);
    const Real xz = scale * averaged_prism_sum(
        [](const Real x, const Real y, const Real z,
           const Real a, const Real b, const Real c) {
            return F2(z, x, y, a, b, c);
        }, r, source, target);
    return {static_cast<double>(xx), static_cast<double>(xy),
            static_cast<double>(xz), static_cast<double>(yy),
            static_cast<double>(yz), static_cast<double>(zz)};
}

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
