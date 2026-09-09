// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/rectangular_prism.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
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

Real prism_ratio(const Real numerator, const Real denominator)
{
    if (denominator != 0.0L) {
        return numerator / denominator;
    }
    if (numerator == 0.0L) {
        return 0.0L;
    }
    return std::copysign(std::numeric_limits<Real>::infinity(), numerator);
}

Real safe_log_abs(const Real value, const Real epsilon)
{
    return std::log(std::abs(value == 0.0L ? epsilon : value));
}

// MagTense TileRectangularPrismTensor.f90, f_3D/g_3D/h_3D.  The dimensions
// a,b,c are full side lengths; the source routine internally uses a/2 etc.
Real f_3d(const Real a, const Real b, const Real c,
          const Real x, const Real y, const Real z)
{
    const Real u = 0.5L * a - x;
    const Real v = 0.5L * b - y;
    const Real w = 0.5L * c - z;
    const Real distance = std::hypot(std::hypot(u, v), w);
    if (u == 0.0L) {
        // MagTense perturbs the coincident face on both sides, averages the
        // ratios, and only then applies atan in getN_prism_3D.
        const Real u_low = 0.5L * a - 0.9999L * x;
        const Real u_high = 0.5L * a - 1.0001L * x;
        const Real d_low = std::hypot(std::hypot(u_low, v), w);
        const Real d_high = std::hypot(std::hypot(u_high, v), w);
        return std::atan(0.5L *
                         (prism_ratio(v * w, u_low * d_low) +
                          prism_ratio(v * w, u_high * d_high)));
    }
    return safe_atan_ratio(v * w, u * distance);
}

Real g_3d(const Real a, const Real b, const Real c,
          const Real x, const Real y, const Real z)
{
    const Real u = 0.5L * a - x;
    const Real v = 0.5L * b - y;
    const Real w = 0.5L * c - z;
    const Real distance = std::hypot(std::hypot(u, v), w);
    if (v == 0.0L) {
        const Real v_low = 0.5L * b - 0.9999L * y;
        const Real v_high = 0.5L * b - 1.0001L * y;
        const Real d_low = std::hypot(std::hypot(u, v_low), w);
        const Real d_high = std::hypot(std::hypot(u, v_high), w);
        return std::atan(0.5L *
                         (prism_ratio(u * w, v_low * d_low) +
                          prism_ratio(u * w, v_high * d_high)));
    }
    return safe_atan_ratio(u * w, v * distance);
}

Real h_3d(const Real a, const Real b, const Real c,
          const Real x, const Real y, const Real z)
{
    const Real u = 0.5L * a - x;
    const Real v = 0.5L * b - y;
    const Real w = 0.5L * c - z;
    const Real distance = std::hypot(std::hypot(u, v), w);
    if (w == 0.0L) {
        const Real w_low = 0.5L * c - 0.9999L * z;
        const Real w_high = 0.5L * c - 1.0001L * z;
        const Real d_low = std::hypot(std::hypot(u, v), w_low);
        const Real d_high = std::hypot(std::hypot(u, v), w_high);
        return std::atan(0.5L *
                         (prism_ratio(u * v, w_low * d_low) +
                          prism_ratio(u * v, w_high * d_high)));
    }
    return safe_atan_ratio(u * v, w * distance);
}

Real prism_corner_sum(const Real a, const Real b, const Real c,
                      const Vec3& displacement,
                      Real (*component)(Real, Real, Real, Real, Real, Real))
{
    Real result = 0.0L;
    constexpr Real signs[2] = {-1.0L, 1.0L};
    for (const Real sx : signs) {
        for (const Real sy : signs) {
            for (const Real sz : signs) {
                result += component(a, b, c, sx * displacement.x,
                                    sy * displacement.y, sz * displacement.z);
            }
        }
    }
    return result;
}

Real prism_corner_factor(const Real a, const Real b, const Real c,
                         const Real x, const Real y, const Real z)
{
    const Real u = 0.5L * a - x;
    const Real v = 0.5L * b - y;
    const Real w = 0.5L * c - z;
    const Real distance = std::hypot(std::hypot(u, v), w);
    // This is FF_3D in TileRectangularPrismTensor.f90.  Rationalise for a
    // negative w to avoid losing all significant bits in R+w.  The zero
    // transverse-distance case is exact, rather than a 0/0 division.
    if (w < 0.0L) {
        const Real transverse_squared = u * u + v * v;
        if (transverse_squared == 0.0L) {
            return 0.0L;
        }
        return transverse_squared / (distance - w);
    }
    return distance + w;
}

struct LogProductPair {
    Real numerator{0.0L};
    Real denominator{0.0L};
};

std::optional<LogProductPair> log_ff_products_at(
    const Real a, const Real b, const Real c,
    const Real x, const Real y, const Real z)
{
    // TileNComponents.getN_prism_3D forms these four numerator and four
    // denominator factors (FF_3D, or its cyclic GG/HH counterpart).
    const Real numerator[4] = {
        prism_corner_factor(a, b, c, x, y, z),
        prism_corner_factor(-a, -b, c, x, y, z),
        prism_corner_factor(a, -b, -c, x, y, z),
        prism_corner_factor(-a, b, -c, x, y, z)
    };
    const Real denominator[4] = {
        prism_corner_factor(a, -b, c, x, y, z),
        prism_corner_factor(-a, b, c, x, y, z),
        prism_corner_factor(a, b, -c, x, y, z),
        prism_corner_factor(-a, -b, -c, x, y, z)
    };
    LogProductPair result;
    for (const Real value : numerator) {
        if (!(value > 0.0L) || !std::isfinite(value)) {
            return std::nullopt;
        }
        result.numerator += std::log(value);
    }
    for (const Real value : denominator) {
        if (!(value > 0.0L) || !std::isfinite(value)) {
            return std::nullopt;
        }
        result.denominator += std::log(value);
    }
    return result;
}

std::optional<Real> log_ff_ratio_at(const Real a, const Real b, const Real c,
                                    const Real x, const Real y, const Real z)
{
    const std::optional<LogProductPair> products =
        log_ff_products_at(a, b, c, x, y, z);
    if (!products) {
        return std::nullopt;
    }
    return products->numerator - products->denominator;
}

Real log_ff_ratio(const Real a, const Real b, const Real c,
                  const Real x, const Real y, const Real z)
{
    // The MagTense F/F product is the xy component (with cyclic coordinate
    // permutations for yz/xz).  Reflection symmetry makes this component
    // exactly zero on either of its two coordinate planes.  In particular,
    // do not replace an exact zero here with an arbitrary epsilon: the
    // getF_limit perturbation is a two-sided positional limit and its
    // products cancel on these symmetry planes.
    if (x == 0.0L || y == 0.0L) {
        return 0.0L;
    }
    if (const std::optional<Real> result =
            log_ff_ratio_at(a, b, c, x, y, z)) {
        return *result;
    }

    // MagTense getF_limit evaluates at 0.9999 and 1.0001 times the position
    // when one factor vanishes.  Average the products in log space so the
    // limiting branch cannot overflow or underflow before taking their ratio.
    const auto perturbed = [](const Real value, const Real factor) {
        // Match TileRectangularPrismTensor.getF_limit exactly: zero
        // coordinates remain zero under a multiplicative perturbation.
        return factor * value;
    };
    const Real x_low = perturbed(x, 0.9999L);
    const Real y_low = perturbed(y, 0.9999L);
    const Real z_low = perturbed(z, 0.9999L);
    const Real x_high = perturbed(x, 1.0001L);
    const Real y_high = perturbed(y, 1.0001L);
    const Real z_high = perturbed(z, 1.0001L);
    const std::optional<LogProductPair> low =
        log_ff_products_at(a, b, c, x_low, y_low, z_low);
    const std::optional<LogProductPair> high =
        log_ff_products_at(a, b, c, x_high, y_high, z_high);
    if (low && high) {
        const auto log_sum_exp = [](const Real first, const Real second) {
            const Real larger = std::max(first, second);
            const Real smaller = std::min(first, second);
            return larger + std::log1p(std::exp(smaller - larger));
        };
        // MagTense averages products, not ratios:
        // log((nom_low+nom_high)/(den_low+den_high)).
        return log_sum_exp(low->numerator, high->numerator) -
            log_sum_exp(low->denominator, high->denominator);
    }

    // A prism field is finite away from a degenerate prism.  This fallback is
    // only reachable for an extreme floating-point corner; retain a bounded
    // one-sided limit rather than returning NaN from log(0).
    throw std::domain_error(
        "rectangular-prism off-diagonal limit is outside floating-point range");
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
    const Real a = source.hx;
    const Real b = source.hy;
    const Real c = source.hz;
    const Real volume = a * b * c;
    const Vec3& r = target_minus_source_representative;
    const Real diagonal_scale = -1.0L / (four_pi * volume);
    const Real off_diagonal_scale = 1.0L / (four_pi * volume);
    const Real xx = diagonal_scale * prism_corner_sum(a, b, c, r, f_3d);
    const Real yy = diagonal_scale * prism_corner_sum(a, b, c, r, g_3d);
    const Real zz = diagonal_scale * prism_corner_sum(a, b, c, r, h_3d);
    const Real xy = off_diagonal_scale * log_ff_ratio(a, b, c,
                                                       r.x, r.y, r.z);
    const Real yz = off_diagonal_scale * log_ff_ratio(b, c, a,
                                                       r.y, r.z, r.x);
    const Real xz = off_diagonal_scale * log_ff_ratio(c, a, b,
                                                       r.z, r.x, r.y);
    return {static_cast<double>(xx), static_cast<double>(xy),
            static_cast<double>(xz), static_cast<double>(yy),
            static_cast<double>(yz), static_cast<double>(zz)};
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

} // namespace cdfmm
