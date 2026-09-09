// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <numbers>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cdfmm/rectangular_prism.hpp"

using namespace cdfmm;

namespace {

// These helpers follow the MagTense getN_prism_3D and getAvgN_prism_3D
// convention: h contains full side lengths, M is magnetisation, and the
// returned tensor maps the total moment m = V*M to H.  The F1/F2 primitives
// below are written locally so these tests do not compare an implementation
// against itself.
double safe_asinh_ratio(double numerator, double a, double b)
{
    const double denominator = std::hypot(a, b);
    return denominator == 0.0 ? 0.0 : std::asinh(numerator / denominator);
}

double safe_atan_ratio(double numerator, double denominator)
{
    if (denominator != 0.0) {
        return std::atan(numerator / denominator);
    }
    return numerator == 0.0
        ? 0.0
        : std::copysign(0.5 * std::numbers::pi, numerator);
}

// In TileRectangularPrismTensor.f90, a zero face denominator is evaluated at
// 0.9999/1.0001 times the coordinate, and the two ratios are averaged before
// atan is applied.  For an exact face this symmetric branch is zero (also for
// a nonzero numerator), rather than the one-sided principal atan limit above.
double magtense_atan_ratio(double numerator, double denominator)
{
    return denominator == 0.0 ? 0.0 : std::atan(numerator / denominator);
}

double reference_f1(double x, double y, double z)
{
    const double radius = std::hypot(std::hypot(x, y), z);
    if (radius == 0.0) {
        return 0.0;
    }
    return 0.5 * y * (z * z - x * x) * safe_asinh_ratio(y, x, z) +
           0.5 * z * (y * y - x * x) * safe_asinh_ratio(z, x, y) -
           x * y * z * safe_atan_ratio(y * z, x * radius) +
           (2.0 * x * x - y * y - z * z) * radius / 6.0;
}

double reference_f2(double x, double y, double z)
{
    const double radius = std::hypot(std::hypot(x, y), z);
    if (radius == 0.0) {
        return 0.0;
    }
    return x * y * z * safe_asinh_ratio(z, x, y) +
           y * (3.0 * z * z - y * y) * safe_asinh_ratio(x, y, z) / 6.0 +
           x * (3.0 * z * z - x * x) * safe_asinh_ratio(y, x, z) / 6.0 -
           z * z * z * safe_atan_ratio(x * y, z * radius) / 6.0 -
           z * y * y * safe_atan_ratio(x * z, y * radius) / 2.0 -
           z * x * x * safe_atan_ratio(y * z, x * radius) / 2.0 -
           x * y * radius / 3.0;
}

template <typename Primitive>
double signed_corner_sum(Vec3 displacement, RectangularPrism source,
                         RectangularPrism target, Primitive primitive)
{
    constexpr std::array<int, 4> signs{1, -1, -1, 1};
    const std::array<double, 4> x_offsets{
        -0.5 * (source.hx + target.hx),
        -0.5 * source.hx + 0.5 * target.hx,
        0.5 * source.hx - 0.5 * target.hx,
        0.5 * (source.hx + target.hx)};
    const std::array<double, 4> y_offsets{
        -0.5 * (source.hy + target.hy),
        -0.5 * source.hy + 0.5 * target.hy,
        0.5 * source.hy - 0.5 * target.hy,
        0.5 * (source.hy + target.hy)};
    const std::array<double, 4> z_offsets{
        -0.5 * (source.hz + target.hz),
        -0.5 * source.hz + 0.5 * target.hz,
        0.5 * source.hz - 0.5 * target.hz,
        0.5 * (source.hz + target.hz)};
    double sum = 0.0;
    for (int ix = 0; ix < 4; ++ix) {
        for (int iy = 0; iy < 4; ++iy) {
            for (int iz = 0; iz < 4; ++iz) {
                sum += signs[ix] * signs[iy] * signs[iz] * primitive(
                    displacement.x + x_offsets[ix],
                    displacement.y + y_offsets[iy],
                    displacement.z + z_offsets[iz]);
            }
        }
    }
    return sum;
}

PairTensor reference_prism_prism(Vec3 displacement,
                                 RectangularPrism source,
                                 RectangularPrism target)
{
    const double scale = 1.0 /
        (4.0 * std::numbers::pi * source.volume() * target.volume());
    return {
        scale * signed_corner_sum(displacement, source, target, reference_f1),
        scale * signed_corner_sum(displacement, source, target,
                                   [](double x, double y, double z) {
                                       return reference_f2(x, y, z);
                                   }),
        scale * signed_corner_sum(displacement, source, target,
                                   [](double x, double y, double z) {
                                       return reference_f2(x, z, y);
                                   }),
        scale * signed_corner_sum(displacement, source, target,
                                   [](double x, double y, double z) {
                                       return reference_f1(y, x, z);
                                   }),
        scale * signed_corner_sum(displacement, source, target,
                                   [](double x, double y, double z) {
                                       return reference_f2(y, z, x);
                                   }),
        scale * signed_corner_sum(displacement, source, target,
                                   [](double x, double y, double z) {
                                       return reference_f1(z, y, x);
                                   })};
}

PairTensor reference_prism_point(Vec3 displacement, RectangularPrism source)
{
    std::array<double, 3> diagonal{};
    std::array<double, 3> off_diagonal{};
    for (int ix = 0; ix < 2; ++ix) {
        for (int iy = 0; iy < 2; ++iy) {
            for (int iz = 0; iz < 2; ++iz) {
                const double x = displacement.x + (ix == 0 ? -0.5 : 0.5) * source.hx;
                const double y = displacement.y + (iy == 0 ? -0.5 : 0.5) * source.hy;
                const double z = displacement.z + (iz == 0 ? -0.5 : 0.5) * source.hz;
                const double radius = std::hypot(std::hypot(x, y), z);
                const double sign = (ix + iy + iz) % 2 == 0 ? -1.0 : 1.0;
                diagonal[0] += sign * magtense_atan_ratio(y * z, x * radius);
                diagonal[1] += sign * magtense_atan_ratio(x * z, y * radius);
                diagonal[2] += sign * magtense_atan_ratio(x * y, z * radius);
                off_diagonal[0] += sign * safe_asinh_ratio(z, x, y);
                off_diagonal[1] += sign * safe_asinh_ratio(y, x, z);
                off_diagonal[2] += sign * safe_asinh_ratio(x, y, z);
            }
        }
    }
    const double scale = 1.0 / (4.0 * std::numbers::pi * source.volume());
    return {-scale * diagonal[0], scale * off_diagonal[0],
            scale * off_diagonal[1], -scale * diagonal[1],
            scale * off_diagonal[2], -scale * diagonal[2]};
}

Vec3 apply(PairTensor tensor, Vec3 moment)
{
    return {tensor.xx * moment.x + tensor.xy * moment.y + tensor.xz * moment.z,
            tensor.xy * moment.x + tensor.yy * moment.y + tensor.yz * moment.z,
            tensor.xz * moment.x + tensor.yz * moment.y + tensor.zz * moment.z};
}

void require_tensor_close(PairTensor actual, PairTensor expected, double margin)
{
    REQUIRE(actual.xx == Catch::Approx(expected.xx).margin(margin));
    REQUIRE(actual.xy == Catch::Approx(expected.xy).margin(margin));
    REQUIRE(actual.xz == Catch::Approx(expected.xz).margin(margin));
    REQUIRE(actual.yy == Catch::Approx(expected.yy).margin(margin));
    REQUIRE(actual.yz == Catch::Approx(expected.yz).margin(margin));
    REQUIRE(actual.zz == Catch::Approx(expected.zz).margin(margin));
}

} // namespace

TEST_CASE("rectangular prism point tensor matches MagTense convention",
          "[rectangular_prism][magtense]")
{
    const RectangularPrism prism{0.8, 1.3, 0.5};
    const Vec3 displacement{1.1, -0.7, 0.9};
    require_tensor_close(rectangular_prism_point_tensor(displacement, prism),
                         reference_prism_point(displacement, prism), 2.0e-13);
}

TEST_CASE("rectangular prism point tensor handles faces, edges, and interior",
          "[rectangular_prism][magtense]")
{
    const RectangularPrism prism{1.0, 0.6, 0.4};
    // MagTense uses its two-sided getF_limit branch exactly on a face.  These
    // are the resulting dimensionless diagonal N entries for this prism;
    // CDFMM divides by source volume because its tensor maps total moment.
    const double inverse_volume = 1.0 / prism.volume();
    require_tensor_close(
        rectangular_prism_point_tensor({0.5, 0.0, 0.0}, prism),
        {-0.01794740212956352 * inverse_volume, 0.0, 0.0,
         -0.17829870830164718 * inverse_volume, 0.0,
         -0.30375388956878935 * inverse_volume},
        4.0e-13);
    require_tensor_close(
        rectangular_prism_point_tensor({0.5, 0.3, 0.0}, prism),
        {-0.01608625487078998 * inverse_volume,
         1.3493129140126805 * inverse_volume, 0.0,
         -0.0437041337406494 * inverse_volume, 0.0,
         -0.19020961138856063 * inverse_volume},
        4.0e-13);

    for (const Vec3 displacement : {Vec3{0.1, 0.05, -0.02}}) {
        require_tensor_close(rectangular_prism_point_tensor(displacement, prism),
                             reference_prism_point(displacement, prism),
                             4.0e-13);
    }
}

TEST_CASE("rectangular prism pair tensor matches averaged MagTense F1/F2",
          "[rectangular_prism][magtense]")
{
    const RectangularPrism source{0.8, 1.3, 0.5};
    const RectangularPrism target{0.4, 0.7, 0.9};
    for (const Vec3 displacement : {Vec3{1.1, -0.7, 0.9},
                                    Vec3{0.4, 0.35, 0.2},
                                    Vec3{0.0, 0.0, 0.0}}) {
        require_tensor_close(
            rectangular_prism_rectangular_prism_tensor(displacement, source,
                                                       target),
            reference_prism_prism(displacement, source, target), 4.0e-12);
    }
}

TEST_CASE("rectangular prism tensors obey reciprocity and far-field limit",
          "[rectangular_prism][magtense]")
{
    const RectangularPrism source{0.7, 1.1, 0.9};
    const RectangularPrism target{0.5, 0.8, 0.6};
    const Vec3 displacement{40.0, -31.0, 27.0};
    const PairTensor pair = rectangular_prism_rectangular_prism_tensor(
        displacement, source, target);
    const PairTensor reverse = rectangular_prism_rectangular_prism_tensor(
        displacement * -1.0, target, source);
    require_tensor_close(pair, reverse, 2.0e-16);

    const PairTensor point = rectangular_prism_point_tensor(displacement, source);
    const Vec3 moment{0.37, -0.21, 0.48};
    const Vec3 pair_field = apply(pair, moment);
    const Vec3 point_field = apply(point, moment);
    REQUIRE(pair_field.x == Catch::Approx(point_field.x).epsilon(1.0e-4));
    REQUIRE(pair_field.y == Catch::Approx(point_field.y).epsilon(1.0e-4));
    REQUIRE(pair_field.z == Catch::Approx(point_field.z).epsilon(1.0e-4));
}

TEST_CASE("rectangular prism tensor preserves total-moment normalisation",
          "[rectangular_prism][magtense]")
{
    const RectangularPrism prism{0.6, 0.9, 1.2};
    const Vec3 displacement{2.0, 1.5, -1.1};
    const Vec3 magnetisation{0.8, -0.4, 0.3};
    const Vec3 total_moment = magnetisation * prism.volume();
    const Vec3 from_total = apply(rectangular_prism_point_tensor(displacement,
                                                                  prism),
                                   total_moment);
    const Vec3 expected = apply(reference_prism_point(displacement, prism),
                                total_moment);
    REQUIRE(from_total.x == Catch::Approx(expected.x).margin(2.0e-13));
    REQUIRE(from_total.y == Catch::Approx(expected.y).margin(2.0e-13));
    REQUIRE(from_total.z == Catch::Approx(expected.z).margin(2.0e-13));
}
