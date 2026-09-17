// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <limits>
#include <numbers>

// The exact field tensor of a uniformly magnetised rectangular prism at a
// point (MagTense TileRectangularPrismTensor / TileNComponents), written once
// for any floating-point type.  The production tensor evaluates it in
// `long double` on the host (`rectangular_prism.cpp`); the representation
// benchmarks instantiate it in `double` and `float` on the CPU and inside a
// CUDA kernel to measure procedural reconstruction.  Every expression and its
// evaluation order are those of the original host implementation, so the
// `long double` instantiation reproduces the production results bit for bit.
//
// The kernel contains no exceptions: the one unreachable-in-practice fallback
// of the off-diagonal limit returns NaN, which the host wrapper turns into the
// original `std::domain_error`.
#if defined(__CUDACC__)
#define CDFMM_PRISM_KERNEL_HOST_DEVICE __host__ __device__
#else
#define CDFMM_PRISM_KERNEL_HOST_DEVICE
#endif

namespace cdfmm::detail::prism_kernel {

template <typename Real>
CDFMM_PRISM_KERNEL_HOST_DEVICE inline Real safe_atan_ratio(
    const Real numerator, const Real denominator) {
  if (denominator != Real{0}) {
    return std::atan(numerator / denominator);
  }
  if (numerator == Real{0}) {
    return Real{0};
  }
  return std::copysign(static_cast<Real>(0.5L) * std::numbers::pi_v<Real>,
                       numerator);
}

template <typename Real>
CDFMM_PRISM_KERNEL_HOST_DEVICE inline Real prism_ratio(const Real numerator,
                                                       const Real denominator) {
  if (denominator != Real{0}) {
    return numerator / denominator;
  }
  if (numerator == Real{0}) {
    return Real{0};
  }
  return std::copysign(std::numeric_limits<Real>::infinity(), numerator);
}

template <typename Real>
CDFMM_PRISM_KERNEL_HOST_DEVICE inline Real hypot3(const Real u, const Real v,
                                                  const Real w) {
  return std::hypot(std::hypot(u, v), w);
}

// MagTense TileRectangularPrismTensor.f90, f_3D/g_3D/h_3D.  The dimensions
// a,b,c are full side lengths; the source routine internally uses a/2 etc.
// A coincident face is perturbed on both sides, the ratios averaged, and
// only then is atan applied (getN_prism_3D).
template <typename Real>
CDFMM_PRISM_KERNEL_HOST_DEVICE inline Real f_3d(const Real a, const Real b,
                                                const Real c, const Real x,
                                                const Real y, const Real z) {
  const Real u = static_cast<Real>(0.5L) * a - x;
  const Real v = static_cast<Real>(0.5L) * b - y;
  const Real w = static_cast<Real>(0.5L) * c - z;
  const Real distance = hypot3(u, v, w);
  if (u == Real{0}) {
    const Real u_low = static_cast<Real>(0.5L) * a - static_cast<Real>(0.9999L) * x;
    const Real u_high = static_cast<Real>(0.5L) * a - static_cast<Real>(1.0001L) * x;
    const Real d_low = hypot3(u_low, v, w);
    const Real d_high = hypot3(u_high, v, w);
    return std::atan(static_cast<Real>(0.5L) *
                     (prism_ratio(v * w, u_low * d_low) +
                      prism_ratio(v * w, u_high * d_high)));
  }
  return safe_atan_ratio(v * w, u * distance);
}

template <typename Real>
CDFMM_PRISM_KERNEL_HOST_DEVICE inline Real g_3d(const Real a, const Real b,
                                                const Real c, const Real x,
                                                const Real y, const Real z) {
  const Real u = static_cast<Real>(0.5L) * a - x;
  const Real v = static_cast<Real>(0.5L) * b - y;
  const Real w = static_cast<Real>(0.5L) * c - z;
  const Real distance = hypot3(u, v, w);
  if (v == Real{0}) {
    const Real v_low = static_cast<Real>(0.5L) * b - static_cast<Real>(0.9999L) * y;
    const Real v_high = static_cast<Real>(0.5L) * b - static_cast<Real>(1.0001L) * y;
    const Real d_low = hypot3(u, v_low, w);
    const Real d_high = hypot3(u, v_high, w);
    return std::atan(static_cast<Real>(0.5L) *
                     (prism_ratio(u * w, v_low * d_low) +
                      prism_ratio(u * w, v_high * d_high)));
  }
  return safe_atan_ratio(u * w, v * distance);
}

template <typename Real>
CDFMM_PRISM_KERNEL_HOST_DEVICE inline Real h_3d(const Real a, const Real b,
                                                const Real c, const Real x,
                                                const Real y, const Real z) {
  const Real u = static_cast<Real>(0.5L) * a - x;
  const Real v = static_cast<Real>(0.5L) * b - y;
  const Real w = static_cast<Real>(0.5L) * c - z;
  const Real distance = hypot3(u, v, w);
  if (w == Real{0}) {
    const Real w_low = static_cast<Real>(0.5L) * c - static_cast<Real>(0.9999L) * z;
    const Real w_high = static_cast<Real>(0.5L) * c - static_cast<Real>(1.0001L) * z;
    const Real d_low = hypot3(u, v, w_low);
    const Real d_high = hypot3(u, v, w_high);
    return std::atan(static_cast<Real>(0.5L) *
                     (prism_ratio(u * v, w_low * d_low) +
                      prism_ratio(u * v, w_high * d_high)));
  }
  return safe_atan_ratio(u * v, w * distance);
}

// Sum of `component` over the eight signed corner images of the displacement.
template <typename Real, typename Component>
CDFMM_PRISM_KERNEL_HOST_DEVICE inline Real prism_corner_sum(
    const Real a, const Real b, const Real c, const Real dx, const Real dy,
    const Real dz, Component component) {
  Real result = Real{0};
  const Real signs[2] = {Real{-1}, Real{1}};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        result += component(a, b, c, signs[i] * dx, signs[j] * dy, signs[k] * dz);
      }
    }
  }
  return result;
}

// FF_3D in TileRectangularPrismTensor.f90.  Rationalised for a negative w to
// avoid losing all significant bits in R+w; the zero transverse-distance
// case is exact rather than a 0/0 division.
template <typename Real>
CDFMM_PRISM_KERNEL_HOST_DEVICE inline Real prism_corner_factor(
    const Real a, const Real b, const Real c, const Real x, const Real y,
    const Real z) {
  const Real u = static_cast<Real>(0.5L) * a - x;
  const Real v = static_cast<Real>(0.5L) * b - y;
  const Real w = static_cast<Real>(0.5L) * c - z;
  const Real distance = hypot3(u, v, w);
  if (w < Real{0}) {
    const Real transverse_squared = u * u + v * v;
    if (transverse_squared == Real{0}) {
      return Real{0};
    }
    return transverse_squared / (distance - w);
  }
  return distance + w;
}

template <typename Real>
struct LogProductPair {
  Real numerator{0};
  Real denominator{0};
  bool valid{false};
};

// TileNComponents.getN_prism_3D forms these four numerator and four
// denominator factors (FF_3D, or its cyclic GG/HH counterpart) and takes the
// logarithm of their products; `valid` is false when a factor vanishes.
template <typename Real>
CDFMM_PRISM_KERNEL_HOST_DEVICE inline LogProductPair<Real> log_ff_products_at(
    const Real a, const Real b, const Real c, const Real x, const Real y,
    const Real z) {
  const Real numerator[4] = {
      prism_corner_factor(a, b, c, x, y, z),
      prism_corner_factor(-a, -b, c, x, y, z),
      prism_corner_factor(a, -b, -c, x, y, z),
      prism_corner_factor(-a, b, -c, x, y, z)};
  const Real denominator[4] = {
      prism_corner_factor(a, -b, c, x, y, z),
      prism_corner_factor(-a, b, c, x, y, z),
      prism_corner_factor(a, b, -c, x, y, z),
      prism_corner_factor(-a, -b, -c, x, y, z)};
  LogProductPair<Real> result;
  for (int index = 0; index < 4; ++index) {
    const Real value = numerator[index];
    if (!(value > Real{0}) || !std::isfinite(value)) {
      return result;
    }
    result.numerator += std::log(value);
  }
  for (int index = 0; index < 4; ++index) {
    const Real value = denominator[index];
    if (!(value > Real{0}) || !std::isfinite(value)) {
      return result;
    }
    result.denominator += std::log(value);
  }
  result.valid = true;
  return result;
}

// The MagTense F/F product is the xy component (with cyclic coordinate
// permutations for yz/xz).  Reflection symmetry makes this component exactly
// zero on either of its two coordinate planes; that exact zero must not be
// replaced by an epsilon because the getF_limit perturbation is a two-sided
// positional limit whose products cancel on these symmetry planes.  Returns
// NaN for the unreachable-in-practice floating-point corner in which even the
// perturbed limit is outside the representable range.
template <typename Real>
CDFMM_PRISM_KERNEL_HOST_DEVICE inline Real log_ff_ratio(const Real a, const Real b,
                                                        const Real c, const Real x,
                                                        const Real y, const Real z) {
  if (x == Real{0} || y == Real{0}) {
    return Real{0};
  }
  const LogProductPair<Real> direct = log_ff_products_at(a, b, c, x, y, z);
  if (direct.valid) {
    return direct.numerator - direct.denominator;
  }

  // MagTense getF_limit evaluates at 0.9999 and 1.0001 times the position
  // when one factor vanishes (zero coordinates remain zero under the
  // multiplicative perturbation).  The products are averaged in log space so
  // the limiting branch cannot overflow or underflow before the ratio.
  const Real x_low = static_cast<Real>(0.9999L) * x;
  const Real y_low = static_cast<Real>(0.9999L) * y;
  const Real z_low = static_cast<Real>(0.9999L) * z;
  const Real x_high = static_cast<Real>(1.0001L) * x;
  const Real y_high = static_cast<Real>(1.0001L) * y;
  const Real z_high = static_cast<Real>(1.0001L) * z;
  const LogProductPair<Real> low = log_ff_products_at(a, b, c, x_low, y_low, z_low);
  const LogProductPair<Real> high =
      log_ff_products_at(a, b, c, x_high, y_high, z_high);
  if (low.valid && high.valid) {
    const auto log_sum_exp = [](const Real first, const Real second) {
      const Real larger = first > second ? first : second;
      const Real smaller = first > second ? second : first;
      return larger + std::log1p(std::exp(smaller - larger));
    };
    // MagTense averages products, not ratios:
    // log((nom_low+nom_high)/(den_low+den_high)).
    return log_sum_exp(low.numerator, high.numerator) -
           log_sum_exp(low.denominator, high.denominator);
  }
  return std::numeric_limits<Real>::quiet_NaN();
}

/**
 * @brief Six components (xx, xy, xz, yy, yz, zz) of the exact point tensor of
 * a prism with full side lengths a, b, c at displacement (rx, ry, rz) from its
 * centre, mapping the total moment to the field (includes 1/V).
 */
template <typename Real>
CDFMM_PRISM_KERNEL_HOST_DEVICE inline void rectangular_prism_point_tensor_components(
    const Real a, const Real b, const Real c, const Real rx, const Real ry,
    const Real rz, Real (&tensor)[6]) {
  constexpr Real four_pi = static_cast<Real>(4.0L) * std::numbers::pi_v<Real>;
  const Real volume = a * b * c;
  const Real diagonal_scale = static_cast<Real>(-1.0L) / (four_pi * volume);
  const Real off_diagonal_scale = static_cast<Real>(1.0L) / (four_pi * volume);
  const Real xx = diagonal_scale *
                  prism_corner_sum(a, b, c, rx, ry, rz,
                                   [](const Real a_, const Real b_, const Real c_,
                                      const Real x, const Real y, const Real z) {
                                     return f_3d(a_, b_, c_, x, y, z);
                                   });
  const Real yy = diagonal_scale *
                  prism_corner_sum(a, b, c, rx, ry, rz,
                                   [](const Real a_, const Real b_, const Real c_,
                                      const Real x, const Real y, const Real z) {
                                     return g_3d(a_, b_, c_, x, y, z);
                                   });
  const Real zz = diagonal_scale *
                  prism_corner_sum(a, b, c, rx, ry, rz,
                                   [](const Real a_, const Real b_, const Real c_,
                                      const Real x, const Real y, const Real z) {
                                     return h_3d(a_, b_, c_, x, y, z);
                                   });
  const Real xy = off_diagonal_scale * log_ff_ratio(a, b, c, rx, ry, rz);
  const Real yz = off_diagonal_scale * log_ff_ratio(b, c, a, ry, rz, rx);
  const Real xz = off_diagonal_scale * log_ff_ratio(c, a, b, rz, rx, ry);
  tensor[0] = xx;
  tensor[1] = xy;
  tensor[2] = xz;
  tensor[3] = yy;
  tensor[4] = yz;
  tensor[5] = zz;
}

} // namespace cdfmm::detail::prism_kernel

#undef CDFMM_PRISM_KERNEL_HOST_DEVICE
