// SPDX-License-Identifier: Apache-2.0
#pragma once

// Allocation-free recurrence for the real regular solid harmonics and their
// Cartesian gradients, shared by the CPU and CUDA procedural point P2M/L2P
// executors.  It reproduces the convention of `regular_solid_harmonics`
// (docs/math/conventions.md, docs/math/spherical-expansions.md) without the per-mode polynomial
// tables: the recurrence runs in the factorial normalisation
//
//     Q_l^m(r) = r^l P_l^m(cos theta) e^{i m phi} / (l + m)!,   0 <= m <= l,
//
// (Condon--Shortley phase inside P_l^m) whose three-term, diagonal and
// gradient relations have rational coefficients only,
//
//     Q_0^0 = 1,
//     Q_l^l = -(x + i y) Q_{l-1}^{l-1} / (2 l),
//     Q_l^m = [(2l - 1) z Q_{l-1}^m - |r|^2 Q_{l-2}^m] / ((l - m)(l + m)),
//     d/dz Q_l^m = Q_{l-1}^m,
//     (d/dx + i d/dy) Q_l^m = Q_{l-1}^{m+1},
//     (d/dx - i d/dy) Q_l^m = -Q_{l-1}^{m-1},   Q_l^{-m} = (-1)^m conj(Q_l^m),
//
// and the repository's real tesseral mode is a constant multiple of one of
// its parts:
//
//     R_{l,0}  = f_{l,0}  Re Q_l^0,
//     R_{l,m}  = f_{l,m}  Re Q_l^m,   R_{l,-m} = f_{l,m} Im Q_l^m   (m > 0),
//     f_{l,m}  = sqrt((l - m)! (l + m)!) * (m == 0 ? 1 : sqrt(2) (-1)^m).
//
// `visit_regular_solid_harmonics` streams the Q-normalised value and gradient
// of every real mode in the plan's coefficient order (index l^2 + l + m);
// `regular_solid_harmonic_mode_factors` supplies the f_{l,m} table, which an
// executor multiplies in once per leaf.  Splitting the constant out keeps the
// hot loop free of square roots on both targets and lets a caller fold the
// operator's own constants (1/(4 pi), the L2P minus sign) into the same table.

#include <cmath>
#include <cstddef>
#include <vector>

#if defined(__CUDACC__)
#define CDFMM_SOLID_HARMONIC_HOST_DEVICE __host__ __device__
#else
#define CDFMM_SOLID_HARMONIC_HOST_DEVICE
#endif

// NOTE(cdfmm): `#pragma unroll` is nvcc's device-pass spelling.  nvcc leaves
// __CUDACC__ defined while it hands the host half of a .cu to the host
// compiler, so keying the hint on __CUDACC__ makes the host compiler parse a
// pragma it does not know.  __CUDA_ARCH__ is defined only in the device pass,
// which is the pass that understands it; the host half of a .cu then takes the
// same host spelling a .cpp translation unit already takes.
#if defined(__CUDA_ARCH__)
#define CDFMM_SOLID_HARMONIC_UNROLL _Pragma("unroll")
#elif defined(__GNUC__) && !defined(__clang__)
#define CDFMM_SOLID_HARMONIC_UNROLL _Pragma("GCC unroll 32")
#else
#define CDFMM_SOLID_HARMONIC_UNROLL
#endif

namespace cdfmm::solid_harmonics {

/// Largest expansion order the fixed-size recurrence rows are compiled for.
inline constexpr int max_recurrence_order = 15;

/**
 * @brief Streams the Q-normalised regular solid harmonics of degree <= P at
 *        `r = (x, y, z)` and their gradients.
 *
 * `visit(index, value, gx, gy, gz)` is called once per real mode in
 * coefficient order with the Q-normalised part named above; multiplying the
 * five numbers by `f_{l,|m|}` (see `regular_solid_harmonic_mode_factors`)
 * gives R_{l,m}(r) and grad R_{l,m}(r).  The rows are fixed-size arrays that
 * stay in registers once the degree loops are unrolled, so the function
 * allocates nothing and is usable in device code.  `Scalar` needs only
 * `+ - * /`, unary minus and construction from an integer or double
 * constant, so a SIMD lane pack serves as well as `float` or `double`.
 */
template <int P, typename Scalar, typename Visit>
CDFMM_SOLID_HARMONIC_HOST_DEVICE inline void
visit_regular_solid_harmonics(const Scalar x, const Scalar y, const Scalar z,
                              Visit&& visit) {
  static_assert(P >= 0 && P <= max_recurrence_order,
                "expansion order outside the compiled recurrence range");
  const Scalar r2 = x * x + y * y + z * z;
  // Rows of degree l-2, l-1 and l. Entry m of a row is written only once the
  // degree reaches m, so every entry above a row's degree is still zero; the
  // gradient relations read one entry past the previous degree, hence the
  // extra slot.
  Scalar prev2_re[P + 2] = {};
  Scalar prev2_im[P + 2] = {};
  Scalar prev_re[P + 2] = {};
  Scalar prev_im[P + 2] = {};
  Scalar cur_re[P + 2] = {};
  Scalar cur_im[P + 2] = {};
  const Scalar zero = static_cast<Scalar>(0);
  const Scalar one = static_cast<Scalar>(1);
  const Scalar half = static_cast<Scalar>(0.5);
  cur_re[0] = one;
  visit(0, one, zero, zero, zero);

  CDFMM_SOLID_HARMONIC_UNROLL
  for (int l = 1; l <= P; ++l) {
    CDFMM_SOLID_HARMONIC_UNROLL
    for (int m = 0; m < P + 2; ++m) {
      prev2_re[m] = prev_re[m];
      prev2_im[m] = prev_im[m];
      prev_re[m] = cur_re[m];
      prev_im[m] = cur_im[m];
    }
    // Three-term recurrence for m < l; Q_{l-2}^{l-1} is a zero slot.
    CDFMM_SOLID_HARMONIC_UNROLL
    for (int m = 0; m < l; ++m) {
      const Scalar inverse = one / static_cast<Scalar>((l - m) * (l + m));
      const Scalar a = static_cast<Scalar>(2 * l - 1) * inverse;
      cur_re[m] = a * z * prev_re[m] - inverse * r2 * prev2_re[m];
      cur_im[m] = a * z * prev_im[m] - inverse * r2 * prev2_im[m];
    }
    // Diagonal step: Q_l^l = -(x + i y) Q_{l-1}^{l-1} / (2 l).
    {
      const Scalar d = static_cast<Scalar>(-1) / static_cast<Scalar>(2 * l);
      cur_re[l] = d * (x * prev_re[l - 1] - y * prev_im[l - 1]);
      cur_im[l] = d * (x * prev_im[l - 1] + y * prev_re[l - 1]);
    }
    // Gradients of degree l from the degree l-1 row:
    //   plus  = (d/dx + i d/dy) Q_l^m =  Q_{l-1}^{m+1}
    //   minus = (d/dx - i d/dy) Q_l^m = -Q_{l-1}^{m-1}
    //   d/dx = (plus + minus) / 2,   d/dy = -i (plus - minus) / 2.
    const int base = l * l + l;
    CDFMM_SOLID_HARMONIC_UNROLL
    for (int m = 0; m <= l; ++m) {
      const Scalar plus_re = prev_re[m + 1];
      const Scalar plus_im = prev_im[m + 1];
      Scalar minus_re;
      Scalar minus_im;
      if (m == 0) {
        // Q_{l-1}^{-1} = -conj(Q_{l-1}^{1}).
        minus_re = prev_re[1];
        minus_im = -prev_im[1];
      } else {
        minus_re = -prev_re[m - 1];
        minus_im = -prev_im[m - 1];
      }
      const Scalar dx_re = half * (plus_re + minus_re);
      const Scalar dx_im = half * (plus_im + minus_im);
      const Scalar dy_re = half * (plus_im - minus_im);
      const Scalar dy_im = -half * (plus_re - minus_re);
      const Scalar dz_re = prev_re[m];
      const Scalar dz_im = prev_im[m];
      if (m == 0) {
        visit(base, cur_re[0], dx_re, dy_re, dz_re);
      } else {
        visit(base + m, cur_re[m], dx_re, dy_re, dz_re);
        visit(base - m, cur_im[m], dx_im, dy_im, dz_im);
      }
    }
  }
}

/**
 * @brief Per-mode factors f_{l,|m|} that turn the Q-normalised parts streamed
 *        by `visit_regular_solid_harmonics` into R_{l,m} (host only).
 */
inline std::vector<double> regular_solid_harmonic_mode_factors(const int p) {
  const auto factorial = [](const int n) {
    long double result = 1.0L;
    for (int value = 2; value <= n; ++value) {
      result *= static_cast<long double>(value);
    }
    return result;
  };
  std::vector<double> factors(static_cast<std::size_t>((p + 1) * (p + 1)));
  for (int l = 0; l <= p; ++l) {
    for (int m = -l; m <= l; ++m) {
      const int order = m < 0 ? -m : m;
      long double factor = std::sqrt(factorial(l - order) * factorial(l + order));
      if (order != 0) {
        factor *= std::sqrt(2.0L) * (order % 2 == 0 ? 1.0L : -1.0L);
      }
      factors[static_cast<std::size_t>(l * l + l + m)] =
          static_cast<double>(factor);
    }
  }
  return factors;
}

} // namespace cdfmm::solid_harmonics
