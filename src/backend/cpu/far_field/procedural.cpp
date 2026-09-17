// SPDX-License-Identifier: Apache-2.0

#include "procedural.hpp"

#include <algorithm>
#include <stdexcept>
#include <type_traits>

#include "cdfmm/math/potential_field.hpp"
#include "lanes.hpp"
#include "operators/point_expansion_kernel.hpp"

namespace cdfmm::detail::cpu {

namespace {

// Runs `f(std::integral_constant<int, P>{})` for the compiled order P.
template <typename F>
void dispatch_order(const int order, F&& f) {
  switch (order) {
  case 1: f(std::integral_constant<int, 1>{}); return;
  case 2: f(std::integral_constant<int, 2>{}); return;
  case 3: f(std::integral_constant<int, 3>{}); return;
  case 4: f(std::integral_constant<int, 4>{}); return;
  case 5: f(std::integral_constant<int, 5>{}); return;
  case 6: f(std::integral_constant<int, 6>{}); return;
  case 7: f(std::integral_constant<int, 7>{}); return;
  case 8: f(std::integral_constant<int, 8>{}); return;
  case 9: f(std::integral_constant<int, 9>{}); return;
  case 10: f(std::integral_constant<int, 10>{}); return;
  default:
    throw std::logic_error(
        "procedural point expansion order is outside the compiled range");
  }
}

template <typename Scalar>
std::vector<Scalar> narrow(const std::vector<double>& values) {
  std::vector<Scalar> result(values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    result[index] = static_cast<Scalar>(values[index]);
  }
  return result;
}

} // namespace

template <typename Scalar>
ProceduralPointExpansion<Scalar>::ProceduralPointExpansion(const int order)
    : order_(order),
      p2m_factors_(
          narrow<Scalar>(operators::point_expansion::p2m_mode_factors(order))),
      l2p_field_factors_(narrow<Scalar>(
          operators::point_expansion::l2p_field_mode_factors(order))),
      l2p_potential_factors_(narrow<Scalar>(
          operators::point_expansion::l2p_potential_mode_factors(order))) {
  if (order < 1 || order > max_order) {
    throw std::invalid_argument(
        "procedural point expansion supports orders 1 to 10");
  }
}

template <typename Scalar>
template <int P, typename Moment>
void ProceduralPointExpansion<Scalar>::apply_p2m_order(
    const Vec3& centre, const std::span<const Vec3> positions,
    const std::span<const Moment> moments, Scalar* M) const {
  using Pack = Lanes<Scalar, lanes>;
  constexpr int C = (P + 1) * (P + 1);
  Pack acc[C];
  const std::size_t count = positions.size();
  for (std::size_t base = 0; base < count; base += lanes) {
    Pack dx;
    Pack dy;
    Pack dz;
    Pack mx;
    Pack my;
    Pack mz;
    const int valid =
        static_cast<int>(std::min<std::size_t>(lanes, count - base));
    for (int k = 0; k < valid; ++k) {
      const Vec3 d = positions[base + static_cast<std::size_t>(k)] - centre;
      const Moment m = moments[base + static_cast<std::size_t>(k)];
      dx.v[k] = static_cast<Scalar>(d.x);
      dy.v[k] = static_cast<Scalar>(d.y);
      dz.v[k] = static_cast<Scalar>(d.z);
      mx.v[k] = static_cast<Scalar>(m.x);
      my.v[k] = static_cast<Scalar>(m.y);
      mz.v[k] = static_cast<Scalar>(m.z);
    }
    operators::point_expansion::accumulate_point_p2m<P, Pack>(dx, dy, dz, mx,
                                                              my, mz, acc);
  }
  for (int index = 0; index < C; ++index) {
    M[index] += p2m_factors_[static_cast<std::size_t>(index)] *
                acc[index].sum();
  }
}

template <typename Scalar>
template <int P, typename Result>
void ProceduralPointExpansion<Scalar>::apply_l2p_order(
    const Vec3& centre, const std::span<const Vec3> positions,
    const Scalar* L, const std::span<Result> results, const bool field,
    const bool potential) const {
  using Pack = Lanes<Scalar, lanes>;
  constexpr int C = (P + 1) * (P + 1);
  // The leaf's locals scaled once by the mode factors (and the L2P sign).
  Scalar scaled_field[C];
  Scalar scaled_potential[C];
  for (int index = 0; index < C; ++index) {
    const auto mode = static_cast<std::size_t>(index);
    scaled_field[index] = l2p_field_factors_[mode] * L[index];
    scaled_potential[index] = l2p_potential_factors_[mode] * L[index];
  }
  const std::size_t count = positions.size();
  for (std::size_t base = 0; base < count; base += lanes) {
    Pack dx;
    Pack dy;
    Pack dz;
    const int valid =
        static_cast<int>(std::min<std::size_t>(lanes, count - base));
    for (int k = 0; k < valid; ++k) {
      const Vec3 d = positions[base + static_cast<std::size_t>(k)] - centre;
      dx.v[k] = static_cast<Scalar>(d.x);
      dy.v[k] = static_cast<Scalar>(d.y);
      dz.v[k] = static_cast<Scalar>(d.z);
    }
    Pack Hx;
    Pack Hy;
    Pack Hz;
    Pack phi;
    if (field) {
      operators::point_expansion::accumulate_point_l2p_field<P, Pack, Scalar>(
          dx, dy, dz, scaled_field, Hx, Hy, Hz);
    }
    if (potential) {
      operators::point_expansion::accumulate_point_l2p_potential<P, Pack,
                                                                 Scalar>(
          dx, dy, dz, scaled_potential, phi);
    }
    for (int k = 0; k < valid; ++k) {
      Result& result = results[base + static_cast<std::size_t>(k)];
      result.H.x = Hx.v[k];
      result.H.y = Hy.v[k];
      result.H.z = Hz.v[k];
      result.phi = phi.v[k];
    }
  }
}

template <typename Scalar>
template <typename Moment>
void ProceduralPointExpansion<Scalar>::apply_p2m(
    const Vec3& centre, const std::span<const Vec3> positions,
    const std::span<const Moment> moments, Scalar* M) const {
  dispatch_order(order_, [&](auto order_tag) {
    apply_p2m_order<decltype(order_tag)::value, Moment>(centre, positions,
                                                        moments, M);
  });
}

template <typename Scalar>
template <typename Result>
void ProceduralPointExpansion<Scalar>::apply_l2p(
    const Vec3& centre, const std::span<const Vec3> positions,
    const Scalar* L, const std::span<Result> results, const bool field,
    const bool potential) const {
  dispatch_order(order_, [&](auto order_tag) {
    apply_l2p_order<decltype(order_tag)::value, Result>(
        centre, positions, L, results, field, potential);
  });
}

template class ProceduralPointExpansion<double>;
template class ProceduralPointExpansion<float>;
template void ProceduralPointExpansion<double>::apply_p2m<Vec3>(
    const Vec3&, std::span<const Vec3>, std::span<const Vec3>, double*) const;
template void ProceduralPointExpansion<float>::apply_p2m<FloatVec3>(
    const Vec3&, std::span<const Vec3>, std::span<const FloatVec3>,
    float*) const;
template void ProceduralPointExpansion<double>::apply_l2p<PotentialField>(
    const Vec3&, std::span<const Vec3>, const double*,
    std::span<PotentialField>, bool, bool) const;
template void ProceduralPointExpansion<float>::apply_l2p<FloatPotentialField>(
    const Vec3&, std::span<const Vec3>, const float*,
    std::span<FloatPotentialField>, bool, bool) const;

} // namespace cdfmm::detail::cpu
