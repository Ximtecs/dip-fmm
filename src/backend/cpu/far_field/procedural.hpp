// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "cdfmm/math/vec3.hpp"

namespace cdfmm::detail::cpu {

/**
 * @brief Procedural point-source P2M and point-target L2P (spherical basis).
 *
 * Instead of streaming the precomputed coefficient rows (`3 C` scalars per
 * source or target), the executor recomputes the point operators from the
 * sorted positions during every evaluation through the shared kernels in
 * `operators/point_expansion_kernel.hpp`: the positions of a leaf stay cache
 * resident and the recurrence is multiply-add only. Points are processed in
 * SIMD packs of `lanes` (four FP64 or eight FP32 values); a leaf's tail pack
 * is padded with zero moments and zero displacements, which contribute
 * nothing. The only retained data are three small per-mode factor tables.
 *
 * The executor is compiled for orders 1 to `max_order`; the plan falls back
 * to the precomputed rows outside that range.
 */
template <typename Scalar>
class ProceduralPointExpansion {
public:
  static constexpr int lanes = sizeof(Scalar) == 8 ? 4 : 8;
  // Equal to operators::point_expansion::max_procedural_order (checked in
  // procedural.cpp, which includes the kernels).
  static constexpr int max_order = 15;

  ProceduralPointExpansion() = default;
  explicit ProceduralPointExpansion(int order);

  [[nodiscard]] int order() const noexcept { return order_; }

  /**
   * @brief Accumulates one leaf's P2M: `M += 1/(4 pi) sum_s m_s . grad R(x_s - centre)`.
   *
   * `positions` and `moments` are the leaf's slices of the sorted arrays.
   */
  template <typename Moment>
  void apply_p2m(const Vec3& centre, std::span<const Vec3> positions,
                 std::span<const Moment> moments, Scalar* M) const;

  /**
   * @brief Evaluates one leaf's L2P for every target of the leaf.
   *
   * Writes `results[t].H = -sum L grad R(x_t - centre)` when `field` is set
   * and `results[t].phi = sum L R(x_t - centre)` when `potential` is set;
   * the other member is set to zero, like the precomputed path.
   */
  template <typename Result>
  void apply_l2p(const Vec3& centre, std::span<const Vec3> positions,
                 const Scalar* L, std::span<Result> results, bool field,
                 bool potential) const;

  /// Retained bytes of the P2M factor table.
  [[nodiscard]] std::size_t p2m_memory_bytes() const noexcept {
    return p2m_factors_.size() * sizeof(Scalar);
  }
  /// Retained bytes of the L2P field and potential factor tables.
  [[nodiscard]] std::size_t l2p_memory_bytes() const noexcept {
    return (l2p_field_factors_.size() + l2p_potential_factors_.size()) *
           sizeof(Scalar);
  }
  /// Retained bytes: the three factor tables.
  [[nodiscard]] std::size_t memory_bytes() const noexcept {
    return p2m_memory_bytes() + l2p_memory_bytes();
  }

private:
  template <int P, typename Moment>
  void apply_p2m_order(const Vec3& centre, std::span<const Vec3> positions,
                       std::span<const Moment> moments, Scalar* M) const;
  template <int P, typename Result>
  void apply_l2p_order(const Vec3& centre, std::span<const Vec3> positions,
                       const Scalar* L, std::span<Result> results, bool field,
                       bool potential) const;

  int order_{0};
  std::vector<Scalar> p2m_factors_{};
  std::vector<Scalar> l2p_field_factors_{};
  std::vector<Scalar> l2p_potential_factors_{};
};

extern template class ProceduralPointExpansion<double>;
extern template class ProceduralPointExpansion<float>;

} // namespace cdfmm::detail::cpu
