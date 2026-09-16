// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "cdfmm/math/vec3.hpp"
#include "cdfmm/tree/static_topology.hpp"

namespace cdfmm::detail::cpu {

/**
 * @brief Position-based list-1 P2P for point sources and point targets.
 *
 * Instead of streaming stored pair tensors (48 or 24 bytes per pair) from
 * memory, the executor recomputes each point-dipole pair from the sorted
 * positions, which stay cache resident, using the authoritative kernel in
 * `operators/p2p_point_kernel.hpp`.  Per target leaf the neighbour sources of
 * all canonical list-1 records are gathered once into a structure-of-arrays
 * scratch (positions with their periodic image shift, moments and sorted
 * index), then every target of the leaf sweeps that contiguous range.  The
 * self pair is excluded through the sorted identity index of the target,
 * exactly like the tensor executors.  Each target row is summed by one
 * thread in record order, so results are deterministic for a given binary.
 *
 * The scratch is sized once at construction for the widest leaf
 * neighbourhood and the OpenMP team size seen then. If a later evaluation
 * runs with a larger team, the scratch grows once before the parallel region
 * so every thread keeps the gathered path; a thread still outside the
 * capacity falls back to sweeping the records without gathering. Evaluations
 * with an unchanged team never allocate.
 */
template <typename Scalar>
class PointGeometryP2P {
public:
  PointGeometryP2P() = default;
  PointGeometryP2P(const StaticFmmTopology& topology, int thread_capacity);

  /**
   * @brief Accumulates the near field of every target: `H[t] += sum_s T(t,s) m_s`.
   *
   * @param sorted_self_indices  Sorted source index excluded per target, or
   *                             -1; empty means no exclusion.
   */
  template <typename Moment, typename Field>
  void apply(const StaticFmmTopology& topology,
             std::span<const Moment> sorted_moments, std::span<Field> H,
             std::span<const int> sorted_self_indices);

  /** @brief Accumulates the near-field potential into `results[t].phi`. */
  template <typename Moment, typename Result>
  void apply_potential(const StaticFmmTopology& topology,
                       std::span<const Moment> sorted_moments,
                       std::span<Result> results,
                       std::span<const int> sorted_self_indices) const;

  [[nodiscard]] std::size_t memory_bytes() const noexcept {
    return values_.size() * sizeof(Scalar) + indices_.size() * sizeof(int);
  }
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] int thread_capacity() const noexcept { return thread_capacity_; }

private:
  void reserve_threads(int thread_capacity);
  void ensure_thread_capacity();

  /// Neighbour sources per thread: six SoA rows (x, y, z, mx, my, mz).
  std::vector<Scalar> values_{};
  std::vector<int> indices_{};
  std::size_t capacity_{0};
  int thread_capacity_{0};
};

extern template class PointGeometryP2P<double>;

} // namespace cdfmm::detail::cpu
