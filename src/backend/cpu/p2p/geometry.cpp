// SPDX-License-Identifier: Apache-2.0

#include "geometry.hpp"

#include <algorithm>

#ifdef CDFMM_USE_OPENMP
#include <omp.h>
#endif

#include "cdfmm/core/precision.hpp"
#include "cdfmm/math/potential_field.hpp"
#include "operators/p2p_point_kernel.hpp"

namespace cdfmm::detail::cpu {

namespace {

int current_thread() {
#ifdef CDFMM_USE_OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

// Sweeps one gathered neighbourhood for one target.  The self pair is
// removed without control flow: its displacement is replaced by a unit
// vector (so the formula stays finite) and its contribution is weighted by
// zero, which keeps the loop a straight SIMD reduction.
template <typename Scalar>
inline void sweep_gathered(const Scalar* sx, const Scalar* sy, const Scalar* sz,
                           const Scalar* mx, const Scalar* my, const Scalar* mz,
                           const int* index, const int count,
                           const Scalar tx, const Scalar ty, const Scalar tz,
                           const int self, Scalar& Hx, Scalar& Hy,
                           Scalar& Hz) {
  Scalar hx = Scalar{0};
  Scalar hy = Scalar{0};
  Scalar hz = Scalar{0};
#pragma omp simd reduction(+ : hx, hy, hz)
  for (int k = 0; k < count; ++k) {
    const bool excluded = index[k] == self;
    const Scalar weight = excluded ? Scalar{0} : Scalar{1};
    const Scalar rx = excluded ? Scalar{1} : tx - sx[k];
    const Scalar ry = excluded ? Scalar{0} : ty - sy[k];
    const Scalar rz = excluded ? Scalar{0} : tz - sz[k];
    Scalar cx = Scalar{0};
    Scalar cy = Scalar{0};
    Scalar cz = Scalar{0};
    operators::p2p::accumulate_point_dipole_field(rx, ry, rz, mx[k], my[k],
                                                  mz[k], cx, cy, cz);
    hx += weight * cx;
    hy += weight * cy;
    hz += weight * cz;
  }
  Hx += hx;
  Hy += hy;
  Hz += hz;
}

} // namespace

template <typename Scalar>
PointGeometryP2P<Scalar>::PointGeometryP2P(const StaticFmmTopology& topology,
                                           const int thread_capacity)
    : thread_capacity_(std::max(thread_capacity, 1)) {
  const auto& offsets = topology.p2p_target_leaf_offsets;
  for (std::size_t leaf = 0; leaf + 1 < offsets.size(); ++leaf) {
    std::size_t neighbours = 0;
    for (int row = offsets[leaf]; row < offsets[leaf + 1]; ++row) {
      neighbours +=
          topology.p2p_leaf_records[static_cast<std::size_t>(row)].source_count;
    }
    capacity_ = std::max(capacity_, neighbours);
  }
  // Round the per-thread stride to a whole number of cache lines.
  capacity_ = (capacity_ + 15) / 16 * 16;
  values_.assign(static_cast<std::size_t>(thread_capacity_) * 6 * capacity_,
                 Scalar{0});
  indices_.assign(static_cast<std::size_t>(thread_capacity_) * capacity_, 0);
}

template <typename Scalar>
template <typename Moment, typename Field>
void PointGeometryP2P<Scalar>::apply(
    const StaticFmmTopology& topology,
    const std::span<const Moment> sorted_moments, const std::span<Field> H,
    const std::span<const int> sorted_self_indices) {
  const auto& records = topology.p2p_leaf_records;
  const auto& offsets = topology.p2p_target_leaf_offsets;
  const std::span<const Vec3> sources = topology.sorted_source_positions;
  const std::span<const Vec3> targets = topology.sorted_target_positions;
  const int leaf_count = static_cast<int>(offsets.size()) - 1;

#pragma omp parallel for schedule(dynamic, 4) if (leaf_count >= 8)
  for (int leaf = 0; leaf < leaf_count; ++leaf) {
    const int row_begin = offsets[static_cast<std::size_t>(leaf)];
    const int row_end = offsets[static_cast<std::size_t>(leaf + 1)];
    if (row_begin == row_end) {
      continue;
    }
    const StaticP2PLeafRecord& first = records[static_cast<std::size_t>(row_begin)];
    const std::size_t target_begin = first.target_begin;
    const std::size_t target_end = first.target_begin + first.target_count;

    std::size_t neighbours = 0;
    for (int row = row_begin; row < row_end; ++row) {
      neighbours += records[static_cast<std::size_t>(row)].source_count;
    }
    const int thread = current_thread();
    const bool gathered = thread < thread_capacity_ && neighbours <= capacity_;
    if (gathered) {
      // Gather the whole neighbourhood once per target leaf; the image shift
      // of a periodic record is folded into the source position.
      Scalar* base =
          values_.data() + static_cast<std::size_t>(thread) * 6 * capacity_;
      Scalar* sx = base;
      Scalar* sy = sx + capacity_;
      Scalar* sz = sy + capacity_;
      Scalar* mx = sz + capacity_;
      Scalar* my = mx + capacity_;
      Scalar* mz = my + capacity_;
      int* index = indices_.data() + static_cast<std::size_t>(thread) * capacity_;
      std::size_t fill = 0;
      for (int row = row_begin; row < row_end; ++row) {
        const StaticP2PLeafRecord& record = records[static_cast<std::size_t>(row)];
        for (std::size_t s = 0; s < record.source_count; ++s) {
          const std::size_t source = record.source_begin + s;
          const Vec3 position = sources[source] + record.source_shift;
          const Moment moment = sorted_moments[source];
          sx[fill] = static_cast<Scalar>(position.x);
          sy[fill] = static_cast<Scalar>(position.y);
          sz[fill] = static_cast<Scalar>(position.z);
          mx[fill] = static_cast<Scalar>(moment.x);
          my[fill] = static_cast<Scalar>(moment.y);
          mz[fill] = static_cast<Scalar>(moment.z);
          index[fill] = record.skip_for_identity ? static_cast<int>(source) : -2;
          ++fill;
        }
      }
      const int count = static_cast<int>(fill);
      for (std::size_t target = target_begin; target < target_end; ++target) {
        const int self = sorted_self_indices.empty()
            ? -1 : sorted_self_indices[target];
        const Vec3 position = targets[target];
        Scalar Hx = Scalar{0};
        Scalar Hy = Scalar{0};
        Scalar Hz = Scalar{0};
        sweep_gathered(sx, sy, sz, mx, my, mz, index, count,
                       static_cast<Scalar>(position.x),
                       static_cast<Scalar>(position.y),
                       static_cast<Scalar>(position.z), self, Hx, Hy, Hz);
        Field& field = H[target];
        field.x += Hx;
        field.y += Hy;
        field.z += Hz;
      }
      continue;
    }

    // Capacity fallback: sweep the records directly without gathering.
    for (std::size_t target = target_begin; target < target_end; ++target) {
      const int self = sorted_self_indices.empty()
          ? -1 : sorted_self_indices[target];
      const Vec3 position = targets[target];
      Scalar Hx = Scalar{0};
      Scalar Hy = Scalar{0};
      Scalar Hz = Scalar{0};
      for (int row = row_begin; row < row_end; ++row) {
        const StaticP2PLeafRecord& record = records[static_cast<std::size_t>(row)];
        for (std::size_t s = 0; s < record.source_count; ++s) {
          const std::size_t source = record.source_begin + s;
          if (record.skip_for_identity && static_cast<int>(source) == self) {
            continue;
          }
          const Vec3 shifted = sources[source] + record.source_shift;
          const Moment moment = sorted_moments[source];
          operators::p2p::accumulate_point_dipole_field(
              static_cast<Scalar>(position.x - shifted.x),
              static_cast<Scalar>(position.y - shifted.y),
              static_cast<Scalar>(position.z - shifted.z),
              static_cast<Scalar>(moment.x), static_cast<Scalar>(moment.y),
              static_cast<Scalar>(moment.z), Hx, Hy, Hz);
        }
      }
      Field& field = H[target];
      field.x += Hx;
      field.y += Hy;
      field.z += Hz;
    }
  }
}

// The potential is requested rarely; it sweeps the records directly.
template <typename Scalar>
template <typename Moment, typename Result>
void PointGeometryP2P<Scalar>::apply_potential(
    const StaticFmmTopology& topology,
    const std::span<const Moment> sorted_moments,
    const std::span<Result> results,
    const std::span<const int> sorted_self_indices) const {
  const auto& records = topology.p2p_leaf_records;
  const auto& offsets = topology.p2p_target_leaf_offsets;
  const std::span<const Vec3> sources = topology.sorted_source_positions;
  const std::span<const Vec3> targets = topology.sorted_target_positions;
  const int leaf_count = static_cast<int>(offsets.size()) - 1;
#pragma omp parallel for schedule(dynamic, 4) if (leaf_count >= 8)
  for (int leaf = 0; leaf < leaf_count; ++leaf) {
    const int row_begin = offsets[static_cast<std::size_t>(leaf)];
    const int row_end = offsets[static_cast<std::size_t>(leaf + 1)];
    if (row_begin == row_end) {
      continue;
    }
    const StaticP2PLeafRecord& first = records[static_cast<std::size_t>(row_begin)];
    for (std::size_t target = first.target_begin;
         target < first.target_begin + first.target_count; ++target) {
      const int self = sorted_self_indices.empty()
          ? -1 : sorted_self_indices[target];
      const Vec3 position = targets[target];
      Scalar phi = Scalar{0};
      for (int row = row_begin; row < row_end; ++row) {
        const StaticP2PLeafRecord& record = records[static_cast<std::size_t>(row)];
        for (std::size_t s = 0; s < record.source_count; ++s) {
          const std::size_t source = record.source_begin + s;
          if (record.skip_for_identity && static_cast<int>(source) == self) {
            continue;
          }
          const Vec3 shifted = sources[source] + record.source_shift;
          const Moment moment = sorted_moments[source];
          phi += operators::p2p::point_dipole_potential(
              static_cast<Scalar>(position.x - shifted.x),
              static_cast<Scalar>(position.y - shifted.y),
              static_cast<Scalar>(position.z - shifted.z),
              static_cast<Scalar>(moment.x), static_cast<Scalar>(moment.y),
              static_cast<Scalar>(moment.z));
        }
      }
      results[target].phi += phi;
    }
  }
}

// Both plan precisions evaluate the pair formula in FP64 (measured to cost
// nothing over FP32 arithmetic here) and round once into the FP32 field, so
// the FP32 plan keeps the accuracy of its former FP64-computed tensors.
template class PointGeometryP2P<double>;
template void PointGeometryP2P<double>::apply<Vec3, Vec3>(
    const StaticFmmTopology&, std::span<const Vec3>, std::span<Vec3>,
    std::span<const int>);
template void PointGeometryP2P<double>::apply<FloatVec3, FloatVec3>(
    const StaticFmmTopology&, std::span<const FloatVec3>, std::span<FloatVec3>,
    std::span<const int>);
template void PointGeometryP2P<double>::apply_potential<FloatVec3, FloatPotentialField>(
    const StaticFmmTopology&, std::span<const FloatVec3>,
    std::span<FloatPotentialField>, std::span<const int>) const;

} // namespace cdfmm::detail::cpu
