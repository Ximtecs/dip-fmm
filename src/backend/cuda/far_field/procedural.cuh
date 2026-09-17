// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <stdexcept>

#include "backend/cuda/common/error.hpp"
#include "operators/point_expansion_kernel.hpp"

namespace cdfmm::cuda_far_field_detail {

/// One occupied leaf of the procedural point executors: its node index and
/// the contiguous sorted range of its points.
struct ProceduralLeaf {
  int node{0};
  int begin{0};
  int count{0};
  int padding{0};
};

/// One point's displacement from its leaf centre, padded to one aligned load.
template <typename Scalar>
struct alignas(4 * sizeof(Scalar)) ProceduralPoint {
  Scalar x{0};
  Scalar y{0};
  Scalar z{0};
  Scalar w{0};
};

inline constexpr int procedural_threads = 128;

namespace {

// Lane groups of `lanes_per_leaf` consecutive lanes (a power of two, at most
// 32) own one leaf each; a warp holds `32 / lanes_per_leaf` leaves, so small
// leaves do not idle most of the warp. The lanes of a group stride through the
// leaf's sources, each running the recurrence for its own source and keeping
// the `C` Q-normalised sums in registers; the group then reduces with
// shuffles and one lane scales the sums by the mode factors into the leaf's
// multipole slot. A leaf belongs to exactly one group, so no atomics are
// needed; the shuffle loops are uniform across the warp.
template <int P, typename Scalar, typename Vector>
__global__ void __launch_bounds__(procedural_threads) procedural_p2m_kernel(
    const ProceduralLeaf *__restrict__ leaves, const int leaf_count,
    const int lanes_per_leaf,
    const ProceduralPoint<Scalar> *__restrict__ displacements,
    const Vector *__restrict__ moments, const Scalar *__restrict__ factors,
    Scalar *__restrict__ multipoles) {
  constexpr int C = (P + 1) * (P + 1);
  const int lane = static_cast<int>(threadIdx.x & 31);
  const int warp =
      static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int groups_per_warp = 32 / lanes_per_leaf;
  const int group = lane / lanes_per_leaf;
  const int lane_in_group = lane - group * lanes_per_leaf;
  const int leaf_index = warp * groups_per_warp + group;
  const bool active = leaf_index < leaf_count;
  ProceduralLeaf leaf;
  if (active) {
    leaf = leaves[leaf_index];
  }
  Scalar acc[C];
#pragma unroll
  for (int index = 0; index < C; ++index) {
    acc[index] = Scalar{0};
  }
  for (int local = lane_in_group; local < leaf.count; local += lanes_per_leaf) {
    const int source = leaf.begin + local;
    const ProceduralPoint<Scalar> d = displacements[source];
    const Vector m = moments[source];
    operators::point_expansion::accumulate_point_p2m<P, Scalar>(
        d.x, d.y, d.z, static_cast<Scalar>(m.x), static_cast<Scalar>(m.y),
        static_cast<Scalar>(m.z), acc);
  }
#pragma unroll
  for (int index = 0; index < C; ++index) {
    for (int offset = lanes_per_leaf >> 1; offset > 0; offset >>= 1) {
      acc[index] += __shfl_xor_sync(0xffffffffU, acc[index], offset);
    }
  }
  if (active && lane_in_group == 0) {
    Scalar *M = multipoles + static_cast<std::size_t>(leaf.node) * C;
#pragma unroll
    for (int index = 0; index < C; ++index) {
      M[index] += factors[index] * acc[index];
    }
  }
}

// Same lane-group layout over the target leaves. Every lane evaluates whole
// targets: it scales the leaf's locals once into registers and sweeps its
// targets, so no reduction is needed and each target is written by one lane.
template <int P, typename Scalar, typename Vector>
__global__ void __launch_bounds__(procedural_threads) procedural_l2p_kernel(
    const ProceduralLeaf *__restrict__ leaves, const int leaf_count,
    const int lanes_per_leaf,
    const ProceduralPoint<Scalar> *__restrict__ displacements,
    const Scalar *__restrict__ locals, const Scalar *__restrict__ factors,
    Vector *__restrict__ fields) {
  constexpr int C = (P + 1) * (P + 1);
  const int lane = static_cast<int>(threadIdx.x & 31);
  const int warp =
      static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int groups_per_warp = 32 / lanes_per_leaf;
  const int group = lane / lanes_per_leaf;
  const int lane_in_group = lane - group * lanes_per_leaf;
  const int leaf_index = warp * groups_per_warp + group;
  if (leaf_index >= leaf_count) {
    return;
  }
  const ProceduralLeaf leaf = leaves[leaf_index];
  const Scalar *L = locals + static_cast<std::size_t>(leaf.node) * C;
  Scalar scaled[C];
#pragma unroll
  for (int index = 0; index < C; ++index) {
    scaled[index] = factors[index] * L[index];
  }
  for (int local = lane_in_group; local < leaf.count; local += lanes_per_leaf) {
    const int target = leaf.begin + local;
    const ProceduralPoint<Scalar> d = displacements[target];
    Scalar hx = Scalar{0};
    Scalar hy = Scalar{0};
    Scalar hz = Scalar{0};
    operators::point_expansion::accumulate_point_l2p_field<P, Scalar, Scalar>(
        d.x, d.y, d.z, scaled, hx, hy, hz);
    Vector &field = fields[target];
    field.x += hx;
    field.y += hy;
    field.z += hz;
  }
}

inline int procedural_blocks(const int leaf_count, const int lanes_per_leaf) {
  const int groups_per_warp = 32 / lanes_per_leaf;
  const int warps = (leaf_count + groups_per_warp - 1) / groups_per_warp;
  constexpr int warps_per_block = procedural_threads / 32;
  return (warps + warps_per_block - 1) / warps_per_block;
}

// Launches the instantiation compiled for `order`.
template <typename Scalar, typename Vector>
void launch_procedural_p2m(const int order, const ProceduralLeaf *leaves,
                           const int leaf_count, const int lanes_per_leaf,
                           const ProceduralPoint<Scalar> *displacements,
                           const Vector *moments, const Scalar *factors,
                           Scalar *multipoles, cudaStream_t stream) {
  if (leaf_count == 0) {
    return;
  }
  const int blocks = procedural_blocks(leaf_count, lanes_per_leaf);
  switch (order) {
#define CDFMM_PROCEDURAL_P2M_CASE(P)                                         \
  case P:                                                                    \
    procedural_p2m_kernel<P, Scalar, Vector>                                 \
        <<<blocks, procedural_threads, 0, stream>>>(                         \
            leaves, leaf_count, lanes_per_leaf, displacements, moments,      \
            factors, multipoles);                                            \
    break;
    CDFMM_PROCEDURAL_P2M_CASE(1)
    CDFMM_PROCEDURAL_P2M_CASE(2)
    CDFMM_PROCEDURAL_P2M_CASE(3)
    CDFMM_PROCEDURAL_P2M_CASE(4)
    CDFMM_PROCEDURAL_P2M_CASE(5)
    CDFMM_PROCEDURAL_P2M_CASE(6)
    CDFMM_PROCEDURAL_P2M_CASE(7)
    CDFMM_PROCEDURAL_P2M_CASE(8)
    CDFMM_PROCEDURAL_P2M_CASE(9)
    CDFMM_PROCEDURAL_P2M_CASE(10)
#undef CDFMM_PROCEDURAL_P2M_CASE
  default:
    throw std::logic_error(
        "procedural CUDA P2M order is outside the compiled range");
  }
  cuda_detail::check_cuda(cudaGetLastError(),
                          "launch procedural CUDA P2M kernel");
}

template <typename Scalar, typename Vector>
void launch_procedural_l2p(const int order, const ProceduralLeaf *leaves,
                           const int leaf_count, const int lanes_per_leaf,
                           const ProceduralPoint<Scalar> *displacements,
                           const Scalar *locals, const Scalar *factors,
                           Vector *fields, cudaStream_t stream) {
  if (leaf_count == 0) {
    return;
  }
  const int blocks = procedural_blocks(leaf_count, lanes_per_leaf);
  switch (order) {
#define CDFMM_PROCEDURAL_L2P_CASE(P)                                         \
  case P:                                                                    \
    procedural_l2p_kernel<P, Scalar, Vector>                                 \
        <<<blocks, procedural_threads, 0, stream>>>(                         \
            leaves, leaf_count, lanes_per_leaf, displacements, locals,       \
            factors, fields);                                                \
    break;
    CDFMM_PROCEDURAL_L2P_CASE(1)
    CDFMM_PROCEDURAL_L2P_CASE(2)
    CDFMM_PROCEDURAL_L2P_CASE(3)
    CDFMM_PROCEDURAL_L2P_CASE(4)
    CDFMM_PROCEDURAL_L2P_CASE(5)
    CDFMM_PROCEDURAL_L2P_CASE(6)
    CDFMM_PROCEDURAL_L2P_CASE(7)
    CDFMM_PROCEDURAL_L2P_CASE(8)
    CDFMM_PROCEDURAL_L2P_CASE(9)
    CDFMM_PROCEDURAL_L2P_CASE(10)
#undef CDFMM_PROCEDURAL_L2P_CASE
  default:
    throw std::logic_error(
        "procedural CUDA L2P order is outside the compiled range");
  }
  cuda_detail::check_cuda(cudaGetLastError(),
                          "launch procedural CUDA L2P kernel");
}

} // namespace

} // namespace cdfmm::cuda_far_field_detail
