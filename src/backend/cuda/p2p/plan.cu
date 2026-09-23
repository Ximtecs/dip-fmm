// SPDX-License-Identifier: Apache-2.0
//
// CUDA execution of the exact near field: device views, upload routines and
// kernels for every P2P packing, plus the hybrid backend's `CudaP2PPlan` that
// owns them on a private stream.  All kernels apply the same operator,
// H_i += sum_j T_ij m_j, and differ only in where T_ij comes from:
//
//   canonical rows    one thread per target walks its CSR row of blocks
//   compact rows      the same with SoA components
//   leaf blocks       one warp per dense (target leaf, source leaf) block,
//                     lanes = targets x source slots, shuffle reduction,
//                     atomic accumulation across blocks of one target leaf
//   point geometry    the leaf-block lane layout over list-1 records, but
//                     the pair tensor is recomputed from the resident
//                     positions with the shared point-dipole formula
//   signed dictionary one-, two- or four-byte tokens into a packed dictionary
//                     of already-signed tensors, with three executors chosen
//                     by the execution policy: source-warp (one block per
//                     32-target tile, warps split the sources), target-owned
//                     (one thread per target) and power-of-two microtiles
//                     (one warp per T-target tile, T from 32 down to 1)
//   BSR(3)            cuSPARSE SpMV over 3x3 blocks with identity baked in
//
// Identity semantics are those of the packing: the executors that receive
// `self_indices` skip a pair only when its block carries `skip_for_identity`;
// the dictionary and BSR packings encode the fixed map at construction.  The
// device copy of a leaf-style packing is source-major inside a block so that
// one tensor component of consecutive targets is a contiguous load.  All
// uploads happen once at construction; evaluations only launch kernels.

#include "cdfmm/backend/cuda/p2p.hpp"
#include "backend/cuda/p2p/internal.hpp"
#include "backend/cuda/common/diagnostic_event.hpp"
#include "backend/cuda/common/error.hpp"
#include "operators/p2p_point_kernel.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cdfmm {
namespace cuda_p2p_detail {

using cuda_detail::check_cuda;

void check_cusparse(const cusparseStatus_t status, const char *operation) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string(operation) + ": " + cusparseGetErrorString(status));
  }
}

// Canonical AoS rows: one thread per target, one block record per pair.
// Assigns rather than accumulates, so no clear pass is needed.
template <typename Block, typename Vector>
__global__ void static_p2p_kernel(const int target_count,
                                  const int *row_offsets,
                                  const Block *blocks,
                                  const Vector *moments,
                                  const int *self_indices,
                                  Vector *fields) {
  const int target = blockIdx.x * blockDim.x + threadIdx.x;
  if (target >= target_count) {
    return;
  }
  Vector field{};
  const int self = self_indices[target];
  for (int entry = row_offsets[target]; entry < row_offsets[target + 1];
       ++entry) {
    const Block tensor = blocks[entry];
    if (tensor.skip_for_identity != 0 && tensor.source == self) {
      continue;
    }
    const Vector moment = moments[tensor.source];
    accumulate_static_dipole_block(tensor, moment, field);
  }
  fields[target] = field;
}

template <typename Block, typename Vector>
void launch_static_p2p(const CudaP2PDeviceView<Block> &plan,
                       const Vector *moments,
                       const int *self_indices,
                       Vector *fields,
                       cudaStream_t stream) {
  if (plan.target_count == 0) {
    return;
  }
  static_p2p_kernel<<<
      (plan.target_count + static_operator_threads - 1) /
          static_operator_threads,
      static_operator_threads, 0, stream>>>(
      plan.target_count, plan.row_offsets, plan.blocks, moments, self_indices,
      fields);
  check_cuda(cudaGetLastError(), "launch canonical static P2P kernel");
}

// Compact (SoA) rows: the six components live in separate planes of
// `interaction_count` values each.
template <typename Scalar, typename Vector>
__global__ void compact_p2p_kernel(
    const int target_count,
    const int* row_offsets,
    const int* source_indices,
    const unsigned char* skip_for_identity,
    const Scalar* tensors,
    const std::size_t interaction_count,
    const Vector* moments,
    const int* self_indices,
    Vector* fields)
{
  const int target = blockIdx.x * blockDim.x + threadIdx.x;
  if (target >= target_count) {
    return;
  }

  Vector field{};
  const int self = self_indices[target];
  for (int entry = row_offsets[target]; entry < row_offsets[target + 1];
       ++entry) {
    const int source = source_indices[entry];
    if (skip_for_identity[entry] != 0 && source == self) {
      continue;
    }
    const std::size_t index = static_cast<std::size_t>(entry);
    const Vector moment = moments[source];
    const Scalar xx = tensors[index];
    const Scalar xy = tensors[interaction_count + index];
    const Scalar xz = tensors[2 * interaction_count + index];
    const Scalar yy = tensors[3 * interaction_count + index];
    const Scalar yz = tensors[4 * interaction_count + index];
    const Scalar zz = tensors[5 * interaction_count + index];
    field.x += xx * moment.x + xy * moment.y + xz * moment.z;
    field.y += xy * moment.x + yy * moment.y + yz * moment.z;
    field.z += xz * moment.x + yz * moment.y + zz * moment.z;
  }
  fields[target] = field;
}

template <typename Scalar, typename Vector>
void launch_compact_p2p(
    const CudaCompactP2PDeviceView<Scalar>& plan,
    const Vector* moments,
    const int* self_indices,
    Vector* fields,
    cudaStream_t stream)
{
  if (plan.target_count == 0) {
    return;
  }
  compact_p2p_kernel<<<
      (plan.target_count + static_operator_threads - 1) /
          static_operator_threads,
      static_operator_threads, 0, stream>>>(
      plan.target_count, plan.row_offsets, plan.source_indices,
      plan.skip_for_identity, plan.tensors, plan.interaction_count, moments,
      self_indices, fields);
  check_cuda(cudaGetLastError(), "launch compact static P2P kernel");
}

constexpr int leaf_p2p_threads = 128;

// One warp per dense (target leaf, source leaf) block. Lanes are laid out as
// `stride` consecutive targets (the smallest power of two covering the leaf,
// at most 32) times `32 / stride` source slots, so every tensor component
// load is a contiguous run of `target_count` values (the CUDA copy of the
// leaf plan is source-major inside a block) and the moment of one source is
// uniform across a lane group. Source-slot partial fields are combined with
// shuffles and the block's contribution is added atomically, because the
// other source leaves of the same target leaf run in other warps. The number
// of warps in flight is the number of dense blocks, which keeps the memory
// system busy even for trees with a few particles per leaf.
template <typename Scalar, typename Vector>
__global__ void __launch_bounds__(leaf_p2p_threads) leaf_p2p_kernel(
    const int *__restrict__ block_target_leaves,
    const int *__restrict__ target_begins,
    const int *__restrict__ target_counts,
    const StaticP2PLeafBlock *__restrict__ leaf_blocks,
    const Scalar *__restrict__ tensors, const std::size_t interaction_count,
    const int block_count, const Vector *__restrict__ moments,
    const int *__restrict__ self_indices, Vector *__restrict__ fields) {
  const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int lane = static_cast<int>(threadIdx.x & 31);
  if (warp >= block_count) {
    return;
  }
  const int target_leaf = block_target_leaves[warp];
  const int target_begin = target_begins[target_leaf];
  const int target_count = target_counts[target_leaf];
  const StaticP2PLeafBlock block = leaf_blocks[warp];

  int stride = 32;
  while (stride > 1 && (stride >> 1) >= target_count) {
    stride >>= 1;
  }
  const int source_slots = 32 / stride;
  const int source_slot = lane / stride;
  const int target_slot = lane - source_slot * stride;

  for (int target_base = 0; target_base < target_count;
       target_base += stride) {
    const int local_target = target_base + target_slot;
    const bool active = local_target < target_count;
    const int target = target_begin + (active ? local_target : 0);
    const int self = active ? self_indices[target] : -1;
    Scalar hx = Scalar{0};
    Scalar hy = Scalar{0};
    Scalar hz = Scalar{0};
    for (int local_source = source_slot; local_source < block.source_count;
         local_source += source_slots) {
      const int source = block.source_begin + local_source;
      if (!active || (block.skip_for_identity != 0 && source == self)) {
        continue;
      }
      const Vector moment = moments[source];
      const std::size_t index = block.tensor_offset +
          static_cast<std::size_t>(local_source) * target_count + local_target;
      const Scalar xx = tensors[index];
      const Scalar xy = tensors[interaction_count + index];
      const Scalar xz = tensors[2 * interaction_count + index];
      const Scalar yy = tensors[3 * interaction_count + index];
      const Scalar yz = tensors[4 * interaction_count + index];
      const Scalar zz = tensors[5 * interaction_count + index];
      hx += xx * moment.x + xy * moment.y + xz * moment.z;
      hy += xy * moment.x + yy * moment.y + yz * moment.z;
      hz += xz * moment.x + yz * moment.y + zz * moment.z;
    }
    for (int offset = stride; offset < 32; offset <<= 1) {
      hx += __shfl_xor_sync(0xffffffffU, hx, offset);
      hy += __shfl_xor_sync(0xffffffffU, hy, offset);
      hz += __shfl_xor_sync(0xffffffffU, hz, offset);
    }
    if (active && source_slot == 0) {
      atomicAdd(&fields[target].x, hx);
      atomicAdd(&fields[target].y, hy);
      atomicAdd(&fields[target].z, hz);
    }
  }
}

constexpr int point_geometry_p2p_threads = 128;

// Position-based point P2P: one warp per canonical list-1 record (target
// leaf, source leaf, image) with the leaf-block lane layout, `stride`
// consecutive targets times `32 / stride` source slots. Instead of six stored
// tensor components a lane reads one aligned position and one moment per
// source, both uniform across a lane group, and recomputes the point-dipole
// pair from the displacement (plus the record's image shift) with the shared
// formula of `operators/p2p_point_kernel.hpp`. The self pair is removed
// without control flow (unit displacement, zero weight) exactly like the CPU
// executor, so the source loop stays uniform. Source-slot partial fields are
// combined with shuffles and the record's contribution is added atomically,
// because the other records of the same target leaf run in other warps.
template <typename Scalar, typename Vector>
__global__ void __launch_bounds__(point_geometry_p2p_threads)
point_geometry_p2p_kernel(
    const CudaPointGeometryP2PRecord<Scalar> *__restrict__ records,
    const int record_count,
    const typename CudaScalar4<Scalar>::type *__restrict__ source_positions,
    const typename CudaScalar4<Scalar>::type *__restrict__ target_positions,
    const Vector *__restrict__ moments, const int *__restrict__ self_indices,
    Vector *__restrict__ fields) {
  const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int lane = static_cast<int>(threadIdx.x & 31);
  if (warp >= record_count) {
    return;
  }
  const CudaPointGeometryP2PRecord<Scalar> record = records[warp];

  int stride = 32;
  while (stride > 1 && (stride >> 1) >= record.target_count) {
    stride >>= 1;
  }
  const int source_slots = 32 / stride;
  const int source_slot = lane / stride;
  const int target_slot = lane - source_slot * stride;

  for (int target_base = 0; target_base < record.target_count;
       target_base += stride) {
    const int local_target = target_base + target_slot;
    const bool active = local_target < record.target_count;
    const int target = record.target_begin + (active ? local_target : 0);
    const int self = active ? self_indices[target] : -1;
    const auto target_position = target_positions[target];
    const Scalar tx = target_position.x;
    const Scalar ty = target_position.y;
    const Scalar tz = target_position.z;
    Scalar hx = Scalar{0};
    Scalar hy = Scalar{0};
    Scalar hz = Scalar{0};
    for (int local_source = source_slot; local_source < record.source_count;
         local_source += source_slots) {
      const int source = record.source_begin + local_source;
      const auto source_position = source_positions[source];
      const Vector moment = moments[source];
      const bool excluded =
          !active || (record.skip_for_identity != 0 && source == self);
      const Scalar weight = excluded ? Scalar{0} : Scalar{1};
      const Scalar rx =
          excluded ? Scalar{1} : tx - (source_position.x + record.shift_x);
      const Scalar ry =
          excluded ? Scalar{0} : ty - (source_position.y + record.shift_y);
      const Scalar rz =
          excluded ? Scalar{0} : tz - (source_position.z + record.shift_z);
      Scalar cx = Scalar{0};
      Scalar cy = Scalar{0};
      Scalar cz = Scalar{0};
      operators::p2p::accumulate_point_dipole_field(
          rx, ry, rz, static_cast<Scalar>(moment.x),
          static_cast<Scalar>(moment.y), static_cast<Scalar>(moment.z), cx, cy,
          cz);
      hx += weight * cx;
      hy += weight * cy;
      hz += weight * cz;
    }
    for (int offset = stride; offset < 32; offset <<= 1) {
      hx += __shfl_xor_sync(0xffffffffU, hx, offset);
      hy += __shfl_xor_sync(0xffffffffU, hy, offset);
      hz += __shfl_xor_sync(0xffffffffU, hz, offset);
    }
    if (active && source_slot == 0) {
      atomicAdd(&fields[target].x, hx);
      atomicAdd(&fields[target].y, hy);
      atomicAdd(&fields[target].z, hz);
    }
  }
}

template <typename Scalar, typename Vector>
void launch_point_geometry_p2p_typed(
    const CudaPointGeometryP2PDeviceView<Scalar> &plan, const Vector *moments,
    const int *self_indices, Vector *fields, cudaStream_t stream) {
  if (plan.target_count == 0) {
    return;
  }
  // Records accumulate atomically, so the fields start from zero.
  check_cuda(cudaMemsetAsync(fields, 0,
                             static_cast<std::size_t>(plan.target_count) *
                                 sizeof(Vector),
                             stream),
             "clear position-based P2P fields");
  if (plan.record_count == 0) {
    return;
  }
  constexpr int warps_per_block = point_geometry_p2p_threads / 32;
  point_geometry_p2p_kernel<<<(plan.record_count + warps_per_block - 1) /
                                  warps_per_block,
                              point_geometry_p2p_threads, 0, stream>>>(
      plan.records, plan.record_count, plan.source_positions,
      plan.target_positions, moments, self_indices, fields);
  check_cuda(cudaGetLastError(), "launch position-based point P2P kernel");
}

constexpr int cuda_dictionary_warp_size = 32;
constexpr int cuda_dictionary_target_tile_size = 32;
constexpr int cuda_dictionary_max_source_warps = 8;

constexpr int cuda_dictionary_target_owned_threads = 256;

// The device dictionary stores each signed variant as one 16-byte-aligned
// float4/double4 plus a float2/double2, so a variant is fetched with two
// vector loads instead of six scalar gathers.
static CudaPackedTensor6<float> make_cuda_packed_tensor6(
    const float xx,
    const float xy,
    const float xz,
    const float yy,
    const float yz,
    const float zz) {

  CudaPackedTensor6<float> result{};
  result.a = make_float4(xx, xy, xz, yy);
  result.b = make_float2(yz, zz);
  return result;
}

static CudaPackedTensor6<double> make_cuda_packed_tensor6(
    const double xx,
    const double xy,
    const double xz,
    const double yy,
    const double yz,
    const double zz) {

  CudaPackedTensor6<double> result{};
#if CUDART_VERSION >= 13000
  result.a = make_double4_16a(xx, xy, xz, yy);
#else
  result.a = make_double4(xx, xy, xz, yy);
#endif
  result.b = make_double2(yz, zz);
  return result;
}

// Work list of the source-warp executor: one entry per (leaf, 32-target tile).
static void build_cuda_dictionary_tiles(
    const std::span<const int> target_counts, const int target_tile_size,
    std::vector<int> &tile_leaf_indices,
    std::vector<int> &tile_target_offsets) {
  if (target_tile_size <= 0) {
    throw std::invalid_argument(
        "CUDA signed dictionary target tile must be positive");
  }
  tile_leaf_indices.clear();
  tile_target_offsets.clear();
  for (int leaf = 0; leaf < static_cast<int>(target_counts.size()); ++leaf) {
    const int target_count = target_counts[static_cast<std::size_t>(leaf)];
    for (int target_offset = 0; target_offset < target_count;
         target_offset += target_tile_size) {
      tile_leaf_indices.push_back(leaf);
      tile_target_offsets.push_back(target_offset);
    }
  }
}

/** @brief Host schedule for power-of-two target microtiles. */
struct CudaDictionaryMicrotileSchedule {
  std::vector<int> leaf_indices{};
  std::vector<int> target_offsets{};
  // [begin32, begin16, begin8, begin4, begin2, begin1, end].
  std::array<int, 7> class_offsets{};
};

/**
 * @brief Decomposes each leaf's target range into power-of-two microtiles.
 *
 * Tiles are emitted in width-class order so each class can use one compile-time
 * warp layout. Within a leaf, greedy descending widths produce an exact binary
 * decomposition and cover every target exactly once.
 */
static CudaDictionaryMicrotileSchedule build_cuda_dictionary_microtiles(
    const std::span<const int> target_counts) {
  constexpr std::array<int, 6> widths{32, 16, 8, 4, 2, 1};
  std::array<std::vector<int>, widths.size()> leaves;
  std::array<std::vector<int>, widths.size()> offsets;

  for (int leaf = 0; leaf < static_cast<int>(target_counts.size()); ++leaf) {
    int target_offset = 0;
    int remaining = target_counts[static_cast<std::size_t>(leaf)];
    for (std::size_t class_index = 0; class_index < widths.size();
         ++class_index) {
      const int width = widths[class_index];
      while (remaining >= width) {
        leaves[class_index].push_back(leaf);
        offsets[class_index].push_back(target_offset);
        target_offset += width;
        remaining -= width;
      }
    }
    if (remaining != 0) {
      throw std::logic_error(
          "power-of-two CUDA dictionary decomposition failed");
    }
  }

  CudaDictionaryMicrotileSchedule result;
  result.class_offsets[0] = 0;
  for (std::size_t class_index = 0; class_index < widths.size();
       ++class_index) {
    result.leaf_indices.insert(result.leaf_indices.end(),
                               leaves[class_index].begin(),
                               leaves[class_index].end());
    result.target_offsets.insert(result.target_offsets.end(),
                                 offsets[class_index].begin(),
                                 offsets[class_index].end());
    result.class_offsets[class_index + 1] =
        static_cast<int>(result.leaf_indices.size());
  }
  return result;
}

template <typename Scalar, typename Vector>
void launch_leaf_p2p(
    const CudaLeafP2PDeviceView<Scalar>& plan,
    const Vector* moments,
    const int* self_indices,
    Vector* fields,
    cudaStream_t stream)
{
  if (plan.target_count == 0) {
    return;
  }
  // Blocks accumulate atomically, so the fields start from zero.
  check_cuda(
      cudaMemsetAsync(
          fields, 0, static_cast<std::size_t>(plan.target_count) * sizeof(Vector),
          stream),
      "clear leaf P2P fields");
  if (plan.block_count == 0) {
    return;
  }
  constexpr int warps_per_block = leaf_p2p_threads / 32;
  leaf_p2p_kernel<<<(plan.block_count + warps_per_block - 1) / warps_per_block,
                    leaf_p2p_threads, 0, stream>>>(
      plan.block_target_leaves, plan.target_begins, plan.target_counts,
      plan.leaf_blocks, plan.tensors, plan.interaction_count, plan.block_count,
      moments, self_indices, fields);
  check_cuda(cudaGetLastError(), "launch leaf-block static P2P kernel");
}

// Source-warp dictionary executor: one block per 32-target tile of a leaf.
// Every warp of the block holds the same 32 targets in its lanes and takes a
// different 32-source slice of each source leaf; a source's moment is loaded
// once per warp and broadcast with shuffles while the 32 lanes read their
// consecutive source-major tokens.  With more than one warp the per-warp
// partial fields are reduced through shared memory.  It is the executor of
// choice at high leaf occupancy, where the source slices keep every warp
// busy; the two executors below serve lower occupancies.
template <typename Scalar, typename Vector, typename Token>
__global__ void signed_dictionary_p2p_kernel(
    const int *__restrict__ target_begins,
    const int *__restrict__ target_counts,
    const int *__restrict__ leaf_row_offsets,
    const StaticP2PLeafBlock *__restrict__ leaf_blocks,
    const int *__restrict__ tile_leaf_indices,
    const int *__restrict__ tile_target_offsets,
    const Token *__restrict__ tokens,
    const CudaPackedTensor6<Scalar> *__restrict__ tensors,
    const Vector *__restrict__ moments,
    Vector *__restrict__ fields) {

  constexpr unsigned full_warp_mask = 0xffffffffu;

  const int lane =
      static_cast<int>(threadIdx.x) &
      (cuda_dictionary_warp_size - 1);

  const int warp =
      static_cast<int>(threadIdx.x) /
      cuda_dictionary_warp_size;

  const int warp_count =
      static_cast<int>(blockDim.x) /
      cuda_dictionary_warp_size;

  /*
   * Every warp works on the same 32 target lanes.
   * Different warps own different subsets of the sources.
   */
  const int work =
      static_cast<int>(blockIdx.x);

  const int target_leaf =
      tile_leaf_indices[work];

  const int tile_offset =
      tile_target_offsets[work];

  const int target_begin =
      target_begins[target_leaf];

  const int target_count =
      target_counts[target_leaf];

  const int local_target =
      tile_offset + lane;

  const bool active_target =
      local_target < target_count;

  const int target =
      target_begin + local_target;

  Scalar Hx = Scalar{0};
  Scalar Hy = Scalar{0};
  Scalar Hz = Scalar{0};

  /*
   * Each source group contains 32 sources per participating warp.
   */
  const int sources_per_group =
      warp_count * cuda_dictionary_warp_size;

  for (int block_index = leaf_row_offsets[target_leaf];
       block_index < leaf_row_offsets[target_leaf + 1];
       ++block_index) {

    const StaticP2PLeafBlock block =
        leaf_blocks[block_index];

    for (int group_base = 0;
         group_base < block.source_count;
         group_base += sources_per_group) {

      const int warp_source_base =
          group_base +
          warp * cuda_dictionary_warp_size;

      const int remaining =
          block.source_count - warp_source_base;

      const int warp_source_count =
          remaining <= 0
              ? 0
              : min(cuda_dictionary_warp_size, remaining);

      /*
       * Each lane loads one source moment.
       * The warp therefore performs a coalesced 32-moment global load.
       */
      Scalar lane_mx = Scalar{0};
      Scalar lane_my = Scalar{0};
      Scalar lane_mz = Scalar{0};

      if (lane < warp_source_count) {
        const Vector moment =
            moments[
                block.source_begin +
                warp_source_base +
                lane];

        lane_mx = static_cast<Scalar>(moment.x);
        lane_my = static_cast<Scalar>(moment.y);
        lane_mz = static_cast<Scalar>(moment.z);
      }

      /*
       * Broadcast each source through warp shuffles.
       * For a given source, the 32 lanes read consecutive source-major tokens.
       */
      for (int source_lane = 0;
           source_lane < warp_source_count;
           ++source_lane) {

        const Scalar mx =
            __shfl_sync(
                full_warp_mask,
                lane_mx,
                source_lane);

        const Scalar my =
            __shfl_sync(
                full_warp_mask,
                lane_my,
                source_lane);

        const Scalar mz =
            __shfl_sync(
                full_warp_mask,
                lane_mz,
                source_lane);

        if (active_target) {
          const int local_source =
              warp_source_base + source_lane;

          const std::size_t token_index =
              block.tensor_offset +
              static_cast<std::size_t>(local_source) *
                  static_cast<std::size_t>(target_count) +
              static_cast<std::size_t>(local_target);

          const std::size_t variant =
              static_cast<std::size_t>(
                  tokens[token_index]);

          const CudaPackedTensor6<Scalar> tensor =
              tensors[variant];

          const Scalar xx = tensor.a.x;
          const Scalar xy = tensor.a.y;
          const Scalar xz = tensor.a.z;
          const Scalar yy = tensor.a.w;
          const Scalar yz = tensor.b.x;
          const Scalar zz = tensor.b.y;

          Hx += xx * mx + xy * my + xz * mz;
          Hy += xy * mx + yy * my + yz * mz;
          Hz += xz * mx + yz * my + zz * mz;
        }
      }
    }
  }

  /*
   * With one source warp, no cross-warp reduction is required.
   */
  if (warp_count == 1) {
    if (active_target) {
      fields[target].x = Hx;
      fields[target].y = Hy;
      fields[target].z = Hz;
    }
    return;
  }

  /*
   * Store one partial H per source warp / target lane.
   */
  extern __shared__ unsigned char shared_storage[];

  Scalar *partial_hx =
      reinterpret_cast<Scalar *>(shared_storage);

  Scalar *partial_hy =
      partial_hx + blockDim.x;

  Scalar *partial_hz =
      partial_hy + blockDim.x;

  const int thread =
      static_cast<int>(threadIdx.x);

  partial_hx[thread] = Hx;
  partial_hy[thread] = Hy;
  partial_hz[thread] = Hz;

  __syncthreads();

  /*
   * Warp 0 reduces the source-warp partial sums for its corresponding
   * 32 target lanes.
   */
  if (warp == 0 && active_target) {
    Scalar total_x = Scalar{0};
    Scalar total_y = Scalar{0};
    Scalar total_z = Scalar{0};

#pragma unroll
    for (int source_warp = 0;
         source_warp < cuda_dictionary_max_source_warps;
         ++source_warp) {

      if (source_warp < warp_count) {
        const int index =
            source_warp *
                cuda_dictionary_warp_size +
            lane;

        total_x += partial_hx[index];
        total_y += partial_hy[index];
        total_z += partial_hz[index];
      }
    }

    fields[target].x = total_x;
    fields[target].y = total_y;
    fields[target].z = total_z;
  }
}

// Target-owned dictionary executor: one thread per target walks every source
// of every block in its leaf row.  Consecutive threads are consecutive
// targets of one leaf, so their token reads for a given source are
// contiguous; no reduction or atomics are needed.
template <typename Scalar, typename Vector, typename Token>
__global__ void signed_dictionary_target_owned_p2p_kernel(
    const int target_count,
    const int *__restrict__ target_begins,
    const int *__restrict__ target_counts,
    const int *__restrict__ target_leaf_for_target,
    const int *__restrict__ leaf_row_offsets,
    const StaticP2PLeafBlock *__restrict__ leaf_blocks,
    const Token *__restrict__ tokens,
    const CudaPackedTensor6<Scalar> *__restrict__ tensors,
    const Vector *__restrict__ moments,
    Vector *__restrict__ fields) {

  const int target =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

  if (target >= target_count) {
    return;
  }

  const int target_leaf =
      target_leaf_for_target[target];

  const int target_begin =
      target_begins[target_leaf];

  const int leaf_target_count =
      target_counts[target_leaf];

  const int local_target =
      target - target_begin;

  Scalar Hx = Scalar{0};
  Scalar Hy = Scalar{0};
  Scalar Hz = Scalar{0};

  for (int block_index = leaf_row_offsets[target_leaf];
       block_index < leaf_row_offsets[target_leaf + 1];
       ++block_index) {

    const StaticP2PLeafBlock block =
        leaf_blocks[block_index];

    std::size_t token_index =
        block.tensor_offset +
        static_cast<std::size_t>(local_target);

    for (int local_source = 0;
         local_source < block.source_count;
         ++local_source) {

      const std::size_t variant =
          static_cast<std::size_t>(tokens[token_index]);

      const CudaPackedTensor6<Scalar> tensor =
          tensors[variant];

      const Vector moment =
          moments[block.source_begin + local_source];

      const Scalar mx = static_cast<Scalar>(moment.x);
      const Scalar my = static_cast<Scalar>(moment.y);
      const Scalar mz = static_cast<Scalar>(moment.z);

      const Scalar xx = tensor.a.x;
      const Scalar xy = tensor.a.y;
      const Scalar xz = tensor.a.z;
      const Scalar yy = tensor.a.w;
      const Scalar yz = tensor.b.x;
      const Scalar zz = tensor.b.y;

      Hx += xx * mx + xy * my + xz * mz;
      Hy += xy * mx + yy * my + yz * mz;
      Hz += xz * mx + yz * my + zz * mz;

      token_index +=
          static_cast<std::size_t>(leaf_target_count);
    }
  }

  fields[target].x = Hx;
  fields[target].y = Hy;
  fields[target].z = Hz;
}

/**
 * @brief Executes one power-of-two target microtile per warp.
 *
 * A width-T microtile assigns T lanes to each target and reuses each source
 * moment across those T lanes. The source-major token layout remains unchanged
 * from the source-warp dictionary executor.
 */
template <int T, typename Scalar, typename Vector, typename Token>
__global__ void signed_dictionary_microtile_p2p_kernel(
    const int *__restrict__ target_begins,
    const int *__restrict__ target_counts,
    const int *__restrict__ leaf_row_offsets,
    const StaticP2PLeafBlock *__restrict__ leaf_blocks,
    const int *__restrict__ work_leaf_indices,
    const int *__restrict__ work_target_offsets,
    const int work_count,
    const Token *__restrict__ tokens,
    const CudaPackedTensor6<Scalar> *__restrict__ tensors,
    const Vector *__restrict__ moments,
    Vector *__restrict__ fields) {
  static_assert(T == 1 || T == 2 || T == 4 || T == 8 || T == 16 || T == 32);
  constexpr int warp_size = cuda_dictionary_warp_size;
  constexpr int source_lanes = warp_size / T;
  constexpr unsigned full_mask = 0xffffffffu;

  const int lane = static_cast<int>(threadIdx.x) & (warp_size - 1);
  const int warp_in_block = static_cast<int>(threadIdx.x) >> 5;
  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
  const int work = static_cast<int>(blockIdx.x) * warps_per_block +
                  warp_in_block;
  if (work >= work_count) {
    return;
  }

  const int target_leaf = work_leaf_indices[work];
  const int microtile_offset = work_target_offsets[work];
  const int target_begin = target_begins[target_leaf];
  const int target_count = target_counts[target_leaf];
  const int target_slot = lane % T;
  const int source_slot = lane / T;
  const int local_target = microtile_offset + target_slot;
  const bool target_active = local_target < target_count;
  const int target = target_begin + local_target;

  Scalar Hx = Scalar{0};
  Scalar Hy = Scalar{0};
  Scalar Hz = Scalar{0};
  const int row_begin = leaf_row_offsets[target_leaf];
  const int row_end = leaf_row_offsets[target_leaf + 1];

  for (int block_index = row_begin; block_index < row_end; ++block_index) {
    const StaticP2PLeafBlock block = leaf_blocks[block_index];
    for (int source_base = 0; source_base < block.source_count;
         source_base += source_lanes) {
      const int local_source = source_base + source_slot;
      const bool source_active = local_source < block.source_count;
      Scalar mx = Scalar{0};
      Scalar my = Scalar{0};
      Scalar mz = Scalar{0};
      const int subgroup_leader = source_slot * T;
      if (target_slot == 0 && source_active) {
        const Vector moment = moments[block.source_begin + local_source];
        mx = static_cast<Scalar>(moment.x);
        my = static_cast<Scalar>(moment.y);
        mz = static_cast<Scalar>(moment.z);
      }

      // Every lane participates in the full-mask shuffle. Inactive source
      // slots retain zero moments, while inactive target slots retain zero
      // partial fields; this keeps the mask valid for partial source groups.
      mx = __shfl_sync(full_mask, mx, subgroup_leader);
      my = __shfl_sync(full_mask, my, subgroup_leader);
      mz = __shfl_sync(full_mask, mz, subgroup_leader);

      if (source_active && target_active) {
        const std::size_t token_index =
            block.tensor_offset +
            static_cast<std::size_t>(local_source) *
                static_cast<std::size_t>(target_count) +
            static_cast<std::size_t>(local_target);
        const std::size_t variant = static_cast<std::size_t>(
            tokens[token_index]);
        const CudaPackedTensor6<Scalar> tensor = tensors[variant];
        const Scalar xx = tensor.a.x;
        const Scalar xy = tensor.a.y;
        const Scalar xz = tensor.a.z;
        const Scalar yy = tensor.a.w;
        const Scalar yz = tensor.b.x;
        const Scalar zz = tensor.b.y;
        Hx += xx * mx + xy * my + xz * mz;
        Hy += xy * mx + yy * my + yz * mz;
        Hz += xz * mx + yz * my + zz * mz;
      }
    }
  }

  // Reduce source-slot partials within each T-lane target subgroup.
  for (int delta = T; delta < warp_size; delta *= 2) {
    Hx += __shfl_xor_sync(full_mask, Hx, delta);
    Hy += __shfl_xor_sync(full_mask, Hy, delta);
    Hz += __shfl_xor_sync(full_mask, Hz, delta);
  }

  // The schedule is an exact decomposition, so every active target is owned
  // by one microtile. Assignment is therefore race-free and starts fresh.
  if (source_slot == 0 && target_active) {
    fields[target].x = Hx;
    fields[target].y = Hy;
    fields[target].z = Hz;
  }
}

template <int T, typename Scalar, typename Vector, typename Token>
void launch_dictionary_microtile_class(
    const CudaSignedDictionaryP2PDeviceView<Scalar> &plan,
    const int class_index, const Vector *moments, Vector *fields,
    cudaStream_t stream) {
  constexpr int threads = 128;
  constexpr int warps_per_block = threads / cuda_dictionary_warp_size;
  const int begin = plan.microtile_class_offsets[
      static_cast<std::size_t>(class_index)];
  const int end = plan.microtile_class_offsets[
      static_cast<std::size_t>(class_index + 1)];
  const int count = end - begin;
  if (count == 0) {
    return;
  }
  const int blocks = (count + warps_per_block - 1) / warps_per_block;
  signed_dictionary_microtile_p2p_kernel<T, Scalar, Vector, Token>
      <<<blocks, threads, 0, stream>>>(
          plan.target_begins, plan.target_counts, plan.leaf_row_offsets,
          plan.leaf_blocks, plan.microtile_leaf_indices + begin,
          plan.microtile_target_offsets + begin, count,
          static_cast<const Token *>(plan.tokens), plan.tensors, moments,
          fields);
  check_cuda(cudaGetLastError(),
             "launch power-of-two dictionary P2P kernel");
}

template <typename Scalar, typename Vector, typename Token>
void launch_dictionary_power2_microtiles(
    const CudaSignedDictionaryP2PDeviceView<Scalar> &plan,
    const Vector *moments, Vector *fields, cudaStream_t stream) {
  launch_dictionary_microtile_class<32, Scalar, Vector, Token>(
      plan, 0, moments, fields, stream);
  launch_dictionary_microtile_class<16, Scalar, Vector, Token>(
      plan, 1, moments, fields, stream);
  launch_dictionary_microtile_class<8, Scalar, Vector, Token>(
      plan, 2, moments, fields, stream);
  launch_dictionary_microtile_class<4, Scalar, Vector, Token>(
      plan, 3, moments, fields, stream);
  launch_dictionary_microtile_class<2, Scalar, Vector, Token>(
      plan, 4, moments, fields, stream);
  launch_dictionary_microtile_class<1, Scalar, Vector, Token>(
      plan, 5, moments, fields, stream);
}

// Dispatch to the executor the policy selected at upload.  All three assign
// every target exactly once, so none needs a clear pass.
template <typename Scalar, typename Vector, typename Token>
void launch_signed_dictionary_p2p_typed(
    const CudaSignedDictionaryP2PDeviceView<Scalar> &plan,
    const Vector *moments,
    Vector *fields,
    cudaStream_t stream) {

  if (plan.target_count == 0) {
    return;
  }

  if (plan.target_owned) {
    const int block_count =
        (plan.target_count +
         cuda_dictionary_target_owned_threads - 1) /
        cuda_dictionary_target_owned_threads;

    signed_dictionary_target_owned_p2p_kernel<
        Scalar, Vector, Token>
        <<<block_count,
           cuda_dictionary_target_owned_threads,
           0,
           stream>>>(
            plan.target_count,
            plan.target_begins,
            plan.target_counts,
            plan.target_leaf_for_target,
            plan.leaf_row_offsets,
            plan.leaf_blocks,
            static_cast<const Token *>(plan.tokens),
            plan.tensors,
            moments,
            fields);

    check_cuda(
        cudaGetLastError(),
        "launch target-owned signed tensor-dictionary P2P kernel");

    return;
  }

  if (plan.power2_microtiles) {
    launch_dictionary_power2_microtiles<Scalar, Vector, Token>(
        plan, moments, fields, stream);
    return;
  }

  if (plan.tile_count == 0) {
    return;
  }

  const std::size_t shared_bytes =
      plan.threads_per_block == cuda_dictionary_warp_size
          ? 0
          : 3 *
                static_cast<std::size_t>(
                    plan.threads_per_block) *
                sizeof(Scalar);

  signed_dictionary_p2p_kernel<Scalar, Vector, Token>
      <<<plan.tile_count,
         plan.threads_per_block,
         shared_bytes,
         stream>>>(
          plan.target_begins,
          plan.target_counts,
          plan.leaf_row_offsets,
          plan.leaf_blocks,
          plan.tile_leaf_indices,
          plan.tile_target_offsets,
          static_cast<const Token *>(plan.tokens),
          plan.tensors,
          moments,
          fields);

  check_cuda(
      cudaGetLastError(),
      "launch source-parallel signed tensor-dictionary P2P kernel");
}

template <typename Scalar, typename Vector>
void launch_signed_dictionary_p2p(
    const CudaSignedDictionaryP2PDeviceView<Scalar> &plan,
    const Vector *moments, Vector *fields, cudaStream_t stream) {
  switch (plan.token_width_bytes) {
  case 1:
    launch_signed_dictionary_p2p_typed<Scalar, Vector, std::uint8_t>(
        plan, moments, fields, stream);
    break;
  case 2:
    launch_signed_dictionary_p2p_typed<Scalar, Vector, std::uint16_t>(
        plan, moments, fields, stream);
    break;
  case 4:
    launch_signed_dictionary_p2p_typed<Scalar, Vector, std::uint32_t>(
        plan, moments, fields, stream);
    break;
  default:
    throw std::invalid_argument(
        "invalid CUDA tensor-dictionary token width");
  }
}

// Upload one signed dictionary and prepare the work lists of all three
// executors (the choice is fixed here, but the schedules are cheap and the
// statistics report them).  The source-warp block size is the smallest
// number of warps that covers the largest source leaf, capped at eight.
template <typename Scalar, typename Vector, typename HostPlan>
void upload_cuda_signed_dictionary(
    const HostPlan &host,
    CudaSignedDictionaryP2PDeviceView<Scalar> &device,
    CudaPlanStatistics &statistics,
    const bool target_owned,
    const bool power2_microtiles) {

  device.target_count = host.target_count;
  device.variant_count = host.variant_count();
  device.token_width_bytes = host.token_width_bytes;
  device.target_owned = target_owned;
  device.power2_microtiles = power2_microtiles;

  std::vector<int> tile_leaf_indices;
  std::vector<int> tile_target_offsets;
  const CudaDictionaryMicrotileSchedule microtile_schedule =
      build_cuda_dictionary_microtiles(
          std::span<const int>(host.target_counts));
  std::vector<int> target_leaf_for_target;

  if (host.target_begins.size() != host.target_counts.size()) {
    throw std::invalid_argument(
        "CUDA signed dictionary target metadata are inconsistent");
  }

  build_cuda_dictionary_tiles(
      std::span<const int>(host.target_counts),
      cuda_dictionary_target_tile_size,
      tile_leaf_indices,
      tile_target_offsets);

  device.tile_count = static_cast<int>(tile_leaf_indices.size());
  device.microtile_count =
      static_cast<int>(microtile_schedule.leaf_indices.size());
  device.microtile_class_offsets = microtile_schedule.class_offsets;

  if (target_owned) {
    device.threads_per_block = cuda_dictionary_target_owned_threads;

    target_leaf_for_target.assign(
        static_cast<std::size_t>(host.target_count),
        -1);

    for (int leaf = 0;
         leaf < static_cast<int>(host.target_begins.size());
         ++leaf) {

      const int begin =
          host.target_begins[static_cast<std::size_t>(leaf)];

      const int count =
          host.target_counts[static_cast<std::size_t>(leaf)];

      if (begin < 0 ||
          count < 0 ||
          begin + count > host.target_count) {
        throw std::invalid_argument(
            "CUDA signed dictionary target leaf bounds are invalid");
      }

      for (int local_target = 0;
           local_target < count;
           ++local_target) {

        const int target = begin + local_target;

        int &mapped_leaf =
            target_leaf_for_target[
                static_cast<std::size_t>(target)];

        if (mapped_leaf != -1) {
          throw std::invalid_argument(
              "CUDA dictionary target appears in multiple leaves");
        }

        mapped_leaf = leaf;
      }
    }

    if (std::find(
            target_leaf_for_target.begin(),
            target_leaf_for_target.end(),
            -1) != target_leaf_for_target.end()) {
      throw std::invalid_argument(
          "CUDA signed dictionary target leaf map is incomplete");
    }
  } else if (power2_microtiles) {
    device.threads_per_block = 128;
  } else {
    int maximum_source_count = 0;

    for (const StaticP2PLeafBlock &block : host.blocks) {
      maximum_source_count =
          std::max(maximum_source_count, block.source_count);
    }

    int source_warps = 1;

    while (source_warps < cuda_dictionary_max_source_warps &&
           source_warps * cuda_dictionary_warp_size <
               maximum_source_count) {
      source_warps *= 2;
    }

    device.threads_per_block =
        source_warps * cuda_dictionary_warp_size;
  }

  const void *token_source = nullptr;
  std::size_t token_bytes = 0;

  switch (host.token_width_bytes) {
  case 1:
    token_source = host.tokens8.data();
    token_bytes =
        host.tokens8.size() * sizeof(std::uint8_t);
    break;

  case 2:
    token_source = host.tokens16.data();
    token_bytes =
        host.tokens16.size() * sizeof(std::uint16_t);
    break;

  case 4:
    token_source = host.tokens32.data();
    token_bytes =
        host.tokens32.size() * sizeof(std::uint32_t);
    break;

  default:
    throw std::invalid_argument(
        "invalid CUDA tensor-dictionary token width");
  }

  for (const auto &component : host.tensors) {
    if (component.size() != device.variant_count) {
      throw std::invalid_argument(
          "CUDA signed tensor-dictionary components are inconsistent");
    }
  }

  std::vector<CudaPackedTensor6<Scalar>> packed_tensors(
      device.variant_count);

  for (std::size_t variant = 0;
       variant < device.variant_count;
       ++variant) {

    packed_tensors[variant] =
        make_cuda_packed_tensor6(
            static_cast<Scalar>(host.tensors[0][variant]),
            static_cast<Scalar>(host.tensors[1][variant]),
            static_cast<Scalar>(host.tensors[2][variant]),
            static_cast<Scalar>(host.tensors[3][variant]),
            static_cast<Scalar>(host.tensors[4][variant]),
            static_cast<Scalar>(host.tensors[5][variant]));
  }

  const std::size_t target_begin_bytes =
      host.target_begins.size() * sizeof(int);

  const std::size_t target_count_bytes =
      host.target_counts.size() * sizeof(int);

  const std::size_t target_leaf_map_bytes =
      target_leaf_for_target.size() * sizeof(int);

  const std::size_t row_bytes =
      host.leaf_row_offsets.size() * sizeof(int);

  const std::size_t block_bytes =
      host.blocks.size() * sizeof(StaticP2PLeafBlock);

  const std::size_t tile_leaf_bytes =
      tile_leaf_indices.size() * sizeof(int);

  const std::size_t tile_offset_bytes =
      tile_target_offsets.size() * sizeof(int);

  const std::size_t microtile_leaf_bytes =
      microtile_schedule.leaf_indices.size() * sizeof(int);

  const std::size_t microtile_offset_bytes =
      microtile_schedule.target_offsets.size() * sizeof(int);

  const std::size_t tensor_bytes =
      packed_tensors.size() *
      sizeof(CudaPackedTensor6<Scalar>);

  const auto allocate =
      [](auto **pointer,
         const std::size_t bytes,
         const char *operation) {

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void **>(pointer),
                std::max(bytes, std::size_t{1})),
            operation);
      };

  allocate(
      &device.target_begins,
      target_begin_bytes,
      "allocate signed dictionary target begins");

  allocate(
      &device.target_counts,
      target_count_bytes,
      "allocate signed dictionary target counts");

  allocate(
      &device.target_leaf_for_target,
      target_leaf_map_bytes,
      "allocate signed dictionary target leaf map");

  allocate(
      &device.leaf_row_offsets,
      row_bytes,
      "allocate signed dictionary leaf rows");

  allocate(
      &device.leaf_blocks,
      block_bytes,
      "allocate signed dictionary leaf blocks");

  allocate(
      &device.tile_leaf_indices,
      tile_leaf_bytes,
      "allocate signed dictionary tile leaves");

  allocate(
      &device.tile_target_offsets,
      tile_offset_bytes,
      "allocate signed dictionary tile offsets");

  allocate(
      &device.microtile_leaf_indices,
      microtile_leaf_bytes,
      "allocate signed dictionary microtile leaves");

  allocate(
      &device.microtile_target_offsets,
      microtile_offset_bytes,
      "allocate signed dictionary microtile offsets");

  allocate(
      &device.tensors,
      tensor_bytes,
      "allocate packed signed dictionary tensors");

  allocate(
      &device.tokens,
      token_bytes,
      "allocate signed dictionary tokens");

  const auto upload =
      [](void *destination,
         const void *source,
         const std::size_t bytes,
         const char *operation) {

        if (bytes != 0) {
          check_cuda(
              cudaMemcpy(
                  destination,
                  source,
                  bytes,
                  cudaMemcpyHostToDevice),
              operation);
        }
      };

  upload(
      device.target_begins,
      host.target_begins.data(),
      target_begin_bytes,
      "upload signed dictionary target begins");

  upload(
      device.target_counts,
      host.target_counts.data(),
      target_count_bytes,
      "upload signed dictionary target counts");

  upload(
      device.target_leaf_for_target,
      target_leaf_for_target.data(),
      target_leaf_map_bytes,
      "upload signed dictionary target leaf map");

  upload(
      device.leaf_row_offsets,
      host.leaf_row_offsets.data(),
      row_bytes,
      "upload signed dictionary leaf rows");

  upload(
      device.leaf_blocks,
      host.blocks.data(),
      block_bytes,
      "upload signed dictionary leaf blocks");

  upload(
      device.tile_leaf_indices,
      tile_leaf_indices.data(),
      tile_leaf_bytes,
      "upload signed dictionary tile leaves");

  upload(
      device.tile_target_offsets,
      tile_target_offsets.data(),
      tile_offset_bytes,
      "upload signed dictionary tile offsets");

  upload(
      device.microtile_leaf_indices,
      microtile_schedule.leaf_indices.data(),
      microtile_leaf_bytes,
      "upload signed dictionary microtile leaves");

  upload(
      device.microtile_target_offsets,
      microtile_schedule.target_offsets.data(),
      microtile_offset_bytes,
      "upload signed dictionary microtile offsets");

  upload(
      device.tensors,
      packed_tensors.data(),
      tensor_bytes,
      "upload packed signed dictionary tensors");

  upload(
      device.tokens,
      token_source,
      token_bytes,
      "upload signed dictionary tokens");

  const std::size_t leaf_metadata_bytes =
      target_begin_bytes +
      target_count_bytes +
      target_leaf_map_bytes +
      block_bytes +
      tile_leaf_bytes +
      tile_offset_bytes +
      microtile_leaf_bytes +
      microtile_offset_bytes;

  const std::size_t total_bytes =
      row_bytes +
      leaf_metadata_bytes +
      tensor_bytes +
      token_bytes;

  statistics.setup_h2d_bytes += total_bytes;
  statistics.persistent_device_bytes += total_bytes;

  statistics.p2p_interaction_count =
      host.token_count();

  statistics.p2p_tensor_bytes =
      tensor_bytes;

  statistics.p2p_index_bytes =
      token_bytes;

  statistics.p2p_row_metadata_bytes =
      row_bytes;

  statistics.p2p_leaf_metadata_bytes =
      leaf_metadata_bytes;

  statistics.p2p_identity_bytes = 0;

  statistics.p2p_scratch_bytes =
      target_owned || power2_microtiles ||
              device.threads_per_block ==
                  cuda_dictionary_warp_size
          ? 0
          : 3 *
                static_cast<std::size_t>(
                    device.threads_per_block) *
                sizeof(Scalar);

  statistics.p2p_threads_per_block =
      device.threads_per_block;
}

// cuSPARSE descriptors of the BSR(3) plan: the sparse matrix over 3x3 blocks
// and the dense moment/field vectors it multiplies, bound once to the
// device buffers the evaluation reuses, plus the SpMV workspace.
template <typename Scalar>
void initialise_bsr_p2p(
    CudaBsrP2PDeviceView<Scalar>& plan,
    const void* input_values,
    void* output_values,
    const cudaDataType value_type,
    const char* operation_prefix) {
  check_cusparse(
      cusparseCreateBsr(
          &plan.descriptor, plan.target_count, plan.source_count,
          plan.interaction_count, 3, 3, plan.row_offsets, plan.source_indices,
          plan.values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
          CUSPARSE_INDEX_BASE_ZERO, value_type, CUSPARSE_ORDER_ROW),
      operation_prefix);

  check_cusparse(
      cusparseCreateConstDnVec(
          &plan.input_descriptor,
          static_cast<int64_t>(plan.source_count) * 3, input_values,
          value_type),
      "create cuSPARSE BSR input vector descriptor");
  check_cusparse(
      cusparseCreateDnVec(
          &plan.output_descriptor,
          static_cast<int64_t>(plan.target_count) * 3, output_values,
          value_type),
      "create cuSPARSE BSR output vector descriptor");

  constexpr Scalar alpha = static_cast<Scalar>(1);
  constexpr Scalar beta = static_cast<Scalar>(0);
  check_cusparse(
      cusparseSpMV_bufferSize(
          plan.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
          plan.descriptor, plan.input_descriptor, &beta, plan.output_descriptor,
          value_type, CUSPARSE_SPMV_ALG_DEFAULT, &plan.workspace_size),
      "size cuSPARSE BSR P2P workspace");
  if (plan.workspace_size != 0) {
    check_cuda(cudaMalloc(&plan.workspace, plan.workspace_size),
               "allocate cuSPARSE BSR P2P workspace");
  }
}

void launch_bsr_p2p(
    const CudaBsrP2PDeviceView<double>& plan,
    Vec3* fields,
    cudaStream_t stream)
{
  static_assert(sizeof(Vec3) == 3 * sizeof(double));
  if (plan.target_count == 0) {
    return;
  }
  if (plan.interaction_count == 0) {
    check_cuda(
        cudaMemsetAsync(
            fields, 0,
            static_cast<std::size_t>(plan.target_count) * sizeof(Vec3), stream),
        "clear empty cuSPARSE BSR P2P fields");
    return;
  }
  constexpr double alpha = 1.0;
  constexpr double beta = 0.0;
  check_cusparse(cusparseSetStream(plan.handle, stream),
                 "set cuSPARSE P2P stream");
  check_cusparse(
      cusparseSpMV(
          plan.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
          plan.descriptor, plan.input_descriptor, &beta, plan.output_descriptor,
          CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, plan.workspace),
      "launch cuSPARSE BSR(3) P2P");
}

void launch_bsr_p2p(
    const CudaBsrP2PDeviceView<float>& plan,
    FloatVec3* fields,
    cudaStream_t stream)
{
  static_assert(sizeof(FloatVec3) == 3 * sizeof(float));
  if (plan.target_count == 0) {
    return;
  }
  if (plan.interaction_count == 0) {
    check_cuda(cudaMemsetAsync(
                   fields, 0,
                   static_cast<std::size_t>(plan.target_count) *
                       sizeof(FloatVec3),
                   stream),
               "clear empty FP32 cuSPARSE BSR P2P fields");
    return;
  }
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  check_cusparse(cusparseSetStream(plan.handle, stream),
                 "set FP32 cuSPARSE P2P stream");
  check_cusparse(
      cusparseSpMV(
          plan.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
          plan.descriptor, plan.input_descriptor, &beta, plan.output_descriptor,
          CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, plan.workspace),
      "launch FP32 cuSPARSE BSR(3) P2P");
}

void launch_static_p2p(const CudaP2PDeviceView<StaticDipoleBlock> &plan,
                       const Vec3 *moments, const int *self_indices,
                       Vec3 *fields, cudaStream_t stream) {
  launch_static_p2p<StaticDipoleBlock, Vec3>(
      plan, moments, self_indices, fields, stream);
}

void launch_static_p2p(
    const CudaP2PDeviceView<FloatStaticDipoleBlock> &plan,
    const FloatVec3 *moments, const int *self_indices, FloatVec3 *fields,
    cudaStream_t stream) {
  launch_static_p2p<FloatStaticDipoleBlock, FloatVec3>(
      plan, moments, self_indices, fields, stream);
}

void launch_compact_p2p(const CudaCompactP2PDeviceView<double> &plan,
                        const Vec3 *moments, const int *self_indices,
                        Vec3 *fields, cudaStream_t stream) {
  launch_compact_p2p<double, Vec3>(
      plan, moments, self_indices, fields, stream);
}

void launch_compact_p2p(const CudaCompactP2PDeviceView<float> &plan,
                        const FloatVec3 *moments, const int *self_indices,
                        FloatVec3 *fields, cudaStream_t stream) {
  launch_compact_p2p<float, FloatVec3>(
      plan, moments, self_indices, fields, stream);
}

void launch_leaf_p2p(const CudaLeafP2PDeviceView<double> &plan,
                     const Vec3 *moments, const int *self_indices,
                     Vec3 *fields, cudaStream_t stream) {
  launch_leaf_p2p<double, Vec3>(plan, moments, self_indices, fields, stream);
}

void launch_leaf_p2p(const CudaLeafP2PDeviceView<float> &plan,
                     const FloatVec3 *moments, const int *self_indices,
                     FloatVec3 *fields, cudaStream_t stream) {
  launch_leaf_p2p<float, FloatVec3>(
      plan, moments, self_indices, fields, stream);
}

void launch_signed_dictionary_p2p(
    const CudaSignedDictionaryP2PDeviceView<double> &plan,
    const Vec3 *moments, Vec3 *fields, cudaStream_t stream) {
  launch_signed_dictionary_p2p<double, Vec3>(plan, moments, fields, stream);
}

void launch_signed_dictionary_p2p(
    const CudaSignedDictionaryP2PDeviceView<float> &plan,
    const FloatVec3 *moments, FloatVec3 *fields, cudaStream_t stream) {
  launch_signed_dictionary_p2p<float, FloatVec3>(
      plan, moments, fields, stream);
}

void upload_cuda_signed_dictionary(
    const StaticP2PSignedTensorDictionaryPlan &host,
    CudaSignedDictionaryP2PDeviceView<double> &device,
    CudaPlanStatistics &statistics, const bool target_owned,
    const bool power2_microtiles) {
  upload_cuda_signed_dictionary<
      double, Vec3, StaticP2PSignedTensorDictionaryPlan>(
      host, device, statistics, target_owned, power2_microtiles);
}

void upload_cuda_signed_dictionary(
    const FloatStaticP2PSignedTensorDictionaryPlan &host,
    CudaSignedDictionaryP2PDeviceView<float> &device,
    CudaPlanStatistics &statistics, const bool target_owned,
    const bool power2_microtiles) {
  upload_cuda_signed_dictionary<
      float, FloatVec3, FloatStaticP2PSignedTensorDictionaryPlan>(
      host, device, statistics, target_owned, power2_microtiles);
}

void initialise_bsr_p2p_resources(
    CudaBsrP2PDeviceView<double> &plan, const Vec3 *input_values,
    Vec3 *output_values, const char *handle_operation,
    const char *descriptor_operation) {
  check_cusparse(cusparseCreate(&plan.handle), handle_operation);
  initialise_bsr_p2p<double>(plan, input_values, output_values, CUDA_R_64F,
                             descriptor_operation);
}

void initialise_bsr_p2p_resources(
    CudaBsrP2PDeviceView<float> &plan, const FloatVec3 *input_values,
    FloatVec3 *output_values, const char *handle_operation,
    const char *descriptor_operation) {
  check_cusparse(cusparseCreate(&plan.handle), handle_operation);
  initialise_bsr_p2p<float>(plan, input_values, output_values, CUDA_R_32F,
                            descriptor_operation);
}

template <typename HostPlan, typename Block>
void upload_cuda_canonical_impl(
    const HostPlan &host, CudaP2PDeviceView<Block> &device,
    CudaPlanStatistics &statistics, const cudaStream_t stream,
    const char *allocation_operation, const char *upload_operation) {
  device.target_count = host.target_count;
  const std::size_t row_bytes = host.row_offsets.size() * sizeof(int);
  const std::size_t block_bytes = host.blocks.size() * sizeof(Block);
  check_cuda(cudaMalloc(&device.row_offsets,
                        std::max(row_bytes, sizeof(int))),
             allocation_operation);
  check_cuda(cudaMalloc(&device.blocks,
                        std::max(block_bytes, sizeof(Block))),
             allocation_operation);
  if (row_bytes != 0) {
    check_cuda(cudaMemcpyAsync(device.row_offsets, host.row_offsets.data(),
                               row_bytes, cudaMemcpyHostToDevice, stream),
               upload_operation);
  }
  if (block_bytes != 0) {
    check_cuda(cudaMemcpyAsync(device.blocks, host.blocks.data(), block_bytes,
                               cudaMemcpyHostToDevice, stream),
               upload_operation);
  }
  statistics.setup_h2d_bytes += row_bytes + block_bytes;
}

// Uploads a dense leaf packing. Portable leaf tensors are target-major; the
// CUDA copy transposes every block to source-major so that the lanes of one
// warp read consecutive targets. The per-block target leaf map lets one warp
// own one block without searching the leaf row offsets.
template <typename Scalar, typename Plan>
void upload_cuda_leaf_typed(const Plan &leaf,
                            CudaLeafP2PDeviceView<Scalar> &device,
                            CudaPlanStatistics &statistics,
                            const char *allocation_operation,
                            const char *upload_operation) {
  device.target_count = leaf.target_count;
  device.target_leaf_count = static_cast<int>(leaf.target_begins.size());
  device.block_count = static_cast<int>(leaf.blocks.size());
  device.interaction_count = leaf.tensors[0].size();
  device.threads_per_block = leaf_p2p_threads;

  std::vector<int> block_target_leaves(leaf.blocks.size(), 0);
  for (std::size_t target_leaf = 0; target_leaf < leaf.target_begins.size();
       ++target_leaf) {
    for (int block_index = leaf.leaf_row_offsets[target_leaf];
         block_index < leaf.leaf_row_offsets[target_leaf + 1]; ++block_index) {
      block_target_leaves[static_cast<std::size_t>(block_index)] =
          static_cast<int>(target_leaf);
    }
  }

  const std::size_t target_metadata_bytes =
      (leaf.target_begins.size() + leaf.target_counts.size()) * sizeof(int);
  const std::size_t row_bytes = leaf.leaf_row_offsets.size() * sizeof(int);
  const std::size_t block_leaf_bytes = block_target_leaves.size() * sizeof(int);
  const std::size_t block_bytes =
      leaf.blocks.size() * sizeof(StaticP2PLeafBlock);
  const std::size_t tensor_bytes = leaf.tensors[0].size() * 6 * sizeof(Scalar);
  const auto allocate = [&](void **pointer, const std::size_t bytes) {
    check_cuda(cudaMalloc(pointer, std::max(bytes, std::size_t{1})),
               allocation_operation);
  };
  allocate(reinterpret_cast<void **>(&device.target_begins),
           leaf.target_begins.size() * sizeof(int));
  allocate(reinterpret_cast<void **>(&device.target_counts),
           leaf.target_counts.size() * sizeof(int));
  allocate(reinterpret_cast<void **>(&device.leaf_row_offsets), row_bytes);
  allocate(reinterpret_cast<void **>(&device.block_target_leaves),
           block_leaf_bytes);
  allocate(reinterpret_cast<void **>(&device.leaf_blocks), block_bytes);
  allocate(reinterpret_cast<void **>(&device.tensors), tensor_bytes);

  const auto upload = [&](void *destination, const void *source,
                          const std::size_t bytes) {
    if (bytes != 0) {
      check_cuda(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice),
                 upload_operation);
    }
  };
  upload(device.target_begins, leaf.target_begins.data(),
         leaf.target_begins.size() * sizeof(int));
  upload(device.target_counts, leaf.target_counts.data(),
         leaf.target_counts.size() * sizeof(int));
  upload(device.leaf_row_offsets, leaf.leaf_row_offsets.data(), row_bytes);
  upload(device.block_target_leaves, block_target_leaves.data(),
         block_leaf_bytes);
  upload(device.leaf_blocks, leaf.blocks.data(), block_bytes);

  // Each component plane is transposed into one reused staging buffer and
  // uploaded before the next, so the host holds one plane rather than a
  // second copy of all six; the uploads and the device layout are unchanged.
  std::vector<Scalar> transposed(leaf.tensors[0].size());
  for (std::size_t component = 0; component < 6; ++component) {
    for (std::size_t target_leaf = 0; target_leaf < leaf.target_begins.size();
         ++target_leaf) {
      const int target_count = leaf.target_counts[target_leaf];
      for (int block_index = leaf.leaf_row_offsets[target_leaf];
           block_index < leaf.leaf_row_offsets[target_leaf + 1]; ++block_index) {
        const StaticP2PLeafBlock &block =
            leaf.blocks[static_cast<std::size_t>(block_index)];
        for (int local_target = 0; local_target < target_count; ++local_target) {
          for (int local_source = 0; local_source < block.source_count;
               ++local_source) {
            const std::size_t source_index = block.tensor_offset +
                static_cast<std::size_t>(local_target) * block.source_count +
                local_source;
            const std::size_t destination_index = block.tensor_offset +
                static_cast<std::size_t>(local_source) * target_count +
                local_target;
            transposed[destination_index] = leaf.tensors[component][source_index];
          }
        }
      }
    }
    upload(device.tensors + component * leaf.tensors[0].size(),
           transposed.data(), transposed.size() * sizeof(Scalar));
  }

  const std::size_t total_bytes = target_metadata_bytes + row_bytes +
      block_leaf_bytes + block_bytes + tensor_bytes;
  statistics.setup_h2d_bytes += total_bytes;
  statistics.persistent_device_bytes += total_bytes;
  statistics.p2p_interaction_count = leaf.tensors[0].size();
  statistics.p2p_tensor_bytes = tensor_bytes;
  statistics.p2p_row_metadata_bytes = row_bytes;
  statistics.p2p_leaf_metadata_bytes =
      target_metadata_bytes + block_leaf_bytes + block_bytes;
  statistics.p2p_scratch_bytes = 0;
  statistics.p2p_threads_per_block = leaf_p2p_threads;
}

void upload_cuda_leaf(const StaticP2PLeafPlan &host,
                      CudaLeafP2PDeviceView<double> &device,
                      CudaPlanStatistics &statistics,
                      const char *allocation_operation,
                      const char *upload_operation) {
  upload_cuda_leaf_typed(host, device, statistics, allocation_operation,
                         upload_operation);
}

void upload_cuda_leaf(const FloatStaticP2PLeafPlan &host,
                      CudaLeafP2PDeviceView<float> &device,
                      CudaPlanStatistics &statistics,
                      const char *allocation_operation,
                      const char *upload_operation) {
  upload_cuda_leaf_typed(host, device, statistics, allocation_operation,
                         upload_operation);
}

// The position-based executor needs no canonical tensors: it uploads the
// sorted positions (one aligned four-scalar slot each) and the canonical
// list-1 records with their shift and identity marker. Positions are stored
// relative to their own leaf centre and each record's shift is the source
// leaf centre minus the target leaf centre plus the image shift, computed in
// FP64 and rounded once: the displacement `x_t - x_s` is then formed from
// leaf-sized numbers and an (exactly representable) centre difference, which
// keeps the FP32 pair error near the rounding level instead of the
// cancellation of two root-frame coordinates. Sources and targets share one
// position array when they are the same sorted points in the same leaves.
template <typename Scalar>
void upload_cuda_point_geometry_typed(
    const StaticFmmTopology &topology,
    CudaPointGeometryP2PDeviceView<Scalar> &device,
    CudaPlanStatistics &statistics, const char *allocation_operation,
    const char *upload_operation) {
  using Scalar4 = typename CudaScalar4<Scalar>::type;
  device.target_count =
      static_cast<int>(topology.sorted_target_positions.size());
  device.record_count = static_cast<int>(topology.p2p_leaf_records.size());
  device.threads_per_block = point_geometry_p2p_threads;
  const auto &nodes = topology.nodes;

  std::vector<CudaPointGeometryP2PRecord<Scalar>> records;
  records.reserve(topology.p2p_leaf_records.size());
  std::size_t interactions = 0;
  for (const StaticP2PLeafRecord &record : topology.p2p_leaf_records) {
    const Vec3 shift =
        nodes[static_cast<std::size_t>(record.source_leaf)].centre -
        nodes[static_cast<std::size_t>(record.target_leaf)].centre +
        record.source_shift;
    CudaPointGeometryP2PRecord<Scalar> packed;
    packed.target_begin = static_cast<int>(record.target_begin);
    packed.target_count = static_cast<int>(record.target_count);
    packed.source_begin = static_cast<int>(record.source_begin);
    packed.source_count = static_cast<int>(record.source_count);
    packed.shift_x = static_cast<Scalar>(shift.x);
    packed.shift_y = static_cast<Scalar>(shift.y);
    packed.shift_z = static_cast<Scalar>(shift.z);
    packed.skip_for_identity = record.skip_for_identity ? 1 : 0;
    records.push_back(packed);
    interactions += record.target_count * record.source_count;
  }
  device.interaction_count = interactions;

  // Every sorted point lies in exactly one occupied leaf range.
  const auto pack_positions = [&nodes](const std::vector<Vec3> &positions,
                                       std::span<const StaticLeafRange> leaves) {
    std::vector<Scalar4> packed(positions.size());
    for (const StaticLeafRange &leaf : leaves) {
      const Vec3 centre = nodes[static_cast<std::size_t>(leaf.node)].centre;
      for (std::size_t index = leaf.begin; index < leaf.begin + leaf.count;
           ++index) {
        const Vec3 d = positions[index] - centre;
        packed[index].x = static_cast<Scalar>(d.x);
        packed[index].y = static_cast<Scalar>(d.y);
        packed[index].z = static_cast<Scalar>(d.z);
        packed[index].w = Scalar{0};
      }
    }
    return packed;
  };
  const std::vector<Scalar4> sources =
      pack_positions(topology.sorted_source_positions, topology.source_leaves);
  std::vector<Scalar4> targets =
      pack_positions(topology.sorted_target_positions, topology.target_leaves);
  const bool shared_positions =
      targets.size() == sources.size() &&
      std::equal(targets.begin(), targets.end(), sources.begin(),
                 [](const Scalar4 &a, const Scalar4 &b) {
                   return a.x == b.x && a.y == b.y && a.z == b.z;
                 });
  device.shared_positions = shared_positions;
  if (shared_positions) {
    targets.clear();
  }

  const std::size_t record_bytes =
      records.size() * sizeof(CudaPointGeometryP2PRecord<Scalar>);
  const std::size_t source_bytes = sources.size() * sizeof(Scalar4);
  const std::size_t target_bytes = targets.size() * sizeof(Scalar4);
  const auto allocate = [&](void **pointer, const std::size_t bytes) {
    check_cuda(cudaMalloc(pointer, std::max(bytes, std::size_t{1})),
               allocation_operation);
  };
  const auto upload = [&](void *destination, const void *source,
                          const std::size_t bytes) {
    if (bytes != 0) {
      check_cuda(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice),
                 upload_operation);
    }
  };
  allocate(reinterpret_cast<void **>(&device.records), record_bytes);
  allocate(reinterpret_cast<void **>(&device.source_positions), source_bytes);
  upload(device.records, records.data(), record_bytes);
  upload(device.source_positions, sources.data(), source_bytes);
  if (shared_positions) {
    device.target_positions = device.source_positions;
  } else {
    allocate(reinterpret_cast<void **>(&device.target_positions), target_bytes);
    upload(device.target_positions, targets.data(), target_bytes);
  }

  const std::size_t total_bytes = record_bytes + source_bytes + target_bytes;
  statistics.setup_h2d_bytes += total_bytes;
  statistics.persistent_device_bytes += total_bytes;
  statistics.p2p_interaction_count = interactions;
  statistics.p2p_tensor_bytes = 0;
  statistics.p2p_index_bytes = 0;
  statistics.p2p_row_metadata_bytes = 0;
  statistics.p2p_leaf_metadata_bytes = record_bytes;
  statistics.p2p_geometry_bytes = source_bytes + target_bytes;
  statistics.p2p_scratch_bytes = 0;
  statistics.p2p_threads_per_block = point_geometry_p2p_threads;
}

void upload_cuda_point_geometry(
    const StaticFmmTopology &topology,
    CudaPointGeometryP2PDeviceView<double> &device,
    CudaPlanStatistics &statistics, const char *allocation_operation,
    const char *upload_operation) {
  upload_cuda_point_geometry_typed(topology, device, statistics,
                                   allocation_operation, upload_operation);
}

void upload_cuda_point_geometry(
    const StaticFmmTopology &topology,
    CudaPointGeometryP2PDeviceView<float> &device,
    CudaPlanStatistics &statistics, const char *allocation_operation,
    const char *upload_operation) {
  upload_cuda_point_geometry_typed(topology, device, statistics,
                                   allocation_operation, upload_operation);
}

void launch_point_geometry_p2p(
    const CudaPointGeometryP2PDeviceView<double> &plan, const Vec3 *moments,
    const int *self_indices, Vec3 *fields, cudaStream_t stream) {
  launch_point_geometry_p2p_typed(plan, moments, self_indices, fields, stream);
}

void launch_point_geometry_p2p(
    const CudaPointGeometryP2PDeviceView<float> &plan,
    const FloatVec3 *moments, const int *self_indices, FloatVec3 *fields,
    cudaStream_t stream) {
  launch_point_geometry_p2p_typed(plan, moments, self_indices, fields, stream);
}

void upload_cuda_canonical(
    const StaticP2POperator &host,
    CudaP2PDeviceView<StaticDipoleBlock> &device,
    CudaPlanStatistics &statistics, const cudaStream_t stream,
    const char *allocation_operation, const char *upload_operation) {
  upload_cuda_canonical_impl(host, device, statistics, stream,
                             allocation_operation, upload_operation);
}

void upload_cuda_canonical(
    const FloatStaticP2POperator &host,
    CudaP2PDeviceView<FloatStaticDipoleBlock> &device,
    CudaPlanStatistics &statistics, const cudaStream_t stream,
    const char *allocation_operation, const char *upload_operation) {
  upload_cuda_canonical_impl(host, device, statistics, stream,
                             allocation_operation, upload_operation);
}

template <typename HostPlan, typename Scalar, typename Vector>
void upload_cuda_bsr_impl(
    const HostPlan &host, CudaBsrP2PDeviceView<Scalar> &device,
    const Vector *input_values, Vector *output_values,
    CudaPlanStatistics &statistics, const cudaStream_t stream,
    const char *allocation_operation, const char *upload_operation,
    const char *handle_operation, const char *descriptor_operation) {
  device.target_count = host.target_count;
  device.source_count = host.source_count;
  device.interaction_count = static_cast<int>(host.source_indices.size());
  const std::size_t row_bytes = host.row_offsets.size() * sizeof(int);
  const std::size_t index_bytes = host.source_indices.size() * sizeof(int);
  const std::size_t value_bytes = host.values.size() * sizeof(Scalar);
  const auto allocate = [&](auto **pointer, const std::size_t bytes) {
    check_cuda(cudaMalloc(reinterpret_cast<void **>(pointer),
                          std::max(bytes, std::size_t{1})),
               allocation_operation);
  };
  allocate(&device.row_offsets, row_bytes);
  allocate(&device.source_indices, index_bytes);
  allocate(&device.values, value_bytes);
  initialise_bsr_p2p_resources(device, input_values, output_values,
                               handle_operation, descriptor_operation);
  const auto upload = [&](void *destination, const void *source,
                          const std::size_t bytes) {
    if (bytes != 0) {
      check_cuda(cudaMemcpyAsync(destination, source, bytes,
                                 cudaMemcpyHostToDevice, stream),
                 upload_operation);
    }
  };
  upload(device.row_offsets, host.row_offsets.data(), row_bytes);
  upload(device.source_indices, host.source_indices.data(), index_bytes);
  upload(device.values, host.values.data(), value_bytes);
  statistics.setup_h2d_bytes += row_bytes + index_bytes + value_bytes;
}

void upload_cuda_bsr(
    const StaticP2PBsrPlan &host, CudaBsrP2PDeviceView<double> &device,
    const Vec3 *input_values, Vec3 *output_values,
    CudaPlanStatistics &statistics, const cudaStream_t stream,
    const char *allocation_operation, const char *upload_operation,
    const char *handle_operation, const char *descriptor_operation) {
  upload_cuda_bsr_impl(host, device, input_values, output_values, statistics,
                       stream, allocation_operation, upload_operation,
                       handle_operation, descriptor_operation);
}

void upload_cuda_bsr(
    const FloatStaticP2PBsrPlan &host, CudaBsrP2PDeviceView<float> &device,
    const FloatVec3 *input_values, FloatVec3 *output_values,
    CudaPlanStatistics &statistics, const cudaStream_t stream,
    const char *allocation_operation, const char *upload_operation,
    const char *handle_operation, const char *descriptor_operation) {
  upload_cuda_bsr_impl(host, device, input_values, output_values, statistics,
                       stream, allocation_operation, upload_operation,
                       handle_operation, descriptor_operation);
}

// Release helpers: every view is value-initialised, unused pointers are null
// and cudaFree accepts null, so releasing an unpopulated view is a no-op.
// Each resets the view so a second release is harmless.
template <typename Block>
void release_canonical_p2p(CudaP2PDeviceView<Block> &plan) noexcept {
  cudaFree(plan.row_offsets);
  cudaFree(plan.blocks);
  plan = {};
}

template <typename Scalar>
void release_compact_p2p(CudaCompactP2PDeviceView<Scalar> &plan) noexcept {
  cudaFree(plan.row_offsets);
  cudaFree(plan.source_indices);
  cudaFree(plan.skip_for_identity);
  cudaFree(plan.tensors);
  plan = {};
}

template <typename Scalar>
void release_leaf_p2p(CudaLeafP2PDeviceView<Scalar> &plan) noexcept {
  cudaFree(plan.target_begins);
  cudaFree(plan.target_counts);
  cudaFree(plan.leaf_row_offsets);
  cudaFree(plan.block_target_leaves);
  cudaFree(plan.leaf_blocks);
  cudaFree(plan.tensors);
  plan = {};
}

template <typename Scalar>
void release_dictionary_p2p(
    CudaSignedDictionaryP2PDeviceView<Scalar> &plan) noexcept {
  cudaFree(plan.target_begins);
  cudaFree(plan.target_counts);
  cudaFree(plan.target_leaf_for_target);
  cudaFree(plan.leaf_row_offsets);
  cudaFree(plan.leaf_blocks);
  cudaFree(plan.tile_leaf_indices);
  cudaFree(plan.tile_target_offsets);
  cudaFree(plan.microtile_leaf_indices);
  cudaFree(plan.microtile_target_offsets);
  cudaFree(plan.tensors);
  cudaFree(plan.tokens);
  plan = {};
}

template <typename Scalar>
void release_bsr_p2p(CudaBsrP2PDeviceView<Scalar> &plan) noexcept {
  if (plan.input_descriptor != nullptr) {
    cusparseDestroyDnVec(plan.input_descriptor);
  }
  if (plan.output_descriptor != nullptr) {
    cusparseDestroyDnVec(plan.output_descriptor);
  }
  if (plan.descriptor != nullptr) {
    cusparseDestroySpMat(plan.descriptor);
  }
  if (plan.handle != nullptr) {
    cusparseDestroy(plan.handle);
  }
  cudaFree(plan.row_offsets);
  cudaFree(plan.source_indices);
  cudaFree(plan.values);
  cudaFree(plan.workspace);
  plan = {};
}

void release_p2p_device_view(
    CudaP2PDeviceView<StaticDipoleBlock> &plan) noexcept {
  release_canonical_p2p(plan);
}

void release_p2p_device_view(
    CudaP2PDeviceView<FloatStaticDipoleBlock> &plan) noexcept {
  release_canonical_p2p(plan);
}

void release_p2p_device_view(
    CudaCompactP2PDeviceView<double> &plan) noexcept {
  release_compact_p2p(plan);
}

void release_p2p_device_view(
    CudaCompactP2PDeviceView<float> &plan) noexcept {
  release_compact_p2p(plan);
}

void release_p2p_device_view(CudaLeafP2PDeviceView<double> &plan) noexcept {
  release_leaf_p2p(plan);
}

void release_p2p_device_view(CudaLeafP2PDeviceView<float> &plan) noexcept {
  release_leaf_p2p(plan);
}

template <typename Scalar>
void release_point_geometry_p2p(
    CudaPointGeometryP2PDeviceView<Scalar> &plan) noexcept {
  cudaFree(plan.records);
  cudaFree(plan.source_positions);
  if (!plan.shared_positions) {
    cudaFree(plan.target_positions);
  }
  plan = {};
}

void release_p2p_device_view(
    CudaPointGeometryP2PDeviceView<double> &plan) noexcept {
  release_point_geometry_p2p(plan);
}

void release_p2p_device_view(
    CudaPointGeometryP2PDeviceView<float> &plan) noexcept {
  release_point_geometry_p2p(plan);
}

void release_p2p_device_view(
    CudaSignedDictionaryP2PDeviceView<double> &plan) noexcept {
  release_dictionary_p2p(plan);
}

void release_p2p_device_view(
    CudaSignedDictionaryP2PDeviceView<float> &plan) noexcept {
  release_dictionary_p2p(plan);
}

void release_p2p_device_view(CudaBsrP2PDeviceView<double> &plan) noexcept {
  release_bsr_p2p(plan);
}

void release_p2p_device_view(CudaBsrP2PDeviceView<float> &plan) noexcept {
  release_bsr_p2p(plan);
}

} // namespace cuda_p2p_detail

using namespace cuda_p2p_detail;

// The hybrid backend's near-field plan.  Exactly one device view (of the
// plan's precision) is populated according to `kind`; the others stay empty
// and release as no-ops.  The plan owns a private non-blocking stream so its
// work overlaps the CPU far field, and the begin/finish protocol below lets
// the caller enqueue the whole round trip, do other work, and collect the
// result later.  `dynamic_self_identities` says whether the identity map is
// uploaded per evaluation (true for packings that read it at run time
// without a fixed map) or was fixed at construction.
struct CudaP2PPlan::Implementation {
  enum class Kind {
    Canonical,
    Compact,
    Leaf,
    TensorDictionary,
    Bsr,
    PointGeometry
  };

  int source_count{0};
  int target_count{0};
  bool fp32{false};
  Kind kind{Kind::Canonical};
  CudaP2PDeviceView<StaticDipoleBlock> canonical{};
  CudaCompactP2PDeviceView<double> compact{};
  CudaLeafP2PDeviceView<double> leaf{};
  CudaSignedDictionaryP2PDeviceView<double> dictionary{};
  CudaBsrP2PDeviceView<double> bsr{};
  CudaP2PDeviceView<FloatStaticDipoleBlock> canonical_float{};
  CudaCompactP2PDeviceView<float> compact_float{};
  CudaLeafP2PDeviceView<float> leaf_float{};
  CudaSignedDictionaryP2PDeviceView<float> dictionary_float{};
  CudaBsrP2PDeviceView<float> bsr_float{};
  CudaPointGeometryP2PDeviceView<double> point_geometry{};
  CudaPointGeometryP2PDeviceView<float> point_geometry_float{};
  Vec3* moments{nullptr};
  int* self_indices{nullptr};
  Vec3* fields{nullptr};
  Vec3* pinned_moments{nullptr};
  int* pinned_self_indices{nullptr};
  Vec3* pinned_fields{nullptr};
  FloatVec3 *moments_float{nullptr};
  FloatVec3 *fields_float{nullptr};
  FloatVec3 *pinned_moments_float{nullptr};
  FloatVec3 *pinned_fields_float{nullptr};
  cudaStream_t stream{};
  // `d2h` is the functional completion point that finish_evaluate waits on
  // (cudaEventDisableTiming).  The others are diagnostic, recorded only at
  // TimingLevel::Detailed; `finished` is the timing twin of `d2h`.
  cudaEvent_t d2h{};
  cudaEvent_t start{};
  cudaEvent_t h2d{};
  cudaEvent_t kernel{};
  cudaEvent_t finished{};
  TimingLevel timing_level{TimingLevel::Off};
  CudaPlanStatistics statistics{};
  CudaEvaluationTimings timings{};
  std::vector<int> fixed_self_indices{};
  bool dynamic_self_identities{true};
  bool pending{false};
};

namespace {

// The h2d / kernel / d2h lanes of one completed evaluation; `total` is their
// sum because the plan's stream is serial.
template <typename Implementation>
void read_p2p_timings(Implementation &plan, const char *what) {
  CudaEvaluationTimings &timings = plan.timings;
  timings = {};
  timings.timing_level = TimingLevel::Detailed;
  timings.h2d_seconds =
      cuda_detail::diagnostic_elapsed_seconds(plan.start, plan.h2d, what);
  timings.kernel_seconds =
      cuda_detail::diagnostic_elapsed_seconds(plan.h2d, plan.kernel, what);
  timings.p2p_seconds = timings.kernel_seconds;
  timings.d2h_seconds = cuda_detail::diagnostic_elapsed_seconds(
      plan.kernel, plan.finished, what);
  timings.total_seconds =
      timings.h2d_seconds + timings.kernel_seconds + timings.d2h_seconds;
}

} // namespace

CudaP2PPlan::CudaP2PPlan(
    const int source_count,
    const int target_count,
    const std::span<const int> fixed_self_indices,
    const bool device_self_indices)
    : CudaP2PPlan(source_count, target_count, fixed_self_indices,
                  device_self_indices, StaticPrecision::Float64) {}

// Common constructor: the stream, events and the per-evaluation moment /
// identity / field buffers (device and pinned) of the plan's precision.  The
// packing-specific constructors delegate here and then upload their view.
// `device_self_indices` is false for the packings whose executors take no
// identity array (dictionary, BSR).
CudaP2PPlan::CudaP2PPlan(
    const int source_count, const int target_count,
    const std::span<const int> fixed_self_indices,
    const bool device_self_indices, const StaticPrecision precision)
    : implementation_(new Implementation{})
{
  auto &plan = *implementation_;
  plan.fp32 = precision == StaticPrecision::Float32;
  plan.source_count = source_count;
  plan.target_count = target_count;
  if (!fixed_self_indices.empty() &&
      fixed_self_indices.size() != static_cast<std::size_t>(target_count)) {
    throw std::invalid_argument("fixed CUDA P2P identity dimensions are inconsistent");
  }
  plan.dynamic_self_identities = fixed_self_indices.empty();
  plan.fixed_self_indices.assign(
      fixed_self_indices.begin(), fixed_self_indices.end());
  check_cuda(cudaStreamCreateWithFlags(&plan.stream, cudaStreamNonBlocking),
             "create static P2P stream");
  check_cuda(cudaEventCreateWithFlags(&plan.d2h, cudaEventDisableTiming),
             "create static P2P event");
  for (cudaEvent_t *event : {&plan.start, &plan.h2d, &plan.kernel,
                             &plan.finished}) {
    check_cuda(cudaEventCreate(event), "create static P2P event");
  }
  if (plan.fp32) {
    check_cuda(cudaMalloc(&plan.moments_float,
                          std::max(source_count * sizeof(FloatVec3),
                                   sizeof(FloatVec3))),
               "allocate FP32 P2P moments");
  } else {
    check_cuda(cudaMalloc(&plan.moments,
                          std::max(source_count * sizeof(Vec3), sizeof(Vec3))),
               "allocate P2P moments");
  }
  if (device_self_indices) {
    check_cuda(cudaMalloc(&plan.self_indices,
                          std::max(target_count * sizeof(int), sizeof(int))),
               "allocate P2P identities");
  }
  if (plan.fp32) {
    check_cuda(cudaMalloc(&plan.fields_float,
                          std::max(target_count * sizeof(FloatVec3),
                                   sizeof(FloatVec3))),
               "allocate FP32 P2P fields");
    check_cuda(cudaMallocHost(
                   &plan.pinned_moments_float,
                   std::max(source_count * sizeof(FloatVec3),
                            sizeof(FloatVec3))),
               "allocate pinned FP32 P2P moments");
  } else {
    check_cuda(cudaMalloc(&plan.fields,
                          std::max(target_count * sizeof(Vec3), sizeof(Vec3))),
               "allocate P2P fields");
    check_cuda(cudaMallocHost(
                   &plan.pinned_moments,
                   std::max(source_count * sizeof(Vec3), sizeof(Vec3))),
               "allocate pinned P2P moments");
  }
  if (device_self_indices && plan.dynamic_self_identities) {
    check_cuda(cudaMallocHost(
                   &plan.pinned_self_indices,
                   std::max(target_count * sizeof(int), sizeof(int))),
               "allocate pinned P2P identities");
  }
  if (plan.fp32) {
    check_cuda(cudaMallocHost(
                   &plan.pinned_fields_float,
                   std::max(target_count * sizeof(FloatVec3),
                            sizeof(FloatVec3))),
               "allocate pinned FP32 P2P fields");
  } else {
    check_cuda(cudaMallocHost(
                   &plan.pinned_fields,
                   std::max(target_count * sizeof(Vec3), sizeof(Vec3))),
               "allocate pinned P2P fields");
  }

  const std::size_t vector_bytes =
      plan.fp32 ? sizeof(FloatVec3) : sizeof(Vec3);
  plan.statistics.scalar_bytes = plan.fp32 ? sizeof(float) : sizeof(double);
  plan.statistics.persistent_device_bytes =
      static_cast<std::size_t>(source_count + target_count) * vector_bytes +
      (device_self_indices
           ? static_cast<std::size_t>(target_count) * sizeof(int)
           : 0);
  plan.statistics.p2p_identity_bytes = device_self_indices
      ? static_cast<std::size_t>(target_count) * sizeof(int)
      : 0;
  if (device_self_indices && !fixed_self_indices.empty()) {
    check_cuda(cudaMemcpy(
                   plan.self_indices, fixed_self_indices.data(),
                   fixed_self_indices.size_bytes(), cudaMemcpyHostToDevice),
               "upload fixed P2P identities");
    plan.statistics.setup_h2d_bytes += fixed_self_indices.size_bytes();
  }
  plan.statistics.plan_generation_count = 1;
  plan.statistics.static_upload_count = 1;
  plan.statistics.static_p2p_upload_count = 1;
  plan.statistics.geometry_upload_count = 1;
}

CudaP2PPlan::CudaP2PPlan(
    const StaticP2POperator &operator_map,
    const std::span<const int> fixed_self_indices)
    : CudaP2PPlan(
          operator_map.source_count, operator_map.target_count,
          fixed_self_indices, true)
{
  auto& plan = *implementation_;
  plan.kind = Implementation::Kind::Canonical;
  plan.canonical.target_count = plan.target_count;
  const std::size_t row_bytes = operator_map.row_offsets.size() * sizeof(int);
  const std::size_t block_bytes =
      operator_map.blocks.size() * sizeof(StaticDipoleBlock);
  check_cuda(cudaMalloc(&plan.canonical.row_offsets,
                        std::max(row_bytes, sizeof(int))),
             "allocate canonical P2P rows");
  check_cuda(cudaMalloc(&plan.canonical.blocks,
                        std::max(block_bytes, sizeof(StaticDipoleBlock))),
             "allocate canonical P2P blocks");
  check_cuda(cudaMemcpy(plan.canonical.row_offsets,
                        operator_map.row_offsets.data(), row_bytes,
                        cudaMemcpyHostToDevice),
             "upload canonical P2P rows");
  if (block_bytes != 0) {
    check_cuda(cudaMemcpy(plan.canonical.blocks, operator_map.blocks.data(),
                          block_bytes, cudaMemcpyHostToDevice),
               "upload canonical P2P blocks");
  }
  plan.statistics.setup_h2d_bytes += row_bytes + block_bytes;
  plan.statistics.persistent_device_bytes += row_bytes + block_bytes;
  plan.statistics.p2p_interaction_count = operator_map.blocks.size();
  plan.statistics.p2p_tensor_bytes =
      operator_map.blocks.size() * 6 * sizeof(double);
  plan.statistics.p2p_index_bytes =
      operator_map.blocks.size() * 3 * sizeof(int);
  plan.statistics.p2p_row_metadata_bytes = row_bytes;
  plan.statistics.p2p_threads_per_block = static_operator_threads;
}

CudaP2PPlan::CudaP2PPlan(
    const StaticP2PCompactPlan& compact,
    const std::span<const int> fixed_self_indices)
    : CudaP2PPlan(
          compact.source_count, compact.target_count, fixed_self_indices, true)
{
  auto& plan = *implementation_;
  plan.kind = Implementation::Kind::Compact;
  plan.compact.target_count = plan.target_count;
  plan.compact.interaction_count = compact.source_indices.size();
  const std::size_t row_bytes = compact.row_offsets.size() * sizeof(int);
  const std::size_t index_bytes = compact.source_indices.size() * sizeof(int);
  const std::size_t tensor_bytes =
      compact.source_indices.size() * 6 * sizeof(double);
  check_cuda(cudaMalloc(&plan.compact.row_offsets,
                        std::max(row_bytes, sizeof(int))),
             "allocate compact P2P rows");
  check_cuda(cudaMalloc(&plan.compact.source_indices,
                        std::max(index_bytes, sizeof(int))),
             "allocate compact P2P sources");
  const std::size_t identity_bytes =
      compact.skip_for_identity.size() * sizeof(unsigned char);
  check_cuda(cudaMalloc(&plan.compact.skip_for_identity,
                        std::max(identity_bytes, sizeof(unsigned char))),
             "allocate compact P2P identity markers");
  check_cuda(cudaMalloc(&plan.compact.tensors,
                        std::max(tensor_bytes, sizeof(double))),
             "allocate compact P2P tensors");
  check_cuda(cudaMemcpy(plan.compact.row_offsets, compact.row_offsets.data(),
                        row_bytes, cudaMemcpyHostToDevice),
             "upload compact P2P rows");
  if (index_bytes != 0) {
    check_cuda(cudaMemcpy(plan.compact.source_indices,
                          compact.source_indices.data(), index_bytes,
                          cudaMemcpyHostToDevice),
               "upload compact P2P sources");
    check_cuda(cudaMemcpy(plan.compact.skip_for_identity,
                          compact.skip_for_identity.data(), identity_bytes,
                          cudaMemcpyHostToDevice),
               "upload compact P2P identity markers");
    for (std::size_t component = 0; component < 6; ++component) {
      check_cuda(cudaMemcpy(
                     plan.compact.tensors +
                         component * compact.source_indices.size(),
                     compact.tensors[component].data(),
                     compact.tensors[component].size() * sizeof(double),
                     cudaMemcpyHostToDevice),
                 "upload compact P2P tensor component");
    }
  }
  plan.statistics.setup_h2d_bytes +=
      row_bytes + index_bytes + identity_bytes + tensor_bytes;
  plan.statistics.persistent_device_bytes +=
      row_bytes + index_bytes + identity_bytes + tensor_bytes;
  plan.statistics.p2p_interaction_count = compact.source_indices.size();
  plan.statistics.p2p_tensor_bytes = tensor_bytes;
  plan.statistics.p2p_index_bytes = index_bytes + identity_bytes;
  plan.statistics.p2p_row_metadata_bytes = row_bytes;
  plan.statistics.p2p_threads_per_block = static_operator_threads;
}

CudaP2PPlan::CudaP2PPlan(
    const FloatStaticP2POperator &operator_map,
    const std::span<const int> fixed_self_indices)
    : CudaP2PPlan(operator_map.source_count, operator_map.target_count,
                  fixed_self_indices, true, StaticPrecision::Float32) {
  auto &plan = *implementation_;
  plan.kind = Implementation::Kind::Canonical;
  plan.canonical_float.target_count = plan.target_count;
  const std::size_t row_bytes = operator_map.row_offsets.size() * sizeof(int);
  const std::size_t block_bytes =
      operator_map.blocks.size() * sizeof(FloatStaticDipoleBlock);
  check_cuda(cudaMalloc(&plan.canonical_float.row_offsets,
                        std::max(row_bytes, sizeof(int))),
             "allocate FP32 canonical P2P rows");
  check_cuda(cudaMalloc(&plan.canonical_float.blocks,
                        std::max(block_bytes, sizeof(FloatStaticDipoleBlock))),
             "allocate FP32 canonical P2P blocks");
  if (row_bytes != 0) {
    check_cuda(cudaMemcpy(plan.canonical_float.row_offsets,
                          operator_map.row_offsets.data(), row_bytes,
                          cudaMemcpyHostToDevice),
               "upload FP32 canonical P2P rows");
  }
  if (block_bytes != 0) {
    check_cuda(cudaMemcpy(plan.canonical_float.blocks,
                          operator_map.blocks.data(), block_bytes,
                          cudaMemcpyHostToDevice),
               "upload FP32 canonical P2P blocks");
  }
  plan.statistics.setup_h2d_bytes += row_bytes + block_bytes;
  plan.statistics.persistent_device_bytes += row_bytes + block_bytes;
  plan.statistics.p2p_interaction_count = operator_map.blocks.size();
  plan.statistics.p2p_tensor_bytes =
      operator_map.blocks.size() * 6 * sizeof(float);
  plan.statistics.p2p_index_bytes =
      operator_map.blocks.size() * 3 * sizeof(int);
  plan.statistics.p2p_row_metadata_bytes = row_bytes;
  plan.statistics.p2p_threads_per_block = static_operator_threads;
}

CudaP2PPlan::CudaP2PPlan(
    const FloatStaticP2PCompactPlan &compact,
    const std::span<const int> fixed_self_indices)
    : CudaP2PPlan(compact.source_count, compact.target_count,
                  fixed_self_indices, true, StaticPrecision::Float32) {
  auto &plan = *implementation_;
  plan.kind = Implementation::Kind::Compact;
  plan.compact_float.target_count = plan.target_count;
  plan.compact_float.interaction_count = compact.source_indices.size();
  const std::size_t row_bytes = compact.row_offsets.size() * sizeof(int);
  const std::size_t index_bytes = compact.source_indices.size() * sizeof(int);
  const std::size_t tensor_bytes =
      compact.source_indices.size() * 6 * sizeof(float);
  check_cuda(cudaMalloc(&plan.compact_float.row_offsets,
                        std::max(row_bytes, sizeof(int))),
             "allocate FP32 compact P2P rows");
  check_cuda(cudaMalloc(&plan.compact_float.source_indices,
                        std::max(index_bytes, sizeof(int))),
             "allocate FP32 compact P2P sources");
  const std::size_t identity_bytes =
      compact.skip_for_identity.size() * sizeof(unsigned char);
  check_cuda(cudaMalloc(&plan.compact_float.skip_for_identity,
                        std::max(identity_bytes, sizeof(unsigned char))),
             "allocate FP32 compact P2P identity markers");
  check_cuda(cudaMalloc(&plan.compact_float.tensors,
                        std::max(tensor_bytes, sizeof(float))),
             "allocate FP32 compact P2P tensors");
  if (row_bytes != 0) {
    check_cuda(cudaMemcpy(plan.compact_float.row_offsets,
                          compact.row_offsets.data(), row_bytes,
                          cudaMemcpyHostToDevice),
               "upload FP32 compact P2P rows");
  }
  if (index_bytes != 0) {
    check_cuda(cudaMemcpy(plan.compact_float.source_indices,
                          compact.source_indices.data(), index_bytes,
                          cudaMemcpyHostToDevice),
               "upload FP32 compact P2P sources");
    check_cuda(cudaMemcpy(plan.compact_float.skip_for_identity,
                          compact.skip_for_identity.data(), identity_bytes,
                          cudaMemcpyHostToDevice),
               "upload FP32 compact P2P identity markers");
    for (std::size_t component = 0; component < 6; ++component) {
      check_cuda(cudaMemcpy(
                     plan.compact_float.tensors +
                         component * compact.source_indices.size(),
                     compact.tensors[component].data(),
                     compact.tensors[component].size() * sizeof(float),
                     cudaMemcpyHostToDevice),
                 "upload FP32 compact P2P tensor component");
    }
  }
  plan.statistics.setup_h2d_bytes +=
      row_bytes + index_bytes + identity_bytes + tensor_bytes;
  plan.statistics.persistent_device_bytes +=
      row_bytes + index_bytes + identity_bytes + tensor_bytes;
  plan.statistics.p2p_interaction_count = compact.source_indices.size();
  plan.statistics.p2p_tensor_bytes = tensor_bytes;
  plan.statistics.p2p_index_bytes = index_bytes + identity_bytes;
  plan.statistics.p2p_row_metadata_bytes = row_bytes;
  plan.statistics.p2p_threads_per_block = static_operator_threads;
}

CudaP2PPlan::CudaP2PPlan(
    const StaticP2PLeafPlan& leaf,
    const std::span<const int> fixed_self_indices)
    : CudaP2PPlan(
          leaf.source_count, leaf.target_count, fixed_self_indices, true)
{
  auto& plan = *implementation_;
  plan.kind = Implementation::Kind::Leaf;
  upload_cuda_leaf(leaf, plan.leaf, plan.statistics,
                   "allocate leaf P2P data", "upload leaf P2P data");
}

CudaP2PPlan::CudaP2PPlan(
    const FloatStaticP2PLeafPlan &leaf,
    const std::span<const int> fixed_self_indices)
    : CudaP2PPlan(leaf.source_count, leaf.target_count, fixed_self_indices,
                  true, StaticPrecision::Float32) {
  auto &plan = *implementation_;
  plan.kind = Implementation::Kind::Leaf;
  upload_cuda_leaf(leaf, plan.leaf_float, plan.statistics,
                   "allocate FP32 leaf P2P data", "upload FP32 leaf P2P data");
}

CudaP2PPlan::CudaP2PPlan(
    const StaticP2PSignedTensorDictionaryPlan &dictionary,
    const bool target_owned,
    const bool power2_microtiles)
    : CudaP2PPlan(
          dictionary.source_count,
          dictionary.target_count,
          {},
          false) {

  auto &plan = *implementation_;
  plan.kind = Implementation::Kind::TensorDictionary;
  plan.dynamic_self_identities = false;

  upload_cuda_signed_dictionary<double, Vec3>(
      dictionary,
      plan.dictionary,
      plan.statistics,
      target_owned,
      power2_microtiles);
}

CudaP2PPlan::CudaP2PPlan(
    const FloatStaticP2PSignedTensorDictionaryPlan &dictionary,
    const bool target_owned,
    const bool power2_microtiles)
    : CudaP2PPlan(
          dictionary.source_count,
          dictionary.target_count,
          {},
          false,
          StaticPrecision::Float32) {

  auto &plan = *implementation_;
  plan.kind = Implementation::Kind::TensorDictionary;
  plan.dynamic_self_identities = false;

  upload_cuda_signed_dictionary<float, FloatVec3>(
      dictionary,
      plan.dictionary_float,
      plan.statistics,
      target_owned,
      power2_microtiles);
}

CudaP2PPlan::CudaP2PPlan(const StaticP2PBsrPlan& bsr)
    : CudaP2PPlan(
          bsr.source_count, bsr.target_count, bsr.target_source_indices, false)
{
  auto& plan = *implementation_;
  plan.kind = Implementation::Kind::Bsr;
  plan.dynamic_self_identities = false;
  plan.bsr.target_count = bsr.target_count;
  plan.bsr.source_count = bsr.source_count;
  plan.bsr.interaction_count = static_cast<int>(bsr.source_indices.size());
  const std::size_t row_bytes = bsr.row_offsets.size() * sizeof(int);
  const std::size_t index_bytes = bsr.source_indices.size() * sizeof(int);
  const std::size_t tensor_bytes = bsr.values.size() * sizeof(double);
  check_cuda(cudaMalloc(&plan.bsr.row_offsets,
                        std::max(row_bytes, sizeof(int))),
             "allocate cuSPARSE BSR rows");
  check_cuda(cudaMalloc(&plan.bsr.source_indices,
                        std::max(index_bytes, sizeof(int))),
             "allocate cuSPARSE BSR sources");
  check_cuda(cudaMalloc(&plan.bsr.values,
                        std::max(tensor_bytes, sizeof(double))),
             "allocate cuSPARSE BSR values");
  const auto upload = [](void* destination, const void* source,
                         const std::size_t bytes, const char* operation) {
    if (bytes != 0) {
      check_cuda(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice),
                 operation);
    }
  };
  upload(plan.bsr.row_offsets, bsr.row_offsets.data(), row_bytes,
         "upload cuSPARSE BSR rows");
  upload(plan.bsr.source_indices, bsr.source_indices.data(), index_bytes,
         "upload cuSPARSE BSR sources");
  upload(plan.bsr.values, bsr.values.data(), tensor_bytes,
         "upload cuSPARSE BSR values");
  initialise_bsr_p2p_resources(
      plan.bsr, plan.moments, plan.fields, "create cuSPARSE P2P handle",
      "create cuSPARSE P2P BSR descriptor");

  plan.statistics.setup_h2d_bytes += row_bytes + index_bytes + tensor_bytes;
  plan.statistics.persistent_device_bytes +=
      row_bytes + index_bytes + tensor_bytes + plan.bsr.workspace_size;
  plan.statistics.p2p_scratch_bytes = plan.bsr.workspace_size;
  plan.statistics.p2p_interaction_count = bsr.source_indices.size();
  plan.statistics.p2p_tensor_bytes = tensor_bytes;
  plan.statistics.p2p_index_bytes = index_bytes;
  plan.statistics.p2p_row_metadata_bytes = row_bytes;
  plan.statistics.p2p_threads_per_block = 0;
}

CudaP2PPlan::CudaP2PPlan(const FloatStaticP2PBsrPlan &bsr)
    : CudaP2PPlan(bsr.source_count, bsr.target_count,
                  bsr.target_source_indices, false,
                  StaticPrecision::Float32) {
  auto &plan = *implementation_;
  plan.kind = Implementation::Kind::Bsr;
  plan.dynamic_self_identities = false;
  plan.bsr_float.target_count = bsr.target_count;
  plan.bsr_float.source_count = bsr.source_count;
  plan.bsr_float.interaction_count =
      static_cast<int>(bsr.source_indices.size());
  const std::size_t row_bytes = bsr.row_offsets.size() * sizeof(int);
  const std::size_t index_bytes = bsr.source_indices.size() * sizeof(int);
  const std::size_t tensor_bytes = bsr.values.size() * sizeof(float);
  check_cuda(cudaMalloc(&plan.bsr_float.row_offsets,
                        std::max(row_bytes, sizeof(int))),
             "allocate FP32 cuSPARSE BSR rows");
  check_cuda(cudaMalloc(&plan.bsr_float.source_indices,
                        std::max(index_bytes, sizeof(int))),
             "allocate FP32 cuSPARSE BSR sources");
  check_cuda(cudaMalloc(&plan.bsr_float.values,
                        std::max(tensor_bytes, sizeof(float))),
             "allocate FP32 cuSPARSE BSR values");
  const auto upload = [](void *destination, const void *source,
                         const std::size_t bytes, const char *operation) {
    if (bytes != 0) {
      check_cuda(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice),
                 operation);
    }
  };
  upload(plan.bsr_float.row_offsets, bsr.row_offsets.data(), row_bytes,
         "upload FP32 cuSPARSE BSR rows");
  upload(plan.bsr_float.source_indices, bsr.source_indices.data(), index_bytes,
         "upload FP32 cuSPARSE BSR sources");
  upload(plan.bsr_float.values, bsr.values.data(), tensor_bytes,
         "upload FP32 cuSPARSE BSR values");
  initialise_bsr_p2p_resources(
      plan.bsr_float, plan.moments_float, plan.fields_float,
      "create FP32 cuSPARSE P2P handle",
      "create FP32 cuSPARSE P2P BSR descriptor");
  plan.statistics.setup_h2d_bytes += row_bytes + index_bytes + tensor_bytes;
  plan.statistics.persistent_device_bytes +=
      row_bytes + index_bytes + tensor_bytes + plan.bsr_float.workspace_size;
  plan.statistics.p2p_scratch_bytes = plan.bsr_float.workspace_size;
  plan.statistics.p2p_interaction_count = bsr.source_indices.size();
  plan.statistics.p2p_tensor_bytes = tensor_bytes;
  plan.statistics.p2p_index_bytes = index_bytes;
  plan.statistics.p2p_row_metadata_bytes = row_bytes;
  plan.statistics.p2p_threads_per_block = 0;
}

CudaP2PPlan::CudaP2PPlan(const StaticFmmTopology &topology,
                         const StaticPrecision precision,
                         const std::span<const int> fixed_self_indices)
    : CudaP2PPlan(static_cast<int>(topology.sorted_source_positions.size()),
                  static_cast<int>(topology.sorted_target_positions.size()),
                  fixed_self_indices, true, precision) {
  auto &plan = *implementation_;
  plan.kind = Implementation::Kind::PointGeometry;
  if (plan.fp32) {
    upload_cuda_point_geometry(topology, plan.point_geometry_float,
                               plan.statistics,
                               "allocate position-based P2P data",
                               "upload position-based P2P data");
  } else {
    upload_cuda_point_geometry(topology, plan.point_geometry, plan.statistics,
                               "allocate position-based P2P data",
                               "upload position-based P2P data");
  }
}

CudaP2PPlan::~CudaP2PPlan() {
  if (implementation_ == nullptr) {
    return;
  }
  auto& plan = *implementation_;
  cancel_evaluate();
  release_p2p_device_view(plan.point_geometry);
  release_p2p_device_view(plan.point_geometry_float);
  release_p2p_device_view(plan.canonical);
  release_p2p_device_view(plan.compact);
  release_p2p_device_view(plan.leaf);
  release_p2p_device_view(plan.dictionary);
  release_p2p_device_view(plan.bsr);
  release_p2p_device_view(plan.canonical_float);
  release_p2p_device_view(plan.compact_float);
  release_p2p_device_view(plan.leaf_float);
  release_p2p_device_view(plan.dictionary_float);
  release_p2p_device_view(plan.bsr_float);
  cudaFree(plan.moments);
  cudaFree(plan.self_indices);
  cudaFree(plan.fields);
  cudaFreeHost(plan.pinned_moments);
  cudaFreeHost(plan.pinned_self_indices);
  cudaFreeHost(plan.pinned_fields);
  cudaFree(plan.moments_float);
  cudaFree(plan.fields_float);
  cudaFreeHost(plan.pinned_moments_float);
  cudaFreeHost(plan.pinned_fields_float);
  cudaEventDestroy(plan.start);
  cudaEventDestroy(plan.h2d);
  cudaEventDestroy(plan.kernel);
  cudaEventDestroy(plan.finished);
  cudaEventDestroy(plan.d2h);
  cudaStreamDestroy(plan.stream);
  delete implementation_;
}

// Enqueue one complete near-field evaluation (H2D, kernel, D2H) on the plan's
// stream and return without waiting.  A fixed identity map must be passed
// unchanged on every call; a dynamic map is staged and uploaded each time.
// Any failure after `pending` is set drains the stream so the plan stays
// reusable.  `finish_evaluate` collects the result and the event timings.
void CudaP2PPlan::begin_evaluate(
    const std::span<const Vec3> moments,
    const std::span<const int> target_source_indices) {
  auto &plan = *implementation_;
  if (moments.size() != static_cast<std::size_t>(plan.source_count) ||
      target_source_indices.size() !=
          static_cast<std::size_t>(plan.target_count)) {
    throw std::invalid_argument("CUDA static P2P dimensions are inconsistent");
  }
  const bool uses_self_identities =
      plan.kind != Implementation::Kind::TensorDictionary;
  if (uses_self_identities && !plan.dynamic_self_identities &&
      !std::equal(target_source_indices.begin(), target_source_indices.end(),
                  plan.fixed_self_indices.begin(),
                  plan.fixed_self_indices.end())) {
    throw std::invalid_argument(
        "fixed CUDA P2P identity map changed; rebuild the static plan");
  }
  if (plan.pending) {
    throw std::logic_error("CUDA static P2P evaluation is already pending");
  }
  std::copy(moments.begin(), moments.end(), plan.pinned_moments);
  if (uses_self_identities && plan.dynamic_self_identities) {
    std::copy(target_source_indices.begin(), target_source_indices.end(),
              plan.pinned_self_indices);
  }
  plan.pending = true;
  const bool detailed = plan.timing_level == TimingLevel::Detailed;
  try {
    cuda_detail::record_diagnostic(detailed, plan.start, plan.stream,
                                   "record P2P start");
    check_cuda(cudaMemcpyAsync(plan.moments, plan.pinned_moments,
                               moments.size_bytes(), cudaMemcpyHostToDevice,
                               plan.stream),
               "upload P2P moments");
    if (uses_self_identities && plan.dynamic_self_identities) {
      check_cuda(cudaMemcpyAsync(
                     plan.self_indices, plan.pinned_self_indices,
                     target_source_indices.size_bytes(),
                     cudaMemcpyHostToDevice, plan.stream),
                 "upload P2P identities");
    }
    cuda_detail::record_diagnostic(detailed, plan.h2d, plan.stream,
                                   "record P2P upload");
    switch (plan.kind) {
    case Implementation::Kind::Canonical:
      launch_static_p2p(plan.canonical, plan.moments, plan.self_indices,
                        plan.fields, plan.stream);
      break;
    case Implementation::Kind::Compact:
      launch_compact_p2p(plan.compact, plan.moments, plan.self_indices,
                         plan.fields, plan.stream);
      break;
    case Implementation::Kind::Leaf:
      launch_leaf_p2p(plan.leaf, plan.moments, plan.self_indices,
                      plan.fields, plan.stream);
      break;
    case Implementation::Kind::PointGeometry:
      launch_point_geometry_p2p(plan.point_geometry, plan.moments,
                                plan.self_indices, plan.fields, plan.stream);
      break;
    case Implementation::Kind::TensorDictionary:
      launch_signed_dictionary_p2p(plan.dictionary, plan.moments, plan.fields,
                                   plan.stream);
      break;
    case Implementation::Kind::Bsr:
      launch_bsr_p2p(plan.bsr, plan.fields, plan.stream);
      break;
    }
    cuda_detail::record_diagnostic(detailed, plan.kernel, plan.stream,
                                   "record P2P kernel");
    check_cuda(cudaMemcpyAsync(plan.pinned_fields, plan.fields,
                               static_cast<std::size_t>(plan.target_count) *
                                   sizeof(Vec3),
                               cudaMemcpyDeviceToHost, plan.stream),
               "download P2P fields");
    check_cuda(cudaEventRecord(plan.d2h, plan.stream), "record P2P download");
    cuda_detail::record_diagnostic(detailed, plan.finished, plan.stream,
                                   "record P2P finished");
  } catch (...) {
    cancel_evaluate();
    throw;
  }
  plan.statistics.evaluation_h2d_bytes = moments.size_bytes() +
      (uses_self_identities && plan.dynamic_self_identities
           ? target_source_indices.size_bytes()
           : 0);
  plan.statistics.evaluation_d2h_bytes =
      static_cast<std::size_t>(plan.target_count) * sizeof(Vec3);
  ++plan.statistics.evaluation_h2d_calls;
  ++plan.statistics.evaluation_d2h_calls;
}

void CudaP2PPlan::finish_evaluate(const std::span<Vec3> fields) {
  auto &plan = *implementation_;
  if (!plan.pending) {
    throw std::logic_error("CUDA static P2P has no pending evaluation");
  }
  if (fields.size() != static_cast<std::size_t>(plan.target_count)) {
    throw std::invalid_argument("CUDA static P2P output size is inconsistent");
  }
  check_cuda(cudaEventSynchronize(plan.d2h), "synchronise static P2P");
  std::copy(plan.pinned_fields, plan.pinned_fields + fields.size(),
            fields.begin());
  if (plan.timing_level == TimingLevel::Detailed) {
    read_p2p_timings(plan, "time static P2P phase");
  }
  plan.pending = false;
}

// FP32 form of the protocol above over the FP32 buffers and views.
void CudaP2PPlan::begin_evaluate(
    const std::span<const FloatVec3> moments,
    const std::span<const int> target_source_indices) {
  auto &plan = *implementation_;
  if (!plan.fp32 ||
      moments.size() != static_cast<std::size_t>(plan.source_count) ||
      target_source_indices.size() !=
          static_cast<std::size_t>(plan.target_count)) {
    throw std::invalid_argument("CUDA FP32 P2P dimensions are inconsistent");
  }
  const bool uses_self_identities =
      plan.kind != Implementation::Kind::TensorDictionary;
  if (uses_self_identities && !plan.dynamic_self_identities &&
      !std::equal(target_source_indices.begin(), target_source_indices.end(),
                  plan.fixed_self_indices.begin(),
                  plan.fixed_self_indices.end())) {
    throw std::invalid_argument(
        "fixed CUDA FP32 P2P identity map changed; rebuild the plan");
  }
  if (plan.pending) {
    throw std::logic_error("CUDA FP32 P2P evaluation is already pending");
  }
  std::copy(moments.begin(), moments.end(), plan.pinned_moments_float);
  if (uses_self_identities && plan.dynamic_self_identities) {
    std::copy(target_source_indices.begin(), target_source_indices.end(),
              plan.pinned_self_indices);
  }
  plan.pending = true;
  const bool detailed = plan.timing_level == TimingLevel::Detailed;
  try {
    cuda_detail::record_diagnostic(detailed, plan.start, plan.stream,
                                   "record FP32 P2P start");
    check_cuda(cudaMemcpyAsync(plan.moments_float, plan.pinned_moments_float,
                               moments.size_bytes(), cudaMemcpyHostToDevice,
                               plan.stream),
               "upload FP32 P2P moments");
    if (uses_self_identities && plan.dynamic_self_identities) {
      check_cuda(cudaMemcpyAsync(
                     plan.self_indices, plan.pinned_self_indices,
                     target_source_indices.size_bytes(),
                     cudaMemcpyHostToDevice, plan.stream),
                 "upload FP32 P2P identities");
    }
    cuda_detail::record_diagnostic(detailed, plan.h2d, plan.stream,
                                   "record FP32 P2P upload");
    switch (plan.kind) {
    case Implementation::Kind::Canonical:
      launch_static_p2p(plan.canonical_float, plan.moments_float,
                        plan.self_indices, plan.fields_float, plan.stream);
      break;
    case Implementation::Kind::Compact:
      launch_compact_p2p(plan.compact_float, plan.moments_float,
                         plan.self_indices, plan.fields_float, plan.stream);
      break;
    case Implementation::Kind::Leaf:
      launch_leaf_p2p(plan.leaf_float, plan.moments_float, plan.self_indices,
                      plan.fields_float, plan.stream);
      break;
    case Implementation::Kind::PointGeometry:
      launch_point_geometry_p2p(plan.point_geometry_float, plan.moments_float,
                                plan.self_indices, plan.fields_float,
                                plan.stream);
      break;
    case Implementation::Kind::TensorDictionary:
      launch_signed_dictionary_p2p(plan.dictionary_float, plan.moments_float,
                                   plan.fields_float, plan.stream);
      break;
    case Implementation::Kind::Bsr:
      launch_bsr_p2p(plan.bsr_float, plan.fields_float, plan.stream);
      break;
    }
    cuda_detail::record_diagnostic(detailed, plan.kernel, plan.stream,
                                   "record FP32 P2P kernel");
    check_cuda(cudaMemcpyAsync(
                   plan.pinned_fields_float, plan.fields_float,
                   static_cast<std::size_t>(plan.target_count) *
                       sizeof(FloatVec3),
                   cudaMemcpyDeviceToHost, plan.stream),
               "download FP32 P2P fields");
    check_cuda(cudaEventRecord(plan.d2h, plan.stream),
               "record FP32 P2P download");
    cuda_detail::record_diagnostic(detailed, plan.finished, plan.stream,
                                   "record FP32 P2P finished");
  } catch (...) {
    cancel_evaluate();
    throw;
  }
  plan.statistics.evaluation_h2d_bytes = moments.size_bytes() +
      (uses_self_identities && plan.dynamic_self_identities
           ? target_source_indices.size_bytes()
           : 0);
  plan.statistics.evaluation_d2h_bytes =
      static_cast<std::size_t>(plan.target_count) * sizeof(FloatVec3);
  ++plan.statistics.evaluation_h2d_calls;
  ++plan.statistics.evaluation_d2h_calls;
}

void CudaP2PPlan::finish_evaluate(const std::span<FloatVec3> fields) {
  auto &plan = *implementation_;
  if (!plan.fp32 || !plan.pending) {
    throw std::logic_error("CUDA FP32 P2P has no pending evaluation");
  }
  if (fields.size() != static_cast<std::size_t>(plan.target_count)) {
    throw std::invalid_argument("CUDA FP32 P2P output size is inconsistent");
  }
  check_cuda(cudaEventSynchronize(plan.d2h), "synchronise FP32 P2P");
  std::copy(plan.pinned_fields_float,
            plan.pinned_fields_float + fields.size(), fields.begin());
  if (plan.timing_level == TimingLevel::Detailed) {
    read_p2p_timings(plan, "time FP32 P2P phase");
  }
  plan.pending = false;
}

// Abandon a pending evaluation: drain the stream so no kernel still reads the
// staging buffers, then clear the flag.  Called on the error paths and from
// the destructor.
void CudaP2PPlan::cancel_evaluate() noexcept {
  if (implementation_ == nullptr || !implementation_->pending) {
    return;
  }
  cudaStreamSynchronize(implementation_->stream);
  implementation_->pending = false;
}

// Synchronous convenience form of begin/finish.
void CudaP2PPlan::evaluate(const std::span<const Vec3> moments,
                           const std::span<const int> target_source_indices,
                           const std::span<Vec3> fields) {
  begin_evaluate(moments, target_source_indices);
  try {
    finish_evaluate(fields);
  } catch (...) {
    cancel_evaluate();
    throw;
  }
}

void CudaP2PPlan::evaluate(
    const std::span<const FloatVec3> moments,
    const std::span<const int> target_source_indices,
    const std::span<FloatVec3> fields) {
  begin_evaluate(moments, target_source_indices);
  try {
    finish_evaluate(fields);
  } catch (...) {
    cancel_evaluate();
    throw;
  }
}

const CudaPlanStatistics &CudaP2PPlan::statistics() const noexcept {
  return implementation_->statistics;
}

const CudaEvaluationTimings &CudaP2PPlan::timings() const noexcept {
  return implementation_->timings;
}

TimingLevel CudaP2PPlan::timing_level() const noexcept {
  return implementation_->timing_level;
}

void CudaP2PPlan::set_timing_level(const TimingLevel level) noexcept {
  implementation_->timing_level = level;
  implementation_->timings = {};
  implementation_->timings.timing_level = level;
}

} // namespace cdfmm
