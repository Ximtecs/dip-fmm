// SPDX-License-Identifier: Apache-2.0
//
// Portable executors for the two dictionary packings of the exact near-field
// operator.  Both keep the canonical list-1 topology as dense target/source
// leaf blocks and replace every stored six-component tensor by a token into a
// shared dictionary of distinct tensors:
//
//   * the plain Tensor6 dictionary stores unsigned magnitudes plus a per-token
//     six-bit sign mask and is decoded per pair (`tensor6_token_id`,
//     `tensor6_token_sign_mask`);
//   * the signed dictionary stores already-signed variants and one-, two- or
//     four-byte ids, so its inner loop is a pure gather/FMA over a tile of
//     targets with no sign reconstruction.
//
// Neither executor reads geometry: the identity semantics of a block are
// carried by `skip_for_identity`, and the signed dictionary encodes fixed point
// self pairs as an exact zero variant at construction, so it needs no identity
// map at all.  The arithmetic and the per-target accumulation order of every
// SIMD specialisation match the portable loop they replace.

#include "cdfmm/backend/cpu/p2p.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "cdfmm/plan/p2p/tensor_dictionary.hpp"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace cdfmm {

// Plain Tensor6 dictionary, FP32.  Tokens are laid out target-major within a
// leaf block: the `source_count` tokens of one local target are contiguous at
// `tensor_offset + local_target * source_count`.  One thread owns a whole
// target leaf, so no two threads accumulate into the same target.
void apply_static_p2p_tensor_dictionary_plan(
    const FloatStaticP2PTensorDictionaryPlan &plan,
    const std::span<const FloatVec3> dipole_moments,
    const std::span<FloatVec3> H,
    const std::span<const int> target_source_indices) {
  if (dipole_moments.size() != static_cast<std::size_t>(plan.source_count) ||
      H.size() != static_cast<std::size_t>(plan.target_count) ||
      (!target_source_indices.empty() && target_source_indices.size() != H.size())) {
    throw std::invalid_argument("FP32 Tensor6 dictionary P2P dimensions are inconsistent");
  }
#pragma omp parallel for schedule(static) if (plan.target_begins.size() >= 8)
  for (int target_leaf = 0;
       target_leaf < static_cast<int>(plan.target_begins.size()); ++target_leaf) {
    const int target_begin = plan.target_begins[static_cast<std::size_t>(target_leaf)];
    const int target_count = plan.target_counts[static_cast<std::size_t>(target_leaf)];
    for (int local_target = 0; local_target < target_count; ++local_target) {
      const int target = target_begin + local_target;
      const int self = target_source_indices.empty() ? -1 : target_source_indices[static_cast<std::size_t>(target)];
      FloatVec3 field{};
      for (int block_index = plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
           block_index < plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf) + 1]; ++block_index) {
        const StaticP2PLeafBlock &block = plan.blocks[static_cast<std::size_t>(block_index)];
        const std::size_t begin = block.tensor_offset + static_cast<std::size_t>(local_target) * block.source_count;
        for (int local_source = 0; local_source < block.source_count; ++local_source) {
          const int source = block.source_begin + local_source;
          if (block.skip_for_identity != 0 && source == self) continue;
          const std::uint32_t token = plan.tokens[begin + local_source];
          const auto id = tensor6_token_id(token);
          const auto signs = tensor6_token_sign_mask(token);
          // Components are ordered xx, xy, xz, yy, yz, zz; bit `component` of
          // the mask flips the sign of that component of the stored variant.
          const auto value = [&](int component) {
            const float coefficient = plan.tensors[static_cast<std::size_t>(component)][id];
            return (signs & (1U << component)) != 0U ? -coefficient : coefficient;
          };
          const auto moment = dipole_moments[static_cast<std::size_t>(source)];
          const float xx = value(0), xy = value(1), xz = value(2),
                      yy = value(3), yz = value(4), zz = value(5);
          field.x += xx * moment.x + xy * moment.y + xz * moment.z;
          field.y += xy * moment.x + yy * moment.y + yz * moment.z;
          field.z += xz * moment.x + yz * moment.y + zz * moment.z;
        }
      }
      H[static_cast<std::size_t>(target)] += field;
    }
  }
}

// Plain Tensor6 dictionary, FP64.  Same layout and semantics as the FP32
// executor above.
void apply_static_p2p_tensor_dictionary_plan(
    const StaticP2PTensorDictionaryPlan &plan,
    const std::span<const Vec3> dipole_moments, const std::span<Vec3> H,
    const std::span<const int> target_source_indices) {
  if (dipole_moments.size() != static_cast<std::size_t>(plan.source_count) ||
      H.size() != static_cast<std::size_t>(plan.target_count) ||
      (!target_source_indices.empty() && target_source_indices.size() != H.size())) {
    throw std::invalid_argument("Tensor6 dictionary P2P dimensions are inconsistent");
  }
#pragma omp parallel for schedule(static) if (plan.target_begins.size() >= 8)
  for (int target_leaf = 0;
       target_leaf < static_cast<int>(plan.target_begins.size()); ++target_leaf) {
    const int target_begin = plan.target_begins[static_cast<std::size_t>(target_leaf)];
    const int target_count = plan.target_counts[static_cast<std::size_t>(target_leaf)];
    for (int local_target = 0; local_target < target_count; ++local_target) {
      const int target = target_begin + local_target;
      const int self = target_source_indices.empty() ? -1 :
          target_source_indices[static_cast<std::size_t>(target)];
      Vec3 field{};
      for (int block_index = plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
           block_index < plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf) + 1];
           ++block_index) {
        const StaticP2PLeafBlock &block = plan.blocks[static_cast<std::size_t>(block_index)];
        const std::size_t begin = block.tensor_offset +
            static_cast<std::size_t>(local_target) * block.source_count;
        for (int local_source = 0; local_source < block.source_count; ++local_source) {
          const int source = block.source_begin + local_source;
          if (block.skip_for_identity != 0 && source == self) {
            continue;
          }
          const std::uint32_t token = plan.tokens[begin + local_source];
          const std::uint32_t id = tensor6_token_id(token);
          const std::uint8_t signs = tensor6_token_sign_mask(token);
          const auto signed_value = [&](const int component) {
            const double value = plan.tensors[static_cast<std::size_t>(component)][id];
            return (signs & (1U << component)) != 0U ? -value : value;
          };
          const Vec3 moment = dipole_moments[static_cast<std::size_t>(source)];
          const double xx = signed_value(0), xy = signed_value(1),
                       xz = signed_value(2), yy = signed_value(3),
                       yz = signed_value(4), zz = signed_value(5);
          field.x += xx * moment.x + xy * moment.y + xz * moment.z;
          field.y += xy * moment.x + yy * moment.y + yz * moment.z;
          field.z += xz * moment.x + yz * moment.y + zz * moment.z;
        }
      }
      H[static_cast<std::size_t>(target)] += field;
    }
  }
}

namespace {

// Signed dictionary layout, shared by every executor below.  The plan splits
// each target leaf into tiles of at most `target_tile_size` (<= 128) targets;
// `tile_leaf_indices[work]` / `tile_target_offsets[work]` name one tile, and
// one OpenMP iteration owns one tile, so the disjoint tiles need no atomics.
// Within a leaf block the tokens are source-major: the tokens of one source
// against all `target_count` targets of the leaf are contiguous, so a tile of
// consecutive targets reads a contiguous token run per source.

// Whole-tile variant: accumulates the complete tile in three stack arrays and
// lets the compiler vectorise the target lanes.  Production uses the microtile
// executor below; this one is retained as the simpler reference and as the
// measurable alternative exercised by `benchmark_p2p`.
template <typename Scalar, typename Vector, typename Plan, typename Token>
void apply_signed_tensor_dictionary_whole_tile_impl(
    const Plan &plan, const std::span<const Vector> moments,
    const std::span<Vector> fields, const std::span<const Token> tokens) {
  constexpr int max_tile_size = 128;
  const int work_count = static_cast<int>(plan.tile_leaf_indices.size());
  const Scalar *const xx = plan.tensors[0].data();
  const Scalar *const xy = plan.tensors[1].data();
  const Scalar *const xz = plan.tensors[2].data();
  const Scalar *const yy = plan.tensors[3].data();
  const Scalar *const yz = plan.tensors[4].data();
  const Scalar *const zz = plan.tensors[5].data();
  const Token *const token_data = tokens.data();
#pragma omp parallel for schedule(static) if (work_count >= 8)
  for (int work = 0; work < work_count; ++work) {
    const int target_leaf = plan.tile_leaf_indices[static_cast<std::size_t>(work)];
    const int local_begin = plan.tile_target_offsets[static_cast<std::size_t>(work)];
    const int target_begin = plan.target_begins[static_cast<std::size_t>(target_leaf)];
    const int target_count = plan.target_counts[static_cast<std::size_t>(target_leaf)];
    const int lanes = std::min(plan.target_tile_size, target_count - local_begin);
    std::array<Scalar, max_tile_size> Hx{};
    std::array<Scalar, max_tile_size> Hy{};
    std::array<Scalar, max_tile_size> Hz{};
    for (int block_index = plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
         block_index < plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf) + 1];
         ++block_index) {
      const StaticP2PLeafBlock &block = plan.blocks[static_cast<std::size_t>(block_index)];
      for (int local_source = 0; local_source < block.source_count; ++local_source) {
        const Vector moment = moments[static_cast<std::size_t>(block.source_begin + local_source)];
        const Token *const block_tokens = token_data + block.tensor_offset +
            static_cast<std::size_t>(local_source) * target_count + local_begin;
#pragma omp simd
        for (int lane = 0; lane < lanes; ++lane) {
          const std::size_t variant = block_tokens[lane];
          Hx[static_cast<std::size_t>(lane)] += xx[variant] * moment.x +
              xy[variant] * moment.y + xz[variant] * moment.z;
          Hy[static_cast<std::size_t>(lane)] += xy[variant] * moment.x +
              yy[variant] * moment.y + yz[variant] * moment.z;
          Hz[static_cast<std::size_t>(lane)] += xz[variant] * moment.x +
              yz[variant] * moment.y + zz[variant] * moment.z;
        }
      }
    }
    for (int lane = 0; lane < lanes; ++lane) {
      fields[static_cast<std::size_t>(target_begin + local_begin + lane)] +=
          Vector{Hx[static_cast<std::size_t>(lane)], Hy[static_cast<std::size_t>(lane)],
                 Hz[static_cast<std::size_t>(lane)]};
    }
  }
}

#if defined(__AVX2__) && defined(__FMA__)

// Widen four (FP64 lanes) or eight (FP32 lanes) narrow tokens to the 32-bit
// gather indices the AVX2 gathers need.  Narrow tokens are copied through an
// integer to avoid an unaligned vector load of fewer than 16 bytes.
template <typename Token>
[[nodiscard]] inline __m128i load_four_variant_ids(
    const Token *const tokens) noexcept {
  if constexpr (sizeof(Token) == 1) {
    std::uint32_t packed{};
    std::memcpy(&packed, tokens, sizeof(packed));
    return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(static_cast<int>(packed)));
  } else if constexpr (sizeof(Token) == 2) {
    std::uint64_t packed{};
    std::memcpy(&packed, tokens, sizeof(packed));
    return _mm_cvtepu16_epi32(_mm_cvtsi64_si128(static_cast<long long>(packed)));
  } else {
    return _mm_loadu_si128(reinterpret_cast<const __m128i *>(tokens));
  }
}

template <typename Token>
[[nodiscard]] inline __m256i load_eight_variant_ids(
    const Token *const tokens) noexcept {
  if constexpr (sizeof(Token) == 1) {
    std::uint64_t packed{};
    std::memcpy(&packed, tokens, sizeof(packed));
    return _mm256_cvtepu8_epi32(
        _mm_cvtsi64_si128(static_cast<long long>(packed)));
  } else if constexpr (sizeof(Token) == 2) {
    return _mm256_cvtepu16_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(tokens)));
  } else {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(tokens));
  }
}

// NOTE(cdfmm): the microtile bodies are kept out of line so that the six
// gathered vectors and three accumulators stay register resident per call
// instead of being spilled by the enclosing tile loop.
#if defined(__GNUC__) || defined(__clang__)
#define CDFMM_SIGNED_P2P_NOINLINE __attribute__((noinline))
#else
#define CDFMM_SIGNED_P2P_NOINLINE
#endif

// One full-width microtile of four FP64 targets: for every source of every
// block in the leaf row, gather the six signed components of the four
// variants and FMA them against the broadcast moment.  The three accumulators
// are the symmetric tensor product H = T m with T = [[xx,xy,xz],[xy,yy,yz],
// [xz,yz,zz]].
template <typename Vector, typename Plan, typename Token>
CDFMM_SIGNED_P2P_NOINLINE void apply_signed_microtile_avx2_f64(
    const Plan &plan, const std::span<const Vector> moments,
    const std::span<Vector> fields, const Token *const token_data,
    const int target_leaf, const int target_begin, const int target_count,
    const int local_begin) {
  const double *const xx = plan.tensors[0].data();
  const double *const xy = plan.tensors[1].data();
  const double *const xz = plan.tensors[2].data();
  const double *const yy = plan.tensors[3].data();
  const double *const yz = plan.tensors[4].data();
  const double *const zz = plan.tensors[5].data();
  __m256d hx = _mm256_setzero_pd();
  __m256d hy = _mm256_setzero_pd();
  __m256d hz = _mm256_setzero_pd();
  for (int block_index =
           plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
       block_index <
       plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf) + 1];
       ++block_index) {
    const StaticP2PLeafBlock &block =
        plan.blocks[static_cast<std::size_t>(block_index)];
    for (int local_source = 0; local_source < block.source_count;
         ++local_source) {
      const Vector moment =
          moments[static_cast<std::size_t>(block.source_begin + local_source)];
      const Token *const block_tokens = token_data + block.tensor_offset +
          static_cast<std::size_t>(local_source) * target_count + local_begin;
      const __m128i ids = load_four_variant_ids(block_tokens);
      const __m256d vxx = _mm256_i32gather_pd(xx, ids, 8);
      const __m256d vxy = _mm256_i32gather_pd(xy, ids, 8);
      const __m256d vxz = _mm256_i32gather_pd(xz, ids, 8);
      const __m256d vyy = _mm256_i32gather_pd(yy, ids, 8);
      const __m256d vyz = _mm256_i32gather_pd(yz, ids, 8);
      const __m256d vzz = _mm256_i32gather_pd(zz, ids, 8);
      const __m256d mx = _mm256_set1_pd(moment.x);
      const __m256d my = _mm256_set1_pd(moment.y);
      const __m256d mz = _mm256_set1_pd(moment.z);
      hx = _mm256_fmadd_pd(vxz, mz,
                           _mm256_fmadd_pd(vxy, my,
                                          _mm256_fmadd_pd(vxx, mx, hx)));
      hy = _mm256_fmadd_pd(vyz, mz,
                           _mm256_fmadd_pd(vyy, my,
                                          _mm256_fmadd_pd(vxy, mx, hy)));
      hz = _mm256_fmadd_pd(vzz, mz,
                           _mm256_fmadd_pd(vyz, my,
                                          _mm256_fmadd_pd(vxz, mx, hz)));
    }
  }
  alignas(32) std::array<double, 4> hx_values{}, hy_values{}, hz_values{};
  _mm256_store_pd(hx_values.data(), hx);
  _mm256_store_pd(hy_values.data(), hy);
  _mm256_store_pd(hz_values.data(), hz);
  for (int lane = 0; lane < 4; ++lane) {
    fields[static_cast<std::size_t>(target_begin + local_begin + lane)] +=
        Vector{hx_values[static_cast<std::size_t>(lane)],
               hy_values[static_cast<std::size_t>(lane)],
               hz_values[static_cast<std::size_t>(lane)]};
  }
}

// The FP32 counterpart: eight targets per microtile.
template <typename Vector, typename Plan, typename Token>
CDFMM_SIGNED_P2P_NOINLINE void apply_signed_microtile_avx2_f32(
    const Plan &plan, const std::span<const Vector> moments,
    const std::span<Vector> fields, const Token *const token_data,
    const int target_leaf, const int target_begin, const int target_count,
    const int local_begin) {
  const float *const xx = plan.tensors[0].data();
  const float *const xy = plan.tensors[1].data();
  const float *const xz = plan.tensors[2].data();
  const float *const yy = plan.tensors[3].data();
  const float *const yz = plan.tensors[4].data();
  const float *const zz = plan.tensors[5].data();
  __m256 hx = _mm256_setzero_ps();
  __m256 hy = _mm256_setzero_ps();
  __m256 hz = _mm256_setzero_ps();
  for (int block_index =
           plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
       block_index <
       plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf) + 1];
       ++block_index) {
    const StaticP2PLeafBlock &block =
        plan.blocks[static_cast<std::size_t>(block_index)];
    for (int local_source = 0; local_source < block.source_count;
         ++local_source) {
      const Vector moment =
          moments[static_cast<std::size_t>(block.source_begin + local_source)];
      const Token *const block_tokens = token_data + block.tensor_offset +
          static_cast<std::size_t>(local_source) * target_count + local_begin;
      const __m256i ids = load_eight_variant_ids(block_tokens);
      const __m256 vxx = _mm256_i32gather_ps(xx, ids, 4);
      const __m256 vxy = _mm256_i32gather_ps(xy, ids, 4);
      const __m256 vxz = _mm256_i32gather_ps(xz, ids, 4);
      const __m256 vyy = _mm256_i32gather_ps(yy, ids, 4);
      const __m256 vyz = _mm256_i32gather_ps(yz, ids, 4);
      const __m256 vzz = _mm256_i32gather_ps(zz, ids, 4);
      const __m256 mx = _mm256_set1_ps(moment.x);
      const __m256 my = _mm256_set1_ps(moment.y);
      const __m256 mz = _mm256_set1_ps(moment.z);
      hx = _mm256_fmadd_ps(vxz, mz,
                           _mm256_fmadd_ps(vxy, my,
                                          _mm256_fmadd_ps(vxx, mx, hx)));
      hy = _mm256_fmadd_ps(vyz, mz,
                           _mm256_fmadd_ps(vyy, my,
                                          _mm256_fmadd_ps(vxy, mx, hy)));
      hz = _mm256_fmadd_ps(vzz, mz,
                           _mm256_fmadd_ps(vyz, my,
                                          _mm256_fmadd_ps(vxz, mx, hz)));
    }
  }
  alignas(32) std::array<float, 8> hx_values{}, hy_values{}, hz_values{};
  _mm256_store_ps(hx_values.data(), hx);
  _mm256_store_ps(hy_values.data(), hy);
  _mm256_store_ps(hz_values.data(), hz);
  for (int lane = 0; lane < 8; ++lane) {
    fields[static_cast<std::size_t>(target_begin + local_begin + lane)] +=
        Vector{hx_values[static_cast<std::size_t>(lane)],
               hy_values[static_cast<std::size_t>(lane)],
               hz_values[static_cast<std::size_t>(lane)]};
  }
}

#undef CDFMM_SIGNED_P2P_NOINLINE
#endif

// Portable microtile of `lanes` <= width targets.  It is the whole path on
// non-AVX2 builds and the remainder path (a partial final microtile) on AVX2
// builds; the arithmetic is identical to the intrinsic bodies.
template <typename Scalar, typename Vector, typename Plan, typename Token>
inline void apply_signed_microtile_portable(
    const Plan &plan, const std::span<const Vector> moments,
    const std::span<Vector> fields, const Token *const token_data,
    const int target_leaf, const int target_begin, const int target_count,
    const int local_begin, const int lanes) {
  constexpr int width = std::is_same_v<Scalar, float> ? 8 : 4;
  std::array<Scalar, width> hx{}, hy{}, hz{};
  for (int block_index =
           plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
       block_index <
       plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf) + 1];
       ++block_index) {
    const StaticP2PLeafBlock &block =
        plan.blocks[static_cast<std::size_t>(block_index)];
    for (int local_source = 0; local_source < block.source_count;
         ++local_source) {
      const Vector moment =
          moments[static_cast<std::size_t>(block.source_begin + local_source)];
      const Token *const block_tokens = token_data + block.tensor_offset +
          static_cast<std::size_t>(local_source) * target_count + local_begin;
#pragma omp simd
      for (int lane = 0; lane < lanes; ++lane) {
        const std::size_t variant = block_tokens[lane];
        hx[static_cast<std::size_t>(lane)] +=
            plan.tensors[0][variant] * moment.x +
            plan.tensors[1][variant] * moment.y +
            plan.tensors[2][variant] * moment.z;
        hy[static_cast<std::size_t>(lane)] +=
            plan.tensors[1][variant] * moment.x +
            plan.tensors[3][variant] * moment.y +
            plan.tensors[4][variant] * moment.z;
        hz[static_cast<std::size_t>(lane)] +=
            plan.tensors[2][variant] * moment.x +
            plan.tensors[4][variant] * moment.y +
            plan.tensors[5][variant] * moment.z;
      }
    }
  }
  for (int lane = 0; lane < lanes; ++lane) {
    fields[static_cast<std::size_t>(target_begin + local_begin + lane)] +=
        Vector{hx[static_cast<std::size_t>(lane)],
               hy[static_cast<std::size_t>(lane)],
               hz[static_cast<std::size_t>(lane)]};
  }
}

// Microtile variant (the production CPU executor): walks each tile in
// register-width steps of 8 FP32 or 4 FP64 targets and finishes any remainder
// with the portable body.
template <typename Scalar, typename Vector, typename Plan, typename Token>
void apply_signed_tensor_dictionary_microtile_impl(
    const Plan &plan, const std::span<const Vector> moments,
    const std::span<Vector> fields, const std::span<const Token> tokens) {
  constexpr int width = std::is_same_v<Scalar, float> ? 8 : 4;
  const int work_count = static_cast<int>(plan.tile_leaf_indices.size());
  const Token *const token_data = tokens.data();
#pragma omp parallel for schedule(static) if (work_count >= 8)
  for (int work = 0; work < work_count; ++work) {
    const int target_leaf = plan.tile_leaf_indices[static_cast<std::size_t>(work)];
    const int tile_begin = plan.tile_target_offsets[static_cast<std::size_t>(work)];
    const int target_begin = plan.target_begins[static_cast<std::size_t>(target_leaf)];
    const int target_count = plan.target_counts[static_cast<std::size_t>(target_leaf)];
    const int tile_end =
        std::min(target_count, tile_begin + plan.target_tile_size);
    int local_begin = tile_begin;
    for (; local_begin + width <= tile_end; local_begin += width) {
#if defined(__AVX2__) && defined(__FMA__)
      if constexpr (std::is_same_v<Scalar, float>) {
        apply_signed_microtile_avx2_f32(
            plan, moments, fields, token_data, target_leaf, target_begin,
            target_count, local_begin);
      } else {
        apply_signed_microtile_avx2_f64(
            plan, moments, fields, token_data, target_leaf, target_begin,
            target_count, local_begin);
      }
#else
      apply_signed_microtile_portable<Scalar>(
          plan, moments, fields, token_data, target_leaf, target_begin,
          target_count, local_begin, width);
#endif
    }
    if (local_begin < tile_end) {
      apply_signed_microtile_portable<Scalar>(
          plan, moments, fields, token_data, target_leaf, target_begin,
          target_count, local_begin, tile_end - local_begin);
    }
  }
}

// Shape checks shared by both signed executors.  The tile size bound matches
// the whole-tile stack arrays; the token width must be one the plan can store.
template <typename Scalar, typename Vector, typename Plan>
void validate_signed_tensor_dictionary(const Plan &plan,
                                       const std::span<const Vector> moments,
                                       const std::span<Vector> fields) {
  if (moments.size() != static_cast<std::size_t>(plan.source_count) ||
      fields.size() != static_cast<std::size_t>(plan.target_count)) {
    throw std::invalid_argument("signed Tensor6 dictionary P2P dimensions are inconsistent");
  }
  if (plan.target_tile_size <= 0 || plan.target_tile_size > 128 ||
      plan.tile_leaf_indices.size() != plan.tile_target_offsets.size()) {
    throw std::invalid_argument("signed Tensor6 dictionary plan is malformed");
  }
  if (plan.token_width_bytes != 1 && plan.token_width_bytes != 2 &&
      plan.token_width_bytes != 4) {
    throw std::invalid_argument("signed Tensor6 dictionary token width is invalid");
  }
}

// Dispatch on the token width the plan chose at construction (the narrowest
// that indexes every variant) so the inner loops are instantiated per width.
template <typename Scalar, typename Vector, typename Plan>
void apply_signed_tensor_dictionary(const Plan &plan,
                                    const std::span<const Vector> moments,
                                    const std::span<Vector> fields) {
  validate_signed_tensor_dictionary<Scalar>(plan, moments, fields);
  if (plan.token_width_bytes == 1) {
    apply_signed_tensor_dictionary_microtile_impl<Scalar>(
        plan, moments, fields, std::span{plan.tokens8});
  } else if (plan.token_width_bytes == 2) {
    apply_signed_tensor_dictionary_microtile_impl<Scalar>(
        plan, moments, fields, std::span{plan.tokens16});
  } else {
    apply_signed_tensor_dictionary_microtile_impl<Scalar>(
        plan, moments, fields, std::span{plan.tokens32});
  }
}

template <typename Scalar, typename Vector, typename Plan>
void apply_signed_tensor_dictionary_whole_tile(
    const Plan &plan, const std::span<const Vector> moments,
    const std::span<Vector> fields) {
  validate_signed_tensor_dictionary<Scalar>(plan, moments, fields);
  if (plan.token_width_bytes == 1) {
    apply_signed_tensor_dictionary_whole_tile_impl<Scalar>(
        plan, moments, fields, std::span{plan.tokens8});
  } else if (plan.token_width_bytes == 2) {
    apply_signed_tensor_dictionary_whole_tile_impl<Scalar>(
        plan, moments, fields, std::span{plan.tokens16});
  } else {
    apply_signed_tensor_dictionary_whole_tile_impl<Scalar>(
        plan, moments, fields, std::span{plan.tokens32});
  }
}

} // namespace

// Public entry points: the signed dictionary carries its own identity encoding,
// so unlike the other stored-tensor executors these take no identity map.
void apply_static_p2p_signed_tensor_dictionary_plan(
    const StaticP2PSignedTensorDictionaryPlan &plan,
    const std::span<const Vec3> dipole_moments, const std::span<Vec3> H) {
  apply_signed_tensor_dictionary<double>(plan, dipole_moments, H);
}

void apply_static_p2p_signed_tensor_dictionary_plan_whole_tile(
    const StaticP2PSignedTensorDictionaryPlan &plan,
    const std::span<const Vec3> dipole_moments, const std::span<Vec3> H) {
  apply_signed_tensor_dictionary_whole_tile<double>(plan, dipole_moments, H);
}

void apply_static_p2p_signed_tensor_dictionary_plan(
    const FloatStaticP2PSignedTensorDictionaryPlan &plan,
    const std::span<const FloatVec3> dipole_moments,
    const std::span<FloatVec3> H) {
  apply_signed_tensor_dictionary<float>(plan, dipole_moments, H);
}

void apply_static_p2p_signed_tensor_dictionary_plan_whole_tile(
    const FloatStaticP2PSignedTensorDictionaryPlan &plan,
    const std::span<const FloatVec3> dipole_moments,
    const std::span<FloatVec3> H) {
  apply_signed_tensor_dictionary_whole_tile<float>(plan, dipole_moments, H);
}

// Diagnostics reported in the initialisation summary and by the benchmarks.
const char *static_p2p_signed_simd_path() noexcept {
#if defined(__AVX2__) && defined(__FMA__)
  return "avx2-gather-fma";
#else
  return "compiler-simd-portable";
#endif
}

int static_p2p_signed_simd_width(const bool single_precision) noexcept {
  return single_precision ? 8 : 4;
}

} // namespace cdfmm
