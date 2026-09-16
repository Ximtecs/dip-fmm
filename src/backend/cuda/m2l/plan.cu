// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/backend/cuda/m2l.hpp"
#include "backend/cuda/m2l/internal.hpp"
#include "backend/cuda/common/error.hpp"
#include "backend/cuda/common/runtime.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace cdfmm::cuda_m2l_detail {

using cuda_detail::check_cuda;

namespace {
template <typename Scalar>
__global__ void scale_m2l_multipoles_kernel(
    const Scalar *multipoles, const int *node_levels,
    const Scalar *multipole_scaling, const int coefficient_count,
    const std::size_t value_count, Scalar *scaled_multipoles) {
  const std::size_t value_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (value_index >= value_count) {
    return;
  }

  const int node = static_cast<int>(value_index / coefficient_count);
  const int alpha = static_cast<int>(value_index % coefficient_count);
  const int level = node_levels[node];
  scaled_multipoles[value_index] =
      level < 0
          ? multipoles[value_index]
          : multipole_scaling[static_cast<std::size_t>(level) *
                                  coefficient_count +
                              alpha] *
                multipoles[value_index];
}

template <typename Scalar>
__global__ void apply_scaled_m2l_rows_kernel(
    const Scalar *matrices, const CudaM2LActiveRow *active_rows,
    const int *sources, const int *matrix_ids, const Scalar *local_scaling,
    const int active_row_count, const int coefficient_count,
    const Scalar *scaled_multipoles, Scalar *locals) {
  const std::size_t output =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t output_count =
      static_cast<std::size_t>(active_row_count) * coefficient_count;
  if (output >= output_count) {
    return;
  }

  const int active_row_index =
      static_cast<int>(output / coefficient_count);
  const int beta = static_cast<int>(output % coefficient_count);
  const CudaM2LActiveRow row = active_rows[active_row_index];
  const std::size_t coefficient_stride =
      static_cast<std::size_t>(coefficient_count);
  Scalar value = Scalar{0};

  // Threads own distinct (target, beta) outputs. This makes accumulation
  // deterministic and needs no atomics even when transfer classes repeat.
  for (int interaction = row.interaction_begin;
       interaction < row.interaction_end; ++interaction) {
    const std::size_t source_base =
        static_cast<std::size_t>(sources[interaction]) * coefficient_stride;
    const std::size_t matrix_base =
        static_cast<std::size_t>(matrix_ids[interaction]) *
        coefficient_stride * coefficient_stride;
    for (int alpha = 0; alpha < coefficient_count; ++alpha) {
      value += matrices[matrix_base +
                        static_cast<std::size_t>(alpha) * coefficient_stride +
                        beta] *
               scaled_multipoles[source_base + alpha];
    }
  }

  // Local scaling depends only on the target level and beta, so applying it
  // after the complete reduction removes it from the interaction/alpha loop.
  const std::size_t local_index =
      static_cast<std::size_t>(row.target) * coefficient_stride + beta;
  locals[local_index] +=
      local_scaling[static_cast<std::size_t>(row.level) * coefficient_stride +
                    beta] *
      value;
}

template <typename Scalar>
__global__ void apply_unscaled_m2l_rows_kernel(
    const Scalar *matrices, const CudaM2LActiveRow *active_rows,
    const int *sources, const int *matrix_ids,
    const Scalar *multipole_scaling, const Scalar *local_scaling,
    const int *node_levels, const int active_row_count, const int coefficient_count,
    const Scalar *multipoles, Scalar *locals) {
  const std::size_t output =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t output_count =
      static_cast<std::size_t>(active_row_count) * coefficient_count;
  if (output >= output_count) {
    return;
  }

  const int active_row_index =
      static_cast<int>(output / coefficient_count);
  const int beta = static_cast<int>(output % coefficient_count);
  const CudaM2LActiveRow row = active_rows[active_row_index];
  const std::size_t coefficient_stride =
      static_cast<std::size_t>(coefficient_count);
  const std::size_t scale_base =
      static_cast<std::size_t>(row.level) * coefficient_stride;
  Scalar value = Scalar{0};

  for (int interaction = row.interaction_begin;
       interaction < row.interaction_end; ++interaction) {
    const std::size_t source_base =
        static_cast<std::size_t>(sources[interaction]) * coefficient_stride;
    const std::size_t matrix_base =
        static_cast<std::size_t>(matrix_ids[interaction]) *
        coefficient_stride * coefficient_stride;
    for (int alpha = 0; alpha < coefficient_count; ++alpha) {
      value += matrices[matrix_base +
                        static_cast<std::size_t>(alpha) * coefficient_stride +
                        beta] *
               multipole_scaling[static_cast<std::size_t>(node_levels[sources[interaction]]) * coefficient_stride + alpha] *
               multipoles[source_base + alpha];
    }
  }

  const std::size_t local_index =
      static_cast<std::size_t>(row.target) * coefficient_stride + beta;
  locals[local_index] += local_scaling[scale_base + beta] * value;
}

/** @brief One block of same-transfer-class M2L pairs. */
struct CudaM2LClassGroup {
  int matrix_id{0};
  int pair_begin{0};
  int pair_end{0};
};

inline constexpr int m2l_group_threads = 256;
inline constexpr int m2l_group_alpha_tile = 16;
inline constexpr int m2l_group_pairs_per_thread = 8;

// Transfer-class grouped M2L. Every block owns one shared matrix and up to
// `pairs_per_block` (source, target) pairs using it. The matrix is streamed
// through shared memory in alpha slabs, so each matrix value is fetched from
// global memory once per block and reused for every pair, instead of once per
// interaction as in the target-row kernel. Threads own one beta and eight
// consecutive pairs; the eight multipole values they need per alpha are
// broadcast shared-memory loads. Multipole level scaling is applied while
// staging, and local level scaling before the final accumulation, so the
// arithmetic matches the target-row kernels exactly up to summation order.
// Different classes contribute to the same target, hence the atomic adds.
template <typename Scalar>
__global__ void __launch_bounds__(m2l_group_threads)
apply_grouped_m2l_kernel(const Scalar *__restrict__ matrices,
                         const CudaM2LClassGroup *__restrict__ groups,
                         const int *__restrict__ pair_sources,
                         const int *__restrict__ pair_targets,
                         const int *__restrict__ pair_source_levels,
                         const int *__restrict__ pair_target_levels,
                         const Scalar *__restrict__ multipole_scaling,
                         const Scalar *__restrict__ local_scaling,
                         const int coefficient_count, const int pairs_per_block,
                         const Scalar *__restrict__ multipoles,
                         Scalar *__restrict__ locals) {
  extern __shared__ __align__(16) unsigned char m2l_group_shared[];
  Scalar *matrix_tile = reinterpret_cast<Scalar *>(m2l_group_shared);
  // 16 * n * sizeof(Scalar) is a multiple of 16 bytes, so the multipole tile
  // keeps 16-byte alignment for the vector loads below.
  Scalar *multipole_tile =
      matrix_tile + m2l_group_alpha_tile * coefficient_count;

  constexpr int pairs_per_thread = m2l_group_pairs_per_thread;
  const int n = coefficient_count;
  const CudaM2LClassGroup group = groups[blockIdx.x];
  const int pair_count = group.pair_end - group.pair_begin;
  const int slot = static_cast<int>(threadIdx.x) / n;
  const int beta = static_cast<int>(threadIdx.x) - slot * n;
  const bool computes = slot < pairs_per_block / pairs_per_thread;
  const int first_pair = slot * pairs_per_thread;
  const std::size_t matrix_base =
      static_cast<std::size_t>(group.matrix_id) * n * n;

  Scalar accumulator[pairs_per_thread];
#pragma unroll
  for (int q = 0; q < pairs_per_thread; ++q) {
    accumulator[q] = Scalar{0};
  }

  for (int alpha_begin = 0; alpha_begin < n;
       alpha_begin += m2l_group_alpha_tile) {
    const int alpha_count = min(m2l_group_alpha_tile, n - alpha_begin);
    // Matrix rows alpha_begin.. are contiguous: a plain coalesced copy.
    for (int index = static_cast<int>(threadIdx.x); index < alpha_count * n;
         index += m2l_group_threads) {
      matrix_tile[index] =
          matrices[matrix_base + static_cast<std::size_t>(alpha_begin) * n +
                   index];
    }
    // Pre-scaled source multipoles, transposed to [alpha][pair] so that the
    // eight pairs of one thread are contiguous. Missing pairs of a partial
    // group stage zeros and contribute nothing.
    for (int index = static_cast<int>(threadIdx.x);
         index < alpha_count * pairs_per_block; index += m2l_group_threads) {
      const int pair = index / alpha_count;
      const int alpha = index - pair * alpha_count;
      Scalar value = Scalar{0};
      if (pair < pair_count) {
        const int pair_index = group.pair_begin + pair;
        const int source = pair_sources[pair_index];
        const int source_level = pair_source_levels[pair_index];
        value = multipole_scaling[static_cast<std::size_t>(source_level) * n +
                                  alpha_begin + alpha] *
                multipoles[static_cast<std::size_t>(source) * n + alpha_begin +
                           alpha];
      }
      multipole_tile[alpha * pairs_per_block + pair] = value;
    }
    __syncthreads();
    if (computes) {
      for (int alpha = 0; alpha < alpha_count; ++alpha) {
        const Scalar matrix_value = matrix_tile[alpha * n + beta];
        const Scalar *values =
            multipole_tile + alpha * pairs_per_block + first_pair;
        if constexpr (sizeof(Scalar) == sizeof(float)) {
          const float4 low = *reinterpret_cast<const float4 *>(values);
          const float4 high = *reinterpret_cast<const float4 *>(values + 4);
          accumulator[0] += matrix_value * low.x;
          accumulator[1] += matrix_value * low.y;
          accumulator[2] += matrix_value * low.z;
          accumulator[3] += matrix_value * low.w;
          accumulator[4] += matrix_value * high.x;
          accumulator[5] += matrix_value * high.y;
          accumulator[6] += matrix_value * high.z;
          accumulator[7] += matrix_value * high.w;
        } else {
#pragma unroll
          for (int q = 0; q < pairs_per_thread; q += 2) {
            const double2 pair_values =
                *reinterpret_cast<const double2 *>(values + q);
            accumulator[q] += matrix_value * pair_values.x;
            accumulator[q + 1] += matrix_value * pair_values.y;
          }
        }
      }
    }
    __syncthreads();
  }

  if (!computes) {
    return;
  }
#pragma unroll
  for (int q = 0; q < pairs_per_thread; ++q) {
    const int pair = first_pair + q;
    if (pair < pair_count) {
      const int pair_index = group.pair_begin + pair;
      const int target = pair_targets[pair_index];
      const int target_level = pair_target_levels[pair_index];
      atomicAdd(locals + static_cast<std::size_t>(target) * n + beta,
                local_scaling[static_cast<std::size_t>(target_level) * n +
                              beta] *
                    accumulator[q]);
    }
  }
}

/**
 * @brief Shared persistent CUDA executor for a canonical static M2L plan.
 *
 * Both CUDA backends own this exact type. The partial backend supplies copied
 * coefficient buffers, while CUDA-full supplies its resident hierarchy and
 * far-field stream. Immutable mathematical data and derived execution metadata
 * are uploaded once and no evaluation performs allocation or synchronisation.
 */
template <typename Scalar, typename Plan>
class CudaM2LExecutionStorage {
public:
  CudaM2LExecutionStorage(const Plan &data, cudaStream_t stream) {
    initialise(data, stream);
  }

  ~CudaM2LExecutionStorage() {
    cudaFree(matrices_);
    cudaFree(active_rows_);
    cudaFree(sources_);
    cudaFree(matrix_ids_);
    cudaFree(node_levels_);
    cudaFree(multipole_scaling_);
    cudaFree(local_scaling_);
    cudaFree(scaled_multipoles_);
    cudaFree(groups_);
    cudaFree(pair_sources_);
    cudaFree(pair_targets_);
    cudaFree(pair_source_levels_);
    cudaFree(pair_target_levels_);
  }

  CudaM2LExecutionStorage(const CudaM2LExecutionStorage &) = delete;
  CudaM2LExecutionStorage &operator=(const CudaM2LExecutionStorage &) = delete;

  void enqueue(const Scalar *multipoles, Scalar *locals, cudaStream_t stream,
               cudaEvent_t scale_complete) const {
    if (group_count_ != 0) {
      // The grouped kernel scales multipoles while staging, so there is no
      // separate scaling pass; the event still marks the phase boundary.
      check_cuda(cudaEventRecord(scale_complete, stream),
                 "record M2L scaling completion");
      apply_grouped_m2l_kernel<<<group_count_, m2l_group_threads,
                                 group_shared_bytes_, stream>>>(
          matrices_, groups_, pair_sources_, pair_targets_,
          pair_source_levels_, pair_target_levels_, multipole_scaling_,
          local_scaling_, coefficient_count_, pairs_per_block_, multipoles,
          locals);
      check_cuda(cudaGetLastError(), "launch grouped static M2L kernel");
      return;
    }
    const std::size_t coefficient_values =
        static_cast<std::size_t>(node_count_) * coefficient_count_;
    if (scaled_multipoles_ != nullptr && coefficient_values != 0) {
      constexpr int scaling_threads = 256;
      scale_m2l_multipoles_kernel<<<
          (coefficient_values + scaling_threads - 1) / scaling_threads,
          scaling_threads, 0, stream>>>(
          multipoles, node_levels_, multipole_scaling_, coefficient_count_,
          coefficient_values, scaled_multipoles_);
      check_cuda(cudaGetLastError(), "launch M2L multipole scaling kernel");
    }
    check_cuda(cudaEventRecord(scale_complete, stream),
               "record M2L scaling completion");

    const std::size_t outputs =
        static_cast<std::size_t>(active_row_count_) * coefficient_count_;
    if (outputs == 0) {
      return;
    }
    const int block_count = static_cast<int>(
        (outputs + threads_per_block_ - 1) / threads_per_block_);
    if (scaled_multipoles_ != nullptr) {
      apply_scaled_m2l_rows_kernel<<<block_count, threads_per_block_, 0,
                                     stream>>>(
          matrices_, active_rows_, sources_, matrix_ids_, local_scaling_,
          active_row_count_, coefficient_count_, scaled_multipoles_, locals);
    } else {
      // Extremely large plans can exceed the bounded scratch policy. The
      // fallback preserves the target-owned reduction without extra storage.
      apply_unscaled_m2l_rows_kernel<<<block_count, threads_per_block_, 0,
                                       stream>>>(
          matrices_, active_rows_, sources_, matrix_ids_, multipole_scaling_,
          local_scaling_, node_levels_, active_row_count_, coefficient_count_, multipoles,
          locals);
    }
    check_cuda(cudaGetLastError(), "launch optimised static M2L kernel");
  }

  [[nodiscard]] const CudaPlanStatistics &statistics() const noexcept {
    return statistics_;
  }

private:
  template <typename T>
  static void allocate(T **pointer, const std::size_t count,
                       const char *description) {
    if (count == 0) {
      *pointer = nullptr;
      return;
    }
    check_cuda(cudaMalloc(reinterpret_cast<void **>(pointer),
                          count * sizeof(T)),
               description);
  }

  template <typename T>
  static void upload(T *destination, const std::vector<T> &values,
                     cudaStream_t stream, const char *description) {
    if (!values.empty()) {
      check_cuda(cudaMemcpyAsync(destination, values.data(),
                                 values.size() * sizeof(T),
                                 cudaMemcpyHostToDevice, stream),
                 description);
    }
  }

  void initialise(const Plan &data, cudaStream_t stream) {
    if (data.coefficient_count < 0 || data.matrix_count < 0 ||
        data.level_count < 0 || data.target_row_offsets.empty()) {
      throw std::invalid_argument("canonical M2L dimensions are invalid");
    }
    if (data.source_nodes.size() != data.matrix_ids.size() ||
        (!data.interaction_levels.empty() &&
         data.source_nodes.size() != data.interaction_levels.size()) ||
        (!data.source_levels.empty() &&
         data.source_levels.size() != data.source_nodes.size()) ||
        (!data.target_levels.empty() &&
         data.target_levels.size() != data.source_nodes.size())) {
      throw std::invalid_argument("canonical M2L interaction arrays differ");
    }

    coefficient_count_ = data.coefficient_count;
    node_count_ = static_cast<int>(data.target_row_offsets.size()) - 1;
    std::vector<int> node_levels(static_cast<std::size_t>(node_count_), -1);
    if (!data.node_levels.empty()) {
      if (data.node_levels.size() != static_cast<std::size_t>(node_count_)) {
        throw std::invalid_argument("canonical M2L node levels are invalid");
      }
      node_levels = data.node_levels;
    } else {
      if (data.level_target_begin.size() !=
              static_cast<std::size_t>(data.level_count) ||
          data.level_target_end.size() !=
              static_cast<std::size_t>(data.level_count)) {
        throw std::invalid_argument("canonical M2L level bounds are invalid");
      }
      for (int level = 0; level < data.level_count; ++level) {
        const int begin = data.level_target_begin[static_cast<std::size_t>(level)];
        const int end = data.level_target_end[static_cast<std::size_t>(level)];
        if (begin < 0 || end < begin || end > node_count_) {
          throw std::invalid_argument("canonical M2L level bound is invalid");
        }
        std::fill(node_levels.begin() + begin, node_levels.begin() + end, level);
      }
    }

    std::vector<CudaM2LActiveRow> active_rows;
    active_rows.reserve(static_cast<std::size_t>(node_count_));
    for (int target = 0; target < node_count_; ++target) {
      const int begin = data.target_row_offsets[static_cast<std::size_t>(target)];
      const int end = data.target_row_offsets[static_cast<std::size_t>(target + 1)];
      if (begin < 0 || end < begin ||
          end > static_cast<int>(data.source_nodes.size())) {
        throw std::invalid_argument("canonical M2L row offset is invalid");
      }
      if (begin == end) {
        continue;
      }
      const int level = data.target_levels.empty()
          ? (data.interaction_levels.empty()
                 ? node_levels[static_cast<std::size_t>(target)]
                 : data.interaction_levels[static_cast<std::size_t>(begin)])
          : data.target_levels[static_cast<std::size_t>(begin)];
      if (level < 0 || level >= data.level_count ||
          node_levels[static_cast<std::size_t>(target)] != level) {
        throw std::invalid_argument("canonical M2L target level is invalid");
      }
      for (int interaction = begin; interaction < end; ++interaction) {
        const std::size_t index = static_cast<std::size_t>(interaction);
        const int source_node = data.source_nodes[index];
        const int interaction_target_level = data.target_levels.empty()
            ? (data.interaction_levels.empty() ? level
                                               : data.interaction_levels[index])
            : data.target_levels[index];
        const int interaction_source_level =
            (source_node >= 0 && source_node < node_count_)
                ? (data.source_levels.empty()
                       ? node_levels[static_cast<std::size_t>(source_node)]
                       : data.source_levels[index])
                : -1;
        if (interaction_target_level != level ||
            source_node < 0 || source_node >= node_count_ ||
            interaction_source_level < 0 ||
            interaction_source_level >= data.level_count ||
            interaction_source_level != node_levels[static_cast<std::size_t>(source_node)] ||
            data.matrix_ids[index] < 0 ||
            data.matrix_ids[index] >= data.matrix_count) {
          throw std::invalid_argument("canonical M2L interaction is invalid");
        }
      }
      active_rows.push_back({target, level, begin, end});
    }

    const std::size_t matrix_values =
        static_cast<std::size_t>(data.matrix_count) * coefficient_count_ *
        coefficient_count_;
    const std::size_t scaling_values =
        static_cast<std::size_t>(data.level_count) * coefficient_count_;
    if (data.matrices.size() != matrix_values ||
        data.multipole_scaling.size() != scaling_values ||
        data.local_scaling.size() != scaling_values) {
      throw std::invalid_argument("canonical M2L table dimensions are invalid");
    }

    active_row_count_ = static_cast<int>(active_rows.size());
    allocate(&matrices_, data.matrices.size(), "allocate M2L matrices");
    allocate(&active_rows_, active_rows.size(), "allocate M2L active rows");
    allocate(&sources_, data.source_nodes.size(), "allocate M2L sources");
    allocate(&matrix_ids_, data.matrix_ids.size(), "allocate M2L matrix IDs");
    allocate(&node_levels_, node_levels.size(), "allocate M2L node levels");
    allocate(&multipole_scaling_, data.multipole_scaling.size(),
             "allocate M2L multipole scaling");
    allocate(&local_scaling_, data.local_scaling.size(),
             "allocate M2L local scaling");
    upload(matrices_, data.matrices, stream, "upload M2L matrices");
    upload(active_rows_, active_rows, stream, "upload M2L active rows");
    upload(sources_, data.source_nodes, stream, "upload M2L sources");
    upload(matrix_ids_, data.matrix_ids, stream, "upload M2L matrix IDs");
    upload(node_levels_, node_levels, stream, "upload M2L node levels");
    upload(multipole_scaling_, data.multipole_scaling, stream,
           "upload M2L multipole scaling");
    upload(local_scaling_, data.local_scaling, stream,
           "upload M2L local scaling");

    // Transfer-class grouping for the grouped kernel: pairs are counting-sorted
    // by matrix id and cut into blocks of `pairs_per_block_`. Every block then
    // stages exactly one matrix. Coefficient counts above one block of threads
    // fall back to the target-row kernels.
    const int slots = coefficient_count_ > 0
        ? m2l_group_threads / coefficient_count_
        : 0;
    if (slots > 0 && !active_rows.empty()) {
      pairs_per_block_ = slots * m2l_group_pairs_per_thread;
      std::vector<int> class_offsets(static_cast<std::size_t>(data.matrix_count) + 1, 0);
      for (const CudaM2LActiveRow &row : active_rows) {
        for (int interaction = row.interaction_begin;
             interaction < row.interaction_end; ++interaction) {
          ++class_offsets[static_cast<std::size_t>(
              data.matrix_ids[static_cast<std::size_t>(interaction)]) + 1];
        }
      }
      for (int matrix = 0; matrix < data.matrix_count; ++matrix) {
        class_offsets[static_cast<std::size_t>(matrix) + 1] +=
            class_offsets[static_cast<std::size_t>(matrix)];
      }
      const std::size_t pair_count =
          class_offsets[static_cast<std::size_t>(data.matrix_count)];
      std::vector<int> pair_sources(pair_count);
      std::vector<int> pair_targets(pair_count);
      std::vector<int> pair_source_levels(pair_count);
      std::vector<int> pair_target_levels(pair_count);
      std::vector<int> cursor(class_offsets.begin(), class_offsets.end() - 1);
      for (const CudaM2LActiveRow &row : active_rows) {
        for (int interaction = row.interaction_begin;
             interaction < row.interaction_end; ++interaction) {
          const std::size_t index = static_cast<std::size_t>(interaction);
          const int source = data.source_nodes[index];
          const std::size_t slot_index = static_cast<std::size_t>(
              cursor[static_cast<std::size_t>(data.matrix_ids[index])]++);
          pair_sources[slot_index] = source;
          pair_targets[slot_index] = row.target;
          pair_source_levels[slot_index] =
              node_levels[static_cast<std::size_t>(source)];
          pair_target_levels[slot_index] = row.level;
        }
      }
      std::vector<CudaM2LClassGroup> groups;
      for (int matrix = 0; matrix < data.matrix_count; ++matrix) {
        const int begin = class_offsets[static_cast<std::size_t>(matrix)];
        const int end = class_offsets[static_cast<std::size_t>(matrix) + 1];
        for (int group_begin = begin; group_begin < end;
             group_begin += pairs_per_block_) {
          groups.push_back(
              {matrix, group_begin, std::min(end, group_begin + pairs_per_block_)});
        }
      }
      group_count_ = static_cast<int>(groups.size());
      group_shared_bytes_ =
          static_cast<std::size_t>(m2l_group_alpha_tile) *
          (static_cast<std::size_t>(coefficient_count_) + pairs_per_block_) *
          sizeof(Scalar);
      allocate(&groups_, groups.size(), "allocate M2L class groups");
      allocate(&pair_sources_, pair_sources.size(), "allocate M2L pair sources");
      allocate(&pair_targets_, pair_targets.size(), "allocate M2L pair targets");
      allocate(&pair_source_levels_, pair_source_levels.size(),
               "allocate M2L pair source levels");
      allocate(&pair_target_levels_, pair_target_levels.size(),
               "allocate M2L pair target levels");
      upload(groups_, groups, stream, "upload M2L class groups");
      upload(pair_sources_, pair_sources, stream, "upload M2L pair sources");
      upload(pair_targets_, pair_targets, stream, "upload M2L pair targets");
      upload(pair_source_levels_, pair_source_levels, stream,
             "upload M2L pair source levels");
      upload(pair_target_levels_, pair_target_levels, stream,
             "upload M2L pair target levels");
      grouped_metadata_bytes_ =
          groups.size() * sizeof(CudaM2LClassGroup) +
          4 * pair_count * sizeof(int);
    }

    // Pre-scaled multipoles are demand-sized. Their persistent allocation is
    // bounded by both total and currently free device memory so construction
    // remains safe for very large geometries.
    const std::size_t scratch_bytes =
        static_cast<std::size_t>(node_count_) * coefficient_count_ *
        sizeof(Scalar);
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
               "query CUDA memory for M2L scratch");
    const std::size_t scratch_limit =
        std::min(total_bytes / 10, free_bytes / 4);
    if (group_count_ == 0 && scratch_bytes != 0 &&
        scratch_bytes <= scratch_limit) {
      const cudaError_t status = cudaMalloc(
          reinterpret_cast<void **>(&scaled_multipoles_), scratch_bytes);
      if (status != cudaSuccess) {
        scaled_multipoles_ = nullptr;
        cudaGetLastError();
      }
    }

    int minimum_grid_size = 0;
    int suggested_block_size = 0;
    check_cuda(cudaOccupancyMaxPotentialBlockSize(
                   &minimum_grid_size, &suggested_block_size,
                   apply_scaled_m2l_rows_kernel<Scalar>, 0, 0),
               "select M2L launch configuration");
    if (suggested_block_size <= 64) {
      threads_per_block_ = 64;
    } else if (suggested_block_size <= 128) {
      threads_per_block_ = 128;
    } else {
      threads_per_block_ = 256;
    }

    statistics_.m2l_unique_matrix_count = data.matrix_count;
    statistics_.scalar_bytes = sizeof(Scalar);
    statistics_.m2l_matrix_bytes = data.matrices.size() * sizeof(Scalar);
    statistics_.m2l_interaction_count = data.source_nodes.size();
    statistics_.m2l_active_row_count = active_rows.size();
    statistics_.m2l_interaction_metadata_bytes =
        active_rows.size() * sizeof(CudaM2LActiveRow) +
        data.source_nodes.size() * sizeof(int) +
        data.matrix_ids.size() * sizeof(int) +
        node_levels.size() * sizeof(int) +
        grouped_metadata_bytes_;
    statistics_.m2l_scratch_bytes =
        scaled_multipoles_ == nullptr ? 0 : scratch_bytes;
    statistics_.m2l_threads_per_block = threads_per_block_;
    statistics_.setup_h2d_bytes =
        statistics_.m2l_matrix_bytes +
        statistics_.m2l_interaction_metadata_bytes +
        (data.multipole_scaling.size() + data.local_scaling.size()) *
            sizeof(Scalar);
    statistics_.persistent_device_bytes =
        statistics_.setup_h2d_bytes + statistics_.m2l_scratch_bytes;
    statistics_.plan_generation_count = 1;
    statistics_.static_upload_count = 1;
    statistics_.static_m2l_upload_count = 1;
    statistics_.geometry_upload_count = 1;
  }

  Scalar *matrices_{nullptr};
  CudaM2LActiveRow *active_rows_{nullptr};
  int *sources_{nullptr};
  int *matrix_ids_{nullptr};
  int *node_levels_{nullptr};
  Scalar *multipole_scaling_{nullptr};
  Scalar *local_scaling_{nullptr};
  Scalar *scaled_multipoles_{nullptr};
  CudaM2LClassGroup *groups_{nullptr};
  int *pair_sources_{nullptr};
  int *pair_targets_{nullptr};
  int *pair_source_levels_{nullptr};
  int *pair_target_levels_{nullptr};
  int group_count_{0};
  int pairs_per_block_{0};
  std::size_t group_shared_bytes_{0};
  std::size_t grouped_metadata_bytes_{0};
  int node_count_{0};
  int active_row_count_{0};
  int coefficient_count_{0};
  int threads_per_block_{128};
  CudaPlanStatistics statistics_{};
};
} // namespace

template <typename Scalar, typename Plan>
struct CudaM2LExecutionPlan<Scalar, Plan>::Implementation {
  CudaM2LExecutionStorage<Scalar, Plan> storage;

  Implementation(const Plan &data, cudaStream_t stream) : storage(data, stream) {}
};

template <typename Scalar, typename Plan>
CudaM2LExecutionPlan<Scalar, Plan>::CudaM2LExecutionPlan(
    const Plan &data, cudaStream_t stream)
    : implementation_(new Implementation(data, stream)) {}

template <typename Scalar, typename Plan>
CudaM2LExecutionPlan<Scalar, Plan>::~CudaM2LExecutionPlan() {
  delete implementation_;
}

template <typename Scalar, typename Plan>
void CudaM2LExecutionPlan<Scalar, Plan>::enqueue(
    const Scalar *multipoles, Scalar *locals, cudaStream_t stream,
    cudaEvent_t scale_complete) const {
  implementation_->storage.enqueue(multipoles, locals, stream, scale_complete);
}

template <typename Scalar, typename Plan>
const CudaPlanStatistics &
CudaM2LExecutionPlan<Scalar, Plan>::statistics() const noexcept {
  return implementation_->storage.statistics();
}

template class CudaM2LExecutionPlan<double, StaticM2LPlan>;
template class CudaM2LExecutionPlan<float, FloatStaticM2LPlan>;

} // namespace cdfmm::cuda_m2l_detail

namespace cdfmm {

using cuda_detail::check_cuda;

//------------------------------------------------------------------------------
// Standalone CUDA M2L plan
//------------------------------------------------------------------------------

struct CudaM2LPlan::Implementation {
  bool fp32{false};
  int coefficient_count{0};
  int node_count{0};
  std::size_t coefficient_values{0};
  // Pinned staging keeps the per-evaluation coefficient round trip on the
  // asynchronous copy engines instead of driver-staged pageable copies.
  double *host_multipoles{nullptr};
  double *host_locals{nullptr};
  cuda_m2l_detail::CudaM2LExecutionPlan<double, StaticM2LPlan> *executor{nullptr};
  float *host_multipoles_float{nullptr};
  float *host_locals_float{nullptr};
  cuda_m2l_detail::CudaM2LExecutionPlan<float, FloatStaticM2LPlan> *executor_float{nullptr};
  float *multipoles_float{nullptr};
  float *locals_float{nullptr};
  double* multipoles{nullptr};
  double* locals{nullptr};
  cudaStream_t stream{nullptr};
  cudaEvent_t start{nullptr};
  cudaEvent_t h2d{nullptr};
  cudaEvent_t scale{nullptr};
  cudaEvent_t kernel{nullptr};
  cudaEvent_t d2h{nullptr};
  CudaPlanStatistics statistics{};
  CudaEvaluationTimings timings{};
};

CudaM2LPlan::CudaM2LPlan(const FloatStaticM2LPlan &data)
    : implementation_(new Implementation{}) {
  if (!cuda_runtime_available()) {
    delete implementation_;
    implementation_ = nullptr;
    throw std::runtime_error("CudaM2L requires an available CUDA device");
  }
  auto &plan = *implementation_;
  plan.fp32 = true;
  plan.coefficient_count = data.coefficient_count;
  plan.node_count = static_cast<int>(data.target_row_offsets.size()) - 1;
  const std::size_t coefficient_values =
      static_cast<std::size_t>(plan.node_count) * plan.coefficient_count;
  plan.coefficient_values = coefficient_values;
  check_cuda(cudaMallocHost(reinterpret_cast<void **>(&plan.host_multipoles_float),
                            std::max(coefficient_values * sizeof(float),
                                     std::size_t{1})),
             "allocate pinned FP32 M2L multipoles");
  check_cuda(cudaMallocHost(reinterpret_cast<void **>(&plan.host_locals_float),
                            std::max(coefficient_values * sizeof(float),
                                     std::size_t{1})),
             "allocate pinned FP32 M2L locals");
  check_cuda(cudaStreamCreateWithFlags(&plan.stream, cudaStreamNonBlocking),
             "create canonical FP32 M2L stream");
  check_cuda(cudaEventCreate(&plan.start), "create M2L event");
  check_cuda(cudaEventCreate(&plan.h2d), "create M2L event");
  check_cuda(cudaEventCreate(&plan.scale), "create M2L event");
  check_cuda(cudaEventCreate(&plan.kernel), "create M2L event");
  check_cuda(cudaEventCreate(&plan.d2h), "create M2L event");
  const auto allocate = [](auto **pointer, const std::size_t bytes) {
    check_cuda(cudaMalloc(reinterpret_cast<void **>(pointer),
                          std::max(bytes, std::size_t{1})),
               "allocate FP32 M2L data");
  };
  allocate(&plan.multipoles_float, coefficient_values * sizeof(float));
  allocate(&plan.locals_float, coefficient_values * sizeof(float));
  plan.executor_float =
      new cuda_m2l_detail::CudaM2LExecutionPlan<float, FloatStaticM2LPlan>(data, plan.stream);
  check_cuda(cudaStreamSynchronize(plan.stream), "finish FP32 M2L upload");
  plan.statistics = plan.executor_float->statistics();
  plan.statistics.persistent_device_bytes +=
      2 * coefficient_values * sizeof(float);
}

CudaM2LPlan::CudaM2LPlan(const StaticM2LPlan& data)
    : implementation_(new Implementation{}) {
  if (!cuda_runtime_available()) {
    delete implementation_;
    implementation_ = nullptr;
    throw std::runtime_error("CudaM2L requires an available CUDA device");
  }
  auto& plan = *implementation_;
  plan.coefficient_count = data.coefficient_count;
  plan.node_count = static_cast<int>(data.target_row_offsets.size()) - 1;
  const std::size_t coefficient_values =
      static_cast<std::size_t>(plan.node_count) * plan.coefficient_count;
  plan.coefficient_values = coefficient_values;
  check_cuda(cudaMallocHost(reinterpret_cast<void **>(&plan.host_multipoles),
                            std::max(coefficient_values * sizeof(double),
                                     std::size_t{1})),
             "allocate pinned M2L multipoles");
  check_cuda(cudaMallocHost(reinterpret_cast<void **>(&plan.host_locals),
                            std::max(coefficient_values * sizeof(double),
                                     std::size_t{1})),
             "allocate pinned M2L locals");
  check_cuda(cudaStreamCreateWithFlags(&plan.stream, cudaStreamNonBlocking),
             "create canonical M2L stream");
  check_cuda(cudaEventCreate(&plan.start), "create M2L event");
  check_cuda(cudaEventCreate(&plan.h2d), "create M2L event");
  check_cuda(cudaEventCreate(&plan.scale), "create M2L event");
  check_cuda(cudaEventCreate(&plan.kernel), "create M2L event");
  check_cuda(cudaEventCreate(&plan.d2h), "create M2L event");
  const auto allocate = [](auto** pointer, const std::size_t bytes) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(pointer),
                          std::max(bytes, std::size_t{1})), "allocate M2L data");
  };
  allocate(&plan.multipoles, coefficient_values * sizeof(double));
  allocate(&plan.locals, coefficient_values * sizeof(double));
  plan.executor =
      new cuda_m2l_detail::CudaM2LExecutionPlan<double, StaticM2LPlan>(data, plan.stream);
  check_cuda(cudaStreamSynchronize(plan.stream), "finish M2L plan upload");
  plan.statistics = plan.executor->statistics();
  plan.statistics.persistent_device_bytes +=
      2 * coefficient_values * sizeof(double);
}

CudaM2LPlan::~CudaM2LPlan() {
  if (!implementation_) return;
  auto& plan = *implementation_;
  delete plan.executor;
  delete plan.executor_float;
  cudaFree(plan.multipoles); cudaFree(plan.locals);
  cudaFree(plan.multipoles_float);
  cudaFree(plan.locals_float);
  cudaFreeHost(plan.host_multipoles);
  cudaFreeHost(plan.host_locals);
  cudaFreeHost(plan.host_multipoles_float);
  cudaFreeHost(plan.host_locals_float);
  cudaEventDestroy(plan.start); cudaEventDestroy(plan.h2d);
  cudaEventDestroy(plan.scale);
  cudaEventDestroy(plan.kernel); cudaEventDestroy(plan.d2h);
  cudaStreamDestroy(plan.stream); delete implementation_;
}

void CudaM2LPlan::evaluate(const std::span<const float> multipoles,
                           const std::span<float> locals) {
  auto &plan = *implementation_;
  const std::size_t values = plan.coefficient_values;
  if (!plan.fp32 || multipoles.size() != values || locals.size() != values) {
    throw std::invalid_argument("CUDA FP32 M2L coefficient dimensions differ");
  }
  std::copy(multipoles.begin(), multipoles.end(), plan.host_multipoles_float);
  check_cuda(cudaEventRecord(plan.start, plan.stream), "record M2L start");
  if (values != 0) {
    check_cuda(cudaMemcpyAsync(plan.multipoles_float,
                               plan.host_multipoles_float,
                               values * sizeof(float), cudaMemcpyHostToDevice,
                               plan.stream),
               "upload FP32 M2L multipoles");
  }
  check_cuda(cudaEventRecord(plan.h2d, plan.stream), "record M2L H2D");
  if (values != 0) {
    check_cuda(cudaMemsetAsync(plan.locals_float, 0, values * sizeof(float),
                               plan.stream),
               "clear FP32 M2L locals");
  }
  plan.executor_float->enqueue(plan.multipoles_float, plan.locals_float,
                               plan.stream, plan.scale);
  check_cuda(cudaEventRecord(plan.kernel, plan.stream), "record M2L kernel");
  if (values != 0) {
    check_cuda(cudaMemcpyAsync(plan.host_locals_float, plan.locals_float,
                               values * sizeof(float), cudaMemcpyDeviceToHost,
                               plan.stream),
               "download FP32 M2L locals");
  }
  check_cuda(cudaEventRecord(plan.d2h, plan.stream), "record M2L D2H");
  check_cuda(cudaEventSynchronize(plan.d2h), "wait for FP32 M2L");
  std::copy(plan.host_locals_float, plan.host_locals_float + values,
            locals.begin());
  const auto elapsed = [](cudaEvent_t first, cudaEvent_t second) {
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, first, second),
               "time FP32 M2L");
    return static_cast<double>(milliseconds) * 1.0e-3;
  };
  plan.timings = {};
  plan.timings.h2d_seconds = elapsed(plan.start, plan.h2d);
  plan.timings.scale_seconds = elapsed(plan.h2d, plan.scale);
  plan.timings.multiply_seconds = elapsed(plan.scale, plan.kernel);
  plan.timings.kernel_seconds =
      plan.timings.scale_seconds + plan.timings.multiply_seconds;
  plan.timings.m2l_seconds = plan.timings.kernel_seconds;
  plan.timings.d2h_seconds = elapsed(plan.kernel, plan.d2h);
  plan.timings.total_seconds = plan.timings.h2d_seconds +
      plan.timings.kernel_seconds + plan.timings.d2h_seconds;
  plan.statistics.evaluation_h2d_bytes = values * sizeof(float);
  plan.statistics.evaluation_d2h_bytes = values * sizeof(float);
  ++plan.statistics.evaluation_h2d_calls;
  ++plan.statistics.evaluation_d2h_calls;
}

void CudaM2LPlan::evaluate(const std::span<const double> multipoles,
                           const std::span<double> locals) {
  auto& plan = *implementation_;
  const std::size_t values = plan.coefficient_values;
  if (plan.fp32 || multipoles.size() != values || locals.size() != values) {
    throw std::invalid_argument("CUDA M2L coefficient dimensions differ");
  }
  std::copy(multipoles.begin(), multipoles.end(), plan.host_multipoles);
  check_cuda(cudaEventRecord(plan.start, plan.stream), "record M2L start");
  if (values != 0) {
    check_cuda(cudaMemcpyAsync(plan.multipoles, plan.host_multipoles,
                               values * sizeof(double), cudaMemcpyHostToDevice,
                               plan.stream),
               "upload M2L multipoles");
  }
  check_cuda(cudaEventRecord(plan.h2d, plan.stream), "record M2L H2D");
  if (values != 0) {
    check_cuda(cudaMemsetAsync(plan.locals, 0, values * sizeof(double),
                               plan.stream),
               "clear M2L locals");
  }
  plan.executor->enqueue(plan.multipoles, plan.locals, plan.stream, plan.scale);
  check_cuda(cudaEventRecord(plan.kernel, plan.stream), "record M2L kernel");
  if (values != 0) {
    check_cuda(cudaMemcpyAsync(plan.host_locals, plan.locals,
                               values * sizeof(double), cudaMemcpyDeviceToHost,
                               plan.stream),
               "download M2L locals");
  }
  check_cuda(cudaEventRecord(plan.d2h, plan.stream), "record M2L D2H");
  check_cuda(cudaEventSynchronize(plan.d2h), "wait for M2L");
  std::copy(plan.host_locals, plan.host_locals + values, locals.begin());
  const auto elapsed = [](cudaEvent_t first, cudaEvent_t second) {
    float ms = 0.0F; check_cuda(cudaEventElapsedTime(&ms, first, second), "time M2L");
    return static_cast<double>(ms) * 1.0e-3;
  };
  plan.timings = {};
  plan.timings.h2d_seconds = elapsed(plan.start, plan.h2d);
  plan.timings.scale_seconds = elapsed(plan.h2d, plan.scale);
  plan.timings.multiply_seconds = elapsed(plan.scale, plan.kernel);
  plan.timings.kernel_seconds =
      plan.timings.scale_seconds + plan.timings.multiply_seconds;
  plan.timings.m2l_seconds = plan.timings.kernel_seconds;
  plan.timings.d2h_seconds = elapsed(plan.kernel, plan.d2h);
  plan.timings.total_seconds =
      plan.timings.h2d_seconds + plan.timings.kernel_seconds +
      plan.timings.d2h_seconds;
  plan.statistics.evaluation_h2d_bytes = values * sizeof(double);
  plan.statistics.evaluation_d2h_bytes = values * sizeof(double);
  ++plan.statistics.evaluation_h2d_calls;
  ++plan.statistics.evaluation_d2h_calls;
}

const CudaPlanStatistics& CudaM2LPlan::statistics() const noexcept {
  return implementation_->statistics;
}
const CudaEvaluationTimings& CudaM2LPlan::timings() const noexcept {
  return implementation_->timings;
}



} // namespace cdfmm
