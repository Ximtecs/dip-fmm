// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/cuda_direct.hpp"
#include "cdfmm/cuda_cuboid.hpp"
#include "cuda_fmm_plan.hpp"
#include "profile.hpp"
#include "cuda_m2l_plan.hpp"
#include "cuda_p2p_plan.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <algorithm>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace cdfmm {
namespace {

// This translation unit contains both CUDA execution branches because they
// share persistent plan storage, streams, and low-level launch utilities.
// Far-field kernels implement P2M -> M2M -> M2L -> L2L -> L2P; near-field
// kernels implement list1 P2P.  They consume the same canonical host plans as
// the CPU executors, and separate streams retain near/far overlap.  Splitting
// these kernels physically would add no execution abstraction and is deferred
// until their shared device infrastructure can remain equally explicit.


void check_cuda(const cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

void check_cusparse(const cusparseStatus_t status, const char* operation)
{
  if (status != CUSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string(operation) + ": " + cusparseGetErrorString(status));
  }
}

void check_cublas(const cublasStatus_t status, const char* operation)
{
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string(operation) + ": cuBLAS status " +
        std::to_string(static_cast<int>(status)));
  }
}

__global__ void permute_moments_kernel(const Vec3 *input,
                                       const int *permutation, const int count,
                                       Vec3 *sorted) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    sorted[index] = input[permutation[index]];
    }
}

__global__ void apply_entries_kernel(const StaticOperatorEntry *entries,
                                     const std::size_t count,
                                     const double *input, double *output) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const StaticOperatorEntry entry = entries[index];
        atomicAdd(output + entry.output, entry.value * input[entry.input]);
  }
}

__global__ void
apply_shared_translation_kernel(const StaticOperatorEntry *matrices,
                                const CudaTranslationInteraction *interactions,
                                const std::size_t interaction_count,
                                const int entries_per_matrix,
                                const int coefficient_count, const int level,
                                const double *input, double *output) {
  const std::size_t item = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t item_count = interaction_count * entries_per_matrix;
  if (item >= item_count) {
    return;
  }
  const std::size_t interaction_index = item / entries_per_matrix;
  const CudaTranslationInteraction interaction =
      interactions[interaction_index];
  if (interaction.level != level) {
    return;
  }
  const int matrix_entry = static_cast<int>(item % entries_per_matrix);
  const StaticOperatorEntry entry =
      matrices[static_cast<std::size_t>(interaction.matrix_id) *
                   entries_per_matrix +
               matrix_entry];
  atomicAdd(output +
                static_cast<std::size_t>(interaction.target_node) *
                    coefficient_count +
                entry.output,
            entry.value *
                input[static_cast<std::size_t>(interaction.source_node) *
                          coefficient_count +
                      entry.input]);
}

/** @brief One non-empty canonical M2L target row on the device. */
struct CudaM2LActiveRow {
  int target{0};
  int level{0};
  int interaction_begin{0};
  int interaction_end{0};
};

__global__ void scale_m2l_multipoles_kernel(
    const double *multipoles, const int *node_levels,
    const double *multipole_scaling, const int coefficient_count,
    const std::size_t value_count, double *scaled_multipoles) {
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

__global__ void apply_scaled_m2l_rows_kernel(
    const double *matrices, const CudaM2LActiveRow *active_rows,
    const int *sources, const int *matrix_ids, const double *local_scaling,
    const int active_row_count, const int coefficient_count,
    const double *scaled_multipoles, double *locals) {
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
  double value = 0.0;

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

__global__ void apply_unscaled_m2l_rows_kernel(
    const double *matrices, const CudaM2LActiveRow *active_rows,
    const int *sources, const int *matrix_ids,
    const double *multipole_scaling, const double *local_scaling,
    const int active_row_count, const int coefficient_count,
    const double *multipoles, double *locals) {
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
  double value = 0.0;

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
               multipole_scaling[scale_base + alpha] *
               multipoles[source_base + alpha];
    }
  }

  const std::size_t local_index =
      static_cast<std::size_t>(row.target) * coefficient_stride + beta;
  locals[local_index] += local_scaling[scale_base + beta] * value;
}

/**
 * @brief Shared persistent CUDA executor for a canonical static M2L plan.
 *
 * Both CUDA backends own this exact type. The partial backend supplies copied
 * coefficient buffers, while CUDA-full supplies its resident hierarchy and
 * far-field stream. Immutable mathematical data and derived execution metadata
 * are uploaded once and no evaluation performs allocation or synchronisation.
 */
class CudaM2LExecutionPlan {
public:
  CudaM2LExecutionPlan(const StaticM2LPlan &data, cudaStream_t stream) {
    initialise(data, stream);
  }

  ~CudaM2LExecutionPlan() {
    cudaFree(matrices_);
    cudaFree(active_rows_);
    cudaFree(sources_);
    cudaFree(matrix_ids_);
    cudaFree(node_levels_);
    cudaFree(multipole_scaling_);
    cudaFree(local_scaling_);
    cudaFree(scaled_multipoles_);
  }

  CudaM2LExecutionPlan(const CudaM2LExecutionPlan &) = delete;
  CudaM2LExecutionPlan &operator=(const CudaM2LExecutionPlan &) = delete;

  void enqueue(const double *multipoles, double *locals, cudaStream_t stream,
               cudaEvent_t scale_complete) const {
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
          local_scaling_, active_row_count_, coefficient_count_, multipoles,
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

  void initialise(const StaticM2LPlan &data, cudaStream_t stream) {
    if (data.coefficient_count < 0 || data.matrix_count < 0 ||
        data.level_count < 0 || data.target_row_offsets.empty()) {
      throw std::invalid_argument("canonical M2L dimensions are invalid");
    }
    if (data.source_nodes.size() != data.matrix_ids.size() ||
        data.source_nodes.size() != data.interaction_levels.size()) {
      throw std::invalid_argument("canonical M2L interaction arrays differ");
    }

    coefficient_count_ = data.coefficient_count;
    node_count_ = static_cast<int>(data.target_row_offsets.size()) - 1;
    std::vector<int> node_levels(static_cast<std::size_t>(node_count_), -1);
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
      const int level = data.interaction_levels[static_cast<std::size_t>(begin)];
      if (level < 0 || level >= data.level_count ||
          node_levels[static_cast<std::size_t>(target)] != level) {
        throw std::invalid_argument("canonical M2L target level is invalid");
      }
      for (int interaction = begin; interaction < end; ++interaction) {
        const std::size_t index = static_cast<std::size_t>(interaction);
        if (data.interaction_levels[index] != level ||
            data.source_nodes[index] < 0 ||
            data.source_nodes[index] >= node_count_ ||
            node_levels[static_cast<std::size_t>(data.source_nodes[index])] !=
                level ||
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

    // Pre-scaled multipoles are demand-sized. Their persistent allocation is
    // bounded by both total and currently free device memory so construction
    // remains safe for very large geometries.
    const std::size_t scratch_bytes =
        static_cast<std::size_t>(node_count_) * coefficient_count_ *
        sizeof(double);
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes),
               "query CUDA memory for M2L scratch");
    const std::size_t scratch_limit =
        std::min(total_bytes / 10, free_bytes / 4);
    if (scratch_bytes != 0 && scratch_bytes <= scratch_limit) {
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
                   apply_scaled_m2l_rows_kernel, 0, 0),
               "select M2L launch configuration");
    if (suggested_block_size <= 64) {
      threads_per_block_ = 64;
    } else if (suggested_block_size <= 128) {
      threads_per_block_ = 128;
    } else {
      threads_per_block_ = 256;
    }

    statistics_.m2l_unique_matrix_count = data.matrix_count;
    statistics_.m2l_matrix_bytes = data.matrices.size() * sizeof(double);
    statistics_.m2l_interaction_count = data.source_nodes.size();
    statistics_.m2l_active_row_count = active_rows.size();
    statistics_.m2l_interaction_metadata_bytes =
        active_rows.size() * sizeof(CudaM2LActiveRow) +
        data.source_nodes.size() * sizeof(int) +
        data.matrix_ids.size() * sizeof(int) +
        node_levels.size() * sizeof(int);
    statistics_.m2l_scratch_bytes =
        scaled_multipoles_ == nullptr ? 0 : scratch_bytes;
    statistics_.m2l_threads_per_block = threads_per_block_;
    statistics_.setup_h2d_bytes =
        statistics_.m2l_matrix_bytes +
        statistics_.m2l_interaction_metadata_bytes +
        (data.multipole_scaling.size() + data.local_scaling.size()) *
            sizeof(double);
    statistics_.persistent_device_bytes =
        statistics_.setup_h2d_bytes + statistics_.m2l_scratch_bytes;
    statistics_.plan_generation_count = 1;
    statistics_.static_upload_count = 1;
    statistics_.static_m2l_upload_count = 1;
    statistics_.geometry_upload_count = 1;
  }

  double *matrices_{nullptr};
  CudaM2LActiveRow *active_rows_{nullptr};
  int *sources_{nullptr};
  int *matrix_ids_{nullptr};
  int *node_levels_{nullptr};
  double *multipole_scaling_{nullptr};
  double *local_scaling_{nullptr};
  double *scaled_multipoles_{nullptr};
  int node_count_{0};
  int active_row_count_{0};
  int coefficient_count_{0};
  int threads_per_block_{128};
  CudaPlanStatistics statistics_{};
};

constexpr int static_operator_threads = 256;

__global__ void combine_order_kernel(const Vec3 *far_fields,
                                     const Vec3 *near_fields,
                                     const int *target_permutation,
                                     const int count, Vec3 *output) {
  const int sorted = blockIdx.x * blockDim.x + threadIdx.x;
  if (sorted < count) {
    Vec3 value = far_fields[sorted];
        value.x += near_fields[sorted].x;
        value.y += near_fields[sorted].y;
        value.z += near_fields[sorted].z;
        output[target_permutation[sorted]] = value;
  }
}

__global__ void dipole_field_kernel(
    const Vec3 *targets, const Vec3 *sources, const Vec3 *moments,
    const int *self_indices, const std::size_t source_count,
    const std::size_t target_count, Vec3 *fields, double *potentials,
    const bool compute_field, const bool compute_potential) {
  const std::size_t target_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (target_index >= target_count) {
    return;
    }

    const Vec3 target = targets[target_index];
    Vec3 H;
    H.x = 0.0;
    H.y = 0.0;
    H.z = 0.0;
    double phi = 0.0;
    constexpr double scale = 0.079577471545947667884441881686257181;
    const int self_index = self_indices[target_index];

    for (std::size_t source_index = 0; source_index < source_count;
         ++source_index) {
        if (static_cast<int>(source_index) == self_index) {
            continue;
        }
        const Vec3 source = sources[source_index];
        const Vec3 moment = moments[source_index];
        const double rx = target.x - source.x;
        const double ry = target.y - source.y;
        const double rz = target.z - source.z;
    const double r2 = rx * rx + ry * ry + rz * rz;
    const double inverse_r = rsqrt(r2);
    const double inverse_r3 = inverse_r * inverse_r * inverse_r;
    const double moment_dot_r = moment.x * rx + moment.y * ry + moment.z * rz;

    if (compute_potential) {
      phi += scale * moment_dot_r * inverse_r3;
        }
        if (compute_field) {
            const double factor = 3.0 * moment_dot_r * inverse_r3 / r2;
            H.x += scale * (rx * factor - moment.x * inverse_r3);
            H.y += scale * (ry * factor - moment.y * inverse_r3);
            H.z += scale * (rz * factor - moment.z * inverse_r3);
        }
    }

    if (compute_field) {
        fields[target_index] = H;
    }
    if (compute_potential) {
        potentials[target_index] = phi;
    }
}

} // namespace

struct CudaDenseDirectPlan::Implementation {
  std::size_t source_count{0};
  std::size_t target_count{0};
  std::size_t matrix_value_count{0};
  double* device_matrices{nullptr};
  double* device_moments{nullptr};
  double* device_fields{nullptr};
  double* pinned_moments{nullptr};
  double* pinned_fields{nullptr};
  cudaStream_t stream{nullptr};
  cublasHandle_t cublas_handle{nullptr};

  ~Implementation()
  {
    cudaFree(device_matrices);
    cudaFree(device_moments);
    cudaFree(device_fields);
    cudaFreeHost(pinned_moments);
    cudaFreeHost(pinned_fields);
    if (cublas_handle != nullptr) {
      cublasDestroy(cublas_handle);
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }
};

struct CudaDirectPlan::Implementation {
    std::size_t source_count{0};
    std::size_t target_count{0};
    Vec3* device_sources{nullptr};
    Vec3* device_targets{nullptr};
    Vec3* device_moments{nullptr};
    Vec3* device_fields{nullptr};
    double* device_potentials{nullptr};
    int* device_self_indices{nullptr};
    Vec3* pinned_moments{nullptr};
    Vec3* pinned_fields{nullptr};
    double* pinned_potentials{nullptr};
    std::vector<int> self_indices{};
    cudaStream_t stream{nullptr};
    cudaEvent_t evaluation_start{nullptr};
    cudaEvent_t h2d_complete{nullptr};
    cudaEvent_t kernel_complete{nullptr};
    cudaEvent_t d2h_complete{nullptr};
    CudaPlanStatistics statistics{};
  CudaEvaluationTimings evaluation_timings{};
};

bool cuda_runtime_available() noexcept {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

bool cuda_compiled() noexcept { return true; }

bool cuda_direct_available() noexcept { return cuda_runtime_available(); }

bool cuda_dense_direct_available() noexcept
{
  return cuda_runtime_available();
}

bool cuda_m2l_available() noexcept { return cuda_m2l_p2p_available(); }

bool cuda_m2l_p2p_available() noexcept { return cuda_runtime_available(); }

bool cuda_full_available() noexcept { return cuda_runtime_available(); }

std::string cuda_runtime_description() {
  int device = 0;
  cudaDeviceProp properties{};
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties");
    return std::string(properties.name) + " (compute capability " +
        std::to_string(properties.major) + "." +
        std::to_string(properties.minor) + ")";
}

CudaDenseDirectPlan::CudaDenseDirectPlan(
    const std::span<const Vec3> source_positions,
    const std::span<const Vec3> target_positions,
    const SourceGeometry source_geometry,
    const TargetGeometry target_geometry,
    const std::span<const CuboidSize> source_sizes,
    const std::span<const CuboidSize> target_sizes,
    const std::span<const int> target_source_indices)
    : implementation_(new Implementation{})
{
  try {
    if (!cuda_runtime_available()) {
      throw std::runtime_error(
          "CUDA dense direct backend requested, but no CUDA-capable device "
          "is available");
    }

    auto& plan = *implementation_;
    plan.source_count = source_positions.size();
    plan.target_count = target_positions.size();
    if (plan.source_count > static_cast<std::size_t>(
            std::numeric_limits<int>::max()) ||
        plan.target_count > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
      throw std::overflow_error(
          "CUDA dense direct dimensions exceed the cuBLAS integer limit");
    }
    if (plan.source_count != 0 &&
        plan.target_count > std::numeric_limits<std::size_t>::max() /
            plan.source_count) {
      throw std::overflow_error("CUDA dense direct tensor size overflow");
    }
    plan.matrix_value_count = plan.source_count * plan.target_count;
    if (plan.matrix_value_count >
        std::numeric_limits<std::size_t>::max() /
            (6 * sizeof(double))) {
      throw std::overflow_error("CUDA dense direct tensor byte-size overflow");
    }

    // Build the exact six-component geometry tensor using the common CPU
    // implementation, then retain only its device copy after construction.
    const DenseDirectPlan host_plan(
        source_positions, target_positions, source_geometry, target_geometry,
        source_sizes, target_sizes, target_source_indices);

    check_cuda(cudaStreamCreateWithFlags(&plan.stream, cudaStreamNonBlocking),
               "create CUDA dense direct stream");
    check_cublas(cublasCreate(&plan.cublas_handle),
                 "create CUDA dense direct cuBLAS handle");
    check_cublas(cublasSetStream(plan.cublas_handle, plan.stream),
                 "set CUDA dense direct cuBLAS stream");

    const std::size_t tensor_bytes = tensor_memory_bytes();
    const std::size_t moment_bytes = 3 * plan.source_count * sizeof(double);
    const std::size_t field_bytes = 3 * plan.target_count * sizeof(double);
    check_cuda(cudaMalloc(&plan.device_matrices,
                          std::max(tensor_bytes, sizeof(double))),
               "allocate CUDA dense direct tensors");
    check_cuda(cudaMalloc(&plan.device_moments,
                          std::max(moment_bytes, sizeof(double))),
               "allocate CUDA dense direct moments");
    check_cuda(cudaMalloc(&plan.device_fields,
                          std::max(field_bytes, sizeof(double))),
               "allocate CUDA dense direct fields");
    check_cuda(cudaMallocHost(&plan.pinned_moments,
                              std::max(moment_bytes, sizeof(double))),
               "allocate pinned CUDA dense direct moments");
    check_cuda(cudaMallocHost(&plan.pinned_fields,
                              std::max(field_bytes, sizeof(double))),
               "allocate pinned CUDA dense direct fields");

    const std::size_t matrix_bytes =
        plan.matrix_value_count * sizeof(double);
    for (std::size_t component = 0; component < 6; ++component) {
      check_cuda(
          cudaMemcpyAsync(
              plan.device_matrices + component * plan.matrix_value_count,
              host_plan.matrices()[component].data(), matrix_bytes,
              cudaMemcpyHostToDevice, plan.stream),
          "upload CUDA dense direct tensor component");
    }
    check_cuda(cudaStreamSynchronize(plan.stream),
               "complete CUDA dense direct tensor upload");
  } catch (...) {
    delete implementation_;
    implementation_ = nullptr;
    throw;
  }
}

CudaDenseDirectPlan::~CudaDenseDirectPlan()
{
  delete implementation_;
}

std::vector<Vec3> CudaDenseDirectPlan::evaluate(
    const std::span<const Vec3> total_moments)
{
  auto& plan = *implementation_;
  if (total_moments.size() != plan.source_count) {
    throw std::invalid_argument(
        "CUDA dense direct plan requires one moment per source");
  }
  if (plan.target_count == 0) {
    return {};
  }
  if (plan.source_count == 0) {
    return std::vector<Vec3>(plan.target_count);
  }

  for (std::size_t source = 0; source < plan.source_count; ++source) {
    plan.pinned_moments[source] = total_moments[source].x;
    plan.pinned_moments[plan.source_count + source] = total_moments[source].y;
    plan.pinned_moments[2 * plan.source_count + source] =
        total_moments[source].z;
  }
  const std::size_t moment_bytes =
      3 * plan.source_count * sizeof(double);
  check_cuda(cudaMemcpyAsync(plan.device_moments, plan.pinned_moments,
                             moment_bytes, cudaMemcpyHostToDevice, plan.stream),
             "upload CUDA dense direct moments");

  const double alpha = 1.0;
  const double zero = 0.0;
  const double one = 1.0;
  const int source_count = static_cast<int>(plan.source_count);
  const int target_count = static_cast<int>(plan.target_count);
  const auto apply_component = [&](const std::size_t matrix_component,
                                   const std::size_t moment_component,
                                   const std::size_t field_component,
                                   const bool add) {
    // Target-major row storage is equivalent to a column-major Ns x Nt
    // matrix. Its transpose therefore applies the intended Nt x Ns operator.
    check_cublas(
        cublasDgemv(
            plan.cublas_handle, CUBLAS_OP_T, source_count, target_count,
            &alpha,
            plan.device_matrices +
                matrix_component * plan.matrix_value_count,
            source_count,
            plan.device_moments + moment_component * plan.source_count, 1,
            add ? &one : &zero,
            plan.device_fields + field_component * plan.target_count, 1),
        "apply CUDA dense direct tensor component");
  };

  apply_component(0, 0, 0, false);
  apply_component(1, 1, 0, true);
  apply_component(2, 2, 0, true);
  apply_component(1, 0, 1, false);
  apply_component(3, 1, 1, true);
  apply_component(4, 2, 1, true);
  apply_component(2, 0, 2, false);
  apply_component(4, 1, 2, true);
  apply_component(5, 2, 2, true);

  const std::size_t field_bytes = 3 * plan.target_count * sizeof(double);
  check_cuda(cudaMemcpyAsync(plan.pinned_fields, plan.device_fields,
                             field_bytes, cudaMemcpyDeviceToHost, plan.stream),
             "download CUDA dense direct fields");
  check_cuda(cudaStreamSynchronize(plan.stream),
             "complete CUDA dense direct evaluation");

  std::vector<Vec3> result(plan.target_count);
  for (std::size_t target = 0; target < plan.target_count; ++target) {
    result[target] = {
        plan.pinned_fields[target],
        plan.pinned_fields[plan.target_count + target],
        plan.pinned_fields[2 * plan.target_count + target]
    };
  }
  return result;
}

std::size_t CudaDenseDirectPlan::source_count() const noexcept
{
  return implementation_->source_count;
}

std::size_t CudaDenseDirectPlan::target_count() const noexcept
{
  return implementation_->target_count;
}

std::size_t CudaDenseDirectPlan::tensor_memory_bytes() const noexcept
{
  return 6 * implementation_->matrix_value_count * sizeof(double);
}

std::size_t CudaDenseDirectPlan::persistent_device_bytes() const noexcept
{
  return tensor_memory_bytes() +
      3 * (implementation_->source_count + implementation_->target_count) *
          sizeof(double);
}

CudaDirectPlan::CudaDirectPlan(
    const std::span<const Vec3> source_positions,
    const std::span<const Vec3> target_positions,
    const std::span<const int> target_source_indices
)
    : implementation_(new Implementation{}) {
  if (!cuda_runtime_available()) {
    delete implementation_;
    implementation_ = nullptr;
    throw std::runtime_error(
        "CUDA backend requested, but no CUDA-capable device is available");
  }
  auto &plan = *implementation_;
  plan.source_count = source_positions.size();
  plan.target_count = target_positions.size();
  if (!target_source_indices.empty() &&
      target_source_indices.size() != plan.target_count) {
    delete implementation_;
    implementation_ = nullptr;
    throw std::invalid_argument(
        "CudaDirectPlan requires one source identity per target");
  }
  for (const int source_index : target_source_indices) {
    if (source_index < -1 ||
        source_index >= static_cast<int>(plan.source_count)) {
      delete implementation_;
      implementation_ = nullptr;
      throw std::invalid_argument(
          "CudaDirectPlan source identity is out of range");
    }
  }
  plan.self_indices.assign(plan.target_count, -1);
  if (!target_source_indices.empty()) {
    std::copy(
        target_source_indices.begin(),
        target_source_indices.end(),
        plan.self_indices.begin()
    );
  }
    check_cuda(cudaStreamCreateWithFlags(&plan.stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
    check_cuda(cudaEventCreate(&plan.evaluation_start),
               "cudaEventCreate evaluation start");
    check_cuda(cudaEventCreate(&plan.h2d_complete),
               "cudaEventCreate H2D complete");
    check_cuda(cudaEventCreate(&plan.kernel_complete),
               "cudaEventCreate kernel complete");
    check_cuda(cudaEventCreate(&plan.d2h_complete),
               "cudaEventCreate D2H complete");

    const std::size_t source_bytes = plan.source_count * sizeof(Vec3);
  const std::size_t target_bytes = plan.target_count * sizeof(Vec3);
  const std::size_t source_allocation = std::max(source_bytes, sizeof(Vec3));
  const std::size_t target_allocation = std::max(target_bytes, sizeof(Vec3));
  check_cuda(cudaMalloc(&plan.device_sources, source_allocation),
             "cudaMalloc sources");
  check_cuda(cudaMalloc(&plan.device_targets, target_allocation),
             "cudaMalloc targets");
  check_cuda(cudaMalloc(&plan.device_moments, source_allocation),
             "cudaMalloc moments");
  check_cuda(cudaMalloc(&plan.device_fields, target_allocation),
             "cudaMalloc fields");
  check_cuda(
      cudaMalloc(&plan.device_potentials,
                 std::max(plan.target_count * sizeof(double), sizeof(double))),
      "cudaMalloc potentials");
  check_cuda(cudaMalloc(&plan.device_self_indices,
                          std::max(plan.target_count * sizeof(int), sizeof(int))),
               "cudaMalloc identities");
    check_cuda(cudaMallocHost(&plan.pinned_moments, source_allocation),
             "cudaMallocHost moments");
  check_cuda(cudaMallocHost(&plan.pinned_fields, target_allocation),
             "cudaMallocHost fields");
  check_cuda(cudaMallocHost(
                 &plan.pinned_potentials,
                 std::max(plan.target_count * sizeof(double), sizeof(double))),
             "cudaMallocHost potentials");

    check_cuda(cudaMemcpyAsync(plan.device_sources, source_positions.data(),
                               source_bytes, cudaMemcpyHostToDevice, plan.stream),
               "upload source geometry");
    check_cuda(cudaMemcpyAsync(plan.device_targets, target_positions.data(),
                               target_bytes, cudaMemcpyHostToDevice, plan.stream),
               "upload target geometry");
    check_cuda(cudaMemcpyAsync(plan.device_self_indices, plan.self_indices.data(),
                               plan.target_count * sizeof(int),
                               cudaMemcpyHostToDevice, plan.stream),
             "upload initial identity map");
  check_cuda(cudaStreamSynchronize(plan.stream), "synchronise CUDA setup");

  plan.statistics.setup_h2d_bytes =
      source_bytes + target_bytes + plan.target_count * sizeof(int);
  plan.statistics.persistent_device_bytes =
      2 * source_bytes + 2 * target_bytes +
      plan.target_count * (sizeof(double) + sizeof(int));
  plan.statistics.plan_generation_count = 1;
  plan.statistics.static_upload_count = 1;
  plan.statistics.static_m2l_upload_count = 0;
}

CudaDirectPlan::~CudaDirectPlan() {
  if (implementation_ == nullptr) {
    return;
  }
    auto& plan = *implementation_;
    cudaFree(plan.device_sources);
    cudaFree(plan.device_targets);
    cudaFree(plan.device_moments);
    cudaFree(plan.device_fields);
    cudaFree(plan.device_potentials);
    cudaFree(plan.device_self_indices);
    cudaFreeHost(plan.pinned_moments);
    cudaFreeHost(plan.pinned_fields);
    cudaFreeHost(plan.pinned_potentials);
    cudaEventDestroy(plan.evaluation_start);
    cudaEventDestroy(plan.h2d_complete);
    cudaEventDestroy(plan.kernel_complete);
    cudaEventDestroy(plan.d2h_complete);
    cudaStreamDestroy(plan.stream);
    delete implementation_;
}

void CudaDirectPlan::evaluate(
    const std::span<const Vec3> moments,
    const std::span<PotentialField> results,
    const OutputFlags output
) {
  auto &plan = *implementation_;
  if (moments.size() != plan.source_count ||
      results.size() != plan.target_count) {
    throw std::invalid_argument("CUDA evaluation array size mismatch");
  }
  std::copy(moments.begin(), moments.end(), plan.pinned_moments);
    const std::size_t moment_bytes = plan.source_count * sizeof(Vec3);
    check_cuda(cudaEventRecord(plan.evaluation_start, plan.stream),
               "record CUDA evaluation start");
    check_cuda(cudaMemcpyAsync(plan.device_moments, plan.pinned_moments,
                               moment_bytes, cudaMemcpyHostToDevice, plan.stream),
               "upload dipole moments");
    plan.statistics.evaluation_h2d_bytes = moment_bytes;
    plan.statistics.evaluation_h2d_calls = 1;

  check_cuda(cudaEventRecord(plan.h2d_complete, plan.stream),
             "record CUDA H2D completion");

  constexpr int threads = 128;
  const int blocks =
      static_cast<int>((plan.target_count + threads - 1) / threads);
  const bool field = has_flag(output, OutputFlags::Field);
  const bool potential = has_flag(output, OutputFlags::Potential);
  if (plan.target_count != 0) {
    dipole_field_kernel<<<blocks, threads, 0, plan.stream>>>(
        plan.device_targets, plan.device_sources, plan.device_moments,
        plan.device_self_indices, plan.source_count, plan.target_count,
        plan.device_fields, plan.device_potentials, field, potential);
    check_cuda(cudaGetLastError(), "launch CUDA FMM evaluation");
  }
  check_cuda(cudaEventRecord(plan.kernel_complete, plan.stream),
               "record CUDA kernel completion");

    plan.statistics.evaluation_d2h_bytes = 0;
    plan.statistics.evaluation_d2h_calls = 0;
  if (field) {
    const std::size_t field_bytes = plan.target_count * sizeof(Vec3);
    check_cuda(cudaMemcpyAsync(plan.pinned_fields, plan.device_fields,
                               field_bytes, cudaMemcpyDeviceToHost,
                               plan.stream),
               "download magnetic field");
    plan.statistics.evaluation_d2h_bytes += field_bytes;
    ++plan.statistics.evaluation_d2h_calls;
    }
    if (potential) {
    const std::size_t potential_bytes = plan.target_count * sizeof(double);
    check_cuda(cudaMemcpyAsync(plan.pinned_potentials, plan.device_potentials,
                               potential_bytes, cudaMemcpyDeviceToHost,
                               plan.stream),
               "download scalar potential");
    plan.statistics.evaluation_d2h_bytes += potential_bytes;
    ++plan.statistics.evaluation_d2h_calls;
  }
    check_cuda(cudaEventRecord(plan.d2h_complete, plan.stream),
               "record CUDA D2H completion");
    check_cuda(cudaEventSynchronize(plan.d2h_complete),
               "complete CUDA evaluation");
  float h2d_milliseconds = 0.0F;
  float kernel_milliseconds = 0.0F;
  float d2h_milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&h2d_milliseconds, plan.evaluation_start,
                                  plan.h2d_complete),
             "measure CUDA H2D time");
  check_cuda(cudaEventElapsedTime(&kernel_milliseconds, plan.h2d_complete,
                                  plan.kernel_complete),
             "measure CUDA kernel time");
  check_cuda(cudaEventElapsedTime(&d2h_milliseconds, plan.kernel_complete,
                                  plan.d2h_complete),
             "measure CUDA D2H time");
  plan.evaluation_timings.h2d_seconds =
      static_cast<double>(h2d_milliseconds) * 1.0e-3;
    plan.evaluation_timings.kernel_seconds =
        static_cast<double>(kernel_milliseconds) * 1.0e-3;
    plan.evaluation_timings.d2h_seconds =
        static_cast<double>(d2h_milliseconds) * 1.0e-3;
    for (std::size_t index = 0; index < plan.target_count; ++index) {
        results[index].H = field ? plan.pinned_fields[index] : Vec3{};
        results[index].phi = potential ? plan.pinned_potentials[index] : 0.0;
  }
}

std::size_t CudaDirectPlan::source_count() const noexcept {
  return implementation_->source_count;
}

std::size_t CudaDirectPlan::target_count() const noexcept {
  return implementation_->target_count;
}

const CudaPlanStatistics &CudaDirectPlan::statistics() const noexcept {
  return implementation_->statistics;
}

const CudaEvaluationTimings &
CudaDirectPlan::evaluation_timings() const noexcept {
  return implementation_->evaluation_timings;
}

std::vector<PotentialField> cuda_direct_p2p_reference(
    const std::span<const Vec3> targets, const std::span<const Vec3> sources,
    const std::span<const Vec3> moments, const OutputFlags output,
    const std::span<const int> target_source_indices) {
  if (moments.size() != sources.size()) {
    throw std::invalid_argument(
        "cuda_direct_p2p_reference requires one moment per source");
  }
  if (!target_source_indices.empty() &&
      target_source_indices.size() != targets.size()) {
    throw std::invalid_argument(
        "cuda_direct_p2p_reference identity map has incorrect length");
  }

  CudaDirectPlan plan(sources, targets, target_source_indices);
    std::vector<PotentialField> results(targets.size());
    plan.evaluate(moments, results, output);
    return results;
}

//------------------------------------------------------------------------------
// Hybrid static M2L plan
//------------------------------------------------------------------------------

struct CudaM2LPlan::Implementation {
  int coefficient_count{0};
  int node_count{0};
  std::vector<double> host_multipoles{};
  std::vector<double> host_locals{};
  CudaM2LExecutionPlan *executor{nullptr};
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
  plan.host_multipoles.resize(coefficient_values);
  plan.host_locals.resize(coefficient_values);
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
  plan.executor = new CudaM2LExecutionPlan(data, plan.stream);
  check_cuda(cudaStreamSynchronize(plan.stream), "finish M2L plan upload");
  plan.statistics = plan.executor->statistics();
  plan.statistics.persistent_device_bytes +=
      2 * coefficient_values * sizeof(double);
}

CudaM2LPlan::~CudaM2LPlan() {
  if (!implementation_) return;
  auto& plan = *implementation_;
  delete plan.executor;
  cudaFree(plan.multipoles); cudaFree(plan.locals);
  cudaEventDestroy(plan.start); cudaEventDestroy(plan.h2d);
  cudaEventDestroy(plan.scale);
  cudaEventDestroy(plan.kernel); cudaEventDestroy(plan.d2h);
  cudaStreamDestroy(plan.stream); delete implementation_;
}

void CudaM2LPlan::evaluate(const std::span<const double> multipoles,
                           const std::span<double> locals) {
  auto& plan = *implementation_;
  if (multipoles.size() != plan.host_multipoles.size() ||
      locals.size() != plan.host_locals.size()) {
    throw std::invalid_argument("CUDA M2L coefficient dimensions differ");
  }
  std::copy(multipoles.begin(), multipoles.end(), plan.host_multipoles.begin());
  check_cuda(cudaEventRecord(plan.start, plan.stream), "record M2L start");
  if (!plan.host_multipoles.empty()) {
    check_cuda(cudaMemcpyAsync(plan.multipoles, plan.host_multipoles.data(),
        plan.host_multipoles.size() * sizeof(double), cudaMemcpyHostToDevice,
        plan.stream), "upload M2L multipoles");
  }
  check_cuda(cudaEventRecord(plan.h2d, plan.stream), "record M2L H2D");
  if (!plan.host_locals.empty()) {
    check_cuda(cudaMemsetAsync(plan.locals, 0,
        plan.host_locals.size() * sizeof(double), plan.stream),
        "clear M2L locals");
  }
  plan.executor->enqueue(plan.multipoles, plan.locals, plan.stream, plan.scale);
  check_cuda(cudaEventRecord(plan.kernel, plan.stream), "record M2L kernel");
  if (!plan.host_locals.empty()) {
    check_cuda(cudaMemcpyAsync(plan.host_locals.data(), plan.locals,
        plan.host_locals.size() * sizeof(double), cudaMemcpyDeviceToHost,
        plan.stream), "download M2L locals");
  }
  check_cuda(cudaEventRecord(plan.d2h, plan.stream), "record M2L D2H");
  check_cuda(cudaEventSynchronize(plan.d2h), "wait for M2L");
  std::copy(plan.host_locals.begin(), plan.host_locals.end(), locals.begin());
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
  plan.statistics.evaluation_h2d_bytes =
      plan.host_multipoles.size() * sizeof(double);
  plan.statistics.evaluation_d2h_bytes =
      plan.host_locals.size() * sizeof(double);
  ++plan.statistics.evaluation_h2d_calls;
  ++plan.statistics.evaluation_d2h_calls;
}

const CudaPlanStatistics& CudaM2LPlan::statistics() const noexcept {
  return implementation_->statistics;
}
const CudaEvaluationTimings& CudaM2LPlan::timings() const noexcept {
  return implementation_->timings;
}

namespace {

__global__ void static_p2p_kernel(const int target_count,
                                  const int *row_offsets,
                                  const StaticDipoleBlock *blocks,
                                  const Vec3 *moments, const int *self_indices,
                                  Vec3 *fields) {
  const int target = blockIdx.x * blockDim.x + threadIdx.x;
  if (target >= target_count) {
    return;
  }
  Vec3 field;
  field.x = 0.0;
  field.y = 0.0;
  field.z = 0.0;
  const int self = self_indices[target];
  for (int entry = row_offsets[target]; entry < row_offsets[target + 1];
       ++entry) {
    const StaticDipoleBlock tensor = blocks[entry];
    if (tensor.source == self) {
      continue;
    }
    const Vec3 moment = moments[tensor.source];
    accumulate_static_dipole_block(tensor, moment, field);
  }
  fields[target] = field;
}

struct CudaP2PDeviceView {
  int target_count{0};
  int *row_offsets{nullptr};
  StaticDipoleBlock *blocks{nullptr};
};

void launch_static_p2p(const CudaP2PDeviceView &plan,
                       const Vec3 *moments,
                       const int *self_indices,
                       Vec3 *fields,
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

__global__ void compact_p2p_kernel(
    const int target_count,
    const int* row_offsets,
    const int* source_indices,
    const double* tensors,
    const std::size_t interaction_count,
    const Vec3* moments,
    const int* self_indices,
    Vec3* fields)
{
  const int target = blockIdx.x * blockDim.x + threadIdx.x;
  if (target >= target_count) {
    return;
  }

  Vec3 field{};
  const int self = self_indices[target];
  for (int entry = row_offsets[target]; entry < row_offsets[target + 1];
       ++entry) {
    const int source = source_indices[entry];
    if (source == self) {
      continue;
    }
    const std::size_t index = static_cast<std::size_t>(entry);
    const Vec3 moment = moments[source];
    const double xx = tensors[index];
    const double xy = tensors[interaction_count + index];
    const double xz = tensors[2 * interaction_count + index];
    const double yy = tensors[3 * interaction_count + index];
    const double yz = tensors[4 * interaction_count + index];
    const double zz = tensors[5 * interaction_count + index];
    field.x += xx * moment.x + xy * moment.y + xz * moment.z;
    field.y += xy * moment.x + yy * moment.y + yz * moment.z;
    field.z += xz * moment.x + yz * moment.y + zz * moment.z;
  }
  fields[target] = field;
}

struct CudaCompactP2PDeviceView {
  int target_count{0};
  std::size_t interaction_count{0};
  int* row_offsets{nullptr};
  int* source_indices{nullptr};
  double* tensors{nullptr};
};

void launch_compact_p2p(
    const CudaCompactP2PDeviceView& plan,
    const Vec3* moments,
    const int* self_indices,
    Vec3* fields,
    cudaStream_t stream)
{
  if (plan.target_count == 0) {
    return;
  }
  compact_p2p_kernel<<<
      (plan.target_count + static_operator_threads - 1) /
          static_operator_threads,
      static_operator_threads, 0, stream>>>(
      plan.target_count, plan.row_offsets, plan.source_indices, plan.tensors,
      plan.interaction_count, moments, self_indices, fields);
  check_cuda(cudaGetLastError(), "launch compact static P2P kernel");
}

constexpr int p2p_source_batch_size = 128;

__global__ void leaf_p2p_kernel(
    const int* target_begins,
    const int* target_counts,
    const int* leaf_row_offsets,
    const StaticP2PLeafBlock* leaf_blocks,
    const double* tensors,
    const std::size_t interaction_count,
    const Vec3* moments,
    const int* self_indices,
    Vec3* fields)
{
  extern __shared__ Vec3 shared_moments[];
  const int target_leaf = blockIdx.x;
  const int target_begin = target_begins[target_leaf];
  const int target_count = target_counts[target_leaf];

  // All threads execute every source-batch barrier. Threads outside the final
  // target batch participate in staging but do not accumulate or write.
  for (int target_base = 0; target_base < target_count;
       target_base += blockDim.x) {
    const int local_target = target_base + threadIdx.x;
    const bool active = local_target < target_count;
    const int target = target_begin + local_target;
    const int self = active ? self_indices[target] : -1;
    Vec3 field{};

    for (int block_index = leaf_row_offsets[target_leaf];
         block_index < leaf_row_offsets[target_leaf + 1]; ++block_index) {
      const StaticP2PLeafBlock leaf_block = leaf_blocks[block_index];
      for (int source_base = 0; source_base < leaf_block.source_count;
           source_base += p2p_source_batch_size) {
        const int batch_count = min(
            p2p_source_batch_size, leaf_block.source_count - source_base);
        for (int local_source = threadIdx.x; local_source < batch_count;
             local_source += blockDim.x) {
          shared_moments[local_source] =
              moments[leaf_block.source_begin + source_base + local_source];
        }
        __syncthreads();

        if (active) {
          for (int batch_source = 0; batch_source < batch_count;
               ++batch_source) {
            const int local_source = source_base + batch_source;
            const int source = leaf_block.source_begin + local_source;
            if (source == self) {
              continue;
            }

            // CUDA tensors are transposed during setup to source-major order.
            // Consecutive target threads therefore read consecutive entries.
            const std::size_t index = leaf_block.tensor_offset +
                static_cast<std::size_t>(local_source) * target_count +
                local_target;
            const Vec3 moment = shared_moments[batch_source];
            const double xx = tensors[index];
            const double xy = tensors[interaction_count + index];
            const double xz = tensors[2 * interaction_count + index];
            const double yy = tensors[3 * interaction_count + index];
            const double yz = tensors[4 * interaction_count + index];
            const double zz = tensors[5 * interaction_count + index];
            field.x += xx * moment.x + xy * moment.y + xz * moment.z;
            field.y += xy * moment.x + yy * moment.y + yz * moment.z;
            field.z += xz * moment.x + yz * moment.y + zz * moment.z;
          }
        }
        __syncthreads();
      }
    }
    if (active) {
      fields[target] = field;
    }
  }
}

struct CudaLeafP2PDeviceView {
  int target_count{0};
  int target_leaf_count{0};
  int threads_per_block{0};
  std::size_t interaction_count{0};
  int* target_begins{nullptr};
  int* target_counts{nullptr};
  int* leaf_row_offsets{nullptr};
  StaticP2PLeafBlock* leaf_blocks{nullptr};
  double* tensors{nullptr};
};

void launch_leaf_p2p(
    const CudaLeafP2PDeviceView& plan,
    const Vec3* moments,
    const int* self_indices,
    Vec3* fields,
    cudaStream_t stream)
{
  if (plan.target_count == 0) {
    return;
  }
  check_cuda(
      cudaMemsetAsync(
          fields, 0, static_cast<std::size_t>(plan.target_count) * sizeof(Vec3),
          stream),
      "clear leaf P2P fields");
  if (plan.target_leaf_count == 0) {
    return;
  }
  leaf_p2p_kernel<<<
      plan.target_leaf_count, plan.threads_per_block,
      p2p_source_batch_size * sizeof(Vec3), stream>>>(
      plan.target_begins, plan.target_counts, plan.leaf_row_offsets,
      plan.leaf_blocks, plan.tensors, plan.interaction_count, moments,
      self_indices, fields);
  check_cuda(cudaGetLastError(), "launch leaf-block static P2P kernel");
}

struct CudaBsrP2PDeviceView {
  int target_count{0};
  int source_count{0};
  int interaction_count{0};
  int* row_offsets{nullptr};
  int* source_indices{nullptr};
  double* values{nullptr};
  cusparseHandle_t handle{nullptr};
  cusparseMatDescr_t descriptor{nullptr};
};

void launch_bsr_p2p(
    const CudaBsrP2PDeviceView& plan,
    const Vec3* moments,
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
      cusparseDbsrmv(
          plan.handle, CUSPARSE_DIRECTION_ROW,
          CUSPARSE_OPERATION_NON_TRANSPOSE, plan.target_count,
          plan.source_count, plan.interaction_count, &alpha, plan.descriptor,
          plan.values, plan.row_offsets, plan.source_indices, 3,
          reinterpret_cast<const double*>(moments), &beta,
          reinterpret_cast<double*>(fields)),
      "launch cuSPARSE BSR(3) P2P");
}

} // namespace

struct CudaP2PPlan::Implementation {
  enum class Kind {
    Canonical,
    Compact,
    Leaf,
    Bsr
  };

  int source_count{0};
  int target_count{0};
  Kind kind{Kind::Canonical};
  CudaP2PDeviceView canonical{};
  CudaCompactP2PDeviceView compact{};
  CudaLeafP2PDeviceView leaf{};
  CudaBsrP2PDeviceView bsr{};
  Vec3* moments{nullptr};
  int* self_indices{nullptr};
  Vec3* fields{nullptr};
  Vec3* pinned_moments{nullptr};
  int* pinned_self_indices{nullptr};
  Vec3* pinned_fields{nullptr};
  cudaStream_t stream{};
  cudaEvent_t start{};
  cudaEvent_t h2d{};
  cudaEvent_t kernel{};
  cudaEvent_t d2h{};
  CudaPlanStatistics statistics{};
  CudaEvaluationTimings timings{};
  std::vector<int> fixed_self_indices{};
  bool dynamic_self_identities{true};
  bool pending{false};
};

CudaP2PPlan::CudaP2PPlan(
    const int source_count,
    const int target_count,
    const std::span<const int> fixed_self_indices,
    const bool device_self_indices)
    : implementation_(new Implementation{})
{
  auto &plan = *implementation_;
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
  check_cuda(cudaEventCreate(&plan.start), "create static P2P event");
  check_cuda(cudaEventCreate(&plan.h2d), "create static P2P event");
  check_cuda(cudaEventCreate(&plan.kernel), "create static P2P event");
  check_cuda(cudaEventCreate(&plan.d2h), "create static P2P event");
  check_cuda(cudaMalloc(&plan.moments,
                        std::max(source_count * sizeof(Vec3),
                                 sizeof(Vec3))),
             "allocate P2P moments");
  if (device_self_indices) {
    check_cuda(cudaMalloc(&plan.self_indices,
                          std::max(target_count * sizeof(int), sizeof(int))),
               "allocate P2P identities");
  }
  check_cuda(cudaMalloc(&plan.fields,
                        std::max(target_count * sizeof(Vec3), sizeof(Vec3))),
             "allocate P2P fields");
  check_cuda(cudaMallocHost(&plan.pinned_moments,
                            std::max(source_count * sizeof(Vec3), sizeof(Vec3))),
             "allocate pinned P2P moments");
  if (device_self_indices && plan.dynamic_self_identities) {
    check_cuda(cudaMallocHost(
                   &plan.pinned_self_indices,
                   std::max(target_count * sizeof(int), sizeof(int))),
               "allocate pinned P2P identities");
  }
  check_cuda(cudaMallocHost(&plan.pinned_fields,
                            std::max(target_count * sizeof(Vec3), sizeof(Vec3))),
             "allocate pinned P2P fields");

  plan.statistics.persistent_device_bytes =
      static_cast<std::size_t>(source_count + target_count) * sizeof(Vec3) +
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
      operator_map.blocks.size() * 2 * sizeof(int);
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
  plan.statistics.setup_h2d_bytes += row_bytes + index_bytes + tensor_bytes;
  plan.statistics.persistent_device_bytes +=
      row_bytes + index_bytes + tensor_bytes;
  plan.statistics.p2p_interaction_count = compact.source_indices.size();
  plan.statistics.p2p_tensor_bytes = tensor_bytes;
  plan.statistics.p2p_index_bytes = index_bytes;
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
  plan.leaf.target_count = plan.target_count;
  plan.leaf.target_leaf_count = static_cast<int>(leaf.target_begins.size());
  plan.leaf.interaction_count = leaf.tensors[0].size();
  int maximum_target_count = 1;
  for (const int target_count : leaf.target_counts) {
    maximum_target_count = std::max(maximum_target_count, target_count);
  }
  plan.leaf.threads_per_block = 32;
  while (plan.leaf.threads_per_block < maximum_target_count &&
         plan.leaf.threads_per_block < 256) {
    plan.leaf.threads_per_block *= 2;
  }

  const std::size_t target_metadata_bytes =
      (leaf.target_begins.size() + leaf.target_counts.size()) * sizeof(int);
  const std::size_t row_bytes = leaf.leaf_row_offsets.size() * sizeof(int);
  const std::size_t block_bytes =
      leaf.blocks.size() * sizeof(StaticP2PLeafBlock);
  const std::size_t tensor_bytes =
      leaf.tensors[0].size() * 6 * sizeof(double);
  check_cuda(cudaMalloc(&plan.leaf.target_begins,
                        std::max(leaf.target_begins.size() * sizeof(int),
                                 sizeof(int))),
             "allocate leaf P2P target begins");
  check_cuda(cudaMalloc(&plan.leaf.target_counts,
                        std::max(leaf.target_counts.size() * sizeof(int),
                                 sizeof(int))),
             "allocate leaf P2P target counts");
  check_cuda(cudaMalloc(&plan.leaf.leaf_row_offsets,
                        std::max(row_bytes, sizeof(int))),
             "allocate leaf P2P rows");
  check_cuda(cudaMalloc(&plan.leaf.leaf_blocks,
                        std::max(block_bytes, sizeof(StaticP2PLeafBlock))),
             "allocate leaf P2P blocks");
  check_cuda(cudaMalloc(&plan.leaf.tensors,
                        std::max(tensor_bytes, sizeof(double))),
             "allocate leaf P2P tensors");

  const auto upload = [](void* destination, const void* source,
                         const std::size_t bytes, const char* operation) {
    if (bytes != 0) {
      check_cuda(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice),
                 operation);
    }
  };
  upload(plan.leaf.target_begins, leaf.target_begins.data(),
         leaf.target_begins.size() * sizeof(int),
         "upload leaf P2P target begins");
  upload(plan.leaf.target_counts, leaf.target_counts.data(),
         leaf.target_counts.size() * sizeof(int),
         "upload leaf P2P target counts");
  upload(plan.leaf.leaf_row_offsets, leaf.leaf_row_offsets.data(), row_bytes,
         "upload leaf P2P rows");
  upload(plan.leaf.leaf_blocks, leaf.blocks.data(), block_bytes,
         "upload leaf P2P blocks");

  // Portable leaf tensors are target-major. CUDA transposes each dense leaf
  // rectangle so neighbouring target threads read consecutive coefficients.
  std::array<std::vector<double>, 6> transposed;
  for (auto& component : transposed) {
    component.resize(leaf.tensors[0].size());
  }
  for (std::size_t target_leaf = 0;
       target_leaf < leaf.target_begins.size(); ++target_leaf) {
    const int target_count = leaf.target_counts[target_leaf];
    for (int block_index = leaf.leaf_row_offsets[target_leaf];
         block_index < leaf.leaf_row_offsets[target_leaf + 1]; ++block_index) {
      const StaticP2PLeafBlock& block =
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
          for (std::size_t component = 0; component < 6; ++component) {
            transposed[component][destination_index] =
                leaf.tensors[component][source_index];
          }
        }
      }
    }
  }
  for (std::size_t component = 0; component < 6; ++component) {
    upload(plan.leaf.tensors + component * leaf.tensors[0].size(),
           transposed[component].data(),
           transposed[component].size() * sizeof(double),
           "upload transposed leaf P2P tensor component");
  }

  plan.statistics.setup_h2d_bytes +=
      target_metadata_bytes + row_bytes + block_bytes + tensor_bytes;
  plan.statistics.persistent_device_bytes +=
      target_metadata_bytes + row_bytes + block_bytes + tensor_bytes;
  plan.statistics.p2p_interaction_count = leaf.tensors[0].size();
  plan.statistics.p2p_tensor_bytes = tensor_bytes;
  plan.statistics.p2p_row_metadata_bytes = row_bytes;
  plan.statistics.p2p_leaf_metadata_bytes =
      target_metadata_bytes + block_bytes;
  plan.statistics.p2p_scratch_bytes =
      p2p_source_batch_size * sizeof(Vec3);
  plan.statistics.p2p_threads_per_block = plan.leaf.threads_per_block;
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
  check_cusparse(cusparseCreate(&plan.bsr.handle),
                 "create cuSPARSE P2P handle");
  check_cusparse(cusparseCreateMatDescr(&plan.bsr.descriptor),
                 "create cuSPARSE P2P descriptor");
  check_cusparse(
      cusparseSetMatType(plan.bsr.descriptor, CUSPARSE_MATRIX_TYPE_GENERAL),
      "set cuSPARSE P2P matrix type");
  check_cusparse(
      cusparseSetMatIndexBase(
          plan.bsr.descriptor, CUSPARSE_INDEX_BASE_ZERO),
      "set cuSPARSE P2P index base");

  plan.statistics.setup_h2d_bytes += row_bytes + index_bytes + tensor_bytes;
  plan.statistics.persistent_device_bytes +=
      row_bytes + index_bytes + tensor_bytes;
  plan.statistics.p2p_interaction_count = bsr.source_indices.size();
  plan.statistics.p2p_tensor_bytes = tensor_bytes;
  plan.statistics.p2p_index_bytes = index_bytes;
  plan.statistics.p2p_row_metadata_bytes = row_bytes;
  plan.statistics.p2p_threads_per_block = 0;
}

CudaP2PPlan::~CudaP2PPlan() {
  if (implementation_ == nullptr) {
    return;
  }
  auto& plan = *implementation_;
  cancel_evaluate();
  cudaFree(plan.canonical.row_offsets);
  cudaFree(plan.canonical.blocks);
  cudaFree(plan.compact.row_offsets);
  cudaFree(plan.compact.source_indices);
  cudaFree(plan.compact.tensors);
  cudaFree(plan.leaf.target_begins);
  cudaFree(plan.leaf.target_counts);
  cudaFree(plan.leaf.leaf_row_offsets);
  cudaFree(plan.leaf.leaf_blocks);
  cudaFree(plan.leaf.tensors);
  if (plan.bsr.descriptor != nullptr) {
    cusparseDestroyMatDescr(plan.bsr.descriptor);
  }
  if (plan.bsr.handle != nullptr) {
    cusparseDestroy(plan.bsr.handle);
  }
  cudaFree(plan.bsr.row_offsets);
  cudaFree(plan.bsr.source_indices);
  cudaFree(plan.bsr.values);
  cudaFree(plan.moments);
  cudaFree(plan.self_indices);
  cudaFree(plan.fields);
  cudaFreeHost(plan.pinned_moments);
  cudaFreeHost(plan.pinned_self_indices);
  cudaFreeHost(plan.pinned_fields);
  cudaEventDestroy(plan.start);
  cudaEventDestroy(plan.h2d);
  cudaEventDestroy(plan.kernel);
  cudaEventDestroy(plan.d2h);
  cudaStreamDestroy(plan.stream);
  delete implementation_;
}

void CudaP2PPlan::begin_evaluate(
    const std::span<const Vec3> moments,
    const std::span<const int> target_source_indices) {
  auto &plan = *implementation_;
  if (moments.size() != static_cast<std::size_t>(plan.source_count) ||
      target_source_indices.size() !=
          static_cast<std::size_t>(plan.target_count)) {
    throw std::invalid_argument("CUDA static P2P dimensions are inconsistent");
  }
  if (!plan.dynamic_self_identities &&
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
  if (plan.dynamic_self_identities) {
    std::copy(target_source_indices.begin(), target_source_indices.end(),
              plan.pinned_self_indices);
  }
  plan.pending = true;
  try {
    check_cuda(cudaEventRecord(plan.start, plan.stream), "record P2P start");
    check_cuda(cudaMemcpyAsync(plan.moments, plan.pinned_moments,
                               moments.size_bytes(), cudaMemcpyHostToDevice,
                               plan.stream),
               "upload P2P moments");
    if (plan.dynamic_self_identities) {
      check_cuda(cudaMemcpyAsync(
                     plan.self_indices, plan.pinned_self_indices,
                     target_source_indices.size_bytes(),
                     cudaMemcpyHostToDevice, plan.stream),
                 "upload P2P identities");
    }
    check_cuda(cudaEventRecord(plan.h2d, plan.stream), "record P2P upload");
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
    case Implementation::Kind::Bsr:
      launch_bsr_p2p(plan.bsr, plan.moments, plan.fields, plan.stream);
      break;
    }
    check_cuda(cudaEventRecord(plan.kernel, plan.stream), "record P2P kernel");
    check_cuda(cudaMemcpyAsync(plan.pinned_fields, plan.fields,
                               static_cast<std::size_t>(plan.target_count) *
                                   sizeof(Vec3),
                               cudaMemcpyDeviceToHost, plan.stream),
               "download P2P fields");
    check_cuda(cudaEventRecord(plan.d2h, plan.stream), "record P2P download");
  } catch (...) {
    cancel_evaluate();
    throw;
  }
  plan.statistics.evaluation_h2d_bytes = moments.size_bytes() +
      (plan.dynamic_self_identities ? target_source_indices.size_bytes() : 0);
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
  auto elapsed = [](cudaEvent_t first, cudaEvent_t second) {
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, first, second),
               "time static P2P phase");
    return static_cast<double>(milliseconds) * 1.0e-3;
  };
  plan.timings = {};
  plan.timings.h2d_seconds = elapsed(plan.start, plan.h2d);
  plan.timings.kernel_seconds = elapsed(plan.h2d, plan.kernel);
  plan.timings.p2p_seconds = plan.timings.kernel_seconds;
  plan.timings.d2h_seconds = elapsed(plan.kernel, plan.d2h);
  plan.timings.total_seconds = plan.timings.h2d_seconds +
      plan.timings.kernel_seconds + plan.timings.d2h_seconds;
  plan.pending = false;
}

void CudaP2PPlan::cancel_evaluate() noexcept {
  if (implementation_ == nullptr || !implementation_->pending) {
    return;
  }
    cudaStreamSynchronize(implementation_->stream);
    implementation_->pending = false;
}

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

const CudaPlanStatistics &CudaP2PPlan::statistics() const noexcept {
  return implementation_->statistics;
}

const CudaEvaluationTimings &CudaP2PPlan::timings() const noexcept {
  return implementation_->timings;
}

//------------------------------------------------------------------------------
// Complete static CUDA FMM
//------------------------------------------------------------------------------

struct CudaFullPlan::Implementation {
    int coefficient_count{0};
    int node_count{0};
    int source_count{0};
    int target_count{0};
    int* source_permutation{nullptr};
    int* target_permutation{nullptr};
    int* self_indices{nullptr};
  CudaP2PDeviceView p2p{};
  CudaBsrP2PDeviceView p2p_bsr{};
  bool use_p2p_bsr{false};
  StaticOperatorEntry *entries{nullptr};
  StaticOperatorEntry *m2m_matrices{nullptr};
  StaticOperatorEntry *l2l_matrices{nullptr};
  CudaTranslationInteraction *m2m_interactions{nullptr};
  CudaTranslationInteraction *l2l_interactions{nullptr};
  CudaM2LExecutionPlan *m2l{nullptr};
  std::vector<std::size_t> offsets{};
  std::vector<std::size_t> counts{};
  Vec3 *moments{nullptr};
    Vec3* sorted_moments{nullptr};
    double* multipoles{nullptr};
    double* locals{nullptr};
    Vec3* far_fields{nullptr};
    Vec3* near_fields{nullptr};
    Vec3* final_fields{nullptr};
    Vec3* pinned_moments{nullptr};
    Vec3* pinned_fields{nullptr};
    cudaStream_t far_field_stream{};
    cudaStream_t near_field_stream{};
    cudaEvent_t evaluation_start{};
    cudaEvent_t moments_ready{};
    cudaEvent_t p2m_complete{};
    cudaEvent_t m2m_complete{};
    cudaEvent_t m2l_scale_complete{};
    cudaEvent_t m2l_complete{};
    cudaEvent_t l2l_complete{};
    cudaEvent_t l2p_complete{};
    cudaEvent_t p2p_start{};
    cudaEvent_t p2p_complete{};
    cudaEvent_t combination_start{};
    cudaEvent_t combination_complete{};
    cudaEvent_t d2h_complete{};
    CudaPlanStatistics statistics{};
    CudaEvaluationTimings timings{};
  std::vector<int> fixed_self_indices{};
  bool identity_initialised{false};
  int p2m_stage{0};
  int leaf_level{0};
  int m2m_entries_per_matrix{0};
  int l2l_entries_per_matrix{0};
  std::size_t m2m_interaction_count{0};
  std::size_t l2l_interaction_count{0};
  int l2p_stage{0};
  std::size_t p2p_block_count{0};
};

CudaFullPlan::CudaFullPlan(const CudaFullPlanData &data)
    : implementation_(new Implementation{}) {
  auto &plan = *implementation_;
  plan.coefficient_count = data.coefficient_count;
  plan.node_count = data.node_count;
    plan.source_count = data.source_count;
  plan.target_count = data.target_count;
  plan.use_p2p_bsr = data.use_p2p_bsr;
  plan.p2p.target_count = plan.target_count;
  plan.p2p_bsr.target_count = plan.target_count;
  plan.p2p_bsr.source_count = plan.source_count;
  plan.p2p_bsr.interaction_count =
      static_cast<int>(data.p2p_bsr.source_indices.size());
  plan.p2p_block_count = data.use_p2p_bsr
                             ? data.p2p_bsr.source_indices.size()
                             : data.p2p.blocks.size();
    std::vector<StaticOperatorEntry> entries;
    const auto append_stage = [&](const auto& stage_entries) {
        plan.offsets.push_back(entries.size());
        plan.counts.push_back(stage_entries.size());
        entries.insert(entries.end(), stage_entries.begin(), stage_entries.end());
    return static_cast<int>(plan.offsets.size() - 1);
  };
  plan.p2m_stage = append_stage(data.p2m);
  plan.l2p_stage = append_stage(data.l2p);
  plan.leaf_level = data.m2l.level_count - 1;
  plan.m2m_entries_per_matrix = data.m2m.entries_per_matrix;
  plan.l2l_entries_per_matrix = data.l2l.entries_per_matrix;
  plan.m2m_interaction_count = data.m2m.interactions.size();
  plan.l2l_interaction_count = data.l2l.interactions.size();

  check_cuda(cudaStreamCreateWithFlags(&plan.far_field_stream,
                                       cudaStreamNonBlocking),
             "create full FMM far-field stream");
  check_cuda(cudaStreamCreateWithFlags(&plan.near_field_stream,
                                       cudaStreamNonBlocking),
             "create full FMM near-field stream");
  const std::array<cudaEvent_t *, 13> events{
      &plan.evaluation_start,
      &plan.moments_ready,
      &plan.p2m_complete,
      &plan.m2m_complete,
      &plan.m2l_scale_complete,
      &plan.m2l_complete,
      &plan.l2l_complete,
      &plan.l2p_complete,
      &plan.p2p_start,
      &plan.p2p_complete,
      &plan.combination_start,
      &plan.combination_complete,
      &plan.d2h_complete,
  };
  for (cudaEvent_t *event : events) {
    check_cuda(cudaEventCreate(event), "create full FMM event");
  }
    const auto allocate = [](auto** pointer, const std::size_t bytes) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(pointer),
                          std::max(bytes, std::size_t{1})),
               "allocate full FMM buffer");
  };
  const std::size_t source_bytes =
      static_cast<std::size_t>(plan.source_count) * sizeof(Vec3);
  const std::size_t target_bytes =
      static_cast<std::size_t>(plan.target_count) * sizeof(Vec3);
  const std::size_t coefficient_bytes =
      static_cast<std::size_t>(plan.node_count) * plan.coefficient_count *
      sizeof(double);
  allocate(&plan.source_permutation,
           data.source_permutation.size() * sizeof(int));
  allocate(&plan.target_permutation,
           data.target_permutation.size() * sizeof(int));
  if (!plan.use_p2p_bsr) {
    allocate(&plan.self_indices,
             static_cast<std::size_t>(plan.target_count) * sizeof(int));
    allocate(&plan.p2p.row_offsets, data.p2p.row_offsets.size() * sizeof(int));
    allocate(&plan.p2p.blocks,
             data.p2p.blocks.size() * sizeof(StaticDipoleBlock));
  } else {
    allocate(&plan.p2p_bsr.row_offsets,
             data.p2p_bsr.row_offsets.size() * sizeof(int));
    allocate(&plan.p2p_bsr.source_indices,
             data.p2p_bsr.source_indices.size() * sizeof(int));
    allocate(&plan.p2p_bsr.values,
             data.p2p_bsr.values.size() * sizeof(double));
    check_cusparse(cusparseCreate(&plan.p2p_bsr.handle),
                   "create full FMM cuSPARSE P2P handle");
    check_cusparse(cusparseCreateMatDescr(&plan.p2p_bsr.descriptor),
                   "create full FMM cuSPARSE P2P descriptor");
    check_cusparse(
        cusparseSetMatType(plan.p2p_bsr.descriptor,
                           CUSPARSE_MATRIX_TYPE_GENERAL),
        "set full FMM cuSPARSE P2P matrix type");
    check_cusparse(
        cusparseSetMatIndexBase(plan.p2p_bsr.descriptor,
                                CUSPARSE_INDEX_BASE_ZERO),
        "set full FMM cuSPARSE P2P index base");
  }
  allocate(&plan.entries, entries.size() * sizeof(StaticOperatorEntry));
  allocate(&plan.m2m_matrices,
           data.m2m.matrices.size() * sizeof(StaticOperatorEntry));
  allocate(&plan.l2l_matrices,
           data.l2l.matrices.size() * sizeof(StaticOperatorEntry));
  allocate(&plan.m2m_interactions,
           data.m2m.interactions.size() * sizeof(CudaTranslationInteraction));
  allocate(&plan.l2l_interactions,
           data.l2l.interactions.size() * sizeof(CudaTranslationInteraction));
  allocate(&plan.moments, source_bytes);
  allocate(&plan.sorted_moments, source_bytes);
  allocate(&plan.multipoles, coefficient_bytes);
    allocate(&plan.locals, coefficient_bytes);
  allocate(&plan.far_fields, target_bytes);
  allocate(&plan.near_fields, target_bytes);
  allocate(&plan.final_fields, target_bytes);
  check_cuda(cudaMallocHost(&plan.pinned_moments,
                            std::max(source_bytes, std::size_t{1})),
             "allocate pinned full FMM moments");
  check_cuda(cudaMallocHost(&plan.pinned_fields,
                            std::max(target_bytes, std::size_t{1})),
             "allocate pinned full FMM fields");
  const auto upload = [&](void *destination, const void *source,
                          const std::size_t bytes) {
        if (bytes != 0) {
            check_cuda(cudaMemcpyAsync(destination, source, bytes,
                                       cudaMemcpyHostToDevice,
                                       plan.far_field_stream),
                       "upload full FMM static data");
            plan.statistics.setup_h2d_bytes += bytes;
        }
    };
    upload(plan.source_permutation, data.source_permutation.data(),
           data.source_permutation.size() * sizeof(int));
    upload(plan.target_permutation, data.target_permutation.data(),
           data.target_permutation.size() * sizeof(int));
  if (!plan.use_p2p_bsr) {
    upload(plan.p2p.row_offsets, data.p2p.row_offsets.data(),
           data.p2p.row_offsets.size() * sizeof(int));
    upload(plan.p2p.blocks, data.p2p.blocks.data(),
           data.p2p.blocks.size() * sizeof(StaticDipoleBlock));
  } else {
    upload(plan.p2p_bsr.row_offsets, data.p2p_bsr.row_offsets.data(),
           data.p2p_bsr.row_offsets.size() * sizeof(int));
    upload(plan.p2p_bsr.source_indices, data.p2p_bsr.source_indices.data(),
           data.p2p_bsr.source_indices.size() * sizeof(int));
    upload(plan.p2p_bsr.values, data.p2p_bsr.values.data(),
           data.p2p_bsr.values.size() * sizeof(double));
  }
  upload(plan.entries, entries.data(),
         entries.size() * sizeof(StaticOperatorEntry));
  upload(plan.m2m_matrices, data.m2m.matrices.data(),
         data.m2m.matrices.size() * sizeof(StaticOperatorEntry));
  upload(plan.l2l_matrices, data.l2l.matrices.data(),
         data.l2l.matrices.size() * sizeof(StaticOperatorEntry));
  upload(plan.m2m_interactions, data.m2m.interactions.data(),
         data.m2m.interactions.size() * sizeof(CudaTranslationInteraction));
  upload(plan.l2l_interactions, data.l2l.interactions.data(),
         data.l2l.interactions.size() * sizeof(CudaTranslationInteraction));
  if (data.has_fixed_self_indices) {
    plan.fixed_self_indices = data.fixed_self_indices;
    plan.identity_initialised = true;
    if (!plan.use_p2p_bsr && !data.fixed_self_indices.empty()) {
      upload(plan.self_indices, data.fixed_self_indices.data(),
             data.fixed_self_indices.size() * sizeof(int));
    }
  }
  plan.m2l = new CudaM2LExecutionPlan(data.m2l, plan.far_field_stream);
  check_cuda(cudaStreamSynchronize(plan.far_field_stream),
             "finish full FMM setup upload");
  plan.statistics.m2m_unique_matrix_count = data.m2m.matrix_count;
  plan.statistics.m2m_matrix_bytes =
      data.m2m.matrices.size() * sizeof(StaticOperatorEntry);
  const CudaPlanStatistics &m2l_statistics = plan.m2l->statistics();
  plan.statistics.m2l_unique_matrix_count =
      m2l_statistics.m2l_unique_matrix_count;
  plan.statistics.m2l_matrix_bytes = m2l_statistics.m2l_matrix_bytes;
  plan.statistics.m2l_interaction_metadata_bytes =
      m2l_statistics.m2l_interaction_metadata_bytes;
  plan.statistics.m2l_interaction_count =
      m2l_statistics.m2l_interaction_count;
  plan.statistics.m2l_active_row_count =
      m2l_statistics.m2l_active_row_count;
  plan.statistics.m2l_scratch_bytes = m2l_statistics.m2l_scratch_bytes;
  plan.statistics.m2l_threads_per_block =
      m2l_statistics.m2l_threads_per_block;
  plan.statistics.setup_h2d_bytes += m2l_statistics.setup_h2d_bytes;
  plan.statistics.l2l_unique_matrix_count = data.l2l.matrix_count;
  plan.statistics.l2l_matrix_bytes =
      data.l2l.matrices.size() * sizeof(StaticOperatorEntry);
  plan.statistics.persistent_device_bytes =
      plan.statistics.setup_h2d_bytes + 2 * source_bytes + 3 * target_bytes +
      2 * coefficient_bytes +
      (plan.use_p2p_bsr || data.has_fixed_self_indices
           ? 0
           : static_cast<std::size_t>(plan.target_count) * sizeof(int)) +
      plan.statistics.m2l_scratch_bytes;
  plan.statistics.plan_generation_count = 1;
  plan.statistics.static_upload_count = 1;
    plan.statistics.static_m2l_upload_count = 1;
    plan.statistics.static_p2p_upload_count = 1;
  plan.statistics.p2p_interaction_count = plan.p2p_block_count;
  if (plan.use_p2p_bsr) {
    plan.statistics.p2p_tensor_bytes =
        data.p2p_bsr.values.size() * sizeof(double);
    plan.statistics.p2p_index_bytes =
        data.p2p_bsr.source_indices.size() * sizeof(int);
    plan.statistics.p2p_row_metadata_bytes =
        data.p2p_bsr.row_offsets.size() * sizeof(int);
    plan.statistics.p2p_identity_bytes = 0;
    plan.statistics.p2p_threads_per_block = 0;
  } else {
    plan.statistics.p2p_tensor_bytes =
        data.p2p.blocks.size() * 6 * sizeof(double);
    plan.statistics.p2p_index_bytes =
        data.p2p.blocks.size() * 2 * sizeof(int);
    plan.statistics.p2p_row_metadata_bytes =
        data.p2p.row_offsets.size() * sizeof(int);
    plan.statistics.p2p_identity_bytes =
        static_cast<std::size_t>(plan.target_count) * sizeof(int);
    plan.statistics.p2p_threads_per_block = static_operator_threads;
  }
  plan.statistics.geometry_upload_count = 1;
}

CudaFullPlan::~CudaFullPlan() {
  if (implementation_ == nullptr) {
    return;
  }
    auto& plan = *implementation_;
    cudaFree(plan.source_permutation);
    cudaFree(plan.target_permutation);
    cudaFree(plan.self_indices);
  cudaFree(plan.p2p.row_offsets);
  cudaFree(plan.p2p.blocks);
  if (plan.p2p_bsr.descriptor != nullptr) {
    cusparseDestroyMatDescr(plan.p2p_bsr.descriptor);
  }
  if (plan.p2p_bsr.handle != nullptr) {
    cusparseDestroy(plan.p2p_bsr.handle);
  }
  cudaFree(plan.p2p_bsr.row_offsets);
  cudaFree(plan.p2p_bsr.source_indices);
  cudaFree(plan.p2p_bsr.values);
  cudaFree(plan.entries);
  cudaFree(plan.m2m_matrices);
  cudaFree(plan.l2l_matrices);
  cudaFree(plan.m2m_interactions);
  cudaFree(plan.l2l_interactions);
  delete plan.m2l;
  cudaFree(plan.moments);
  cudaFree(plan.sorted_moments);
  cudaFree(plan.multipoles);
    cudaFree(plan.locals);
    cudaFree(plan.far_fields);
    cudaFree(plan.near_fields);
    cudaFree(plan.final_fields);
    cudaFreeHost(plan.pinned_moments);
    cudaFreeHost(plan.pinned_fields);
    const std::array<cudaEvent_t, 13> events{
        plan.evaluation_start,
        plan.moments_ready,
        plan.p2m_complete,
        plan.m2m_complete,
        plan.m2l_scale_complete,
        plan.m2l_complete,
        plan.l2l_complete,
        plan.l2p_complete,
        plan.p2p_start,
        plan.p2p_complete,
        plan.combination_start,
        plan.combination_complete,
        plan.d2h_complete,
    };
    for (cudaEvent_t event : events) {
      cudaEventDestroy(event);
    }
    cudaStreamDestroy(plan.near_field_stream);
    cudaStreamDestroy(plan.far_field_stream);
    delete implementation_;
}

void CudaFullPlan::evaluate(const std::span<const Vec3> moments,
                            const std::span<Vec3> fields,
                            const std::span<const int> sorted_self_indices) {
  auto &plan = *implementation_;
  if (moments.size() != static_cast<std::size_t>(plan.source_count) ||
      fields.size() != static_cast<std::size_t>(plan.target_count) ||
      sorted_self_indices.size() !=
          static_cast<std::size_t>(plan.target_count)) {
    throw std::invalid_argument("full CUDA FMM dimensions are inconsistent");
  }
  if (!plan.identity_initialised) {
    plan.fixed_self_indices.assign(sorted_self_indices.begin(),
                                   sorted_self_indices.end());
    if (!sorted_self_indices.empty()) {
      check_cuda(cudaMemcpy(plan.self_indices, sorted_self_indices.data(),
                            sorted_self_indices.size_bytes(),
                            cudaMemcpyHostToDevice),
                 "upload static self identities");
      plan.statistics.setup_h2d_bytes += sorted_self_indices.size_bytes();
    }
        plan.identity_initialised = true;
  } else if (!std::equal(sorted_self_indices.begin(), sorted_self_indices.end(),
                         plan.fixed_self_indices.begin())) {
    throw std::invalid_argument(
        "CudaFull identity map changed; rebuild the static plan");
  }
  std::copy(moments.begin(), moments.end(), plan.pinned_moments);
  constexpr int threads = 256;
  const auto launch_stage = [&](const int stage, const double *input,
                                double *output) {
    const std::size_t count = plan.counts[static_cast<std::size_t>(stage)];
    if (count != 0) {
      apply_entries_kernel<<<(count + threads - 1) / threads, threads, 0,
                             plan.far_field_stream>>>(
          plan.entries + plan.offsets[stage], count, input, output);
    }
  };
  check_cuda(cudaEventRecord(plan.evaluation_start, plan.far_field_stream),
             "record full FMM start");
  {
    detail::ProfileRange transfer_range{"cdfmm/input_preparation/moments_h2d"};
    if (!moments.empty()) {
      check_cuda(cudaMemcpyAsync(plan.moments, plan.pinned_moments,
                                 moments.size_bytes(), cudaMemcpyHostToDevice,
                                 plan.far_field_stream),
                 "upload full FMM moments");
      detail::ProfileRange permutation_range{
          "cdfmm/input_preparation/moment_permutation"};
      permute_moments_kernel<<<(plan.source_count + threads - 1) / threads,
                               threads, 0, plan.far_field_stream>>>(
          plan.moments, plan.source_permutation, plan.source_count,
          plan.sorted_moments);
    }
  }
  check_cuda(cudaEventRecord(plan.moments_ready, plan.far_field_stream),
             "record full FMM moments ready");

  // P2P and the far-field hierarchy both consume the immutable sorted moments.
  // Once permutation is complete they have no data dependency until final
  // field combination, so retain them on independent non-blocking streams.
  check_cuda(cudaStreamWaitEvent(plan.near_field_stream, plan.moments_ready, 0),
             "wait for full FMM moments on near-field stream");
  check_cuda(cudaEventRecord(plan.p2p_start, plan.near_field_stream),
             "record full FMM P2P start");
  {
    detail::ProfileRange p2p_range{"cdfmm/near_field/p2p"};
    if (plan.use_p2p_bsr) {
      launch_bsr_p2p(plan.p2p_bsr, plan.sorted_moments, plan.near_fields,
                     plan.near_field_stream);
    } else {
      launch_static_p2p(plan.p2p, plan.sorted_moments, plan.self_indices,
                        plan.near_fields, plan.near_field_stream);
    }
  }
  check_cuda(cudaEventRecord(plan.p2p_complete, plan.near_field_stream),
             "record full FMM P2P completion");

  {
    detail::ProfileRange p2m_range{"cdfmm/far_field/p2m"};
    check_cuda(cudaMemsetAsync(plan.multipoles, 0,
                               static_cast<std::size_t>(plan.node_count) *
                                   plan.coefficient_count * sizeof(double),
                               plan.far_field_stream),
               "clear full FMM multipoles");
    launch_stage(plan.p2m_stage,
                 reinterpret_cast<double *>(plan.sorted_moments),
                 plan.multipoles);
  }
  check_cuda(cudaEventRecord(plan.p2m_complete, plan.far_field_stream),
             "record P2M");
  // Kernels for one level can update parent coefficients concurrently, but a
  // parent level must not consume them early. Launching levels into one stream
  // supplies the required child-to-parent ordering without a host barrier.
  detail::ProfileRange m2m_range{"cdfmm/far_field/m2m"};
  for (int level = plan.leaf_level; level >= 1; --level) {
    const std::size_t items =
        plan.m2m_interaction_count * plan.m2m_entries_per_matrix;
    if (items != 0) {
      apply_shared_translation_kernel<<<(items + threads - 1) / threads,
                                        threads, 0, plan.far_field_stream>>>(
          plan.m2m_matrices, plan.m2m_interactions, plan.m2m_interaction_count,
          plan.m2m_entries_per_matrix, plan.coefficient_count, level,
          plan.multipoles, plan.multipoles);
    }
  }
  check_cuda(cudaEventRecord(plan.m2m_complete, plan.far_field_stream),
             "record M2M");
  m2m_range.end();
  detail::ProfileRange m2l_range{"cdfmm/far_field/m2l"};
  check_cuda(cudaMemsetAsync(plan.locals, 0,
                             static_cast<std::size_t>(plan.node_count) *
                                 plan.coefficient_count * sizeof(double),
                             plan.far_field_stream),
             "clear full FMM locals");
  plan.m2l->enqueue(plan.multipoles, plan.locals, plan.far_field_stream,
                    plan.m2l_scale_complete);
  check_cuda(cudaEventRecord(plan.m2l_complete, plan.far_field_stream),
             "record M2L");
  m2l_range.end();
  detail::ProfileRange l2l_range{"cdfmm/far_field/l2l"};
  // The downward dependency is the reverse: each parent local must be complete
  // before the next level translates it to children. Stream order enforces it.
  for (int level = 1; level <= plan.leaf_level; ++level) {
    const std::size_t items =
        plan.l2l_interaction_count * plan.l2l_entries_per_matrix;
    if (items != 0) {
      apply_shared_translation_kernel<<<(items + threads - 1) / threads,
                                        threads, 0, plan.far_field_stream>>>(
          plan.l2l_matrices, plan.l2l_interactions, plan.l2l_interaction_count,
          plan.l2l_entries_per_matrix, plan.coefficient_count, level,
          plan.locals, plan.locals);
    }
  }
  check_cuda(cudaEventRecord(plan.l2l_complete, plan.far_field_stream),
             "record L2L");
  l2l_range.end();
  detail::ProfileRange l2p_range{"cdfmm/far_field/l2p"};
  check_cuda(cudaMemsetAsync(plan.far_fields, 0,
                             static_cast<std::size_t>(plan.target_count) *
                                 sizeof(Vec3),
                             plan.far_field_stream),
             "clear far fields");
  launch_stage(plan.l2p_stage, plan.locals,
               reinterpret_cast<double *>(plan.far_fields));
  check_cuda(cudaEventRecord(plan.l2p_complete, plan.far_field_stream),
             "record L2P");
  l2p_range.end();

  // Final combination is the first operation that consumes both branches.
  // A device-side event wait avoids an earlier host synchronisation and keeps
  // all available near/far overlap.
  check_cuda(cudaStreamWaitEvent(plan.far_field_stream, plan.p2p_complete, 0),
             "wait for full FMM P2P before combination");
  check_cuda(cudaEventRecord(plan.combination_start, plan.far_field_stream),
             "record full FMM combination start");
  detail::ProfileRange combine_range{"cdfmm/combine"};
  if (plan.target_count != 0) {
    // Combining and unsorting on-device keeps intermediate far/near fields
    // private to the plan; repeated field evaluations download only user-order H.
    combine_order_kernel<<<(plan.target_count + threads - 1) / threads, threads,
                           0, plan.far_field_stream>>>(
        plan.far_fields, plan.near_fields, plan.target_permutation,
        plan.target_count, plan.final_fields);
  }
  check_cuda(cudaEventRecord(plan.combination_complete, plan.far_field_stream),
             "record accumulation");
  combine_range.end();
  {
    detail::ProfileRange transfer_range{"cdfmm/output/final_field_d2h"};
    if (!fields.empty()) {
      check_cuda(cudaMemcpyAsync(plan.pinned_fields, plan.final_fields,
                                 fields.size_bytes(), cudaMemcpyDeviceToHost,
                                 plan.far_field_stream),
                 "download full FMM fields");
    }
  }
  check_cuda(cudaEventRecord(plan.d2h_complete, plan.far_field_stream),
             "record field download");
  check_cuda(cudaEventSynchronize(plan.d2h_complete),
             "wait for full FMM evaluation");
  std::copy(plan.pinned_fields, plan.pinned_fields + fields.size(),
            fields.begin());
  const auto elapsed = [](const cudaEvent_t first, const cudaEvent_t second) {
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, first, second),
               "time full FMM phase");
    return static_cast<double>(milliseconds) * 1.0e-3;
  };
  plan.timings = {};
  plan.timings.h2d_seconds =
      elapsed(plan.evaluation_start, plan.moments_ready);
  plan.timings.p2m_seconds =
      elapsed(plan.moments_ready, plan.p2m_complete);
  plan.timings.m2m_seconds = elapsed(plan.p2m_complete, plan.m2m_complete);
  plan.timings.m2l_seconds = elapsed(plan.m2m_complete, plan.m2l_complete);
  plan.timings.scale_seconds =
      elapsed(plan.m2m_complete, plan.m2l_scale_complete);
  plan.timings.multiply_seconds =
      elapsed(plan.m2l_scale_complete, plan.m2l_complete);
  plan.timings.l2l_seconds = elapsed(plan.m2l_complete, plan.l2l_complete);
  plan.timings.l2p_seconds = elapsed(plan.l2l_complete, plan.l2p_complete);
  plan.timings.p2p_seconds = elapsed(plan.p2p_start, plan.p2p_complete);
  plan.timings.accumulation_seconds =
      elapsed(plan.combination_start, plan.combination_complete);
  plan.timings.d2h_seconds =
      elapsed(plan.combination_complete, plan.d2h_complete);
  plan.timings.kernel_seconds =
      plan.timings.p2m_seconds + plan.timings.m2m_seconds +
      plan.timings.m2l_seconds + plan.timings.l2l_seconds +
      plan.timings.l2p_seconds + plan.timings.p2p_seconds +
      plan.timings.accumulation_seconds;
  plan.timings.total_seconds =
      elapsed(plan.evaluation_start, plan.d2h_complete);
  plan.statistics.evaluation_h2d_bytes = moments.size_bytes();
  plan.statistics.evaluation_d2h_bytes = fields.size_bytes();
    ++plan.statistics.evaluation_h2d_calls;
  ++plan.statistics.evaluation_d2h_calls;
}

const CudaPlanStatistics &CudaFullPlan::statistics() const noexcept {
  return implementation_->statistics;
}

const CudaEvaluationTimings &CudaFullPlan::timings() const noexcept {
  return implementation_->timings;
}

} // namespace cdfmm
