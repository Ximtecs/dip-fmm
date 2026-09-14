// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/backend/cuda/dense_direct.hpp"

#include "../common/error.hpp"
#include "../common/runtime.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cdfmm {
namespace {

using cuda_detail::check_cuda;

void check_cublas(const cublasStatus_t status, const char *operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string(operation) + ": cuBLAS status " +
        std::to_string(static_cast<int>(status)));
  }
}

} // namespace

struct CudaDenseDirectPlan::Implementation {
  std::size_t source_count{0};
  std::size_t target_count{0};
  std::size_t matrix_value_count{0};
  StaticPrecision static_precision{StaticPrecision::Float64};
  float *device_matrices_f32{nullptr};
  float *device_moments_f32{nullptr};
  float *device_fields_f32{nullptr};
  float *pinned_moments_f32{nullptr};
  float *pinned_fields_f32{nullptr};
  double *device_matrices{nullptr};
  double *device_moments{nullptr};
  double *device_fields{nullptr};
  double *pinned_moments{nullptr};
  double *pinned_fields{nullptr};
  cudaStream_t stream{nullptr};
  cublasHandle_t cublas_handle{nullptr};

  ~Implementation() {
    cudaFree(device_matrices_f32);
    cudaFree(device_moments_f32);
    cudaFree(device_fields_f32);
    cudaFreeHost(pinned_moments_f32);
    cudaFreeHost(pinned_fields_f32);
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

bool cuda_dense_direct_available() noexcept {
  return cuda_runtime_available();
}

CudaDenseDirectPlan::CudaDenseDirectPlan(
    const std::span<const Vec3> source_positions,
    const std::span<const Vec3> target_positions,
    const SourceGeometry source_geometry,
    const TargetGeometry target_geometry,
    const std::span<const CuboidSize> source_sizes,
    const std::span<const CuboidSize> target_sizes,
    const std::span<const int> target_source_indices,
    const StaticPrecision static_precision,
    const std::span<const Tetrahedron> source_tetrahedra,
    const std::span<const Tetrahedron> target_tetrahedra,
    const SourceModel source_model,
    const TargetModel target_model)
    : implementation_(new Implementation{}) {
  try {
    if (!cuda_runtime_available()) {
      throw std::runtime_error(
          "CUDA dense direct backend requested, but no CUDA-capable device "
          "is available");
    }

    auto &plan = *implementation_;
    plan.source_count = source_positions.size();
    plan.target_count = target_positions.size();
    plan.static_precision = static_precision;
    if (plan.source_count > static_cast<std::size_t>(
                                std::numeric_limits<int>::max()) ||
        plan.target_count > static_cast<std::size_t>(
                                std::numeric_limits<int>::max())) {
      throw std::overflow_error(
          "CUDA dense direct dimensions exceed the cuBLAS integer limit");
    }
    if (plan.source_count != 0 &&
        plan.target_count >
            std::numeric_limits<std::size_t>::max() / plan.source_count) {
      throw std::overflow_error("CUDA dense direct tensor size overflow");
    }
    plan.matrix_value_count = plan.source_count * plan.target_count;
    const std::size_t scalar_bytes =
        static_precision == StaticPrecision::Float32 ? sizeof(float)
                                                       : sizeof(double);
    if (plan.matrix_value_count >
        std::numeric_limits<std::size_t>::max() / (6 * scalar_bytes)) {
      throw std::overflow_error("CUDA dense direct tensor byte-size overflow");
    }

    // Build the exact six-component geometry tensor using the common CPU
    // implementation, then retain only its device copy after construction.
    const DenseDirectPlan host_plan(
        source_positions, target_positions, source_geometry, target_geometry,
        source_sizes, target_sizes, target_source_indices, static_precision,
        source_tetrahedra, target_tetrahedra, source_model, target_model);

    check_cuda(cudaStreamCreateWithFlags(&plan.stream, cudaStreamNonBlocking),
               "create CUDA dense direct stream");
    check_cublas(cublasCreate(&plan.cublas_handle),
                 "create CUDA dense direct cuBLAS handle");
    check_cublas(cublasSetStream(plan.cublas_handle, plan.stream),
                 "set CUDA dense direct cuBLAS stream");

    const std::size_t tensor_bytes = tensor_memory_bytes();
    const std::size_t moment_bytes = 3 * plan.source_count * scalar_bytes;
    const std::size_t field_bytes = 3 * plan.target_count * scalar_bytes;
    if (static_precision == StaticPrecision::Float32) {
      check_cuda(cudaMalloc(&plan.device_matrices_f32,
                            std::max(tensor_bytes, sizeof(float))),
                 "allocate FP32 CUDA dense direct tensors");
      check_cuda(cudaMalloc(&plan.device_moments_f32,
                            std::max(moment_bytes, sizeof(float))),
                 "allocate FP32 CUDA dense direct moments");
      check_cuda(cudaMalloc(&plan.device_fields_f32,
                            std::max(field_bytes, sizeof(float))),
                 "allocate FP32 CUDA dense direct fields");
      check_cuda(cudaMallocHost(&plan.pinned_moments_f32,
                                std::max(moment_bytes, sizeof(float))),
                 "allocate pinned FP32 CUDA dense direct moments");
      check_cuda(cudaMallocHost(&plan.pinned_fields_f32,
                                std::max(field_bytes, sizeof(float))),
                 "allocate pinned FP32 CUDA dense direct fields");
      const std::size_t matrix_bytes = plan.matrix_value_count * sizeof(float);
      for (std::size_t component = 0; component < 6; ++component) {
        check_cuda(cudaMemcpyAsync(
                       plan.device_matrices_f32 +
                           component * plan.matrix_value_count,
                       host_plan.float_matrices()[component].data(),
                       matrix_bytes, cudaMemcpyHostToDevice, plan.stream),
                   "upload FP32 CUDA dense direct tensor component");
      }
    } else {
      check_cuda(cudaMalloc(&plan.device_matrices,
                            std::max(tensor_bytes, sizeof(double))),
                 "allocate FP64 CUDA dense direct tensors");
      check_cuda(cudaMalloc(&plan.device_moments,
                            std::max(moment_bytes, sizeof(double))),
                 "allocate FP64 CUDA dense direct moments");
      check_cuda(cudaMalloc(&plan.device_fields,
                            std::max(field_bytes, sizeof(double))),
                 "allocate FP64 CUDA dense direct fields");
      check_cuda(cudaMallocHost(&plan.pinned_moments,
                                std::max(moment_bytes, sizeof(double))),
                 "allocate pinned FP64 CUDA dense direct moments");
      check_cuda(cudaMallocHost(&plan.pinned_fields,
                                std::max(field_bytes, sizeof(double))),
                 "allocate pinned FP64 CUDA dense direct fields");
      const std::size_t matrix_bytes = plan.matrix_value_count * sizeof(double);
      for (std::size_t component = 0; component < 6; ++component) {
        check_cuda(cudaMemcpyAsync(
                       plan.device_matrices + component * plan.matrix_value_count,
                       host_plan.matrices()[component].data(), matrix_bytes,
                       cudaMemcpyHostToDevice, plan.stream),
                   "upload FP64 CUDA dense direct tensor component");
      }
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
    const std::span<const Vec3> total_moments) {
  auto &plan = *implementation_;
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

  const int source_count = static_cast<int>(plan.source_count);
  const int target_count = static_cast<int>(plan.target_count);
  std::vector<Vec3> result(plan.target_count);
  if (plan.static_precision == StaticPrecision::Float32) {
    for (std::size_t source = 0; source < plan.source_count; ++source) {
      plan.pinned_moments_f32[source] =
          static_cast<float>(total_moments[source].x);
      plan.pinned_moments_f32[plan.source_count + source] =
          static_cast<float>(total_moments[source].y);
      plan.pinned_moments_f32[2 * plan.source_count + source] =
          static_cast<float>(total_moments[source].z);
    }
    const std::size_t moment_bytes = 3 * plan.source_count * sizeof(float);
    check_cuda(cudaMemcpyAsync(plan.device_moments_f32,
                               plan.pinned_moments_f32, moment_bytes,
                               cudaMemcpyHostToDevice, plan.stream),
               "upload FP32 CUDA dense direct moments");

    const float alpha = 1.0F;
    const float zero = 0.0F;
    const float one = 1.0F;
    const auto apply_component = [&](const std::size_t matrix_component,
                                     const std::size_t moment_component,
                                     const std::size_t field_component,
                                     const bool add) {
      check_cublas(
          cublasSgemv(plan.cublas_handle, CUBLAS_OP_T, source_count,
                      target_count, &alpha,
                      plan.device_matrices_f32 +
                          matrix_component * plan.matrix_value_count,
                      source_count,
                      plan.device_moments_f32 + moment_component * plan.source_count,
                      1, add ? &one : &zero,
                      plan.device_fields_f32 + field_component * plan.target_count,
                      1),
          "apply FP32 CUDA dense direct tensor component");
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

    const std::size_t field_bytes = 3 * plan.target_count * sizeof(float);
    check_cuda(cudaMemcpyAsync(plan.pinned_fields_f32, plan.device_fields_f32,
                               field_bytes, cudaMemcpyDeviceToHost, plan.stream),
               "download FP32 CUDA dense direct fields");
    check_cuda(cudaStreamSynchronize(plan.stream),
               "complete FP32 CUDA dense direct evaluation");
    for (std::size_t target = 0; target < plan.target_count; ++target) {
      result[target] = {
          static_cast<double>(plan.pinned_fields_f32[target]),
          static_cast<double>(plan.pinned_fields_f32[plan.target_count + target]),
          static_cast<double>(plan.pinned_fields_f32[2 * plan.target_count + target])};
    }
  } else {
    for (std::size_t source = 0; source < plan.source_count; ++source) {
      plan.pinned_moments[source] = total_moments[source].x;
      plan.pinned_moments[plan.source_count + source] =
          total_moments[source].y;
      plan.pinned_moments[2 * plan.source_count + source] =
          total_moments[source].z;
    }
    const std::size_t moment_bytes = 3 * plan.source_count * sizeof(double);
    check_cuda(cudaMemcpyAsync(plan.device_moments, plan.pinned_moments,
                               moment_bytes, cudaMemcpyHostToDevice, plan.stream),
               "upload FP64 CUDA dense direct moments");

    const double alpha = 1.0;
    const double zero = 0.0;
    const double one = 1.0;
    const auto apply_component = [&](const std::size_t matrix_component,
                                     const std::size_t moment_component,
                                     const std::size_t field_component,
                                     const bool add) {
      check_cublas(
          cublasDgemv(plan.cublas_handle, CUBLAS_OP_T, source_count,
                      target_count, &alpha,
                      plan.device_matrices +
                          matrix_component * plan.matrix_value_count,
                      source_count,
                      plan.device_moments + moment_component * plan.source_count,
                      1, add ? &one : &zero,
                      plan.device_fields + field_component * plan.target_count,
                      1),
          "apply FP64 CUDA dense direct tensor component");
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
               "download FP64 CUDA dense direct fields");
    check_cuda(cudaStreamSynchronize(plan.stream),
               "complete FP64 CUDA dense direct evaluation");
    for (std::size_t target = 0; target < plan.target_count; ++target) {
      result[target] = {
          plan.pinned_fields[target],
          plan.pinned_fields[plan.target_count + target],
          plan.pinned_fields[2 * plan.target_count + target]};
    }
  }
  return result;
}

std::size_t CudaDenseDirectPlan::source_count() const noexcept {
  return implementation_->source_count;
}

std::size_t CudaDenseDirectPlan::target_count() const noexcept {
  return implementation_->target_count;
}

std::size_t CudaDenseDirectPlan::tensor_memory_bytes() const noexcept {
  const std::size_t scalar_bytes =
      implementation_->static_precision == StaticPrecision::Float32
          ? sizeof(float)
          : sizeof(double);
  return 6 * implementation_->matrix_value_count * scalar_bytes;
}

std::size_t CudaDenseDirectPlan::persistent_device_bytes() const noexcept {
  return tensor_memory_bytes() +
         3 * (implementation_->source_count + implementation_->target_count) *
             (implementation_->static_precision == StaticPrecision::Float32
                  ? sizeof(float)
                  : sizeof(double));
}

StaticPrecision CudaDenseDirectPlan::static_precision() const noexcept {
  return implementation_->static_precision;
}

} // namespace cdfmm
