// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/backend/cuda/direct.hpp"

#include "../common/diagnostic_event.hpp"
#include "../common/error.hpp"
#include "../common/runtime.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace cdfmm {
namespace {

using cuda_detail::check_cuda;

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

bool cuda_direct_available() noexcept {
  return cuda_runtime_available();
}

struct CudaDirectPlan::Implementation {
  std::size_t source_count{0};
  std::size_t target_count{0};
  Vec3 *device_sources{nullptr};
  Vec3 *device_targets{nullptr};
  Vec3 *device_moments{nullptr};
  Vec3 *device_fields{nullptr};
  double *device_potentials{nullptr};
  int *device_self_indices{nullptr};
  Vec3 *pinned_moments{nullptr};
  Vec3 *pinned_fields{nullptr};
  double *pinned_potentials{nullptr};
  std::vector<int> self_indices{};
  cudaStream_t stream{nullptr};
  // `d2h_complete` is the functional completion point
  // (cudaEventDisableTiming); the others are diagnostic and recorded only at
  // TimingLevel::Detailed, with `evaluation_end` the twin of `d2h_complete`.
  cudaEvent_t d2h_complete{nullptr};
  cudaEvent_t evaluation_start{nullptr};
  cudaEvent_t h2d_complete{nullptr};
  cudaEvent_t kernel_complete{nullptr};
  cudaEvent_t evaluation_end{nullptr};
  TimingLevel timing_level{TimingLevel::Off};
  CudaPlanStatistics statistics{};
  CudaEvaluationTimings evaluation_timings{};
};

CudaDirectPlan::CudaDirectPlan(
    const std::span<const Vec3> source_positions,
    const std::span<const Vec3> target_positions,
    const std::span<const int> target_source_indices)
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
    std::copy(target_source_indices.begin(), target_source_indices.end(),
              plan.self_indices.begin());
  }

  check_cuda(cudaStreamCreateWithFlags(&plan.stream, cudaStreamNonBlocking),
             "cudaStreamCreateWithFlags");
  check_cuda(cudaEventCreateWithFlags(&plan.d2h_complete,
                                      cudaEventDisableTiming),
             "cudaEventCreate D2H complete");
  for (cudaEvent_t *event : {&plan.evaluation_start, &plan.h2d_complete,
                             &plan.kernel_complete, &plan.evaluation_end}) {
    check_cuda(cudaEventCreate(event), "cudaEventCreate timing event");
  }

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
  check_cuda(cudaMalloc(
                 &plan.device_potentials,
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
  auto &plan = *implementation_;
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
  cudaEventDestroy(plan.evaluation_end);
  cudaEventDestroy(plan.d2h_complete);
  cudaStreamDestroy(plan.stream);
  delete implementation_;
}

void CudaDirectPlan::evaluate(
    const std::span<const Vec3> moments,
    const std::span<PotentialField> results,
    const OutputFlags output) {
  auto &plan = *implementation_;
  if (moments.size() != plan.source_count ||
      results.size() != plan.target_count) {
    throw std::invalid_argument("CUDA evaluation array size mismatch");
  }
  std::copy(moments.begin(), moments.end(), plan.pinned_moments);
  const std::size_t moment_bytes = plan.source_count * sizeof(Vec3);
  const bool detailed = plan.timing_level == TimingLevel::Detailed;
  cuda_detail::record_diagnostic(detailed, plan.evaluation_start, plan.stream,
                                 "record CUDA evaluation start");
  check_cuda(cudaMemcpyAsync(plan.device_moments, plan.pinned_moments,
                             moment_bytes, cudaMemcpyHostToDevice, plan.stream),
             "upload dipole moments");
  plan.statistics.evaluation_h2d_bytes = moment_bytes;
  plan.statistics.evaluation_h2d_calls = 1;
  cuda_detail::record_diagnostic(detailed, plan.h2d_complete, plan.stream,
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
  cuda_detail::record_diagnostic(detailed, plan.kernel_complete, plan.stream,
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
  cuda_detail::record_diagnostic(detailed, plan.evaluation_end, plan.stream,
                                 "record CUDA evaluation end");
  check_cuda(cudaEventSynchronize(plan.d2h_complete),
             "complete CUDA evaluation");

  if (detailed) {
    CudaEvaluationTimings &timings = plan.evaluation_timings;
    timings = {};
    timings.timing_level = TimingLevel::Detailed;
    timings.h2d_seconds = cuda_detail::diagnostic_elapsed_seconds(
        plan.evaluation_start, plan.h2d_complete, "measure CUDA H2D time");
    timings.kernel_seconds = cuda_detail::diagnostic_elapsed_seconds(
        plan.h2d_complete, plan.kernel_complete, "measure CUDA kernel time");
    timings.d2h_seconds = cuda_detail::diagnostic_elapsed_seconds(
        plan.kernel_complete, plan.evaluation_end, "measure CUDA D2H time");
    timings.total_seconds =
        timings.h2d_seconds + timings.kernel_seconds + timings.d2h_seconds;
  }
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

TimingLevel CudaDirectPlan::timing_level() const noexcept {
  return implementation_->timing_level;
}

void CudaDirectPlan::set_timing_level(const TimingLevel level) noexcept {
  implementation_->timing_level = level;
  implementation_->evaluation_timings = {};
  implementation_->evaluation_timings.timing_level = level;
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

} // namespace cdfmm
