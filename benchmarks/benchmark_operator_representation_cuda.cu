// SPDX-License-Identifier: Apache-2.0

#include "benchmark_operator_representation_cuda.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "geometry/primitives/rectangular_prism_point_kernel.hpp"

namespace cdfmm_bench {

namespace {

void check(const cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

struct DevicePrism {
  double hx;
  double hy;
  double hz;
};

constexpr int threads_per_block = 128;

// One thread per work item (one target, one dense source range).  Every
// pair tensor is reconstructed in `Real` from the retained positions and the
// prism record of the pair's prism body, applied to the source moment and
// the partial field is added atomically to the target, because the other
// source ranges of the same target run in other threads.  The kernel is
// transcendental-bound (about 24 atan and 24 log per pair), so the simple
// one-item-per-thread mapping suffices to saturate the device.
template <typename Real, typename Vector>
__global__ void __launch_bounds__(threads_per_block) procedural_prism_kernel(
    const ProceduralWorkItem* __restrict__ items, const int item_count,
    const double* __restrict__ source_positions,
    const double* __restrict__ target_positions,
    const DevicePrism* __restrict__ prisms, const int common_prism,
    const int prism_is_source, const int skip_self,
    const Vector* __restrict__ moments, Vector* __restrict__ fields) {
  const int item_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (item_index >= item_count) {
    return;
  }
  const ProceduralWorkItem item = items[item_index];
  const int target = item.target;
  const double tx = target_positions[3 * target];
  const double ty = target_positions[3 * target + 1];
  const double tz = target_positions[3 * target + 2];
  const DevicePrism target_prism = prisms[common_prism ? 0 : target];
  Real hx = Real{0};
  Real hy = Real{0};
  Real hz = Real{0};
  for (int local = 0; local < item.source_count; ++local) {
    const int source = item.source_begin + local;
    if (skip_self != 0 && source == target) {
      // The identity map names this body as its own source: a point source's
      // coincident pair is the singular one the canonical operator omits.
      continue;
    }
    const DevicePrism prism =
        prism_is_source ? prisms[common_prism ? 0 : source] : target_prism;
    // The displacement is formed in FP64 like the production builder; the
    // reconstruction precision is the template parameter.
    const Real rx = static_cast<Real>(tx - source_positions[3 * source]);
    const Real ry = static_cast<Real>(ty - source_positions[3 * source + 1]);
    const Real rz = static_cast<Real>(tz - source_positions[3 * source + 2]);
    Real tensor[6];
    cdfmm::detail::prism_kernel::rectangular_prism_point_tensor_components<Real>(
        static_cast<Real>(prism.hx), static_cast<Real>(prism.hy),
        static_cast<Real>(prism.hz), rx, ry, rz, tensor);
    const Vector moment = moments[source];
    const Real mx = static_cast<Real>(moment.x);
    const Real my = static_cast<Real>(moment.y);
    const Real mz = static_cast<Real>(moment.z);
    hx += tensor[0] * mx + tensor[1] * my + tensor[2] * mz;
    hy += tensor[1] * mx + tensor[3] * my + tensor[4] * mz;
    hz += tensor[2] * mx + tensor[4] * my + tensor[5] * mz;
  }
  using Component = decltype(Vector{}.x);
  atomicAdd(&fields[target].x, static_cast<Component>(hx));
  atomicAdd(&fields[target].y, static_cast<Component>(hy));
  atomicAdd(&fields[target].z, static_cast<Component>(hz));
}

template <typename T>
T* upload(const std::vector<T>& values) {
  T* device = nullptr;
  check(cudaMalloc(&device, values.size() * sizeof(T)), "cudaMalloc");
  check(cudaMemcpy(device, values.data(), values.size() * sizeof(T),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy H2D");
  return device;
}

} // namespace

struct ProceduralPrismCudaPlan::Implementation {
  ProceduralWorkItem* items{nullptr};
  int item_count{0};
  double* source_positions{nullptr};
  double* target_positions{nullptr};
  DevicePrism* prisms{nullptr};
  int common_prism{1};
  int prism_is_source{1};
  int skip_self{0};
  int source_count{0};
  int target_count{0};
  std::size_t device_bytes{0};
  void* moments{nullptr};
  void* fields{nullptr};
  std::size_t vector_capacity_bytes{0};
  cudaEvent_t start{};
  cudaEvent_t stop{};

  template <typename Real, typename Vector>
  double run(const std::span<const Vector> moments_host,
             const std::span<Vector> fields_host) {
    const std::size_t moment_bytes = moments_host.size() * sizeof(Vector);
    const std::size_t field_bytes = fields_host.size() * sizeof(Vector);
    if (moment_bytes > vector_capacity_bytes || field_bytes > vector_capacity_bytes) {
      if (moments != nullptr) {
        cudaFree(moments);
        cudaFree(fields);
      }
      vector_capacity_bytes = moment_bytes > field_bytes ? moment_bytes : field_bytes;
      check(cudaMalloc(&moments, vector_capacity_bytes), "cudaMalloc moments");
      check(cudaMalloc(&fields, vector_capacity_bytes), "cudaMalloc fields");
    }
    check(cudaMemcpy(moments, moments_host.data(), moment_bytes,
                     cudaMemcpyHostToDevice),
          "upload moments");
    check(cudaMemset(fields, 0, field_bytes), "zero fields");
    check(cudaEventRecord(start), "record start");
    const int blocks = (item_count + threads_per_block - 1) / threads_per_block;
    procedural_prism_kernel<Real, Vector><<<blocks, threads_per_block>>>(
        items, item_count, source_positions, target_positions, prisms,
        common_prism, prism_is_source, skip_self,
        static_cast<const Vector*>(moments), static_cast<Vector*>(fields));
    check(cudaGetLastError(), "procedural prism kernel launch");
    check(cudaEventRecord(stop), "record stop");
    check(cudaMemcpy(fields_host.data(), fields, field_bytes,
                     cudaMemcpyDeviceToHost),
          "download fields");
    check(cudaEventSynchronize(stop), "synchronise");
    float milliseconds = 0.0F;
    check(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed");
    return static_cast<double>(milliseconds) * 1.0e-3;
  }
};

ProceduralPrismCudaPlan::ProceduralPrismCudaPlan(const ProceduralPrismScene& scene)
    : implementation_(std::make_unique<Implementation>()) {
  Implementation& plan = *implementation_;
  plan.item_count = static_cast<int>(scene.items.size());
  plan.items = upload(scene.items);
  std::vector<double> flat;
  flat.reserve(scene.source_positions.size() * 3);
  for (const cdfmm::Vec3& position : scene.source_positions) {
    flat.push_back(position.x);
    flat.push_back(position.y);
    flat.push_back(position.z);
  }
  plan.source_positions = upload(flat);
  plan.source_count = static_cast<int>(scene.source_positions.size());
  flat.clear();
  for (const cdfmm::Vec3& position : scene.target_positions) {
    flat.push_back(position.x);
    flat.push_back(position.y);
    flat.push_back(position.z);
  }
  plan.target_positions = upload(flat);
  plan.target_count = static_cast<int>(scene.target_positions.size());
  std::vector<DevicePrism> prisms;
  for (const cdfmm::RectangularPrism& prism : scene.prisms) {
    prisms.push_back({prism.hx, prism.hy, prism.hz});
  }
  plan.prisms = upload(prisms);
  plan.common_prism = prisms.size() == 1 ? 1 : 0;
  plan.prism_is_source = scene.prism_is_source ? 1 : 0;
  plan.skip_self = scene.skip_point_self_pair ? 1 : 0;
  plan.device_bytes = scene.items.size() * sizeof(ProceduralWorkItem) +
                      (scene.source_positions.size() + scene.target_positions.size()) *
                          3 * sizeof(double) +
                      prisms.size() * sizeof(DevicePrism);
  check(cudaEventCreate(&plan.start), "create event");
  check(cudaEventCreate(&plan.stop), "create event");
}

ProceduralPrismCudaPlan::~ProceduralPrismCudaPlan() {
  Implementation& plan = *implementation_;
  cudaFree(plan.items);
  cudaFree(plan.source_positions);
  cudaFree(plan.target_positions);
  cudaFree(plan.prisms);
  if (plan.moments != nullptr) {
    cudaFree(plan.moments);
    cudaFree(plan.fields);
  }
  cudaEventDestroy(plan.start);
  cudaEventDestroy(plan.stop);
}

double ProceduralPrismCudaPlan::evaluate(
    const std::span<const cdfmm::FloatVec3> moments,
    const std::span<cdfmm::FloatVec3> fields, const bool double_math) {
  if (double_math) {
    return implementation_->run<double, cdfmm::FloatVec3>(moments, fields);
  }
  return implementation_->run<float, cdfmm::FloatVec3>(moments, fields);
}

double ProceduralPrismCudaPlan::evaluate(const std::span<const cdfmm::Vec3> moments,
                                         const std::span<cdfmm::Vec3> fields) {
  return implementation_->run<double, cdfmm::Vec3>(moments, fields);
}

std::size_t ProceduralPrismCudaPlan::device_bytes() const noexcept {
  return implementation_->device_bytes;
}

std::size_t ProceduralPrismCudaPlan::item_count() const noexcept {
  return static_cast<std::size_t>(implementation_->item_count);
}

} // namespace cdfmm_bench
