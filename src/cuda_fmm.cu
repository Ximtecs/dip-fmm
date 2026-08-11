// SPDX-License-Identifier: Apache-2.0

#include "cuda_fmm_plan.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <numbers>
#include <stdexcept>
#include <string>

namespace cdfmm {
namespace {

void check_cuda(const cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

__global__ void dipole_field_kernel(const Vec3* targets,
                                    const Vec3* sources,
                                    const Vec3* moments,
                                    const int* self_indices,
                                    const std::size_t source_count,
                                    const std::size_t target_count,
                                    Vec3* fields,
                                    double* potentials,
                                    const bool compute_field,
                                    const bool compute_potential)
{
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
        const double moment_dot_r =
            moment.x * rx + moment.y * ry + moment.z * rz;

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

struct CudaFmmPlan::Implementation {
    ExecutionBackend backend{ExecutionBackend::CudaFull};
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
    CudaPlanStatistics statistics{};
};

bool cuda_runtime_available() noexcept
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string cuda_runtime_description()
{
    int device = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties");
    return std::string(properties.name) + " (compute capability " +
        std::to_string(properties.major) + "." +
        std::to_string(properties.minor) + ")";
}

CudaFmmPlan::CudaFmmPlan(const std::span<const Vec3> source_positions,
                         const std::span<const Vec3> target_positions,
                         const ExecutionBackend backend)
    : implementation_(new Implementation{})
{
    if (!cuda_runtime_available()) {
        delete implementation_;
        implementation_ = nullptr;
        throw std::runtime_error(
            "CUDA backend requested, but no CUDA-capable device is available"
        );
    }
    auto& plan = *implementation_;
    plan.backend = backend;
    plan.source_count = source_positions.size();
    plan.target_count = target_positions.size();
    plan.self_indices.assign(plan.target_count, -1);
    check_cuda(cudaStreamCreateWithFlags(&plan.stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");

    const std::size_t source_bytes = plan.source_count * sizeof(Vec3);
    const std::size_t target_bytes = plan.target_count * sizeof(Vec3);
    const std::size_t source_allocation = std::max(source_bytes, sizeof(Vec3));
    const std::size_t target_allocation = std::max(target_bytes, sizeof(Vec3));
    check_cuda(cudaMalloc(&plan.device_sources, source_allocation), "cudaMalloc sources");
    check_cuda(cudaMalloc(&plan.device_targets, target_allocation), "cudaMalloc targets");
    check_cuda(cudaMalloc(&plan.device_moments, source_allocation), "cudaMalloc moments");
    check_cuda(cudaMalloc(&plan.device_fields, target_allocation), "cudaMalloc fields");
    check_cuda(cudaMalloc(&plan.device_potentials,
                          std::max(plan.target_count * sizeof(double), sizeof(double))),
               "cudaMalloc potentials");
    check_cuda(cudaMalloc(&plan.device_self_indices,
                          std::max(plan.target_count * sizeof(int), sizeof(int))),
               "cudaMalloc identities");
    check_cuda(cudaMallocHost(&plan.pinned_moments, source_allocation),
               "cudaMallocHost moments");
    check_cuda(cudaMallocHost(&plan.pinned_fields, target_allocation),
               "cudaMallocHost fields");
    check_cuda(cudaMallocHost(&plan.pinned_potentials,
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

    plan.statistics.setup_h2d_bytes = source_bytes + target_bytes +
        plan.target_count * sizeof(int);
    plan.statistics.persistent_device_bytes = 2 * source_bytes +
        2 * target_bytes + plan.target_count * (sizeof(double) + sizeof(int));
    plan.statistics.plan_generation_count = 1;
    plan.statistics.static_upload_count = 1;
    plan.statistics.static_m2l_upload_count = 0;
}

CudaFmmPlan::~CudaFmmPlan()
{
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
    cudaStreamDestroy(plan.stream);
    delete implementation_;
}

void CudaFmmPlan::evaluate(const std::span<const Vec3> moments,
                           const std::span<PotentialField> results,
                           const OutputFlags output,
                           const std::span<const int> target_source_indices)
{
    auto& plan = *implementation_;
    if (moments.size() != plan.source_count || results.size() != plan.target_count) {
        throw std::invalid_argument("CUDA evaluation array size mismatch");
    }
    std::copy(moments.begin(), moments.end(), plan.pinned_moments);
    const std::size_t moment_bytes = plan.source_count * sizeof(Vec3);
    check_cuda(cudaMemcpyAsync(plan.device_moments, plan.pinned_moments,
                               moment_bytes, cudaMemcpyHostToDevice, plan.stream),
               "upload dipole moments");
    plan.statistics.evaluation_h2d_bytes = moment_bytes;
    plan.statistics.evaluation_h2d_calls = 1;

    if (!target_source_indices.empty() &&
        !std::equal(target_source_indices.begin(), target_source_indices.end(),
                    plan.self_indices.begin())) {
        std::copy(target_source_indices.begin(), target_source_indices.end(),
                  plan.self_indices.begin());
        check_cuda(cudaMemcpyAsync(plan.device_self_indices,
                                   plan.self_indices.data(),
                                   plan.target_count * sizeof(int),
                                   cudaMemcpyHostToDevice, plan.stream),
                   "upload changed identity map");
        plan.statistics.evaluation_h2d_bytes +=
            plan.target_count * sizeof(int);
        ++plan.statistics.evaluation_h2d_calls;
    }

    constexpr int threads = 128;
    const int blocks = static_cast<int>((plan.target_count + threads - 1) / threads);
    const bool field = has_flag(output, OutputFlags::Field);
    const bool potential = has_flag(output, OutputFlags::Potential);
    if (plan.target_count != 0) {
        dipole_field_kernel<<<blocks, threads, 0, plan.stream>>>(
            plan.device_targets, plan.device_sources, plan.device_moments,
            plan.device_self_indices, plan.source_count, plan.target_count,
            plan.device_fields, plan.device_potentials, field, potential
        );
        check_cuda(cudaGetLastError(), "launch CUDA FMM evaluation");
    }

    plan.statistics.evaluation_d2h_bytes = 0;
    plan.statistics.evaluation_d2h_calls = 0;
    if (field) {
        const std::size_t field_bytes = plan.target_count * sizeof(Vec3);
        check_cuda(cudaMemcpyAsync(plan.pinned_fields, plan.device_fields,
                                   field_bytes, cudaMemcpyDeviceToHost, plan.stream),
                   "download magnetic field");
        plan.statistics.evaluation_d2h_bytes += field_bytes;
        ++plan.statistics.evaluation_d2h_calls;
    }
    if (potential) {
        const std::size_t potential_bytes = plan.target_count * sizeof(double);
        check_cuda(cudaMemcpyAsync(plan.pinned_potentials, plan.device_potentials,
                                   potential_bytes, cudaMemcpyDeviceToHost,
                                   plan.stream), "download scalar potential");
        plan.statistics.evaluation_d2h_bytes += potential_bytes;
        ++plan.statistics.evaluation_d2h_calls;
    }
    check_cuda(cudaStreamSynchronize(plan.stream), "complete CUDA evaluation");
    for (std::size_t index = 0; index < plan.target_count; ++index) {
        results[index].H = field ? plan.pinned_fields[index] : Vec3{};
        results[index].phi = potential ? plan.pinned_potentials[index] : 0.0;
    }
}

const CudaPlanStatistics& CudaFmmPlan::statistics() const noexcept
{
    return implementation_->statistics;
}

} // namespace cdfmm
