// SPDX-License-Identifier: Apache-2.0

#include "cuda_fmm_plan.hpp"
#include "cuda_m2l_plan.hpp"
#include "cuda_p2p_plan.hpp"
#include "cdfmm/cuda_direct.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

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

void check_cublas(const cublasStatus_t status, const char* operation)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed");
    }
}

__global__ void gather_m2l_kernel(const double* multipoles,
                                  const int* sources,
                                  const std::size_t interaction_count,
                                  const int coefficient_count,
                                  double* gathered)
{
    const std::size_t value = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t value_count = interaction_count * coefficient_count;
    if (value >= value_count) {
        return;
    }
    const std::size_t interaction = value / coefficient_count;
    const int coefficient = static_cast<int>(value % coefficient_count);
    gathered[value] = multipoles[
        static_cast<std::size_t>(sources[interaction]) * coefficient_count +
        coefficient
    ];
}

__global__ void scatter_m2l_kernel(const double* translated,
                                   const int* targets,
                                   const std::size_t interaction_count,
                                   const int coefficient_count,
                                   double* locals)
{
    const std::size_t value = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t value_count = interaction_count * coefficient_count;
    if (value >= value_count) {
        return;
    }
    const std::size_t interaction = value / coefficient_count;
    const int coefficient = static_cast<int>(value % coefficient_count);
    atomicAdd(
        locals + static_cast<std::size_t>(targets[interaction]) *
            coefficient_count + coefficient,
        translated[value]
    );
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
    ExecutionBackend backend{ExecutionBackend::CpuStatic};
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

bool cuda_runtime_available() noexcept
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

bool cuda_compiled() noexcept
{
    return true;
}

bool cuda_direct_available() noexcept
{
    return cuda_runtime_available();
}

bool cuda_m2l_available() noexcept
{
    return cuda_m2l_p2p_available();
}

bool cuda_m2l_p2p_available() noexcept
{
    return cuda_runtime_available();
}

bool cuda_full_available() noexcept
{
    return false;
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
    cudaEventDestroy(plan.evaluation_start);
    cudaEventDestroy(plan.h2d_complete);
    cudaEventDestroy(plan.kernel_complete);
    cudaEventDestroy(plan.d2h_complete);
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
    check_cuda(cudaEventRecord(plan.evaluation_start, plan.stream),
               "record CUDA evaluation start");
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
    check_cuda(cudaEventRecord(plan.h2d_complete, plan.stream),
               "record CUDA H2D completion");

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
    check_cuda(cudaEventRecord(plan.kernel_complete, plan.stream),
               "record CUDA kernel completion");

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
    check_cuda(cudaEventRecord(plan.d2h_complete, plan.stream),
               "record CUDA D2H completion");
    check_cuda(cudaEventSynchronize(plan.d2h_complete),
               "complete CUDA evaluation");
    float h2d_milliseconds = 0.0F;
    float kernel_milliseconds = 0.0F;
    float d2h_milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(
                   &h2d_milliseconds,
                   plan.evaluation_start,
                   plan.h2d_complete
               ),
               "measure CUDA H2D time");
    check_cuda(cudaEventElapsedTime(
                   &kernel_milliseconds,
                   plan.h2d_complete,
                   plan.kernel_complete
               ),
               "measure CUDA kernel time");
    check_cuda(cudaEventElapsedTime(
                   &d2h_milliseconds,
                   plan.kernel_complete,
                   plan.d2h_complete
               ),
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

const CudaPlanStatistics& CudaFmmPlan::statistics() const noexcept
{
    return implementation_->statistics;
}

const CudaEvaluationTimings&
CudaFmmPlan::evaluation_timings() const noexcept
{
    return implementation_->evaluation_timings;
}

std::vector<PotentialField> cuda_direct_p2p_reference(
    const std::span<const Vec3> targets,
    const std::span<const Vec3> sources,
    const std::span<const Vec3> moments,
    const OutputFlags output,
    const std::span<const int> target_source_indices)
{
    if (moments.size() != sources.size()) {
        throw std::invalid_argument(
            "cuda_direct_p2p_reference requires one moment per source"
        );
    }
    if (!target_source_indices.empty() &&
        target_source_indices.size() != targets.size()) {
        throw std::invalid_argument(
            "cuda_direct_p2p_reference identity map has incorrect length"
        );
    }

    CudaFmmPlan plan(sources, targets);
    std::vector<PotentialField> results(targets.size());
    plan.evaluate(moments, results, output, target_source_indices);
    return results;
}

//------------------------------------------------------------------------------
// Hybrid static M2L plan
//------------------------------------------------------------------------------

struct CudaM2LPlan::Implementation {
    struct Group {
        std::size_t matrix_offset{0};
        std::size_t interaction_offset{0};
        int columns{0};
    };

    int coefficient_count{0};
    std::vector<int> source_nodes{};
    std::vector<int> target_nodes{};
    std::vector<Group> groups{};
    std::vector<double> host_multipoles{};
    std::vector<double> host_locals{};
    double* matrices{nullptr};
    int* sources{nullptr};
    int* targets{nullptr};
    double* multipoles{nullptr};
    double* locals{nullptr};
    double* gathered{nullptr};
    double* translated{nullptr};
    cudaStream_t stream{nullptr};
    cublasHandle_t cublas{nullptr};
    cudaEvent_t start{nullptr};
    cudaEvent_t h2d{nullptr};
    cudaEvent_t gather{nullptr};
    cudaEvent_t multiply{nullptr};
    cudaEvent_t scatter{nullptr};
    cudaEvent_t d2h{nullptr};
    CudaPlanStatistics statistics{};
    CudaEvaluationTimings timings{};
};

CudaM2LPlan::CudaM2LPlan(
    const int coefficient_count,
    const std::span<const CudaM2LGroupView> group_views)
    : implementation_(new Implementation{})
{
    if (!cuda_runtime_available()) {
        delete implementation_;
        implementation_ = nullptr;
        throw std::runtime_error("CudaM2L requires an available CUDA device");
    }
    Implementation& plan = *implementation_;
    plan.coefficient_count = coefficient_count;

    std::vector<double> matrices;
    std::vector<int> sources;
    std::vector<int> targets;
    for (const CudaM2LGroupView& view : group_views) {
        Implementation::Group group;
        group.matrix_offset = matrices.size();
        group.interaction_offset = sources.size();
        group.columns = static_cast<int>(view.sources.size());
        plan.groups.push_back(group);
        matrices.insert(matrices.end(), view.matrix.begin(), view.matrix.end());
        sources.insert(sources.end(), view.sources.begin(), view.sources.end());
        targets.insert(targets.end(), view.targets.begin(), view.targets.end());
    }

    // Compact node indices make the dynamic transfers contain coefficients
    // required by M2L rather than the complete uniform tree.
    plan.source_nodes = sources;
    plan.target_nodes = targets;
    std::sort(plan.source_nodes.begin(), plan.source_nodes.end());
    plan.source_nodes.erase(
        std::unique(plan.source_nodes.begin(), plan.source_nodes.end()),
        plan.source_nodes.end()
    );
    std::sort(plan.target_nodes.begin(), plan.target_nodes.end());
    plan.target_nodes.erase(
        std::unique(plan.target_nodes.begin(), plan.target_nodes.end()),
        plan.target_nodes.end()
    );
    for (int& source : sources) {
        source = static_cast<int>(std::lower_bound(
            plan.source_nodes.begin(), plan.source_nodes.end(), source
        ) - plan.source_nodes.begin());
    }
    for (int& target : targets) {
        target = static_cast<int>(std::lower_bound(
            plan.target_nodes.begin(), plan.target_nodes.end(), target
        ) - plan.target_nodes.begin());
    }

    const std::size_t source_values = plan.source_nodes.size() *
        static_cast<std::size_t>(coefficient_count);
    const std::size_t target_values = plan.target_nodes.size() *
        static_cast<std::size_t>(coefficient_count);
    const std::size_t interaction_values = sources.size() *
        static_cast<std::size_t>(coefficient_count);
    plan.host_multipoles.resize(source_values);
    plan.host_locals.resize(target_values);

    check_cuda(cudaStreamCreateWithFlags(&plan.stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags M2L");
    check_cublas(cublasCreate(&plan.cublas), "cublasCreate");
    check_cublas(cublasSetStream(plan.cublas, plan.stream), "cublasSetStream");
    check_cuda(cudaEventCreate(&plan.start), "cudaEventCreate");
    check_cuda(cudaEventCreate(&plan.h2d), "cudaEventCreate");
    check_cuda(cudaEventCreate(&plan.gather), "cudaEventCreate");
    check_cuda(cudaEventCreate(&plan.multiply), "cudaEventCreate");
    check_cuda(cudaEventCreate(&plan.scatter), "cudaEventCreate");
    check_cuda(cudaEventCreate(&plan.d2h), "cudaEventCreate");

    const auto allocate = [](auto** pointer, const std::size_t bytes) {
        if (bytes != 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(pointer), bytes),
                       "cudaMalloc M2L buffer");
        }
    };
    allocate(&plan.matrices, matrices.size() * sizeof(double));
    allocate(&plan.sources, sources.size() * sizeof(int));
    allocate(&plan.targets, targets.size() * sizeof(int));
    allocate(&plan.multipoles, source_values * sizeof(double));
    allocate(&plan.locals, target_values * sizeof(double));
    allocate(&plan.gathered, interaction_values * sizeof(double));
    allocate(&plan.translated, interaction_values * sizeof(double));

    if (!matrices.empty()) {
        check_cuda(cudaMemcpyAsync(plan.matrices, matrices.data(),
            matrices.size() * sizeof(double), cudaMemcpyHostToDevice,
            plan.stream), "upload static M2L matrices");
    }
    if (!sources.empty()) {
        check_cuda(cudaMemcpyAsync(plan.sources, sources.data(),
            sources.size() * sizeof(int), cudaMemcpyHostToDevice, plan.stream),
            "upload static M2L sources");
        check_cuda(cudaMemcpyAsync(plan.targets, targets.data(),
            targets.size() * sizeof(int), cudaMemcpyHostToDevice, plan.stream),
            "upload static M2L targets");
    }
    check_cuda(cudaStreamSynchronize(plan.stream), "finish static M2L upload");
    plan.statistics.setup_h2d_bytes = matrices.size() * sizeof(double) +
        (sources.size() + targets.size()) * sizeof(int);
    plan.statistics.persistent_device_bytes =
        plan.statistics.setup_h2d_bytes +
        (source_values + target_values + 2 * interaction_values) * sizeof(double);
    plan.statistics.plan_generation_count = 1;
    plan.statistics.static_upload_count = 1;
    plan.statistics.static_m2l_upload_count = 1;
}

CudaM2LPlan::~CudaM2LPlan()
{
    if (implementation_ == nullptr) {
        return;
    }
    Implementation& plan = *implementation_;
    cudaFree(plan.matrices);
    cudaFree(plan.sources);
    cudaFree(plan.targets);
    cudaFree(plan.multipoles);
    cudaFree(plan.locals);
    cudaFree(plan.gathered);
    cudaFree(plan.translated);
    cudaEventDestroy(plan.start);
    cudaEventDestroy(plan.h2d);
    cudaEventDestroy(plan.gather);
    cudaEventDestroy(plan.multiply);
    cudaEventDestroy(plan.scatter);
    cudaEventDestroy(plan.d2h);
    cublasDestroy(plan.cublas);
    cudaStreamDestroy(plan.stream);
    delete implementation_;
}

void CudaM2LPlan::evaluate(
    const std::span<const std::vector<double>> multipoles,
    const std::span<std::vector<double>> raw_locals)
{
    Implementation& plan = *implementation_;
    const std::size_t n = static_cast<std::size_t>(plan.coefficient_count);
    for (std::size_t packed = 0; packed < plan.source_nodes.size(); ++packed) {
        std::copy(multipoles[static_cast<std::size_t>(plan.source_nodes[packed])].begin(),
                  multipoles[static_cast<std::size_t>(plan.source_nodes[packed])].end(),
                  plan.host_multipoles.begin() + static_cast<std::ptrdiff_t>(packed * n));
    }

    check_cuda(cudaEventRecord(plan.start, plan.stream), "record M2L start");
    if (!plan.host_multipoles.empty()) {
        check_cuda(cudaMemcpyAsync(plan.multipoles, plan.host_multipoles.data(),
            plan.host_multipoles.size() * sizeof(double), cudaMemcpyHostToDevice,
            plan.stream), "upload multipoles");
    }
    check_cuda(cudaEventRecord(plan.h2d, plan.stream), "record M2L H2D");
    const std::size_t interactions = plan.groups.empty() ? 0 :
        plan.groups.back().interaction_offset + plan.groups.back().columns;
    if (interactions != 0) {
        constexpr int threads = 256;
        const std::size_t values = interactions * n;
        gather_m2l_kernel<<<(values + threads - 1) / threads, threads, 0,
            plan.stream>>>(plan.multipoles, plan.sources, interactions,
                           plan.coefficient_count, plan.gathered);
    }
    check_cuda(cudaEventRecord(plan.gather, plan.stream), "record M2L gather");
    constexpr double one = 1.0;
    constexpr double zero = 0.0;
    for (const Implementation::Group& group : plan.groups) {
        check_cublas(cublasDgemm(
            plan.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            plan.coefficient_count, group.columns, plan.coefficient_count,
            &one, plan.matrices + group.matrix_offset, plan.coefficient_count,
            plan.gathered + group.interaction_offset * n, plan.coefficient_count,
            &zero, plan.translated + group.interaction_offset * n,
            plan.coefficient_count
        ), "cublasDgemm static M2L");
    }
    check_cuda(cudaEventRecord(plan.multiply, plan.stream), "record M2L multiply");
    if (!plan.host_locals.empty()) {
        check_cuda(cudaMemsetAsync(plan.locals, 0,
            plan.host_locals.size() * sizeof(double), plan.stream),
            "clear raw M2L locals");
    }
    if (interactions != 0) {
        constexpr int threads = 256;
        const std::size_t values = interactions * n;
        scatter_m2l_kernel<<<(values + threads - 1) / threads, threads, 0,
            plan.stream>>>(plan.translated, plan.targets, interactions,
                           plan.coefficient_count, plan.locals);
    }
    check_cuda(cudaEventRecord(plan.scatter, plan.stream), "record M2L scatter");
    if (!plan.host_locals.empty()) {
        check_cuda(cudaMemcpyAsync(plan.host_locals.data(), plan.locals,
            plan.host_locals.size() * sizeof(double), cudaMemcpyDeviceToHost,
            plan.stream), "download raw M2L locals");
    }
    check_cuda(cudaEventRecord(plan.d2h, plan.stream), "record M2L D2H");
    check_cuda(cudaEventSynchronize(plan.d2h), "wait for CUDA M2L");

    for (std::size_t packed = 0; packed < plan.target_nodes.size(); ++packed) {
        std::copy(plan.host_locals.begin() + static_cast<std::ptrdiff_t>(packed * n),
                  plan.host_locals.begin() + static_cast<std::ptrdiff_t>((packed + 1) * n),
                  raw_locals[static_cast<std::size_t>(plan.target_nodes[packed])].begin());
    }
    const auto elapsed = [](cudaEvent_t first, cudaEvent_t second) {
        float milliseconds = 0.0F;
        check_cuda(cudaEventElapsedTime(&milliseconds, first, second),
                   "time CUDA M2L phase");
        return static_cast<double>(milliseconds) * 1.0e-3;
    };
    plan.timings.h2d_seconds = elapsed(plan.start, plan.h2d);
    plan.timings.gather_seconds = elapsed(plan.h2d, plan.gather);
    plan.timings.multiply_seconds = elapsed(plan.gather, plan.multiply);
    plan.timings.scatter_seconds = elapsed(plan.multiply, plan.scatter);
    plan.timings.d2h_seconds = elapsed(plan.scatter, plan.d2h);
    plan.timings.kernel_seconds = plan.timings.gather_seconds +
        plan.timings.multiply_seconds + plan.timings.scatter_seconds;
    plan.statistics.evaluation_h2d_bytes =
        plan.host_multipoles.size() * sizeof(double);
    plan.statistics.evaluation_d2h_bytes =
        plan.host_locals.size() * sizeof(double);
    plan.statistics.evaluation_h2d_calls = 1;
    plan.statistics.evaluation_d2h_calls = 1;
}

const CudaPlanStatistics& CudaM2LPlan::statistics() const noexcept
{
    return implementation_->statistics;
}

const CudaEvaluationTimings& CudaM2LPlan::timings() const noexcept
{
    return implementation_->timings;
}

namespace {

__global__ void static_p2p_kernel(
    const int target_count,
    const int* row_offsets,
    const StaticDipoleBlock* blocks,
    const Vec3* moments,
    const int* self_indices,
    Vec3* fields)
{
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
        field.x += tensor.xx * moment.x + tensor.xy * moment.y +
            tensor.xz * moment.z;
        field.y += tensor.xy * moment.x + tensor.yy * moment.y +
            tensor.yz * moment.z;
        field.z += tensor.xz * moment.x + tensor.yz * moment.y +
            tensor.zz * moment.z;
    }
    fields[target] = field;
}

} // namespace

struct CudaP2PPlan::Implementation {
    int source_count{0};
    int target_count{0};
    int* row_offsets{nullptr};
    StaticDipoleBlock* blocks{nullptr};
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
    bool pending{false};
};

CudaP2PPlan::CudaP2PPlan(const StaticP2POperator& operator_map)
    : implementation_(new Implementation{})
{
    auto& plan = *implementation_;
    plan.source_count = operator_map.source_count;
    plan.target_count = operator_map.target_count;
    check_cuda(cudaStreamCreateWithFlags(&plan.stream, cudaStreamNonBlocking),
               "create static P2P stream");
    check_cuda(cudaEventCreate(&plan.start), "create static P2P event");
    check_cuda(cudaEventCreate(&plan.h2d), "create static P2P event");
    check_cuda(cudaEventCreate(&plan.kernel), "create static P2P event");
    check_cuda(cudaEventCreate(&plan.d2h), "create static P2P event");
    const std::size_t row_bytes = operator_map.row_offsets.size() * sizeof(int);
    const std::size_t block_bytes = operator_map.blocks.size() *
        sizeof(StaticDipoleBlock);
    check_cuda(cudaMalloc(&plan.row_offsets, std::max(row_bytes, sizeof(int))),
               "allocate P2P rows");
    check_cuda(cudaMalloc(&plan.blocks,
                          std::max(block_bytes, sizeof(StaticDipoleBlock))),
               "allocate P2P blocks");
    check_cuda(cudaMalloc(&plan.moments, std::max(
                              operator_map.source_count * sizeof(Vec3),
                              sizeof(Vec3))),
               "allocate P2P moments");
    check_cuda(cudaMalloc(&plan.self_indices, std::max(
                              operator_map.target_count * sizeof(int),
                              sizeof(int))),
               "allocate P2P identities");
    check_cuda(cudaMalloc(&plan.fields, std::max(
                              operator_map.target_count * sizeof(Vec3),
                              sizeof(Vec3))),
               "allocate P2P fields");
    check_cuda(cudaMallocHost(&plan.pinned_moments, std::max(
                                  operator_map.source_count * sizeof(Vec3),
                                  sizeof(Vec3))),
               "allocate pinned P2P moments");
    check_cuda(cudaMallocHost(&plan.pinned_self_indices, std::max(
                                  operator_map.target_count * sizeof(int),
                                  sizeof(int))),
               "allocate pinned P2P identities");
    check_cuda(cudaMallocHost(&plan.pinned_fields, std::max(
                                  operator_map.target_count * sizeof(Vec3),
                                  sizeof(Vec3))),
               "allocate pinned P2P fields");
    check_cuda(cudaMemcpy(plan.row_offsets, operator_map.row_offsets.data(),
                          row_bytes, cudaMemcpyHostToDevice), "upload P2P rows");
    if (block_bytes != 0) {
        check_cuda(cudaMemcpy(plan.blocks, operator_map.blocks.data(), block_bytes,
                              cudaMemcpyHostToDevice), "upload P2P blocks");
    }
    plan.statistics.setup_h2d_bytes = row_bytes + block_bytes;
    plan.statistics.persistent_device_bytes = row_bytes + block_bytes +
        (operator_map.source_count + operator_map.target_count) * sizeof(Vec3) +
        operator_map.target_count * sizeof(int);
    plan.statistics.plan_generation_count = 1;
    plan.statistics.static_upload_count = 1;
    plan.statistics.static_p2p_upload_count = 1;
}

CudaP2PPlan::~CudaP2PPlan()
{
    if (implementation_ == nullptr) {
        return;
    }
    auto& plan = *implementation_;
    cancel_evaluate();
    cudaFree(plan.row_offsets);
    cudaFree(plan.blocks);
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
    const std::span<const int> target_source_indices)
{
    auto& plan = *implementation_;
    if (moments.size() != static_cast<std::size_t>(plan.source_count) ||
        target_source_indices.size() !=
            static_cast<std::size_t>(plan.target_count)) {
        throw std::invalid_argument("CUDA static P2P dimensions are inconsistent");
    }
    if (plan.pending) {
        throw std::logic_error("CUDA static P2P evaluation is already pending");
    }
    std::copy(moments.begin(), moments.end(), plan.pinned_moments);
    std::copy(target_source_indices.begin(), target_source_indices.end(),
              plan.pinned_self_indices);
    plan.pending = true;
    try {
        check_cuda(cudaEventRecord(plan.start, plan.stream), "record P2P start");
        check_cuda(cudaMemcpyAsync(plan.moments, plan.pinned_moments,
                                   moments.size_bytes(), cudaMemcpyHostToDevice,
                                   plan.stream), "upload P2P moments");
        check_cuda(cudaMemcpyAsync(plan.self_indices, plan.pinned_self_indices,
                                   target_source_indices.size_bytes(),
                                   cudaMemcpyHostToDevice, plan.stream),
                   "upload P2P identities");
        check_cuda(cudaEventRecord(plan.h2d, plan.stream), "record P2P upload");
        if (plan.target_count != 0) {
            constexpr int threads = 128;
            static_p2p_kernel<<<
                (plan.target_count + threads - 1) / threads, threads, 0,
                plan.stream>>>(
                plan.target_count, plan.row_offsets, plan.blocks, plan.moments,
                plan.self_indices, plan.fields
            );
            check_cuda(cudaGetLastError(), "launch static P2P kernel");
        }
        check_cuda(cudaEventRecord(plan.kernel, plan.stream),
                   "record P2P kernel");
        check_cuda(cudaMemcpyAsync(
            plan.pinned_fields, plan.fields,
            static_cast<std::size_t>(plan.target_count) * sizeof(Vec3),
            cudaMemcpyDeviceToHost, plan.stream
        ), "download P2P fields");
        check_cuda(cudaEventRecord(plan.d2h, plan.stream),
                   "record P2P download");
    } catch (...) {
        cancel_evaluate();
        throw;
    }
    plan.statistics.evaluation_h2d_bytes = moments.size_bytes() +
        target_source_indices.size_bytes();
    plan.statistics.evaluation_d2h_bytes =
        static_cast<std::size_t>(plan.target_count) * sizeof(Vec3);
    ++plan.statistics.evaluation_h2d_calls;
    ++plan.statistics.evaluation_d2h_calls;
}

void CudaP2PPlan::finish_evaluate(const std::span<Vec3> fields)
{
    auto& plan = *implementation_;
    if (!plan.pending) {
        throw std::logic_error("CUDA static P2P has no pending evaluation");
    }
    if (fields.size() != static_cast<std::size_t>(plan.target_count)) {
        throw std::invalid_argument("CUDA static P2P output size is inconsistent");
    }
    check_cuda(cudaEventSynchronize(plan.d2h), "synchronise static P2P");
    std::copy(
        plan.pinned_fields,
        plan.pinned_fields + fields.size(),
        fields.begin()
    );
    auto elapsed = [](cudaEvent_t first, cudaEvent_t second) {
        float milliseconds = 0.0F;
        check_cuda(cudaEventElapsedTime(&milliseconds, first, second),
                   "time static P2P phase");
        return static_cast<double>(milliseconds) * 1.0e-3;
    };
    plan.timings = {};
    plan.timings.h2d_seconds = elapsed(plan.start, plan.h2d);
    plan.timings.kernel_seconds = elapsed(plan.h2d, plan.kernel);
    plan.timings.d2h_seconds = elapsed(plan.kernel, plan.d2h);
    plan.pending = false;
}

void CudaP2PPlan::cancel_evaluate() noexcept
{
    if (implementation_ == nullptr || !implementation_->pending) {
        return;
    }
    cudaStreamSynchronize(implementation_->stream);
    implementation_->pending = false;
}

void CudaP2PPlan::evaluate(const std::span<const Vec3> moments,
                           const std::span<const int> target_source_indices,
                           const std::span<Vec3> fields)
{
    begin_evaluate(moments, target_source_indices);
    try {
        finish_evaluate(fields);
    } catch (...) {
        cancel_evaluate();
        throw;
    }
}

const CudaPlanStatistics& CudaP2PPlan::statistics() const noexcept
{
    return implementation_->statistics;
}

const CudaEvaluationTimings& CudaP2PPlan::timings() const noexcept
{
    return implementation_->timings;
}

} // namespace cdfmm
