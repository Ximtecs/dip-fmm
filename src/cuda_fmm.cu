// SPDX-License-Identifier: Apache-2.0

#include "cuda_fmm_plan.hpp"
#include "profile.hpp"
#include "cuda_m2l_plan.hpp"
#include "backend/cuda/common/error.hpp"
#include "backend/cuda/common/runtime.hpp"
#include "backend/cuda/m2l/internal.hpp"
#include "backend/cuda/p2p/internal.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace cdfmm {

using cuda_p2p_detail::CudaBsrP2PDeviceView;
using cuda_p2p_detail::CudaP2PDeviceView;
using cuda_p2p_detail::CudaSignedDictionaryP2PDeviceView;
using cuda_p2p_detail::launch_bsr_p2p;
using cuda_p2p_detail::launch_signed_dictionary_p2p;
using cuda_p2p_detail::launch_static_p2p;
using cuda_p2p_detail::release_p2p_device_view;
using cuda_p2p_detail::static_operator_threads;
using cuda_p2p_detail::upload_cuda_bsr;
using cuda_p2p_detail::upload_cuda_canonical;
using cuda_p2p_detail::upload_cuda_signed_dictionary;
using cuda_m2l_detail::CudaM2LExecutionPlan;

namespace {

// This translation unit retains far-field kernels and complete FMM
// orchestration. List-1 execution is provided by the CUDA P2P backend while
// the separate streams retain near/far overlap.

using cuda_detail::check_cuda;

template <typename Vector>
__global__ void permute_moments_kernel(const Vector *input,
                                       const int *permutation, const int count,
                                       Vector *sorted) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    sorted[index] = input[permutation[index]];
    }
}

template <typename Entry, typename Scalar>
__global__ void apply_entries_kernel(const Entry *entries,
                                     const std::size_t count,
                                     const Scalar *input, Scalar *output) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const Entry entry = entries[index];
        atomicAdd(output + entry.output, entry.value * input[entry.input]);
  }
}

template <typename Entry, typename Scalar>
__global__ void
apply_shared_translation_kernel(const Entry *matrices,
                                const CudaTranslationInteraction *interactions,
                                const std::size_t interaction_count,
                                const int entries_per_matrix,
                                const int coefficient_count,
                                const int* coefficient_degrees,
                                const int level,
                                const Scalar *input, Scalar *output) {
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
  const Entry entry =
      matrices[static_cast<std::size_t>(interaction.matrix_id) *
                   entries_per_matrix +
               matrix_entry];
  const int degree_difference =
      coefficient_degrees[entry.output] - coefficient_degrees[entry.input];
  const int power = degree_difference < 0 ? -degree_difference
                                         : degree_difference;
  const Scalar scaled_value =
      ldexp(static_cast<Scalar>(entry.value), -(level - 1) * power);
  atomicAdd(output +
                static_cast<std::size_t>(interaction.target_node) *
                    coefficient_count +
                entry.output,
            scaled_value *
                input[static_cast<std::size_t>(interaction.source_node) *
                          coefficient_count +
                      entry.input]);
}


template <typename Vector>
__global__ void combine_order_kernel(const Vector *far_fields,
                                     const Vector *near_fields,
                                     const int *target_permutation,
                                     const int count, Vector *output) {
  const int sorted = blockIdx.x * blockDim.x + threadIdx.x;
  if (sorted < count) {
    Vector value = far_fields[sorted];
        value.x += near_fields[sorted].x;
        value.y += near_fields[sorted].y;
        value.z += near_fields[sorted].z;
        output[target_permutation[sorted]] = value;
  }
}

} // namespace

bool cuda_m2l_available() noexcept { return cuda_m2l_p2p_available(); }

bool cuda_m2l_p2p_available() noexcept { return cuda_runtime_available(); }

bool cuda_full_available() noexcept { return cuda_runtime_available(); }


//------------------------------------------------------------------------------
// Complete static CUDA FMM
//------------------------------------------------------------------------------

struct CudaFullPlan::Implementation {
    bool fp32{false};
    int coefficient_count{0};
    int node_count{0};
    int source_count{0};
    int target_count{0};
    int* source_permutation{nullptr};
    int* target_permutation{nullptr};
    int* coefficient_degrees{nullptr};
    int* self_indices{nullptr};
    CudaP2PDeviceView<StaticDipoleBlock> p2p{};
    CudaBsrP2PDeviceView<double> p2p_bsr{};
    CudaSignedDictionaryP2PDeviceView<double> p2p_dictionary{};
    CudaP2PDeviceView<FloatStaticDipoleBlock> p2p_float{};
    CudaBsrP2PDeviceView<float> p2p_bsr_float{};
    CudaSignedDictionaryP2PDeviceView<float> p2p_dictionary_float{};
  bool use_p2p_bsr{false};
  bool use_p2p_dictionary{false};
  bool p2p_dictionary_power2_microtiles{false};
  StaticOperatorEntry *entries{nullptr};
  StaticOperatorEntry *m2m_matrices{nullptr};
  StaticOperatorEntry *l2l_matrices{nullptr};
  CudaTranslationInteraction *m2m_interactions{nullptr};
  CudaTranslationInteraction *l2l_interactions{nullptr};
  CudaM2LExecutionPlan<double, StaticM2LPlan> *m2l{nullptr};
  FloatStaticOperatorEntry *entries_float{nullptr};
  FloatStaticOperatorEntry *m2m_matrices_float{nullptr};
  FloatStaticOperatorEntry *l2l_matrices_float{nullptr};
  CudaM2LExecutionPlan<float, FloatStaticM2LPlan> *m2l_float{nullptr};
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
    FloatVec3 *moments_float{nullptr};
    FloatVec3 *sorted_moments_float{nullptr};
    float *multipoles_float{nullptr};
    float *locals_float{nullptr};
    FloatVec3 *far_fields_float{nullptr};
    FloatVec3 *near_fields_float{nullptr};
    FloatVec3 *final_fields_float{nullptr};
    FloatVec3 *pinned_moments_float{nullptr};
    FloatVec3 *pinned_fields_float{nullptr};
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
  int maximum_level{0};
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
  plan.use_p2p_dictionary = data.use_p2p_dictionary;
  plan.p2p_dictionary_power2_microtiles =
      data.p2p_dictionary_power2_microtiles;
  if (plan.use_p2p_bsr && plan.use_p2p_dictionary) {
    throw std::invalid_argument(
        "full CUDA P2P cannot select BSR and tensor dictionary together");
  }
  plan.p2p.target_count = plan.target_count;
  plan.p2p_bsr.target_count = plan.target_count;
  plan.p2p_bsr.source_count = plan.source_count;
  plan.p2p_bsr.interaction_count =
      static_cast<int>(data.p2p_bsr.source_indices.size());
  plan.p2p_block_count = data.use_p2p_dictionary
      ? data.p2p_dictionary.token_count()
      : (data.use_p2p_bsr ? data.p2p_bsr.source_indices.size()
                          : data.p2p.blocks.size());
    std::vector<StaticOperatorEntry> entries;
    const auto append_stage = [&](const auto& stage_entries) {
        plan.offsets.push_back(entries.size());
        plan.counts.push_back(stage_entries.size());
        entries.insert(entries.end(), stage_entries.begin(), stage_entries.end());
    return static_cast<int>(plan.offsets.size() - 1);
  };
  plan.p2m_stage = append_stage(data.p2m);
  plan.l2p_stage = append_stage(data.l2p);
  plan.maximum_level = data.m2l.level_count - 1;
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
  allocate(&plan.coefficient_degrees,
           data.coefficient_degrees.size() * sizeof(int));
  if (plan.use_p2p_dictionary) {
    upload_cuda_signed_dictionary(
        data.p2p_dictionary,
        plan.p2p_dictionary,
        plan.statistics,
        data.p2p_dictionary_target_owned,
        data.p2p_dictionary_power2_microtiles);
  } else if (!plan.use_p2p_bsr) {
    allocate(&plan.self_indices,
             static_cast<std::size_t>(plan.target_count) * sizeof(int));
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
    upload(plan.coefficient_degrees, data.coefficient_degrees.data(),
           data.coefficient_degrees.size() * sizeof(int));
  if (plan.use_p2p_dictionary) {
    // The dictionary metadata and values were uploaded together above.
  } else if (!plan.use_p2p_bsr) {
    upload_cuda_canonical(
        data.p2p, plan.p2p, plan.statistics, plan.far_field_stream,
        "allocate full FMM buffer", "upload full FMM static data");
  } else {
    upload_cuda_bsr(
        data.p2p_bsr, plan.p2p_bsr, plan.sorted_moments, plan.near_fields,
        plan.statistics, plan.far_field_stream, "allocate full FMM buffer",
        "upload full FMM static data", "create full FMM cuSPARSE P2P handle",
        "create full FMM cuSPARSE P2P BSR descriptor");
    plan.statistics.p2p_scratch_bytes = plan.p2p_bsr.workspace_size;
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
    if (!plan.use_p2p_bsr && !plan.use_p2p_dictionary &&
        !data.fixed_self_indices.empty()) {
      upload(plan.self_indices, data.fixed_self_indices.data(),
             data.fixed_self_indices.size() * sizeof(int));
    }
  }
  plan.m2l = new CudaM2LExecutionPlan<double, StaticM2LPlan>(
      data.m2l, plan.far_field_stream);
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
      (plan.use_p2p_bsr || plan.use_p2p_dictionary ||
               data.has_fixed_self_indices
           ? 0
           : static_cast<std::size_t>(plan.target_count) * sizeof(int)) +
      plan.statistics.m2l_scratch_bytes + plan.statistics.p2p_scratch_bytes;
  plan.statistics.plan_generation_count = 1;
  plan.statistics.static_upload_count = 1;
    plan.statistics.static_m2l_upload_count = 1;
    plan.statistics.static_p2p_upload_count = 1;
  plan.statistics.p2p_interaction_count = plan.p2p_block_count;
  if (plan.use_p2p_dictionary) {
    plan.statistics.p2p_interaction_count =
        data.p2p_dictionary.token_count();
    plan.statistics.p2p_identity_bytes = 0;
  } else if (plan.use_p2p_bsr) {
    plan.statistics.p2p_tensor_bytes =
        data.p2p_bsr.values.size() * sizeof(double);
    plan.statistics.p2p_index_bytes =
        data.p2p_bsr.source_indices.size() * sizeof(int);
      plan.statistics.p2p_row_metadata_bytes =
          data.p2p_bsr.row_offsets.size() * sizeof(int);
    plan.statistics.p2p_identity_bytes = 0;
    plan.statistics.p2p_scratch_bytes = plan.p2p_bsr.workspace_size;
    plan.statistics.p2p_threads_per_block = 0;
  } else {
    plan.statistics.p2p_tensor_bytes =
        data.p2p.blocks.size() * 6 * sizeof(double);
    plan.statistics.p2p_index_bytes =
        data.p2p.blocks.size() * 3 * sizeof(int);
    plan.statistics.p2p_row_metadata_bytes =
        data.p2p.row_offsets.size() * sizeof(int);
    plan.statistics.p2p_identity_bytes =
        static_cast<std::size_t>(plan.target_count) * sizeof(int);
    plan.statistics.p2p_threads_per_block = static_operator_threads;
  }
  plan.statistics.geometry_upload_count = 1;
}

CudaFullPlan::CudaFullPlan(const FloatCudaFullPlanData &data)
    : implementation_(new Implementation{}) {
  auto &plan = *implementation_;
  plan.fp32 = true;
  plan.coefficient_count = data.coefficient_count;
  plan.node_count = data.node_count;
  plan.source_count = data.source_count;
  plan.target_count = data.target_count;
  plan.use_p2p_bsr = data.use_p2p_bsr;
  plan.use_p2p_dictionary = data.use_p2p_dictionary;
  plan.p2p_dictionary_power2_microtiles =
      data.p2p_dictionary_power2_microtiles;
  if (plan.use_p2p_bsr && plan.use_p2p_dictionary) {
    throw std::invalid_argument(
        "FP32 full CUDA P2P cannot select BSR and tensor dictionary together");
  }
  plan.p2p_float.target_count = plan.target_count;
  plan.p2p_bsr_float.target_count = plan.target_count;
  plan.p2p_bsr_float.source_count = plan.source_count;
  plan.p2p_bsr_float.interaction_count =
      static_cast<int>(data.p2p_bsr.source_indices.size());
  plan.p2p_block_count = data.use_p2p_dictionary
      ? data.p2p_dictionary.token_count()
      : (data.use_p2p_bsr ? data.p2p_bsr.source_indices.size()
                          : data.p2p.blocks.size());

  std::vector<FloatStaticOperatorEntry> entries;
  const auto append_stage = [&](const auto &stage_entries) {
    plan.offsets.push_back(entries.size());
    plan.counts.push_back(stage_entries.size());
    entries.insert(entries.end(), stage_entries.begin(), stage_entries.end());
    return static_cast<int>(plan.offsets.size() - 1);
  };
  plan.p2m_stage = append_stage(data.p2m);
  plan.l2p_stage = append_stage(data.l2p);
  plan.maximum_level = data.m2l.level_count - 1;
  plan.m2m_entries_per_matrix = data.m2m.entries_per_matrix;
  plan.l2l_entries_per_matrix = data.l2l.entries_per_matrix;
  plan.m2m_interaction_count = data.m2m.interactions.size();
  plan.l2l_interaction_count = data.l2l.interactions.size();

  check_cuda(cudaStreamCreateWithFlags(&plan.far_field_stream,
                                       cudaStreamNonBlocking),
             "create FP32 full FMM far-field stream");
  check_cuda(cudaStreamCreateWithFlags(&plan.near_field_stream,
                                       cudaStreamNonBlocking),
             "create FP32 full FMM near-field stream");
  const std::array<cudaEvent_t *, 13> events{
      &plan.evaluation_start, &plan.moments_ready, &plan.p2m_complete,
      &plan.m2m_complete, &plan.m2l_scale_complete, &plan.m2l_complete,
      &plan.l2l_complete, &plan.l2p_complete, &plan.p2p_start,
      &plan.p2p_complete, &plan.combination_start,
      &plan.combination_complete, &plan.d2h_complete};
  for (cudaEvent_t *event : events) {
    check_cuda(cudaEventCreate(event), "create FP32 full FMM event");
  }
  const auto allocate = [](auto **pointer, const std::size_t bytes) {
    check_cuda(cudaMalloc(reinterpret_cast<void **>(pointer),
                          std::max(bytes, std::size_t{1})),
               "allocate FP32 full FMM buffer");
  };
  const std::size_t source_bytes =
      static_cast<std::size_t>(plan.source_count) * sizeof(FloatVec3);
  const std::size_t target_bytes =
      static_cast<std::size_t>(plan.target_count) * sizeof(FloatVec3);
  const std::size_t coefficient_bytes =
      static_cast<std::size_t>(plan.node_count) * plan.coefficient_count *
      sizeof(float);
  allocate(&plan.source_permutation,
           data.source_permutation.size() * sizeof(int));
  allocate(&plan.target_permutation,
           data.target_permutation.size() * sizeof(int));
  allocate(&plan.coefficient_degrees,
           data.coefficient_degrees.size() * sizeof(int));
  if (plan.use_p2p_dictionary) {
    upload_cuda_signed_dictionary(
        data.p2p_dictionary,
        plan.p2p_dictionary_float,
        plan.statistics,
        data.p2p_dictionary_target_owned,
        data.p2p_dictionary_power2_microtiles);
  } else if (!plan.use_p2p_bsr) {
    allocate(&plan.self_indices,
             static_cast<std::size_t>(plan.target_count) * sizeof(int));
  }
  allocate(&plan.entries_float,
           entries.size() * sizeof(FloatStaticOperatorEntry));
  allocate(&plan.m2m_matrices_float,
           data.m2m.matrices.size() * sizeof(FloatStaticOperatorEntry));
  allocate(&plan.l2l_matrices_float,
           data.l2l.matrices.size() * sizeof(FloatStaticOperatorEntry));
  allocate(&plan.m2m_interactions,
           data.m2m.interactions.size() * sizeof(CudaTranslationInteraction));
  allocate(&plan.l2l_interactions,
           data.l2l.interactions.size() * sizeof(CudaTranslationInteraction));
  allocate(&plan.moments_float, source_bytes);
  allocate(&plan.sorted_moments_float, source_bytes);
  allocate(&plan.multipoles_float, coefficient_bytes);
  allocate(&plan.locals_float, coefficient_bytes);
  allocate(&plan.far_fields_float, target_bytes);
  allocate(&plan.near_fields_float, target_bytes);
  allocate(&plan.final_fields_float, target_bytes);
  check_cuda(cudaMallocHost(&plan.pinned_moments_float,
                            std::max(source_bytes, std::size_t{1})),
             "allocate pinned FP32 full FMM moments");
  check_cuda(cudaMallocHost(&plan.pinned_fields_float,
                            std::max(target_bytes, std::size_t{1})),
             "allocate pinned FP32 full FMM fields");

  const auto upload = [&](void *destination, const void *source,
                          const std::size_t bytes) {
    if (bytes != 0) {
      check_cuda(cudaMemcpyAsync(destination, source, bytes,
                                 cudaMemcpyHostToDevice,
                                 plan.far_field_stream),
                 "upload FP32 full FMM static data");
      plan.statistics.setup_h2d_bytes += bytes;
    }
  };
  upload(plan.source_permutation, data.source_permutation.data(),
         data.source_permutation.size() * sizeof(int));
  upload(plan.target_permutation, data.target_permutation.data(),
         data.target_permutation.size() * sizeof(int));
  upload(plan.coefficient_degrees, data.coefficient_degrees.data(),
         data.coefficient_degrees.size() * sizeof(int));
  if (plan.use_p2p_dictionary) {
    // The dictionary metadata and values were uploaded together above.
  } else if (!plan.use_p2p_bsr) {
    upload_cuda_canonical(
        data.p2p, plan.p2p_float, plan.statistics, plan.far_field_stream,
        "allocate FP32 full FMM buffer", "upload FP32 full FMM static data");
  } else {
    upload_cuda_bsr(
        data.p2p_bsr, plan.p2p_bsr_float, plan.sorted_moments_float,
        plan.near_fields_float, plan.statistics, plan.far_field_stream,
        "allocate FP32 full FMM buffer", "upload FP32 full FMM static data",
        "create FP32 full FMM cuSPARSE handle",
        "create FP32 full FMM cuSPARSE P2P BSR descriptor");
    plan.statistics.p2p_scratch_bytes = plan.p2p_bsr_float.workspace_size;
  }
  upload(plan.entries_float, entries.data(),
         entries.size() * sizeof(FloatStaticOperatorEntry));
  upload(plan.m2m_matrices_float, data.m2m.matrices.data(),
         data.m2m.matrices.size() * sizeof(FloatStaticOperatorEntry));
  upload(plan.l2l_matrices_float, data.l2l.matrices.data(),
         data.l2l.matrices.size() * sizeof(FloatStaticOperatorEntry));
  upload(plan.m2m_interactions, data.m2m.interactions.data(),
         data.m2m.interactions.size() * sizeof(CudaTranslationInteraction));
  upload(plan.l2l_interactions, data.l2l.interactions.data(),
         data.l2l.interactions.size() * sizeof(CudaTranslationInteraction));
  if (data.has_fixed_self_indices) {
    plan.fixed_self_indices = data.fixed_self_indices;
    plan.identity_initialised = true;
    if (!plan.use_p2p_bsr && !plan.use_p2p_dictionary &&
        !data.fixed_self_indices.empty()) {
      upload(plan.self_indices, data.fixed_self_indices.data(),
             data.fixed_self_indices.size() * sizeof(int));
    }
  }
  plan.m2l_float =
      new CudaM2LExecutionPlan<float, FloatStaticM2LPlan>(
          data.m2l, plan.far_field_stream);
  check_cuda(cudaStreamSynchronize(plan.far_field_stream),
             "finish FP32 full FMM setup upload");

  plan.statistics.scalar_bytes = sizeof(float);
  plan.statistics.m2m_unique_matrix_count = data.m2m.matrix_count;
  plan.statistics.m2m_matrix_bytes =
      data.m2m.matrices.size() * sizeof(FloatStaticOperatorEntry);
  const CudaPlanStatistics &m2l_statistics = plan.m2l_float->statistics();
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
      data.l2l.matrices.size() * sizeof(FloatStaticOperatorEntry);
  plan.statistics.persistent_device_bytes =
      plan.statistics.setup_h2d_bytes + 2 * source_bytes + 3 * target_bytes +
      2 * coefficient_bytes +
      (plan.use_p2p_bsr || plan.use_p2p_dictionary ||
               data.has_fixed_self_indices
           ? 0
           : static_cast<std::size_t>(plan.target_count) * sizeof(int)) +
      plan.statistics.m2l_scratch_bytes + plan.statistics.p2p_scratch_bytes;
  plan.statistics.plan_generation_count = 1;
  plan.statistics.static_upload_count = 1;
  plan.statistics.static_m2l_upload_count = 1;
  plan.statistics.static_p2p_upload_count = 1;
  plan.statistics.p2p_interaction_count = plan.p2p_block_count;
  if (plan.use_p2p_dictionary) {
    plan.statistics.p2p_interaction_count =
        data.p2p_dictionary.token_count();
    plan.statistics.p2p_identity_bytes = 0;
  } else if (plan.use_p2p_bsr) {
    plan.statistics.p2p_tensor_bytes =
        data.p2p_bsr.values.size() * sizeof(float);
    plan.statistics.p2p_index_bytes =
        data.p2p_bsr.source_indices.size() * sizeof(int);
      plan.statistics.p2p_row_metadata_bytes =
          data.p2p_bsr.row_offsets.size() * sizeof(int);
    plan.statistics.p2p_scratch_bytes = plan.p2p_bsr_float.workspace_size;
    plan.statistics.p2p_threads_per_block = 0;
  } else {
    plan.statistics.p2p_tensor_bytes =
        data.p2p.blocks.size() * 6 * sizeof(float);
    plan.statistics.p2p_index_bytes =
        data.p2p.blocks.size() * 3 * sizeof(int);
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
  release_p2p_device_view(plan.p2p);
  release_p2p_device_view(plan.p2p_bsr);
  release_p2p_device_view(plan.p2p_dictionary);
  release_p2p_device_view(plan.p2p_float);
  release_p2p_device_view(plan.p2p_bsr_float);
  release_p2p_device_view(plan.p2p_dictionary_float);
  cudaFree(plan.entries);
  cudaFree(plan.coefficient_degrees);
  cudaFree(plan.m2m_matrices);
  cudaFree(plan.l2l_matrices);
  cudaFree(plan.m2m_interactions);
  cudaFree(plan.l2l_interactions);
  delete plan.m2l;
  cudaFree(plan.entries_float);
  cudaFree(plan.m2m_matrices_float);
  cudaFree(plan.l2l_matrices_float);
  delete plan.m2l_float;
  cudaFree(plan.moments);
  cudaFree(plan.sorted_moments);
  cudaFree(plan.multipoles);
    cudaFree(plan.locals);
    cudaFree(plan.far_fields);
    cudaFree(plan.near_fields);
    cudaFree(plan.final_fields);
    cudaFreeHost(plan.pinned_moments);
    cudaFreeHost(plan.pinned_fields);
    cudaFree(plan.moments_float);
    cudaFree(plan.sorted_moments_float);
    cudaFree(plan.multipoles_float);
    cudaFree(plan.locals_float);
    cudaFree(plan.far_fields_float);
    cudaFree(plan.near_fields_float);
    cudaFree(plan.final_fields_float);
    cudaFreeHost(plan.pinned_moments_float);
    cudaFreeHost(plan.pinned_fields_float);
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
    if (!plan.use_p2p_bsr && !plan.use_p2p_dictionary &&
        !sorted_self_indices.empty()) {
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
      permutation_range.end();
    }
    transfer_range.end();
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
    if (plan.use_p2p_dictionary) {
      launch_signed_dictionary_p2p(
          plan.p2p_dictionary, plan.sorted_moments, plan.near_fields,
          plan.near_field_stream);
    } else if (plan.use_p2p_bsr) {
      launch_bsr_p2p(plan.p2p_bsr, plan.near_fields, plan.near_field_stream);
    } else {
      launch_static_p2p(plan.p2p, plan.sorted_moments, plan.self_indices,
                        plan.near_fields, plan.near_field_stream);
    }
    p2p_range.end();
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
    p2m_range.end();
  }
  check_cuda(cudaEventRecord(plan.p2m_complete, plan.far_field_stream),
             "record P2M");
  // Kernels for one level can update parent coefficients concurrently, but a
  // parent level must not consume them early. Launching levels into one stream
  // supplies the required child-to-parent ordering without a host barrier.
  detail::ProfileRange m2m_range{"cdfmm/far_field/m2m"};
  for (int level = plan.maximum_level; level >= 1; --level) {
    const std::size_t items =
        plan.m2m_interaction_count * plan.m2m_entries_per_matrix;
    if (items != 0) {
      apply_shared_translation_kernel<<<(items + threads - 1) / threads,
                                        threads, 0, plan.far_field_stream>>>(
          plan.m2m_matrices, plan.m2m_interactions, plan.m2m_interaction_count,
          plan.m2m_entries_per_matrix, plan.coefficient_count,
          plan.coefficient_degrees, level, plan.multipoles, plan.multipoles);
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
  for (int level = 1; level <= plan.maximum_level; ++level) {
    const std::size_t items =
        plan.l2l_interaction_count * plan.l2l_entries_per_matrix;
    if (items != 0) {
      apply_shared_translation_kernel<<<(items + threads - 1) / threads,
                                        threads, 0, plan.far_field_stream>>>(
          plan.l2l_matrices, plan.l2l_interactions, plan.l2l_interaction_count,
          plan.l2l_entries_per_matrix, plan.coefficient_count,
          plan.coefficient_degrees, level, plan.locals, plan.locals);
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
    transfer_range.end();
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

void CudaFullPlan::evaluate(
    const std::span<const FloatVec3> moments,
    const std::span<FloatVec3> fields,
    const std::span<const int> sorted_self_indices) {
  auto &plan = *implementation_;
  if (!plan.fp32 ||
      moments.size() != static_cast<std::size_t>(plan.source_count) ||
      fields.size() != static_cast<std::size_t>(plan.target_count) ||
      sorted_self_indices.size() !=
          static_cast<std::size_t>(plan.target_count)) {
    throw std::invalid_argument(
        "full FP32 CUDA FMM dimensions are inconsistent");
  }
  if (!plan.identity_initialised) {
    plan.fixed_self_indices.assign(sorted_self_indices.begin(),
                                   sorted_self_indices.end());
    if (!plan.use_p2p_bsr && !plan.use_p2p_dictionary &&
        !sorted_self_indices.empty()) {
      check_cuda(cudaMemcpy(plan.self_indices, sorted_self_indices.data(),
                            sorted_self_indices.size_bytes(),
                            cudaMemcpyHostToDevice),
                 "upload FP32 static self identities");
      plan.statistics.setup_h2d_bytes += sorted_self_indices.size_bytes();
    }
    plan.identity_initialised = true;
  } else if (!std::equal(sorted_self_indices.begin(),
                         sorted_self_indices.end(),
                         plan.fixed_self_indices.begin())) {
    throw std::invalid_argument(
        "CudaFull FP32 identity map changed; rebuild the static plan");
  }
  std::copy(moments.begin(), moments.end(), plan.pinned_moments_float);
  constexpr int threads = 256;
  const auto launch_stage = [&](const int stage, const float *input,
                                float *output) {
    const std::size_t count = plan.counts[static_cast<std::size_t>(stage)];
    if (count != 0) {
      apply_entries_kernel<<<(count + threads - 1) / threads, threads, 0,
                             plan.far_field_stream>>>(
          plan.entries_float + plan.offsets[stage], count, input, output);
    }
  };

  check_cuda(cudaEventRecord(plan.evaluation_start, plan.far_field_stream),
             "record FP32 full FMM start");
  if (!moments.empty()) {
    check_cuda(cudaMemcpyAsync(plan.moments_float, plan.pinned_moments_float,
                               moments.size_bytes(), cudaMemcpyHostToDevice,
                               plan.far_field_stream),
               "upload FP32 full FMM moments");
    permute_moments_kernel<<<
        (plan.source_count + threads - 1) / threads, threads, 0,
        plan.far_field_stream>>>(plan.moments_float, plan.source_permutation,
                                plan.source_count,
                                plan.sorted_moments_float);
  }
  check_cuda(cudaEventRecord(plan.moments_ready, plan.far_field_stream),
             "record FP32 full FMM moments ready");

  check_cuda(cudaStreamWaitEvent(plan.near_field_stream, plan.moments_ready, 0),
             "wait for FP32 full FMM moments on P2P stream");
  check_cuda(cudaEventRecord(plan.p2p_start, plan.near_field_stream),
             "record FP32 full FMM P2P start");
  if (plan.use_p2p_dictionary) {
    launch_signed_dictionary_p2p(
        plan.p2p_dictionary_float, plan.sorted_moments_float,
        plan.near_fields_float, plan.near_field_stream);
  } else if (plan.use_p2p_bsr) {
    launch_bsr_p2p(plan.p2p_bsr_float, plan.near_fields_float,
                   plan.near_field_stream);
  } else {
    launch_static_p2p(plan.p2p_float, plan.sorted_moments_float,
                      plan.self_indices, plan.near_fields_float,
                      plan.near_field_stream);
  }
  check_cuda(cudaEventRecord(plan.p2p_complete, plan.near_field_stream),
             "record FP32 full FMM P2P completion");

  const std::size_t coefficient_values =
      static_cast<std::size_t>(plan.node_count) * plan.coefficient_count;
  check_cuda(cudaMemsetAsync(plan.multipoles_float, 0,
                             coefficient_values * sizeof(float),
                             plan.far_field_stream),
             "clear FP32 full FMM multipoles");
  launch_stage(plan.p2m_stage,
               reinterpret_cast<float *>(plan.sorted_moments_float),
               plan.multipoles_float);
  check_cuda(cudaEventRecord(plan.p2m_complete, plan.far_field_stream),
             "record FP32 P2M");

  for (int level = plan.maximum_level; level >= 1; --level) {
    const std::size_t items =
        plan.m2m_interaction_count * plan.m2m_entries_per_matrix;
    if (items != 0) {
      apply_shared_translation_kernel<<<
          (items + threads - 1) / threads, threads, 0,
          plan.far_field_stream>>>(
          plan.m2m_matrices_float, plan.m2m_interactions,
          plan.m2m_interaction_count, plan.m2m_entries_per_matrix,
          plan.coefficient_count, plan.coefficient_degrees, level,
          plan.multipoles_float,
          plan.multipoles_float);
    }
  }
  check_cuda(cudaEventRecord(plan.m2m_complete, plan.far_field_stream),
             "record FP32 M2M");

  check_cuda(cudaMemsetAsync(plan.locals_float, 0,
                             coefficient_values * sizeof(float),
                             plan.far_field_stream),
             "clear FP32 full FMM locals");
  plan.m2l_float->enqueue(plan.multipoles_float, plan.locals_float,
                          plan.far_field_stream,
                          plan.m2l_scale_complete);
  check_cuda(cudaEventRecord(plan.m2l_complete, plan.far_field_stream),
             "record FP32 M2L");

  for (int level = 1; level <= plan.maximum_level; ++level) {
    const std::size_t items =
        plan.l2l_interaction_count * plan.l2l_entries_per_matrix;
    if (items != 0) {
      apply_shared_translation_kernel<<<
          (items + threads - 1) / threads, threads, 0,
          plan.far_field_stream>>>(
          plan.l2l_matrices_float, plan.l2l_interactions,
          plan.l2l_interaction_count, plan.l2l_entries_per_matrix,
          plan.coefficient_count, plan.coefficient_degrees, level,
          plan.locals_float,
          plan.locals_float);
    }
  }
  check_cuda(cudaEventRecord(plan.l2l_complete, plan.far_field_stream),
             "record FP32 L2L");

  check_cuda(cudaMemsetAsync(plan.far_fields_float, 0,
                             static_cast<std::size_t>(plan.target_count) *
                                 sizeof(FloatVec3),
                             plan.far_field_stream),
             "clear FP32 far fields");
  launch_stage(plan.l2p_stage, plan.locals_float,
               reinterpret_cast<float *>(plan.far_fields_float));
  check_cuda(cudaEventRecord(plan.l2p_complete, plan.far_field_stream),
             "record FP32 L2P");

  check_cuda(cudaStreamWaitEvent(plan.far_field_stream, plan.p2p_complete, 0),
             "wait for FP32 P2P before combination");
  check_cuda(cudaEventRecord(plan.combination_start, plan.far_field_stream),
             "record FP32 combination start");
  if (plan.target_count != 0) {
    combine_order_kernel<<<
        (plan.target_count + threads - 1) / threads, threads, 0,
        plan.far_field_stream>>>(
        plan.far_fields_float, plan.near_fields_float,
        plan.target_permutation, plan.target_count, plan.final_fields_float);
  }
  check_cuda(cudaEventRecord(plan.combination_complete, plan.far_field_stream),
             "record FP32 combination");
  if (!fields.empty()) {
    check_cuda(cudaMemcpyAsync(plan.pinned_fields_float,
                               plan.final_fields_float, fields.size_bytes(),
                               cudaMemcpyDeviceToHost,
                               plan.far_field_stream),
               "download FP32 full FMM fields");
  }
  check_cuda(cudaEventRecord(plan.d2h_complete, plan.far_field_stream),
             "record FP32 field download");
  check_cuda(cudaEventSynchronize(plan.d2h_complete),
             "wait for FP32 full FMM evaluation");
  std::copy(plan.pinned_fields_float,
            plan.pinned_fields_float + fields.size(), fields.begin());

  const auto elapsed = [](const cudaEvent_t first, const cudaEvent_t second) {
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, first, second),
               "time FP32 full FMM phase");
    return static_cast<double>(milliseconds) * 1.0e-3;
  };
  plan.timings = {};
  plan.timings.h2d_seconds =
      elapsed(plan.evaluation_start, plan.moments_ready);
  plan.timings.p2m_seconds = elapsed(plan.moments_ready, plan.p2m_complete);
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

void CudaFullPlan::copy_far_fields(std::span<Vec3> fields) const {
  const auto& plan = *implementation_;
  if (fields.size() != static_cast<std::size_t>(plan.target_count)) {
    throw std::invalid_argument("diagnostic target count mismatch");
  }
  if (plan.far_fields_float != nullptr) {
    std::vector<FloatVec3> values(fields.size());
    check_cuda(cudaMemcpy(values.data(), plan.far_fields_float,
                          values.size() * sizeof(FloatVec3), cudaMemcpyDeviceToHost),
               "copy diagnostic far fields");
    for (std::size_t i = 0; i < fields.size(); ++i) {
      fields[i] = {values[i].x, values[i].y, values[i].z};
    }
  } else {
    check_cuda(cudaMemcpy(fields.data(), plan.far_fields, fields.size_bytes(),
                          cudaMemcpyDeviceToHost), "copy diagnostic far fields");
  }
}

} // namespace cdfmm
