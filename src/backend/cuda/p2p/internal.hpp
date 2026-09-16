// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "cdfmm/plan/static_plan.hpp"
#include "cdfmm/timings.hpp"

namespace cdfmm::cuda_p2p_detail {

inline constexpr int static_operator_threads = 256;

template <typename Scalar>
struct CudaPackedTensor6;

// CUDA 13 split double4 into explicitly aligned variants. Keep the legacy
// spelling for older toolkits, where the alignment-specific type is absent.
#if CUDART_VERSION >= 13000
using CudaDouble4 = double4_16a;
#else
using CudaDouble4 = double4;
#endif

template <>
struct alignas(16) CudaPackedTensor6<float> {
  float4 a;  // xx, xy, xz, yy
  float2 b;  // yz, zz
};

template <>
struct alignas(16) CudaPackedTensor6<double> {
  CudaDouble4 a;  // xx, xy, xz, yy
  double2 b;  // yz, zz
};

template <typename Block>
struct CudaP2PDeviceView {
  int target_count{0};
  int *row_offsets{nullptr};
  Block *blocks{nullptr};
};

template <typename Scalar>
struct CudaCompactP2PDeviceView {
  int target_count{0};
  std::size_t interaction_count{0};
  int *row_offsets{nullptr};
  int *source_indices{nullptr};
  /// Canonical per-pair identity marker (see StaticP2PCompactPlan).
  unsigned char *skip_for_identity{nullptr};
  Scalar *tensors{nullptr};
};

template <typename Scalar>
struct CudaLeafP2PDeviceView {
  int target_count{0};
  int target_leaf_count{0};
  int block_count{0};
  int threads_per_block{0};
  std::size_t interaction_count{0};
  int *target_begins{nullptr};
  int *target_counts{nullptr};
  int *leaf_row_offsets{nullptr};
  /// Target leaf of every dense block; one warp executes one block.
  int *block_target_leaves{nullptr};
  StaticP2PLeafBlock *leaf_blocks{nullptr};
  Scalar *tensors{nullptr};
};

template <typename Scalar>
struct CudaSignedDictionaryP2PDeviceView {
  int target_count{0};
  int tile_count{0};
  int microtile_count{0};
  int threads_per_block{32};
  bool target_owned{false};
  bool power2_microtiles{false};
  std::size_t variant_count{0};
  int *target_begins{nullptr};
  int *target_counts{nullptr};
  int *target_leaf_for_target{nullptr};
  int *leaf_row_offsets{nullptr};
  StaticP2PLeafBlock *leaf_blocks{nullptr};
  int *tile_leaf_indices{nullptr};
  int *tile_target_offsets{nullptr};
  int *microtile_leaf_indices{nullptr};
  int *microtile_target_offsets{nullptr};
  std::array<int, 7> microtile_class_offsets{};
  CudaPackedTensor6<Scalar> *tensors{nullptr};
  void *tokens{nullptr};
  std::uint8_t token_width_bytes{0};
};

template <typename Scalar>
struct CudaBsrP2PDeviceView {
  int target_count{0};
  int source_count{0};
  int interaction_count{0};
  int *row_offsets{nullptr};
  int *source_indices{nullptr};
  Scalar *values{nullptr};
  cusparseHandle_t handle{nullptr};
  cusparseSpMatDescr_t descriptor{nullptr};
  cusparseConstDnVecDescr_t input_descriptor{nullptr};
  cusparseDnVecDescr_t output_descriptor{nullptr};
  void *workspace{nullptr};
  std::size_t workspace_size{0};
};

void launch_static_p2p(const CudaP2PDeviceView<StaticDipoleBlock> &plan,
                       const Vec3 *moments, const int *self_indices,
                       Vec3 *fields, cudaStream_t stream);
void launch_static_p2p(
    const CudaP2PDeviceView<FloatStaticDipoleBlock> &plan,
    const FloatVec3 *moments, const int *self_indices, FloatVec3 *fields,
    cudaStream_t stream);

void launch_compact_p2p(const CudaCompactP2PDeviceView<double> &plan,
                        const Vec3 *moments, const int *self_indices,
                        Vec3 *fields, cudaStream_t stream);
void launch_compact_p2p(const CudaCompactP2PDeviceView<float> &plan,
                        const FloatVec3 *moments, const int *self_indices,
                        FloatVec3 *fields, cudaStream_t stream);

void launch_leaf_p2p(const CudaLeafP2PDeviceView<double> &plan,
                     const Vec3 *moments, const int *self_indices,
                     Vec3 *fields, cudaStream_t stream);
void launch_leaf_p2p(const CudaLeafP2PDeviceView<float> &plan,
                     const FloatVec3 *moments, const int *self_indices,
                     FloatVec3 *fields, cudaStream_t stream);

void launch_signed_dictionary_p2p(
    const CudaSignedDictionaryP2PDeviceView<double> &plan,
    const Vec3 *moments, Vec3 *fields, cudaStream_t stream);
void launch_signed_dictionary_p2p(
    const CudaSignedDictionaryP2PDeviceView<float> &plan,
    const FloatVec3 *moments, FloatVec3 *fields, cudaStream_t stream);

void upload_cuda_signed_dictionary(
    const StaticP2PSignedTensorDictionaryPlan &host,
    CudaSignedDictionaryP2PDeviceView<double> &device,
    CudaPlanStatistics &statistics, bool target_owned,
    bool power2_microtiles);

void upload_cuda_canonical(
    const StaticP2POperator &host,
    CudaP2PDeviceView<StaticDipoleBlock> &device,
    CudaPlanStatistics &statistics, cudaStream_t stream,
    const char *allocation_operation, const char *upload_operation);
void upload_cuda_canonical(
    const FloatStaticP2POperator &host,
    CudaP2PDeviceView<FloatStaticDipoleBlock> &device,
    CudaPlanStatistics &statistics, cudaStream_t stream,
    const char *allocation_operation, const char *upload_operation);

void upload_cuda_leaf(
    const StaticP2PLeafPlan &host, CudaLeafP2PDeviceView<double> &device,
    CudaPlanStatistics &statistics, const char *allocation_operation,
    const char *upload_operation);
void upload_cuda_leaf(
    const FloatStaticP2PLeafPlan &host, CudaLeafP2PDeviceView<float> &device,
    CudaPlanStatistics &statistics, const char *allocation_operation,
    const char *upload_operation);

void upload_cuda_bsr(
    const StaticP2PBsrPlan &host, CudaBsrP2PDeviceView<double> &device,
    const Vec3 *input_values, Vec3 *output_values,
    CudaPlanStatistics &statistics, cudaStream_t stream,
    const char *allocation_operation, const char *upload_operation,
    const char *handle_operation, const char *descriptor_operation);
void upload_cuda_bsr(
    const FloatStaticP2PBsrPlan &host, CudaBsrP2PDeviceView<float> &device,
    const FloatVec3 *input_values, FloatVec3 *output_values,
    CudaPlanStatistics &statistics, cudaStream_t stream,
    const char *allocation_operation, const char *upload_operation,
    const char *handle_operation, const char *descriptor_operation);
void upload_cuda_signed_dictionary(
    const FloatStaticP2PSignedTensorDictionaryPlan &host,
    CudaSignedDictionaryP2PDeviceView<float> &device,
    CudaPlanStatistics &statistics, bool target_owned,
    bool power2_microtiles);

void initialise_bsr_p2p_resources(
    CudaBsrP2PDeviceView<double> &plan, const Vec3 *input_values,
    Vec3 *output_values, const char *handle_operation,
    const char *descriptor_operation);
void initialise_bsr_p2p_resources(
    CudaBsrP2PDeviceView<float> &plan, const FloatVec3 *input_values,
    FloatVec3 *output_values, const char *handle_operation,
    const char *descriptor_operation);
void launch_bsr_p2p(const CudaBsrP2PDeviceView<double> &plan,
                    Vec3 *fields, cudaStream_t stream);
void launch_bsr_p2p(const CudaBsrP2PDeviceView<float> &plan,
                    FloatVec3 *fields, cudaStream_t stream);

void release_p2p_device_view(
    CudaP2PDeviceView<StaticDipoleBlock> &plan) noexcept;
void release_p2p_device_view(
    CudaP2PDeviceView<FloatStaticDipoleBlock> &plan) noexcept;
void release_p2p_device_view(
    CudaCompactP2PDeviceView<double> &plan) noexcept;
void release_p2p_device_view(
    CudaCompactP2PDeviceView<float> &plan) noexcept;
void release_p2p_device_view(CudaLeafP2PDeviceView<double> &plan) noexcept;
void release_p2p_device_view(CudaLeafP2PDeviceView<float> &plan) noexcept;
void release_p2p_device_view(
    CudaSignedDictionaryP2PDeviceView<double> &plan) noexcept;
void release_p2p_device_view(
    CudaSignedDictionaryP2PDeviceView<float> &plan) noexcept;
void release_p2p_device_view(CudaBsrP2PDeviceView<double> &plan) noexcept;
void release_p2p_device_view(CudaBsrP2PDeviceView<float> &plan) noexcept;

} // namespace cdfmm::cuda_p2p_detail
