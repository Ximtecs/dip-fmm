// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifdef CDFMM_ENABLE_CUDA
#include <cuda_runtime_api.h>
#else
// Keep the internal declaration header usable by the structured non-CUDA
// full-FMM stub without requiring a CUDA toolkit during host-only builds.
using cudaStream_t = void *;
#endif

#include <span>
#include <vector>

#include "cdfmm/plan/static_coefficient.hpp"
#include "cdfmm/timings.hpp"
#include "cdfmm/tree/static_topology.hpp"

namespace cdfmm {

/** @brief One compact node-to-node translation using a shared matrix. */
struct CudaTranslationInteraction {
  int source_node{0};
  int target_node{0};
  int matrix_id{0};
  int level{0};
};

namespace cuda_far_field_detail {

inline constexpr int far_field_threads = 256;

/** @brief Host view of immutable CUDA P2M/M2M/L2L/L2P execution data. */
template <typename Entry> struct CudaFarFieldStaticData {
  int coefficient_count{0};
  int maximum_level{0};
  std::span<const int> coefficient_degrees{};
  std::span<const Entry> p2m{};
  std::span<const Entry> l2p{};
  std::span<const Entry> m2m_matrices{};
  std::span<const CudaTranslationInteraction> m2m_interactions{};
  int m2m_entries_per_matrix{0};
  int m2m_matrix_count{0};
  std::span<const Entry> l2l_matrices{};
  std::span<const CudaTranslationInteraction> l2l_interactions{};
  int l2l_entries_per_matrix{0};
  int l2l_matrix_count{0};
  /// Procedural point P2M/L2P: when set, the corresponding entry span is
  /// ignored and the executor recomputes the operator from the positions of
  /// `topology` during every evaluation (spherical basis, orders 1..15).
  bool procedural_p2m{false};
  bool procedural_l2p{false};
  int expansion_order{0};
  int p2m_lanes_per_leaf{32};
  int l2p_lanes_per_leaf{32};
  const StaticFmmTopology *topology{nullptr};
};

/**
 * @brief Persistent CUDA executor for the non-M2L far-field passes.
 *
 * Immutable operator entries, shared translation tables, and degree metadata
 * are uploaded once. Evaluation buffers remain owned by the complete FMM
 * plan and are supplied to each enqueue operation.
 */
template <typename Scalar, typename Entry> class CudaFarFieldExecutionPlan {
public:
  CudaFarFieldExecutionPlan(const CudaFarFieldStaticData<Entry> &data,
                            cudaStream_t stream);
  ~CudaFarFieldExecutionPlan();

  CudaFarFieldExecutionPlan(const CudaFarFieldExecutionPlan &) = delete;
  CudaFarFieldExecutionPlan &
  operator=(const CudaFarFieldExecutionPlan &) = delete;

  void enqueue_p2m(const Scalar *input, Scalar *output,
                   cudaStream_t stream) const;
  void enqueue_m2m(const Scalar *input, Scalar *output,
                   cudaStream_t stream) const;
  void enqueue_l2l(const Scalar *input, Scalar *output,
                   cudaStream_t stream) const;
  void enqueue_l2p(const Scalar *input, Scalar *output,
                   cudaStream_t stream) const;

  [[nodiscard]] const CudaPlanStatistics &statistics() const noexcept;

private:
  struct Implementation;
  Implementation *implementation_{nullptr};
};

extern template class CudaFarFieldExecutionPlan<double, StaticOperatorEntry>;
extern template class CudaFarFieldExecutionPlan<float,
                                                FloatStaticOperatorEntry>;

} // namespace cuda_far_field_detail
} // namespace cdfmm
