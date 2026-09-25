// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <string>
#include <vector>

#include "cdfmm/backend/cuda/availability.hpp"
#include "cdfmm/core/output_flags.hpp"
#include "cdfmm/math/potential_field.hpp"
#include "cdfmm/plan/static_plan.hpp"
#include "cdfmm/timings.hpp"
#include "cdfmm/tree/static_topology.hpp"
#include "backend/cuda/common/runtime.hpp"
#include "backend/cuda/far_field/internal.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Full static FMM plan
//------------------------------------------------------------------------------

/** @brief One level of dependency-ordered static coefficient applications. */
struct CudaStaticLevel {
  std::vector<StaticOperatorEntry> entries{};
};

/** @brief Shared sparse translation classes and their node relations. */
struct CudaSharedTranslationData {
  std::vector<StaticOperatorEntry> matrices{};
  std::vector<CudaTranslationInteraction> interactions{};
  int entries_per_matrix{0};
  int matrix_count{0};
};

/** @brief FP32 shared sparse translations and node relations. */
struct FloatCudaSharedTranslationData {
  std::vector<FloatStaticOperatorEntry> matrices{};
  std::vector<CudaTranslationInteraction> interactions{};
  int entries_per_matrix{0};
  int matrix_count{0};
};

/** @brief Immutable CPU-built operators consumed by the full CUDA plan. */
struct CudaFullPlanData {
  int coefficient_count{0};
    int node_count{0};
    int source_count{0};
    int target_count{0};
  std::vector<int> source_permutation{};
  std::vector<int> target_permutation{};
  std::vector<int> coefficient_degrees{};
  std::vector<StaticOperatorEntry> p2m{};
  CudaSharedTranslationData m2m{};
  StaticM2LPlan m2l{};
  CudaSharedTranslationData l2l{};
  std::vector<StaticOperatorEntry> l2p{};
  StaticP2POperator p2p{};
  StaticP2PBsrPlan p2p_bsr{};
  StaticP2PSignedTensorDictionaryPlan p2p_dictionary{};
  StaticP2PLeafPlan p2p_leaf{};
  /** @brief Plan topology: the position-based P2P executor and the
   *  procedural point P2M/L2P read positions and leaf ranges from it during
   *  construction only. */
  const StaticFmmTopology *topology{nullptr};
  /** @brief Procedural point P2M/L2P (spherical basis, orders 1..15):
   *  recompute the operator from the positions instead of streaming the
   *  entry lists, which are then left empty. */
  bool procedural_p2m{false};
  bool procedural_l2p{false};
  int expansion_order{0};
  int p2m_lanes_per_leaf{32};
  int l2p_lanes_per_leaf{32};
  std::vector<int> fixed_self_indices{};
  bool use_p2p_bsr{false};
  bool use_p2p_dictionary{false};
  /** @brief Selects the warp-per-block dense leaf packing. */
  bool use_p2p_leaf{false};
  /** @brief Selects the position-based point executor (no pair tensors). */
  bool use_p2p_point_geometry{false};
  /** @brief Selects one-thread-per-target dictionary P2P execution. */
  bool p2p_dictionary_target_owned{false};
  /** @brief Selects power-of-two target microtiles for dictionary P2P. */
  bool p2p_dictionary_power2_microtiles{false};
  /** @brief Creates the far-field stream at the greatest stream priority. */
  bool far_field_stream_priority{false};
  bool has_fixed_self_indices{false};
};

/** @brief Immutable FP32 operators consumed by the full CUDA plan. */
struct FloatCudaFullPlanData {
  int coefficient_count{0};
  int node_count{0};
  int source_count{0};
  int target_count{0};
  std::vector<int> source_permutation{};
  std::vector<int> target_permutation{};
  std::vector<int> coefficient_degrees{};
  std::vector<FloatStaticOperatorEntry> p2m{};
  FloatCudaSharedTranslationData m2m{};
  FloatStaticM2LPlan m2l{};
  FloatCudaSharedTranslationData l2l{};
  std::vector<FloatStaticOperatorEntry> l2p{};
  FloatStaticP2POperator p2p{};
  FloatStaticP2PBsrPlan p2p_bsr{};
  FloatStaticP2PSignedTensorDictionaryPlan p2p_dictionary{};
  FloatStaticP2PLeafPlan p2p_leaf{};
  /** @brief Plan topology: the position-based P2P executor and the
   *  procedural point P2M/L2P read positions and leaf ranges from it during
   *  construction only. */
  const StaticFmmTopology *topology{nullptr};
  /** @brief Procedural point P2M/L2P (spherical basis, orders 1..15):
   *  recompute the operator from the positions instead of streaming the
   *  entry lists, which are then left empty. */
  bool procedural_p2m{false};
  bool procedural_l2p{false};
  int expansion_order{0};
  int p2m_lanes_per_leaf{32};
  int l2p_lanes_per_leaf{32};
  std::vector<int> fixed_self_indices{};
  bool use_p2p_bsr{false};
  bool use_p2p_dictionary{false};
  /** @brief Selects the warp-per-block dense leaf packing. */
  bool use_p2p_leaf{false};
  /** @brief Selects the position-based point executor (no pair tensors). */
  bool use_p2p_point_geometry{false};
  /** @brief Selects one-thread-per-target dictionary P2P execution. */
  bool p2p_dictionary_target_owned{false};
  /** @brief Selects power-of-two target microtiles for dictionary P2P. */
  bool p2p_dictionary_power2_microtiles{false};
  /** @brief Creates the far-field stream at the greatest stream priority. */
  bool far_field_stream_priority{false};
  bool has_fixed_self_indices{false};
};

/**
 * @brief Owns a complete device-resident static CUDA FMM evaluation plan.
 *
 * All geometry and operator data are uploaded during construction. The first
 * identity map fixes self interaction metadata; changing it requires rebuilding
 * the plan. A normal field evaluation transfers only moments and final fields.
 */
class CudaFullPlan {
public:
    explicit CudaFullPlan(const CudaFullPlanData& data);
    explicit CudaFullPlan(const FloatCudaFullPlanData& data);
    ~CudaFullPlan();
  CudaFullPlan(const CudaFullPlan &) = delete;
  CudaFullPlan &operator=(const CudaFullPlan &) = delete;

  void evaluate(std::span<const Vec3> moments, std::span<Vec3> fields,
                std::span<const int> sorted_self_indices);
  void evaluate(std::span<const FloatVec3> moments,
                std::span<FloatVec3> fields,
                std::span<const int> sorted_self_indices);
  void copy_far_fields(std::span<Vec3> fields) const;

  /**
   * @brief Pinned host staging buffers for one evaluation.
   *
   * Writing the scaled moments straight into `pinned_moments()` and reading
   * the user-ordered fields straight from `pinned_fields()` removes one host
   * copy on each side of the device round trip. `evaluate()` recognises the
   * aliased spans and skips its own staging copies.
   */
  [[nodiscard]] std::span<Vec3> pinned_moments() noexcept;
  [[nodiscard]] std::span<FloatVec3> pinned_moments_float() noexcept;
  [[nodiscard]] std::span<Vec3> pinned_fields() noexcept;
  [[nodiscard]] std::span<FloatVec3> pinned_fields_float() noexcept;
  [[nodiscard]] const CudaPlanStatistics &statistics() const noexcept;
  /// @brief Device lanes of the latest evaluation; populated only at `Detailed`.
  [[nodiscard]] const CudaEvaluationTimings &timings() const noexcept;
  /// @brief Level of device timing collected by later evaluations (default `Off`).
  [[nodiscard]] TimingLevel timing_level() const noexcept;
  /// @brief Selects the level of device timing; safe between evaluations.
  void set_timing_level(TimingLevel level) noexcept;

private:
    struct Implementation;
    Implementation* implementation_{nullptr};
};

} // namespace cdfmm
