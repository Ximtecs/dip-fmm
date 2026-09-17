// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "cdfmm/backend/execution.hpp"
#include "cdfmm/core/precision.hpp"

namespace cdfmm::cuda_policy {

/** @brief Derived list-1 P2P representation executed by the CUDA backends. */
enum class CudaP2PPacking {
  CanonicalRows,
  LeafBlock,
  Bsr3,
  SignedDictionary,
  /// Point-dipole pairs recomputed from the resident sorted positions; the
  /// CUDA counterpart of `P2PExecutionPacking::PointGeometry`.
  PointGeometry
};

/** @brief Kernel used for the signed tensor-dictionary packing. */
enum class CudaDictionaryExecutor { SourceWarp, TargetOwned, PowerOfTwoMicrotiles };

/**
 * @brief Plan and option facts the CUDA execution policy is resolved from.
 *
 * Every quantity is available once the topology and options exist; nothing is
 * measured at run time. Counts describe the constructed tree, not the user's
 * nominal N and depth.
 */
struct CudaExecutionPolicyInputs {
  StaticPrecision precision{StaticPrecision::Float32};
  SpatialLayout spatial_layout{SpatialLayout::General};
  /// True when a CUDA backend executes P2P; CPU backends keep their own policy.
  bool cuda_backend{false};
  /// Point-dipole near-field sources (geometry or near-field model).
  bool effective_point_source{true};
  /// Point near-field targets (geometry or near-field model).
  bool effective_point_target{true};
  bool periodic{false};
  /// A fixed target-to-source identity map is available in sorted order.
  bool fixed_identity_available{false};
  /// Explicit user options; they take precedence over the layout hint.
  bool explicit_reduced_symmetry{false};
  /// Explicit packing request (`UniformFmmOptions::p2p_packing`), already
  /// accepted by `explicit_packing_rejection`; resolution honours it verbatim.
  std::optional<CudaP2PPacking> explicit_packing{};
  bool explicit_dictionary_target_owned{false};
  bool explicit_dictionary_power2_microtiles{false};
  std::size_t source_count{0};
  std::size_t target_count{0};
  int expansion_order{0};
  int coefficient_count{0};
  int tree_depth{0};
  std::size_t occupied_target_leaf_count{0};
  /// Targets per occupied target leaf, from the constructed topology.
  double mean_leaf_occupancy{0.0};
  std::size_t p2p_pair_count{0};
  std::size_t m2l_translation_count{0};
  /// Estimated persistent bytes of a BSR(3) plan and the configured budget
  /// (diagnostic only since leaf blocks became the general default).
  std::size_t bsr_estimate_bytes{0};
  std::size_t bsr_budget_bytes{0};
};

/** @brief Concrete CUDA execution choices for one plan. */
struct CudaExecutionPolicy {
  CudaP2PPacking p2p_packing{CudaP2PPacking::CanonicalRows};
  CudaDictionaryExecutor dictionary_executor{CudaDictionaryExecutor::SourceWarp};
  /// True when the P2P packing came from the layout hint, not an explicit
  /// option; such a dictionary is kept only if the built plan compresses
  /// (token width of at most two bytes, i.e. at most 65535 variants).
  bool dictionary_from_layout{false};
  /// Pairs per thread of the grouped M2L kernel.
  int m2l_pairs_per_thread{8};
  /// Create the device-resident backend's far-field stream at the greatest
  /// stream priority so its short latency-bound kernels are scheduled ahead
  /// of the concurrent P2P kernel (see `far_field_stream_priority`).
  bool far_field_stream_priority{true};
  /// Lane group per M2M/L2L output for levels with at most `translation_wide_outputs`.
  int translation_wide_lanes{32};
  /// Lane group per M2M/L2L output for larger levels.
  int translation_lanes{4};
  std::size_t translation_wide_outputs{65536};
};

/** @brief Resolves every CUDA strategy choice deterministically from the inputs. */
[[nodiscard]] CudaExecutionPolicy
resolve_cuda_execution_policy(const CudaExecutionPolicyInputs &inputs);

/**
 * @brief Explains why an explicitly requested packing cannot execute a plan.
 *
 * Returns nullptr when the packing is valid for the plan facts. The reasons
 * are representational (periodic image records, identity handling baked in at
 * construction), never the geometry that produced the tensors.
 */
[[nodiscard]] const char *
explicit_packing_rejection(const CudaExecutionPolicyInputs &inputs,
                           CudaP2PPacking packing) noexcept;

/** @brief Grouped M2L pairs per thread for one plan (used by the executor). */
[[nodiscard]] int m2l_pairs_per_thread(StaticPrecision precision,
                                       std::size_t m2l_translation_count);

/** @brief Lane group per translated output for one M2M/L2L level launch. */
[[nodiscard]] int translation_lanes_for_outputs(std::size_t outputs);

/**
 * @brief Estimated device time per list-1 pair of one P2P packing in
 *        picoseconds, measured on the RTX 5090 (Phase 3A/3B.5 records).
 *
 * Only ratios against the M2L estimate matter; the values are the kernel
 * times of the 128-points-per-leaf random workload divided by its pair count.
 */
[[nodiscard]] double p2p_picoseconds_per_pair(CudaP2PPacking packing,
                                              StaticPrecision precision) noexcept;

/**
 * @brief Stream-priority rule for a plan whose P2P packing and precision are
 *        known (see the three-argument overload for the rule itself).
 */
[[nodiscard]] bool far_field_stream_priority(std::size_t p2p_pair_count,
                                             std::size_t m2l_translation_count,
                                             int coefficient_count,
                                             CudaP2PPacking packing,
                                             StaticPrecision precision) noexcept;

/**
 * @brief Whether the full backend's far-field stream should outrank its P2P
 * stream, from the estimated device costs of the two branches.
 *
 * This overload assumes the FP32 leaf-block kernel cost (the Phase-3 P2P
 * unification calibration); the packing-aware overload above is what the
 * resolved policy uses.
 *
 * Measured on the RTX 5090 (Phase-3 P2P unification): with the far field
 * prioritised, evaluations were 4-10 % faster wherever the far field is
 * shorter than a few times the P2P kernel (its many short kernels no longer
 * stretch behind the SM-saturating P2P kernel, which absorbs the delay), but
 * 3 % slower when a deep tree makes the far field several times longer than
 * P2P (the small P2P kernel is then starved to the very end instead of hiding
 * inside the far field). The rule keeps equal priorities in that regime.
 */
[[nodiscard]] bool far_field_stream_priority(std::size_t p2p_pair_count,
                                             std::size_t m2l_translation_count,
                                             int coefficient_count) noexcept;

/** @brief Leaf occupancy below which the power-of-two microtile executor is chosen. */
[[nodiscard]] double dictionary_microtile_occupancy_limit();

/** @brief Leaf occupancy from which the source-warp executor is chosen. */
[[nodiscard]] double dictionary_source_warp_occupancy_limit();

/**
 * @brief Largest token width (bytes) a layout-selected dictionary may have.
 *
 * Measured on 32768 irregular bodies (3.47M variants, four-byte tokens): the
 * CUDA dictionary kernels were 1.9-3.6x slower than leaf blocks, while a
 * lattice (187-344 variants, one-byte tokens) was 3x faster; two-byte tokens
 * keep the dictionary within about 1.5 MB, which streams from cache.
 */
[[nodiscard]] std::uint8_t dictionary_layout_max_token_width_bytes() noexcept;

/**
 * @brief Lane-group width of the procedural point P2M/L2P kernels.
 *
 * A group of this many consecutive lanes (a power of two from 1 to 32) owns
 * one leaf, so a warp holds several small leaves instead of idling; the width
 * is the smallest power of two covering the mean leaf occupancy.
 */
[[nodiscard]] int procedural_lanes_per_leaf(double mean_leaf_occupancy) noexcept;

[[nodiscard]] const char *name(CudaP2PPacking packing) noexcept;
[[nodiscard]] const char *name(CudaDictionaryExecutor executor) noexcept;
[[nodiscard]] const char *name(SpatialLayout layout) noexcept;

} // namespace cdfmm::cuda_policy
