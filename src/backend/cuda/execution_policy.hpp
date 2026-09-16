// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

#include "cdfmm/backend/execution.hpp"
#include "cdfmm/core/precision.hpp"

namespace cdfmm::cuda_policy {

/** @brief Derived list-1 P2P representation executed by the CUDA backends. */
enum class CudaP2PPacking { CanonicalRows, LeafBlock, Bsr3, SignedDictionary };

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
  bool periodic{false};
  /// A fixed target-to-source identity map is available in sorted order.
  bool fixed_identity_available{false};
  /// Explicit user options; they take precedence over the layout hint.
  bool explicit_reduced_symmetry{false};
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
  /// Estimated persistent bytes of a BSR(3) plan and the configured budget.
  std::size_t bsr_estimate_bytes{0};
  std::size_t bsr_budget_bytes{0};
};

/** @brief Concrete CUDA execution choices for one plan. */
struct CudaExecutionPolicy {
  CudaP2PPacking p2p_packing{CudaP2PPacking::CanonicalRows};
  CudaDictionaryExecutor dictionary_executor{CudaDictionaryExecutor::SourceWarp};
  /// True when the P2P packing came from the layout hint, not an explicit option.
  bool dictionary_from_layout{false};
  /// Pairs per thread of the grouped M2L kernel.
  int m2l_pairs_per_thread{8};
  /// Lane group per M2M/L2L output for levels with at most `translation_wide_outputs`.
  int translation_wide_lanes{32};
  /// Lane group per M2M/L2L output for larger levels.
  int translation_lanes{4};
  std::size_t translation_wide_outputs{65536};
};

/** @brief Resolves every CUDA strategy choice deterministically from the inputs. */
[[nodiscard]] CudaExecutionPolicy
resolve_cuda_execution_policy(const CudaExecutionPolicyInputs &inputs);

/** @brief Grouped M2L pairs per thread for one plan (used by the executor). */
[[nodiscard]] int m2l_pairs_per_thread(StaticPrecision precision,
                                       std::size_t m2l_translation_count);

/** @brief Lane group per translated output for one M2M/L2L level launch. */
[[nodiscard]] int translation_lanes_for_outputs(std::size_t outputs);

/** @brief Leaf occupancy below which the power-of-two microtile executor is chosen. */
[[nodiscard]] double dictionary_microtile_occupancy_limit();

/** @brief Leaf occupancy from which the source-warp executor is chosen. */
[[nodiscard]] double dictionary_source_warp_occupancy_limit();

[[nodiscard]] const char *name(CudaP2PPacking packing) noexcept;
[[nodiscard]] const char *name(CudaDictionaryExecutor executor) noexcept;
[[nodiscard]] const char *name(SpatialLayout layout) noexcept;

} // namespace cdfmm::cuda_policy
