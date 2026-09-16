// SPDX-License-Identifier: Apache-2.0

#include "backend/cuda/execution_policy.hpp"

namespace cdfmm::cuda_policy {

namespace {

// Measured on an RTX 5090 (agent_docs/performance_optimization.md, Phase 3A
// and the policy calibration). Thresholds are plan quantities, never timings.

// FP32 plans with this many translations keep the GPU full with 16 pairs per
// thread, which halves the shared-memory matrix loads per FMA; smaller FP32
// plans and all FP64 plans (register pressure) use 8.
constexpr std::size_t m2l_wide_pair_translations = 250000;
constexpr int m2l_wide_pairs = 16;
constexpr int m2l_narrow_pairs = 8;

// Translation levels with few outputs are latency-bound and take a whole
// warp per output; large levels are throughput-bound and take four lanes.
constexpr std::size_t translation_wide_outputs = 65536;
constexpr int translation_wide_lanes = 32;
constexpr int translation_narrow_lanes = 4;

// Regular-grid dictionary executor, three occupancy regimes. Below 48
// targets per leaf a warp tile leaves lanes idle and the power-of-two
// microtile kernel wins (best or tied at 8-32 per leaf); from 48 to 64 per
// leaf the target-owned kernel is best; from 80 per leaf upwards the
// source-warp kernel wins in both precisions (1.1-2.1x over target-owned at
// 80-192 per leaf). Calibrated on lattices of 24k-262k points; see
// agent_docs/performance_optimization.md.
constexpr double dictionary_microtile_occupancy = 48.0;
constexpr double dictionary_source_warp_occupancy = 72.0;

} // namespace

int m2l_pairs_per_thread(const StaticPrecision precision,
                         const std::size_t m2l_translation_count) {
  return (precision == StaticPrecision::Float32 &&
          m2l_translation_count >= m2l_wide_pair_translations)
             ? m2l_wide_pairs
             : m2l_narrow_pairs;
}

int translation_lanes_for_outputs(const std::size_t outputs) {
  return outputs <= translation_wide_outputs ? translation_wide_lanes
                                             : translation_narrow_lanes;
}

double dictionary_microtile_occupancy_limit() {
  return dictionary_microtile_occupancy;
}

double dictionary_source_warp_occupancy_limit() {
  return dictionary_source_warp_occupancy;
}

const char *
explicit_packing_rejection(const CudaExecutionPolicyInputs &inputs,
                           const CudaP2PPacking packing) noexcept {
  const bool identity_baked_in =
      inputs.effective_point_source && !inputs.fixed_identity_available;
  switch (packing) {
  case CudaP2PPacking::CanonicalRows:
    return nullptr;
  case CudaP2PPacking::LeafBlock:
    if (inputs.periodic) {
      return "LeafBlock packs one dense block per (target leaf, source leaf) "
             "pair of the canonical rows and cannot represent periodic image "
             "records; periodic plans use CanonicalAos";
    }
    return nullptr;
  case CudaP2PPacking::Bsr3:
    if (inputs.periodic) {
      return "CudaBsr3 stores one sparse block per (target, source) pair and "
             "cannot represent periodic image records; periodic plans use "
             "CanonicalAos";
    }
    if (identity_baked_in) {
      return "CudaBsr3 bakes the point-source self exclusion into its values "
             "at construction; supply fixed_target_source_indices or select "
             "CanonicalAos or LeafBlock";
    }
    return nullptr;
  case CudaP2PPacking::SignedDictionary:
    if (inputs.periodic) {
      return "TensorDictionary packs dense leaf pairs of the canonical rows "
             "and cannot represent periodic image records; periodic plans use "
             "CanonicalAos";
    }
    if (identity_baked_in) {
      return "TensorDictionary encodes point-source self pairs as its zero "
             "variant at construction; supply fixed_target_source_indices or "
             "select CanonicalAos or LeafBlock";
    }
    return nullptr;
  }
  return "unknown CUDA P2P packing";
}

CudaExecutionPolicy
resolve_cuda_execution_policy(const CudaExecutionPolicyInputs &inputs) {
  CudaExecutionPolicy policy;
  policy.m2l_pairs_per_thread =
      m2l_pairs_per_thread(inputs.precision, inputs.m2l_translation_count);
  policy.translation_wide_lanes = translation_wide_lanes;
  policy.translation_lanes = translation_narrow_lanes;
  policy.translation_wide_outputs = translation_wide_outputs;

  if (inputs.explicit_packing.has_value()) {
    // An explicit request is honoured verbatim; the caller has already
    // rejected packings the plan cannot execute. The executor flags keep
    // their documented meaning for the dictionary.
    policy.p2p_packing = *inputs.explicit_packing;
    policy.dictionary_executor =
        inputs.explicit_dictionary_target_owned
            ? CudaDictionaryExecutor::TargetOwned
            : (inputs.explicit_dictionary_power2_microtiles
                   ? CudaDictionaryExecutor::PowerOfTwoMicrotiles
                   : CudaDictionaryExecutor::SourceWarp);
    return policy;
  }

  // The signed dictionary encodes fixed point-source self pairs as the zero
  // variant and has no periodic image support, so it needs a point source
  // with a fixed identity map on a non-periodic plan. Finite sources may use
  // it through the explicit option only (their self fields are physical).
  const bool dictionary_valid =
      !inputs.periodic &&
      (!inputs.effective_point_source || inputs.fixed_identity_available);
  const bool layout_dictionary =
      inputs.cuda_backend && inputs.spatial_layout == SpatialLayout::RegularGrid &&
      inputs.effective_point_source && dictionary_valid;

  if (inputs.explicit_reduced_symmetry && dictionary_valid) {
    // Explicit request: the executor flags keep their documented meaning
    // (target-owned wins over power-of-two; neither means the source-warp
    // kernel).
    policy.p2p_packing = CudaP2PPacking::SignedDictionary;
    policy.dictionary_executor =
        inputs.explicit_dictionary_target_owned
            ? CudaDictionaryExecutor::TargetOwned
            : (inputs.explicit_dictionary_power2_microtiles
                   ? CudaDictionaryExecutor::PowerOfTwoMicrotiles
                   : CudaDictionaryExecutor::SourceWarp);
    return policy;
  }
  if (layout_dictionary) {
    policy.p2p_packing = CudaP2PPacking::SignedDictionary;
    policy.dictionary_from_layout = true;
    if (inputs.explicit_dictionary_target_owned) {
      policy.dictionary_executor = CudaDictionaryExecutor::TargetOwned;
    } else if (inputs.explicit_dictionary_power2_microtiles) {
      policy.dictionary_executor = CudaDictionaryExecutor::PowerOfTwoMicrotiles;
    } else {
      if (inputs.mean_leaf_occupancy < dictionary_microtile_occupancy) {
        policy.dictionary_executor =
            CudaDictionaryExecutor::PowerOfTwoMicrotiles;
      } else if (inputs.mean_leaf_occupancy <
                 dictionary_source_warp_occupancy) {
        policy.dictionary_executor = CudaDictionaryExecutor::TargetOwned;
      } else {
        policy.dictionary_executor = CudaDictionaryExecutor::SourceWarp;
      }
    }
    return policy;
  }

  // General defaults (Phase 3A): dense leaf blocks for non-periodic point
  // sources; BSR(3) within its budget for finite sources or fixed-identity
  // point sources, otherwise canonical rows.
  if (inputs.effective_point_source && !inputs.periodic) {
    policy.p2p_packing = CudaP2PPacking::LeafBlock;
    return policy;
  }
  const bool bsr_identity_compatible =
      !inputs.effective_point_source || inputs.fixed_identity_available;
  if (!inputs.periodic && bsr_identity_compatible &&
      inputs.bsr_estimate_bytes <= inputs.bsr_budget_bytes) {
    policy.p2p_packing = CudaP2PPacking::Bsr3;
    return policy;
  }
  policy.p2p_packing = CudaP2PPacking::CanonicalRows;
  return policy;
}

const char *name(const CudaP2PPacking packing) noexcept {
  switch (packing) {
  case CudaP2PPacking::CanonicalRows:
    return "canonical_rows";
  case CudaP2PPacking::LeafBlock:
    return "leaf_block";
  case CudaP2PPacking::Bsr3:
    return "cusparse_bsr3";
  case CudaP2PPacking::SignedDictionary:
    return "signed_dictionary";
  }
  return "unknown";
}

const char *name(const CudaDictionaryExecutor executor) noexcept {
  switch (executor) {
  case CudaDictionaryExecutor::SourceWarp:
    return "source_warp";
  case CudaDictionaryExecutor::TargetOwned:
    return "target_owned";
  case CudaDictionaryExecutor::PowerOfTwoMicrotiles:
    return "power2_microtiles";
  }
  return "unknown";
}

const char *name(const SpatialLayout layout) noexcept {
  switch (layout) {
  case SpatialLayout::General:
    return "general";
  case SpatialLayout::RegularGrid:
    return "regular_grid";
  }
  return "unknown";
}

} // namespace cdfmm::cuda_policy
