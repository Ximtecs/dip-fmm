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

// Device-cost estimates behind the stream-priority rule: about 18 ps per
// list-1 pair for the leaf-block kernel (3.95 ms for 192M pairs) and 0.22 ps
// per M2L multiply-add (translations x coefficient_count^2; 2.6 ms for 4.9M
// translations of 49 coefficients), both from the FP32 measurements of the
// Phase-3 P2P unification. Only their ratio matters.
constexpr double p2p_picoseconds_per_pair = 18.0;
constexpr double m2l_picoseconds_per_flop = 0.22;
constexpr double far_field_dominance_ratio = 3.0;

} // namespace

bool far_field_stream_priority(const std::size_t p2p_pair_count,
                               const std::size_t m2l_translation_count,
                               const int coefficient_count) noexcept {
  const double p2p_picoseconds =
      p2p_picoseconds_per_pair * static_cast<double>(p2p_pair_count);
  const double far_field_picoseconds =
      m2l_picoseconds_per_flop * static_cast<double>(m2l_translation_count) *
      static_cast<double>(coefficient_count) *
      static_cast<double>(coefficient_count);
  return far_field_picoseconds < far_field_dominance_ratio * p2p_picoseconds;
}

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

std::uint8_t dictionary_layout_max_token_width_bytes() noexcept { return 2; }

const char *
explicit_packing_rejection(const CudaExecutionPolicyInputs &inputs,
                           const CudaP2PPacking packing) noexcept {
  // Periodic image records are ordinary dense leaf pairs (tagged with their
  // image ordinal) or merged sparse blocks, so no packing depends on
  // periodicity; only identity handling baked in at construction matters.
  const bool identity_baked_in =
      inputs.effective_point_source && !inputs.fixed_identity_available;
  switch (packing) {
  case CudaP2PPacking::CanonicalRows:
  case CudaP2PPacking::LeafBlock:
    return nullptr;
  case CudaP2PPacking::Bsr3:
    if (identity_baked_in) {
      return "CudaBsr3 bakes the point-source self exclusion into its values "
             "at construction; supply fixed_target_source_indices or select "
             "CanonicalAos or LeafBlock";
    }
    return nullptr;
  case CudaP2PPacking::SignedDictionary:
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
  policy.far_field_stream_priority = far_field_stream_priority(
      inputs.p2p_pair_count, inputs.m2l_translation_count,
      inputs.coefficient_count);

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
  // variant, so a point source needs a fixed identity map; a finite source's
  // self field is physical and needs nothing. Periodic image records are
  // dense leaf pairs like any other, so periodicity does not restrict any
  // stored-tensor packing. The regular-grid hint selects the dictionary for
  // any geometry on every backend: a lattice of identical bodies has as few
  // distinct tensors as a point lattice (248 prism / 187 tetrahedron variants
  // for 6.2M pairs) and the same executors, measured 2-7.5x faster than the
  // SoA rows on the CPU and 3x faster than leaf blocks on CUDA. Whether the
  // derived dictionary actually compresses is checked on the built plan
  // (token width), and a poorly compressing hint falls back to the defaults.
  const bool dictionary_valid =
      !inputs.effective_point_source || inputs.fixed_identity_available;
  const bool layout_dictionary =
      inputs.spatial_layout == SpatialLayout::RegularGrid && dictionary_valid;

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

  // General default: dense leaf blocks for every geometry. Phase 3A chose
  // them for point sources; the P2P unification measured them against BSR(3)
  // on finite bodies too (117 vs 191 us on a 32^3 prism/tetrahedron lattice,
  // 142 vs 217 us on irregular bodies, 8 bodies per leaf, FP32) once the leaf
  // packing carried the identity metadata, so the geometry-based BSR default
  // is gone. BSR(3) and canonical rows remain explicit packings; periodic
  // plans follow the same rule since their image records pack identically.
  policy.p2p_packing = CudaP2PPacking::LeafBlock;
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
