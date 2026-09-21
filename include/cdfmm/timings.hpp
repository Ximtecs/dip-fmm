// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

#include "cdfmm/core/timing.hpp"
#include "cdfmm/tree/statistics.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Public timing types
//------------------------------------------------------------------------------

/**
 * @brief Wall-clock breakdown for one or more dipole evaluations.
 *
 * Which fields are populated depends on `timing_level`:
 *
 * - `TimingLevel::Off`: nothing is measured; every field keeps its default
 *   and `evaluations` still counts the evaluations represented.
 * - `TimingLevel::Coarse`: `total` always; `far_field` and `p2p` when the
 *   corresponding branch runs on the CPU (they overlap on the CUDA backends
 *   and are then left uncollected); `cuda_p2p_wait` for the hybrid backend.
 * - `TimingLevel::Detailed`: every phase below, including the CUDA lanes.
 *
 * `far_field` is the wall time of the complete CPU hierarchy (moment
 * preparation through L2P); the per-phase fields partition it.  `m2l_scale`,
 * `m2l_gather`, `m2l_multiply` and `m2l_scatter` partition `m2l`.  The CUDA
 * lanes run on their own streams and overlap the host phases: compare them
 * with `total`, never add them to it.
 */
struct EvaluationTimings {
    /// @brief Time spent permuting moments into Morton order.
    PhaseTiming moment_permutation{};
    /// @brief Time spent clearing multipole coefficients.
    PhaseTiming multipole_reset{};
    /// @brief Time spent accumulating leaf multipoles.
    PhaseTiming p2m{};
    /// @brief Time spent translating child multipoles to their parents.
    PhaseTiming m2m{};
    /// @brief Time spent clearing local coefficients.
    PhaseTiming local_reset{};
    /// @brief Time spent translating parent locals to their children.
    PhaseTiming l2l{};
    /// @brief Time spent translating well-separated multipoles to locals.
    PhaseTiming m2l{};
    /// @brief Time spent pre-scaling multipoles for CUDA M2L execution.
    PhaseTiming m2l_scale{};
    /// @brief Time spent packing source multipoles for grouped M2L.
    PhaseTiming m2l_gather{};
    /// @brief Time spent applying grouped dense M2L matrices.
    PhaseTiming m2l_multiply{};
    /// @brief Time spent accumulating grouped results into target locals.
    PhaseTiming m2l_scatter{};
    /// @brief Time spent evaluating local expansions at targets.
    PhaseTiming l2p{};
    /// @brief Time spent evaluating direct near-field interactions.
    PhaseTiming p2p{};
    /// @brief Time spent restoring results to user target order.
    PhaseTiming result_unpermutation{};
    /// @brief Device-stream time spent uploading per-evaluation CUDA inputs.
    PhaseTiming cuda_h2d{};
    /// @brief Device-stream time spent executing the CUDA evaluation kernel.
    PhaseTiming cuda_kernel{};
    /// @brief Device-stream time spent downloading requested CUDA outputs.
    PhaseTiming cuda_d2h{};
    /// @brief Device-stream time spent uploading multipoles for CUDA M2L.
    PhaseTiming cuda_m2l_h2d{};
    /// @brief Device-stream time spent downloading locals from CUDA M2L.
    PhaseTiming cuda_m2l_d2h{};
    /// @brief Device-stream time spent uploading dynamic CUDA P2P inputs.
    PhaseTiming cuda_p2p_h2d{};
    /// @brief Device-stream time spent applying the static CUDA P2P tensor.
    PhaseTiming cuda_p2p_kernel{};
    /// @brief Device-stream time spent downloading CUDA P2P fields.
    PhaseTiming cuda_p2p_d2h{};
    /// @brief Host time spent at the single final CUDA P2P synchronisation.
    PhaseTiming cuda_p2p_wait{};
    /// @brief Wall time of the complete CPU far-field hierarchy (`Coarse`).
    PhaseTiming far_field{};
    /// @brief Total complete-evaluation time.
    PhaseTiming total{};
    /// @brief Number of complete evaluations represented by these timings.
    std::uint64_t evaluations{0};
    /// @brief Level at which these timings were collected; `Off` means that
    /// every timing field is an uncollected default, not a measurement.
    TimingLevel timing_level{TimingLevel::Off};
};

/**
 * @brief One-time cost and storage of the immutable static CPU plan.
 *
 * Byte, count and cache-hit fields are always populated.  The `PhaseTiming`
 * fields follow `timing_level`: none at `TimingLevel::Off`, `total_setup` and
 * `total` at `TimingLevel::Coarse`, every construction subphase at
 * `TimingLevel::Detailed`.  The tree's own `TreeBuildTimings` are copied into
 * `tree_construction` only at `Detailed`.
 */
struct StaticPlanStatistics {
    /// @brief Level at which the timing fields below were collected.
    TimingLevel timing_level{TimingLevel::Off};
    /// @brief Time spent converting physical inputs to the canonical cube.
    PhaseTiming normalisation{};
    /// @brief Time spent constructing the immutable uniform tree.
    PhaseTiming tree_construction{};
    /// @brief Time spent extracting canonical static topology and geometry.
    PhaseTiming topology_construction{};
    /// @brief Time spent locating and validating the universal cache file.
    PhaseTiming universal_cache_lookup{};
    /// @brief Time spent loading a valid universal operator bank.
    PhaseTiming universal_cache_load{};
    /// @brief Time spent analytically building universal translations.
    PhaseTiming universal_operator_build{};
    /// @brief Time spent serialising universal and periodic operator files.
    PhaseTiming universal_cache_write{};
    /// @brief Time spent locating and validating the periodic operator file.
    PhaseTiming periodic_cache_lookup{};
    /// @brief Time spent loading a valid periodic root operator.
    PhaseTiming periodic_cache_load{};
    /// @brief Time spent analytically building a periodic root operator.
    PhaseTiming periodic_operator_build{};
    /// @brief Time spent hashing canonical geometry and plan options.
    PhaseTiming geometry_hash{};
    /// @brief Time spent locating and validating the geometry cache file.
    PhaseTiming geometry_cache_lookup{};
    /// @brief Time spent loading a valid complete geometry plan.
    PhaseTiming geometry_cache_load{};
    /// @brief Time spent serialising a complete geometry plan.
    PhaseTiming geometry_cache_write{};
    /// @brief Time spent deriving backend-specific host packing.
    PhaseTiming backend_packing{};
    /// @brief Time spent deriving the CPU far-field execution packing.
    PhaseTiming far_field_packing{};
    /// @brief Time spent creating and uploading persistent CUDA state.
    PhaseTiming cuda_upload{};
    /// @brief Complete constructor setup time, including backend creation.
    PhaseTiming total_setup{};
    /// @brief Whether the universal operator bank was loaded from the cache.
    bool universal_cache_hit{false};
    /// @brief Whether the periodic root operator was loaded from the cache.
    bool periodic_cache_hit{false};
    /// @brief Whether the complete geometry plan was loaded from the cache.
    bool geometry_cache_hit{false};
    /// @brief Validated cache bytes read during this construction.
    std::size_t cache_bytes_read{0};
    /// @brief Cache bytes written during this construction.
    std::size_t cache_bytes_written{0};
    /// @brief Maximum expansion degree selected by the plan.
    int expansion_order{0};
    /// @brief Number of real coefficients stored for each node expansion.
    std::size_t coefficient_count{0};
    /// @brief True for the real spherical-harmonic representation.
    bool spherical{false};
    /// @brief Bytes used by one selected execution scalar.
    std::size_t scalar_bytes{sizeof(double)};
    /// @brief Maximum number of integer M2L offsets in a uniform 3-D octree.
    static constexpr std::size_t theoretical_maximum_m2l_classes = 316;
    /// @brief Time spent constructing leaf P2M maps.
    PhaseTiming p2m_plan{};
    /// @brief Time spent constructing shared M2M maps.
    PhaseTiming m2m_plan{};
    /// @brief Time spent constructing grouped M2L maps.
    PhaseTiming m2l_plan{};
    /// @brief Time spent constructing shared L2L maps.
    PhaseTiming l2l_plan{};
    /// @brief Time spent constructing target L2P rows.
    PhaseTiming l2p_plan{};
    /// @brief Time spent constructing the compact list1 dipole tensors.
    ///
    /// This is the whole near-field stage.  The three timings below decompose
    /// it into materialising the interaction list, building the exact
    /// canonical operator, and deriving the execution packings from it.
    PhaseTiming p2p_tensor_plan{};
    /// @brief Time spent expanding leaf records into near-field interactions.
    PhaseTiming p2p_interaction_setup{};
    /// @brief Time spent building the exact canonical near-field operator.
    PhaseTiming p2p_canonical_operator{};
    /// @brief Time spent deriving compact, dictionary and packing forms.
    PhaseTiming p2p_derived_packing{};
    /// @brief Time spent converting the FP64 static plan to FP32.
    PhaseTiming precision_conversion{};
    /// @brief Time spent identifying unique integer M2L displacement classes.
    PhaseTiming transfer_discovery{};
    /// @brief Time spent generating retained numerical operator values.
    PhaseTiming operator_construction{};
    /// @brief Time spent allocating persistent plan and execution buffers.
    PhaseTiming buffer_allocation{};
    /// @brief Total static-plan construction time.
    PhaseTiming total{};
    /// @brief Number of retained M2L displacement classes.
    std::size_t transfer_classes{0};
    /// @brief Number of canonical M2L source-to-target interactions.
    std::size_t interactions{0};
    /// @brief Bytes occupied by all retained numerical operators.
    std::size_t operator_bytes{0};
    /// @brief Bytes occupied by fixed source-to-multipole maps.
    std::size_t p2m_operator_bytes{0};
    /// @brief Bytes occupied by all interaction metadata.
    std::size_t interaction_bytes{0};
    /// @brief Bytes occupied by reusable host execution scratch.
    std::size_t scratch_bytes{0};
    /// @brief Mutable expansions, moments, fields, and result storage.
    std::size_t state_bytes{0};
    /// @brief Bytes occupied by persistent multipole coefficient state.
    std::size_t multipole_state_bytes{0};
    /// @brief Bytes occupied by persistent local coefficient state.
    std::size_t local_state_bytes{0};
    /// @brief Bytes occupied by moments, fields, and other mutable state.
    std::size_t other_state_bytes{0};
    /// @brief Bytes occupied by fixed target local-evaluation rows.
    std::size_t l2p_operator_bytes{0};
    /// @brief Bytes occupied by retained list1 values and metadata.
    std::size_t near_field_operator_bytes{0};
    /// @brief Bytes occupied by immutable uniform-tree storage.
    std::size_t tree_bytes{0};
    /// @brief Bytes occupied by immutable canonical topology and geometry.
    std::size_t topology_bytes{0};
    /// @brief Universal M2M templates stored (exactly eight child classes).
    std::size_t m2m_operators{0};
    /// @brief Complete-tree parent-child relations represented by M2M IDs.
    std::size_t m2m_theoretical_interactions{0};
    /// @brief Bytes occupied by the shared M2M operator table.
    std::size_t m2m_operator_bytes{0};
    /// @brief Dense M2L matrices stored after unused classes are discarded.
    std::size_t m2l_operators{0};
    /// @brief Bytes occupied by the dense M2L operator table.
    std::size_t m2l_operator_bytes{0};
    /// @brief Bytes occupied by M2L source, target, and level metadata.
    std::size_t m2l_interaction_bytes{0};
    /// @brief Universal L2L templates stored (exactly eight child classes).
    std::size_t l2l_operators{0};
    /// @brief Complete-tree parent-child relations represented by L2L IDs.
    std::size_t l2l_theoretical_interactions{0};
    /// @brief Bytes occupied by the shared L2L operator table.
    std::size_t l2l_operator_bytes{0};
    /// @brief True because directly executable M2L matrices use dense storage.
    bool dense{true};
    /// @brief Sparse execution is disabled pending favourable benchmark evidence.
    bool sparse{false};
    /// @brief No accuracy-changing coefficient pruning is performed.
    bool numerically_pruned{false};
    /// @brief No hot-path rotations, reflections, or permutations are used.
    bool symmetry_compressed{false};

    /// @brief Returns storage occupied only by translation operator tables.
    [[nodiscard]] std::size_t translation_operator_bytes() const {
        return m2m_operator_bytes + m2l_operator_bytes + l2l_operator_bytes;
    }
    /// @brief Number of particle pairs represented by the list1 tensor.
    std::size_t p2p_interactions{0};
    /// @brief Six-coefficient tensor storage.
    std::size_t p2p_value_bytes{0};
    /// @brief Row and source/target index storage.
    std::size_t p2p_index_bytes{0};
    /// @brief Complete persistent bytes of the canonical particle-row packing.
    std::size_t p2p_canonical_total_bytes{0};
    /// @brief Number of retained magnitude Tensor6 dictionary entries.
    std::size_t p2p_unique_tensors{0};
    /// @brief Number of packed Tensor6 interaction tokens.
    std::size_t p2p_dictionary_tokens{0};
    /// @brief Bytes used by each packed Tensor6 interaction token.
    std::size_t p2p_dictionary_token_width_bytes{0};
    /// @brief Bytes occupied by packed Tensor6 interaction tokens.
    std::size_t p2p_dictionary_token_bytes{0};
    /// @brief Bytes occupied by unique Tensor6 dictionary values.
    std::size_t p2p_dictionary_tensor_bytes{0};
    /// @brief Complete persistent bytes of the selected dictionary packing.
    std::size_t p2p_dictionary_total_bytes{0};
    /// @brief Number of times the immutable plan has been constructed.
    std::uint64_t construction_count{0};

    /// @brief Returns retained host storage excluding immutable tree storage.
    [[nodiscard]] std::size_t total_bytes() const {
        return operator_bytes + interaction_bytes + scratch_bytes + state_bytes;
    }

    /// @brief Returns the complete host plan, including immutable tree and topology storage.
    [[nodiscard]] std::size_t total_persistent_bytes() const {
        return total_bytes() + tree_bytes + topology_bytes;
    }
};

/** @brief Host/device traffic and persistent storage for a CUDA plan. */
struct CudaPlanStatistics {
    /// @brief Bytes used by one selected device execution scalar.
    std::size_t scalar_bytes{sizeof(double)};
    /// @brief Number of unique uploaded M2M matrices.
    std::size_t m2m_unique_matrix_count{0};
    /// @brief Bytes of the uploaded M2M matrices and their schedule.
    std::size_t m2m_matrix_bytes{0};
    /// @brief Number of unique uploaded M2L matrices.
    std::size_t m2l_unique_matrix_count{0};
    /// @brief Bytes of the uploaded M2L matrices.
    std::size_t m2l_matrix_bytes{0};
    /// @brief Bytes occupied by uploaded M2L interaction metadata.
    std::size_t m2l_interaction_metadata_bytes{0};
    /// @brief Number of canonical source-to-target M2L interactions.
    std::size_t m2l_interaction_count{0};
    /// @brief Number of non-empty target rows executed by CUDA M2L.
    std::size_t m2l_active_row_count{0};
    /// @brief Persistent CUDA M2L evaluation scratch in bytes.
    std::size_t m2l_scratch_bytes{0};
    /// @brief Threads used by each CUDA M2L target-row kernel block.
    int m2l_threads_per_block{0};
    /// @brief Number of unique uploaded L2L matrices.
    std::size_t l2l_unique_matrix_count{0};
    /// @brief Bytes of the uploaded L2L matrices and their schedule.
    std::size_t l2l_matrix_bytes{0};
    /// @brief Number of particle pairs in the uploaded P2P packing.
    std::size_t p2p_interaction_count{0};
    /// @brief Bytes occupied by uploaded P2P tensor values or dictionary variants.
    std::size_t p2p_tensor_bytes{0};
    /// @brief Bytes occupied by uploaded P2P source indices or tokens.
    std::size_t p2p_index_bytes{0};
    /// @brief Bytes occupied by uploaded P2P row offsets.
    std::size_t p2p_row_metadata_bytes{0};
    /// @brief Bytes occupied by uploaded P2P leaf ranges, block records and tile schedules.
    std::size_t p2p_leaf_metadata_bytes{0};
    /// @brief Bytes occupied by immutable P2P self-identity metadata.
    std::size_t p2p_identity_bytes{0};
    /// @brief Bytes occupied by persistent P2P evaluation scratch.
    std::size_t p2p_scratch_bytes{0};
    /// @brief Bytes occupied by the resident sorted positions of the
    /// position-based point P2P executor (zero for stored-tensor packings).
    std::size_t p2p_geometry_bytes{0};
    /// @brief Threads per block for a custom P2P kernel, or zero for cuSPARSE.
    int p2p_threads_per_block{0};
    /// @brief Bytes uploaded once at construction.
    std::size_t setup_h2d_bytes{0};
    /// @brief Bytes uploaded by the most recent evaluation.
    std::size_t evaluation_h2d_bytes{0};
    /// @brief Bytes downloaded by the most recent evaluation.
    std::size_t evaluation_d2h_bytes{0};
    /// @brief Number of per-evaluation uploads so far.
    std::uint64_t evaluation_h2d_calls{0};
    /// @brief Number of per-evaluation downloads so far.
    std::uint64_t evaluation_d2h_calls{0};
    /// @brief Total persistent device allocation owned by the plan.
    std::size_t persistent_device_bytes{0};
    /// @brief Number of device plans constructed from this payload.
    std::uint64_t plan_generation_count{0};
    /// @brief Number of static operator uploads (always one per plan).
    std::uint64_t static_upload_count{0};
    /// @brief Number of static M2L uploads (always one per plan).
    std::uint64_t static_m2l_upload_count{0};
    /// @brief Number of static P2P uploads (always one per plan).
    std::uint64_t static_p2p_upload_count{0};
    /// @brief Number of immutable geometry-metadata uploads.
    std::uint64_t geometry_upload_count{0};
};

/**
 * @brief Device-stream phase timings for the most recent CUDA evaluation.
 *
 * Every value is measured between CUDA events on the plan's streams.  Phases
 * that ran on different streams overlap, so they are lanes, not addends:
 * compare them with `total_seconds` rather than summing them.  Fields a plan
 * does not use stay zero.  The diagnostic events exist only at
 * `TimingLevel::Detailed`; below it no timing event is recorded, no elapsed
 * time is queried, and every field stays zero with `timing_level` saying so.
 */
struct CudaEvaluationTimings {
  TimingLevel timing_level{TimingLevel::Off};  ///< Level the lanes below were collected at.
    double h2d_seconds{0.0};           ///< Upload of the changing inputs.
    double gather_seconds{0.0};        ///< Grouped M2L: packing source multipoles (unused by the current kernels).
    double multiply_seconds{0.0};      ///< M2L matrix application.
    double scatter_seconds{0.0};       ///< Grouped M2L: accumulation into locals (unused by the current kernels).
    double scale_seconds{0.0};         ///< M2L multipole pre-scaling pass.
    double kernel_seconds{0.0};        ///< Sum of the kernel phases; a diagnostic, not a wall time.
    double d2h_seconds{0.0};           ///< Download of the requested outputs.
    double p2m_seconds{0.0};           ///< Device P2M (CudaFull only).
    double m2m_seconds{0.0};           ///< Device M2M (CudaFull only).
    double m2l_seconds{0.0};           ///< Device M2L including its scaling pass.
    double l2l_seconds{0.0};           ///< Device L2L (CudaFull only).
    double l2p_seconds{0.0};           ///< Device L2P (CudaFull only).
    double p2p_seconds{0.0};           ///< Device near field, on its own stream.
    double accumulation_seconds{0.0};  ///< Combination and unsorting of near and far fields (CudaFull only).
    double total_seconds{0.0};         ///< First event to last event on the plan's primary stream.
};

} // namespace cdfmm
