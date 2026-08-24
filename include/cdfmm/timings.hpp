// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

namespace cdfmm {

//------------------------------------------------------------------------------
// Public timing types
//------------------------------------------------------------------------------

/** @brief Accumulated monotonic wall time and invocation count for one phase.
 */
struct PhaseTiming {
  /// @brief Accumulated wall time in seconds.
  double total_seconds{0.0};
    /// @brief Number of observations included in the accumulated time.
  std::uint64_t calls{0};

  /// @brief Adds one elapsed wall-time observation.
  void add(double seconds) {
    total_seconds += seconds;
    ++calls;
  }
};

/** @brief Wall-clock breakdown for fixed uniform-tree geometry construction. */
struct TreeBuildTimings {
    /// @brief Time spent determining or validating the root bounds.
    PhaseTiming root_bounds{};
    /// @brief Time spent constructing the complete node array.
    PhaseTiming node_construction{};
    /// @brief Time spent building parent, child, and level topology.
    PhaseTiming topology{};
    /// @brief Time spent assigning Morton keys to source positions.
    PhaseTiming source_morton{};
    /// @brief Time spent sorting sources by Morton key.
    PhaseTiming source_sorting{};
    /// @brief Time spent assigning Morton keys to target positions.
    PhaseTiming target_morton{};
    /// @brief Time spent sorting targets by Morton key.
    PhaseTiming target_sorting{};
    /// @brief Time spent assigning source and target ranges to nodes.
    PhaseTiming ranges{};
    /// @brief Time spent constructing near- and far-field interaction lists.
    PhaseTiming interaction_lists{};
    /// @brief Total tree construction time.
    PhaseTiming total{};
};

/** @brief Wall-clock breakdown for one or more dipole evaluations. */
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
    /// @brief Total complete-evaluation time.
    PhaseTiming total{};
    /// @brief Number of complete evaluations represented by these timings.
    std::uint64_t evaluations{0};
};

/** @brief One-time cost and storage of the immutable static CPU plan. */
struct StaticPlanStatistics {
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
    PhaseTiming p2p_tensor_plan{};
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
    /// @brief Shared M2M matrices stored (eight child classes per used level).
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
    /// @brief Shared L2L matrices stored (eight child classes per used level).
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
    /// @brief Number of times the immutable plan has been constructed.
    std::uint64_t construction_count{0};

    /// @brief Returns retained host storage excluding immutable tree storage.
    [[nodiscard]] std::size_t total_bytes() const {
        return operator_bytes + interaction_bytes + scratch_bytes + state_bytes;
    }

    /// @brief Returns the complete host plan, including immutable tree storage.
    [[nodiscard]] std::size_t total_persistent_bytes() const {
        return total_bytes() + tree_bytes;
    }
};

/** @brief Host/device traffic and persistent storage for a CUDA plan. */
struct CudaPlanStatistics {
    /// @brief Bytes used by one selected device execution scalar.
    std::size_t scalar_bytes{sizeof(double)};
    /// @brief Number and bytes of unique uploaded M2M matrices.
    std::size_t m2m_unique_matrix_count{0};
    std::size_t m2m_matrix_bytes{0};
    /// @brief Number and bytes of unique uploaded M2L matrices.
    std::size_t m2l_unique_matrix_count{0};
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
    /// @brief Number and bytes of unique uploaded L2L matrices.
    std::size_t l2l_unique_matrix_count{0};
    std::size_t l2l_matrix_bytes{0};
    /// @brief Number of particle pairs in the uploaded P2P packing.
    std::size_t p2p_interaction_count{0};
    /// @brief Bytes occupied by P2P values, indices, and packing metadata.
    std::size_t p2p_tensor_bytes{0};
    std::size_t p2p_index_bytes{0};
    std::size_t p2p_row_metadata_bytes{0};
    std::size_t p2p_leaf_metadata_bytes{0};
    /// @brief Bytes occupied by immutable P2P self-identity metadata.
    std::size_t p2p_identity_bytes{0};
    /// @brief Bytes occupied by persistent P2P evaluation scratch.
    std::size_t p2p_scratch_bytes{0};
    /// @brief Threads per block for a custom P2P kernel, or zero for cuSPARSE.
    int p2p_threads_per_block{0};
    /// @brief Immutable setup traffic and per-evaluation dynamic traffic.
    std::size_t setup_h2d_bytes{0};
    std::size_t evaluation_h2d_bytes{0};
    std::size_t evaluation_d2h_bytes{0};
    /// @brief Dynamic transfer counts accumulated over evaluations.
    std::uint64_t evaluation_h2d_calls{0};
    std::uint64_t evaluation_d2h_calls{0};
    /// @brief Total persistent device allocation owned by the plan.
    std::size_t persistent_device_bytes{0};
    /// @brief Number of device plans constructed from this payload.
    std::uint64_t plan_generation_count{0};
    /// @brief Counts of all static, M2L, and P2P upload operations.
    std::uint64_t static_upload_count{0};
    std::uint64_t static_m2l_upload_count{0};
    std::uint64_t static_p2p_upload_count{0};
    /// @brief Number of immutable geometry-metadata uploads.
    std::uint64_t geometry_upload_count{0};
};

/** @brief Device-stream phase timings for the most recent CUDA evaluation. */
struct CudaEvaluationTimings {
    double h2d_seconds{0.0};
    double gather_seconds{0.0};
    double multiply_seconds{0.0};
    double scatter_seconds{0.0};
    double scale_seconds{0.0};
    double kernel_seconds{0.0};
    double d2h_seconds{0.0};
    double p2m_seconds{0.0};
    double m2m_seconds{0.0};
    double m2l_seconds{0.0};
    double l2l_seconds{0.0};
    double l2p_seconds{0.0};
    double p2p_seconds{0.0};
    double accumulation_seconds{0.0};
    double total_seconds{0.0};
};

} // namespace cdfmm
