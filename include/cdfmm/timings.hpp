// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cstddef>

namespace cdfmm {

//------------------------------------------------------------------------------
// Public timing types
//------------------------------------------------------------------------------

/** @brief Accumulated monotonic wall time and invocation count for one phase. */
struct PhaseTiming {
    /// @brief Accumulated wall time in seconds.
    double total_seconds{0.0};
    /// @brief Number of observations included in the accumulated time.
    std::uint64_t calls{0};

    /// @brief Adds one elapsed wall-time observation.
    void add(double seconds)
    {
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
    /// @brief Total complete-evaluation time.
    PhaseTiming total{};
    /// @brief Number of complete evaluations represented by these timings.
    std::uint64_t evaluations{0};
};

/** @brief One-time cost and storage of the immutable static CPU plan. */
struct StaticPlanStatistics {
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
    PhaseTiming transfer_discovery{};
    PhaseTiming operator_construction{};
    PhaseTiming buffer_allocation{};
    PhaseTiming total{};
    std::size_t transfer_classes{0};
    std::size_t interactions{0};
    std::size_t operator_bytes{0};
    std::size_t interaction_bytes{0};
    std::size_t scratch_bytes{0};
    /// @brief Number of times the immutable plan has been constructed.
    std::uint64_t construction_count{0};

    /// @brief Returns the approximate total persistent plan storage.
    [[nodiscard]] std::size_t total_bytes() const
    {
        return operator_bytes + interaction_bytes + scratch_bytes;
    }
};

/** @brief Host/device traffic and persistent storage for a CUDA plan. */
struct CudaPlanStatistics {
    std::size_t setup_h2d_bytes{0};
    std::size_t evaluation_h2d_bytes{0};
    std::size_t evaluation_d2h_bytes{0};
    std::uint64_t evaluation_h2d_calls{0};
    std::uint64_t evaluation_d2h_calls{0};
    std::size_t persistent_device_bytes{0};
    std::uint64_t plan_generation_count{0};
    std::uint64_t static_upload_count{0};
    std::uint64_t static_m2l_upload_count{0};
};

/** @brief Device-stream phase timings for the most recent CUDA evaluation. */
struct CudaEvaluationTimings {
    double h2d_seconds{0.0};
    double gather_seconds{0.0};
    double multiply_seconds{0.0};
    double scatter_seconds{0.0};
    double kernel_seconds{0.0};
    double d2h_seconds{0.0};
};

} // namespace cdfmm
