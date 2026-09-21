// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

#include "cdfmm/core/timing.hpp"

namespace cdfmm::detail::dense_direct {

/**
 * @brief Per-phase cost and transient storage of one dense direct construction.
 *
 * NOTE(cdfmm): this is an internal diagnostic record, not part of the dense
 * plan's interface.  `DenseDirectPlan` fills the thread-local instance below
 * on every construction so the in-tree construction benchmark can attribute
 * setup cost without the public header, the C ABI or the Python API growing a
 * measurement-only surface.  The `PhaseTiming` fields follow the
 * constructor's `TimingLevel`: none at `Off` (no clock is read), `total` at
 * `Coarse`, every phase at `Detailed`; a handful of clock reads per plan,
 * never per pair.  Counts and bytes are filled at every level.
 */
struct ConstructionStatistics {
    /// @brief Level the timing fields below were collected at.
    TimingLevel timing_level{TimingLevel::Off};
    /// @brief Time spent validating the supplied geometry records.
    PhaseTiming validation{};
    /// @brief Time spent sizing and zeroing the six dense matrices.
    PhaseTiming allocation{};
    /// @brief Time spent deriving one prepared body per distinct record.
    PhaseTiming geometry_preparation{};
    /// @brief Time spent grouping pairs by bitwise-identical operator inputs.
    PhaseTiming classification{};
    /// @brief Time spent evaluating exact pair tensors.
    PhaseTiming tensor_build{};
    /// @brief Time spent scattering built tensors into the dense matrices.
    PhaseTiming materialisation{};
    /// @brief Complete constructor time.
    PhaseTiming total{};

    /// @brief Source-target pairs this plan describes, i.e. `Ns * Nt`.
    std::size_t pair_count{0};
    /// @brief Exact tensors actually evaluated.
    std::size_t built_tensor_count{0};
    /// @brief Distinct bodies prepared: distinct source plus target records.
    std::size_t prepared_body_count{0};
    /// @brief Whether exact classification was kept rather than abandoned.
    bool classified{false};

    // WARNING(cdfmm): when classification is abandoned the build and the
    // scatter are one fused loop, because separating them would need a
    // complete `PairTensor` array -- 48 bytes per pair, far larger than the
    // matrices themselves.  That fused loop is reported under `tensor_build`
    // and `materialisation` stays zero.  A row is fused exactly when
    // `classified` is false.

    /// @brief Retained bytes of the six dense matrices.
    std::size_t matrix_bytes{0};
    /// @brief Transient bytes held by prepared source and target bodies.
    std::size_t prepared_geometry_bytes{0};
    /// @brief Transient bytes held by the class map and representative list.
    std::size_t class_map_bytes{0};
    /// @brief Transient bytes held by the distinct built tensors.
    std::size_t unique_tensor_bytes{0};
};

/**
 * @brief Returns the calling thread's record of its most recent construction.
 *
 * The record is overwritten by each `DenseDirectPlan` construction on that
 * thread, so a caller reads it immediately after the plan it describes.
 */
[[nodiscard]] ConstructionStatistics& construction_statistics() noexcept;

} // namespace cdfmm::detail::dense_direct
