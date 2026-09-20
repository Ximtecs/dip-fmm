// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

namespace cdfmm {

/** @brief Persistent-memory breakdown for a static P2P representation. */
struct StaticP2PMemory {
    std::size_t tensor_bytes{0};        ///< Stored tensor values (or dictionary variants).
    std::size_t index_bytes{0};         ///< Source indices, tokens and identity markers.
    std::size_t row_metadata_bytes{0};  ///< Per-target or per-leaf row offsets.
    std::size_t leaf_metadata_bytes{0}; ///< Leaf ranges, block records and tile schedules.
    std::size_t scratch_bytes{0};       ///< Persistent execution scratch owned by the packing.

    /// Sum of the five categories.
    [[nodiscard]] std::size_t total_bytes() const noexcept;
};

} // namespace cdfmm
