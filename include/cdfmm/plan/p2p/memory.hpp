// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

namespace cdfmm {

/** @brief Persistent-memory breakdown for a static P2P representation. */
struct StaticP2PMemory {
    std::size_t tensor_bytes{0};
    std::size_t index_bytes{0};
    std::size_t row_metadata_bytes{0};
    std::size_t leaf_metadata_bytes{0};
    std::size_t scratch_bytes{0};

    [[nodiscard]] std::size_t total_bytes() const noexcept;
};

} // namespace cdfmm
