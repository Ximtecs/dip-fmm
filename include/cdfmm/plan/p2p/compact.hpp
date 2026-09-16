// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <vector>

#include "cdfmm/plan/p2p/canonical.hpp"
#include "cdfmm/plan/p2p/memory.hpp"

namespace cdfmm {

/** @brief Source-index and SoA packing of canonical FP64 particle rows. */
struct StaticP2PCompactPlan {
    int source_count{0};
    int target_count{0};
    std::vector<int> row_offsets{};
    std::vector<int> source_indices{};
    std::vector<unsigned char> skip_for_identity{};
    std::array<std::vector<double>, 3> potential{};
    std::array<std::vector<double>, 6> tensors{};

    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/** @brief Source-index and SoA packing of canonical FP32 particle rows. */
struct FloatStaticP2PCompactPlan {
    int source_count{0};
    int target_count{0};
    std::vector<int> row_offsets{};
    std::vector<int> source_indices{};
    std::vector<unsigned char> skip_for_identity{};
    std::array<std::vector<float>, 3> potential{};
    std::array<std::vector<float>, 6> tensors{};

    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

[[nodiscard]] StaticP2PCompactPlan
build_static_p2p_compact_plan(const StaticP2POperator& operator_map);

[[nodiscard]] FloatStaticP2PCompactPlan
build_static_p2p_compact_plan(const FloatStaticP2POperator& operator_map);

} // namespace cdfmm
