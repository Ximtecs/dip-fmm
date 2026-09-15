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

/**
 * @brief Authoritative canonical-to-compact row rule: writes one already
 * source-indexed compact @p slot from its canonical @p block.
 *
 * This is the single statement of what compact P2P means field-by-field
 * (source index, identity marker, potential, and tensor components). Callers
 * own allocation and iteration order; both the ordinary plan builder above
 * and the fused cache decode in `cache/format.cpp` call this per row so the
 * mapping is defined once. `compact_plan`'s arrays must already be sized to
 * hold `slot`.
 */
inline void assign_static_p2p_compact_row(StaticP2PCompactPlan& compact_plan,
                                          std::size_t slot,
                                          const StaticDipoleBlock& block) noexcept
{
    compact_plan.source_indices[slot] = block.source;
    compact_plan.skip_for_identity[slot] =
        static_cast<unsigned char>(block.skip_for_identity != 0);
    compact_plan.potential[0][slot] = block.px;
    compact_plan.potential[1][slot] = block.py;
    compact_plan.potential[2][slot] = block.pz;
    compact_plan.tensors[0][slot] = block.xx;
    compact_plan.tensors[1][slot] = block.xy;
    compact_plan.tensors[2][slot] = block.xz;
    compact_plan.tensors[3][slot] = block.yy;
    compact_plan.tensors[4][slot] = block.yz;
    compact_plan.tensors[5][slot] = block.zz;
}

/** @brief FP32 counterpart of @ref assign_static_p2p_compact_row. */
inline void assign_static_p2p_compact_row(FloatStaticP2PCompactPlan& compact_plan,
                                          std::size_t slot,
                                          const FloatStaticDipoleBlock& block) noexcept
{
    compact_plan.source_indices[slot] = block.source;
    compact_plan.skip_for_identity[slot] =
        static_cast<unsigned char>(block.skip_for_identity != 0);
    compact_plan.potential[0][slot] = block.px;
    compact_plan.potential[1][slot] = block.py;
    compact_plan.potential[2][slot] = block.pz;
    compact_plan.tensors[0][slot] = block.xx;
    compact_plan.tensors[1][slot] = block.xy;
    compact_plan.tensors[2][slot] = block.xz;
    compact_plan.tensors[3][slot] = block.yy;
    compact_plan.tensors[4][slot] = block.yz;
    compact_plan.tensors[5][slot] = block.zz;
}

} // namespace cdfmm
