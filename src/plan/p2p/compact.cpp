// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/plan/p2p/compact.hpp"

#include "plan/p2p/compact_row.hpp"

namespace cdfmm {

using cdfmm::plan_detail::assign_static_p2p_compact_row;

StaticP2PCompactPlan
build_static_p2p_compact_plan(const StaticP2POperator& operator_map)
{
    StaticP2PCompactPlan result;
    result.source_count = operator_map.source_count;
    result.target_count = operator_map.target_count;
    result.row_offsets = operator_map.row_offsets;
    const std::size_t block_count = operator_map.blocks.size();
    result.source_indices.resize(block_count);
    result.skip_for_identity.resize(block_count);
    for (auto& coefficient : result.potential) {
        coefficient.resize(block_count);
    }
    for (auto& tensor : result.tensors) {
        tensor.resize(block_count);
    }

    for (std::size_t slot = 0; slot < block_count; ++slot) {
        assign_static_p2p_compact_row(result, slot, operator_map.blocks[slot]);
    }
    return result;
}

FloatStaticP2PCompactPlan
build_static_p2p_compact_plan(const FloatStaticP2POperator& operator_map)
{
    FloatStaticP2PCompactPlan result;
    result.source_count = operator_map.source_count;
    result.target_count = operator_map.target_count;
    result.row_offsets = operator_map.row_offsets;
    const std::size_t block_count = operator_map.blocks.size();
    result.source_indices.resize(block_count);
    result.skip_for_identity.resize(block_count);
    for (auto& coefficient : result.potential) {
        coefficient.resize(block_count);
    }
    for (auto& tensor : result.tensors) {
        tensor.resize(block_count);
    }

    for (std::size_t slot = 0; slot < block_count; ++slot) {
        assign_static_p2p_compact_row(result, slot, operator_map.blocks[slot]);
    }
    return result;
}

} // namespace cdfmm
