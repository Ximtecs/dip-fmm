// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/plan/p2p/compact.hpp"

namespace cdfmm {

StaticP2PCompactPlan
build_static_p2p_compact_plan(const StaticP2POperator& operator_map)
{
    StaticP2PCompactPlan result;
    result.source_count = operator_map.source_count;
    result.target_count = operator_map.target_count;
    result.row_offsets = operator_map.row_offsets;
    result.source_indices.reserve(operator_map.blocks.size());
    result.skip_for_identity.reserve(operator_map.blocks.size());
    for (auto& coefficient : result.potential) {
        coefficient.reserve(operator_map.blocks.size());
    }
    for (auto& tensor : result.tensors) {
        tensor.reserve(operator_map.blocks.size());
    }

    for (const StaticDipoleBlock& block : operator_map.blocks) {
        result.source_indices.push_back(block.source);
        result.skip_for_identity.push_back(
            static_cast<unsigned char>(block.skip_for_identity != 0));
        result.potential[0].push_back(block.px);
        result.potential[1].push_back(block.py);
        result.potential[2].push_back(block.pz);
        result.tensors[0].push_back(block.xx);
        result.tensors[1].push_back(block.xy);
        result.tensors[2].push_back(block.xz);
        result.tensors[3].push_back(block.yy);
        result.tensors[4].push_back(block.yz);
        result.tensors[5].push_back(block.zz);
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
    result.source_indices.reserve(operator_map.blocks.size());
    result.skip_for_identity.reserve(operator_map.blocks.size());
    for (auto& coefficient : result.potential) {
        coefficient.reserve(operator_map.blocks.size());
    }
    for (auto& tensor : result.tensors) {
        tensor.reserve(operator_map.blocks.size());
    }

    for (const FloatStaticDipoleBlock& block : operator_map.blocks) {
        result.source_indices.push_back(block.source);
        result.skip_for_identity.push_back(
            static_cast<unsigned char>(block.skip_for_identity != 0));
        result.potential[0].push_back(block.px);
        result.potential[1].push_back(block.py);
        result.potential[2].push_back(block.pz);
        result.tensors[0].push_back(block.xx);
        result.tensors[1].push_back(block.xy);
        result.tensors[2].push_back(block.xz);
        result.tensors[3].push_back(block.yy);
        result.tensors[4].push_back(block.yz);
        result.tensors[5].push_back(block.zz);
    }
    return result;
}

} // namespace cdfmm
