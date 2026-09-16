// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/plan/p2p/bsr.hpp"

#include <cstddef>
#include <stdexcept>

namespace cdfmm {
namespace {

template <typename Canonical, typename Plan, typename Scalar>
Plan build_bsr(
    const Canonical& operator_map,
    const std::span<const int> target_source_indices)
{
    if (!target_source_indices.empty() &&
        target_source_indices.size() !=
            static_cast<std::size_t>(operator_map.target_count)) {
        throw std::invalid_argument(
            "BSR P2P identity dimensions are inconsistent");
    }

    Plan result;
    result.source_count = operator_map.source_count;
    result.target_count = operator_map.target_count;
    result.target_source_indices.assign(target_source_indices.begin(),
                                        target_source_indices.end());
    if (result.target_source_indices.empty()) {
        result.target_source_indices.assign(
            static_cast<std::size_t>(operator_map.target_count), -1);
    }
    result.row_offsets.reserve(operator_map.row_offsets.size());
    result.row_offsets.push_back(0);
    result.source_indices.reserve(operator_map.blocks.size());
    result.values.reserve(operator_map.blocks.size() * 9);

    // A sparse block row holds one block per (target, source). The canonical
    // rows keep the periodic images of one source adjacent, so their tensors
    // are summed into that block; a fixed point-source identity zeroes only
    // the image that carries the identity marker (the primary cell).
    for (int target = 0; target < operator_map.target_count; ++target) {
        const int self_source =
            result.target_source_indices[static_cast<std::size_t>(target)];
        int previous_source = -1;
        for (int entry = operator_map.row_offsets[
                 static_cast<std::size_t>(target)];
             entry < operator_map.row_offsets[
                 static_cast<std::size_t>(target) + 1]; ++entry) {
            const auto& block =
                operator_map.blocks[static_cast<std::size_t>(entry)];
            const bool self = block.skip_for_identity != 0 &&
                block.source == self_source;
            const Scalar xx = self ? Scalar{0} : block.xx;
            const Scalar xy = self ? Scalar{0} : block.xy;
            const Scalar xz = self ? Scalar{0} : block.xz;
            const Scalar yy = self ? Scalar{0} : block.yy;
            const Scalar yz = self ? Scalar{0} : block.yz;
            const Scalar zz = self ? Scalar{0} : block.zz;
            if (block.source == previous_source) {
                Scalar* merged = result.values.data() + result.values.size() - 9;
                merged[0] += xx;
                merged[1] += xy;
                merged[2] += xz;
                merged[3] += xy;
                merged[4] += yy;
                merged[5] += yz;
                merged[6] += xz;
                merged[7] += yz;
                merged[8] += zz;
                continue;
            }
            previous_source = block.source;
            result.source_indices.push_back(block.source);
            result.values.insert(result.values.end(),
                                 {xx, xy, xz, xy, yy, yz, xz, yz, zz});
        }
        result.row_offsets.push_back(
            static_cast<int>(result.source_indices.size()));
    }
    return result;
}

} // namespace

StaticP2PBsrPlan build_static_p2p_bsr_plan(
    const StaticP2POperator& operator_map,
    const std::span<const int> target_source_indices)
{
    return build_bsr<StaticP2POperator, StaticP2PBsrPlan, double>(
        operator_map, target_source_indices);
}

FloatStaticP2PBsrPlan build_static_p2p_bsr_plan(
    const FloatStaticP2POperator& operator_map,
    const std::span<const int> target_source_indices)
{
    return build_bsr<FloatStaticP2POperator, FloatStaticP2PBsrPlan, float>(
        operator_map, target_source_indices);
}

} // namespace cdfmm
