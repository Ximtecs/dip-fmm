// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/plan/p2p/signed_dictionary.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cdfmm/tensor_dictionary.hpp"
#include "dictionary_detail.hpp"

namespace cdfmm {

StaticP2PSignedTensorDictionaryPlan
build_static_p2p_signed_tensor_dictionary_plan(
    const StaticP2POperator& operator_map,
    const std::span<const StaticP2PLeafPair> leaf_pairs,
    const std::span<const int> target_source_indices,
    const int target_tile_size)
{
    if (target_tile_size <= 0 || target_tile_size > 128) {
        throw std::invalid_argument(
            "signed Tensor6 target tile must be in [1, 128]");
    }
    if (!target_source_indices.empty() &&
        target_source_indices.size() !=
            static_cast<std::size_t>(operator_map.target_count)) {
        throw std::invalid_argument(
            "signed Tensor6 dictionary identity dimensions are inconsistent");
    }
    const StaticP2PLeafPlan leaf =
        build_static_p2p_leaf_plan(operator_map, leaf_pairs);
    StaticP2PSignedTensorDictionaryPlan result;
    result.source_count = leaf.source_count;
    result.target_count = leaf.target_count;
    result.target_begins = leaf.target_begins;
    result.target_counts = leaf.target_counts;
    result.leaf_row_offsets = leaf.leaf_row_offsets;
    result.blocks = leaf.blocks;
    result.target_tile_size = target_tile_size;

    // Fixed point self interactions become ordinary exact-zero lookups.
    for (auto& component : result.tensors) {
        component.push_back(0.0);
    }
    std::unordered_map<Tensor6BitKey<double>, std::uint32_t,
                       plan_detail::Tensor6BitKeyHash<double>> variant_ids;
    variant_ids.reserve(leaf.tensors[0].size());
    variant_ids.emplace(tensor6_bit_key(std::array<double, 6>{}), 0U);
    std::vector<std::uint32_t> wide_tokens;
    wide_tokens.reserve(leaf.tensors[0].size());

    for (int target_leaf = 0;
         target_leaf < static_cast<int>(leaf.target_begins.size());
         ++target_leaf) {
        const int target_begin =
            leaf.target_begins[static_cast<std::size_t>(target_leaf)];
        const int target_count =
            leaf.target_counts[static_cast<std::size_t>(target_leaf)];
        for (int local_begin = 0; local_begin < target_count;
             local_begin += target_tile_size) {
            result.tile_leaf_indices.push_back(target_leaf);
            result.tile_target_offsets.push_back(local_begin);
        }
        for (int block_index =
                 leaf.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
             block_index < leaf.leaf_row_offsets[
                 static_cast<std::size_t>(target_leaf) + 1]; ++block_index) {
            StaticP2PLeafBlock& block =
                result.blocks[static_cast<std::size_t>(block_index)];
            const StaticP2PLeafBlock& source_block =
                leaf.blocks[static_cast<std::size_t>(block_index)];
            block.tensor_offset = wide_tokens.size();
            for (int local_source = 0;
                 local_source < block.source_count; ++local_source) {
                const int source = block.source_begin + local_source;
                for (int local_target = 0;
                     local_target < target_count; ++local_target) {
                    const int target = target_begin + local_target;
                    const bool is_self = !target_source_indices.empty() &&
                        source == target_source_indices[
                            static_cast<std::size_t>(target)];
                    if (is_self) {
                        wide_tokens.push_back(0U);
                        continue;
                    }
                    const std::size_t index = source_block.tensor_offset +
                        static_cast<std::size_t>(local_target) *
                            block.source_count +
                        static_cast<std::size_t>(local_source);
                    std::array<double, 6> tensor{};
                    for (int component = 0; component < 6; ++component) {
                        const double value = leaf.tensors[
                            static_cast<std::size_t>(component)][index];
                        tensor[static_cast<std::size_t>(component)] =
                            value == 0.0 ? 0.0 : value;
                    }
                    const auto key = tensor6_bit_key(tensor);
                    const auto [iterator, inserted] = variant_ids.try_emplace(
                        key, static_cast<std::uint32_t>(variant_ids.size()));
                    if (inserted) {
                        for (int component = 0; component < 6; ++component) {
                            result.tensors[static_cast<std::size_t>(component)]
                                .push_back(tensor[
                                    static_cast<std::size_t>(component)]);
                        }
                    }
                    wide_tokens.push_back(iterator->second);
                }
            }
        }
    }
    plan_detail::frequency_order_signed_variants<double>(
        result, wide_tokens, 0U);
    if (result.variant_count() <= 255U) {
        result.token_width_bytes = 1;
        result.tokens8.assign(wide_tokens.begin(), wide_tokens.end());
    } else if (result.variant_count() <= 65535U) {
        result.token_width_bytes = 2;
        result.tokens16.assign(wide_tokens.begin(), wide_tokens.end());
    } else {
        result.token_width_bytes = 4;
        result.tokens32 = std::move(wide_tokens);
    }
    return result;
}

} // namespace cdfmm
