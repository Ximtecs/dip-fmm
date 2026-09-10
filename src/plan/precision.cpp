// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/plan/precision.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cdfmm/tensor_dictionary.hpp"
#include "p2p/dictionary_detail.hpp"

namespace cdfmm {

FloatStaticCoefficientOperator
quantise_static_operator(const StaticCoefficientOperator& source)
{
    FloatStaticCoefficientOperator result;
    result.input_size = source.input_size;
    result.output_size = source.output_size;
    result.entries.reserve(source.entries.size());
    for (const StaticOperatorEntry& entry : source.entries) {
        result.entries.push_back(
            {entry.output, entry.input, static_cast<float>(entry.value)});
    }
    return result;
}

FloatStaticL2PEvaluator
quantise_static_l2p_evaluator(const StaticL2PEvaluator& source)
{
    FloatStaticL2PEvaluator result;
    result.potential.assign(source.potential.begin(), source.potential.end());
    for (std::size_t component = 0; component < 3; ++component) {
        result.field[component].assign(source.field[component].begin(),
                                       source.field[component].end());
    }
    return result;
}

FloatStaticP2POperator
quantise_static_p2p_operator(const StaticP2POperator& source)
{
    FloatStaticP2POperator result;
    result.source_count = source.source_count;
    result.target_count = source.target_count;
    result.row_offsets = source.row_offsets;
    result.blocks.reserve(source.blocks.size());
    for (const StaticDipoleBlock& block : source.blocks) {
        result.blocks.push_back({
            block.target, block.source, static_cast<float>(block.px),
            static_cast<float>(block.py), static_cast<float>(block.pz),
            static_cast<float>(block.xx), static_cast<float>(block.xy),
            static_cast<float>(block.xz), static_cast<float>(block.yy),
            static_cast<float>(block.yz), static_cast<float>(block.zz),
            block.skip_for_identity});
    }
    return result;
}

FloatStaticP2PCompactPlan
quantise_static_p2p_compact_plan(const StaticP2PCompactPlan& source)
{
    FloatStaticP2PCompactPlan result;
    result.source_count = source.source_count;
    result.target_count = source.target_count;
    result.row_offsets = source.row_offsets;
    result.source_indices = source.source_indices;
    result.skip_for_identity = source.skip_for_identity;
    for (std::size_t component = 0; component < 3; ++component) {
        result.potential[component].assign(
            source.potential[component].begin(),
            source.potential[component].end());
    }
    for (std::size_t component = 0; component < 6; ++component) {
        result.tensors[component].assign(source.tensors[component].begin(),
                                         source.tensors[component].end());
    }
    return result;
}

FloatStaticP2PLeafPlan
quantise_static_p2p_leaf_plan(const StaticP2PLeafPlan& source)
{
    FloatStaticP2PLeafPlan result;
    result.source_count = source.source_count;
    result.target_count = source.target_count;
    result.target_begins = source.target_begins;
    result.target_counts = source.target_counts;
    result.leaf_row_offsets = source.leaf_row_offsets;
    result.blocks = source.blocks;
    for (std::size_t component = 0; component < 6; ++component) {
        result.tensors[component].assign(source.tensors[component].begin(),
                                         source.tensors[component].end());
    }
    result.minimum_occupancy = source.minimum_occupancy;
    result.maximum_occupancy = source.maximum_occupancy;
    result.mean_occupancy = source.mean_occupancy;
    result.unique_occupancies = source.unique_occupancies;
    result.uniform_occupancy = source.uniform_occupancy;
    return result;
}

FloatStaticP2PTensorDictionaryPlan
quantise_static_p2p_tensor_dictionary_plan(
    const StaticP2PTensorDictionaryPlan& source)
{
    FloatStaticP2PTensorDictionaryPlan result;
    result.source_count = source.source_count;
    result.target_count = source.target_count;
    result.target_begins = source.target_begins;
    result.target_counts = source.target_counts;
    result.leaf_row_offsets = source.leaf_row_offsets;
    result.blocks = source.blocks;
    result.skip_for_identity = source.skip_for_identity;
    result.tokens.reserve(source.tokens.size());
    std::unordered_map<Tensor6BitKey<float>, std::uint32_t,
                       plan_detail::Tensor6BitKeyHash<float>> dictionary_ids;
    dictionary_ids.reserve(source.tokens.size());
    for (const std::uint32_t source_token : source.tokens) {
        const std::uint32_t source_id = tensor6_token_id(source_token);
        const std::uint8_t source_signs =
            tensor6_token_sign_mask(source_token);
        std::array<float, 6> tensor{};
        for (int component = 0; component < 6; ++component) {
            const float magnitude = static_cast<float>(source.tensors[
                static_cast<std::size_t>(component)][source_id]);
            tensor[static_cast<std::size_t>(component)] =
                (source_signs & (1U << component)) != 0U
                    ? -magnitude : magnitude;
        }
        const auto canonical = canonicalise_tensor6(tensor);
        const auto key = tensor6_bit_key(canonical.values);
        const auto [iterator, inserted] = dictionary_ids.try_emplace(
            key, static_cast<std::uint32_t>(dictionary_ids.size()));
        if (inserted) {
            if (iterator->second >= (1U << 26U)) {
                throw std::overflow_error(
                    "FP32 Tensor6 dictionary exceeds packed token IDs");
            }
            for (int component = 0; component < 6; ++component) {
                result.tensors[static_cast<std::size_t>(component)].push_back(
                    canonical.values[static_cast<std::size_t>(component)]);
            }
        }
        result.tokens.push_back(
            pack_tensor6_token(iterator->second, canonical.sign_mask));
    }
    return result;
}

FloatStaticP2PSignedTensorDictionaryPlan
quantise_static_p2p_signed_tensor_dictionary_plan(
    const StaticP2PSignedTensorDictionaryPlan& source)
{
    FloatStaticP2PSignedTensorDictionaryPlan result;
    result.source_count = source.source_count;
    result.target_count = source.target_count;
    result.target_begins = source.target_begins;
    result.target_counts = source.target_counts;
    result.leaf_row_offsets = source.leaf_row_offsets;
    result.blocks = source.blocks;
    result.tile_leaf_indices = source.tile_leaf_indices;
    result.tile_target_offsets = source.tile_target_offsets;
    result.target_tile_size = source.target_tile_size;
    for (auto& component : result.tensors) {
        component.push_back(0.0F);
    }
    std::unordered_map<Tensor6BitKey<float>, std::uint32_t,
                       plan_detail::Tensor6BitKeyHash<float>> variant_ids;
    variant_ids.emplace(tensor6_bit_key(std::array<float, 6>{}), 0U);
    std::vector<std::uint32_t> wide_tokens;
    wide_tokens.reserve(source.token_count());
    const auto append_token = [&](const std::uint32_t source_variant) {
        std::array<float, 6> tensor{};
        for (int component = 0; component < 6; ++component) {
            const float value = static_cast<float>(source.tensors[
                static_cast<std::size_t>(component)][source_variant]);
            tensor[static_cast<std::size_t>(component)] =
                value == 0.0F ? 0.0F : value;
        }
        const auto [iterator, inserted] = variant_ids.try_emplace(
            tensor6_bit_key(tensor),
            static_cast<std::uint32_t>(variant_ids.size()));
        if (inserted) {
            for (int component = 0; component < 6; ++component) {
                result.tensors[static_cast<std::size_t>(component)].push_back(
                    tensor[static_cast<std::size_t>(component)]);
            }
        }
        wide_tokens.push_back(iterator->second);
    };
    if (source.token_width_bytes == 1) {
        for (const std::uint8_t token : source.tokens8) {
            append_token(token);
        }
    } else if (source.token_width_bytes == 2) {
        for (const std::uint16_t token : source.tokens16) {
            append_token(token);
        }
    } else if (source.token_width_bytes == 4) {
        for (const std::uint32_t token : source.tokens32) {
            append_token(token);
        }
    } else {
        throw std::invalid_argument(
            "signed Tensor6 dictionary token width is invalid");
    }
    plan_detail::frequency_order_signed_variants<float>(
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

FloatStaticP2PBsrPlan
quantise_static_p2p_bsr_plan(const StaticP2PBsrPlan& source)
{
    FloatStaticP2PBsrPlan result;
    result.source_count = source.source_count;
    result.target_count = source.target_count;
    result.row_offsets = source.row_offsets;
    result.source_indices = source.source_indices;
    result.values.assign(source.values.begin(), source.values.end());
    result.target_source_indices = source.target_source_indices;
    return result;
}

FloatStaticM2LPlan quantise_static_m2l_plan(const StaticM2LPlan& source)
{
    FloatStaticM2LPlan result;
    result.coefficient_count = source.coefficient_count;
    result.matrix_count = source.matrix_count;
    result.level_count = source.level_count;
    result.matrices.assign(source.matrices.begin(), source.matrices.end());
    result.multipole_scaling.assign(source.multipole_scaling.begin(),
                                    source.multipole_scaling.end());
    result.local_scaling.assign(source.local_scaling.begin(),
                                source.local_scaling.end());
    result.target_row_offsets = source.target_row_offsets;
    result.source_nodes = source.source_nodes;
    result.matrix_ids = source.matrix_ids;
    result.source_levels = source.source_levels;
    result.target_levels = source.target_levels;
    result.interaction_levels = source.interaction_levels;
    result.level_target_begin = source.level_target_begin;
    result.level_target_end = source.level_target_end;
    result.target_level_offsets = source.target_level_offsets;
    result.target_nodes_by_level = source.target_nodes_by_level;
    result.node_levels = source.node_levels;
    return result;
}

} // namespace cdfmm
