// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/plan/p2p/signed_dictionary.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cdfmm/plan/p2p/tensor_dictionary.hpp"
#include "dictionary_detail.hpp"
#include "signed_dictionary_builder.hpp"

namespace cdfmm {

namespace detail {

SignedTensorDictionaryBuilder::SignedTensorDictionaryBuilder(
    const int source_count, const int target_count,
    const std::span<const int> target_source_indices,
    const int target_tile_size)
    : target_source_indices_(target_source_indices)
{
    if (target_tile_size <= 0 || target_tile_size > 128) {
        throw std::invalid_argument(
            "signed Tensor6 target tile must be in [1, 128]");
    }
    if (!target_source_indices.empty() &&
        target_source_indices.size() !=
            static_cast<std::size_t>(target_count)) {
        throw std::invalid_argument(
            "signed Tensor6 dictionary identity dimensions are inconsistent");
    }
    result_.source_count = source_count;
    result_.target_count = target_count;
    result_.leaf_row_offsets.push_back(0);
    result_.target_tile_size = target_tile_size;

    // Fixed point self interactions become ordinary exact-zero lookups.  Only
    // blocks whose canonical interactions carry the identity marker encode
    // them; a finite-body self tensor is stored like any other value.
    for (auto& component : result_.tensors) {
        component.push_back(0.0);
    }
    variant_ids_.emplace(tensor6_bit_key(std::array<double, 6>{}), 0U);
}

void SignedTensorDictionaryBuilder::push_token(const std::uint32_t token)
{
    if (!wide_ && token > 0xFFFFU) {
        wide_tokens_.assign(narrow_tokens_.begin(), narrow_tokens_.end());
        narrow_tokens_ = {};
        wide_ = true;
    }
    if (wide_) {
        wide_tokens_.push_back(token);
    } else {
        narrow_tokens_.push_back(static_cast<std::uint16_t>(token));
    }
}

std::size_t SignedTensorDictionaryBuilder::token_count() const noexcept
{
    return wide_ ? wide_tokens_.size() : narrow_tokens_.size();
}

void SignedTensorDictionaryBuilder::append(const StaticP2PLeafPlan& leaf)
{
    if (leaf.source_count != result_.source_count ||
        leaf.target_count != result_.target_count) {
        throw std::invalid_argument(
            "signed Tensor6 dictionary chunk dimensions are inconsistent");
    }
    const int block_base = static_cast<int>(result_.blocks.size());
    for (int target_leaf = 0;
         target_leaf < static_cast<int>(leaf.target_begins.size());
         ++target_leaf) {
        const int target_begin =
            leaf.target_begins[static_cast<std::size_t>(target_leaf)];
        const int target_count =
            leaf.target_counts[static_cast<std::size_t>(target_leaf)];
        const int leaf_index = static_cast<int>(result_.target_begins.size());
        result_.target_begins.push_back(target_begin);
        result_.target_counts.push_back(target_count);
        for (int local_begin = 0; local_begin < target_count;
             local_begin += result_.target_tile_size) {
            result_.tile_leaf_indices.push_back(leaf_index);
            result_.tile_target_offsets.push_back(local_begin);
        }
        for (int block_index =
                 leaf.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
             block_index < leaf.leaf_row_offsets[
                 static_cast<std::size_t>(target_leaf) + 1]; ++block_index) {
            const StaticP2PLeafBlock& source_block =
                leaf.blocks[static_cast<std::size_t>(block_index)];
            StaticP2PLeafBlock block = source_block;
            block.tensor_offset = token_count();
            for (int local_source = 0;
                 local_source < block.source_count; ++local_source) {
                const int source = block.source_begin + local_source;
                for (int local_target = 0;
                     local_target < target_count; ++local_target) {
                    const int target = target_begin + local_target;
                    const bool is_self = block.skip_for_identity != 0 &&
                        !target_source_indices_.empty() &&
                        source == target_source_indices_[
                            static_cast<std::size_t>(target)];
                    if (is_self) {
                        push_token(0U);
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
                    const auto [iterator, inserted] = variant_ids_.try_emplace(
                        key, static_cast<std::uint32_t>(variant_ids_.size()));
                    if (inserted) {
                        for (int component = 0; component < 6; ++component) {
                            result_.tensors[static_cast<std::size_t>(component)]
                                .push_back(tensor[
                                    static_cast<std::size_t>(component)]);
                        }
                    }
                    push_token(iterator->second);
                }
            }
            result_.blocks.push_back(block);
        }
        result_.leaf_row_offsets.push_back(
            block_base +
            leaf.leaf_row_offsets[static_cast<std::size_t>(target_leaf) + 1]);
    }
}

StaticP2PSignedTensorDictionaryPlan SignedTensorDictionaryBuilder::finish()
{
    // Renumbering by frequency is the same map whatever the stored width, so
    // the narrowed tokens equal those of a four-byte build.
    if (wide_) {
        plan_detail::frequency_order_signed_variants<double>(
            result_, wide_tokens_, 0U);
    } else {
        plan_detail::frequency_order_signed_variants<double>(
            result_, narrow_tokens_, 0U);
    }
    const auto narrow_to = [&](auto& destination) {
        if (wide_) {
            destination.assign(wide_tokens_.begin(), wide_tokens_.end());
        } else {
            destination.assign(narrow_tokens_.begin(), narrow_tokens_.end());
        }
    };
    if (result_.variant_count() <= 255U) {
        result_.token_width_bytes = 1;
        narrow_to(result_.tokens8);
    } else if (result_.variant_count() <= 65535U) {
        result_.token_width_bytes = 2;
        if (wide_) {
            narrow_to(result_.tokens16);
        } else {
            result_.tokens16 = std::move(narrow_tokens_);
        }
    } else {
        result_.token_width_bytes = 4;
        if (wide_) {
            result_.tokens32 = std::move(wide_tokens_);
        } else {
            narrow_to(result_.tokens32);
        }
    }
    narrow_tokens_ = {};
    wide_tokens_ = {};
    variant_ids_ = {};
    return std::move(result_);
}

} // namespace detail

StaticP2PSignedTensorDictionaryPlan
build_static_p2p_signed_tensor_dictionary_plan(
    const StaticP2POperator& operator_map,
    const std::span<const StaticP2PLeafPair> leaf_pairs,
    const std::span<const int> target_source_indices,
    const int target_tile_size)
{
    detail::SignedTensorDictionaryBuilder builder(
        operator_map.source_count, operator_map.target_count,
        target_source_indices, target_tile_size);
    builder.append(build_static_p2p_leaf_plan(operator_map, leaf_pairs));
    return builder.finish();
}

} // namespace cdfmm
