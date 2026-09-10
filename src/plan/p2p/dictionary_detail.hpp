// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cdfmm/tensor_dictionary.hpp"

namespace cdfmm::plan_detail {

template <typename Scalar>
struct Tensor6BitKeyHash {
    std::size_t operator()(const Tensor6BitKey<Scalar>& key) const noexcept
    {
        std::size_t hash = 0;
        for (const auto value : key.values) {
            hash ^= std::hash<typename Tensor6BitKey<Scalar>::Bits>{}(value) +
                0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        }
        return hash;
    }
};

template <typename Scalar, typename DictionaryPlan, typename LeafPlan>
DictionaryPlan build_tensor_dictionary_from_leaf(
    const LeafPlan& leaf,
    const bool skip_for_identity)
{
    DictionaryPlan result;
    result.source_count = leaf.source_count;
    result.target_count = leaf.target_count;
    result.target_begins = leaf.target_begins;
    result.target_counts = leaf.target_counts;
    result.leaf_row_offsets = leaf.leaf_row_offsets;
    result.blocks = leaf.blocks;
    result.skip_for_identity = skip_for_identity;
    result.tokens.reserve(leaf.tensors[0].size());

    std::unordered_map<Tensor6BitKey<Scalar>, std::uint32_t,
                       Tensor6BitKeyHash<Scalar>> dictionary_ids;
    dictionary_ids.reserve(leaf.tensors[0].size());
    std::vector<std::size_t> frequencies;
    for (std::size_t interaction = 0;
         interaction < leaf.tensors[0].size(); ++interaction) {
        const std::array<Scalar, 6> tensor{
            leaf.tensors[0][interaction], leaf.tensors[1][interaction],
            leaf.tensors[2][interaction], leaf.tensors[3][interaction],
            leaf.tensors[4][interaction], leaf.tensors[5][interaction]};
        const auto canonical = canonicalise_tensor6(tensor);
        const auto key = tensor6_bit_key(canonical.values);
        const auto [iterator, inserted] = dictionary_ids.try_emplace(
            key, static_cast<std::uint32_t>(dictionary_ids.size()));
        if (inserted) {
            if (iterator->second >= (1U << 26U)) {
                throw std::overflow_error(
                    "Tensor6 dictionary exceeds packed token IDs");
            }
            for (int component = 0; component < 6; ++component) {
                result.tensors[static_cast<std::size_t>(component)].push_back(
                    canonical.values[static_cast<std::size_t>(component)]);
            }
            frequencies.push_back(0);
        }
        ++frequencies[iterator->second];
        result.tokens.push_back(
            pack_tensor6_token(iterator->second, canonical.sign_mask));
    }

    // Frequency ordering is stable, so equal-frequency entries retain their
    // deterministic first-seen order without changing interaction order.
    std::vector<std::uint32_t> order(frequencies.size());
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(), [&](const auto left,
                                                     const auto right) {
        return frequencies[left] > frequencies[right];
    });
    std::vector<std::uint32_t> remap(order.size());
    std::array<std::vector<Scalar>, 6> reordered;
    for (std::uint32_t new_id = 0; new_id < order.size(); ++new_id) {
        const std::uint32_t old_id = order[new_id];
        remap[old_id] = new_id;
        for (int component = 0; component < 6; ++component) {
            reordered[static_cast<std::size_t>(component)].push_back(
                result.tensors[static_cast<std::size_t>(component)][old_id]);
        }
    }
    result.tensors = std::move(reordered);
    for (std::uint32_t& token : result.tokens) {
        token = pack_tensor6_token(remap[tensor6_token_id(token)],
                                   tensor6_token_sign_mask(token));
    }
    return result;
}

template <typename Scalar, typename SignedPlan>
void frequency_order_signed_variants(
    SignedPlan& plan,
    std::vector<std::uint32_t>& tokens,
    const std::uint32_t old_zero_variant)
{
    std::vector<std::size_t> frequencies(plan.variant_count(), 0);
    for (const std::uint32_t token : tokens) {
        ++frequencies[token];
    }
    std::vector<std::uint32_t> order(frequencies.size());
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(), [&](const auto left,
                                                     const auto right) {
        return frequencies[left] > frequencies[right];
    });

    std::vector<std::uint32_t> remap(order.size());
    std::array<std::vector<Scalar>, 6> reordered;
    for (auto& component : reordered) {
        component.reserve(order.size());
    }
    for (std::uint32_t new_id = 0; new_id < order.size(); ++new_id) {
        const std::uint32_t old_id = order[new_id];
        remap[old_id] = new_id;
        for (int component = 0; component < 6; ++component) {
            reordered[static_cast<std::size_t>(component)].push_back(
                plan.tensors[static_cast<std::size_t>(component)][old_id]);
        }
    }
    plan.tensors = std::move(reordered);
    for (std::uint32_t& token : tokens) {
        token = remap[token];
    }
    plan.zero_variant_id = remap[old_zero_variant];
}

} // namespace cdfmm::plan_detail
