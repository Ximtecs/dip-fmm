// SPDX-License-Identifier: Apache-2.0
#pragma once

// Incremental construction of the signed Tensor6 dictionary.
//
// The dictionary is a sequence of dense leaf blocks in canonical order whose
// tensors are replaced by tokens into a table of distinct signed variants.
// Variants are numbered in first-seen order while the blocks are visited in
// canonical order, then renumbered by frequency once every token is known.
// Visiting the blocks in the same order in several appends therefore gives
// the same table and the same tokens as one append of all of them, which is
// what lets the plan build the near field one chunk of target leaves at a
// time instead of materialising every pair tensor first.

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "cdfmm/plan/p2p/leaf.hpp"
#include "cdfmm/plan/p2p/signed_dictionary.hpp"
#include "cdfmm/plan/p2p/tensor_dictionary.hpp"
#include "dictionary_detail.hpp"

namespace cdfmm::detail {

class SignedTensorDictionaryBuilder {
public:
  /// `target_source_indices` (target -> source, or -1) encodes fixed point
  /// self pairs as the zero variant; it is either empty or one per target.
  SignedTensorDictionaryBuilder(int source_count, int target_count,
                                std::span<const int> target_source_indices,
                                int target_tile_size);

  /// Appends the next target leaves in canonical order. `leaf` holds dense
  /// blocks of consecutive target leaves, exactly as
  /// `build_static_p2p_leaf_plan` produces them.
  void append(const StaticP2PLeafPlan &leaf);

  /// Renumbers the variants by frequency and narrows the tokens.
  [[nodiscard]] StaticP2PSignedTensorDictionaryPlan finish();

private:
  StaticP2PSignedTensorDictionaryPlan result_{};
  std::span<const int> target_source_indices_{};
  std::unordered_map<Tensor6BitKey<double>, std::uint32_t,
                     plan_detail::Tensor6BitKeyHash<double>>
      variant_ids_{};
  // Tokens are kept at two bytes while every variant id fits (at most 65536
  // variants) and widened once if one does not, so the construction peak is
  // two bytes per pair for every compressing dictionary.
  void push_token(std::uint32_t token);
  [[nodiscard]] std::size_t token_count() const noexcept;

  std::vector<std::uint16_t> narrow_tokens_{};
  std::vector<std::uint32_t> wide_tokens_{};
  bool wide_{false};
};

} // namespace cdfmm::detail
