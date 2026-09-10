// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/plan/p2p/dictionary.hpp"

#include <stdexcept>

#include "dictionary_detail.hpp"

namespace cdfmm {

StaticP2PTensorDictionaryPlan build_static_p2p_tensor_dictionary_plan(
    const StaticP2POperator& operator_map,
    const std::span<const StaticP2PLeafPair> leaf_pairs)
{
    const StaticP2PLeafPlan leaf =
        build_static_p2p_leaf_plan(operator_map, leaf_pairs);
    bool skip_for_identity = false;
    bool first = true;
    for (const StaticDipoleBlock& block : operator_map.blocks) {
        const bool current = block.skip_for_identity != 0;
        if (!first && current != skip_for_identity) {
            throw std::invalid_argument(
                "Tensor6 dictionary requires a uniform self-identity policy");
        }
        skip_for_identity = current;
        first = false;
    }
    return plan_detail::build_tensor_dictionary_from_leaf<
        double, StaticP2PTensorDictionaryPlan>(leaf, skip_for_identity);
}

} // namespace cdfmm
