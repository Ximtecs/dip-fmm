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
    // Identity handling travels with every dense block; the plan-level flag
    // only summarises whether any block omits identity pairs.
    bool skip_for_identity = false;
    for (const StaticP2PLeafBlock& block : leaf.blocks) {
        skip_for_identity = skip_for_identity || block.skip_for_identity != 0;
    }
    return plan_detail::build_tensor_dictionary_from_leaf<
        double, StaticP2PTensorDictionaryPlan>(leaf, skip_for_identity);
}

} // namespace cdfmm
