// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cdfmm/plan/l2p.hpp"
#include "cdfmm/plan/m2l.hpp"
#include "cdfmm/plan/p2p/bsr.hpp"
#include "cdfmm/plan/p2p/compact.hpp"
#include "cdfmm/plan/p2p/dictionary.hpp"
#include "cdfmm/plan/p2p/signed_dictionary.hpp"
#include "cdfmm/plan/static_coefficient.hpp"

namespace cdfmm {

/** @brief Converts FP64 static coefficient data once at plan construction. */
[[nodiscard]] FloatStaticCoefficientOperator quantise_static_operator(
    const StaticCoefficientOperator& source);

/** @brief Converts fixed FP64 L2P rows once at plan construction. */
[[nodiscard]] FloatStaticL2PEvaluator quantise_static_l2p_evaluator(
    const StaticL2PEvaluator& source);

/** @brief Converts authoritative P2P rows to an authoritative FP32 copy. */
[[nodiscard]] FloatStaticP2POperator quantise_static_p2p_operator(
    const StaticP2POperator& source);

[[nodiscard]] FloatStaticP2PCompactPlan quantise_static_p2p_compact_plan(
    const StaticP2PCompactPlan& source);

[[nodiscard]] FloatStaticP2PLeafPlan quantise_static_p2p_leaf_plan(
    const StaticP2PLeafPlan& source);

[[nodiscard]] FloatStaticP2PTensorDictionaryPlan
quantise_static_p2p_tensor_dictionary_plan(
    const StaticP2PTensorDictionaryPlan& source);

[[nodiscard]] FloatStaticP2PSignedTensorDictionaryPlan
quantise_static_p2p_signed_tensor_dictionary_plan(
    const StaticP2PSignedTensorDictionaryPlan& source);

[[nodiscard]] FloatStaticP2PBsrPlan quantise_static_p2p_bsr_plan(
    const StaticP2PBsrPlan& source);

[[nodiscard]] FloatStaticM2LPlan quantise_static_m2l_plan(
    const StaticM2LPlan& source);

} // namespace cdfmm
