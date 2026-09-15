// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>

#include "cdfmm/core/output_flags.hpp"
#include "cdfmm/math/potential_field.hpp"
#include "cdfmm/plan/static_plan.hpp"

namespace cdfmm {

void apply_static_operator(
    const StaticCoefficientOperator& operator_map, std::span<const double> input,
    std::span<double> output);
void apply_static_operator(
    const FloatStaticCoefficientOperator& operator_map,
    std::span<const float> input, std::span<float> output);
void apply_static_coefficient_matrix(
    std::span<const double> matrix, std::span<const double> input,
    std::span<double> output);

[[nodiscard]] PotentialField apply_static_l2p_evaluator(
    const StaticL2PEvaluator& evaluator, std::span<const double> local,
    OutputFlags output = OutputFlags::Field);
[[nodiscard]] FloatPotentialField apply_static_l2p_evaluator(
    const FloatStaticL2PEvaluator& evaluator, std::span<const float> local,
    OutputFlags output = OutputFlags::Field);

} // namespace cdfmm
