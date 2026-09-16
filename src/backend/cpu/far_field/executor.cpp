// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/backend/cpu/far_field.hpp"

#include "entries.hpp"

namespace cdfmm {

void apply_static_operator(const StaticCoefficientOperator& operator_map,
                           const std::span<const double> input,
                           const std::span<double> output) {
  detail::cpu::apply_static_operator(operator_map, input, output);
}

void apply_static_operator(const FloatStaticCoefficientOperator& operator_map,
                           const std::span<const float> input,
                           const std::span<float> output) {
  detail::cpu::apply_static_operator(operator_map, input, output);
}

void apply_static_coefficient_matrix(const std::span<const double> matrix,
                                    const std::span<const double> input,
                                    const std::span<double> output) {
  detail::cpu::apply_static_coefficient_matrix(matrix, input, output);
}

PotentialField apply_static_l2p_evaluator(
    const StaticL2PEvaluator& evaluator, const std::span<const double> local,
    const OutputFlags output) {
  return detail::cpu::apply_static_l2p_evaluator(evaluator, local, output);
}

FloatPotentialField apply_static_l2p_evaluator(
    const FloatStaticL2PEvaluator& evaluator, const std::span<const float> local,
    const OutputFlags output) {
  return detail::cpu::apply_static_l2p_evaluator(evaluator, local, output);
}

} // namespace cdfmm
