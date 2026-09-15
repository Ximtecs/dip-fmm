// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/backend/cpu/far_field.hpp"

#include "internal.hpp"
#include "entries.hpp"
#include "translation.hpp"

namespace cdfmm {

namespace detail::cpu {

CoefficientDegreeView coefficient_degree_view(
    const bool spherical, const MultiIndexSet &cartesian,
    const SphericalHarmonicBasis &spherical_basis) noexcept {
  return spherical ? CoefficientDegreeView{nullptr, &spherical_basis}
                   : CoefficientDegreeView{&cartesian, nullptr};
}

void apply_static_p2m(const StaticCoefficientOperator& operator_map,
                      const std::span<const Vec3> moments,
                      const std::span<double> multipole) {
  apply_static_p2m_entries(operator_map, moments, multipole);
}

void apply_static_p2m(const FloatStaticCoefficientOperator& operator_map,
                      const std::span<const FloatVec3> moments,
                      const std::span<float> multipole) {
  apply_static_p2m_entries(operator_map, moments, multipole);
}

void apply_static_translation(const StaticCoefficientOperator &operator_map,
                              const std::span<double> input,
                              const std::span<double> output,
                              const int level,
                              const CoefficientDegreeView degree) {
  apply_level_scaled_translation(operator_map, input, output, level, degree);
}

void apply_static_translation(
    const FloatStaticCoefficientOperator &operator_map,
    const std::span<float> input, const std::span<float> output,
    const int level, const CoefficientDegreeView degree) {
  apply_level_scaled_translation(operator_map, input, output, level, degree);
}

} // namespace detail::cpu

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
