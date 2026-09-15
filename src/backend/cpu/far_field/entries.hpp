// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>

#include "cdfmm/plan/static_plan.hpp"

namespace cdfmm::detail::cpu {

inline void apply_static_p2m_entries(
    const StaticCoefficientOperator& operator_map,
    const std::span<const Vec3> moments, const std::span<double> multipole) {
  for (const StaticOperatorEntry& entry : operator_map.entries) {
    const Vec3& moment = moments[static_cast<std::size_t>(entry.input / 3)];
    const double component = entry.input % 3 == 0
        ? moment.x : (entry.input % 3 == 1 ? moment.y : moment.z);
    multipole[static_cast<std::size_t>(entry.output)] +=
        entry.value * component;
  }
}

inline void apply_static_p2m_entries(
    const FloatStaticCoefficientOperator& operator_map,
    const std::span<const FloatVec3> moments,
    const std::span<float> multipole) {
  for (const FloatStaticOperatorEntry& entry : operator_map.entries) {
    const FloatVec3& moment = moments[static_cast<std::size_t>(entry.input / 3)];
    const float component = entry.input % 3 == 0
        ? moment.x : (entry.input % 3 == 1 ? moment.y : moment.z);
    multipole[static_cast<std::size_t>(entry.output)] +=
        entry.value * component;
  }
}

inline void apply_static_operator(
    const StaticCoefficientOperator& operator_map,
    const std::span<const double> input, const std::span<double> output) {
  if (input.size() != static_cast<std::size_t>(operator_map.input_size) ||
      output.size() != static_cast<std::size_t>(operator_map.output_size)) {
    throw std::invalid_argument("static operator dimensions are inconsistent");
  }
  for (const StaticOperatorEntry& entry : operator_map.entries) {
    output[static_cast<std::size_t>(entry.output)] += entry.value *
        input[static_cast<std::size_t>(entry.input)];
  }
}

inline void apply_static_operator(
    const FloatStaticCoefficientOperator& operator_map,
    const std::span<const float> input, const std::span<float> output) {
  if (input.size() != static_cast<std::size_t>(operator_map.input_size) ||
      output.size() != static_cast<std::size_t>(operator_map.output_size)) {
    throw std::invalid_argument("static FP32 operator dimensions are inconsistent");
  }
  for (const FloatStaticOperatorEntry& entry : operator_map.entries) {
    output[static_cast<std::size_t>(entry.output)] +=
        entry.value * input[static_cast<std::size_t>(entry.input)];
  }
}

inline void apply_static_coefficient_matrix(
    const std::span<const double> matrix, const std::span<const double> input,
    const std::span<double> output) {
  const std::size_t coefficient_count = input.size();
  if (output.size() != coefficient_count ||
      matrix.size() != coefficient_count * coefficient_count) {
    throw std::invalid_argument(
        "static coefficient matrix dimensions are inconsistent");
  }
  for (std::size_t alpha = 0; alpha < coefficient_count; ++alpha) {
    for (std::size_t beta = 0; beta < coefficient_count; ++beta) {
      output[beta] += matrix[beta + coefficient_count * alpha] * input[alpha];
    }
  }
}

inline PotentialField apply_static_l2p_evaluator(
    const StaticL2PEvaluator& evaluator, const std::span<const double> L,
    const OutputFlags output) {
  if (evaluator.potential.size() != L.size()) {
    throw std::invalid_argument("static L2P dimensions are inconsistent");
  }
  PotentialField result;
  for (std::size_t index = 0; index < L.size(); ++index) {
    if (has_flag(output, OutputFlags::Potential)) {
      result.phi += evaluator.potential[index] * L[index];
    }
    if (has_flag(output, OutputFlags::Field)) {
      result.H.x += evaluator.field[0][index] * L[index];
      result.H.y += evaluator.field[1][index] * L[index];
      result.H.z += evaluator.field[2][index] * L[index];
    }
  }
  return result;
}

inline FloatPotentialField apply_static_l2p_evaluator(
    const FloatStaticL2PEvaluator& evaluator, const std::span<const float> L,
    const OutputFlags output) {
  if (evaluator.potential.size() != L.size()) {
    throw std::invalid_argument("static FP32 L2P dimensions are inconsistent");
  }
  FloatPotentialField result;
  for (std::size_t index = 0; index < L.size(); ++index) {
    if (has_flag(output, OutputFlags::Potential)) {
      result.phi += evaluator.potential[index] * L[index];
    }
    if (has_flag(output, OutputFlags::Field)) {
      result.H.x += evaluator.field[0][index] * L[index];
      result.H.y += evaluator.field[1][index] * L[index];
      result.H.z += evaluator.field[2][index] * L[index];
    }
  }
  return result;
}

} // namespace cdfmm::detail::cpu
