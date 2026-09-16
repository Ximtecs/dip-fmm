// SPDX-License-Identifier: Apache-2.0

#include "packing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace cdfmm::detail::cpu {

namespace {

template <typename Operator, typename Scalar>
PackedTranslationBank<Scalar> pack_translation_bank_impl(
    const std::span<const Operator> operators,
    const std::span<const int> degrees, const int level_count) {
  if (operators.size() != 8) {
    throw std::invalid_argument(
        "translation packing requires the eight child-class operators");
  }
  PackedTranslationBank<Scalar> bank;
  const int n = static_cast<int>(degrees.size());
  bank.coefficient_count = n;
  bank.level_count = std::max(level_count, 0);

  // Column structure: the contiguous output range of every input column.
  for (int child_class = 0; child_class < 8; ++child_class) {
    const auto index = static_cast<std::size_t>(child_class);
    const Operator& operator_map = operators[index];
    if (operator_map.input_size != n || operator_map.output_size != n) {
      throw std::invalid_argument(
          "translation operator dimensions do not match the basis");
    }
    std::vector<int> begin(static_cast<std::size_t>(n), n);
    std::vector<int> end(static_cast<std::size_t>(n), 0);
    for (const auto& entry : operator_map.entries) {
      const auto in = static_cast<std::size_t>(entry.input);
      begin[in] = std::min(begin[in], entry.output);
      end[in] = std::max(end[in], entry.output + 1);
    }
    std::vector<std::size_t> offset(static_cast<std::size_t>(n) + 1, 0);
    for (int in = 0; in < n; ++in) {
      const auto slot = static_cast<std::size_t>(in);
      if (end[slot] <= begin[slot]) {
        begin[slot] = 0;
        end[slot] = 0;
      }
      offset[slot + 1] =
          offset[slot] + static_cast<std::size_t>(end[slot] - begin[slot]);
    }
    bank.class_values[index] = offset[static_cast<std::size_t>(n)];
    bank.column_begin[index] = std::move(begin);
    bank.column_end[index] = std::move(end);
    bank.column_offset[index] = std::move(offset);
  }

  // One value block per (child level, class); level 1 carries the unscaled
  // operator and deeper levels fold in the exact power-of-two degree factor.
  bank.block_offset.assign(static_cast<std::size_t>(bank.level_count) * 8 + 1,
                           0);
  for (int level = 1; level <= bank.level_count; ++level) {
    for (int child_class = 0; child_class < 8; ++child_class) {
      const std::size_t block =
          static_cast<std::size_t>(level - 1) * 8 +
          static_cast<std::size_t>(child_class);
      bank.block_offset[block + 1] =
          bank.block_offset[block] +
          bank.class_values[static_cast<std::size_t>(child_class)];
    }
  }
  bank.values.assign(bank.block_offset.back(), Scalar{0});
  for (int level = 1; level <= bank.level_count; ++level) {
    for (int child_class = 0; child_class < 8; ++child_class) {
      const auto index = static_cast<std::size_t>(child_class);
      Scalar* block = bank.values.data() +
                      bank.block_offset[static_cast<std::size_t>(level - 1) * 8 +
                                        index];
      for (const auto& entry : operators[index].entries) {
        const auto in = static_cast<std::size_t>(entry.input);
        const int power = std::abs(degrees[static_cast<std::size_t>(
                                       entry.output)] -
                                   degrees[in]);
        const Scalar value = std::ldexp(static_cast<Scalar>(entry.value),
                                        -(level - 1) * power);
        block[bank.column_offset[index][in] +
              static_cast<std::size_t>(entry.output -
                                       bank.column_begin[index][in])] += value;
      }
    }
  }
  return bank;
}

template <typename Plan, typename Scalar>
PackedP2M<Scalar> pack_p2m_impl(const std::span<const Plan> plans,
                                const std::size_t source_count,
                                const int coefficient_count) {
  PackedP2M<Scalar> packing;
  packing.coefficient_count = coefficient_count;
  const auto n = static_cast<std::size_t>(coefficient_count);
  packing.values.assign(source_count * 3 * n, Scalar{0});
  for (const Plan& plan : plans) {
    if (plan.operator_map.output_size != coefficient_count ||
        plan.operator_map.input_size !=
            static_cast<int>(3 * plan.count)) {
      throw std::invalid_argument("P2M plan dimensions are inconsistent");
    }
    Scalar* leaf = packing.values.data() + plan.begin * 3 * n;
    for (const auto& entry : plan.operator_map.entries) {
      // `input` enumerates (local source, moment component) as 3 * s + k.
      leaf[static_cast<std::size_t>(entry.input) * n +
           static_cast<std::size_t>(entry.output)] +=
          static_cast<Scalar>(entry.value);
    }
  }
  return packing;
}

template <typename Evaluator, typename Scalar>
PackedL2P<Scalar> pack_l2p_impl(const std::span<const Evaluator> evaluators,
                                const int coefficient_count) {
  PackedL2P<Scalar> packing;
  packing.coefficient_count = coefficient_count;
  const auto n = static_cast<std::size_t>(coefficient_count);
  packing.field.assign(evaluators.size() * 3 * n, Scalar{0});
  packing.potential.assign(evaluators.size() * n, Scalar{0});
  for (std::size_t target = 0; target < evaluators.size(); ++target) {
    const Evaluator& evaluator = evaluators[target];
    if (evaluator.potential.size() != n || evaluator.field[0].size() != n ||
        evaluator.field[1].size() != n || evaluator.field[2].size() != n) {
      throw std::invalid_argument("L2P evaluator dimensions are inconsistent");
    }
    std::copy(evaluator.potential.begin(), evaluator.potential.end(),
              packing.potential.begin() +
                  static_cast<std::ptrdiff_t>(target * n));
    for (std::size_t component = 0; component < 3; ++component) {
      std::copy(evaluator.field[component].begin(),
                evaluator.field[component].end(),
                packing.field.begin() +
                    static_cast<std::ptrdiff_t>((target * 3 + component) * n));
    }
  }
  return packing;
}

} // namespace

PackedTranslationBank<double> pack_translation_bank(
    const std::span<const StaticCoefficientOperator> operators,
    const std::span<const int> degrees, const int level_count) {
  return pack_translation_bank_impl<StaticCoefficientOperator, double>(
      operators, degrees, level_count);
}

PackedTranslationBank<float> pack_translation_bank(
    const std::span<const FloatStaticCoefficientOperator> operators,
    const std::span<const int> degrees, const int level_count) {
  return pack_translation_bank_impl<FloatStaticCoefficientOperator, float>(
      operators, degrees, level_count);
}

PackedP2M<double> pack_p2m(const std::span<const P2MPlan> plans,
                           const std::size_t source_count,
                           const int coefficient_count) {
  return pack_p2m_impl<P2MPlan, double>(plans, source_count,
                                        coefficient_count);
}

PackedP2M<float> pack_p2m(const std::span<const FloatP2MPlan> plans,
                          const std::size_t source_count,
                          const int coefficient_count) {
  return pack_p2m_impl<FloatP2MPlan, float>(plans, source_count,
                                            coefficient_count);
}

PackedL2P<double> pack_l2p(const std::span<const StaticL2PEvaluator> evaluators,
                           const int coefficient_count) {
  return pack_l2p_impl<StaticL2PEvaluator, double>(evaluators,
                                                   coefficient_count);
}

PackedL2P<float> pack_l2p(
    const std::span<const FloatStaticL2PEvaluator> evaluators,
    const int coefficient_count) {
  return pack_l2p_impl<FloatStaticL2PEvaluator, float>(evaluators,
                                                       coefficient_count);
}

} // namespace cdfmm::detail::cpu
