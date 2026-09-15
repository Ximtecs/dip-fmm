// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <cstddef>
#include <span>

namespace cdfmm::detail::cpu {

template <typename Operator, typename Scalar, typename Degree>
inline void apply_level_scaled_translation(const Operator& operator_map,
                                           const std::span<Scalar> input,
                                           const std::span<Scalar> output,
                                           const int level,
                                           const Degree& degree) {
  for (const auto& entry : operator_map.entries) {
    const int power = std::abs(degree(entry.output) - degree(entry.input));
    const Scalar value = std::ldexp(
        static_cast<Scalar>(entry.value), -(level - 1) * power);
    output[static_cast<std::size_t>(entry.output)] +=
        value * input[static_cast<std::size_t>(entry.input)];
  }
}

} // namespace cdfmm::detail::cpu
