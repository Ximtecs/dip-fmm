// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <vector>

namespace cdfmm {

/** @brief One non-zero entry of a fixed sparse coefficient map. */
struct StaticOperatorEntry {
    int output{0};
    int input{0};
    double value{0.0};
};

/** @brief Immutable FP64 sparse coefficient data prepared for fixed geometry. */
struct StaticCoefficientOperator {
    int input_size{0};
    int output_size{0};
    std::vector<StaticOperatorEntry> entries{};
};

/** @brief FP32 counterpart of one sparse static coefficient entry. */
struct FloatStaticOperatorEntry {
    int output{0};
    int input{0};
    float value{0.0F};
};

/** @brief Immutable FP32 sparse coefficient data with no retained FP64 values. */
struct FloatStaticCoefficientOperator {
    int input_size{0};
    int output_size{0};
    std::vector<FloatStaticOperatorEntry> entries{};
};

/** @brief Immutable leaf index and geometry-specific P2M coefficient map. */
struct P2MPlan {
    int leaf{0};
    std::size_t begin{0};
    std::size_t count{0};
    StaticCoefficientOperator operator_map{};
};

/** @brief FP32 leaf index and quantised P2M coefficient map. */
struct FloatP2MPlan {
    int leaf{0};
    std::size_t begin{0};
    std::size_t count{0};
    FloatStaticCoefficientOperator operator_map{};
};

} // namespace cdfmm
