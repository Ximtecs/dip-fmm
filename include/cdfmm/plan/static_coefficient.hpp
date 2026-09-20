// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <vector>

namespace cdfmm {

/** @brief One non-zero entry of a fixed sparse coefficient map. */
struct StaticOperatorEntry {
    int output{0};      ///< Output coefficient index.
    int input{0};       ///< Input index (for P2M: `3 * source + component`).
    double value{0.0};  ///< Coefficient: `output += value * input`.
};

/** @brief Immutable FP64 sparse coefficient data prepared for fixed geometry. */
struct StaticCoefficientOperator {
    int input_size{0};                          ///< Length of the input vector.
    int output_size{0};                         ///< Length of the output vector.
    std::vector<StaticOperatorEntry> entries{}; ///< Non-zero entries in construction order.
};

/** @brief FP32 counterpart of one sparse static coefficient entry. */
struct FloatStaticOperatorEntry {
    int output{0};      ///< Output coefficient index.
    int input{0};       ///< Input index (for P2M: `3 * source + component`).
    float value{0.0F};  ///< Coefficient: `output += value * input`.
};

/** @brief Immutable FP32 sparse coefficient data with no retained FP64 values. */
struct FloatStaticCoefficientOperator {
    int input_size{0};                               ///< Length of the input vector.
    int output_size{0};                              ///< Length of the output vector.
    std::vector<FloatStaticOperatorEntry> entries{}; ///< Non-zero entries in construction order.
};

/** @brief Immutable leaf index and geometry-specific P2M coefficient map. */
struct P2MPlan {
    int leaf{0};                                ///< Compact node id of the source leaf.
    std::size_t begin{0};                       ///< First sorted source of the leaf.
    std::size_t count{0};                       ///< Number of sources in the leaf.
    StaticCoefficientOperator operator_map{};   ///< Leaf-local moments -> multipole coefficients.
};

/** @brief FP32 leaf index and quantised P2M coefficient map. */
struct FloatP2MPlan {
    int leaf{0};                                    ///< Compact node id of the source leaf.
    std::size_t begin{0};                           ///< First sorted source of the leaf.
    std::size_t count{0};                           ///< Number of sources in the leaf.
    FloatStaticCoefficientOperator operator_map{};  ///< Leaf-local moments -> multipole coefficients.
};

} // namespace cdfmm
