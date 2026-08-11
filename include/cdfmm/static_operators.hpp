// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <array>
#include <vector>

#include "cdfmm/operators.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Public types
//------------------------------------------------------------------------------

/** @brief One non-zero entry of a geometry-dependent coefficient map. */
struct StaticOperatorEntry {
    int output{0};
    int input{0};
    double value{0.0};
};

/** @brief Compact exact representation of a sparse linear operator. */
struct StaticCoefficientOperator {
    int input_size{0};
    int output_size{0};
    std::vector<StaticOperatorEntry> entries{};
};

/** @brief Precomputed potential and field rows for one fixed target offset. */
struct StaticL2PEvaluator {
    std::vector<double> potential{};
    std::array<std::vector<double>, 3> field{};
};

//------------------------------------------------------------------------------
// Canonical static mathematical operators
//------------------------------------------------------------------------------

/**
 * @brief Builds the exact dense Cartesian M2L matrix for one displacement.
 *
 * The column-major matrix satisfies L += T(R) M and is shared by execution
 * packings rather than re-derived by individual CPU or CUDA backends.
 */
[[nodiscard]] std::vector<double> build_static_m2l_matrix(
    const MultiIndexSet& basis,
    const Vec3& R
);

/** @brief Builds M = P m for fixed source positions and expansion centre. */
[[nodiscard]] StaticCoefficientOperator build_static_p2m_operator(
    const MultiIndexSet& basis,
    const Vec3& centre,
    std::span<const Vec3> source_positions
);

/** @brief Builds the triangular map M_parent += A(d) M_child. */
[[nodiscard]] StaticCoefficientOperator build_static_m2m_operator(
    const MultiIndexSet& basis,
    const Vec3& d
);

/** @brief Builds the triangular map L_child += B(d) L_parent. */
[[nodiscard]] StaticCoefficientOperator build_static_l2l_operator(
    const MultiIndexSet& basis,
    const Vec3& d
);

/** @brief Builds fixed rows mapping a local expansion to phi and H. */
[[nodiscard]] StaticL2PEvaluator build_static_l2p_evaluator(
    const MultiIndexSet& basis,
    const Vec3& centre,
    const Vec3& target
);

/** @brief Applies a compact static operator additively to an output vector. */
void apply_static_operator(
    const StaticCoefficientOperator& operator_map,
    std::span<const double> input,
    std::span<double> output
);

/** @brief Applies a fixed local evaluator without unused potential work. */
[[nodiscard]] PotentialField apply_static_l2p_evaluator(
    const StaticL2PEvaluator& evaluator,
    std::span<const double> L,
    OutputFlags output = OutputFlags::Field
);

/** @brief Applies a column-major square static coefficient operator. */
void apply_static_coefficient_matrix(
    std::span<const double> matrix,
    std::span<const double> input,
    std::span<double> output
);

} // namespace cdfmm
