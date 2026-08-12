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

/** @brief Six independent entries of one fixed dipole interaction tensor. */
struct StaticDipoleBlock {
    int target{0};
    int source{0};
    double xx{0.0};
    double xy{0.0};
    double xz{0.0};
    double yy{0.0};
    double yz{0.0};
    double zz{0.0};
};

/** @brief Target-row representation of the exact sparse near-field operator. */
struct StaticP2POperator {
    int source_count{0};
    int target_count{0};
    std::vector<int> row_offsets{};
    std::vector<StaticDipoleBlock> blocks{};

    /// @brief Returns persistent storage used by the compact representation.
    [[nodiscard]] std::size_t memory_bytes() const noexcept;
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

/**
 * @brief Builds the exact sparse dipole tensor for an explicit interaction set.
 *
 * Interaction pairs contain sorted target and source indices. The caller is
 * responsible for supplying only list1 pairs; coordinate equality never
 * implies self identity.
 */
[[nodiscard]] StaticP2POperator build_static_p2p_operator(
    std::span<const Vec3> target_positions,
    std::span<const Vec3> source_positions,
    std::span<const std::array<int, 2>> interactions
);

/**
 * @brief Applies the compact static near-field tensor additively.
 *
 * `target_source_indices` is in the same sorted indexing as the operator. An
 * entry equal to a block source is skipped explicitly. Potential is not part
 * of this field-only tensor.
 */
void apply_static_p2p_operator(
    const StaticP2POperator& operator_map,
    std::span<const Vec3> dipole_moments,
    std::span<Vec3> H,
    std::span<const int> target_source_indices = {}
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
