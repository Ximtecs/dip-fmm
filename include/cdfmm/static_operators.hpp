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

/**
 * @brief One non-zero entry of a geometry-dependent coefficient map.
 *
 * Indices address flattened input and output arrays. Entries are immutable
 * after plan construction and may be applied in any order when each output
 * has a single writer, or accumulated additively otherwise.
 */
struct StaticOperatorEntry {
    int output{0};
    int input{0};
    double value{0.0};
};

/**
 * @brief Compact exact representation of a sparse linear operator.
 *
 * This is mathematical operator data rather than evaluation scratch. The
 * entry list omits exact zeros from P2M and triangular translations and is
 * reused for every moment state associated with the fixed geometry.
 */
struct StaticCoefficientOperator {
    int input_size{0};
    int output_size{0};
    std::vector<StaticOperatorEntry> entries{};
};

/**
 * @brief Precomputed potential and field rows for one fixed target offset.
 *
 * All four arrays use `MultiIndexSet` coefficient order. The object is an
 * immutable CPU-side local-evaluation operator; CUDA-full repacks its non-zero
 * field entries during initialisation.
 */
struct StaticL2PEvaluator {
    std::vector<double> potential{};
    std::array<std::vector<double>, 3> field{};
};

/**
 * @brief Six independent entries of one fixed dipole interaction tensor.
 *
 * The target/source indices use Morton-sorted particle order. Symmetry stores
 * only xx, xy, xz, yy, yz and zz; the block contains immutable mathematical
 * data and no per-evaluation moment or field storage.
 */
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

#if defined(__CUDACC__)
#define CDFMM_HOST_DEVICE __host__ __device__
#else
#define CDFMM_HOST_DEVICE
#endif

/** @brief Accumulates one symmetric six-component dipole tensor product. */
CDFMM_HOST_DEVICE inline void accumulate_static_dipole_block(
    const StaticDipoleBlock& block,
    const Vec3& moment,
    Vec3& H
) noexcept {
    H.x += block.xx * moment.x + block.xy * moment.y + block.xz * moment.z;
    H.y += block.xy * moment.x + block.yy * moment.y + block.yz * moment.z;
    H.z += block.xz * moment.x + block.yz * moment.y + block.zz * moment.z;
}

#undef CDFMM_HOST_DEVICE

/**
 * @brief Target-row representation of the exact sparse near-field operator.
 *
 * `row_offsets[t]..row_offsets[t+1]` is the contiguous block range for sorted
 * target `t`. The representation is immutable after construction, is shared
 * by CPU and CUDA packings, and maps changing sorted moments to near fields.
 */
struct StaticP2POperator {
    int source_count{0};
    int target_count{0};
    std::vector<int> row_offsets{};
    std::vector<StaticDipoleBlock> blocks{};

    /// @brief Returns persistent storage used by the compact representation.
    [[nodiscard]] std::size_t memory_bytes() const noexcept;
};

/**
 * @brief Canonical immutable execution plan for normalised static M2L.
 *
 * `matrices` contains column-major, level-independent transfer-class matrices.
 * Multipole and local scaling are laid out `[level][coefficient]` and restore
 * physical box width. For target node `t`, row offsets select parallel entries
 * in `source_nodes`, `matrix_ids`, and `interaction_levels`. CPU, oneMKL, and
 * CUDA execution derive from this single mathematical representation; the
 * vectors contain no mutable evaluation coefficients or device ownership.
 */
struct StaticM2LPlan {
    int coefficient_count{0};
    int matrix_count{0};
    int level_count{0};
    std::vector<double> matrices{};
    std::vector<double> multipole_scaling{};
    std::vector<double> local_scaling{};
    std::vector<int> target_row_offsets{};
    std::vector<int> source_nodes{};
    std::vector<int> matrix_ids{};
    std::vector<int> interaction_levels{};
};

/** @brief Applies one level of the canonical target-row M2L plan. */
void apply_static_m2l_plan(
    const StaticM2LPlan& plan,
    int level,
    std::span<const std::vector<double>> multipoles,
    std::span<std::vector<double>> locals
);

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
