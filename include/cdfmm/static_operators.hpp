// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <array>
#include <vector>

#include "cdfmm/operators.hpp"
#include "cdfmm/cuboid.hpp"

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

/** @brief Persistent-memory breakdown for a static P2P execution packing. */
struct StaticP2PMemory {
  std::size_t tensor_bytes{0};
  std::size_t index_bytes{0};
  std::size_t row_metadata_bytes{0};
  std::size_t leaf_metadata_bytes{0};
  std::size_t scratch_bytes{0};

  /// @brief Returns all persistent and scratch bytes represented above.
  [[nodiscard]] std::size_t total_bytes() const noexcept;
};

/**
 * @brief Source-index and structure-of-arrays packing of canonical P2P rows.
 *
 * Target identity is implied by `row_offsets`. The six tensor streams retain
 * the canonical interaction order, so this packing removes redundant target
 * indices without changing the accumulation order.
 */
struct StaticP2PCompactPlan {
  int source_count{0};
  int target_count{0};
  std::vector<int> row_offsets{};
  std::vector<int> source_indices{};
  std::array<std::vector<double>, 6> tensors{};

  /// @brief Returns the storage breakdown for this execution packing.
  [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/** @brief Particle ranges defining one dense target/source leaf pair. */
struct StaticP2PLeafPair {
  int target_begin{0};
  int target_count{0};
  int source_begin{0};
  int source_count{0};
};

/** @brief Metadata for one compact dense target/source leaf tensor block. */
struct StaticP2PLeafBlock {
  int source_begin{0};
  int source_count{0};
  std::size_t tensor_offset{0};
};

/**
 * @brief Compact leaf-grouped execution packing of the canonical P2P tensor.
 *
 * One row describes each occupied target leaf. Leaf blocks are dense in
 * target-major, source-minor order and therefore need no particle indices.
 * Arbitrary, unequal, and partially occupied leaf sizes are supported.
 */
struct StaticP2PLeafPlan {
  int source_count{0};
  int target_count{0};
  std::vector<int> target_begins{};
  std::vector<int> target_counts{};
  std::vector<int> leaf_row_offsets{};
  std::vector<StaticP2PLeafBlock> blocks{};
  std::array<std::vector<double>, 6> tensors{};
  int minimum_occupancy{0};
  int maximum_occupancy{0};
  double mean_occupancy{0.0};
  int unique_occupancies{0};
  bool uniform_occupancy{false};

  /// @brief Returns the storage breakdown for this execution packing.
  [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/**
 * @brief Full 3 x 3 BSR packing with an immutable self-identity policy.
 *
 * Symmetric canonical tensors are expanded from six to nine coefficients.
 * Entries selected by `target_source_indices` are stored as exact zero blocks
 * because generic sparse libraries cannot skip a changing identity per row.
 */
struct StaticP2PBsrPlan {
  int source_count{0};
  int target_count{0};
  std::vector<int> row_offsets{};
  std::vector<int> source_indices{};
  std::vector<double> values{};
  std::vector<int> target_source_indices{};

  /// @brief Returns known BSR value and index storage.
  [[nodiscard]] StaticP2PMemory memory() const noexcept;
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
    /// Node-index bounds for each level in the level-ordered uniform tree.
    std::vector<int> level_target_begin{};
    std::vector<int> level_target_end{};
};

/** @brief Applies one level of the canonical target-row M2L plan. */
void apply_static_m2l_plan(
    const StaticM2LPlan& plan,
    int level,
    std::span<const double> multipoles,
    std::span<double> locals
);

/** @brief Compatibility overload for independently allocated node vectors. */
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

/** @brief Builds cuboid-averaged P2M for fixed source centres and dimensions. */
[[nodiscard]] StaticCoefficientOperator build_static_cuboid_p2m_operator(
    const MultiIndexSet& basis,
    const Vec3& centre,
    std::span<const Vec3> source_positions,
    std::span<const CuboidSize> source_sizes
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

/** @brief Builds rows evaluating the volume average over a fixed cuboid target. */
[[nodiscard]] StaticL2PEvaluator build_static_cuboid_l2p_evaluator(
    const MultiIndexSet& basis,
    const Vec3& centre,
    const Vec3& target,
    const CuboidSize& target_size
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

/** @brief Packs canonical particle rows into six contiguous tensor streams. */
[[nodiscard]] StaticP2PCompactPlan
build_static_p2p_compact_plan(const StaticP2POperator &operator_map);

/**
 * @brief Packs dense target/source leaf pairs from a canonical P2P operator.
 *
 * The supplied pairs must partition complete canonical rows into dense leaf
 * rectangles. This explicit contract prevents an execution packing from
 * silently changing the mathematical interaction set.
 */
[[nodiscard]] StaticP2PLeafPlan
build_static_p2p_leaf_plan(const StaticP2POperator &operator_map,
                           std::span<const StaticP2PLeafPair> leaf_pairs);

/** @brief Expands canonical tensors into full BSR(3) blocks. */
[[nodiscard]] StaticP2PBsrPlan
build_static_p2p_bsr_plan(const StaticP2POperator &operator_map,
                          std::span<const int> target_source_indices = {});

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

/** @brief Applies the source-only SoA particle-row packing additively. */
void apply_static_p2p_compact_plan(
    const StaticP2PCompactPlan &plan, std::span<const Vec3> dipole_moments,
    std::span<Vec3> H, std::span<const int> target_source_indices = {});

/** @brief Applies the compact leaf-grouped tensor packing additively. */
void apply_static_p2p_leaf_plan(
    const StaticP2PLeafPlan &plan, std::span<const Vec3> dipole_moments,
    std::span<Vec3> H, std::span<const int> target_source_indices = {});

/** @brief Applies the portable full BSR(3) reference packing additively. */
void apply_static_p2p_bsr_plan(const StaticP2PBsrPlan &plan,
                               std::span<const Vec3> dipole_moments,
                               std::span<Vec3> H,
                               std::span<const int> target_source_indices = {});

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
