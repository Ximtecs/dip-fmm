// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "cdfmm/cuboid.hpp"
#include "cdfmm/operators.hpp"
#include "cdfmm/periodic.hpp"
#include "cdfmm/precision.hpp"
#include "cdfmm/spherical_harmonics.hpp"
#include "cdfmm/tensor_dictionary.hpp"

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
 * @brief Fixed potential row and six independent dipole-field tensor entries.
 *
 * The target/source indices use Morton-sorted particle order. Symmetry stores
 * only xx, xy, xz, yy, yz and zz. The px, py, and pz row maps a moment to
 * scalar potential. The block contains no per-evaluation state.
 */
struct StaticDipoleBlock {
    int target{0};
    int source{0};
    double px{0.0};
    double py{0.0};
    double pz{0.0};
    double xx{0.0};
    double xy{0.0};
    double xz{0.0};
    double yy{0.0};
    double yz{0.0};
    double zz{0.0};
    /// Non-zero only for the central image eligible for identity exclusion.
    int skip_for_identity{1};
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
  /// Central-image marker controlling target/source identity exclusion.
  std::vector<unsigned char> skip_for_identity{};
  /// Coefficients mapping a dipole moment to scalar potential.
  std::array<std::vector<double>, 3> potential{};
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
 * @brief Leaf-topology P2P packing with one packed Tensor6 token per pair.
 *
 * A token stores a dictionary ID and six independent component sign bits.
 * Source and target particle indices remain implied by dense leaf blocks.
 */
struct StaticP2PTensorDictionaryPlan {
  int source_count{0};
  int target_count{0};
  std::vector<int> target_begins{};
  std::vector<int> target_counts{};
  std::vector<int> leaf_row_offsets{};
  std::vector<StaticP2PLeafBlock> blocks{};
  std::array<std::vector<double>, 6> tensors{};
  std::vector<std::uint32_t> tokens{};
  bool skip_for_identity{false};

  [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/**
 * @brief CPU-oriented signed Tensor6 dictionary execution packing.
 *
 * Tensor entries are already signed and token streams are source-major within
 * a dense leaf pair.  The execution loop therefore performs only a token
 * lookup followed by six SoA dictionary gathers.  `zero_variant` represents
 * point-dipole self interactions when fixed identities were supplied while
 * building the plan.
 */
struct StaticP2PSignedTensorDictionaryPlan {
  int source_count{0};
  int target_count{0};
  std::vector<int> target_begins{};
  std::vector<int> target_counts{};
  std::vector<int> leaf_row_offsets{};
  std::vector<StaticP2PLeafBlock> blocks{};
  std::vector<int> tile_leaf_indices{};
  std::vector<int> tile_target_offsets{};
  std::array<std::vector<double>, 6> tensors{};
  std::vector<std::uint8_t> tokens8{};
  std::vector<std::uint16_t> tokens16{};
  std::vector<std::uint32_t> tokens32{};
  std::uint8_t token_width_bytes{1};
  int target_tile_size{64};

  [[nodiscard]] std::size_t variant_count() const noexcept {
    return tensors[0].size();
  }
  [[nodiscard]] std::size_t token_count() const noexcept {
    return tokens8.size() + tokens16.size() + tokens32.size();
  }
  [[nodiscard]] StaticP2PMemory memory() const noexcept;
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

//------------------------------------------------------------------------------
// FP32 execution representations
//------------------------------------------------------------------------------

/** @brief FP32 counterpart of one sparse static coefficient entry. */
struct FloatStaticOperatorEntry {
    int output{0};
    int input{0};
    float value{0.0F};
};

/** @brief FP32 sparse coefficient operator with no retained FP64 values. */
struct FloatStaticCoefficientOperator {
    int input_size{0};
    int output_size{0};
    std::vector<FloatStaticOperatorEntry> entries{};
};

/** @brief FP32 local-evaluation rows. */
struct FloatStaticL2PEvaluator {
    std::vector<float> potential{};
    std::array<std::vector<float>, 3> field{};
};

/** @brief FP32 symmetric near-field tensor and particle indices. */
struct FloatStaticDipoleBlock {
    int target{0};
    int source{0};
    float px{0.0F};
    float py{0.0F};
    float pz{0.0F};
    float xx{0.0F};
    float xy{0.0F};
    float xz{0.0F};
    float yy{0.0F};
    float yz{0.0F};
    float zz{0.0F};
    int skip_for_identity{1};
};

/** @brief Accumulates one FP32 symmetric dipole tensor product. */
CDFMM_HOST_DEVICE inline void accumulate_static_dipole_block(
    const FloatStaticDipoleBlock& block,
    const FloatVec3& moment,
    FloatVec3& H
) noexcept {
    H.x += block.xx * moment.x + block.xy * moment.y + block.xz * moment.z;
    H.y += block.xy * moment.x + block.yy * moment.y + block.yz * moment.z;
    H.z += block.xz * moment.x + block.yz * moment.y + block.zz * moment.z;
}

#undef CDFMM_HOST_DEVICE

/** @brief Canonical FP32 target-row near-field operator. */
struct FloatStaticP2POperator {
    int source_count{0};
    int target_count{0};
    std::vector<int> row_offsets{};
    std::vector<FloatStaticDipoleBlock> blocks{};

    [[nodiscard]] std::size_t memory_bytes() const noexcept;
};

/** @brief FP32 source-only SoA near-field packing. */
struct FloatStaticP2PCompactPlan {
    int source_count{0};
    int target_count{0};
    std::vector<int> row_offsets{};
    std::vector<int> source_indices{};
    std::vector<unsigned char> skip_for_identity{};
    std::array<std::vector<float>, 3> potential{};
    std::array<std::vector<float>, 6> tensors{};

    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/** @brief One exact near-field pair with a periodically shifted source. */
struct StaticP2PInteraction {
    int target{0};
    int source{0};
    Vec3 source_shift{};
    /// True only for the zero-shift image eligible for identity exclusion.
    bool skip_for_identity{true};
};

/** @brief FP32 compact leaf-grouped near-field packing. */
struct FloatStaticP2PLeafPlan {
    int source_count{0};
    int target_count{0};
    std::vector<int> target_begins{};
    std::vector<int> target_counts{};
    std::vector<int> leaf_row_offsets{};
    std::vector<StaticP2PLeafBlock> blocks{};
    std::array<std::vector<float>, 6> tensors{};
    int minimum_occupancy{0};
    int maximum_occupancy{0};
    double mean_occupancy{0.0};
    int unique_occupancies{0};
    bool uniform_occupancy{false};

    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/** @brief FP32 counterpart of the leaf-topology Tensor6 dictionary packing. */
struct FloatStaticP2PTensorDictionaryPlan {
  int source_count{0};
  int target_count{0};
  std::vector<int> target_begins{};
  std::vector<int> target_counts{};
  std::vector<int> leaf_row_offsets{};
  std::vector<StaticP2PLeafBlock> blocks{};
  std::array<std::vector<float>, 6> tensors{};
  std::vector<std::uint32_t> tokens{};
  bool skip_for_identity{false};

  [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/** @brief FP32 counterpart of the CPU signed Tensor6 dictionary packing. */
struct FloatStaticP2PSignedTensorDictionaryPlan {
  int source_count{0};
  int target_count{0};
  std::vector<int> target_begins{};
  std::vector<int> target_counts{};
  std::vector<int> leaf_row_offsets{};
  std::vector<StaticP2PLeafBlock> blocks{};
  std::vector<int> tile_leaf_indices{};
  std::vector<int> tile_target_offsets{};
  std::array<std::vector<float>, 6> tensors{};
  std::vector<std::uint8_t> tokens8{};
  std::vector<std::uint16_t> tokens16{};
  std::vector<std::uint32_t> tokens32{};
  std::uint8_t token_width_bytes{1};
  int target_tile_size{64};

  [[nodiscard]] std::size_t variant_count() const noexcept {
    return tensors[0].size();
  }
  [[nodiscard]] std::size_t token_count() const noexcept {
    return tokens8.size() + tokens16.size() + tokens32.size();
  }
  [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/** @brief Full FP32 BSR(3) near-field packing. */
struct FloatStaticP2PBsrPlan {
    int source_count{0};
    int target_count{0};
    std::vector<int> row_offsets{};
    std::vector<int> source_indices{};
    std::vector<float> values{};
    std::vector<int> target_source_indices{};

    [[nodiscard]] StaticP2PMemory memory() const noexcept;
};

/** @brief Canonical FP32 static M2L plan. */
struct FloatStaticM2LPlan {
    int coefficient_count{0};
    int matrix_count{0};
    int level_count{0};
    std::vector<float> matrices{};
    std::vector<float> multipole_scaling{};
    std::vector<float> local_scaling{};
    std::vector<int> target_row_offsets{};
    std::vector<int> source_nodes{};
    std::vector<int> matrix_ids{};
    std::vector<int> interaction_levels{};
    std::vector<int> level_target_begin{};
    std::vector<int> level_target_end{};
};

[[nodiscard]] FloatStaticCoefficientOperator quantise_static_operator(
    const StaticCoefficientOperator& source);
[[nodiscard]] FloatStaticL2PEvaluator quantise_static_l2p_evaluator(
    const StaticL2PEvaluator& source);
[[nodiscard]] FloatStaticP2POperator quantise_static_p2p_operator(
    const StaticP2POperator& source);
[[nodiscard]] FloatStaticP2PCompactPlan quantise_static_p2p_compact_plan(
    const StaticP2PCompactPlan& source);
[[nodiscard]] FloatStaticP2PLeafPlan quantise_static_p2p_leaf_plan(
    const StaticP2PLeafPlan& source);
[[nodiscard]] FloatStaticP2PTensorDictionaryPlan
quantise_static_p2p_tensor_dictionary_plan(
    const StaticP2PTensorDictionaryPlan &source);
[[nodiscard]] FloatStaticP2PSignedTensorDictionaryPlan
quantise_static_p2p_signed_tensor_dictionary_plan(
    const StaticP2PSignedTensorDictionaryPlan &source);
[[nodiscard]] FloatStaticP2PBsrPlan quantise_static_p2p_bsr_plan(
    const StaticP2PBsrPlan& source);
[[nodiscard]] FloatStaticM2LPlan quantise_static_m2l_plan(
    const StaticM2LPlan& source);

/** @brief Applies one level of the canonical target-row M2L plan. */
void apply_static_m2l_plan(
    const StaticM2LPlan& plan,
    int level,
    std::span<const double> multipoles,
    std::span<double> locals
);

/** @brief Applies one level of an FP32 canonical target-row M2L plan. */
void apply_static_m2l_plan(
    const FloatStaticM2LPlan& plan,
    int level,
    std::span<const float> multipoles,
    std::span<float> locals
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

/** @brief Builds a dense real spherical M2L matrix for one displacement. */
[[nodiscard]] std::vector<double> build_static_m2l_matrix(
    const SphericalHarmonicBasis& basis,
    const Vec3& R
);

/** @brief Builds M = P m for fixed source positions and expansion centre. */
[[nodiscard]] StaticCoefficientOperator build_static_p2m_operator(
    const MultiIndexSet& basis,
    const Vec3& centre,
    std::span<const Vec3> source_positions
);

/** @brief Builds point-dipole P2M directly in the real spherical basis. */
[[nodiscard]] StaticCoefficientOperator build_static_p2m_operator(
    const SphericalHarmonicBasis& basis,
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

/**
 * @brief Builds analytical cuboid-averaged P2M in the real spherical basis.
 *
 * The returned operator maps dipole moments directly to `(p+1)^2` spherical
 * coefficients. Cuboid integration is completed during plan construction.
 */
[[nodiscard]] StaticCoefficientOperator build_static_cuboid_p2m_operator(
    const SphericalHarmonicBasis& basis,
    const Vec3& centre,
    std::span<const Vec3> source_positions,
    std::span<const CuboidSize> source_sizes
);

/** @brief Builds the triangular map M_parent += A(d) M_child. */
[[nodiscard]] StaticCoefficientOperator build_static_m2m_operator(
    const MultiIndexSet& basis,
    const Vec3& d
);

/** @brief Builds a real spherical M2M translation operator. */
[[nodiscard]] StaticCoefficientOperator build_static_m2m_operator(
    const SphericalHarmonicBasis& basis,
    const Vec3& d
);

/** @brief Builds the triangular map L_child += B(d) L_parent. */
[[nodiscard]] StaticCoefficientOperator build_static_l2l_operator(
    const MultiIndexSet& basis,
    const Vec3& d
);

/** @brief Builds a real spherical L2L translation operator. */
[[nodiscard]] StaticCoefficientOperator build_static_l2l_operator(
    const SphericalHarmonicBasis& basis,
    const Vec3& d
);

/** @brief Builds fixed rows mapping a local expansion to phi and H. */
[[nodiscard]] StaticL2PEvaluator build_static_l2p_evaluator(
    const MultiIndexSet& basis,
    const Vec3& centre,
    const Vec3& target
);

/** @brief Builds point-target rows for a real spherical local expansion. */
[[nodiscard]] StaticL2PEvaluator build_static_l2p_evaluator(
    const SphericalHarmonicBasis& basis,
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
 * @brief Builds analytical cuboid-volume rows for spherical local coefficients.
 *
 * The evaluator stores only potential and field rows of spherical width; no
 * Cartesian coefficients or conversion maps are retained at evaluation time.
 */
[[nodiscard]] StaticL2PEvaluator
build_static_cuboid_l2p_evaluator(const SphericalHarmonicBasis &basis,
                                  const Vec3 &centre, const Vec3 &target,
                                  const CuboidSize &target_size);

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
    std::span<const std::array<int, 2>> interactions,
    SourceGeometry source_geometry = SourceGeometry::PointDipole,
    std::span<const CuboidSize> source_sizes = {},
    TargetGeometry target_geometry = TargetGeometry::Point,
    std::span<const CuboidSize> target_sizes = {});

/** @brief Builds exact P2P rows from image-aware periodic interactions. */
[[nodiscard]] StaticP2POperator build_static_p2p_operator(
    std::span<const Vec3> target_positions,
    std::span<const Vec3> source_positions,
    std::span<const StaticP2PInteraction> interactions,
    SourceGeometry source_geometry = SourceGeometry::PointDipole,
    std::span<const CuboidSize> source_sizes = {},
    TargetGeometry target_geometry = TargetGeometry::Point,
    std::span<const CuboidSize> target_sizes = {});

/** @brief Builds the normalised Cartesian zero-k0 root periodiser matrix. */
[[nodiscard]] std::vector<double>
build_static_periodic_m2l_matrix(const MultiIndexSet &basis,
                                 const PeriodicCellOptions &options);

/** @brief Builds the normalised spherical zero-k0 root periodiser matrix. */
[[nodiscard]] std::vector<double>
build_static_periodic_m2l_matrix(const SphericalHarmonicBasis &basis,
                                 const PeriodicCellOptions &options);

/** @brief Packs canonical particle rows into six contiguous tensor streams. */
[[nodiscard]] StaticP2PCompactPlan
build_static_p2p_compact_plan(const StaticP2POperator &operator_map);

/** @brief Packs canonical FP32 particle rows into contiguous tensor streams. */
[[nodiscard]] FloatStaticP2PCompactPlan
build_static_p2p_compact_plan(const FloatStaticP2POperator &operator_map);

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

/** @brief Packs dense leaf blocks into an exact-bit Tensor6 dictionary. */
[[nodiscard]] StaticP2PTensorDictionaryPlan
build_static_p2p_tensor_dictionary_plan(
    const StaticP2POperator &operator_map,
    std::span<const StaticP2PLeafPair> leaf_pairs);

/**
 * @brief Builds source-major, signed Tensor6 CPU execution packing.
 *
 * Point-dipole fixed identities are encoded as the zero variant.  Leave
 * `target_source_indices` empty for cuboid interactions, whose self tensors
 * are physical and must remain in the dictionary.
 */
[[nodiscard]] StaticP2PSignedTensorDictionaryPlan
build_static_p2p_signed_tensor_dictionary_plan(
    const StaticP2POperator &operator_map,
    std::span<const StaticP2PLeafPair> leaf_pairs,
    std::span<const int> target_source_indices = {}, int target_tile_size = 64);

/** @brief Expands canonical tensors into full BSR(3) blocks. */
[[nodiscard]] StaticP2PBsrPlan
build_static_p2p_bsr_plan(const StaticP2POperator &operator_map,
                          std::span<const int> target_source_indices = {});

/** @brief Expands canonical FP32 tensors into full BSR(3) blocks. */
[[nodiscard]] FloatStaticP2PBsrPlan
build_static_p2p_bsr_plan(const FloatStaticP2POperator &operator_map,
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

/** @brief Applies the leaf-topology Tensor6 dictionary packing additively. */
void apply_static_p2p_tensor_dictionary_plan(
    const StaticP2PTensorDictionaryPlan &plan,
    std::span<const Vec3> dipole_moments, std::span<Vec3> H,
    std::span<const int> target_source_indices = {});

/** @brief Applies the CPU source-major signed Tensor6 dictionary packing. */
void apply_static_p2p_signed_tensor_dictionary_plan(
    const StaticP2PSignedTensorDictionaryPlan &plan,
    std::span<const Vec3> dipole_moments, std::span<Vec3> H);

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

void apply_static_operator(
    const FloatStaticCoefficientOperator& operator_map,
    std::span<const float> input,
    std::span<float> output
);

[[nodiscard]] FloatPotentialField apply_static_l2p_evaluator(
    const FloatStaticL2PEvaluator& evaluator,
    std::span<const float> L,
    OutputFlags output = OutputFlags::Field
);

void apply_static_p2p_compact_plan(
    const FloatStaticP2PCompactPlan& plan,
    std::span<const FloatVec3> dipole_moments,
    std::span<FloatVec3> H,
    std::span<const int> target_source_indices = {}
);

void apply_static_p2p_operator(
    const FloatStaticP2POperator& operator_map,
    std::span<const FloatVec3> dipole_moments,
    std::span<FloatVec3> H,
    std::span<const int> target_source_indices = {}
);

void apply_static_p2p_leaf_plan(
    const FloatStaticP2PLeafPlan& plan,
    std::span<const FloatVec3> dipole_moments,
    std::span<FloatVec3> H,
    std::span<const int> target_source_indices = {}
);

/** @brief Applies the FP32 leaf-topology Tensor6 dictionary packing. */
void apply_static_p2p_tensor_dictionary_plan(
    const FloatStaticP2PTensorDictionaryPlan &plan,
    std::span<const FloatVec3> dipole_moments, std::span<FloatVec3> H,
    std::span<const int> target_source_indices = {});

/** @brief Applies the FP32 CPU source-major signed Tensor6 dictionary. */
void apply_static_p2p_signed_tensor_dictionary_plan(
    const FloatStaticP2PSignedTensorDictionaryPlan &plan,
    std::span<const FloatVec3> dipole_moments, std::span<FloatVec3> H);

void apply_static_p2p_bsr_plan(
    const FloatStaticP2PBsrPlan& plan,
    std::span<const FloatVec3> dipole_moments,
    std::span<FloatVec3> H,
    std::span<const int> target_source_indices = {}
);

} // namespace cdfmm
