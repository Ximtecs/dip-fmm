// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "cdfmm/coefficients.hpp"
#include "cdfmm/cuboid.hpp"
#include "cdfmm/multi_index.hpp"
#include "cdfmm/output_flags.hpp"
#include "cdfmm/precision.hpp"
#include "cdfmm/static_operators.hpp"
#include "cdfmm/spherical_harmonics.hpp"
#include "cdfmm/uniform_tree.hpp"

namespace cdfmm {

/** @brief Expansion basis used by the reusable far-field hierarchy. */
enum class ExpansionBasis {
    /// Factorial-normalised total-degree Cartesian Taylor coefficients.
    Cartesian,
    /// Minimal real tesseral solid harmonics ordered by degree then m.
    Spherical
};

/** @brief Implemented M2L strategy for real spherical expansions. */
enum class SphericalM2LBackend {
    /// Reusable dense real matrix for each integer displacement class.
    StaticDense
};

/** @brief Location and implementation used for a complete FMM evaluation. */
enum class ExecutionBackend {
    /// Resolve conservatively to the portable CPU static backend.
    Auto,
    /// Independent Cartesian CPU traversal for validation.
    CpuReference,
    /// Canonical static plan executed on the CPU.
    CpuStatic,
    /// Hybrid CPU hierarchy with CUDA M2L and P2P.
    CudaM2LP2P,
    /// User-facing alias for the hybrid CUDA backend.
    CudaPartial = CudaM2LP2P,
    /// Complete field-only device-resident static FMM.
    CudaFull,
    /// Compatibility alias for the hybrid CUDA backend.
    CudaM2L = CudaM2LP2P,
    /// Compatibility alias for the hybrid CUDA backend.
    CudaM2LStaticP2P = CudaM2LP2P
};

/** @brief Dense multiplication implementation for cached static M2L matrices.
 */
enum class StaticMatrixBackend {
    /// Internal typed dense multiplication.
    Portable,
    /// Class-grouped oneMKL SGEMM or DGEMM.
    OneMkl
};

/** @brief Executor selected for one operator in the shared static traversal. */
enum class StaticOperatorExecutor { Reference, Portable, OneMkl, Cuda };

/** @brief Storage packing selected for repeated list1 P2P execution. */
enum class P2PExecutionPacking {
  Reference,
  CanonicalAos,
  ParticleRowSoa,
  CudaBsr3
};

/**
 * @brief Per-operator executor selection for the canonical static plan.
 *
 * This small value object reports resolved placement; it owns neither
 * operators nor scratch buffers. See `docs/backends.md` for transfer flow.
 */
struct StaticExecutionPlan {
  StaticOperatorExecutor p2m{StaticOperatorExecutor::Portable};
  StaticOperatorExecutor m2m{StaticOperatorExecutor::Portable};
  StaticOperatorExecutor m2l{StaticOperatorExecutor::Portable};
  StaticOperatorExecutor l2l{StaticOperatorExecutor::Portable};
  StaticOperatorExecutor l2p{StaticOperatorExecutor::Portable};
  StaticOperatorExecutor p2p{StaticOperatorExecutor::Portable};
};

/** @brief Reports whether this build includes the oneMKL matrix backend. */
[[nodiscard]] bool one_mkl_available() noexcept;

/** @brief Reports whether the library was compiled with CUDA support. */
[[nodiscard]] bool cuda_compiled() noexcept;

/** @brief Reports whether this build can access a CUDA device. */
[[nodiscard]] bool cuda_available() noexcept;

/** @brief Reports whether the O(N^2) CUDA direct reference is available. */
[[nodiscard]] bool cuda_direct_available() noexcept;

/** @brief Reports whether hybrid CUDA static M2L/P2P is available. */
[[nodiscard]] bool cuda_m2l_p2p_available() noexcept;

/** @brief Compatibility alias for `cuda_m2l_p2p_available()`. */
[[nodiscard]] bool cuda_m2l_available() noexcept;

/** @brief Reports whether a complete device-resident CUDA FMM is implemented.
 */
[[nodiscard]] bool cuda_full_available() noexcept;

/** @brief Returns a concise description of the selected CUDA device. */
[[nodiscard]] std::string cuda_device_description();

//------------------------------------------------------------------------------
// Public types
//------------------------------------------------------------------------------

/**
 * @brief Options for static or reference uniform FMM traversal.
 *
 * Geometry options are kept alongside the expansion order at setup, while
 * dipole moments are supplied separately for each upward evaluation.
 */
struct UniformFmmOptions {
  /// @brief Scalar precision used by operators, state, and execution.
  StaticPrecision precision{StaticPrecision::Float32};
  /// @brief Maximum degree of the selected multipole expansion.
  int expansion_order{4};
  /// @brief Expansion representation; real spherical harmonics are the default.
  ExpansionBasis expansion_basis{ExpansionBasis::Spherical};
  /// @brief Spherical M2L implementation selected at plan construction.
  SphericalM2LBackend spherical_m2l_backend{
      SphericalM2LBackend::StaticDense};
  /// @brief Complete uniform-tree geometry options.
  UniformTreeOptions tree{};
  /// @brief M2L execution backend; static grouped execution is the default.
  enum class M2LBackend { Static, Reference } m2l_backend{M2LBackend::Static};
  /// @brief Multiplication implementation used by cached static M2L matrices.
  StaticMatrixBackend static_matrix_backend{StaticMatrixBackend::Portable};
  /// @brief Complete evaluation backend. Auto safely selects CPU static.
  ExecutionBackend backend{ExecutionBackend::Auto};
  /// @brief Physical model represented by every source.
  SourceGeometry source_geometry{SourceGeometry::PointDipole};
  /// @brief Cuboid dimensions: one common size or one per source in user order.
  std::vector<CuboidSize> source_sizes{};
  /**
   * @brief Uses finite cuboid moments in P2M for uniform-cuboid sources.
   *
   * When false, P2M uses the ordinary point-dipole operator while P2P still
   * uses the exact cuboid-to-point tensor. This comparison mode isolates the
   * accuracy contribution from finite-source P2M moments. It has no effect for
   * point-dipole sources.
   */
  bool use_cuboid_p2m{true};
  /**
   * @brief Optional immutable target-to-source self-identity map.
   *
   * Entries use original user ordering. Supplying the map permits CUDA to
   * embed self exclusions in a cuSPARSE BSR(3) plan; omitting it retains the
   * dynamic-identity canonical CUDA path. The map must contain one entry per
   * target when present.
   */
  std::optional<std::vector<int>> fixed_target_source_indices{};
  /// @brief Maximum persistent bytes permitted for an automatic CUDA BSR plan.
  std::size_t cuda_p2p_bsr_max_bytes{20ULL * 1024ULL * 1024ULL * 1024ULL};
};

using M2LBackend = UniformFmmOptions::M2LBackend;

/**
 * @brief Complete fixed-geometry FMM with reusable static operators.
 *
 * Construction fixes and Morton-sorts the source geometry. Calling
 * `upward_pass` accepts moments in the original user source order and replaces
 * all stored node multipoles. For every node, the resulting expansion
 * represents all source dipoles in that node's subtree about that node's
 * centre.
 *
 * When target geometry is supplied, `evaluate` completes P2M, M2M, M2L, L2L,
 * L2P, and exact list1 P2P. Results are returned in original user target
 * order. Production backends consume one canonical static plan; independently
 * callable passes and node coefficients remain exposed for validation.
 */
class UniformFmm {
public:
  ~UniformFmm();
  UniformFmm(UniformFmm&&) noexcept;
  UniformFmm& operator=(UniformFmm&&) noexcept;
  UniformFmm(const UniformFmm&) = delete;
  UniformFmm& operator=(const UniformFmm&) = delete;
  /**
   * @brief Constructs fixed source geometry for repeated upward passes.
   *
   * @param source_positions Source positions in user order.
   * @param options Expansion and uniform-tree configuration.
   */
  UniformFmm(const std::vector<Vec3> &source_positions,
             const UniformFmmOptions &options = {});

  /**
   * @brief Constructs fixed source and target geometry for complete evaluation.
   *
   * Source and target populations are sorted independently by `UniformTree`.
   * Supplying equal coordinate arrays does not implicitly identify particles;
   * self identities are passed explicitly to `evaluate`.
   *
   * @param source_positions Source positions in user order.
   * @param target_positions Target positions in user order.
   * @param options Expansion and uniform-tree configuration.
   */
  UniformFmm(const std::vector<Vec3> &source_positions,
             const std::vector<Vec3> &target_positions,
             const UniformFmmOptions &options = {});

  /**
   * @brief Recomputes all node multipoles for a new dipole state.
   *
   * Input moments correspond one-to-one with the original source-position
   * ordering supplied to the constructor. They are permuted internally to
   * the tree's Morton order before leaf P2M.
   *
   * @param dipole_moments Dipole moments in original user source order.
   * @throws std::invalid_argument if the number of moments is incorrect.
   */
  void upward_pass(std::span<const Vec3> dipole_moments);

  /**
   * @brief Rebuilds all local expansions by M2L and downward L2L translation.
   *
   * Existing locals are cleared first. At each level, a target node inherits
   * its parent local before accumulating its own list2 interactions.
   */
  void downward_pass();

  /**
   * @brief Evaluates the complete far-field and near-field traversal.
   *
   * The optional identity map has one entry per target in original user order.
   * Entry `i` is the original source index representing the same particle, or
   * -1 for no self identity. Only that precise source is skipped during list1
   * P2P; coordinate equality is deliberately not used as an identity test.
   *
   * @param dipole_moments Moments in original user source order.
   * @param output Requested potential and/or magnetic field components.
   * @param target_source_indices Explicit target-to-source identity map.
   * @return Values in the original user target ordering.
   */
  [[nodiscard]] std::vector<PotentialField>
  evaluate(std::span<const Vec3> dipole_moments,
           OutputFlags output = OutputFlags::Field,
           std::span<const int> target_source_indices = {});

  /**
   * @brief Evaluates into caller-owned storage without allocating result
   * arrays.
   *
   * The object owns mutable expansions, scratch storage, and timing state.
   * Consequently calls on one object must not execute concurrently, although
   * each call may use OpenMP internally.
   */
  void evaluate_into(std::span<const Vec3> dipole_moments,
                     std::span<PotentialField> results,
                     OutputFlags output = OutputFlags::Field,
                     std::span<const int> target_source_indices = {});

  /// @brief Returns timings for the most recent complete evaluation.
  [[nodiscard]] const EvaluationTimings &last_timings() const;
  /// @brief Returns accumulated timings since construction or the last reset.
  [[nodiscard]] const EvaluationTimings &aggregate_timings() const;
  /// @brief Clears accumulated evaluation timings without changing geometry.
  void reset_timings();

  /// @brief Returns the fixed complete uniform-tree geometry.
  [[nodiscard]] const UniformTree &tree() const;
  /// @brief Returns Cartesian ordering; throws for a spherical plan.
  [[nodiscard]] const MultiIndexSet &basis() const;
  /// @brief Returns spherical mode ordering; throws for a Cartesian plan.
  [[nodiscard]] const SphericalHarmonicBasis& spherical_basis() const;
  /// @brief Returns the selected expansion representation.
  [[nodiscard]] ExpansionBasis expansion_basis() const noexcept;
  /// @brief Returns the maximum expansion degree.
  [[nodiscard]] int expansion_order() const noexcept;
  /// @brief Returns the number of stored coefficients per expansion.
  [[nodiscard]] int coefficient_count() const noexcept;
  /// @brief Returns the configured spherical M2L strategy.
  [[nodiscard]] SphericalM2LBackend spherical_m2l_backend() const noexcept;
  /// @brief Returns the selected M2L execution backend.
  [[nodiscard]] M2LBackend m2l_backend() const;
  /// @brief Returns the selected cached static-matrix multiplication backend.
  [[nodiscard]] StaticMatrixBackend static_matrix_backend() const;
  /// @brief Returns the resolved backend used by this evaluator.
  [[nodiscard]] ExecutionBackend backend() const;
  /// @brief Returns the scalar precision selected when constructing the plan.
  [[nodiscard]] StaticPrecision precision() const noexcept;
  /// @brief Returns the executor selected for every canonical static operator.
  [[nodiscard]] StaticExecutionPlan execution_plan() const noexcept;
  /// @brief Returns the resolved P2P storage packing used for evaluation.
  [[nodiscard]] P2PExecutionPacking p2p_execution_packing() const noexcept;
  /// @brief Returns CUDA traffic and persistent-allocation diagnostics.
  [[nodiscard]] const CudaPlanStatistics &cuda_plan_statistics() const;
  /// @brief Returns one-time static-plan timing and memory information.
  [[nodiscard]] const StaticPlanStatistics &static_plan_statistics() const;

  /**
   * @brief Returns a read-only multipole for a flat tree-node index.
   *
   * Coefficients follow the selected Cartesian or spherical basis ordering.
   * Before the first upward pass and for empty subtrees, every coefficient is
   * zero.
   */
  [[nodiscard]] std::span<const double> multipole(int node_index) const;

  /** @brief Returns a node local expansion in the selected basis ordering. */
  [[nodiscard]] std::span<const double> local(int node_index) const;

  /// @brief Returns the root multipole, whose flat node index is zero.
  [[nodiscard]] std::span<const double> root_multipole() const;

  /** @brief Evaluates an FP32 plan without widening its results. */
  [[nodiscard]] std::vector<FloatPotentialField>
  evaluate_float32(std::span<const Vec3> dipole_moments,
                   OutputFlags output = OutputFlags::Field,
                   std::span<const int> target_source_indices = {});

  /** @brief Evaluates an FP32 plan into caller-owned FP32 storage. */
  void evaluate_into_float32(
      std::span<const Vec3> dipole_moments,
      std::span<FloatPotentialField> results,
      OutputFlags output = OutputFlags::Field,
      std::span<const int> target_source_indices = {});

  /** @brief Evaluates an FP64 plan and rejects an FP32 plan. */
  [[nodiscard]] std::vector<PotentialField>
  evaluate_float64(std::span<const Vec3> dipole_moments,
                   OutputFlags output = OutputFlags::Field,
                   std::span<const int> target_source_indices = {});

  /** @brief Returns FP32 multipoles and rejects an FP64 plan. */
  [[nodiscard]] std::span<const float> multipole_float32(int node_index) const;

  /** @brief Returns FP32 locals and rejects an FP64 plan. */
  [[nodiscard]] std::span<const float> local_float32(int node_index) const;

  /** @brief Returns the FP32 root multipole and rejects an FP64 plan. */
  [[nodiscard]] std::span<const float> root_multipole_float32() const;

  /** @brief Returns FP64 multipoles and rejects an FP32 plan. */
  [[nodiscard]] std::span<const double> multipole_float64(int node_index) const;
  /** @brief Returns FP64 locals and rejects an FP32 plan. */
  [[nodiscard]] std::span<const double> local_float64(int node_index) const;
  /** @brief Returns the FP64 root multipole and rejects an FP32 plan. */
  [[nodiscard]] std::span<const double> root_multipole_float64() const;

private:
  class CudaM2LPlanOwner;
  class CudaP2PPlanOwner;
  class CudaFullPlanOwner;
  /** @brief Immutable leaf index and geometry-specific P2M coefficient map. */
  struct P2MPlan {
    int leaf{0};
    StaticCoefficientOperator operator_map{};
  };
  /** @brief FP32 leaf index and quantised P2M coefficient map. */
  struct FloatP2MPlan {
    int leaf{0};
    FloatStaticCoefficientOperator operator_map{};
  };
  /**
   * @brief oneMKL-only gather/GEMM/scatter packing for one transfer class.
   *
   * Source, target, and level metadata is fixed; gathered and translated are
   * persistent scratch reused by evaluations. Portable and CUDA paths consume
   * the canonical `StaticM2LPlan` directly and do not allocate these buffers.
   */
  struct M2LGroup {
    int matrix_id{0};
    std::vector<int> sources{};
    std::vector<int> targets{};
    std::vector<int> levels{};
    std::vector<double> gathered{};
    std::vector<double> translated{};
  };
  /** @brief FP32 oneMKL gather/GEMM/scatter packing. */
  struct FloatM2LGroup {
    int matrix_id{0};
    std::vector<int> sources{};
    std::vector<int> targets{};
    std::vector<int> levels{};
    std::vector<float> gathered{};
    std::vector<float> translated{};
  };

  void build_static_plan();
  void quantise_static_plan_to_float();
  void initialise_source_geometry(const UniformFmmOptions &options);
  void initialise_p2p_policy(const UniformFmmOptions &options);
  void build_cuda_p2p_plan();
  void build_cuda_full_plan();
  [[nodiscard]] int coefficient_degree(int coefficient) const;
  void prepare_moments(std::span<const Vec3> dipole_moments);
  void prepare_moments_float(std::span<const Vec3> dipole_moments);
  void upward_pass_prepared();
  void upward_pass_prepared_float();
  void prepare_self_indices(std::span<const int> target_source_indices);
  [[nodiscard]] std::span<const int>
  resolve_self_indices(std::span<const int> target_source_indices) const;
  void static_m2l(int level);
  void static_m2l_float(int level);
  void cuda_m2l();
  void l2l_downward();
  void downward_pass_float();
  [[nodiscard]] std::span<double> multipole_for_node(int node_index) noexcept;
  [[nodiscard]] std::span<const double> multipole_for_node(
      int node_index) const noexcept;
  [[nodiscard]] std::span<double> local_for_node(int node_index) noexcept;
  [[nodiscard]] std::span<const double> local_for_node(
      int node_index) const noexcept;
  [[nodiscard]] std::span<float> multipole_float_for_node(
      int node_index) noexcept;
  [[nodiscard]] std::span<const float> multipole_float_for_node(
      int node_index) const noexcept;
  [[nodiscard]] std::span<float> local_float_for_node(int node_index) noexcept;
  [[nodiscard]] std::span<const float> local_float_for_node(
      int node_index) const noexcept;

  // Fixed geometry and immutable operator descriptions outlive every call.
  UniformTree tree_;
  MultiIndexSet basis_;
  SphericalHarmonicBasis spherical_basis_;
  ExpansionBasis expansion_basis_{ExpansionBasis::Spherical};
  SphericalM2LBackend spherical_m2l_backend_{
      SphericalM2LBackend::StaticDense};
  M2LBackend m2l_backend_{M2LBackend::Static};
  StaticMatrixBackend static_matrix_backend_{StaticMatrixBackend::Portable};
  StaticPrecision precision_{StaticPrecision::Float32};
  // FP32 operators use root-width-normalised coordinates. This geometric
  // scale remains double precision, like the source and target positions.
  double float_coordinate_scale_{1.0};
  ExecutionBackend backend_{ExecutionBackend::CpuStatic};
  SourceGeometry source_geometry_{SourceGeometry::PointDipole};
  bool use_cuboid_p2m_{false};
  std::vector<CuboidSize> sorted_source_sizes_{};
  P2PExecutionPacking p2p_execution_packing_{P2PExecutionPacking::Reference};
  std::size_t cuda_p2p_bsr_max_bytes_{20ULL * 1024ULL * 1024ULL * 1024ULL};
  mutable CudaPlanStatistics empty_cuda_statistics_{};
  std::unique_ptr<CudaM2LPlanOwner> cuda_m2l_plan_{};
  std::unique_ptr<CudaP2PPlanOwner> cuda_p2p_plan_{};
  std::unique_ptr<CudaFullPlanOwner> cuda_full_plan_{};
  std::vector<M2LGroup> m2l_groups_{};
  std::vector<FloatM2LGroup> m2l_groups_float_{};
  std::vector<P2MPlan> p2m_plans_{};
  std::vector<FloatP2MPlan> p2m_plans_float_{};
  std::vector<std::array<StaticCoefficientOperator, 8>> m2m_operators_{};
  std::vector<std::array<StaticCoefficientOperator, 8>> l2l_operators_{};
  std::vector<StaticL2PEvaluator> l2p_evaluators_{};
  StaticP2POperator p2p_operator_{};
  StaticP2PCompactPlan p2p_compact_plan_{};
  StaticM2LPlan m2l_plan_{};
  std::vector<std::array<FloatStaticCoefficientOperator, 8>>
      m2m_operators_float_{};
  std::vector<std::array<FloatStaticCoefficientOperator, 8>>
      l2l_operators_float_{};
  std::vector<FloatStaticL2PEvaluator> l2p_evaluators_float_{};
  FloatStaticP2POperator p2p_operator_float_{};
  FloatStaticP2PCompactPlan p2p_compact_plan_float_{};
  FloatStaticP2PBsrPlan p2p_bsr_plan_float_{};
  FloatStaticM2LPlan m2l_plan_float_{};
  StaticPlanStatistics static_plan_statistics_{};
  // Mutable coefficient and result storage makes one evaluator non-reentrant.
  // All fixed-width node expansions share one node-major allocation. This
  // removes two heap allocations per node and lets resets stream linearly.
  std::vector<double> multipoles_{};
  std::vector<double> locals_{};
  std::vector<Vec3> sorted_dipole_moments_{};
  std::vector<PotentialField> sorted_results_{};
  std::vector<Vec3> near_fields_{};
  std::vector<float> multipoles_float_{};
  std::vector<float> locals_float_{};
  std::vector<FloatVec3> sorted_dipole_moments_float_{};
  std::vector<FloatPotentialField> sorted_results_float_{};
  std::vector<FloatVec3> near_fields_float_{};
  // Legacy coefficient inspection widens FP32 state only on demand.
  mutable std::vector<double> inspection_widening_buffer_{};
  std::vector<int> sorted_self_indices_{};
  std::optional<std::vector<int>> fixed_target_source_indices_{};
  std::vector<int> fixed_sorted_self_indices_{};
  EvaluationTimings last_timings_{};
  EvaluationTimings aggregate_timings_{};
};

} // namespace cdfmm
