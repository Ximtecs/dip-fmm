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
#include "cdfmm/static_operators.hpp"
#include "cdfmm/uniform_tree.hpp"

namespace cdfmm {

/** @brief Location and implementation used for a complete FMM evaluation. */
enum class ExecutionBackend {
    Auto,
    CpuReference,
    CpuStatic,
    CudaM2LP2P,
    CudaPartial = CudaM2LP2P,
    CudaFull,
    CudaM2L = CudaM2LP2P,
  CudaM2LStaticP2P = CudaM2LP2P
};

/** @brief Dense multiplication implementation for cached static M2L matrices.
 */
enum class StaticMatrixBackend { Portable, OneMkl };

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
  /// @brief Maximum total degree of the Cartesian multipole expansion.
  int expansion_order{4};
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
 * @brief Complete uniform FMM evaluator with reusable static M2L by default.
 *
 * Construction fixes and Morton-sorts the source geometry. Calling
 * `upward_pass` accepts moments in the original user source order and replaces
 * all stored node multipoles. For every node, the resulting expansion
 * represents all source dipoles in that node's subtree about that node's
 * centre.
 *
 * When target geometry is supplied, `evaluate` completes P2M, M2M, M2L, L2L,
 * L2P, and direct list1 P2P. Results are returned in original user target
 * order. The independently callable passes and node coefficients are exposed
 * for validation and education rather than as an optimised execution plan.
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
  /// @brief Returns the Cartesian coefficient basis used by every node.
  [[nodiscard]] const MultiIndexSet &basis() const;
  /// @brief Returns the selected M2L execution backend.
  [[nodiscard]] M2LBackend m2l_backend() const;
  /// @brief Returns the selected cached static-matrix multiplication backend.
  [[nodiscard]] StaticMatrixBackend static_matrix_backend() const;
  /// @brief Returns the resolved backend used by this evaluator.
  [[nodiscard]] ExecutionBackend backend() const;
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
   * Coefficients follow `basis()` ordering. Before the first upward pass and
   * for empty subtrees, every coefficient is zero.
   */
  [[nodiscard]] std::span<const double> multipole(int node_index) const;

  /** @brief Returns a node local expansion in `basis()` ordering. */
  [[nodiscard]] std::span<const double> local(int node_index) const;

  /// @brief Returns the root multipole, whose flat node index is zero.
  [[nodiscard]] std::span<const double> root_multipole() const;

private:
  class CudaM2LPlanOwner;
  class CudaP2PPlanOwner;
  class CudaFullPlanOwner;
  /** @brief Immutable leaf index and geometry-specific P2M coefficient map. */
  struct P2MPlan {
      int leaf{0};
    StaticCoefficientOperator operator_map{};
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

  void build_static_plan();
  void initialise_source_geometry(const UniformFmmOptions &options);
  void initialise_p2p_policy(const UniformFmmOptions &options);
  void build_cuda_p2p_plan();
  void build_cuda_full_plan();
  void prepare_moments(std::span<const Vec3> dipole_moments);
  void upward_pass_prepared();
  void prepare_self_indices(std::span<const int> target_source_indices);
  [[nodiscard]] std::span<const int>
  resolve_self_indices(std::span<const int> target_source_indices) const;
  void static_m2l(int level);
  void cuda_m2l();
  void l2l_downward();
  [[nodiscard]] std::span<double> multipole_for_node(int node_index) noexcept;
  [[nodiscard]] std::span<const double> multipole_for_node(
      int node_index) const noexcept;
  [[nodiscard]] std::span<double> local_for_node(int node_index) noexcept;
  [[nodiscard]] std::span<const double> local_for_node(
      int node_index) const noexcept;

  // Fixed geometry and immutable operator descriptions outlive every call.
  UniformTree tree_;
  MultiIndexSet basis_;
  M2LBackend m2l_backend_{M2LBackend::Static};
  StaticMatrixBackend static_matrix_backend_{StaticMatrixBackend::Portable};
  ExecutionBackend backend_{ExecutionBackend::CpuStatic};
  SourceGeometry source_geometry_{SourceGeometry::PointDipole};
  std::vector<CuboidSize> sorted_source_sizes_{};
  P2PExecutionPacking p2p_execution_packing_{P2PExecutionPacking::Reference};
  std::size_t cuda_p2p_bsr_max_bytes_{20ULL * 1024ULL * 1024ULL * 1024ULL};
  mutable CudaPlanStatistics empty_cuda_statistics_{};
  std::unique_ptr<CudaM2LPlanOwner> cuda_m2l_plan_{};
  std::unique_ptr<CudaP2PPlanOwner> cuda_p2p_plan_{};
  std::unique_ptr<CudaFullPlanOwner> cuda_full_plan_{};
  std::vector<M2LGroup> m2l_groups_{};
  std::vector<P2MPlan> p2m_plans_{};
  std::vector<std::array<StaticCoefficientOperator, 8>> m2m_operators_{};
  std::vector<std::array<StaticCoefficientOperator, 8>> l2l_operators_{};
  std::vector<StaticL2PEvaluator> l2p_evaluators_{};
  StaticP2POperator p2p_operator_{};
  StaticP2PCompactPlan p2p_compact_plan_{};
  StaticM2LPlan m2l_plan_{};
  StaticPlanStatistics static_plan_statistics_{};
  // Mutable coefficient and result storage makes one evaluator non-reentrant.
  // All fixed-width node expansions share one node-major allocation. This
  // removes two heap allocations per node and lets resets stream linearly.
  std::vector<double> multipoles_{};
  std::vector<double> locals_{};
  std::vector<Vec3> sorted_dipole_moments_{};
  std::vector<PotentialField> sorted_results_{};
  std::vector<Vec3> near_fields_{};
  std::vector<int> sorted_self_indices_{};
  std::optional<std::vector<int>> fixed_target_source_indices_{};
  std::vector<int> fixed_sorted_self_indices_{};
  EvaluationTimings last_timings_{};
  EvaluationTimings aggregate_timings_{};
};

} // namespace cdfmm
