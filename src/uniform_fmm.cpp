// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/uniform_fmm.hpp"

#include <algorithm>
#include <chrono>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>

#include "cdfmm/laplace_derivatives.hpp"

#ifdef CDFMM_USE_MKL
#include <mkl.h>
#endif

#ifdef CDFMM_USE_OPENMP
#include <omp.h>
#endif

#include "cdfmm/operators.hpp"
#include "cdfmm/static_operators.hpp"
#include "cuda_fmm_plan.hpp"
#include "cuda_m2l_plan.hpp"
#include "cuda_p2p_plan.hpp"
#include "near_field.hpp"
#include "profile.hpp"
#include "uniform_fmm_internal.hpp"

namespace cdfmm {

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(const Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void accumulate_phase(PhaseTiming &aggregate, const PhaseTiming &value) {
  aggregate.total_seconds += value.total_seconds;
  aggregate.calls += value.calls;
}

void accumulate_timings(EvaluationTimings &aggregate,
                        const EvaluationTimings &value) {
  accumulate_phase(aggregate.moment_permutation, value.moment_permutation);
  accumulate_phase(aggregate.multipole_reset, value.multipole_reset);
  accumulate_phase(aggregate.p2m, value.p2m);
  accumulate_phase(aggregate.m2m, value.m2m);
  accumulate_phase(aggregate.local_reset, value.local_reset);
  accumulate_phase(aggregate.l2l, value.l2l);
  accumulate_phase(aggregate.m2l, value.m2l);
  accumulate_phase(aggregate.m2l_scale, value.m2l_scale);
  accumulate_phase(aggregate.m2l_gather, value.m2l_gather);
  accumulate_phase(aggregate.m2l_multiply, value.m2l_multiply);
  accumulate_phase(aggregate.m2l_scatter, value.m2l_scatter);
  accumulate_phase(aggregate.l2p, value.l2p);
  accumulate_phase(aggregate.p2p, value.p2p);
  accumulate_phase(aggregate.result_unpermutation, value.result_unpermutation);
  accumulate_phase(aggregate.cuda_h2d, value.cuda_h2d);
  accumulate_phase(aggregate.cuda_kernel, value.cuda_kernel);
  accumulate_phase(aggregate.cuda_d2h, value.cuda_d2h);
  accumulate_phase(aggregate.cuda_m2l_h2d, value.cuda_m2l_h2d);
  accumulate_phase(aggregate.cuda_m2l_d2h, value.cuda_m2l_d2h);
  accumulate_phase(aggregate.cuda_p2p_h2d, value.cuda_p2p_h2d);
  accumulate_phase(aggregate.cuda_p2p_kernel, value.cuda_p2p_kernel);
  accumulate_phase(aggregate.cuda_p2p_d2h, value.cuda_p2p_d2h);
  accumulate_phase(aggregate.cuda_p2p_wait, value.cuda_p2p_wait);
  accumulate_phase(aggregate.total, value.total);
  aggregate.evaluations += value.evaluations;
}

class PendingCudaP2PGuard {
public:
  explicit PendingCudaP2PGuard(CudaP2PPlan *plan) noexcept : plan_(plan) {}

  ~PendingCudaP2PGuard() {
    if (active_) {
      plan_->cancel_evaluate();
    }
  }

  void arm() noexcept { active_ = true; }

  void release() noexcept { active_ = false; }

private:
  CudaP2PPlan *plan_{nullptr};
  bool active_{false};
};

} // namespace

//------------------------------------------------------------------------------
// Construction
//------------------------------------------------------------------------------

UniformFmm::UniformFmm(const std::vector<Vec3> &source_positions,
                       const UniformFmmOptions &options)
    : tree_(source_positions, options.tree), basis_(options.expansion_order),
      m2l_backend_(options.m2l_backend),
      static_matrix_backend_(options.static_matrix_backend),
      precision_(options.precision) {
  if (options.expansion_order < 0) {
    throw std::invalid_argument(
        "UniformFmmOptions.expansion_order must be >= 0");
  }
  initialise_source_geometry(options);

  backend_ = options.backend;
  if (backend_ == ExecutionBackend::Auto) {
    backend_ = options.m2l_backend == M2LBackend::Reference
                   ? ExecutionBackend::CpuReference
                   : ExecutionBackend::CpuStatic;
  }
  if (backend_ == ExecutionBackend::CudaM2LP2P && !cuda_m2l_p2p_available()) {
    throw std::runtime_error("CudaM2LP2P is unavailable in this build");
  }
  if (backend_ == ExecutionBackend::CudaFull && !cuda_full_available()) {
    throw std::runtime_error("CudaFull is unavailable in this build");
  }
  m2l_backend_ = backend_ == ExecutionBackend::CpuReference
                     ? M2LBackend::Reference
                     : M2LBackend::Static;
  if (m2l_backend_ == M2LBackend::Static &&
      static_matrix_backend_ == StaticMatrixBackend::OneMkl &&
      !one_mkl_available()) {
    throw std::runtime_error(
        "The oneMKL static-matrix backend is unavailable in this build");
  }

  const std::size_t coefficient_values =
      tree_.nodes().size() * static_cast<std::size_t>(basis_.size());
  if (precision_ == StaticPrecision::Float32) {
    multipoles_float_.assign(coefficient_values, 0.0F);
    locals_float_.assign(coefficient_values, 0.0F);
    sorted_dipole_moments_float_.resize(source_positions.size());
    sorted_results_float_.resize(tree_.sorted_target_positions().size());
    near_fields_float_.resize(tree_.sorted_target_positions().size());
  } else {
    multipoles_.assign(coefficient_values, 0.0);
    locals_.assign(coefficient_values, 0.0);
    sorted_dipole_moments_.resize(source_positions.size());
    sorted_results_.resize(tree_.sorted_target_positions().size());
    near_fields_.resize(tree_.sorted_target_positions().size());
  }
  sorted_self_indices_.resize(tree_.sorted_target_positions().size(), -1);
  initialise_p2p_policy(options);
  if (m2l_backend_ == M2LBackend::Static ||
      precision_ == StaticPrecision::Float32) {
    build_static_plan();
  }
  static_plan_statistics_.state_bytes = precision_ == StaticPrecision::Float32
      ? (multipoles_float_.size() + locals_float_.size()) * sizeof(float) +
            sorted_dipole_moments_float_.size() * sizeof(FloatVec3) +
            sorted_results_float_.size() * sizeof(FloatPotentialField) +
            near_fields_float_.size() * sizeof(FloatVec3) +
            sorted_self_indices_.size() * sizeof(int)
      : (multipoles_.size() + locals_.size()) * sizeof(double) +
            sorted_dipole_moments_.size() * sizeof(Vec3) +
            sorted_results_.size() * sizeof(PotentialField) +
            near_fields_.size() * sizeof(Vec3) +
            sorted_self_indices_.size() * sizeof(int);
  if (backend_ == ExecutionBackend::CudaM2LP2P) {
    if (precision_ == StaticPrecision::Float32) {
      cuda_m2l_plan_ = std::make_unique<CudaM2LPlanOwner>(
          std::make_unique<CudaM2LPlan>(m2l_plan_float_));
    } else {
      cuda_m2l_plan_ = std::make_unique<CudaM2LPlanOwner>(
          std::make_unique<CudaM2LPlan>(m2l_plan_));
    }
    if (!tree_.sorted_target_positions().empty()) {
      build_cuda_p2p_plan();
    }
  }
  if (backend_ == ExecutionBackend::CudaFull) {
    build_cuda_full_plan();
  }
}

UniformFmm::UniformFmm(const std::vector<Vec3> &source_positions,
                       const std::vector<Vec3> &target_positions,
                       const UniformFmmOptions &options)
    : tree_(source_positions, target_positions, options.tree),
      basis_(options.expansion_order),
      static_matrix_backend_(options.static_matrix_backend),
      precision_(options.precision) {
  if (options.expansion_order < 0) {
    throw std::invalid_argument(
        "UniformFmmOptions.expansion_order must be >= 0");
  }
  initialise_source_geometry(options);

  backend_ = options.backend;
  if (backend_ == ExecutionBackend::Auto) {
    backend_ = options.m2l_backend == M2LBackend::Reference
                   ? ExecutionBackend::CpuReference
                   : ExecutionBackend::CpuStatic;
  }
  if (backend_ == ExecutionBackend::CudaM2LP2P && !cuda_m2l_p2p_available()) {
    throw std::runtime_error("CudaM2LP2P is unavailable in this build");
  }
  if (backend_ == ExecutionBackend::CudaFull && !cuda_full_available()) {
    throw std::runtime_error("CudaFull is unavailable in this build");
  }
  m2l_backend_ = backend_ == ExecutionBackend::CpuReference
                     ? M2LBackend::Reference
                     : M2LBackend::Static;
  if (m2l_backend_ == M2LBackend::Static &&
      static_matrix_backend_ == StaticMatrixBackend::OneMkl &&
      !one_mkl_available()) {
    throw std::runtime_error(
        "The oneMKL static-matrix backend is unavailable in this build");
  }

  const std::size_t coefficient_values =
      tree_.nodes().size() * static_cast<std::size_t>(basis_.size());
  if (precision_ == StaticPrecision::Float32) {
    multipoles_float_.assign(coefficient_values, 0.0F);
    locals_float_.assign(coefficient_values, 0.0F);
    sorted_dipole_moments_float_.resize(source_positions.size());
    sorted_results_float_.resize(target_positions.size());
    near_fields_float_.resize(target_positions.size());
  } else {
    multipoles_.assign(coefficient_values, 0.0);
    locals_.assign(coefficient_values, 0.0);
    sorted_dipole_moments_.resize(source_positions.size());
    sorted_results_.resize(target_positions.size());
    near_fields_.resize(target_positions.size());
  }
  sorted_self_indices_.resize(target_positions.size(), -1);
  initialise_p2p_policy(options);
  if (m2l_backend_ == M2LBackend::Static ||
      precision_ == StaticPrecision::Float32) {
    build_static_plan();
  }
  static_plan_statistics_.state_bytes = precision_ == StaticPrecision::Float32
      ? (multipoles_float_.size() + locals_float_.size()) * sizeof(float) +
            sorted_dipole_moments_float_.size() * sizeof(FloatVec3) +
            sorted_results_float_.size() * sizeof(FloatPotentialField) +
            near_fields_float_.size() * sizeof(FloatVec3) +
            sorted_self_indices_.size() * sizeof(int)
      : (multipoles_.size() + locals_.size()) * sizeof(double) +
            sorted_dipole_moments_.size() * sizeof(Vec3) +
            sorted_results_.size() * sizeof(PotentialField) +
            near_fields_.size() * sizeof(Vec3) +
            sorted_self_indices_.size() * sizeof(int);
  if (backend_ == ExecutionBackend::CudaM2LP2P) {
    if (precision_ == StaticPrecision::Float32) {
      cuda_m2l_plan_ = std::make_unique<CudaM2LPlanOwner>(
          std::make_unique<CudaM2LPlan>(m2l_plan_float_));
    } else {
      cuda_m2l_plan_ = std::make_unique<CudaM2LPlanOwner>(
          std::make_unique<CudaM2LPlan>(m2l_plan_));
    }
    build_cuda_p2p_plan();
  }
  if (backend_ == ExecutionBackend::CudaFull) {
    build_cuda_full_plan();
  }
}

void UniformFmm::initialise_p2p_policy(const UniformFmmOptions &options) {
  cuda_p2p_bsr_max_bytes_ = options.cuda_p2p_bsr_max_bytes;
  if (!options.fixed_target_source_indices.has_value()) {
    return;
  }

  const std::vector<int> &identities =
      options.fixed_target_source_indices.value();
  const std::size_t target_count = tree_.sorted_target_positions().size();
  const std::size_t source_count = tree_.sorted_source_positions().size();
  if (identities.size() != target_count) {
    throw std::invalid_argument(
        "fixed_target_source_indices must contain one entry per target");
  }
  for (const int source_index : identities) {
    if (source_index < -1 || source_index >= static_cast<int>(source_count)) {
      throw std::invalid_argument(
          "fixed_target_source_indices contains an invalid source index");
    }
  }

  // Finite cuboid self fields are physical; identity maps only remove
  // singular point-dipole self interactions.
  if (source_geometry_ == SourceGeometry::UniformCuboid) {
    return;
  }

  fixed_target_source_indices_ = identities;
  prepare_self_indices(identities);
  fixed_sorted_self_indices_ = sorted_self_indices_;
}

void UniformFmm::initialise_source_geometry(const UniformFmmOptions &options) {
  source_geometry_ = options.source_geometry;
  use_cuboid_p2m_ =
      source_geometry_ == SourceGeometry::UniformCuboid &&
      options.use_cuboid_p2m;
  const std::size_t count = tree_.sorted_source_positions().size();
  if (source_geometry_ == SourceGeometry::PointDipole) {
    if (!options.source_sizes.empty()) {
      throw std::invalid_argument(
          "point-dipole sources do not accept cuboid sizes");
    }
    return;
  }
  if (options.source_sizes.size() != 1 &&
      options.source_sizes.size() != count) {
    throw std::invalid_argument(
        "cuboid sizes must contain one or one per source");
  }
  if (options.backend == ExecutionBackend::CpuReference ||
      (options.backend == ExecutionBackend::Auto &&
       options.m2l_backend == M2LBackend::Reference)) {
    throw std::invalid_argument("UniformCuboid sources require a static backend");
  }
  if (options.source_sizes.size() == 1) {
    sorted_source_sizes_ = options.source_sizes;
    return;
  }
  sorted_source_sizes_.resize(count);
  const auto permutation = tree_.source_permutation();
  for (std::size_t sorted = 0; sorted < count; ++sorted) {
    sorted_source_sizes_[sorted] = options.source_sizes[permutation[sorted]];
  }
}

void UniformFmm::build_cuda_p2p_plan() {
  if (precision_ == StaticPrecision::Float32) {
    if (fixed_target_source_indices_.has_value() &&
        p2p_bsr_plan_float_.memory().total_bytes() <=
            cuda_p2p_bsr_max_bytes_) {
      cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
          std::make_unique<CudaP2PPlan>(p2p_bsr_plan_float_));
      p2p_execution_packing_ = P2PExecutionPacking::CudaBsr3;
      return;
    }
    const std::span<const int> fixed_identities =
        fixed_target_source_indices_.has_value()
            ? std::span<const int>(fixed_sorted_self_indices_)
            : std::span<const int>{};
    cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
        std::make_unique<CudaP2PPlan>(p2p_operator_float_, fixed_identities));
    p2p_execution_packing_ = P2PExecutionPacking::CanonicalAos;
    return;
  }
  if (fixed_target_source_indices_.has_value()) {
    StaticP2PBsrPlan bsr =
        build_static_p2p_bsr_plan(p2p_operator_, fixed_sorted_self_indices_);
    if (bsr.memory().total_bytes() <= cuda_p2p_bsr_max_bytes_) {
      cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
          std::make_unique<CudaP2PPlan>(bsr));
      p2p_execution_packing_ = P2PExecutionPacking::CudaBsr3;
      return;
    }
  }

  const std::span<const int> fixed_identities =
      fixed_target_source_indices_.has_value()
          ? std::span<const int>(fixed_sorted_self_indices_)
          : std::span<const int>{};
  cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
      std::make_unique<CudaP2PPlan>(p2p_operator_, fixed_identities));
  p2p_execution_packing_ = P2PExecutionPacking::CanonicalAos;
}

void UniformFmm::build_static_plan() {
  // This is the geometry-dependent half of the evaluator. None of the data
  // built here depends on dipole moments, so it remains valid for every later
  // evaluate() call; see docs/static-architecture.md.
  const auto total_start = Clock::now();
  const auto nodes = tree_.nodes();
  using Key = std::tuple<int, int, int>;
  std::map<Key, std::vector<std::pair<int, int>>> classes;

  // FP32 expansion coefficients are expressed in root-width-normalised
  // coordinates. This prevents physical length powers from separately
  // underflowing multipoles and overflowing M2L scaling tables before their
  // product is formed. Moments are divided by scale^3 at the evaluation
  // boundary, so the resulting field retains its physical value.
  const Vec3 root_centre = nodes.empty() ? Vec3{} : nodes.front().centre;
  if (precision_ == StaticPrecision::Float32 && !nodes.empty()) {
    float_coordinate_scale_ = 2.0 * nodes.front().half_width;
  }
  const auto normalise_position = [&](const Vec3 position) {
    return precision_ == StaticPrecision::Float32
        ? (position - root_centre) * (1.0 / float_coordinate_scale_)
        : position;
  };
  const auto normalise_size = [&](const CuboidSize size) {
    return precision_ == StaticPrecision::Float32
        ? CuboidSize{size.hx / float_coordinate_scale_,
                     size.hy / float_coordinate_scale_,
                     size.hz / float_coordinate_scale_}
        : size;
  };

  std::vector<Vec3> operator_source_positions;
  std::vector<Vec3> operator_target_positions;
  std::vector<CuboidSize> operator_source_sizes;
  if (precision_ == StaticPrecision::Float32) {
    operator_source_positions.reserve(tree_.sorted_source_positions().size());
    for (const Vec3 position : tree_.sorted_source_positions()) {
      operator_source_positions.push_back(normalise_position(position));
    }
    operator_target_positions.reserve(tree_.sorted_target_positions().size());
    for (const Vec3 position : tree_.sorted_target_positions()) {
      operator_target_positions.push_back(normalise_position(position));
    }
    operator_source_sizes.reserve(sorted_source_sizes_.size());
    for (const CuboidSize size : sorted_source_sizes_) {
      operator_source_sizes.push_back(normalise_size(size));
    }
  }

  auto phase_start = Clock::now();
  const std::span<const Vec3> sorted_positions =
      precision_ == StaticPrecision::Float32
          ? std::span<const Vec3>(operator_source_positions)
          : tree_.sorted_source_positions();
  const std::span<const Vec3> sorted_targets =
      precision_ == StaticPrecision::Float32
          ? std::span<const Vec3>(operator_target_positions)
          : tree_.sorted_target_positions();
  const std::span<const CuboidSize> source_sizes =
      precision_ == StaticPrecision::Float32
          ? std::span<const CuboidSize>(operator_source_sizes)
          : std::span<const CuboidSize>(sorted_source_sizes_);
  p2m_plans_.reserve(tree_.occupied_source_leaves().size());
  for (const int leaf_index : tree_.occupied_source_leaves()) {
    const TreeNode &leaf = nodes[static_cast<std::size_t>(leaf_index)];
    P2MPlan plan;
    plan.leaf = leaf_index;
    const auto leaf_positions =
        sorted_positions.subspan(leaf.source_begin, leaf.source_count());
    if (use_cuboid_p2m_) {
      const std::span<const CuboidSize> leaf_sizes =
          source_sizes.size() == 1
              ? source_sizes
              : source_sizes.subspan(
                    leaf.source_begin, leaf.source_count());
      plan.operator_map = build_static_cuboid_p2m_operator(
          basis_, normalise_position(leaf.centre), leaf_positions, leaf_sizes);
    } else {
      plan.operator_map = build_static_p2m_operator(
          basis_, normalise_position(leaf.centre), leaf_positions);
    }
    static_plan_statistics_.operator_bytes +=
        plan.operator_map.entries.size() * sizeof(StaticOperatorEntry);
    p2m_plans_.push_back(std::move(plan));
  }
  static_plan_statistics_.p2m_plan.add(elapsed_seconds(phase_start));

  phase_start = Clock::now();
  m2m_operators_.resize(static_cast<std::size_t>(tree_.leaf_level() + 1));
  l2l_operators_.resize(static_cast<std::size_t>(tree_.leaf_level() + 1));
  static_plan_statistics_.m2m_theoretical_interactions =
      nodes.empty() ? 0 : nodes.size() - 1;
  static_plan_statistics_.l2l_theoretical_interactions =
      static_plan_statistics_.m2m_theoretical_interactions;
  for (int level = 1; level <= tree_.leaf_level(); ++level) {
    const double physical_child_half_width =
        nodes[static_cast<std::size_t>(level_offset(level))].half_width;
    const double child_half_width = precision_ == StaticPrecision::Float32
        ? physical_child_half_width / float_coordinate_scale_
        : physical_child_half_width;
    for (int child_class = 0; child_class < 8; ++child_class) {
      // Parity of the three child coordinates identifies one of the eight
      // parent-child displacements. Sharing by class avoids storing a copy of
      // the same triangular map for every tree edge.
      const Vec3 child_offset{
          (child_class & 1) != 0 ? child_half_width : -child_half_width,
          (child_class & 2) != 0 ? child_half_width : -child_half_width,
          (child_class & 4) != 0 ? child_half_width : -child_half_width};
      m2m_operators_[static_cast<std::size_t>(level)][child_class] =
          build_static_m2m_operator(basis_, child_offset * -1.0);
      l2l_operators_[static_cast<std::size_t>(level)][child_class] =
          build_static_l2l_operator(basis_, child_offset);
      const std::size_t m2m_bytes =
          m2m_operators_[static_cast<std::size_t>(level)][child_class]
              .entries.size() *
          sizeof(StaticOperatorEntry);
      const std::size_t l2l_bytes =
          l2l_operators_[static_cast<std::size_t>(level)][child_class]
              .entries.size() *
          sizeof(StaticOperatorEntry);
      static_plan_statistics_.m2m_operator_bytes += m2m_bytes;
      static_plan_statistics_.l2l_operator_bytes += l2l_bytes;
      static_plan_statistics_.operator_bytes += m2m_bytes + l2l_bytes;
      ++static_plan_statistics_.m2m_operators;
      ++static_plan_statistics_.l2l_operators;
    }
  }
  static_plan_statistics_.m2m_plan.add(elapsed_seconds(phase_start));
  // The two triangular families are constructed together from each shared
  // parent-child displacement class.
  static_plan_statistics_.l2l_plan = static_plan_statistics_.m2m_plan;

  phase_start = Clock::now();
  for (const TreeNode &target : nodes) {
    if (target.level == 0 || target.target_count() == 0) {
      continue;
    }
    for (const int source_index : target.list2) {
      const TreeNode &source = nodes[static_cast<std::size_t>(source_index)];
      if (source.source_count() == 0) {
        continue;
      }
      const Key key{target.ix - source.ix, target.iy - source.iy,
                    target.iz - source.iz};
      classes[key].emplace_back(source_index, target.index);
    }
  }
  static_plan_statistics_.transfer_discovery.add(elapsed_seconds(phase_start));

  const int coefficient_count = basis_.size();
  m2l_plan_.coefficient_count = coefficient_count;
  m2l_plan_.matrix_count = static_cast<int>(classes.size());
  m2l_plan_.level_count = tree_.leaf_level() + 1;
  m2l_plan_.target_row_offsets.assign(nodes.size() + 1, 0);
  m2l_plan_.level_target_begin.resize(
      static_cast<std::size_t>(m2l_plan_.level_count));
  m2l_plan_.level_target_end.resize(
      static_cast<std::size_t>(m2l_plan_.level_count));

  const std::size_t scaling_size =
      static_cast<std::size_t>(m2l_plan_.level_count) * coefficient_count;
  m2l_plan_.multipole_scaling.resize(scaling_size);
  m2l_plan_.local_scaling.resize(scaling_size);
  std::vector<double> inverse_width_powers(
      static_cast<std::size_t>(basis_.order() + 2), 1.0);

  for (int level = 0; level <= tree_.leaf_level(); ++level) {
    m2l_plan_.level_target_begin[static_cast<std::size_t>(level)] =
        level_offset(level);
    m2l_plan_.level_target_end[static_cast<std::size_t>(level)] =
        level_offset(level + 1);
    const double physical_box_width =
        2.0 * nodes[static_cast<std::size_t>(level_offset(level))].half_width;
    const double box_width = precision_ == StaticPrecision::Float32
        ? physical_box_width / float_coordinate_scale_
        : physical_box_width;
    const std::size_t scaling_offset =
        static_cast<std::size_t>(level) * coefficient_count;
    // The matrices below are dimensionless and level independent. These two
    // degree-dependent factors restore physical box width; see the M2L
    // normalisation section in docs/math.md.
    inverse_width_powers[0] = 1.0;
    for (int degree = 1; degree <= basis_.order() + 1; ++degree) {
      inverse_width_powers[static_cast<std::size_t>(degree)] =
          inverse_width_powers[static_cast<std::size_t>(degree - 1)] /
          box_width;
    }
    // Each coefficient is filled once from its known degree, rather than
    // rescanning the complete basis for every possible degree.
    for (int index = 0; index < coefficient_count; ++index) {
      const int degree = basis_[index].degree();
      m2l_plan_.multipole_scaling[scaling_offset + index] =
          inverse_width_powers[static_cast<std::size_t>(degree)];
      m2l_plan_.local_scaling[scaling_offset + index] =
          inverse_width_powers[static_cast<std::size_t>(degree + 1)];
    }
  }

  phase_start = Clock::now();
  std::size_t interaction_count = 0;
  for (const auto &[key, interactions] : classes) {
    const auto [dx, dy, dz] = key;
    const Vec3 R{static_cast<double>(dx), static_cast<double>(dy),
                 static_cast<double>(dz)};
    const std::vector<double> matrix = build_static_m2l_matrix(basis_, R);
    m2l_plan_.matrices.insert(m2l_plan_.matrices.end(), matrix.begin(),
                              matrix.end());
    for (const auto [source, target] : interactions) {
      (void)source;
      ++m2l_plan_.target_row_offsets[static_cast<std::size_t>(target) + 1];
      ++interaction_count;
    }
  }
  std::partial_sum(m2l_plan_.target_row_offsets.begin(),
                   m2l_plan_.target_row_offsets.end(),
                   m2l_plan_.target_row_offsets.begin());
  // Convert discovered interactions to CSR-like target rows. A target's
  // contributions are contiguous, giving the portable and CUDA executors one
  // output owner and deterministic accumulation without atomics on the CPU.
  m2l_plan_.source_nodes.resize(interaction_count);
  m2l_plan_.matrix_ids.resize(interaction_count);
  m2l_plan_.interaction_levels.resize(interaction_count);
  std::vector<int> row_cursors = m2l_plan_.target_row_offsets;
  int matrix_id = 0;
  for (const auto &[key, interactions] : classes) {
    (void)key;
    for (const auto [source, target] : interactions) {
      const int slot = row_cursors[static_cast<std::size_t>(target)]++;
      m2l_plan_.source_nodes[static_cast<std::size_t>(slot)] = source;
      m2l_plan_.matrix_ids[static_cast<std::size_t>(slot)] = matrix_id;
      m2l_plan_.interaction_levels[static_cast<std::size_t>(slot)] =
          nodes[static_cast<std::size_t>(target)].level;
    }
    ++matrix_id;
  }
  static_plan_statistics_.operator_construction.add(
      elapsed_seconds(phase_start));
  static_plan_statistics_.m2l_plan =
      static_plan_statistics_.operator_construction;

  phase_start = Clock::now();
  const std::size_t matrix_bytes = m2l_plan_.matrices.size() * sizeof(double);
  const std::size_t scaling_bytes =
      (m2l_plan_.multipole_scaling.size() + m2l_plan_.local_scaling.size()) *
      sizeof(double);
  const std::size_t metadata_bytes =
      (m2l_plan_.target_row_offsets.size() + m2l_plan_.source_nodes.size() +
       m2l_plan_.matrix_ids.size() + m2l_plan_.interaction_levels.size() +
       m2l_plan_.level_target_begin.size() +
       m2l_plan_.level_target_end.size()) *
      sizeof(int);
  static_plan_statistics_.operator_bytes += matrix_bytes + scaling_bytes;
  static_plan_statistics_.m2l_operator_bytes += matrix_bytes + scaling_bytes;
  static_plan_statistics_.interaction_bytes += metadata_bytes;
  static_plan_statistics_.m2l_interaction_bytes += metadata_bytes;
  static_plan_statistics_.interactions = interaction_count;

  // oneMKL uses an execution-only gather/GEMM/scatter packing derived from the
  // canonical target-row plan. Portable CPU and CUDA retain no such buffers.
  if (static_matrix_backend_ == StaticMatrixBackend::OneMkl) {
    m2l_groups_.resize(static_cast<std::size_t>(m2l_plan_.matrix_count));
    for (int id = 0; id < m2l_plan_.matrix_count; ++id) {
      m2l_groups_[static_cast<std::size_t>(id)].matrix_id = id;
    }
    for (std::size_t target = 0;
         target + 1 < m2l_plan_.target_row_offsets.size(); ++target) {
      const int begin = m2l_plan_.target_row_offsets[target];
      const int end = m2l_plan_.target_row_offsets[target + 1];
      for (int interaction = begin; interaction < end; ++interaction) {
        const std::size_t slot = static_cast<std::size_t>(interaction);
        M2LGroup &group =
            m2l_groups_[static_cast<std::size_t>(m2l_plan_.matrix_ids[slot])];
        group.sources.push_back(m2l_plan_.source_nodes[slot]);
        group.targets.push_back(static_cast<int>(target));
        group.levels.push_back(m2l_plan_.interaction_levels[slot]);
      }
    }
    for (M2LGroup &group : m2l_groups_) {
      const std::size_t values =
          static_cast<std::size_t>(coefficient_count) * group.sources.size();
      group.gathered.resize(values);
      group.translated.resize(values);
      const std::size_t group_metadata_bytes =
          (group.sources.size() + group.targets.size() + group.levels.size()) *
          sizeof(int);
      static_plan_statistics_.interaction_bytes += group_metadata_bytes;
      static_plan_statistics_.m2l_interaction_bytes += group_metadata_bytes;
      static_plan_statistics_.scratch_bytes += 2 * values * sizeof(double);
    }
  }
  static_plan_statistics_.transfer_classes = classes.size();
  static_plan_statistics_.m2l_operators = classes.size();
  static_plan_statistics_.buffer_allocation.add(elapsed_seconds(phase_start));
  phase_start = Clock::now();
  l2p_evaluators_.resize(sorted_targets.size());
  for (const int leaf_index : tree_.occupied_target_leaves()) {
    const TreeNode &leaf = nodes[static_cast<std::size_t>(leaf_index)];
    for (std::size_t target = leaf.target_begin; target < leaf.target_end;
         ++target) {
      l2p_evaluators_[target] =
          build_static_l2p_evaluator(
              basis_, normalise_position(leaf.centre), sorted_targets[target]);
      static_plan_statistics_.operator_bytes +=
          4 * static_cast<std::size_t>(basis_.size()) * sizeof(double);
    }
  }
  static_plan_statistics_.l2p_plan.add(elapsed_seconds(phase_start));

  phase_start = Clock::now();
  std::vector<std::array<int, 2>> near_interactions;
  for (const int leaf_index : tree_.occupied_target_leaves()) {
    const TreeNode &leaf = nodes[static_cast<std::size_t>(leaf_index)];
    for (std::size_t target = leaf.target_begin; target < leaf.target_end;
         ++target) {
      for (const int neighbour_index : leaf.list1) {
        const TreeNode &neighbour =
            nodes[static_cast<std::size_t>(neighbour_index)];
        for (std::size_t source = neighbour.source_begin;
             source < neighbour.source_end; ++source) {
          near_interactions.push_back(
              {static_cast<int>(target), static_cast<int>(source)});
        }
      }
    }
  }
  p2p_operator_ = build_static_p2p_operator(
      sorted_targets, sorted_positions, near_interactions, source_geometry_,
      source_sizes);
  p2p_compact_plan_ = build_static_p2p_compact_plan(p2p_operator_);
  if (backend_ == ExecutionBackend::CpuStatic) {
    p2p_execution_packing_ = P2PExecutionPacking::ParticleRowSoa;
  }
  static_plan_statistics_.p2p_interactions = p2p_operator_.blocks.size();
  static_plan_statistics_.p2p_value_bytes =
      p2p_operator_.blocks.size() * 6 * sizeof(double);
  static_plan_statistics_.p2p_index_bytes =
      p2p_compact_plan_.row_offsets.size() * sizeof(int) +
      p2p_compact_plan_.source_indices.size() * sizeof(int);
  static_plan_statistics_.operator_bytes +=
      p2p_operator_.memory_bytes() + p2p_compact_plan_.memory().total_bytes();
  static_plan_statistics_.p2p_tensor_plan.add(elapsed_seconds(phase_start));
  static_plan_statistics_.total.add(elapsed_seconds(total_start));
  ++static_plan_statistics_.construction_count;

  if (precision_ == StaticPrecision::Float32) {
    quantise_static_plan_to_float();
  }
}

void UniformFmm::quantise_static_plan_to_float() {
  p2m_plans_float_.reserve(p2m_plans_.size());
  for (const P2MPlan &plan : p2m_plans_) {
    p2m_plans_float_.push_back(
        {plan.leaf, quantise_static_operator(plan.operator_map)});
  }

  m2m_operators_float_.resize(m2m_operators_.size());
  l2l_operators_float_.resize(l2l_operators_.size());
  for (std::size_t level = 0; level < m2m_operators_.size(); ++level) {
    for (int child_class = 0; child_class < 8; ++child_class) {
      m2m_operators_float_[level][child_class] =
          quantise_static_operator(m2m_operators_[level][child_class]);
      l2l_operators_float_[level][child_class] =
          quantise_static_operator(l2l_operators_[level][child_class]);
    }
  }

  l2p_evaluators_float_.reserve(l2p_evaluators_.size());
  for (const StaticL2PEvaluator &evaluator : l2p_evaluators_) {
    l2p_evaluators_float_.push_back(
        quantise_static_l2p_evaluator(evaluator));
  }
  p2p_operator_float_ = quantise_static_p2p_operator(p2p_operator_);
  p2p_compact_plan_float_ =
      quantise_static_p2p_compact_plan(p2p_compact_plan_);
  if (fixed_target_source_indices_.has_value()) {
    p2p_bsr_plan_float_ = quantise_static_p2p_bsr_plan(
        build_static_p2p_bsr_plan(p2p_operator_, fixed_sorted_self_indices_));
  }
  m2l_plan_float_ = quantise_static_m2l_plan(m2l_plan_);

  m2l_groups_float_.reserve(m2l_groups_.size());
  for (const M2LGroup &group : m2l_groups_) {
    FloatM2LGroup converted;
    converted.matrix_id = group.matrix_id;
    converted.sources = group.sources;
    converted.targets = group.targets;
    converted.levels = group.levels;
    converted.gathered.resize(group.gathered.size());
    converted.translated.resize(group.translated.size());
    m2l_groups_float_.push_back(std::move(converted));
  }

  // Recalculate scalar-dependent storage from the representation that will
  // remain alive. Integer metadata is unchanged by precision selection.
  std::size_t operator_bytes = 0;
  std::size_t m2m_bytes = 0;
  std::size_t l2l_bytes = 0;
  for (const FloatP2MPlan &plan : p2m_plans_float_) {
    operator_bytes +=
        plan.operator_map.entries.size() * sizeof(FloatStaticOperatorEntry);
  }
  for (std::size_t level = 0; level < m2m_operators_float_.size(); ++level) {
    for (int child_class = 0; child_class < 8; ++child_class) {
      m2m_bytes += m2m_operators_float_[level][child_class].entries.size() *
                   sizeof(FloatStaticOperatorEntry);
      l2l_bytes += l2l_operators_float_[level][child_class].entries.size() *
                   sizeof(FloatStaticOperatorEntry);
    }
  }
  const std::size_t m2l_bytes =
      (m2l_plan_float_.matrices.size() +
       m2l_plan_float_.multipole_scaling.size() +
       m2l_plan_float_.local_scaling.size()) *
      sizeof(float);
  std::size_t l2p_bytes = 0;
  for (const FloatStaticL2PEvaluator &evaluator : l2p_evaluators_float_) {
    l2p_bytes += evaluator.potential.size() * sizeof(float);
    for (const std::vector<float> &field : evaluator.field) {
      l2p_bytes += field.size() * sizeof(float);
    }
  }
  operator_bytes += m2m_bytes + l2l_bytes + m2l_bytes + l2p_bytes;
  operator_bytes += p2p_operator_float_.memory_bytes();
  operator_bytes += p2p_compact_plan_float_.memory().total_bytes();

  static_plan_statistics_.scalar_bytes = sizeof(float);
  static_plan_statistics_.operator_bytes = operator_bytes;
  static_plan_statistics_.m2m_operator_bytes = m2m_bytes;
  static_plan_statistics_.m2l_operator_bytes = m2l_bytes;
  static_plan_statistics_.l2l_operator_bytes = l2l_bytes;
  static_plan_statistics_.p2p_value_bytes =
      p2p_operator_float_.blocks.size() * 6 * sizeof(float);
  static_plan_statistics_.scratch_bytes = 0;
  for (const FloatM2LGroup &group : m2l_groups_float_) {
    static_plan_statistics_.scratch_bytes +=
        (group.gathered.size() + group.translated.size()) * sizeof(float);
  }

  // Drop analytical FP64 construction temporaries. An FP32 plan therefore
  // retains no hidden double-precision operator or expansion representation.
  p2m_plans_.clear();
  p2m_plans_.shrink_to_fit();
  m2m_operators_.clear();
  m2m_operators_.shrink_to_fit();
  l2l_operators_.clear();
  l2l_operators_.shrink_to_fit();
  l2p_evaluators_.clear();
  l2p_evaluators_.shrink_to_fit();
  p2p_operator_ = {};
  p2p_compact_plan_ = {};
  m2l_plan_ = {};
  m2l_groups_.clear();
  m2l_groups_.shrink_to_fit();
}

void UniformFmm::build_cuda_full_plan() {
  if (precision_ == StaticPrecision::Float32) {
    FloatCudaFullPlanData data;
    data.coefficient_count = basis_.size();
    data.node_count = static_cast<int>(tree_.nodes().size());
    data.source_count =
        static_cast<int>(tree_.sorted_source_positions().size());
    data.target_count =
        static_cast<int>(tree_.sorted_target_positions().size());
    data.source_permutation.assign(tree_.source_permutation().begin(),
                                   tree_.source_permutation().end());
    data.target_permutation.assign(tree_.target_permutation().begin(),
                                   tree_.target_permutation().end());
    const int n = basis_.size();
    const auto nodes = tree_.nodes();
    for (const FloatP2MPlan &leaf_plan : p2m_plans_float_) {
      const TreeNode &leaf = nodes[static_cast<std::size_t>(leaf_plan.leaf)];
      for (FloatStaticOperatorEntry entry : leaf_plan.operator_map.entries) {
        entry.input += static_cast<int>(leaf.source_begin) * 3;
        entry.output += leaf.index * n;
        data.p2m.push_back(entry);
      }
    }
    if (tree_.leaf_level() > 0) {
      data.m2m.entries_per_matrix = static_cast<int>(
          m2m_operators_float_[1][0].entries.size());
      data.l2l.entries_per_matrix = static_cast<int>(
          l2l_operators_float_[1][0].entries.size());
    }
    for (int level = 1; level <= tree_.leaf_level(); ++level) {
      for (int child_class = 0; child_class < 8; ++child_class) {
        data.m2m.matrices.insert(
            data.m2m.matrices.end(),
            m2m_operators_float_[level][child_class].entries.begin(),
            m2m_operators_float_[level][child_class].entries.end());
        data.l2l.matrices.insert(
            data.l2l.matrices.end(),
            l2l_operators_float_[level][child_class].entries.begin(),
            l2l_operators_float_[level][child_class].entries.end());
      }
    }
    data.m2m.matrix_count = tree_.leaf_level() * 8;
    data.l2l.matrix_count = tree_.leaf_level() * 8;
    for (int level = 1; level <= tree_.leaf_level(); ++level) {
      const int begin = level_offset(level);
      const int end = level_offset(level + 1);
      for (int child_index = begin; child_index < end; ++child_index) {
        const TreeNode &child = nodes[static_cast<std::size_t>(child_index)];
        const int child_class =
            (child.ix & 1) | ((child.iy & 1) << 1) |
            ((child.iz & 1) << 2);
        const int matrix_id = (level - 1) * 8 + child_class;
        if (child.source_count() != 0) {
          data.m2m.interactions.push_back(
              {child.index, child.parent, matrix_id, level});
        }
        if (child.target_count() != 0) {
          data.l2l.interactions.push_back(
              {child.parent, child.index, matrix_id, level});
        }
      }
    }
    data.m2l = m2l_plan_float_;
    for (const int leaf_index : tree_.occupied_target_leaves()) {
      const TreeNode &leaf = nodes[static_cast<std::size_t>(leaf_index)];
      for (std::size_t target = leaf.target_begin;
           target < leaf.target_end; ++target) {
        for (int component = 0; component < 3; ++component) {
          for (int coefficient = 0; coefficient < n; ++coefficient) {
            const float value =
                l2p_evaluators_float_[target].field[component][coefficient];
            if (value != 0.0F) {
              data.l2p.push_back(
                  {static_cast<int>(target) * 3 + component,
                   leaf.index * n + coefficient, value});
            }
          }
        }
      }
    }
    data.p2p = p2p_operator_float_;
    if (fixed_target_source_indices_.has_value()) {
      data.has_fixed_self_indices = true;
      data.fixed_self_indices = fixed_sorted_self_indices_;
      data.p2p_bsr = p2p_bsr_plan_float_;
      data.use_p2p_bsr =
          data.p2p_bsr.memory().total_bytes() <= cuda_p2p_bsr_max_bytes_;
    }
    p2p_execution_packing_ = data.use_p2p_bsr
        ? P2PExecutionPacking::CudaBsr3
        : P2PExecutionPacking::CanonicalAos;
    cuda_full_plan_ = std::make_unique<CudaFullPlanOwner>(
        std::make_unique<CudaFullPlan>(data));
    return;
  }

  CudaFullPlanData data;
  data.coefficient_count = basis_.size();
  data.node_count = static_cast<int>(tree_.nodes().size());
  data.source_count = static_cast<int>(tree_.sorted_source_positions().size());
  data.target_count = static_cast<int>(tree_.sorted_target_positions().size());
  data.source_permutation.assign(tree_.source_permutation().begin(),
                                 tree_.source_permutation().end());
  data.target_permutation.assign(tree_.target_permutation().begin(),
                                 tree_.target_permutation().end());
  const int n = basis_.size();
  const auto nodes = tree_.nodes();

  for (const P2MPlan &leaf_plan : p2m_plans_) {
    const TreeNode &leaf = nodes[static_cast<std::size_t>(leaf_plan.leaf)];
    for (StaticOperatorEntry entry : leaf_plan.operator_map.entries) {
      entry.input += static_cast<int>(leaf.source_begin) * 3;
      entry.output += leaf.index * n;
      data.p2m.push_back(entry);
    }
  }
  if (tree_.leaf_level() > 0) {
    data.m2m.entries_per_matrix =
        static_cast<int>(m2m_operators_[1][0].entries.size());
    data.l2l.entries_per_matrix =
        static_cast<int>(l2l_operators_[1][0].entries.size());
  }
  for (int level = 1; level <= tree_.leaf_level(); ++level) {
    for (int child_class = 0; child_class < 8; ++child_class) {
      const int matrix_id = (level - 1) * 8 + child_class;
      data.m2m.matrices.insert(
          data.m2m.matrices.end(),
          m2m_operators_[level][child_class].entries.begin(),
          m2m_operators_[level][child_class].entries.end());
      data.l2l.matrices.insert(
          data.l2l.matrices.end(),
          l2l_operators_[level][child_class].entries.begin(),
          l2l_operators_[level][child_class].entries.end());
      (void)matrix_id;
    }
  }
  data.m2m.matrix_count = tree_.leaf_level() * 8;
  data.l2l.matrix_count = tree_.leaf_level() * 8;
  for (int level = 1; level <= tree_.leaf_level(); ++level) {
    const int begin = level_offset(level);
    const int end = level_offset(level + 1);
    for (int child_index = begin; child_index < end; ++child_index) {
      const TreeNode &child = nodes[static_cast<std::size_t>(child_index)];
      const int child_class =
          (child.ix & 1) | ((child.iy & 1) << 1) | ((child.iz & 1) << 2);
      const int matrix_id = (level - 1) * 8 + child_class;
      if (child.source_count() != 0) {
        data.m2m.interactions.push_back(
            {child.index, child.parent, matrix_id, level});
      }
      if (child.target_count() != 0) {
        data.l2l.interactions.push_back(
            {child.parent, child.index, matrix_id, level});
      }
    }
  }
  data.m2l = m2l_plan_;
  const auto occupied_leaves = tree_.occupied_target_leaves();
  for (const int leaf_index : occupied_leaves) {
    const TreeNode &leaf = nodes[static_cast<std::size_t>(leaf_index)];
    for (std::size_t target = leaf.target_begin; target < leaf.target_end;
         ++target) {
      for (int component = 0; component < 3; ++component) {
        for (int coefficient = 0; coefficient < n; ++coefficient) {
          const double value =
              l2p_evaluators_[target].field[component][coefficient];
          if (value != 0.0) {
            data.l2p.push_back({static_cast<int>(target) * 3 + component,
                                leaf.index * n + coefficient, value});
          }
        }
      }
    }
  }
  data.p2p = p2p_operator_;
  if (fixed_target_source_indices_.has_value()) {
    data.has_fixed_self_indices = true;
    data.fixed_self_indices = fixed_sorted_self_indices_;
    data.p2p_bsr =
        build_static_p2p_bsr_plan(p2p_operator_, fixed_sorted_self_indices_);
    data.use_p2p_bsr =
        data.p2p_bsr.memory().total_bytes() <= cuda_p2p_bsr_max_bytes_;
  }
  p2p_execution_packing_ = data.use_p2p_bsr ? P2PExecutionPacking::CudaBsr3
                                            : P2PExecutionPacking::CanonicalAos;
  cuda_full_plan_ =
      std::make_unique<CudaFullPlanOwner>(std::make_unique<CudaFullPlan>(data));
}

// Far-field operator execution is implemented in far_field.cpp.  Keeping the
// complete P2M -> M2M -> M2L -> L2L chain together makes this file focus on
// evaluation setup, near/far scheduling, result assembly, and inspection.

std::vector<PotentialField>
UniformFmm::evaluate(std::span<const Vec3> dipole_moments,
                     const OutputFlags output,
                     std::span<const int> target_source_indices) {
  std::vector<PotentialField> results(tree_.sorted_target_positions().size());
  evaluate_into(dipole_moments, results, output, target_source_indices);
  return results;
}

void UniformFmm::prepare_self_indices(
    const std::span<const int> target_source_indices) {
  const auto target_permutation = tree_.target_permutation();
  const auto source_inverse = tree_.source_inverse_permutation();
  std::fill(sorted_self_indices_.begin(), sorted_self_indices_.end(), -1);
  for (std::size_t target_index = 0;
       target_index < target_source_indices.size(); ++target_index) {
    const int original_target = target_permutation[target_index];
    const int original_source =
        target_source_indices[static_cast<std::size_t>(original_target)];
    if (original_source >= 0) {
      sorted_self_indices_[target_index] =
          source_inverse[static_cast<std::size_t>(original_source)];
    }
  }
}

std::span<const int> UniformFmm::resolve_self_indices(
    const std::span<const int> target_source_indices) const {
  if (source_geometry_ == SourceGeometry::UniformCuboid) {
    return {};
  }
  if (!fixed_target_source_indices_.has_value()) {
    return target_source_indices;
  }
  const std::vector<int> &fixed = fixed_target_source_indices_.value();
  if (target_source_indices.empty()) {
    return fixed;
  }
  if (!std::equal(target_source_indices.begin(), target_source_indices.end(),
                  fixed.begin(), fixed.end())) {
    throw std::invalid_argument(
        "fixed target-to-source identity map changed; rebuild the FMM plan");
  }
  return target_source_indices;
}

void UniformFmm::evaluate_into(std::span<const Vec3> dipole_moments,
                               std::span<PotentialField> results,
                               const OutputFlags output,
                               std::span<const int> target_source_indices) {
  if (precision_ == StaticPrecision::Float32) {
    std::vector<FloatPotentialField> float_results(results.size());
    evaluate_into_float32(dipole_moments, float_results, output,
                          target_source_indices);
    for (std::size_t index = 0; index < results.size(); ++index) {
      results[index].phi = static_cast<double>(float_results[index].phi);
      results[index].H = {
          static_cast<double>(float_results[index].H.x),
          static_cast<double>(float_results[index].H.y),
          static_cast<double>(float_results[index].H.z)};
    }
    return;
  }
  detail::ProfileRange evaluation_range{"cdfmm/evaluate"};
  const std::size_t target_count = tree_.sorted_target_positions().size();
  target_source_indices = resolve_self_indices(target_source_indices);
  if (results.size() != target_count) {
    throw std::invalid_argument(
        "UniformFmm::evaluate_into requires one result per target");
  }
  if (!target_source_indices.empty() &&
      target_source_indices.size() != target_count) {
    throw std::invalid_argument(
        "UniformFmm::evaluate identity map has incorrect length");
  }
  for (const int source_index : target_source_indices) {
    if (source_index < -1 ||
        source_index >= static_cast<int>(dipole_moments.size())) {
      throw std::invalid_argument(
          "UniformFmm::evaluate identity map contains an invalid index");
    }
  }

  if (backend_ == ExecutionBackend::CudaFull) {
    if (output != OutputFlags::Field) {
      throw std::invalid_argument(
          "CudaFull currently supports field-only evaluation");
    }
    last_timings_ = {};
    const auto evaluation_start = Clock::now();
    prepare_self_indices(target_source_indices);
    std::vector<Vec3> fields(target_count);
    detail::ProfileRange device_range{"cdfmm/cuda_full"};
    cuda_full_plan_->plan->evaluate(dipole_moments, fields,
                                    sorted_self_indices_);
    for (std::size_t target = 0; target < target_count; ++target) {
      results[target].phi = 0.0;
      results[target].H = fields[target];
    }
    const CudaEvaluationTimings &device = cuda_full_plan_->plan->timings();
    last_timings_.cuda_h2d.add(device.h2d_seconds);
    last_timings_.p2m.add(device.p2m_seconds);
    last_timings_.m2m.add(device.m2m_seconds);
    last_timings_.m2l.add(device.m2l_seconds);
    last_timings_.m2l_scale.add(device.scale_seconds);
    last_timings_.m2l_multiply.add(device.multiply_seconds);
    last_timings_.l2l.add(device.l2l_seconds);
    last_timings_.l2p.add(device.l2p_seconds);
    last_timings_.p2p.add(device.p2p_seconds);
    last_timings_.cuda_p2p_kernel.add(device.p2p_seconds);
    last_timings_.result_unpermutation.add(device.accumulation_seconds);
    last_timings_.cuda_kernel.add(device.kernel_seconds);
    last_timings_.cuda_d2h.add(device.d2h_seconds);
    last_timings_.total.add(elapsed_seconds(evaluation_start));
    last_timings_.evaluations = 1;
    accumulate_timings(aggregate_timings_, last_timings_);
    return;
  }

  last_timings_ = {};
  const auto evaluation_start = Clock::now();
  prepare_moments(dipole_moments);
  prepare_self_indices(target_source_indices);

  const bool use_cuda_p2p =
      execution_plan().p2p == StaticOperatorExecutor::Cuda &&
      has_flag(output, OutputFlags::Field) && cuda_p2p_plan_;
  PendingCudaP2PGuard p2p_guard(use_cuda_p2p ? cuda_p2p_plan_->plan.get()
                                             : nullptr);
  if (use_cuda_p2p) {
    // Near-field P2P is independent of the far-field hierarchy. Starting it
    // now overlaps its private CUDA stream with CPU P2M/M2M and CUDA M2L.
    cuda_p2p_plan_->plan->begin_evaluate(sorted_dipole_moments_,
                                         sorted_self_indices_);
    p2p_guard.arm();
  }

  {
    detail::ProfileRange far_range{"cdfmm/far_field"};
    upward_pass_prepared();
    downward_pass();
  }

  const auto nodes = tree_.nodes();
  const auto targets = tree_.sorted_target_positions();
  const auto sources = tree_.sorted_source_positions();
  const auto target_permutation = tree_.target_permutation();
  const auto occupied_leaves = tree_.occupied_target_leaves();

  auto phase_start = Clock::now();
  detail::ProfileRange l2p_range{"cdfmm/far_field/l2p"};
#pragma omp parallel for schedule(static) if (occupied_leaves.size() >= 8)
  for (std::ptrdiff_t occupied_index = 0;
       occupied_index < static_cast<std::ptrdiff_t>(occupied_leaves.size());
       ++occupied_index) {
    const int leaf_index =
        occupied_leaves[static_cast<std::size_t>(occupied_index)];
    const TreeNode &leaf = nodes[static_cast<std::size_t>(leaf_index)];
    for (std::size_t target_index = leaf.target_begin;
         target_index < leaf.target_end; ++target_index) {
      if (execution_plan().l2p != StaticOperatorExecutor::Reference) {
        sorted_results_[target_index] = apply_static_l2p_evaluator(
            l2p_evaluators_[target_index],
            local_for_node(leaf_index), output);
      } else {
        sorted_results_[target_index] =
            l2p_eval(basis_, leaf.centre, targets[target_index],
                     local_for_node(leaf_index), output);
      }
    }
  }
  last_timings_.l2p.add(elapsed_seconds(phase_start));

  if (execution_plan().p2p != StaticOperatorExecutor::Reference &&
      has_flag(output, OutputFlags::Field)) {
    detail::ProfileRange near_range{"cdfmm/near_field"};
    std::fill(near_fields_.begin(), near_fields_.end(), Vec3{});
    if (use_cuda_p2p) {
      // This is the first point at which final assembly needs the near
      // field, so delaying the wait preserves all available overlap.
      const auto wait_start = Clock::now();
      cuda_p2p_plan_->plan->finish_evaluate(near_fields_);
      last_timings_.cuda_p2p_wait.add(elapsed_seconds(wait_start));
      p2p_guard.release();
      const CudaEvaluationTimings &device = cuda_p2p_plan_->plan->timings();
      last_timings_.cuda_h2d.add(device.h2d_seconds);
      last_timings_.cuda_kernel.add(device.kernel_seconds);
      last_timings_.cuda_d2h.add(device.d2h_seconds);
      last_timings_.cuda_p2p_h2d.add(device.h2d_seconds);
      last_timings_.cuda_p2p_kernel.add(device.kernel_seconds);
      last_timings_.cuda_p2p_d2h.add(device.d2h_seconds);
      last_timings_.p2p.add(device.h2d_seconds + device.kernel_seconds +
                            device.d2h_seconds);
    } else {
      detail::ProfileRange p2p_range{"cdfmm/near_field/p2p"};
      const auto p2p_start = Clock::now();
      detail::evaluate_static_near_field(p2p_compact_plan_,
                                         sorted_dipole_moments_, near_fields_,
                                         sorted_self_indices_);
      last_timings_.p2p.add(elapsed_seconds(p2p_start));
    }
    for (std::size_t target = 0; target < target_count; ++target) {
      sorted_results_[target].H += near_fields_[target];
    }
  }

  phase_start = Clock::now();
  detail::ProfileRange output_range{"cdfmm/output_permutation"};
  const OutputFlags reference_near_output =
      execution_plan().p2p == StaticOperatorExecutor::Reference
          ? output
          : (has_flag(output, OutputFlags::Potential) ? OutputFlags::Potential
                                                      : OutputFlags::None);
  // The compact static tensor represents H only. Potential requests retain
  // the direct list1 formula rather than changing the tensor representation.
  if (reference_near_output != OutputFlags::None) {
    detail::evaluate_reference_near_field(
        tree_, sorted_dipole_moments_, sorted_self_indices_,
        reference_near_output, sorted_results_);
  }
  if (reference_near_output != OutputFlags::None) {
    last_timings_.p2p.add(elapsed_seconds(phase_start));
  }

  phase_start = Clock::now();
#pragma omp parallel for schedule(static) if (target_count >= 256)
  for (std::ptrdiff_t sorted_index = 0;
       sorted_index < static_cast<std::ptrdiff_t>(target_count);
       ++sorted_index) {
    const int original_index =
        target_permutation[static_cast<std::size_t>(sorted_index)];
    results[static_cast<std::size_t>(original_index)] =
        sorted_results_[static_cast<std::size_t>(sorted_index)];
  }
  last_timings_.result_unpermutation.add(elapsed_seconds(phase_start));
  last_timings_.total.add(elapsed_seconds(evaluation_start));
  last_timings_.evaluations = 1;
  accumulate_timings(aggregate_timings_, last_timings_);
}

std::vector<FloatPotentialField> UniformFmm::evaluate_float32(
    const std::span<const Vec3> dipole_moments, const OutputFlags output,
    const std::span<const int> target_source_indices) {
  if (precision_ != StaticPrecision::Float32) {
    throw std::logic_error("evaluate_float32 requires an FP32 FMM plan");
  }
  std::vector<FloatPotentialField> results(
      tree_.sorted_target_positions().size());
  evaluate_into_float32(dipole_moments, results, output,
                        target_source_indices);
  return results;
}

std::vector<PotentialField> UniformFmm::evaluate_float64(
    const std::span<const Vec3> dipole_moments, const OutputFlags output,
    const std::span<const int> target_source_indices) {
  if (precision_ != StaticPrecision::Float64) {
    throw std::logic_error("evaluate_float64 requires an FP64 FMM plan");
  }
  return evaluate(dipole_moments, output, target_source_indices);
}

void UniformFmm::evaluate_into_float32(
    const std::span<const Vec3> dipole_moments,
    const std::span<FloatPotentialField> results, const OutputFlags output,
    std::span<const int> target_source_indices) {
  if (precision_ != StaticPrecision::Float32) {
    throw std::logic_error("evaluate_into_float32 requires an FP32 FMM plan");
  }
  const std::size_t target_count = tree_.sorted_target_positions().size();
  target_source_indices = resolve_self_indices(target_source_indices);
  if (results.size() != target_count) {
    throw std::invalid_argument(
        "UniformFmm::evaluate_into_float32 requires one result per target");
  }
  if (!target_source_indices.empty() &&
      target_source_indices.size() != target_count) {
    throw std::invalid_argument(
        "UniformFmm::evaluate identity map has incorrect length");
  }
  for (const int source_index : target_source_indices) {
    if (source_index < -1 ||
        source_index >= static_cast<int>(dipole_moments.size())) {
      throw std::invalid_argument(
          "UniformFmm::evaluate identity map contains an invalid index");
    }
  }

  if (backend_ == ExecutionBackend::CudaFull) {
    if (output != OutputFlags::Field) {
      throw std::invalid_argument(
          "CudaFull currently supports field-only evaluation");
    }
    last_timings_ = {};
    const auto evaluation_start = Clock::now();
    prepare_self_indices(target_source_indices);
    std::vector<FloatVec3> moments_float(dipole_moments.size());
    for (std::size_t index = 0; index < dipole_moments.size(); ++index) {
      const Vec3 value = dipole_moments[index];
      const double scale = float_coordinate_scale_;
      moments_float[index] = {
          static_cast<float>(value.x / scale / scale / scale),
          static_cast<float>(value.y / scale / scale / scale),
          static_cast<float>(value.z / scale / scale / scale)};
    }
    std::vector<FloatVec3> fields(target_count);
    cuda_full_plan_->plan->evaluate(moments_float, fields,
                                    sorted_self_indices_);
    for (std::size_t target = 0; target < target_count; ++target) {
      results[target].phi = 0.0F;
      results[target].H = fields[target];
    }
    const CudaEvaluationTimings &device = cuda_full_plan_->plan->timings();
    last_timings_.cuda_h2d.add(device.h2d_seconds);
    last_timings_.p2m.add(device.p2m_seconds);
    last_timings_.m2m.add(device.m2m_seconds);
    last_timings_.m2l.add(device.m2l_seconds);
    last_timings_.m2l_scale.add(device.scale_seconds);
    last_timings_.m2l_multiply.add(device.multiply_seconds);
    last_timings_.l2l.add(device.l2l_seconds);
    last_timings_.l2p.add(device.l2p_seconds);
    last_timings_.p2p.add(device.p2p_seconds);
    last_timings_.cuda_p2p_kernel.add(device.p2p_seconds);
    last_timings_.result_unpermutation.add(device.accumulation_seconds);
    last_timings_.cuda_kernel.add(device.kernel_seconds);
    last_timings_.cuda_d2h.add(device.d2h_seconds);
    last_timings_.total.add(elapsed_seconds(evaluation_start));
    last_timings_.evaluations = 1;
    accumulate_timings(aggregate_timings_, last_timings_);
    return;
  }

  last_timings_ = {};
  const auto evaluation_start = Clock::now();
  prepare_moments_float(dipole_moments);
  prepare_self_indices(target_source_indices);
  const bool use_cuda_p2p =
      execution_plan().p2p == StaticOperatorExecutor::Cuda &&
      has_flag(output, OutputFlags::Field) && cuda_p2p_plan_;
  PendingCudaP2PGuard p2p_guard(use_cuda_p2p ? cuda_p2p_plan_->plan.get()
                                             : nullptr);
  if (use_cuda_p2p) {
    cuda_p2p_plan_->plan->begin_evaluate(sorted_dipole_moments_float_,
                                         sorted_self_indices_);
    p2p_guard.arm();
  }
  upward_pass_prepared_float();
  downward_pass_float();

  const auto nodes = tree_.nodes();
  const auto target_permutation = tree_.target_permutation();
  const auto occupied_leaves = tree_.occupied_target_leaves();

  auto phase_start = Clock::now();
#pragma omp parallel for schedule(static) if (occupied_leaves.size() >= 8)
  for (std::ptrdiff_t occupied_index = 0;
       occupied_index < static_cast<std::ptrdiff_t>(occupied_leaves.size());
       ++occupied_index) {
    const int leaf_index =
        occupied_leaves[static_cast<std::size_t>(occupied_index)];
    const TreeNode &leaf = nodes[static_cast<std::size_t>(leaf_index)];
    for (std::size_t target_index = leaf.target_begin;
         target_index < leaf.target_end; ++target_index) {
      sorted_results_float_[target_index] = apply_static_l2p_evaluator(
          l2p_evaluators_float_[target_index],
          local_float_for_node(leaf_index), output);
    }
  }
  last_timings_.l2p.add(elapsed_seconds(phase_start));

  if (has_flag(output, OutputFlags::Field)) {
    phase_start = Clock::now();
    std::fill(near_fields_float_.begin(), near_fields_float_.end(),
              FloatVec3{});
    if (use_cuda_p2p) {
      cuda_p2p_plan_->plan->finish_evaluate(near_fields_float_);
      p2p_guard.release();
      const CudaEvaluationTimings &device = cuda_p2p_plan_->plan->timings();
      last_timings_.cuda_h2d.add(device.h2d_seconds);
      last_timings_.cuda_kernel.add(device.kernel_seconds);
      last_timings_.cuda_d2h.add(device.d2h_seconds);
      last_timings_.cuda_p2p_h2d.add(device.h2d_seconds);
      last_timings_.cuda_p2p_kernel.add(device.kernel_seconds);
      last_timings_.cuda_p2p_d2h.add(device.d2h_seconds);
    } else {
      apply_static_p2p_compact_plan(
          p2p_compact_plan_float_, sorted_dipole_moments_float_,
          near_fields_float_, sorted_self_indices_);
    }
    for (std::size_t target = 0; target < target_count; ++target) {
      sorted_results_float_[target].H += near_fields_float_[target];
    }
    last_timings_.p2p.add(elapsed_seconds(phase_start));
  }

  if (has_flag(output, OutputFlags::Potential)) {
    const FloatStaticP2PCompactPlan &plan = p2p_compact_plan_float_;
#pragma omp parallel for schedule(static) if (target_count >= 64)
    for (std::ptrdiff_t target = 0;
         target < static_cast<std::ptrdiff_t>(target_count); ++target) {
      float potential = 0.0F;
      const int self = sorted_self_indices_[static_cast<std::size_t>(target)];
      const int begin = plan.row_offsets[static_cast<std::size_t>(target)];
      const int end = plan.row_offsets[static_cast<std::size_t>(target) + 1];
      for (int entry = begin; entry < end; ++entry) {
        const std::size_t index = static_cast<std::size_t>(entry);
        const int source = plan.source_indices[index];
        if (source == self) {
          continue;
        }
        const FloatVec3 moment =
            sorted_dipole_moments_float_[static_cast<std::size_t>(source)];
        potential += plan.potential[0][index] * moment.x +
            plan.potential[1][index] * moment.y +
            plan.potential[2][index] * moment.z;
      }
      sorted_results_float_[static_cast<std::size_t>(target)].phi += potential;
    }
  }

  phase_start = Clock::now();
#pragma omp parallel for schedule(static) if (target_count >= 256)
  for (std::ptrdiff_t sorted_index = 0;
       sorted_index < static_cast<std::ptrdiff_t>(target_count);
       ++sorted_index) {
    const int original_index =
        target_permutation[static_cast<std::size_t>(sorted_index)];
    results[static_cast<std::size_t>(original_index)] =
        sorted_results_float_[static_cast<std::size_t>(sorted_index)];
    // The normalised-coordinate potential differs by one power of length;
    // the field is invariant after moment scaling by root_width^-3.
    results[static_cast<std::size_t>(original_index)].phi *=
        static_cast<float>(float_coordinate_scale_);
  }
  last_timings_.result_unpermutation.add(elapsed_seconds(phase_start));
  last_timings_.total.add(elapsed_seconds(evaluation_start));
  last_timings_.evaluations = 1;
  accumulate_timings(aggregate_timings_, last_timings_);
}

//------------------------------------------------------------------------------
// Public inspection
//------------------------------------------------------------------------------

const UniformTree &UniformFmm::tree() const { return tree_; }
const MultiIndexSet &UniformFmm::basis() const { return basis_; }
M2LBackend UniformFmm::m2l_backend() const { return m2l_backend_; }
StaticMatrixBackend UniformFmm::static_matrix_backend() const {
  return static_matrix_backend_;
}
ExecutionBackend UniformFmm::backend() const { return backend_; }
StaticPrecision UniformFmm::precision() const noexcept { return precision_; }
StaticExecutionPlan UniformFmm::execution_plan() const noexcept {
  if (backend_ == ExecutionBackend::CpuReference) {
    return {
        StaticOperatorExecutor::Reference, StaticOperatorExecutor::Reference,
        StaticOperatorExecutor::Reference, StaticOperatorExecutor::Reference,
        StaticOperatorExecutor::Reference, StaticOperatorExecutor::Reference};
  }

  const StaticOperatorExecutor matrix_executor =
      static_matrix_backend_ == StaticMatrixBackend::OneMkl
          ? StaticOperatorExecutor::OneMkl
          : StaticOperatorExecutor::Portable;
  if (backend_ == ExecutionBackend::CudaFull) {
    return {StaticOperatorExecutor::Cuda, StaticOperatorExecutor::Cuda,
            StaticOperatorExecutor::Cuda, StaticOperatorExecutor::Cuda,
            StaticOperatorExecutor::Cuda, StaticOperatorExecutor::Cuda};
  }
  if (backend_ == ExecutionBackend::CudaPartial) {
    return {StaticOperatorExecutor::Portable, StaticOperatorExecutor::Portable,
            StaticOperatorExecutor::Cuda,     StaticOperatorExecutor::Portable,
            StaticOperatorExecutor::Portable, StaticOperatorExecutor::Cuda};
  }
  return {StaticOperatorExecutor::Portable,
          StaticOperatorExecutor::Portable,
          matrix_executor,
          StaticOperatorExecutor::Portable,
          StaticOperatorExecutor::Portable,
          StaticOperatorExecutor::Portable};
}
P2PExecutionPacking UniformFmm::p2p_execution_packing() const noexcept {
  return p2p_execution_packing_;
}
const CudaPlanStatistics &UniformFmm::cuda_plan_statistics() const {
  if (cuda_full_plan_) {
    return cuda_full_plan_->plan->statistics();
  }
  if (!cuda_m2l_plan_) {
    return empty_cuda_statistics_;
  }
  empty_cuda_statistics_ = cuda_m2l_plan_->plan->statistics();
  if (cuda_p2p_plan_) {
    const CudaPlanStatistics &p2p = cuda_p2p_plan_->plan->statistics();
    empty_cuda_statistics_.setup_h2d_bytes += p2p.setup_h2d_bytes;
    empty_cuda_statistics_.evaluation_h2d_bytes += p2p.evaluation_h2d_bytes;
    empty_cuda_statistics_.evaluation_d2h_bytes += p2p.evaluation_d2h_bytes;
    empty_cuda_statistics_.evaluation_h2d_calls += p2p.evaluation_h2d_calls;
    empty_cuda_statistics_.evaluation_d2h_calls += p2p.evaluation_d2h_calls;
    empty_cuda_statistics_.persistent_device_bytes +=
        p2p.persistent_device_bytes;
    empty_cuda_statistics_.p2p_interaction_count = p2p.p2p_interaction_count;
    empty_cuda_statistics_.p2p_tensor_bytes = p2p.p2p_tensor_bytes;
    empty_cuda_statistics_.p2p_index_bytes = p2p.p2p_index_bytes;
    empty_cuda_statistics_.p2p_row_metadata_bytes = p2p.p2p_row_metadata_bytes;
    empty_cuda_statistics_.p2p_leaf_metadata_bytes =
        p2p.p2p_leaf_metadata_bytes;
    empty_cuda_statistics_.p2p_identity_bytes = p2p.p2p_identity_bytes;
    empty_cuda_statistics_.p2p_scratch_bytes = p2p.p2p_scratch_bytes;
    empty_cuda_statistics_.p2p_threads_per_block = p2p.p2p_threads_per_block;
    empty_cuda_statistics_.static_p2p_upload_count =
        p2p.static_p2p_upload_count;
  }
  return empty_cuda_statistics_;
}
const StaticPlanStatistics &UniformFmm::static_plan_statistics() const {
  return static_plan_statistics_;
}

std::span<const double> UniformFmm::multipole(const int node_index) const {
  if (node_index < 0 ||
      static_cast<std::size_t>(node_index) >= tree_.nodes().size()) {
    throw std::out_of_range("multipole node index is out of range");
  }
  if (precision_ == StaticPrecision::Float64) {
    return multipole_for_node(node_index);
  }
  const auto values = multipole_float_for_node(node_index);
  inspection_widening_buffer_.assign(values.begin(), values.end());
  return inspection_widening_buffer_;
}

std::span<const double> UniformFmm::local(const int node_index) const {
  if (node_index < 0 ||
      static_cast<std::size_t>(node_index) >= tree_.nodes().size()) {
    throw std::out_of_range("local node index is out of range");
  }
  if (precision_ == StaticPrecision::Float64) {
    return local_for_node(node_index);
  }
  const auto values = local_float_for_node(node_index);
  inspection_widening_buffer_.assign(values.begin(), values.end());
  return inspection_widening_buffer_;
}

std::span<const double> UniformFmm::root_multipole() const {
  return multipole(0);
}

std::span<const float> UniformFmm::multipole_float32(
    const int node_index) const {
  if (precision_ != StaticPrecision::Float32) {
    throw std::logic_error("multipole_float32 requires an FP32 FMM plan");
  }
  if (node_index < 0 ||
      static_cast<std::size_t>(node_index) >= tree_.nodes().size()) {
    throw std::out_of_range("multipole node index is out of range");
  }
  return multipole_float_for_node(node_index);
}

std::span<const float> UniformFmm::local_float32(const int node_index) const {
  if (precision_ != StaticPrecision::Float32) {
    throw std::logic_error("local_float32 requires an FP32 FMM plan");
  }
  if (node_index < 0 ||
      static_cast<std::size_t>(node_index) >= tree_.nodes().size()) {
    throw std::out_of_range("local node index is out of range");
  }
  return local_float_for_node(node_index);
}

std::span<const float> UniformFmm::root_multipole_float32() const {
  return multipole_float32(0);
}

std::span<const double> UniformFmm::multipole_float64(
    const int node_index) const {
  if (precision_ != StaticPrecision::Float64) {
    throw std::logic_error("multipole_float64 requires an FP64 FMM plan");
  }
  return multipole(node_index);
}

std::span<const double> UniformFmm::local_float64(
    const int node_index) const {
  if (precision_ != StaticPrecision::Float64) {
    throw std::logic_error("local_float64 requires an FP64 FMM plan");
  }
  return local(node_index);
}

std::span<const double> UniformFmm::root_multipole_float64() const {
  return multipole_float64(0);
}

std::span<double> UniformFmm::multipole_for_node(const int node_index) noexcept {
  const std::size_t n = static_cast<std::size_t>(basis_.size());
  return {multipoles_.data() + static_cast<std::size_t>(node_index) * n, n};
}

std::span<const double> UniformFmm::multipole_for_node(
    const int node_index) const noexcept {
  const std::size_t n = static_cast<std::size_t>(basis_.size());
  return {multipoles_.data() + static_cast<std::size_t>(node_index) * n, n};
}

std::span<double> UniformFmm::local_for_node(const int node_index) noexcept {
  const std::size_t n = static_cast<std::size_t>(basis_.size());
  return {locals_.data() + static_cast<std::size_t>(node_index) * n, n};
}

std::span<const double> UniformFmm::local_for_node(
    const int node_index) const noexcept {
  const std::size_t n = static_cast<std::size_t>(basis_.size());
  return {locals_.data() + static_cast<std::size_t>(node_index) * n, n};
}

std::span<float> UniformFmm::multipole_float_for_node(
    const int node_index) noexcept {
  const std::size_t n = static_cast<std::size_t>(basis_.size());
  return {multipoles_float_.data() + static_cast<std::size_t>(node_index) * n,
          n};
}

std::span<const float> UniformFmm::multipole_float_for_node(
    const int node_index) const noexcept {
  const std::size_t n = static_cast<std::size_t>(basis_.size());
  return {multipoles_float_.data() + static_cast<std::size_t>(node_index) * n,
          n};
}

std::span<float> UniformFmm::local_float_for_node(
    const int node_index) noexcept {
  const std::size_t n = static_cast<std::size_t>(basis_.size());
  return {locals_float_.data() + static_cast<std::size_t>(node_index) * n, n};
}

std::span<const float> UniformFmm::local_float_for_node(
    const int node_index) const noexcept {
  const std::size_t n = static_cast<std::size_t>(basis_.size());
  return {locals_float_.data() + static_cast<std::size_t>(node_index) * n, n};
}

const EvaluationTimings &UniformFmm::last_timings() const {
  return last_timings_;
}

const EvaluationTimings &UniformFmm::aggregate_timings() const {
  return aggregate_timings_;
}

void UniformFmm::reset_timings() { aggregate_timings_ = {}; }

UniformFmm::~UniformFmm() = default;
UniformFmm::UniformFmm(UniformFmm &&) noexcept = default;
UniformFmm &UniformFmm::operator=(UniformFmm &&) noexcept = default;

bool one_mkl_available() noexcept {
#ifdef CDFMM_USE_MKL
  return true;
#else
  return false;
#endif
}

bool cuda_available() noexcept { return cuda_runtime_available(); }

std::string cuda_device_description() { return cuda_runtime_description(); }

} // namespace cdfmm
