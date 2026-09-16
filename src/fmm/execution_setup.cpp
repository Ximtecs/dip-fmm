// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/uniform_fmm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

#include "cdfmm/backend/cuda/m2l.hpp"
#include "cdfmm/backend/cuda/p2p.hpp"
#include "cdfmm/operators/operators.hpp"
#include "cdfmm/plan/p2p/leaf.hpp"
#include "cdfmm/plan/precision.hpp"
#include "cdfmm/plan/static_plan.hpp"

#include "backend/cuda/fmm/internal.hpp"
#include "cache/internal.hpp"
#include "fmm/internal.hpp"

namespace cdfmm {

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(const Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

// Dense leaf rectangles of the canonical list-1 topology, in canonical order.
std::vector<StaticP2PLeafPair> leaf_pairs_from_topology(
    const StaticFmmTopology &topology) {
  std::vector<StaticP2PLeafPair> leaf_pairs;
  leaf_pairs.reserve(topology.p2p_leaf_records.size());
  for (const StaticP2PLeafRecord &record : topology.p2p_leaf_records) {
    leaf_pairs.push_back({static_cast<int>(record.target_begin),
                          static_cast<int>(record.target_count),
                          static_cast<int>(record.source_begin),
                          static_cast<int>(record.source_count)});
  }
  return leaf_pairs;
}

// FP32 plans retain no FP64 canonical operator; the leaf builder is FP64-only,
// so widen the stored values without changing them.
StaticP2POperator promote_p2p_operator(const FloatStaticP2POperator &source) {
  StaticP2POperator promoted;
  promoted.source_count = source.source_count;
  promoted.target_count = source.target_count;
  promoted.row_offsets = source.row_offsets;
  promoted.blocks.reserve(source.blocks.size());
  for (const FloatStaticDipoleBlock &block : source.blocks) {
    promoted.blocks.push_back({block.target, block.source, block.px, block.py,
                               block.pz, block.xx, block.xy, block.xz,
                               block.yy, block.yz, block.zz,
                               block.skip_for_identity});
  }
  return promoted;
}

// The policy may select the dictionary before its plan exists; if the plan
// could not be derived, the plan falls back to the General defaults.
cuda_policy::CudaExecutionPolicy effective_cuda_policy(
    const cuda_policy::CudaExecutionPolicyInputs &inputs,
    const cuda_policy::CudaExecutionPolicy &policy,
    const bool dictionary_plan_available) {
  if (policy.p2p_packing != cuda_policy::CudaP2PPacking::SignedDictionary ||
      dictionary_plan_available) {
    return policy;
  }
  cuda_policy::CudaExecutionPolicyInputs fallback = inputs;
  fallback.explicit_reduced_symmetry = false;
  fallback.spatial_layout = SpatialLayout::General;
  return cuda_policy::resolve_cuda_execution_policy(fallback);
}

void validate_model_options(const UniformFmmOptions& options)
{
  switch (options.near_field_source_model) {
  case SourceModel::PointDipole:
  case SourceModel::ExactGeometry:
    break;
  default:
    throw std::invalid_argument("unsupported near-field source model");
  }
  switch (options.near_field_target_model) {
  case TargetModel::Point:
  case TargetModel::ExactGeometry:
    break;
  default:
    throw std::invalid_argument("unsupported near-field target model");
  }
  switch (options.far_field_source_model) {
  case SourceModel::PointDipole:
  case SourceModel::ExactGeometry:
    break;
  default:
    throw std::invalid_argument("unsupported far-field source model");
  }
  switch (options.far_field_target_model) {
  case TargetModel::Point:
  case TargetModel::ExactGeometry:
    break;
  default:
    throw std::invalid_argument("unsupported far-field target model");
  }
}

} // namespace

UniformFmm::MklM2LPlanOwner::MklM2LPlanOwner(const StaticM2LPlan& plan)
    : executor_(plan) {}

UniformFmm::MklM2LPlanOwner::MklM2LPlanOwner(
    const FloatStaticM2LPlan& plan)
    : executor_(plan) {}

detail::mkl::M2LApplyTimings UniformFmm::MklM2LPlanOwner::apply(
    const StaticM2LPlan& plan, const int level,
    const std::span<const double> multipoles,
    const std::span<double> locals) {
  return executor_.apply(plan, level, multipoles, locals);
}

detail::mkl::M2LApplyTimings UniformFmm::MklM2LPlanOwner::apply(
    const FloatStaticM2LPlan& plan, const int level,
    const std::span<const float> multipoles,
    const std::span<float> locals) {
  return executor_.apply(plan, level, multipoles, locals);
}

detail::mkl::M2LStorageStatistics
UniformFmm::MklM2LPlanOwner::statistics() const noexcept {
  return executor_.statistics();
}

UniformFmm::CudaM2LPlanOwner::CudaM2LPlanOwner(
    std::unique_ptr<CudaM2LPlan> value)
    : plan(std::move(value)) {}

UniformFmm::CudaP2PPlanOwner::CudaP2PPlanOwner(
    std::unique_ptr<CudaP2PPlan> value)
    : plan(std::move(value)) {}

UniformFmm::CudaFullPlanOwner::CudaFullPlanOwner(
    std::unique_ptr<CudaFullPlan> value)
    : plan(std::move(value)) {}

void UniformFmm::initialise_execution(const UniformFmmOptions& options) {
  validate_model_options(options);
  if (options.expansion_order < 0) {
    throw std::invalid_argument(
        "UniformFmmOptions.expansion_order must be >= 0");
  }
  if (expansion_basis_ != ExpansionBasis::Cartesian &&
      expansion_basis_ != ExpansionBasis::Spherical) {
    throw std::invalid_argument("unsupported expansion basis");
  }
  if (expansion_basis_ == ExpansionBasis::Spherical &&
      (options.backend == ExecutionBackend::CpuReference ||
       options.m2l_backend == M2LBackend::Reference)) {
    throw std::invalid_argument(
        "spherical expansions require a static M2L execution backend");
  }
  if (expansion_basis_ == ExpansionBasis::Spherical &&
      spherical_m2l_backend_ != SphericalM2LBackend::StaticDense) {
    throw std::invalid_argument("unsupported spherical M2L backend");
  }
  if (periodic_.enabled &&
      (options.backend == ExecutionBackend::CpuReference ||
       options.m2l_backend == M2LBackend::Reference)) {
    throw std::invalid_argument(
        "periodic evaluation requires a static execution backend");
  }
  initialise_source_geometry(options);
  initialise_target_geometry(options);

  const bool effective_finite_source =
      source_geometry_ != SourceGeometry::PointDipole &&
      (near_field_source_model_ == SourceModel::ExactGeometry ||
       far_field_source_model_ == SourceModel::ExactGeometry);
  const bool effective_finite_target =
      target_geometry_ != TargetGeometry::Point &&
      (near_field_target_model_ == TargetModel::ExactGeometry ||
       far_field_target_model_ == TargetModel::ExactGeometry);
  const bool reference_requested =
      options.backend == ExecutionBackend::CpuReference ||
      (options.backend == ExecutionBackend::Auto &&
       options.m2l_backend == M2LBackend::Reference);
  if (reference_requested &&
      (effective_finite_source || effective_finite_target)) {
    throw std::invalid_argument(
        "CpuReference cannot execute selected exact finite source/target stages");
  }

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
      topology_->nodes.size() * static_cast<std::size_t>(coefficient_count());
  const std::size_t source_count = topology_->sorted_source_positions.size();
  const std::size_t target_count = topology_->sorted_target_positions.size();
  if (precision_ == StaticPrecision::Float32) {
    multipoles_float_.assign(coefficient_values, 0.0F);
    locals_float_.assign(coefficient_values, 0.0F);
    sorted_dipole_moments_float_.resize(source_count);
    sorted_results_float_.resize(target_count);
    near_fields_float_.resize(target_count);
  } else {
    multipoles_.assign(coefficient_values, 0.0);
    locals_.assign(coefficient_values, 0.0);
    sorted_dipole_moments_.resize(source_count);
    sorted_results_.resize(target_count);
    near_fields_.resize(target_count);
  }
  sorted_self_indices_.resize(target_count, -1);
  initialise_p2p_policy(options);
  {
    // Plan preparation supplies the geometry/option facts cache identity is
    // computed from; `compute_cache_identity` never reaches into `this`.
    const detail::cache::CacheIdentityInputs inputs{
        expansion_basis_,
        precision_,
        expansion_order(),
        source_geometry_,
        target_geometry_,
        near_field_source_model_,
        near_field_target_model_,
        far_field_source_model_,
        far_field_target_model_,
        use_reduced_symmetry_p2p_,
        periodic_,
        *tree_,
        sorted_source_sizes_,
        sorted_target_sizes_,
        sorted_source_tetrahedra_,
        sorted_target_tetrahedra_,
        fixed_target_source_indices_};
    const detail::cache::CacheIdentity identity =
        detail::cache::compute_cache_identity(supplied_topology_,
                                              options.enable_cache, inputs,
                                              static_plan_statistics_);
    cache_enabled_ = identity.enabled;
    cache_directory_ = identity.directory;
    universal_cache_key_ = identity.universal_key;
    periodic_cache_key_ = identity.periodic_key;
    geometry_cache_key_ = identity.geometry_key;
    geometry_hash_digest_ = identity.geometry_hash_digest;
  }
  if (m2l_backend_ == M2LBackend::Static ||
      precision_ == StaticPrecision::Float32) {
    build_static_plan();
  }
  static_plan_statistics_.state_bytes =
      precision_ == StaticPrecision::Float32
      ? (multipoles_float_.capacity() + locals_float_.capacity()) *
                sizeof(float) +
            sorted_dipole_moments_float_.capacity() * sizeof(FloatVec3) +
            sorted_results_float_.capacity() * sizeof(FloatPotentialField) +
            near_fields_float_.capacity() * sizeof(FloatVec3)
      : (multipoles_.capacity() + locals_.capacity()) * sizeof(double) +
            sorted_dipole_moments_.capacity() * sizeof(Vec3) +
            sorted_results_.capacity() * sizeof(PotentialField) +
            near_fields_.capacity() * sizeof(Vec3);
  static_plan_statistics_.state_bytes +=
      (sorted_self_indices_.capacity() + fixed_sorted_self_indices_.capacity() +
       (fixed_target_source_indices_.has_value()
            ? fixed_target_source_indices_->capacity()
            : 0)) *
          sizeof(int) +
      sorted_source_sizes_.capacity() * sizeof(CuboidSize);
  static_plan_statistics_.state_bytes +=
      sorted_source_tetrahedra_.capacity() * sizeof(Tetrahedron);
  static_plan_statistics_.state_bytes +=
      sorted_target_sizes_.capacity() * sizeof(CuboidSize);
  static_plan_statistics_.state_bytes +=
      sorted_target_tetrahedra_.capacity() * sizeof(Tetrahedron);
  const std::size_t coefficient_scalar_bytes =
      precision_ == StaticPrecision::Float32 ? sizeof(float) : sizeof(double);
  static_plan_statistics_.multipole_state_bytes =
      coefficient_values * coefficient_scalar_bytes;
  static_plan_statistics_.local_state_bytes =
      coefficient_values * coefficient_scalar_bytes;
  static_plan_statistics_.other_state_bytes =
      static_plan_statistics_.state_bytes -
      static_plan_statistics_.multipole_state_bytes -
      static_plan_statistics_.local_state_bytes;
  const auto cuda_setup_start = Clock::now();
  const bool creates_cuda_plan = backend_ == ExecutionBackend::CudaM2LP2P ||
      backend_ == ExecutionBackend::CudaFull;
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
  if (creates_cuda_plan) {
    static_plan_statistics_.cuda_upload.add(
        elapsed_seconds(cuda_setup_start));
  }
}

void UniformFmm::initialise_p2p_policy(const UniformFmmOptions &options) {
  cuda_p2p_bsr_max_bytes_ = options.cuda_p2p_bsr_max_bytes;
  spatial_layout_ = options.spatial_layout;
  use_reduced_symmetry_p2p_ = options.use_reduced_symmetry_p2p;
  cuda_dictionary_target_owned_ =
      options.cuda_dictionary_target_owned;
  cuda_dictionary_power2_microtiles_ =
      options.cuda_dictionary_power2_microtiles;
  signed_p2p_target_tile_size_ = options.signed_p2p_target_tile_size;
  if (signed_p2p_target_tile_size_ <= 0 ||
      signed_p2p_target_tile_size_ > 128) {
    throw std::invalid_argument(
        "signed_p2p_target_tile_size must be in [1, 128]");
  }
  if (!options.fixed_target_source_indices.has_value()) {
    resolve_cuda_execution_policy();
    return;
  }

  const std::vector<int> &identities =
      options.fixed_target_source_indices.value();
  const std::size_t target_count = topology_->sorted_target_positions.size();
  const std::size_t source_count = topology_->sorted_source_positions.size();
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

  // Finite-geometry source self fields are physical; identity maps remove
  // self interactions only when the near-field source is a point dipole.
  const bool effective_point_source =
      source_geometry_ == SourceGeometry::PointDipole ||
      near_field_source_model_ == SourceModel::PointDipole;
  if (!effective_point_source) {
    resolve_cuda_execution_policy();
    return;
  }

  fixed_target_source_indices_ = identities;
  prepare_self_indices(identities);
  fixed_sorted_self_indices_ = sorted_self_indices_;
  resolve_cuda_execution_policy();
}

void UniformFmm::resolve_cuda_execution_policy() {
  // Every input is a constructed-topology or option fact, so the policy is
  // deterministic and available before any packing is derived.
  cuda_policy::CudaExecutionPolicyInputs inputs;
  inputs.precision = precision_;
  inputs.spatial_layout = spatial_layout_;
  inputs.cuda_backend = backend_ == ExecutionBackend::CudaM2LP2P ||
                        backend_ == ExecutionBackend::CudaFull;
  inputs.effective_point_source =
      source_geometry_ == SourceGeometry::PointDipole ||
      near_field_source_model_ == SourceModel::PointDipole;
  inputs.periodic = periodic_.enabled;
  inputs.fixed_identity_available = fixed_target_source_indices_.has_value();
  inputs.explicit_reduced_symmetry = use_reduced_symmetry_p2p_;
  inputs.explicit_dictionary_target_owned = cuda_dictionary_target_owned_;
  inputs.explicit_dictionary_power2_microtiles =
      cuda_dictionary_power2_microtiles_;
  inputs.source_count = topology_->sorted_source_positions.size();
  inputs.target_count = topology_->sorted_target_positions.size();
  inputs.expansion_order = expansion_order();
  inputs.coefficient_count = coefficient_count();
  inputs.tree_depth = topology_->maximum_level;
  inputs.occupied_target_leaf_count = topology_->target_leaves.size();
  inputs.mean_leaf_occupancy =
      inputs.occupied_target_leaf_count == 0
          ? 0.0
          : static_cast<double>(inputs.target_count) /
                static_cast<double>(inputs.occupied_target_leaf_count);
  std::size_t pairs = 0;
  for (const StaticP2PLeafRecord &record : topology_->p2p_leaf_records) {
    pairs += record.target_count * record.source_count;
  }
  inputs.p2p_pair_count = pairs;
  inputs.m2l_translation_count = topology_->m2l_interactions.size();
  // BSR(3) stores nine values and one index per pair plus row metadata.
  inputs.bsr_estimate_bytes =
      pairs * (9 * (precision_ == StaticPrecision::Float32 ? sizeof(float)
                                                            : sizeof(double)) +
               sizeof(int)) +
      (inputs.target_count * 2 + 1) * sizeof(int);
  inputs.bsr_budget_bytes = cuda_p2p_bsr_max_bytes_;
  cuda_policy_ = std::make_unique<CudaExecutionPolicyOwner>();
  cuda_policy_->inputs = inputs;
  cuda_policy_->policy = cuda_policy::resolve_cuda_execution_policy(inputs);
}

void UniformFmm::build_reduced_symmetry_p2p_packing() {
  // The dictionary is derived when the user asked for it or when the CUDA
  // policy selected it from the regular-grid layout hint.
  const bool dictionary_selected =
      cuda_policy_ && cuda_policy_->policy.p2p_packing ==
                          cuda_policy::CudaP2PPacking::SignedDictionary;
  if (!dictionary_selected || periodic_.enabled) {
    return;
  }
  // The branch-free point-dipole executor encodes fixed self pairs as the
  // zero variant.  Dynamic identity maps retain the production SoA path.
  const bool effective_point_source =
      source_geometry_ == SourceGeometry::PointDipole ||
      near_field_source_model_ == SourceModel::PointDipole;
  if (effective_point_source && !fixed_target_source_indices_.has_value()) {
    p2p_tensor_dictionary_plan_.reset();
    return;
  }

  std::vector<StaticP2PLeafPair> leaf_pairs;
  leaf_pairs.reserve(topology_->p2p_leaf_records.size());
  for (const StaticP2PLeafRecord& record : topology_->p2p_leaf_records) {
    leaf_pairs.push_back({static_cast<int>(record.target_begin),
                          static_cast<int>(record.target_count),
                          static_cast<int>(record.source_begin),
                          static_cast<int>(record.source_count)});
  }

  if (geometry_cache_loaded_direct_float_) {
    StaticP2POperator promoted;
    promoted.source_count = p2p_operator_float_.source_count;
    promoted.target_count = p2p_operator_float_.target_count;
    promoted.row_offsets = p2p_operator_float_.row_offsets;
    promoted.blocks.reserve(p2p_operator_float_.blocks.size());
    for (const FloatStaticDipoleBlock &block : p2p_operator_float_.blocks) {
      promoted.blocks.push_back({block.target, block.source, block.px, block.py,
                                 block.pz, block.xx, block.xy, block.xz,
                                 block.yy, block.yz, block.zz,
                                 block.skip_for_identity});
    }
    p2p_tensor_dictionary_plan_ = build_static_p2p_signed_tensor_dictionary_plan(
        promoted, leaf_pairs, fixed_sorted_self_indices_,
        signed_p2p_target_tile_size_);
    return;
  }
  p2p_tensor_dictionary_plan_ = build_static_p2p_signed_tensor_dictionary_plan(
      p2p_operator_, leaf_pairs, fixed_sorted_self_indices_,
      signed_p2p_target_tile_size_);
}

void UniformFmm::initialise_source_geometry(const UniformFmmOptions &options) {
  source_geometry_ = options.source_geometry;
  near_field_source_model_ = options.near_field_source_model;
  far_field_source_model_ = options.far_field_source_model;
  const std::size_t count = topology_->sorted_source_positions.size();
  if (source_geometry_ == SourceGeometry::PointDipole) {
    if (!options.source_sizes.empty() || !options.source_tetrahedra.empty()) {
      throw std::invalid_argument(
          "point-dipole sources do not accept finite geometry records");
    }
    return;
  }
  if (source_geometry_ != SourceGeometry::RectangularPrism) {
    if (!options.source_sizes.empty()) {
      throw std::invalid_argument(
          "non-prism sources do not accept rectangular-prism sizes");
    }
    if (source_geometry_ == SourceGeometry::Tetrahedron) {
      if (options.source_tetrahedra.size() != 1 &&
          options.source_tetrahedra.size() != count) {
        throw std::invalid_argument(
            "tetrahedron geometries must contain one or one per source");
      }
      if (options.source_tetrahedra.size() == 1) {
        sorted_source_tetrahedra_ = options.source_tetrahedra;
      } else {
        sorted_source_tetrahedra_.resize(count);
        for (std::size_t sorted = 0; sorted < count; ++sorted) {
          sorted_source_tetrahedra_[sorted] =
              options.source_tetrahedra[topology_->source_permutation[sorted]];
        }
      }
      for (const Tetrahedron& tetrahedron : sorted_source_tetrahedra_) {
        static_cast<void>(tetrahedron_volume(tetrahedron));
      }
      use_cuboid_p2m_ = false;
      return;
    }
    throw std::invalid_argument("unsupported source geometry");
  }
  if (!options.source_tetrahedra.empty()) {
    throw std::invalid_argument(
        "rectangular-prism sources do not accept tetrahedron records");
  }
  if (options.source_sizes.size() != 1 &&
      options.source_sizes.size() != count) {
    throw std::invalid_argument(
        "cuboid sizes must contain one or one per source");
  }
  if (options.source_sizes.size() == 1) {
    sorted_source_sizes_ = options.source_sizes;
    use_cuboid_p2m_ = far_field_source_model_ == SourceModel::ExactGeometry;
    return;
  }
  sorted_source_sizes_.resize(count);
  const auto permutation = std::span<const int>(topology_->source_permutation);
  for (std::size_t sorted = 0; sorted < count; ++sorted) {
    sorted_source_sizes_[sorted] = options.source_sizes[permutation[sorted]];
  }
  use_cuboid_p2m_ = far_field_source_model_ == SourceModel::ExactGeometry;
}

void UniformFmm::initialise_target_geometry(const UniformFmmOptions &options) {
  target_geometry_ = options.target_geometry;
  near_field_target_model_ = options.near_field_target_model;
  far_field_target_model_ = options.far_field_target_model;
  const std::size_t count = topology_->sorted_target_positions.size();
  if (target_geometry_ == TargetGeometry::Point) {
    if (!options.target_sizes.empty() || !options.target_tetrahedra.empty()) {
      throw std::invalid_argument("point targets do not accept finite geometry records");
    }
    return;
  }
  if (target_geometry_ != TargetGeometry::RectangularPrism) {
    if (!options.target_sizes.empty()) {
      throw std::invalid_argument(
          "non-prism targets do not accept rectangular-prism sizes");
    }
    if (target_geometry_ == TargetGeometry::Tetrahedron) {
      if (options.target_tetrahedra.size() != 1 &&
          options.target_tetrahedra.size() != count) {
        throw std::invalid_argument(
            "tetrahedron geometries must contain one or one per target");
      }
      if (options.target_tetrahedra.size() == 1) {
        sorted_target_tetrahedra_ = options.target_tetrahedra;
      } else {
        sorted_target_tetrahedra_.resize(count);
        for (std::size_t sorted = 0; sorted < count; ++sorted) {
          sorted_target_tetrahedra_[sorted] =
              options.target_tetrahedra[topology_->target_permutation[sorted]];
        }
      }
      for (const Tetrahedron& tetrahedron : sorted_target_tetrahedra_) {
        static_cast<void>(tetrahedron_volume(tetrahedron));
      }
      return;
    }
    throw std::invalid_argument("unsupported target geometry");
  }
  if (!options.target_tetrahedra.empty()) {
    throw std::invalid_argument(
        "rectangular-prism targets do not accept tetrahedron records");
  }
  if (options.target_sizes.size() != 1 &&
      options.target_sizes.size() != count) {
    throw std::invalid_argument(
        "cuboid target sizes must contain one or one per target");
  }
  if (options.target_sizes.size() == 1) {
    sorted_target_sizes_ = options.target_sizes;
    use_cuboid_l2p_ = far_field_target_model_ == TargetModel::ExactGeometry;
    return;
  }
  sorted_target_sizes_.resize(count);
  const auto permutation = std::span<const int>(topology_->target_permutation);
  for (std::size_t sorted = 0; sorted < count; ++sorted) {
    sorted_target_sizes_[sorted] = options.target_sizes[permutation[sorted]];
  }
  use_cuboid_l2p_ = far_field_target_model_ == TargetModel::ExactGeometry;
}

void UniformFmm::build_cuda_p2p_plan() {
  using cuda_policy::CudaDictionaryExecutor;
  using cuda_policy::CudaP2PPacking;
  const bool dictionary_plan_available =
      precision_ == StaticPrecision::Float32
          ? p2p_tensor_dictionary_plan_float_.has_value()
          : p2p_tensor_dictionary_plan_.has_value();
  const cuda_policy::CudaExecutionPolicy policy = effective_cuda_policy(
      cuda_policy_->inputs, cuda_policy_->policy, dictionary_plan_available);
  cuda_policy_->policy = policy;
  const bool target_owned =
      policy.dictionary_executor == CudaDictionaryExecutor::TargetOwned;
  const bool power2_microtiles =
      policy.dictionary_executor == CudaDictionaryExecutor::PowerOfTwoMicrotiles;
  const std::span<const int> fixed_identities =
      fixed_target_source_indices_.has_value()
          ? std::span<const int>(fixed_sorted_self_indices_)
          : std::span<const int>{};
  if (policy.p2p_packing == CudaP2PPacking::SignedDictionary) {
    if (precision_ == StaticPrecision::Float32) {
      cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
          std::make_unique<CudaP2PPlan>(*p2p_tensor_dictionary_plan_float_,
                                        target_owned, power2_microtiles));
    } else {
      cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
          std::make_unique<CudaP2PPlan>(*p2p_tensor_dictionary_plan_,
                                        target_owned, power2_microtiles));
    }
    p2p_execution_packing_ = P2PExecutionPacking::TensorDictionary;
    return;
  }
  if (policy.p2p_packing == CudaP2PPacking::LeafBlock) {
    if (precision_ == StaticPrecision::Float32) {
      cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
          std::make_unique<CudaP2PPlan>(build_cuda_leaf_plan_float(),
                                        fixed_identities));
    } else {
      cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
          std::make_unique<CudaP2PPlan>(build_cuda_leaf_plan(),
                                        fixed_identities));
    }
    p2p_execution_packing_ = P2PExecutionPacking::LeafBlock;
    return;
  }
  if (policy.p2p_packing == CudaP2PPacking::Bsr3) {
    if (precision_ == StaticPrecision::Float32) {
      cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
          std::make_unique<CudaP2PPlan>(p2p_bsr_plan_float_));
    } else {
      const StaticP2PBsrPlan bsr =
          build_static_p2p_bsr_plan(p2p_operator_, fixed_identities);
      cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
          std::make_unique<CudaP2PPlan>(bsr));
    }
    p2p_execution_packing_ = P2PExecutionPacking::CudaBsr3;
    return;
  }
  if (precision_ == StaticPrecision::Float32) {
    cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
        std::make_unique<CudaP2PPlan>(p2p_operator_float_, fixed_identities));
  } else {
    cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
        std::make_unique<CudaP2PPlan>(p2p_operator_, fixed_identities));
  }
  p2p_execution_packing_ = P2PExecutionPacking::CanonicalAos;
}

StaticP2PLeafPlan UniformFmm::build_cuda_leaf_plan() const {
  return build_static_p2p_leaf_plan(p2p_operator_,
                                    leaf_pairs_from_topology(*topology_));
}

FloatStaticP2PLeafPlan UniformFmm::build_cuda_leaf_plan_float() const {
  // An FP32 plan keeps no FP64 operator after quantisation, so the FP64-only
  // leaf builder runs on the widened FP32 values; quantising the result back
  // reproduces the stored FP32 tensors exactly.
  return quantise_static_p2p_leaf_plan(build_static_p2p_leaf_plan(
      promote_p2p_operator(p2p_operator_float_),
      leaf_pairs_from_topology(*topology_)));
}

void UniformFmm::build_backend_packing() {
  if (backend_ != ExecutionBackend::CudaFull) {
    build_cpu_far_field_packing();
  }
  const bool cuda_executes_m2l = backend_ == ExecutionBackend::CudaM2LP2P ||
      backend_ == ExecutionBackend::CudaFull;
  if (static_matrix_backend_ != StaticMatrixBackend::OneMkl ||
      cuda_executes_m2l) {
    return;
  }
  const auto start = Clock::now();
  if (precision_ == StaticPrecision::Float32) {
    mkl_m2l_plan_ = std::make_unique<MklM2LPlanOwner>(m2l_plan_float_);
  } else {
    mkl_m2l_plan_ = std::make_unique<MklM2LPlanOwner>(m2l_plan_);
  }
  const detail::mkl::M2LStorageStatistics storage =
      mkl_m2l_plan_->statistics();
  static_plan_statistics_.interaction_bytes += storage.metadata_bytes;
  static_plan_statistics_.m2l_interaction_bytes += storage.metadata_bytes;
  static_plan_statistics_.scratch_bytes += storage.scratch_bytes;
  static_plan_statistics_.backend_packing.add(elapsed_seconds(start));
}

void UniformFmm::build_cpu_far_field_packing() {
  // The canonical sparse P2M/L2P maps are persisted and consumed by CUDA
  // plan construction; the CPU hierarchy executes a dense, level-scaled
  // packing derived from them once here.  Every backend except CudaFull runs
  // P2M/M2M/L2L/L2P on the CPU, and after packing nothing else reads the
  // per-source/per-target canonical maps, so they are released to keep the
  // resident operator footprint at the packed size.
  const auto start = Clock::now();
  const int n = coefficient_count();
  std::vector<int> degrees(static_cast<std::size_t>(n));
  for (int coefficient = 0; coefficient < n; ++coefficient) {
    degrees[static_cast<std::size_t>(coefficient)] =
        coefficient_degree(coefficient);
  }
  const int level_count = topology_->maximum_level;
  const std::size_t source_count = topology_->sorted_source_positions.size();
  auto owner = std::make_unique<CpuFarFieldPackingOwner>();
  std::size_t canonical_p2m_bytes = 0;
  std::size_t canonical_l2p_bytes = 0;
  std::size_t packed_p2m_bytes = 0;
  std::size_t packed_l2p_bytes = 0;
  std::size_t packed_translation_bytes = 0;
  if (precision_ == StaticPrecision::Float32) {
    auto &packing = owner->fp32;
    packing.p2m = detail::cpu::pack_p2m(
        std::span<const FloatP2MPlan>(p2m_plans_float_), source_count, n);
    packing.m2m = detail::cpu::pack_translation_bank(
        std::span<const FloatStaticCoefficientOperator>(m2m_operators_float_),
        degrees, level_count);
    packing.l2l = detail::cpu::pack_translation_bank(
        std::span<const FloatStaticCoefficientOperator>(l2l_operators_float_),
        degrees, level_count);
    packing.l2p = detail::cpu::pack_l2p(
        std::span<const FloatStaticL2PEvaluator>(l2p_evaluators_float_), n);
    for (const FloatP2MPlan &plan : p2m_plans_float_) {
      canonical_p2m_bytes +=
          plan.operator_map.entries.size() * sizeof(FloatStaticOperatorEntry);
    }
    canonical_l2p_bytes = l2p_evaluators_float_.size() * 4 *
                          static_cast<std::size_t>(n) * sizeof(float);
    packed_p2m_bytes = packing.p2m.memory_bytes();
    packed_l2p_bytes = packing.l2p.memory_bytes();
    packed_translation_bytes =
        packing.m2m.memory_bytes() + packing.l2l.memory_bytes();
    p2m_plans_float_.clear();
    p2m_plans_float_.shrink_to_fit();
    l2p_evaluators_float_.clear();
    l2p_evaluators_float_.shrink_to_fit();
  } else {
    auto &packing = owner->fp64;
    packing.p2m = detail::cpu::pack_p2m(std::span<const P2MPlan>(p2m_plans_),
                                        source_count, n);
    packing.m2m = detail::cpu::pack_translation_bank(
        std::span<const StaticCoefficientOperator>(m2m_operators_), degrees,
        level_count);
    packing.l2l = detail::cpu::pack_translation_bank(
        std::span<const StaticCoefficientOperator>(l2l_operators_), degrees,
        level_count);
    packing.l2p = detail::cpu::pack_l2p(
        std::span<const StaticL2PEvaluator>(l2p_evaluators_), n);
    for (const P2MPlan &plan : p2m_plans_) {
      canonical_p2m_bytes +=
          plan.operator_map.entries.size() * sizeof(StaticOperatorEntry);
    }
    canonical_l2p_bytes = l2p_evaluators_.size() * 4 *
                          static_cast<std::size_t>(n) * sizeof(double);
    packed_p2m_bytes = packing.p2m.memory_bytes();
    packed_l2p_bytes = packing.l2p.memory_bytes();
    packed_translation_bytes =
        packing.m2m.memory_bytes() + packing.l2l.memory_bytes();
    p2m_plans_.clear();
    p2m_plans_.shrink_to_fit();
    l2p_evaluators_.clear();
    l2p_evaluators_.shrink_to_fit();
  }
  // The portable M2L executor applies the canonical plan through a block
  // schedule sorted by transfer class; oneMKL and CUDA M2L keep their own.
  // The class-sorted schedule pays off when the transfer matrices do not fit
  // the per-core L2 (2 MiB on the calibration machine): below about 1 MiB
  // the per-target row kernel already streams them from L2 and the block
  // staging only adds overhead (measured: S FP32, 316 x 25^2 floats).
  const bool portable_m2l = backend_ == ExecutionBackend::CpuStatic &&
      static_matrix_backend_ == StaticMatrixBackend::Portable &&
      m2l_backend_ == M2LBackend::Static;
  constexpr std::size_t schedule_matrix_bytes = std::size_t{1} << 20;
  const std::size_t matrix_bytes = precision_ == StaticPrecision::Float32
      ? m2l_plan_float_.matrices.size() * sizeof(float)
      : m2l_plan_.matrices.size() * sizeof(double);
  if (portable_m2l && matrix_bytes > schedule_matrix_bytes) {
    owner->m2l_schedule = precision_ == StaticPrecision::Float32
        ? detail::cpu::build_m2l_block_schedule(m2l_plan_float_)
        : detail::cpu::build_m2l_block_schedule(m2l_plan_);
    static_plan_statistics_.interaction_bytes +=
        owner->m2l_schedule.memory_bytes();
    static_plan_statistics_.m2l_interaction_bytes +=
        owner->m2l_schedule.memory_bytes();
  }
  cpu_far_field_ = std::move(owner);
  // Report the resident packing instead of the released canonical maps; the
  // eight shared translation operators stay resident and keep their bytes.
  static_plan_statistics_.operator_bytes +=
      packed_p2m_bytes + packed_l2p_bytes + packed_translation_bytes;
  static_plan_statistics_.operator_bytes -=
      std::min(static_plan_statistics_.operator_bytes,
               canonical_p2m_bytes + canonical_l2p_bytes);
  static_plan_statistics_.p2m_operator_bytes = packed_p2m_bytes;
  static_plan_statistics_.l2p_operator_bytes = packed_l2p_bytes;
  static_plan_statistics_.m2m_operator_bytes += packed_translation_bytes / 2;
  static_plan_statistics_.l2l_operator_bytes += packed_translation_bytes / 2;
  static_plan_statistics_.far_field_packing.add(elapsed_seconds(start));
}

void UniformFmm::build_cuda_full_plan() {
  using cuda_policy::CudaDictionaryExecutor;
  using cuda_policy::CudaP2PPacking;
  const bool dictionary_plan_available =
      precision_ == StaticPrecision::Float32
          ? p2p_tensor_dictionary_plan_float_.has_value()
          : p2p_tensor_dictionary_plan_.has_value();
  const cuda_policy::CudaExecutionPolicy policy = effective_cuda_policy(
      cuda_policy_->inputs, cuda_policy_->policy, dictionary_plan_available);
  cuda_policy_->policy = policy;
  const bool target_owned =
      policy.dictionary_executor == CudaDictionaryExecutor::TargetOwned;
  const bool power2_microtiles =
      policy.dictionary_executor == CudaDictionaryExecutor::PowerOfTwoMicrotiles;
  const bool use_leaf = policy.p2p_packing == CudaP2PPacking::LeafBlock;
  const bool use_bsr = policy.p2p_packing == CudaP2PPacking::Bsr3;
  if (precision_ == StaticPrecision::Float32) {
    FloatCudaFullPlanData data;
    data.coefficient_count = coefficient_count();
    data.node_count = static_cast<int>(topology_->nodes.size());
    data.source_count =
        static_cast<int>(topology_->sorted_source_positions.size());
    data.target_count =
        static_cast<int>(topology_->sorted_target_positions.size());
    data.source_permutation = topology_->source_permutation;
    data.target_permutation = topology_->target_permutation;
    const int n = coefficient_count();
    data.coefficient_degrees.reserve(static_cast<std::size_t>(n));
    for (int coefficient = 0; coefficient < n; ++coefficient) {
      data.coefficient_degrees.push_back(coefficient_degree(coefficient));
    }
    const auto &nodes = topology_->nodes;
    for (const FloatP2MPlan &leaf_plan : p2m_plans_float_) {
      const auto &leaf = nodes[static_cast<std::size_t>(leaf_plan.leaf)];
      for (FloatStaticOperatorEntry entry : leaf_plan.operator_map.entries) {
        entry.input += static_cast<int>(leaf_plan.begin) * 3;
        entry.output += leaf.index * n;
        data.p2m.push_back(entry);
      }
    }
    if (topology_->maximum_level > 0) {
      data.m2m.entries_per_matrix =
          static_cast<int>(m2m_operators_float_[0].entries.size());
      data.l2l.entries_per_matrix =
          static_cast<int>(l2l_operators_float_[0].entries.size());
    }
    for (int child_class = 0; child_class < 8; ++child_class) {
      data.m2m.matrices.insert(
          data.m2m.matrices.end(),
          m2m_operators_float_[child_class].entries.begin(),
          m2m_operators_float_[child_class].entries.end());
      data.l2l.matrices.insert(
          data.l2l.matrices.end(),
          l2l_operators_float_[child_class].entries.begin(),
          l2l_operators_float_[child_class].entries.end());
    }
    data.m2m.matrix_count = topology_->maximum_level == 0 ? 0 : 8;
    data.l2l.matrix_count = topology_->maximum_level == 0 ? 0 : 8;
    for (const StaticTranslationEdge& edge : topology_->m2m_edges) {
      if (nodes[static_cast<std::size_t>(edge.source_node)].source_count() != 0) {
        data.m2m.interactions.push_back(
            {edge.source_node, edge.target_node, edge.child_class,
             edge.child_level});
      }
    }
    for (const StaticTranslationEdge& edge : topology_->l2l_edges) {
      if (nodes[static_cast<std::size_t>(edge.target_node)].target_count() != 0) {
        data.l2l.interactions.push_back(
            {edge.source_node, edge.target_node, edge.child_class,
             edge.child_level});
      }
    }
    data.m2l = m2l_plan_float_;
    for (const StaticLeafRange& leaf_range : topology_->target_leaves) {
      const auto &leaf = nodes[static_cast<std::size_t>(leaf_range.node)];
      for (std::size_t target = leaf_range.begin;
           target < leaf_range.begin + leaf_range.count; ++target) {
        for (int component = 0; component < 3; ++component) {
          for (int coefficient = 0; coefficient < n; ++coefficient) {
            const float value =
                l2p_evaluators_float_[target].field[component][coefficient];
            if (value != 0.0F) {
              data.l2p.push_back({static_cast<int>(target) * 3 + component,
                   leaf.index * n + coefficient, value});
            }
          }
        }
      }
    }
    if (fixed_target_source_indices_.has_value()) {
      data.has_fixed_self_indices = true;
      data.fixed_self_indices = fixed_sorted_self_indices_;
    }
    if (policy.p2p_packing == CudaP2PPacking::SignedDictionary) {
      data.use_p2p_dictionary = true;
      data.p2p_dictionary_target_owned = target_owned;
      data.p2p_dictionary_power2_microtiles = power2_microtiles;
      data.p2p_dictionary = std::move(*p2p_tensor_dictionary_plan_float_);
    } else if (use_leaf) {
      data.use_p2p_leaf = true;
      data.p2p_leaf = build_cuda_leaf_plan_float();
    } else if (use_bsr) {
      data.use_p2p_bsr = true;
      data.p2p_bsr = std::move(p2p_bsr_plan_float_);
    } else {
      data.p2p = p2p_operator_float_;
    }
    p2p_execution_packing_ = data.use_p2p_dictionary
        ? P2PExecutionPacking::TensorDictionary
        : (data.use_p2p_bsr
               ? P2PExecutionPacking::CudaBsr3
               : (data.use_p2p_leaf ? P2PExecutionPacking::LeafBlock
                                    : P2PExecutionPacking::CanonicalAos));
    cuda_full_plan_ = std::make_unique<CudaFullPlanOwner>(
        std::make_unique<CudaFullPlan>(data));
    return;
  }

  CudaFullPlanData data;
  data.coefficient_count = coefficient_count();
  data.node_count = static_cast<int>(topology_->nodes.size());
  data.source_count = static_cast<int>(topology_->sorted_source_positions.size());
  data.target_count = static_cast<int>(topology_->sorted_target_positions.size());
  data.source_permutation = topology_->source_permutation;
  data.target_permutation = topology_->target_permutation;
  const int n = coefficient_count();
  data.coefficient_degrees.reserve(static_cast<std::size_t>(n));
  for (int coefficient = 0; coefficient < n; ++coefficient) {
    data.coefficient_degrees.push_back(coefficient_degree(coefficient));
  }
  const auto &nodes = topology_->nodes;

  for (const P2MPlan &leaf_plan : p2m_plans_) {
    const auto &leaf = nodes[static_cast<std::size_t>(leaf_plan.leaf)];
    for (StaticOperatorEntry entry : leaf_plan.operator_map.entries) {
      entry.input += static_cast<int>(leaf_plan.begin) * 3;
      entry.output += leaf.index * n;
      data.p2m.push_back(entry);
    }
  }
  if (topology_->maximum_level > 0) {
    data.m2m.entries_per_matrix =
        static_cast<int>(m2m_operators_[0].entries.size());
    data.l2l.entries_per_matrix =
        static_cast<int>(l2l_operators_[0].entries.size());
  }
  for (int child_class = 0; child_class < 8; ++child_class) {
    data.m2m.matrices.insert(
        data.m2m.matrices.end(),
        m2m_operators_[child_class].entries.begin(),
        m2m_operators_[child_class].entries.end());
    data.l2l.matrices.insert(
        data.l2l.matrices.end(),
        l2l_operators_[child_class].entries.begin(),
        l2l_operators_[child_class].entries.end());
  }
  data.m2m.matrix_count = topology_->maximum_level == 0 ? 0 : 8;
  data.l2l.matrix_count = topology_->maximum_level == 0 ? 0 : 8;
  for (const StaticTranslationEdge& edge : topology_->m2m_edges) {
    if (nodes[static_cast<std::size_t>(edge.source_node)].source_count() != 0) {
      data.m2m.interactions.push_back(
          {edge.source_node, edge.target_node, edge.child_class,
           edge.child_level});
    }
  }
  for (const StaticTranslationEdge& edge : topology_->l2l_edges) {
    if (nodes[static_cast<std::size_t>(edge.target_node)].target_count() != 0) {
      data.l2l.interactions.push_back(
          {edge.source_node, edge.target_node, edge.child_class,
           edge.child_level});
    }
  }
  data.m2l = m2l_plan_;
  for (const StaticLeafRange& leaf_range : topology_->target_leaves) {
    const auto &leaf = nodes[static_cast<std::size_t>(leaf_range.node)];
    for (std::size_t target = leaf_range.begin;
         target < leaf_range.begin + leaf_range.count; ++target) {
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
  if (fixed_target_source_indices_.has_value()) {
    data.has_fixed_self_indices = true;
    data.fixed_self_indices = fixed_sorted_self_indices_;
  }
  if (policy.p2p_packing == CudaP2PPacking::SignedDictionary) {
    data.use_p2p_dictionary = true;
    data.p2p_dictionary_target_owned = target_owned;
    data.p2p_dictionary_power2_microtiles = power2_microtiles;
    data.p2p_dictionary = std::move(*p2p_tensor_dictionary_plan_);
  } else if (use_leaf) {
    data.use_p2p_leaf = true;
    data.p2p_leaf = build_cuda_leaf_plan();
  } else if (use_bsr) {
    const std::span<const int> bsr_identities =
        fixed_target_source_indices_.has_value()
            ? std::span<const int>(fixed_sorted_self_indices_)
            : std::span<const int>{};
    data.p2p_bsr = build_static_p2p_bsr_plan(p2p_operator_, bsr_identities);
    data.use_p2p_bsr = true;
  } else {
    data.p2p = p2p_operator_;
  }
  p2p_execution_packing_ = data.use_p2p_dictionary
      ? P2PExecutionPacking::TensorDictionary
      : (data.use_p2p_bsr
             ? P2PExecutionPacking::CudaBsr3
             : (data.use_p2p_leaf ? P2PExecutionPacking::LeafBlock
                                  : P2PExecutionPacking::CanonicalAos));
  cuda_full_plan_ =
      std::make_unique<CudaFullPlanOwner>(std::make_unique<CudaFullPlan>(data));
}

} // namespace cdfmm
