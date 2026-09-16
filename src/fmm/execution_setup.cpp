// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/uniform_fmm.hpp"

#include <chrono>
#include <cmath>
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

template <typename Operator>
std::size_t estimate_bsr_bytes(const Operator& p2p,
                               const std::size_t scalar_bytes) {
  const std::size_t interactions = p2p.blocks.size();
  const std::size_t tensor_bytes = 9 * interactions * scalar_bytes;
  const std::size_t index_bytes = interactions * sizeof(int);
  // StaticP2PBsrPlan stores one identity entry per target, including -1
  // entries when no self exclusion is requested.
  const std::size_t metadata_bytes =
      (p2p.row_offsets.size() +
       static_cast<std::size_t>(p2p.target_count)) * sizeof(int);
  return tensor_bytes + index_bytes + metadata_bytes;
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
    return;
  }

  fixed_target_source_indices_ = identities;
  prepare_self_indices(identities);
  fixed_sorted_self_indices_ = sorted_self_indices_;
}

void UniformFmm::build_reduced_symmetry_p2p_packing() {
  if (!use_reduced_symmetry_p2p_ || periodic_.enabled) {
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
  // A BSR plan bakes identity suppression into its values and therefore
  // cannot accept a dynamic identity map.  Keep the canonical CUDA plan for
  // effective point sources unless a fixed map was supplied; finite sources
  // retain the BSR fast path because their identity map is intentionally
  // ignored.
  const bool effective_point_source =
      source_geometry_ == SourceGeometry::PointDipole ||
      near_field_source_model_ == SourceModel::PointDipole;
  const bool bsr_identity_compatible =
      !effective_point_source || fixed_target_source_indices_.has_value();
  // Point sources take the dense leaf-block packing: it streams six tensor
  // values per pair instead of nine and keeps every self identity dynamic, so
  // it needs neither a fixed map nor the BSR memory budget. Periodic images
  // repeat leaf pairs and keep the canonical/BSR selection.
  const bool leaf_compatible = effective_point_source && !periodic_.enabled;
  const std::span<const int> fixed_identities =
      fixed_target_source_indices_.has_value()
          ? std::span<const int>(fixed_sorted_self_indices_)
          : std::span<const int>{};
  if (precision_ == StaticPrecision::Float32) {
    if (use_reduced_symmetry_p2p_ &&
        p2p_tensor_dictionary_plan_float_.has_value()) {
      cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
          std::make_unique<CudaP2PPlan>(
              *p2p_tensor_dictionary_plan_float_,
              cuda_dictionary_target_owned_,
              cuda_dictionary_power2_microtiles_));
      p2p_execution_packing_ = P2PExecutionPacking::TensorDictionary;
      return;
    }
    if (leaf_compatible) {
      cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
          std::make_unique<CudaP2PPlan>(build_cuda_leaf_plan_float(),
                                        fixed_identities));
      p2p_execution_packing_ = P2PExecutionPacking::LeafBlock;
      return;
    }
    if (!periodic_.enabled && bsr_identity_compatible &&
        estimate_bsr_bytes(p2p_operator_float_, sizeof(float)) <=
            cuda_p2p_bsr_max_bytes_) {
      cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
          std::make_unique<CudaP2PPlan>(p2p_bsr_plan_float_));
      p2p_execution_packing_ = P2PExecutionPacking::CudaBsr3;
      return;
    }
    cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
        std::make_unique<CudaP2PPlan>(p2p_operator_float_, fixed_identities));
    p2p_execution_packing_ = P2PExecutionPacking::CanonicalAos;
    return;
  }
  if (use_reduced_symmetry_p2p_ &&
      p2p_tensor_dictionary_plan_.has_value()) {
    cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
        std::make_unique<CudaP2PPlan>(
            *p2p_tensor_dictionary_plan_,
            cuda_dictionary_target_owned_,
            cuda_dictionary_power2_microtiles_));
    p2p_execution_packing_ = P2PExecutionPacking::TensorDictionary;
    return;
  }
  if (leaf_compatible) {
    cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
        std::make_unique<CudaP2PPlan>(build_cuda_leaf_plan(),
                                      fixed_identities));
    p2p_execution_packing_ = P2PExecutionPacking::LeafBlock;
    return;
  }
  if (!periodic_.enabled && bsr_identity_compatible &&
      estimate_bsr_bytes(p2p_operator_, sizeof(double)) <=
          cuda_p2p_bsr_max_bytes_) {
    const std::span<const int> bsr_identities =
        fixed_target_source_indices_.has_value()
            ? std::span<const int>(fixed_sorted_self_indices_)
            : std::span<const int>{};
    StaticP2PBsrPlan bsr =
        build_static_p2p_bsr_plan(p2p_operator_, bsr_identities);
    cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
        std::make_unique<CudaP2PPlan>(bsr));
    p2p_execution_packing_ = P2PExecutionPacking::CudaBsr3;
    return;
  }

  cuda_p2p_plan_ = std::make_unique<CudaP2PPlanOwner>(
      std::make_unique<CudaP2PPlan>(p2p_operator_, fixed_identities));
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

void UniformFmm::build_cuda_full_plan() {
  const bool effective_point_source =
      source_geometry_ == SourceGeometry::PointDipole ||
      near_field_source_model_ == SourceModel::PointDipole;
  const bool bsr_identity_compatible =
      !effective_point_source || fixed_target_source_indices_.has_value();
  const bool leaf_compatible = effective_point_source && !periodic_.enabled;
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
    if (use_reduced_symmetry_p2p_ &&
        p2p_tensor_dictionary_plan_float_.has_value()) {
      data.use_p2p_dictionary = true;
      data.p2p_dictionary_target_owned = cuda_dictionary_target_owned_;
      data.p2p_dictionary_power2_microtiles =
          cuda_dictionary_power2_microtiles_;
      data.p2p_dictionary =
          std::move(*p2p_tensor_dictionary_plan_float_);
    } else {
      if (fixed_target_source_indices_.has_value()) {
        data.has_fixed_self_indices = true;
        data.fixed_self_indices = fixed_sorted_self_indices_;
      }
      if (leaf_compatible) {
        data.use_p2p_leaf = true;
        data.p2p_leaf = build_cuda_leaf_plan_float();
      } else if (!periodic_.enabled && bsr_identity_compatible &&
                 estimate_bsr_bytes(p2p_operator_float_, sizeof(float)) <=
                     cuda_p2p_bsr_max_bytes_) {
        data.use_p2p_bsr = true;
      }
    }
    if (data.use_p2p_dictionary) {
      if (fixed_target_source_indices_.has_value()) {
        data.has_fixed_self_indices = true;
        data.fixed_self_indices = fixed_sorted_self_indices_;
      }
    } else if (data.use_p2p_bsr) {
      data.p2p_bsr = std::move(p2p_bsr_plan_float_);
    } else if (!data.use_p2p_leaf) {
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
  if (use_reduced_symmetry_p2p_ &&
      p2p_tensor_dictionary_plan_.has_value()) {
    data.use_p2p_dictionary = true;
    data.p2p_dictionary_target_owned = cuda_dictionary_target_owned_;
    data.p2p_dictionary_power2_microtiles =
        cuda_dictionary_power2_microtiles_;
    data.p2p_dictionary = std::move(*p2p_tensor_dictionary_plan_);
    if (fixed_target_source_indices_.has_value()) {
      data.has_fixed_self_indices = true;
      data.fixed_self_indices = fixed_sorted_self_indices_;
    }
  } else {
    if (fixed_target_source_indices_.has_value()) {
      data.has_fixed_self_indices = true;
      data.fixed_self_indices = fixed_sorted_self_indices_;
    }
    if (leaf_compatible) {
      data.use_p2p_leaf = true;
      data.p2p_leaf = build_cuda_leaf_plan();
    } else if (!periodic_.enabled && bsr_identity_compatible &&
               estimate_bsr_bytes(p2p_operator_, sizeof(double)) <=
                   cuda_p2p_bsr_max_bytes_) {
      const std::span<const int> bsr_identities =
          fixed_target_source_indices_.has_value()
              ? std::span<const int>(fixed_sorted_self_indices_)
              : std::span<const int>{};
      data.p2p_bsr = build_static_p2p_bsr_plan(
          p2p_operator_, bsr_identities);
      data.use_p2p_bsr = true;
    }
  }
  if (!data.use_p2p_dictionary && !data.use_p2p_bsr && !data.use_p2p_leaf) {
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
