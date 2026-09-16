// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/uniform_fmm.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>

#include "cdfmm/backend/cpu/p2p.hpp"
#include "fmm/internal.hpp"

namespace cdfmm {

namespace {

std::string_view name(const ExpansionBasis value) {
  switch (value) {
  case ExpansionBasis::Cartesian:
    return "cartesian";
  case ExpansionBasis::Spherical:
    return "spherical";
  }
  return "unknown";
}

std::string_view name(const SphericalM2LBackend value) {
  switch (value) {
  case SphericalM2LBackend::StaticDense:
    return "static_dense";
  }
  return "unknown";
}

std::string_view name(const ExecutionBackend value) {
  switch (value) {
  case ExecutionBackend::Auto:
    return "auto";
  case ExecutionBackend::CpuReference:
    return "cpu_reference";
  case ExecutionBackend::CpuStatic:
    return "cpu_static";
  case ExecutionBackend::CudaM2LP2P:
    return "cuda_partial";
  case ExecutionBackend::CudaFull:
    return "cuda_full";
  }
  return "unknown";
}

std::string_view name(const M2LBackend value) {
  switch (value) {
  case M2LBackend::Static:
    return "static";
  case M2LBackend::Reference:
    return "reference";
  }
  return "unknown";
}

std::string_view name(const StaticMatrixBackend value) {
  switch (value) {
  case StaticMatrixBackend::Portable:
    return "portable";
  case StaticMatrixBackend::OneMkl:
    return "one_mkl";
  }
  return "unknown";
}

std::string_view name(const StaticOperatorExecutor value) {
  switch (value) {
  case StaticOperatorExecutor::Reference:
    return "reference";
  case StaticOperatorExecutor::Portable:
    return "portable";
  case StaticOperatorExecutor::OneMkl:
    return "one_mkl";
  case StaticOperatorExecutor::Cuda:
    return "cuda";
  }
  return "unknown";
}

std::string_view name(const P2PExecutionPacking value) {
  switch (value) {
  case P2PExecutionPacking::Reference:
    return "reference";
  case P2PExecutionPacking::CanonicalAos:
    return "canonical_aos";
  case P2PExecutionPacking::ParticleRowSoa:
    return "particle_row_soa";
  case P2PExecutionPacking::TensorDictionary:
    return "tensor_dictionary";
  case P2PExecutionPacking::CudaBsr3:
    return "cuda_bsr3";
  case P2PExecutionPacking::LeafBlock:
    return "leaf_block";
  case P2PExecutionPacking::PointGeometry:
    return "point_geometry";
  case P2PExecutionPacking::Auto:
    return "auto";
  }
  return "unknown";
}

std::string_view name(const StaticPrecision value) {
  switch (value) {
  case StaticPrecision::Float32:
    return "float32";
  case StaticPrecision::Float64:
    return "float64";
  }
  return "unknown";
}

std::string_view name(const SourceGeometry value) {
  switch (value) {
  case SourceGeometry::PointDipole:
    return "point_dipole";
  case SourceGeometry::RectangularPrism:
    return "rectangular_prism";
  case SourceGeometry::Tetrahedron:
    return "tetrahedron";
  }
  return "unknown";
}

std::string_view name(const TargetGeometry value) {
  switch (value) {
  case TargetGeometry::Point:
    return "point";
  case TargetGeometry::RectangularPrism:
    return "rectangular_prism";
  case TargetGeometry::Tetrahedron:
    return "tetrahedron";
  }
  return "unknown";
}

std::string_view name(const SourceModel value) {
  switch (value) {
  case SourceModel::PointDipole:
    return "point_dipole";
  case SourceModel::ExactGeometry:
    return "exact_geometry";
  }
  return "unknown";
}

std::string_view name(const TargetModel value) {
  switch (value) {
  case TargetModel::Point:
    return "point";
  case TargetModel::ExactGeometry:
    return "exact_geometry";
  }
  return "unknown";
}


void append_vec3(std::ostringstream& stream, const Vec3& value) {
  stream << '[' << value.x << ", " << value.y << ", " << value.z << ']';
}

void append_size_option(std::ostringstream& stream, const char* label,
                        const std::vector<CuboidSize>& sizes) {
  stream << "  " << label << ".count: " << sizes.size() << '\n';
  if (sizes.size() == 1) {
    stream << "  " << label << ".common: [" << sizes[0].hx << ", "
           << sizes[0].hy << ", " << sizes[0].hz << "]\n";
  } else if (sizes.size() > 1) {
    stream << "  " << label << ".layout: per_particle\n";
  }
}

} // namespace

namespace detail {

std::string_view p2p_packing_name(const P2PExecutionPacking value) noexcept {
  return name(value);
}

std::string_view execution_backend_name(const ExecutionBackend value) noexcept {
  return name(value);
}

} // namespace detail

void UniformFmm::print_initialisation_summary(
    const UniformFmmOptions& options) const {
  const StaticExecutionPlan executors = execution_plan();
  std::ostringstream stream;
  stream << std::boolalpha << std::setprecision(12);
  stream << "[cdfmm] UniformFmm initialisation\n";
  stream << "  source_count: " << tree_->sorted_source_positions().size()
         << '\n';
  stream << "  target_count: " << tree_->sorted_target_positions().size()
         << '\n';
  stream << "  expansion_order: " << expansion_order() << '\n';
  stream << "  coefficient_count: " << coefficient_count() << '\n';
  stream << "  expansion_basis: " << name(expansion_basis_) << '\n';
  stream << "  precision: " << name(precision_) << '\n';
  stream << "  backend.requested: " << name(options.backend) << '\n';
  stream << "  backend.resolved: " << name(backend_) << '\n';
  stream << "  m2l_backend.requested: " << name(options.m2l_backend) << '\n';
  stream << "  m2l_backend.resolved: " << name(m2l_backend_) << '\n';
  stream << "  spherical_m2l_backend: " << name(spherical_m2l_backend_)
         << '\n';
  stream << "  static_matrix_backend: " << name(static_matrix_backend_)
         << '\n';
  stream << "  executor.p2m: " << name(executors.p2m) << '\n';
  stream << "  executor.m2m: " << name(executors.m2m) << '\n';
  stream << "  executor.m2l: " << name(executors.m2l) << '\n';
  stream << "  executor.l2l: " << name(executors.l2l) << '\n';
  stream << "  executor.l2p: " << name(executors.l2p) << '\n';
  stream << "  executor.p2p: " << name(executors.p2p) << '\n';
  stream << "  p2p_packing.requested: " << name(requested_p2p_packing_)
         << '\n';
  stream << "  p2p_packing: " << name(p2p_execution_packing_) << '\n';
  stream << "  spatial_layout: " << cuda_policy::name(spatial_layout_) << '\n';
  if (cuda_policy_ && (backend_ == ExecutionBackend::CudaM2LP2P ||
                       backend_ == ExecutionBackend::CudaFull)) {
    const cuda_policy::CudaExecutionPolicy &policy = cuda_policy_->policy;
    const cuda_policy::CudaExecutionPolicyInputs &inputs =
        cuda_policy_->inputs;
    stream << "  cuda_policy.p2p_packing: "
           << cuda_policy::name(policy.p2p_packing) << '\n';
    if (policy.p2p_packing == cuda_policy::CudaP2PPacking::SignedDictionary) {
      stream << "  cuda_policy.dictionary_executor: "
             << cuda_policy::name(policy.dictionary_executor)
             << (policy.dictionary_from_layout ? " (from spatial_layout)"
                                               : " (explicit)")
             << '\n';
    }
    stream << "  cuda_policy.mean_leaf_occupancy: "
           << inputs.mean_leaf_occupancy << '\n';
    stream << "  cuda_policy.p2p_pairs: " << inputs.p2p_pair_count << '\n';
    stream << "  cuda_policy.m2l_translations: "
           << inputs.m2l_translation_count << '\n';
    stream << "  cuda_policy.m2l_pairs_per_thread: "
           << policy.m2l_pairs_per_thread << '\n';
    stream << "  cuda_policy.translation_lanes: " << policy.translation_wide_lanes
           << " (<= " << policy.translation_wide_outputs << " outputs), "
           << policy.translation_lanes << " (larger levels)" << '\n';
  }
  stream << "  p2p.signed_target_tile_size: "
         << signed_p2p_target_tile_size_ << '\n';
  stream << "  p2p.signed_simd_path: " << static_p2p_signed_simd_path()
         << '\n';
  stream << "  tree.max_level.requested: " << options.tree.max_level << '\n';
  stream << "  tree.max_level.resolved: " << tree_->max_level() << '\n';
  stream << "  tree.include_empty_nodes: "
         << options.tree.include_empty_nodes << '\n';
  stream << "  tree.cubic_root_box: " << options.tree.cubic_root_box << '\n';
  stream << "  tree.root_centre.requested: ";
  if (options.tree.root_centre.has_value()) {
    append_vec3(stream, *options.tree.root_centre);
  } else {
    stream << "auto";
  }
  stream << '\n';
  stream << "  tree.root_centre.physical_resolved: ";
  append_vec3(stream, physical_root_centre_);
  stream << '\n';
  stream << "  tree.root_half_width.requested: ";
  if (options.tree.root_half_width.has_value()) {
    stream << *options.tree.root_half_width;
  } else {
    stream << "auto";
  }
  stream << '\n';
  stream << "  tree.root_half_width.physical_resolved: "
         << 0.5 * physical_root_side_length_ << '\n';
  stream << "  tree.root_side_length.physical_resolved: "
         << physical_root_side_length_ << '\n';
  stream << "  tree.root_centre.internal: ";
  append_vec3(stream, tree_->root_centre());
  stream << '\n';
  stream << "  tree.root_half_width.internal: " << tree_->root_half_width()
         << '\n';
  stream << "  source_geometry: " << name(source_geometry_) << '\n';
  append_size_option(stream, "source_sizes", options.source_sizes);
  stream << "  near_field_source_model: "
         << name(near_field_source_model_) << '\n';
  stream << "  far_field_source_model: "
         << name(far_field_source_model_) << '\n';
  stream << "  target_geometry: " << name(target_geometry_) << '\n';
  append_size_option(stream, "target_sizes", options.target_sizes);
  stream << "  near_field_target_model: "
         << name(near_field_target_model_) << '\n';
  stream << "  far_field_target_model: "
         << name(far_field_target_model_) << '\n';
  stream << "  fixed_target_source_indices.requested: "
         << options.fixed_target_source_indices.has_value() << '\n';
  stream << "  fixed_target_source_indices.active: "
         << fixed_target_source_indices_.has_value() << '\n';
  if (options.fixed_target_source_indices.has_value()) {
    stream << "  fixed_target_source_indices.count: "
           << options.fixed_target_source_indices->size() << '\n';
  }
  stream << "  cuda_p2p_bsr_max_bytes: " << cuda_p2p_bsr_max_bytes_ << '\n';
  stream << "  cuda_dictionary_target_owned: "
         << cuda_dictionary_target_owned_ << '\n';
  stream << "  cuda_dictionary_power2_microtiles: "
         << cuda_dictionary_power2_microtiles_ << '\n';
  stream << "  cache.enabled: " << cache_enabled_ << '\n';
  stream << "  cache.directory: " << cache_directory_ << '\n';
  stream << "  cache.universal.key: " << universal_cache_key_ << '\n';
  stream << "  cache.universal.hit: "
         << static_plan_statistics_.universal_cache_hit << '\n';
  stream << "  cache.periodic.key: "
         << (periodic_cache_key_.empty() ? "disabled" : periodic_cache_key_)
         << '\n';
  stream << "  cache.periodic.hit: "
         << static_plan_statistics_.periodic_cache_hit << '\n';
  stream << "  cache.geometry.key: " << geometry_cache_key_ << '\n';
  stream << "  cache.geometry.hit: "
         << static_plan_statistics_.geometry_cache_hit << '\n';
  stream << "  cache.bytes_read: "
         << static_plan_statistics_.cache_bytes_read << '\n';
  stream << "  cache.bytes_written: "
         << static_plan_statistics_.cache_bytes_written << '\n';
  stream << "  setup.normalisation_seconds: "
         << static_plan_statistics_.normalisation.total_seconds << '\n';
  stream << "  setup.tree_construction_seconds: "
         << static_plan_statistics_.tree_construction.total_seconds << '\n';
  stream << "  setup.topology_construction_seconds: "
         << static_plan_statistics_.topology_construction.total_seconds << '\n';
  stream << "  setup.universal_cache_lookup_seconds: "
         << static_plan_statistics_.universal_cache_lookup.total_seconds
         << '\n';
  stream << "  setup.universal_cache_load_seconds: "
         << static_plan_statistics_.universal_cache_load.total_seconds << '\n';
  stream << "  setup.universal_operator_build_seconds: "
         << static_plan_statistics_.universal_operator_build.total_seconds
         << '\n';
  stream << "  setup.universal_cache_write_seconds: "
         << static_plan_statistics_.universal_cache_write.total_seconds
         << '\n';
  stream << "  setup.periodic_cache_lookup_seconds: "
         << static_plan_statistics_.periodic_cache_lookup.total_seconds
         << '\n';
  stream << "  setup.periodic_cache_load_seconds: "
         << static_plan_statistics_.periodic_cache_load.total_seconds << '\n';
  stream << "  setup.periodic_operator_build_seconds: "
         << static_plan_statistics_.periodic_operator_build.total_seconds
         << '\n';
  stream << "  setup.geometry_hash_seconds: "
         << static_plan_statistics_.geometry_hash.total_seconds << '\n';
  stream << "  setup.geometry_cache_lookup_seconds: "
         << static_plan_statistics_.geometry_cache_lookup.total_seconds << '\n';
  stream << "  setup.geometry_cache_load_seconds: "
         << static_plan_statistics_.geometry_cache_load.total_seconds << '\n';
  stream << "  setup.geometry_cache_write_seconds: "
         << static_plan_statistics_.geometry_cache_write.total_seconds << '\n';
  stream << "  setup.p2m_seconds: "
         << static_plan_statistics_.p2m_plan.total_seconds << '\n';
  stream << "  setup.m2m_seconds: "
         << static_plan_statistics_.m2m_plan.total_seconds << '\n';
  stream << "  setup.m2l_seconds: "
         << static_plan_statistics_.m2l_plan.total_seconds << '\n';
  stream << "  setup.l2l_seconds: "
         << static_plan_statistics_.l2l_plan.total_seconds << '\n';
  stream << "  setup.l2p_seconds: "
         << static_plan_statistics_.l2p_plan.total_seconds << '\n';
  stream << "  setup.p2p_seconds: "
         << static_plan_statistics_.p2p_tensor_plan.total_seconds << '\n';
  stream << "  setup.backend_packing_seconds: "
         << static_plan_statistics_.backend_packing.total_seconds << '\n';
  stream << "  setup.far_field_packing_seconds: "
         << static_plan_statistics_.far_field_packing.total_seconds << '\n';
  stream << "  setup.cuda_upload_seconds: "
         << static_plan_statistics_.cuda_upload.total_seconds << '\n';
  stream << "  setup.static_plan_seconds: "
         << static_plan_statistics_.total.total_seconds << '\n';
  stream << "  setup.total_seconds: "
         << static_plan_statistics_.total_setup.total_seconds << '\n';
  stream << "  periodic.enabled: " << physical_periodic_.enabled << '\n';
  stream << "  periodic.axes: [" << physical_periodic_.axes[0] << ", "
         << physical_periodic_.axes[1] << ", "
         << physical_periodic_.axes[2] << "]\n";
  stream << "  periodic.centre: ";
  append_vec3(stream, physical_periodic_.centre);
  stream << '\n';
  stream << "  periodic.lengths: ";
  append_vec3(stream, physical_periodic_.lengths);
  stream << '\n';
  stream << "  periodic.convention: zero_k0\n";
  stream << "  periodic.setup_tolerance: "
         << physical_periodic_.setup_tolerance
         << '\n';
  stream << "  build.cuda_compiled: " << cuda_compiled() << '\n';
  stream << "  build.one_mkl_available: " << one_mkl_available() << '\n';
  std::cout << stream.str() << std::flush;
}

} // namespace cdfmm
