// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/uniform_fmm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string_view>
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
  case SourceGeometry::UniformCuboid:
    return "uniform_cuboid";
  }
  return "unknown";
}

std::string_view name(const TargetGeometry value) {
  switch (value) {
  case TargetGeometry::Point:
    return "point";
  case TargetGeometry::VolumeAveragedCuboid:
    return "volume_averaged_cuboid";
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

struct UniformFmm::NormalisedGeometry {
  std::vector<Vec3> physical_source_positions{};
  std::vector<Vec3> physical_target_positions{};
  UniformTreeOptions physical_tree_options{};
  std::vector<Vec3> source_positions{};
  std::vector<Vec3> target_positions{};
  UniformFmmOptions options{};
  Vec3 physical_root_centre{};
  double physical_root_side_length{1.0};
  double normalisation_seconds{0.0};
  Clock::time_point construction_start{};
};

UniformFmm::NormalisedGeometry UniformFmm::normalise_geometry(
    const std::vector<Vec3>& source_positions,
    const std::vector<Vec3>& target_positions,
    const UniformFmmOptions& options) {
  const auto normalisation_start = Clock::now();
  validate_periodic_cell(options.periodic);

  const auto finite_position = [](const Vec3& position) {
    return std::isfinite(position.x) && std::isfinite(position.y) &&
        std::isfinite(position.z);
  };
  const auto validate_sizes = [](const std::vector<CuboidSize>& sizes,
                                 const std::size_t count,
                                 const char* description) {
    if (sizes.size() != 1 && sizes.size() != count) {
      throw std::invalid_argument(std::string(description) +
                                  " sizes must contain one or one per object");
    }
    for (const CuboidSize& size : sizes) {
      if (!std::isfinite(size.hx) || !std::isfinite(size.hy) ||
          !std::isfinite(size.hz) || size.hx <= 0.0 || size.hy <= 0.0 ||
          size.hz <= 0.0) {
        throw std::invalid_argument(std::string(description) +
                                    " dimensions must be finite and positive");
      }
    }
  };

  for (const Vec3& position : source_positions) {
    if (!finite_position(position)) {
      throw std::invalid_argument("source positions must be finite");
    }
  }
  for (const Vec3& position : target_positions) {
    if (!finite_position(position)) {
      throw std::invalid_argument("target positions must be finite");
    }
  }
  if (options.source_geometry == SourceGeometry::UniformCuboid) {
    validate_sizes(options.source_sizes, source_positions.size(),
                   "source cuboid");
  }
  if (options.target_geometry == TargetGeometry::VolumeAveragedCuboid) {
    validate_sizes(options.target_sizes, target_positions.size(),
                   "target cuboid");
  }

  Vec3 minimum{std::numeric_limits<double>::infinity(),
               std::numeric_limits<double>::infinity(),
               std::numeric_limits<double>::infinity()};
  Vec3 maximum{-std::numeric_limits<double>::infinity(),
               -std::numeric_limits<double>::infinity(),
               -std::numeric_limits<double>::infinity()};
  bool has_geometry = false;
  const auto include_population = [&](const std::vector<Vec3>& positions,
                                      const std::vector<CuboidSize>& sizes,
                                      const bool cuboids) {
    for (std::size_t index = 0; index < positions.size(); ++index) {
      const CuboidSize size = cuboids
          ? sizes[sizes.size() == 1 ? 0 : index]
          : CuboidSize{};
      const Vec3 half_extent{0.5 * size.hx, 0.5 * size.hy,
                             0.5 * size.hz};
      const Vec3 lower = positions[index] - half_extent;
      const Vec3 upper = positions[index] + half_extent;
      minimum.x = std::min(minimum.x, lower.x);
      minimum.y = std::min(minimum.y, lower.y);
      minimum.z = std::min(minimum.z, lower.z);
      maximum.x = std::max(maximum.x, upper.x);
      maximum.y = std::max(maximum.y, upper.y);
      maximum.z = std::max(maximum.z, upper.z);
      has_geometry = true;
    }
  };
  include_population(source_positions, options.source_sizes,
                     options.source_geometry == SourceGeometry::UniformCuboid);
  include_population(
      target_positions, options.target_sizes,
      options.target_geometry == TargetGeometry::VolumeAveragedCuboid);

  NormalisedGeometry geometry;
  geometry.construction_start = normalisation_start;
  geometry.options = options;
  if (options.periodic.enabled) {
    geometry.physical_root_centre = options.periodic.centre;
    geometry.physical_root_side_length = options.periodic.lengths.x;
  } else {
    if (!has_geometry) {
      minimum = {};
      maximum = {};
    }
    geometry.physical_root_centre = options.tree.root_centre.value_or(
        (minimum + maximum) * 0.5);
    const Vec3 upper_distance = maximum - geometry.physical_root_centre;
    const Vec3 lower_distance = geometry.physical_root_centre - minimum;
    double required_half_width = has_geometry
        ? std::max({upper_distance.x, upper_distance.y, upper_distance.z,
                    lower_distance.x, lower_distance.y, lower_distance.z})
        : 0.0;
    if (required_half_width < 0.0 || !std::isfinite(required_half_width)) {
      throw std::invalid_argument("invalid physical root geometry");
    }
    if (options.tree.root_half_width.has_value()) {
      const double requested = *options.tree.root_half_width;
      if (!std::isfinite(requested) || requested <= 0.0) {
        throw std::invalid_argument(
            "UniformTreeOptions.root_half_width must be finite and positive");
      }
      const double tolerance = 32.0 * std::numeric_limits<double>::epsilon() *
          std::max({1.0, requested, required_half_width});
      if (requested + tolerance < required_half_width) {
        throw std::invalid_argument(
            "complete geometry lies outside the requested root box");
      }
      required_half_width = requested;
    } else if (required_half_width == 0.0) {
      // A finite convention is required for empty or coincident point sets.
      required_half_width = 0.5;
    }
    geometry.physical_root_side_length = 2.0 * required_half_width;
  }

  const double inverse_length = 1.0 / geometry.physical_root_side_length;
  const auto canonicalise = [](const double value) {
    // Inputs translated far from a small geometry lose a few low mantissa
    // bits during subtraction.  A nanounit grid removes that representation
    // noise so physically translated/rescaled copies have identical canonical
    // coordinates and therefore identical reusable plans.
    constexpr double resolution = 1.0e9;
    return std::nearbyint(value * resolution) / resolution;
  };
  const auto physical_positions = [&](const std::vector<Vec3>& positions) {
    std::vector<Vec3> wrapped;
    wrapped.reserve(positions.size());
    for (Vec3 position : positions) {
      if (options.periodic.enabled) {
        position = wrap_periodic_position(position, options.periodic);
      }
      wrapped.push_back(position);
    }
    return wrapped;
  };
  const auto normalise_sizes = [inverse_length](
                                   std::vector<CuboidSize>& sizes) {
    for (CuboidSize& size : sizes) {
      size.hx *= inverse_length;
      size.hy *= inverse_length;
      size.hz *= inverse_length;
    }
  };

  geometry.physical_source_positions = physical_positions(source_positions);
  geometry.physical_target_positions = physical_positions(target_positions);
  geometry.source_positions.reserve(geometry.physical_source_positions.size());
  for (const Vec3& position : geometry.physical_source_positions) {
    const Vec3 value =
        (position - geometry.physical_root_centre) * inverse_length;
    geometry.source_positions.push_back(
        {canonicalise(value.x), canonicalise(value.y), canonicalise(value.z)});
  }
  geometry.target_positions.reserve(geometry.physical_target_positions.size());
  for (const Vec3& position : geometry.physical_target_positions) {
    const Vec3 value =
        (position - geometry.physical_root_centre) * inverse_length;
    geometry.target_positions.push_back(
        {canonicalise(value.x), canonicalise(value.y), canonicalise(value.z)});
  }
  geometry.physical_tree_options = options.tree;
  geometry.physical_tree_options.root_centre = geometry.physical_root_centre;
  geometry.physical_tree_options.root_half_width =
      0.5 * geometry.physical_root_side_length;
  normalise_sizes(geometry.options.source_sizes);
  normalise_sizes(geometry.options.target_sizes);
  const auto canonicalise_sizes = [canonicalise](
                                      std::vector<CuboidSize>& sizes) {
    for (CuboidSize& size : sizes) {
      size.hx = canonicalise(size.hx);
      size.hy = canonicalise(size.hy);
      size.hz = canonicalise(size.hz);
    }
  };
  canonicalise_sizes(geometry.options.source_sizes);
  canonicalise_sizes(geometry.options.target_sizes);
  geometry.options.tree.root_centre = Vec3{};
  geometry.options.tree.root_half_width = 0.5;
  if (geometry.options.periodic.enabled) {
    geometry.options.periodic.centre = {};
    geometry.options.periodic.lengths = {1.0, 1.0, 1.0};
  }
  geometry.normalisation_seconds = elapsed_seconds(normalisation_start);
  return geometry;
}

UniformFmm::UniformFmm(const std::vector<Vec3> &source_positions,
                       const UniformFmmOptions &options)
    : UniformFmm(source_positions, std::vector<Vec3>{}, options) {}

UniformFmm::UniformFmm(const std::vector<Vec3> &source_positions,
                       const std::vector<Vec3> &target_positions,
                       const UniformFmmOptions &options)
    : UniformFmm(normalise_geometry(source_positions, target_positions, options),
                 options) {}

UniformFmm::UniformFmm(NormalisedGeometry geometry,
                       const UniformFmmOptions& physical_options)
    : physical_tree_(geometry.physical_source_positions,
                     geometry.physical_target_positions,
                     geometry.physical_tree_options),
      tree_(geometry.source_positions, geometry.target_positions,
            geometry.options.tree),
      physical_root_centre_(geometry.physical_root_centre),
      physical_root_side_length_(geometry.physical_root_side_length),
      physical_periodic_(physical_options.periodic),
      periodic_(geometry.options.periodic),
      basis_(std::max(geometry.options.expansion_order, 0)),
      spherical_basis_(std::max(geometry.options.expansion_order, 0)),
      expansion_basis_(geometry.options.expansion_basis),
      spherical_m2l_backend_(geometry.options.spherical_m2l_backend),
      m2l_backend_(geometry.options.m2l_backend),
      static_matrix_backend_(geometry.options.static_matrix_backend),
      precision_(geometry.options.precision),
      coordinate_scale_(geometry.physical_root_side_length) {
  const UniformFmmOptions& options = geometry.options;
  static_plan_statistics_.normalisation.add(geometry.normalisation_seconds);
  static_plan_statistics_.tree_construction = tree_.build_timings().total;
  static_plan_statistics_.tree_construction.add(
      physical_tree_.build_timings().total.total_seconds);
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
      tree_.nodes().size() * static_cast<std::size_t>(coefficient_count());
  const std::size_t source_count = tree_.sorted_source_positions().size();
  const std::size_t target_count = tree_.sorted_target_positions().size();
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
  initialise_cache_keys(options);
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
      sorted_target_sizes_.capacity() * sizeof(CuboidSize);
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
  static_plan_statistics_.total_setup.add(
      elapsed_seconds(geometry.construction_start));
  print_initialisation_summary(physical_options);
}

void UniformFmm::print_initialisation_summary(
    const UniformFmmOptions& options) const {
  const StaticExecutionPlan executors = execution_plan();
  std::ostringstream stream;
  stream << std::boolalpha << std::setprecision(12);
  stream << "[cdfmm] UniformFmm initialisation\n";
  stream << "  source_count: " << tree_.sorted_source_positions().size()
         << '\n';
  stream << "  target_count: " << tree_.sorted_target_positions().size()
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
  stream << "  p2p_packing: " << name(p2p_execution_packing_) << '\n';
  stream << "  tree.max_level.requested: " << options.tree.max_level << '\n';
  stream << "  tree.max_level.resolved: " << tree_.max_level() << '\n';
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
  append_vec3(stream, tree_.root_centre());
  stream << '\n';
  stream << "  tree.root_half_width.internal: " << tree_.root_half_width()
         << '\n';
  stream << "  source_geometry: " << name(source_geometry_) << '\n';
  append_size_option(stream, "source_sizes", options.source_sizes);
  stream << "  use_cuboid_p2m: " << use_cuboid_p2m_ << '\n';
  stream << "  target_geometry: " << name(target_geometry_) << '\n';
  append_size_option(stream, "target_sizes", options.target_sizes);
  stream << "  use_cuboid_l2p: " << use_cuboid_l2p_ << '\n';
  stream << "  fixed_target_source_indices.requested: "
         << options.fixed_target_source_indices.has_value() << '\n';
  stream << "  fixed_target_source_indices.active: "
         << fixed_target_source_indices_.has_value() << '\n';
  if (options.fixed_target_source_indices.has_value()) {
    stream << "  fixed_target_source_indices.count: "
           << options.fixed_target_source_indices->size() << '\n';
  }
  stream << "  cuda_p2p_bsr_max_bytes: " << cuda_p2p_bsr_max_bytes_ << '\n';
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

void UniformFmm::initialise_p2p_policy(const UniformFmmOptions &options) {
  cuda_p2p_bsr_max_bytes_ = options.cuda_p2p_bsr_max_bytes;
  use_reduced_symmetry_p2p_ = options.use_reduced_symmetry_p2p;
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

void UniformFmm::build_reduced_symmetry_p2p_packing() {
  if (!use_reduced_symmetry_p2p_ || periodic_.enabled) {
    return;
  }

  std::vector<StaticP2PLeafPair> leaf_pairs;
  const auto nodes = tree_.nodes();
  for (const int leaf_index : tree_.occupied_target_leaves()) {
    const TreeNode &leaf = nodes[static_cast<std::size_t>(leaf_index)];
    for (const int neighbour_index : leaf.list1) {
      const TreeNode &neighbour = nodes[static_cast<std::size_t>(neighbour_index)];
      if (neighbour.source_begin == neighbour.source_end) {
        continue;
      }
      leaf_pairs.push_back({static_cast<int>(leaf.target_begin),
                            static_cast<int>(leaf.target_end - leaf.target_begin),
                            static_cast<int>(neighbour.source_begin),
                            static_cast<int>(neighbour.source_end - neighbour.source_begin)});
    }
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
    p2p_tensor_dictionary_plan_ = build_static_p2p_tensor_dictionary_plan(
        promoted, leaf_pairs);
    return;
  }
  p2p_tensor_dictionary_plan_ = build_static_p2p_tensor_dictionary_plan(
      p2p_operator_, leaf_pairs);
}

void UniformFmm::initialise_source_geometry(const UniformFmmOptions &options) {
  source_geometry_ = options.source_geometry;
  use_cuboid_p2m_ = source_geometry_ == SourceGeometry::UniformCuboid &&
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
    throw std::invalid_argument(
        "UniformCuboid sources require a static backend");
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

void UniformFmm::initialise_target_geometry(const UniformFmmOptions &options) {
  target_geometry_ = options.target_geometry;
  use_cuboid_l2p_ =
      target_geometry_ == TargetGeometry::VolumeAveragedCuboid &&
      options.use_cuboid_l2p;
  const std::size_t count = tree_.sorted_target_positions().size();
  if (target_geometry_ == TargetGeometry::Point) {
    if (!options.target_sizes.empty()) {
      throw std::invalid_argument("point targets do not accept cuboid sizes");
    }
    return;
  }
  if (options.target_sizes.size() != 1 &&
      options.target_sizes.size() != count) {
    throw std::invalid_argument(
        "cuboid target sizes must contain one or one per target");
  }
  if (options.backend == ExecutionBackend::CpuReference ||
      (options.backend == ExecutionBackend::Auto &&
       options.m2l_backend == M2LBackend::Reference)) {
    throw std::invalid_argument("cuboid targets require a static backend");
  }
  if (options.target_sizes.size() == 1) {
    sorted_target_sizes_ = options.target_sizes;
    return;
  }
  sorted_target_sizes_.resize(count);
  const auto permutation = tree_.target_permutation();
  for (std::size_t sorted = 0; sorted < count; ++sorted) {
    sorted_target_sizes_[sorted] = options.target_sizes[permutation[sorted]];
  }
}

void UniformFmm::build_cuda_p2p_plan() {
  if (precision_ == StaticPrecision::Float32) {
    if (!periodic_.enabled && fixed_target_source_indices_.has_value() &&
        p2p_bsr_plan_float_.memory().total_bytes() <= cuda_p2p_bsr_max_bytes_) {
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
  if (!periodic_.enabled && fixed_target_source_indices_.has_value()) {
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

void UniformFmm::build_backend_packing() {
  const bool cuda_executes_m2l = backend_ == ExecutionBackend::CudaM2LP2P ||
      backend_ == ExecutionBackend::CudaFull;
  if (static_matrix_backend_ != StaticMatrixBackend::OneMkl ||
      cuda_executes_m2l) {
    return;
  }
  const auto start = Clock::now();
  const auto pack = [this](const auto& plan, auto& groups,
                           const std::size_t scalar_bytes) {
    groups.clear();
    groups.resize(static_cast<std::size_t>(plan.matrix_count));
    for (int id = 0; id < plan.matrix_count; ++id) {
      groups[static_cast<std::size_t>(id)].matrix_id = id;
    }
    for (std::size_t target = 0;
         target + 1 < plan.target_row_offsets.size(); ++target) {
      const int begin = plan.target_row_offsets[target];
      const int end = plan.target_row_offsets[target + 1];
      for (int interaction = begin; interaction < end; ++interaction) {
        const std::size_t slot = static_cast<std::size_t>(interaction);
        auto& group =
            groups[static_cast<std::size_t>(plan.matrix_ids[slot])];
        group.sources.push_back(plan.source_nodes[slot]);
        group.targets.push_back(static_cast<int>(target));
        group.levels.push_back(plan.interaction_levels[slot]);
      }
    }
    for (auto& group : groups) {
      const std::size_t values =
          static_cast<std::size_t>(coefficient_count()) * group.sources.size();
      group.gathered.resize(values);
      group.translated.resize(values);
      const std::size_t metadata_bytes =
          (group.sources.size() + group.targets.size() + group.levels.size()) *
          sizeof(int);
      static_plan_statistics_.interaction_bytes += metadata_bytes;
      static_plan_statistics_.m2l_interaction_bytes += metadata_bytes;
      static_plan_statistics_.scratch_bytes += 2 * values * scalar_bytes;
    }
  };
  if (precision_ == StaticPrecision::Float32) {
    pack(m2l_plan_float_, m2l_groups_float_, sizeof(float));
  } else {
    pack(m2l_plan_, m2l_groups_, sizeof(double));
  }
  static_plan_statistics_.backend_packing.add(elapsed_seconds(start));
}

void UniformFmm::build_missing_universal_operators(
    const bool universal_available, const bool periodic_required) {
  constexpr std::size_t class_count =
      StaticPlanStatistics::theoretical_maximum_m2l_classes;
  if (!universal_available) {
    const auto shared_start = Clock::now();
    constexpr double child_half_width = 0.25;
    for (int child_class = 0; child_class < 8; ++child_class) {
      const Vec3 child_offset{
          (child_class & 1) != 0 ? child_half_width : -child_half_width,
          (child_class & 2) != 0 ? child_half_width : -child_half_width,
          (child_class & 4) != 0 ? child_half_width : -child_half_width};
      if (expansion_basis_ == ExpansionBasis::Spherical) {
        m2m_operators_[child_class] =
            build_static_m2m_operator(spherical_basis_, child_offset * -1.0);
        l2l_operators_[child_class] =
            build_static_l2l_operator(spherical_basis_, child_offset);
      } else {
        m2m_operators_[child_class] =
            build_static_m2m_operator(basis_, child_offset * -1.0);
        l2l_operators_[child_class] =
            build_static_l2l_operator(basis_, child_offset);
      }
      const std::size_t m2m_bytes = m2m_operators_[child_class].entries.size() *
          sizeof(StaticOperatorEntry);
      const std::size_t l2l_bytes = l2l_operators_[child_class].entries.size() *
          sizeof(StaticOperatorEntry);
      static_plan_statistics_.m2m_operator_bytes += m2m_bytes;
      static_plan_statistics_.l2l_operator_bytes += l2l_bytes;
      static_plan_statistics_.operator_bytes += m2m_bytes + l2l_bytes;
      ++static_plan_statistics_.m2m_operators;
      ++static_plan_statistics_.l2l_operators;
    }
    const double shared_seconds = elapsed_seconds(shared_start);
    static_plan_statistics_.m2m_plan.add(shared_seconds);
    static_plan_statistics_.l2l_plan.add(shared_seconds);
    static_plan_statistics_.universal_operator_build.add(shared_seconds);

    const std::size_t matrix_values =
        static_cast<std::size_t>(coefficient_count()) * coefficient_count();
    m2l_plan_.matrices.resize(class_count * matrix_values);
    std::vector<std::array<int, 3>> displacements;
    displacements.reserve(class_count);
    for (int dx = -3; dx <= 3; ++dx) {
      for (int dy = -3; dy <= 3; ++dy) {
        for (int dz = -3; dz <= 3; ++dz) {
          if (std::abs(dx) <= 1 && std::abs(dy) <= 1 &&
              std::abs(dz) <= 1) {
            continue;
          }
          displacements.push_back({dx, dy, dz});
        }
      }
    }
    const auto m2l_start = Clock::now();
#pragma omp parallel for schedule(dynamic) if(class_count >= 8)
    for (std::ptrdiff_t id = 0;
         id < static_cast<std::ptrdiff_t>(class_count); ++id) {
      const auto [dx, dy, dz] =
          displacements[static_cast<std::size_t>(id)];
      const Vec3 displacement{static_cast<double>(dx),
                              static_cast<double>(dy),
                              static_cast<double>(dz)};
      const std::vector<double> matrix =
          expansion_basis_ == ExpansionBasis::Spherical
          ? build_static_m2l_matrix(spherical_basis_, displacement)
          : build_static_m2l_matrix(basis_, displacement);
      std::copy(matrix.begin(), matrix.end(),
                m2l_plan_.matrices.begin() + id * matrix_values);
    }
    static_plan_statistics_.universal_operator_build.add(
        elapsed_seconds(m2l_start));
  }

  if (periodic_required && !periodic_operator_available_) {
    const auto periodic_start = Clock::now();
    const std::vector<double> matrix =
        expansion_basis_ == ExpansionBasis::Spherical
        ? build_static_periodic_m2l_matrix(spherical_basis_, periodic_)
        : build_static_periodic_m2l_matrix(basis_, periodic_);
    m2l_plan_.matrices.insert(m2l_plan_.matrices.end(), matrix.begin(),
                              matrix.end());
    periodic_operator_available_ = true;
    static_plan_statistics_.periodic_operator_build.add(
        elapsed_seconds(periodic_start));
  }
}

void UniformFmm::build_static_plan() {
  // This is the geometry-dependent half of the evaluator. None of the data
  // built here depends on dipole moments, so it remains valid for every later
  // evaluate() call; see docs/static-architecture.md.
  const auto total_start = Clock::now();
  static_plan_statistics_.expansion_order = expansion_order();
  static_plan_statistics_.coefficient_count =
      static_cast<std::size_t>(coefficient_count());
  static_plan_statistics_.spherical =
      expansion_basis_ == ExpansionBasis::Spherical;
  static_plan_statistics_.tree_bytes = tree_.memory_statistics().total_bytes();
  bool universal_available = load_universal_cache();
  const bool periodic_required = periodic_.enabled && !tree_.nodes().empty() &&
      tree_.nodes().front().source_count() != 0 &&
      tree_.nodes().front().target_count() != 0;
  const bool universal_write_required = !universal_available ||
      (periodic_required && !periodic_operator_available_);
  if (universal_write_required) {
    build_missing_universal_operators(universal_available, periodic_required);
    universal_available = true;
    write_universal_cache();
  }
  if (universal_available) {
    // The universal bank always owns exactly eight child-class templates.
    // Record them on both cache hits and analytical construction paths.
    static_plan_statistics_.m2m_operators = 8;
    static_plan_statistics_.l2l_operators = 8;
  }
  if (universal_available &&
      (!periodic_required || periodic_operator_available_) &&
      load_geometry_cache()) {
    static_plan_statistics_.m2l_operators =
        static_cast<std::size_t>(precision_ == StaticPrecision::Float32
                                     ? m2l_plan_float_.matrix_count
                                     : m2l_plan_.matrix_count);
    static_plan_statistics_.transfer_classes =
        StaticPlanStatistics::theoretical_maximum_m2l_classes;
    static_plan_statistics_.interactions =
        precision_ == StaticPrecision::Float32
            ? m2l_plan_float_.source_nodes.size()
            : m2l_plan_.source_nodes.size();
    static_plan_statistics_.p2p_interactions =
        precision_ == StaticPrecision::Float32
            ? p2p_operator_float_.blocks.size()
            : p2p_operator_.blocks.size();
    static_plan_statistics_.m2m_theoretical_interactions =
        tree_.nodes().empty() ? 0 : tree_.nodes().size() - 1;
    static_plan_statistics_.l2l_theoretical_interactions =
        static_plan_statistics_.m2m_theoretical_interactions;
    try {
      build_reduced_symmetry_p2p_packing();
    } catch (const std::invalid_argument &error) {
      throw std::runtime_error(
          std::string("reduced-symmetry P2P topology is not representable: ") +
          error.what());
    }
    if (precision_ == StaticPrecision::Float64) {
      static_plan_statistics_.p2p_value_bytes =
          p2p_operator_.blocks.size() * 6 * sizeof(double);
      static_plan_statistics_.p2p_index_bytes =
          p2p_compact_plan_.row_offsets.size() * sizeof(int) +
          p2p_compact_plan_.source_indices.size() * sizeof(int) +
          p2p_compact_plan_.skip_for_identity.size() * sizeof(unsigned char);
      static_plan_statistics_.p2p_canonical_total_bytes =
          p2p_compact_plan_.memory().total_bytes();
      if (p2p_tensor_dictionary_plan_.has_value()) {
        static_plan_statistics_.p2p_unique_tensors =
            p2p_tensor_dictionary_plan_->tensors[0].size();
        static_plan_statistics_.p2p_dictionary_token_bytes =
            p2p_tensor_dictionary_plan_->tokens.size() * sizeof(std::uint32_t);
        static_plan_statistics_.p2p_dictionary_tensor_bytes =
            p2p_tensor_dictionary_plan_->tensors[0].size() * 6 * sizeof(double);
        static_plan_statistics_.p2p_dictionary_total_bytes =
            p2p_tensor_dictionary_plan_->memory().total_bytes();
      }
    }
    if (backend_ == ExecutionBackend::CpuStatic) {
      p2p_execution_packing_ = p2p_tensor_dictionary_plan_.has_value()
          ? P2PExecutionPacking::TensorDictionary
          : P2PExecutionPacking::ParticleRowSoa;
    }
    if (precision_ == StaticPrecision::Float32) {
      quantise_static_plan_to_float();
    }
    build_backend_packing();
    static_plan_statistics_.total.add(elapsed_seconds(total_start));
    return;
  }
  const bool universal_cache_hit = universal_available;
  const auto nodes = tree_.nodes();
  using Key = std::tuple<int, int, int>;
  using ClassMap = std::map<Key, std::vector<std::pair<int, int>>>;
  ClassMap classes;

  auto phase_start = Clock::now();
  const std::span<const Vec3> sorted_positions =
      tree_.sorted_source_positions();
  const std::span<const Vec3> sorted_targets =
      tree_.sorted_target_positions();
  const std::span<const CuboidSize> source_sizes = sorted_source_sizes_;
  const std::span<const CuboidSize> target_sizes = sorted_target_sizes_;
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
              : source_sizes.subspan(leaf.source_begin, leaf.source_count());
      plan.operator_map = expansion_basis_ == ExpansionBasis::Spherical
          ? build_static_cuboid_p2m_operator(
                spherical_basis_, leaf.centre,
                leaf_positions, leaf_sizes)
          : build_static_cuboid_p2m_operator(
                basis_, leaf.centre, leaf_positions, leaf_sizes);
    } else if (expansion_basis_ == ExpansionBasis::Spherical) {
      plan.operator_map = build_static_p2m_operator(
          spherical_basis_, leaf.centre, leaf_positions);
    } else {
      plan.operator_map = build_static_p2m_operator(
          basis_, leaf.centre, leaf_positions);
    }
    const std::size_t bytes =
        plan.operator_map.entries.size() * sizeof(StaticOperatorEntry);
    static_plan_statistics_.operator_bytes += bytes;
    static_plan_statistics_.p2m_operator_bytes += bytes;
    p2m_plans_.push_back(std::move(plan));
  }
  static_plan_statistics_.p2m_plan.add(elapsed_seconds(phase_start));

  phase_start = Clock::now();
  static_plan_statistics_.m2m_theoretical_interactions =
      nodes.empty() ? 0 : nodes.size() - 1;
  static_plan_statistics_.l2l_theoretical_interactions =
      static_plan_statistics_.m2m_theoretical_interactions;
  constexpr double canonical_child_half_width = 0.25;
  for (int child_class = 0;
       child_class < (universal_cache_hit ? 0 : 8); ++child_class) {
    // The eight templates use the level-one child displacement. At level l,
    // each translation monomial is multiplied by 2^(-(l-1) degree).
    const Vec3 child_offset{
        (child_class & 1) != 0 ? canonical_child_half_width
                               : -canonical_child_half_width,
        (child_class & 2) != 0 ? canonical_child_half_width
                               : -canonical_child_half_width,
        (child_class & 4) != 0 ? canonical_child_half_width
                               : -canonical_child_half_width};
    if (expansion_basis_ == ExpansionBasis::Spherical) {
      m2m_operators_[child_class] =
          build_static_m2m_operator(spherical_basis_, child_offset * -1.0);
      l2l_operators_[child_class] =
          build_static_l2l_operator(spherical_basis_, child_offset);
    } else {
      m2m_operators_[child_class] =
          build_static_m2m_operator(basis_, child_offset * -1.0);
      l2l_operators_[child_class] =
          build_static_l2l_operator(basis_, child_offset);
    }
    const std::size_t m2m_bytes =
        m2m_operators_[child_class].entries.size() *
        sizeof(StaticOperatorEntry);
    const std::size_t l2l_bytes =
        l2l_operators_[child_class].entries.size() *
        sizeof(StaticOperatorEntry);
    static_plan_statistics_.m2m_operator_bytes += m2m_bytes;
    static_plan_statistics_.l2l_operator_bytes += l2l_bytes;
    static_plan_statistics_.operator_bytes += m2m_bytes + l2l_bytes;
    ++static_plan_statistics_.m2m_operators;
    ++static_plan_statistics_.l2l_operators;
  }
  const double shared_translation_seconds = elapsed_seconds(phase_start);
  static_plan_statistics_.m2m_plan.add(shared_translation_seconds);
  if (!universal_cache_hit) {
    static_plan_statistics_.universal_operator_build.add(
        shared_translation_seconds);
  }
  // The two triangular families are constructed together from each shared
  // parent-child displacement class.
  static_plan_statistics_.l2l_plan = static_plan_statistics_.m2m_plan;

  phase_start = Clock::now();
  for (const TreeNode &target : nodes) {
    if (target.level == 0 || target.target_count() == 0) {
      continue;
    }
    if (periodic_.enabled) {
      const int boxes_per_axis = 1 << target.level;
      const auto identities = build_periodic_list2(
          target.level, {target.ix, target.iy, target.iz});
      for (const PeriodicBoxIdentity& identity : identities) {
        const TreeNode& source =
            nodes[static_cast<std::size_t>(identity.node)];
        if (source.source_count() == 0) {
          continue;
        }
        const Key key{
            target.ix -
                (source.ix + identity.image_shift[0] * boxes_per_axis),
            target.iy -
                (source.iy + identity.image_shift[1] * boxes_per_axis),
            target.iz -
                (source.iz + identity.image_shift[2] * boxes_per_axis)};
        classes[key].emplace_back(identity.node, target.index);
      }
    } else {
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
  }
  static_plan_statistics_.transfer_discovery.add(elapsed_seconds(phase_start));

  const int coefficient_count = this->coefficient_count();
  m2l_plan_.coefficient_count = coefficient_count;
  const bool has_periodic_root = periodic_.enabled && !nodes.empty() &&
      nodes.front().source_count() != 0 && nodes.front().target_count() != 0;
  std::vector<Key> universal_classes;
  universal_classes.reserve(
      StaticPlanStatistics::theoretical_maximum_m2l_classes);
  std::map<Key, int> universal_class_ids;
  for (int dx = -3; dx <= 3; ++dx) {
    for (int dy = -3; dy <= 3; ++dy) {
      for (int dz = -3; dz <= 3; ++dz) {
        if (std::abs(dx) <= 1 && std::abs(dy) <= 1 &&
            std::abs(dz) <= 1) {
          continue;
        }
        const Key key{dx, dy, dz};
        universal_class_ids.emplace(
            key, static_cast<int>(universal_classes.size()));
        universal_classes.push_back(key);
      }
    }
  }
  m2l_plan_.matrix_count = static_cast<int>(universal_classes.size()) +
      (has_periodic_root ? 1 : 0);
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
      static_cast<std::size_t>(expansion_order() + 2), 1.0);

  for (int level = 0; level <= tree_.leaf_level(); ++level) {
    m2l_plan_.level_target_begin[static_cast<std::size_t>(level)] =
        level_offset(level);
    m2l_plan_.level_target_end[static_cast<std::size_t>(level)] =
        level_offset(level + 1);
    const double box_width =
        2.0 * nodes[static_cast<std::size_t>(level_offset(level))].half_width;
    const std::size_t scaling_offset =
        static_cast<std::size_t>(level) * coefficient_count;
    // The matrices below are dimensionless and level independent. These two
    // degree-dependent factors restore physical box width; see the M2L
    // normalisation section in docs/math.md.
    inverse_width_powers[0] = 1.0;
    for (int degree = 1; degree <= expansion_order() + 1; ++degree) {
      inverse_width_powers[static_cast<std::size_t>(degree)] =
          inverse_width_powers[static_cast<std::size_t>(degree - 1)] /
          box_width;
    }
    // Each coefficient is filled once from its known degree, rather than
    // rescanning the complete basis for every possible degree.
    for (int index = 0; index < coefficient_count; ++index) {
      const int degree = coefficient_degree(index);
      m2l_plan_.multipole_scaling[scaling_offset + index] =
          inverse_width_powers[static_cast<std::size_t>(degree)];
      m2l_plan_.local_scaling[scaling_offset + index] =
          inverse_width_powers[static_cast<std::size_t>(degree + 1)];
    }
  }

  phase_start = Clock::now();
  using ClassEntry = ClassMap::value_type;
  std::vector<const ClassEntry*> ordered_classes;
  ordered_classes.reserve(classes.size());
  std::size_t interaction_count = 0;
  for (const ClassEntry& entry : classes) {
    ordered_classes.push_back(&entry);
    interaction_count += entry.second.size();
  }
  if (has_periodic_root) {
    ++interaction_count;
  }
  const std::size_t matrix_values =
      static_cast<std::size_t>(coefficient_count) * coefficient_count;
  m2l_plan_.matrices.resize(
      static_cast<std::size_t>(m2l_plan_.matrix_count) * matrix_values);

  // Displacement classes are independent setup work. Constructing them in
  // parallel substantially reduces high-order plan initialisation while
  // preserving the canonical lexicographic matrix-ID ordering.
  const std::ptrdiff_t class_count =
      static_cast<std::ptrdiff_t>(universal_classes.size());
  const auto universal_m2l_start = Clock::now();
#pragma omp parallel for schedule(dynamic) if (class_count >= 8)
  for (std::ptrdiff_t id = 0; id < class_count; ++id) {
    if (universal_cache_hit) {
      continue;
    }
    const Key& key = universal_classes[static_cast<std::size_t>(id)];
    const auto [dx, dy, dz] = key;
    const Vec3 R{static_cast<double>(dx), static_cast<double>(dy),
                 static_cast<double>(dz)};
    const std::vector<double> matrix =
        expansion_basis_ == ExpansionBasis::Spherical
            ? build_static_m2l_matrix(spherical_basis_, R)
            : build_static_m2l_matrix(basis_, R);
    std::copy(matrix.begin(), matrix.end(),
              m2l_plan_.matrices.begin() + id * matrix_values);
  }
  if (!universal_cache_hit) {
    static_plan_statistics_.universal_operator_build.add(
        elapsed_seconds(universal_m2l_start));
  }

  if (has_periodic_root && !periodic_operator_available_) {
    // Wrapped traversal resolves the central root and its 26 neighbours. The
    // appended self translation represents every more distant lattice image.
    const auto periodic_build_start = Clock::now();
    const std::vector<double> matrix =
        expansion_basis_ == ExpansionBasis::Spherical
            ? build_static_periodic_m2l_matrix(spherical_basis_, periodic_)
            : build_static_periodic_m2l_matrix(basis_, periodic_);
    std::copy(
        matrix.begin(), matrix.end(),
        m2l_plan_.matrices.begin() +
            static_cast<std::ptrdiff_t>(universal_classes.size() *
                                        matrix_values));
    static_plan_statistics_.periodic_operator_build.add(
        elapsed_seconds(periodic_build_start));
    periodic_operator_available_ = true;
  }

  for (const ClassEntry* entry : ordered_classes) {
    for (const auto [source, target] : entry->second) {
      (void)source;
      ++m2l_plan_.target_row_offsets[static_cast<std::size_t>(target) + 1];
    }
  }
  if (has_periodic_root) {
    ++m2l_plan_.target_row_offsets[1];
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
  for (const ClassEntry* entry : ordered_classes) {
    const int matrix_id = universal_class_ids.at(entry->first);
    for (const auto [source, target] : entry->second) {
      const int slot = row_cursors[static_cast<std::size_t>(target)]++;
      m2l_plan_.source_nodes[static_cast<std::size_t>(slot)] = source;
      m2l_plan_.matrix_ids[static_cast<std::size_t>(slot)] = matrix_id;
      m2l_plan_.interaction_levels[static_cast<std::size_t>(slot)] =
          nodes[static_cast<std::size_t>(target)].level;
    }
  }
  if (has_periodic_root) {
    const int slot = row_cursors[0]++;
    m2l_plan_.source_nodes[static_cast<std::size_t>(slot)] = 0;
    m2l_plan_.matrix_ids[static_cast<std::size_t>(slot)] =
        static_cast<int>(universal_classes.size());
    m2l_plan_.interaction_levels[static_cast<std::size_t>(slot)] = 0;
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

  static_plan_statistics_.transfer_classes = universal_classes.size();
  static_plan_statistics_.m2l_operators =
      static_cast<std::size_t>(m2l_plan_.matrix_count);
  static_plan_statistics_.buffer_allocation.add(elapsed_seconds(phase_start));
  phase_start = Clock::now();
  l2p_evaluators_.resize(sorted_targets.size());
  for (const int leaf_index : tree_.occupied_target_leaves()) {
    const TreeNode &leaf = nodes[static_cast<std::size_t>(leaf_index)];
    for (std::size_t target = leaf.target_begin; target < leaf.target_end;
         ++target) {
      l2p_evaluators_[target] =
          expansion_basis_ == ExpansionBasis::Spherical
              ? use_cuboid_l2p_
                    ? build_static_cuboid_l2p_evaluator(
                          spherical_basis_, leaf.centre,
                          sorted_targets[target],
                          target_sizes[target_sizes.size() == 1 ? 0 : target])
                    : build_static_l2p_evaluator(
                          spherical_basis_, leaf.centre,
                          sorted_targets[target])
          : use_cuboid_l2p_
              ? build_static_cuboid_l2p_evaluator(
                    basis_, leaf.centre,
                    sorted_targets[target],
                    target_sizes[target_sizes.size() == 1 ? 0 : target])
              : build_static_l2p_evaluator(basis_, leaf.centre,
                                           sorted_targets[target]);
      const std::size_t bytes =
          4 * static_cast<std::size_t>(coefficient_count) * sizeof(double);
      static_plan_statistics_.operator_bytes += bytes;
      static_plan_statistics_.l2p_operator_bytes += bytes;
    }
  }
  static_plan_statistics_.l2p_plan.add(elapsed_seconds(phase_start));

  phase_start = Clock::now();
  if (periodic_.enabled) {
    std::vector<StaticP2PInteraction> near_interactions;
    for (const int leaf_index : tree_.occupied_target_leaves()) {
      const TreeNode& leaf = nodes[static_cast<std::size_t>(leaf_index)];
      const auto identities = build_periodic_list1(
          leaf.level, {leaf.ix, leaf.iy, leaf.iz});
      for (std::size_t target = leaf.target_begin; target < leaf.target_end;
           ++target) {
        for (const PeriodicBoxIdentity& identity : identities) {
          const TreeNode& neighbour =
              nodes[static_cast<std::size_t>(identity.node)];
          const Vec3 source_shift{
              periodic_.lengths.x * identity.image_shift[0],
              periodic_.lengths.y * identity.image_shift[1],
              periodic_.lengths.z * identity.image_shift[2],
          };
          const bool central_image = identity.image_shift[0] == 0 &&
              identity.image_shift[1] == 0 &&
              identity.image_shift[2] == 0;
          for (std::size_t source = neighbour.source_begin;
               source < neighbour.source_end; ++source) {
            near_interactions.push_back({
                static_cast<int>(target), static_cast<int>(source),
                source_shift, central_image});
          }
        }
      }
    }
    p2p_operator_ = build_static_p2p_operator(
        sorted_targets, sorted_positions, near_interactions, source_geometry_,
        source_sizes, target_geometry_, target_sizes);
  } else {
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
        source_sizes, target_geometry_, target_sizes);
  }
  p2p_compact_plan_ = build_static_p2p_compact_plan(p2p_operator_);
  try {
    build_reduced_symmetry_p2p_packing();
  } catch (const std::invalid_argument &error) {
    throw std::runtime_error(
        std::string("reduced-symmetry P2P topology is not representable: ") +
        error.what());
  }
  if (backend_ == ExecutionBackend::CpuStatic) {
    p2p_execution_packing_ = p2p_tensor_dictionary_plan_
        ? P2PExecutionPacking::TensorDictionary
        : P2PExecutionPacking::ParticleRowSoa;
  }
  static_plan_statistics_.p2p_interactions = p2p_operator_.blocks.size();
  static_plan_statistics_.p2p_value_bytes =
      p2p_operator_.blocks.size() * 6 * sizeof(double);
  static_plan_statistics_.p2p_index_bytes =
      p2p_compact_plan_.row_offsets.size() * sizeof(int) +
      p2p_compact_plan_.source_indices.size() * sizeof(int) +
      p2p_compact_plan_.skip_for_identity.size() * sizeof(unsigned char);
  static_plan_statistics_.p2p_canonical_total_bytes =
      p2p_compact_plan_.memory().total_bytes();
  if (p2p_tensor_dictionary_plan_.has_value()) {
    static_plan_statistics_.p2p_unique_tensors =
        p2p_tensor_dictionary_plan_->tensors[0].size();
    static_plan_statistics_.p2p_dictionary_token_bytes =
        p2p_tensor_dictionary_plan_->tokens.size() * sizeof(std::uint32_t);
    static_plan_statistics_.p2p_dictionary_tensor_bytes =
        p2p_tensor_dictionary_plan_->tensors[0].size() * 6 * sizeof(double);
    static_plan_statistics_.p2p_dictionary_total_bytes =
        p2p_tensor_dictionary_plan_->memory().total_bytes();
  }
  static_plan_statistics_.operator_bytes +=
      p2p_operator_.memory_bytes() + p2p_compact_plan_.memory().total_bytes();
  static_plan_statistics_.near_field_operator_bytes =
      p2p_operator_.memory_bytes() + p2p_compact_plan_.memory().total_bytes();
  static_plan_statistics_.p2p_tensor_plan.add(elapsed_seconds(phase_start));
  static_plan_statistics_.total.add(elapsed_seconds(total_start));
  ++static_plan_statistics_.construction_count;

  write_geometry_cache();

  if (precision_ == StaticPrecision::Float32) {
    quantise_static_plan_to_float();
  }
  // oneMKL derives execution-only gather/GEMM/scatter packing from the same
  // canonical target-row metadata used by portable CPU and CUDA.
  build_backend_packing();
}

void UniformFmm::quantise_static_plan_to_float() {
  if (!geometry_cache_loaded_direct_float_) {
    p2m_plans_float_.reserve(p2m_plans_.size());
    for (const P2MPlan &plan : p2m_plans_) {
      p2m_plans_float_.push_back(
          {plan.leaf, quantise_static_operator(plan.operator_map)});
    }
  }

  for (int child_class = 0; child_class < 8; ++child_class) {
    m2m_operators_float_[child_class] =
        quantise_static_operator(m2m_operators_[child_class]);
    l2l_operators_float_[child_class] =
        quantise_static_operator(l2l_operators_[child_class]);
  }

  if (!geometry_cache_loaded_direct_float_) {
    l2p_evaluators_float_.reserve(l2p_evaluators_.size());
    for (const StaticL2PEvaluator &evaluator : l2p_evaluators_) {
      l2p_evaluators_float_.push_back(
          quantise_static_l2p_evaluator(evaluator));
    }
    p2p_operator_float_ = quantise_static_p2p_operator(p2p_operator_);
    m2l_plan_float_ = quantise_static_m2l_plan(m2l_plan_);
  } else {
    // Universal matrices are stored separately from the geometry plan. Only
    // this shared array still needs conversion after a direct FP32 load.
    m2l_plan_float_.matrices.assign(m2l_plan_.matrices.begin(),
                                    m2l_plan_.matrices.end());
  }

  const auto p2p_packing_start = Clock::now();
  const bool uses_cuda_plan = backend_ == ExecutionBackend::CudaM2LP2P ||
      backend_ == ExecutionBackend::CudaFull;
  if (uses_cuda_plan) {
    p2p_compact_plan_float_ = {};
  } else {
    p2p_compact_plan_float_ =
        build_static_p2p_compact_plan(p2p_operator_float_);
    if (p2p_tensor_dictionary_plan_.has_value()) {
      auto dictionary = quantise_static_p2p_tensor_dictionary_plan(
          *p2p_tensor_dictionary_plan_);
      p2p_tensor_dictionary_plan_float_ = std::move(dictionary);
      p2p_execution_packing_ = P2PExecutionPacking::TensorDictionary;
    }
  }
  if (!periodic_.enabled && fixed_target_source_indices_.has_value()) {
    p2p_bsr_plan_float_ = build_static_p2p_bsr_plan(
        p2p_operator_float_, fixed_sorted_self_indices_);
  }
  static_plan_statistics_.backend_packing.add(
      elapsed_seconds(p2p_packing_start));

  // Recalculate scalar-dependent storage from the representation that will
  // remain alive. Integer metadata is unchanged by precision selection.
  std::size_t operator_bytes = 0;
  std::size_t p2m_bytes = 0;
  std::size_t m2m_bytes = 0;
  std::size_t l2l_bytes = 0;
  for (const FloatP2MPlan &plan : p2m_plans_float_) {
    p2m_bytes +=
        plan.operator_map.entries.size() * sizeof(FloatStaticOperatorEntry);
  }
  for (int child_class = 0; child_class < 8; ++child_class) {
    m2m_bytes += m2m_operators_float_[child_class].entries.size() *
                 sizeof(FloatStaticOperatorEntry);
    l2l_bytes += l2l_operators_float_[child_class].entries.size() *
                 sizeof(FloatStaticOperatorEntry);
  }
  const std::size_t m2l_bytes = (m2l_plan_float_.matrices.size() +
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
  const std::size_t near_field_bytes =
      p2p_operator_float_.memory_bytes() +
      p2p_compact_plan_float_.memory().total_bytes();
  operator_bytes = p2m_bytes + m2m_bytes + l2l_bytes + m2l_bytes + l2p_bytes +
                   near_field_bytes;

  static_plan_statistics_.scalar_bytes = sizeof(float);
  static_plan_statistics_.operator_bytes = operator_bytes;
  static_plan_statistics_.p2m_operator_bytes = p2m_bytes;
  static_plan_statistics_.m2m_operator_bytes = m2m_bytes;
  static_plan_statistics_.m2l_operator_bytes = m2l_bytes;
  static_plan_statistics_.l2l_operator_bytes = l2l_bytes;
  static_plan_statistics_.l2p_operator_bytes = l2p_bytes;
  static_plan_statistics_.near_field_operator_bytes = near_field_bytes;
  static_plan_statistics_.p2p_value_bytes =
      p2p_operator_float_.blocks.size() * 6 * sizeof(float);
  static_plan_statistics_.p2p_index_bytes =
      p2p_operator_float_.row_offsets.size() * sizeof(int) +
      p2p_operator_float_.blocks.size() * 2 * sizeof(int) +
      p2p_compact_plan_float_.source_indices.size() * sizeof(int) +
      p2p_compact_plan_float_.skip_for_identity.size() *
          sizeof(unsigned char);
  static_plan_statistics_.p2p_canonical_total_bytes =
      p2p_compact_plan_float_.memory().total_bytes();
  if (p2p_tensor_dictionary_plan_float_.has_value()) {
    static_plan_statistics_.p2p_unique_tensors =
        p2p_tensor_dictionary_plan_float_->tensors[0].size();
    static_plan_statistics_.p2p_dictionary_token_bytes =
        p2p_tensor_dictionary_plan_float_->tokens.size() *
        sizeof(std::uint32_t);
    static_plan_statistics_.p2p_dictionary_tensor_bytes =
        p2p_tensor_dictionary_plan_float_->tensors[0].size() * 6 *
        sizeof(float);
    static_plan_statistics_.p2p_dictionary_total_bytes =
        p2p_tensor_dictionary_plan_float_->memory().total_bytes();
  }
  static_plan_statistics_.scratch_bytes = 0;
  for (const FloatM2LGroup &group : m2l_groups_float_) {
    static_plan_statistics_.scratch_bytes +=
        (group.gathered.size() + group.translated.size()) * sizeof(float);
  }

  // Drop analytical FP64 construction temporaries. An FP32 plan therefore
  // retains no hidden double-precision operator or expansion representation.
  p2m_plans_.clear();
  p2m_plans_.shrink_to_fit();
  m2m_operators_ = {};
  l2l_operators_ = {};
  l2p_evaluators_.clear();
  l2p_evaluators_.shrink_to_fit();
  p2p_operator_ = {};
  p2p_compact_plan_ = {};
  p2p_tensor_dictionary_plan_.reset();
  m2l_plan_ = {};
  m2l_groups_.clear();
  m2l_groups_.shrink_to_fit();
}

void UniformFmm::build_cuda_full_plan() {
  if (precision_ == StaticPrecision::Float32) {
    FloatCudaFullPlanData data;
    data.coefficient_count = coefficient_count();
    data.node_count = static_cast<int>(tree_.nodes().size());
    data.source_count =
        static_cast<int>(tree_.sorted_source_positions().size());
    data.target_count =
        static_cast<int>(tree_.sorted_target_positions().size());
    data.source_permutation.assign(tree_.source_permutation().begin(),
                                   tree_.source_permutation().end());
    data.target_permutation.assign(tree_.target_permutation().begin(),
                                   tree_.target_permutation().end());
    const int n = coefficient_count();
    data.coefficient_degrees.reserve(static_cast<std::size_t>(n));
    for (int coefficient = 0; coefficient < n; ++coefficient) {
      data.coefficient_degrees.push_back(coefficient_degree(coefficient));
    }
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
    data.m2m.matrix_count = tree_.leaf_level() == 0 ? 0 : 8;
    data.l2l.matrix_count = tree_.leaf_level() == 0 ? 0 : 8;
    for (int level = 1; level <= tree_.leaf_level(); ++level) {
      const int begin = level_offset(level);
      const int end = level_offset(level + 1);
      for (int child_index = begin; child_index < end; ++child_index) {
        const TreeNode &child = nodes[static_cast<std::size_t>(child_index)];
        const int child_class =
            (child.ix & 1) | ((child.iy & 1) << 1) | ((child.iz & 1) << 2);
        const int matrix_id = child_class;
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
      for (std::size_t target = leaf.target_begin; target < leaf.target_end;
           ++target) {
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
      if (!periodic_.enabled) {
        data.use_p2p_bsr =
            p2p_bsr_plan_float_.memory().total_bytes() <=
            cuda_p2p_bsr_max_bytes_;
      }
    }
    if (data.use_p2p_bsr) {
      data.p2p_bsr = std::move(p2p_bsr_plan_float_);
    } else {
      data.p2p = p2p_operator_float_;
    }
    p2p_execution_packing_ = data.use_p2p_bsr
        ? P2PExecutionPacking::CudaBsr3
        : P2PExecutionPacking::CanonicalAos;
    cuda_full_plan_ = std::make_unique<CudaFullPlanOwner>(
        std::make_unique<CudaFullPlan>(data));
    return;
  }

  CudaFullPlanData data;
  data.coefficient_count = coefficient_count();
  data.node_count = static_cast<int>(tree_.nodes().size());
  data.source_count = static_cast<int>(tree_.sorted_source_positions().size());
  data.target_count = static_cast<int>(tree_.sorted_target_positions().size());
  data.source_permutation.assign(tree_.source_permutation().begin(),
                                 tree_.source_permutation().end());
  data.target_permutation.assign(tree_.target_permutation().begin(),
                                 tree_.target_permutation().end());
  const int n = coefficient_count();
  data.coefficient_degrees.reserve(static_cast<std::size_t>(n));
  for (int coefficient = 0; coefficient < n; ++coefficient) {
    data.coefficient_degrees.push_back(coefficient_degree(coefficient));
  }
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
  data.m2m.matrix_count = tree_.leaf_level() == 0 ? 0 : 8;
  data.l2l.matrix_count = tree_.leaf_level() == 0 ? 0 : 8;
  for (int level = 1; level <= tree_.leaf_level(); ++level) {
    const int begin = level_offset(level);
    const int end = level_offset(level + 1);
    for (int child_index = begin; child_index < end; ++child_index) {
      const TreeNode &child = nodes[static_cast<std::size_t>(child_index)];
      const int child_class =
          (child.ix & 1) | ((child.iy & 1) << 1) | ((child.iz & 1) << 2);
      const int matrix_id = child_class;
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
  if (fixed_target_source_indices_.has_value()) {
    data.has_fixed_self_indices = true;
    data.fixed_self_indices = fixed_sorted_self_indices_;
    if (!periodic_.enabled) {
      StaticP2PBsrPlan bsr =
          build_static_p2p_bsr_plan(p2p_operator_, fixed_sorted_self_indices_);
      data.use_p2p_bsr =
          bsr.memory().total_bytes() <= cuda_p2p_bsr_max_bytes_;
      if (data.use_p2p_bsr) {
        data.p2p_bsr = std::move(bsr);
      }
    }
  }
  if (!data.use_p2p_bsr) {
    data.p2p = p2p_operator_;
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
      results[index].H = {static_cast<double>(float_results[index].H.x),
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
    detail::ProfileRange device_range{"cdfmm/cuda_full"};
    const double inverse_volume_scale =
        1.0 / (coordinate_scale_ * coordinate_scale_ * coordinate_scale_);
    for (std::size_t index = 0; index < dipole_moments.size(); ++index) {
      sorted_dipole_moments_[index] =
          dipole_moments[index] * inverse_volume_scale;
    }
    cuda_full_plan_->plan->evaluate(sorted_dipole_moments_, near_fields_,
                                    sorted_self_indices_);
    for (std::size_t target = 0; target < target_count; ++target) {
      results[target].phi = 0.0;
      results[target].H = near_fields_[target];
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
            l2p_evaluators_[target_index], local_for_node(leaf_index), output);
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
      if (p2p_tensor_dictionary_plan_.has_value()) {
        apply_static_p2p_tensor_dictionary_plan(
            *p2p_tensor_dictionary_plan_, sorted_dipole_moments_,
            near_fields_, sorted_self_indices_);
      } else {
        detail::evaluate_static_near_field(p2p_compact_plan_,
                                           sorted_dipole_moments_, near_fields_,
                                           sorted_self_indices_);
      }
      last_timings_.p2p.add(elapsed_seconds(p2p_start));
    }
    for (std::size_t target = 0; target < target_count; ++target) {
      sorted_results_[target].H += near_fields_[target];
    }
  }

  if (periodic_.enabled && has_flag(output, OutputFlags::Potential)) {
    const StaticP2PCompactPlan& plan = p2p_compact_plan_;
#pragma omp parallel for schedule(static) if (target_count >= 64)
    for (std::ptrdiff_t target = 0;
         target < static_cast<std::ptrdiff_t>(target_count); ++target) {
      double potential = 0.0;
      const int self = sorted_self_indices_[static_cast<std::size_t>(target)];
      const int begin = plan.row_offsets[static_cast<std::size_t>(target)];
      const int end = plan.row_offsets[static_cast<std::size_t>(target) + 1];
      for (int entry = begin; entry < end; ++entry) {
        const std::size_t index = static_cast<std::size_t>(entry);
        const int source = plan.source_indices[index];
        if (plan.skip_for_identity[index] != 0 && source == self) {
          continue;
        }
        const Vec3 moment =
            sorted_dipole_moments_[static_cast<std::size_t>(source)];
        potential += plan.potential[0][index] * moment.x +
            plan.potential[1][index] * moment.y +
            plan.potential[2][index] * moment.z;
      }
      sorted_results_[static_cast<std::size_t>(target)].phi += potential;
    }
  }

  phase_start = Clock::now();
  detail::ProfileRange output_range{"cdfmm/output_permutation"};
  const OutputFlags reference_near_output =
      periodic_.enabled
          ? OutputFlags::None
          : execution_plan().p2p == StaticOperatorExecutor::Reference
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
    results[static_cast<std::size_t>(original_index)].phi *=
        coordinate_scale_;
  }
  last_timings_.result_unpermutation.add(elapsed_seconds(phase_start));
  last_timings_.total.add(elapsed_seconds(evaluation_start));
  last_timings_.evaluations = 1;
  accumulate_timings(aggregate_timings_, last_timings_);
}

std::vector<FloatPotentialField>
UniformFmm::evaluate_float32(const std::span<const Vec3> dipole_moments,
                             const OutputFlags output,
    const std::span<const int> target_source_indices) {
  if (precision_ != StaticPrecision::Float32) {
    throw std::logic_error("evaluate_float32 requires an FP32 FMM plan");
  }
  std::vector<FloatPotentialField> results(
      tree_.sorted_target_positions().size());
  evaluate_into_float32(dipole_moments, results, output, target_source_indices);
  return results;
}

std::vector<PotentialField>
UniformFmm::evaluate_float64(const std::span<const Vec3> dipole_moments,
                             const OutputFlags output,
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
  evaluate_into_float32_impl(dipole_moments, results, output,
                             target_source_indices);
}

void UniformFmm::evaluate_into_float32(
    const std::span<const FloatVec3> dipole_moments,
    const std::span<FloatPotentialField> results, const OutputFlags output,
    std::span<const int> target_source_indices) {
  evaluate_into_float32_impl(dipole_moments, results, output,
                             target_source_indices);
}

template <typename Moment>
void UniformFmm::evaluate_into_float32_impl(
    const std::span<const Moment> dipole_moments,
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
    for (std::size_t index = 0; index < dipole_moments.size(); ++index) {
      const Moment value = dipole_moments[index];
      using Scalar = decltype(value.x);
      const Scalar scale = static_cast<Scalar>(coordinate_scale_);
      sorted_dipole_moments_float_[index] = {
          static_cast<float>(value.x / scale / scale / scale),
          static_cast<float>(value.y / scale / scale / scale),
          static_cast<float>(value.z / scale / scale / scale)};
    }
    cuda_full_plan_->plan->evaluate(sorted_dipole_moments_float_,
                                    near_fields_float_, sorted_self_indices_);
    for (std::size_t target = 0; target < target_count; ++target) {
      results[target].phi = 0.0F;
      results[target].H = near_fields_float_[target];
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
      sorted_results_float_[target_index] =
          apply_static_l2p_evaluator(l2p_evaluators_float_[target_index],
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
      if (p2p_tensor_dictionary_plan_float_.has_value()) {
        apply_static_p2p_tensor_dictionary_plan(
            *p2p_tensor_dictionary_plan_float_, sorted_dipole_moments_float_,
            near_fields_float_, sorted_self_indices_);
      } else {
        apply_static_p2p_compact_plan(p2p_compact_plan_float_,
                                      sorted_dipole_moments_float_,
                                      near_fields_float_, sorted_self_indices_);
      }
    }
    for (std::size_t target = 0; target < target_count; ++target) {
      sorted_results_float_[target].H += near_fields_float_[target];
    }
    last_timings_.p2p.add(elapsed_seconds(phase_start));
  }

  if (has_flag(output, OutputFlags::Potential)) {
    const bool has_compact_plan =
        !p2p_compact_plan_float_.row_offsets.empty();
#pragma omp parallel for schedule(static) if (target_count >= 64)
    for (std::ptrdiff_t target = 0;
         target < static_cast<std::ptrdiff_t>(target_count); ++target) {
      float potential = 0.0F;
      const int self = sorted_self_indices_[static_cast<std::size_t>(target)];
      const auto& row_offsets = has_compact_plan
          ? p2p_compact_plan_float_.row_offsets
          : p2p_operator_float_.row_offsets;
      const int begin = row_offsets[static_cast<std::size_t>(target)];
      const int end = row_offsets[static_cast<std::size_t>(target) + 1];
      for (int entry = begin; entry < end; ++entry) {
        const std::size_t index = static_cast<std::size_t>(entry);
        const FloatStaticDipoleBlock* block = has_compact_plan
            ? nullptr
            : &p2p_operator_float_.blocks[index];
        const int source = has_compact_plan
            ? p2p_compact_plan_float_.source_indices[index]
            : block->source;
        const bool skip_for_identity = has_compact_plan
            ? p2p_compact_plan_float_.skip_for_identity[index] != 0
            : block->skip_for_identity != 0;
        if (skip_for_identity && source == self) {
          continue;
        }
        const FloatVec3 moment =
            sorted_dipole_moments_float_[static_cast<std::size_t>(source)];
        if (has_compact_plan) {
          potential += p2p_compact_plan_float_.potential[0][index] * moment.x +
              p2p_compact_plan_float_.potential[1][index] * moment.y +
              p2p_compact_plan_float_.potential[2][index] * moment.z;
        } else {
          potential += block->px * moment.x + block->py * moment.y +
              block->pz * moment.z;
        }
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
        static_cast<float>(coordinate_scale_);
  }
  last_timings_.result_unpermutation.add(elapsed_seconds(phase_start));
  last_timings_.total.add(elapsed_seconds(evaluation_start));
  last_timings_.evaluations = 1;
  accumulate_timings(aggregate_timings_, last_timings_);
}

//------------------------------------------------------------------------------
// Public inspection
//------------------------------------------------------------------------------

const UniformTree &UniformFmm::tree() const { return physical_tree_; }
const Vec3& UniformFmm::physical_root_centre() const noexcept {
  return physical_root_centre_;
}
double UniformFmm::physical_root_side_length() const noexcept {
  return physical_root_side_length_;
}
const PeriodicCellOptions& UniformFmm::periodic_cell() const noexcept
{
  return physical_periodic_;
}
const MultiIndexSet &UniformFmm::basis() const {
  if (expansion_basis_ != ExpansionBasis::Cartesian) {
    throw std::logic_error(
        "basis() is only available for Cartesian expansion plans");
  }
  return basis_;
}
const SphericalHarmonicBasis& UniformFmm::spherical_basis() const {
  if (expansion_basis_ != ExpansionBasis::Spherical) {
    throw std::logic_error(
        "spherical_basis() requires a spherical expansion plan");
  }
  return spherical_basis_;
}
ExpansionBasis UniformFmm::expansion_basis() const noexcept {
  return expansion_basis_;
}
int UniformFmm::expansion_order() const noexcept { return basis_.order(); }
int UniformFmm::coefficient_count() const noexcept {
  return expansion_basis_ == ExpansionBasis::Spherical ? spherical_basis_.size()
      : basis_.size();
}
SphericalM2LBackend UniformFmm::spherical_m2l_backend() const noexcept {
  return spherical_m2l_backend_;
}
int UniformFmm::coefficient_degree(const int coefficient) const {
  return expansion_basis_ == ExpansionBasis::Spherical
      ? spherical_basis_[coefficient].l
      : basis_[coefficient].degree();
}
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
  inspection_widening_buffer_.resize(
      static_cast<std::size_t>(coefficient_count()));
  for (int coefficient = 0; coefficient < coefficient_count(); ++coefficient) {
    const double value = precision_ == StaticPrecision::Float64
        ? multipole_for_node(node_index)[static_cast<std::size_t>(coefficient)]
        : static_cast<double>(multipole_float_for_node(node_index)[
              static_cast<std::size_t>(coefficient)]);
    inspection_widening_buffer_[static_cast<std::size_t>(coefficient)] =
        value * std::pow(coordinate_scale_, coefficient_degree(coefficient) + 2);
  }
  return inspection_widening_buffer_;
}

std::span<const double> UniformFmm::local(const int node_index) const {
  if (node_index < 0 ||
      static_cast<std::size_t>(node_index) >= tree_.nodes().size()) {
    throw std::out_of_range("local node index is out of range");
  }
  inspection_widening_buffer_.resize(
      static_cast<std::size_t>(coefficient_count()));
  for (int coefficient = 0; coefficient < coefficient_count(); ++coefficient) {
    const double value = precision_ == StaticPrecision::Float64
        ? local_for_node(node_index)[static_cast<std::size_t>(coefficient)]
        : static_cast<double>(local_float_for_node(node_index)[
              static_cast<std::size_t>(coefficient)]);
    inspection_widening_buffer_[static_cast<std::size_t>(coefficient)] =
        value * std::pow(coordinate_scale_,
                         1 - coefficient_degree(coefficient));
  }
  return inspection_widening_buffer_;
}

std::span<const double> UniformFmm::root_multipole() const {
  return multipole(0);
}

std::span<const float>
UniformFmm::multipole_float32(const int node_index) const {
  if (precision_ != StaticPrecision::Float32) {
    throw std::logic_error("multipole_float32 requires an FP32 FMM plan");
  }
  if (node_index < 0 ||
      static_cast<std::size_t>(node_index) >= tree_.nodes().size()) {
    throw std::out_of_range("multipole node index is out of range");
  }
  const auto values = multipole_float_for_node(node_index);
  float_inspection_buffer_.resize(values.size());
  for (int coefficient = 0; coefficient < coefficient_count(); ++coefficient) {
    float_inspection_buffer_[static_cast<std::size_t>(coefficient)] =
        values[static_cast<std::size_t>(coefficient)] *
        static_cast<float>(std::pow(
            coordinate_scale_, coefficient_degree(coefficient) + 2));
  }
  return float_inspection_buffer_;
}

std::span<const float> UniformFmm::local_float32(const int node_index) const {
  if (precision_ != StaticPrecision::Float32) {
    throw std::logic_error("local_float32 requires an FP32 FMM plan");
  }
  if (node_index < 0 ||
      static_cast<std::size_t>(node_index) >= tree_.nodes().size()) {
    throw std::out_of_range("local node index is out of range");
  }
  const auto values = local_float_for_node(node_index);
  float_inspection_buffer_.resize(values.size());
  for (int coefficient = 0; coefficient < coefficient_count(); ++coefficient) {
    float_inspection_buffer_[static_cast<std::size_t>(coefficient)] =
        values[static_cast<std::size_t>(coefficient)] *
        static_cast<float>(std::pow(
            coordinate_scale_, 1 - coefficient_degree(coefficient)));
  }
  return float_inspection_buffer_;
}

std::span<const float> UniformFmm::root_multipole_float32() const {
  return multipole_float32(0);
}

std::span<const double>
UniformFmm::multipole_float64(const int node_index) const {
  if (precision_ != StaticPrecision::Float64) {
    throw std::logic_error("multipole_float64 requires an FP64 FMM plan");
  }
  return multipole(node_index);
}

std::span<const double> UniformFmm::local_float64(const int node_index) const {
  if (precision_ != StaticPrecision::Float64) {
    throw std::logic_error("local_float64 requires an FP64 FMM plan");
  }
  return local(node_index);
}

std::span<const double> UniformFmm::root_multipole_float64() const {
  return multipole_float64(0);
}

std::span<double>
UniformFmm::multipole_for_node(const int node_index) noexcept {
  const std::size_t n = static_cast<std::size_t>(coefficient_count());
  return {multipoles_.data() + static_cast<std::size_t>(node_index) * n, n};
}

std::span<const double>
UniformFmm::multipole_for_node(const int node_index) const noexcept {
  const std::size_t n = static_cast<std::size_t>(coefficient_count());
  return {multipoles_.data() + static_cast<std::size_t>(node_index) * n, n};
}

std::span<double> UniformFmm::local_for_node(const int node_index) noexcept {
  const std::size_t n = static_cast<std::size_t>(coefficient_count());
  return {locals_.data() + static_cast<std::size_t>(node_index) * n, n};
}

std::span<const double>
UniformFmm::local_for_node(const int node_index) const noexcept {
  const std::size_t n = static_cast<std::size_t>(coefficient_count());
  return {locals_.data() + static_cast<std::size_t>(node_index) * n, n};
}

std::span<float>
UniformFmm::multipole_float_for_node(const int node_index) noexcept {
  const std::size_t n = static_cast<std::size_t>(coefficient_count());
  return {multipoles_float_.data() + static_cast<std::size_t>(node_index) * n,
          n};
}

std::span<const float>
UniformFmm::multipole_float_for_node(const int node_index) const noexcept {
  const std::size_t n = static_cast<std::size_t>(coefficient_count());
  return {multipoles_float_.data() + static_cast<std::size_t>(node_index) * n,
          n};
}

std::span<float>
UniformFmm::local_float_for_node(const int node_index) noexcept {
  const std::size_t n = static_cast<std::size_t>(coefficient_count());
  return {locals_float_.data() + static_cast<std::size_t>(node_index) * n, n};
}

std::span<const float>
UniformFmm::local_float_for_node(const int node_index) const noexcept {
  const std::size_t n = static_cast<std::size_t>(coefficient_count());
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
