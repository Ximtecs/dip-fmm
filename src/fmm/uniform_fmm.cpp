// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/uniform_fmm.hpp"

#include <cmath>
#include <stdexcept>

#include "backend/cuda/common/runtime.hpp"
#include "fmm/internal.hpp"

namespace cdfmm {

const UniformTree &UniformFmm::tree() const {
  if (!physical_tree_) throw std::logic_error("prebuilt evaluator exposes topology(), not a uniform tree");
  return *physical_tree_;
}
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
SpatialLayout UniformFmm::spatial_layout() const noexcept {
  return spatial_layout_;
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

const std::string& UniformFmm::universal_cache_key() const noexcept {
  return universal_cache_key_;
}

const std::string& UniformFmm::geometry_cache_key() const noexcept {
  return geometry_cache_key_;
}

const std::string& UniformFmm::periodic_cache_key() const noexcept {
  return periodic_cache_key_;
}

std::span<const double> UniformFmm::multipole(const int node_index) const {
  if (node_index < 0 ||
      static_cast<std::size_t>(node_index) >= topology_->nodes.size()) {
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
      static_cast<std::size_t>(node_index) >= topology_->nodes.size()) {
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
  return multipole(topology_->root);
}

std::span<const float>
UniformFmm::multipole_float32(const int node_index) const {
  if (precision_ != StaticPrecision::Float32) {
    throw std::logic_error("multipole_float32 requires an FP32 FMM plan");
  }
  if (node_index < 0 ||
      static_cast<std::size_t>(node_index) >= topology_->nodes.size()) {
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
      static_cast<std::size_t>(node_index) >= topology_->nodes.size()) {
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
  return multipole_float32(topology_->root);
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
  return multipole_float64(topology_->root);
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

bool cuda_available() noexcept { return cuda_runtime_available(); }

std::string cuda_device_description() { return cuda_runtime_description(); }

} // namespace cdfmm
