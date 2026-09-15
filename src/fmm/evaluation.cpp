// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/uniform_fmm.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <type_traits>

#include "cdfmm/backend/cpu/p2p.hpp"
#include "cdfmm/backend/cuda/p2p.hpp"
#include "cdfmm/operators.hpp"

#include "backend/cpu/p2p/near_field.hpp"
#include "fmm/internal.hpp"
#include "profile.hpp"

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

std::vector<PotentialField>
UniformFmm::evaluate(std::span<const Vec3> dipole_moments,
                     const OutputFlags output,
                     std::span<const int> target_source_indices) {
  std::vector<PotentialField> results(topology_->sorted_target_positions.size());
  evaluate_into(dipole_moments, results, output, target_source_indices);
  return results;
}

void UniformFmm::prepare_self_indices(
    const std::span<const int> target_source_indices) {
  const auto target_permutation =
      std::span<const int>(topology_->target_permutation);
  const auto source_inverse =
      std::span<const int>(topology_->source_inverse_permutation);
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
  const bool effective_point_source =
      source_geometry_ == SourceGeometry::PointDipole ||
      near_field_source_model_ == SourceModel::PointDipole;
  if (!effective_point_source) {
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
  const bool effective_finite_source =
      source_geometry_ != SourceGeometry::PointDipole &&
      near_field_source_model_ == SourceModel::ExactGeometry;
  const bool effective_finite_target =
      target_geometry_ != TargetGeometry::Point &&
      near_field_target_model_ == TargetModel::ExactGeometry;
  if (has_flag(output, OutputFlags::Potential) &&
      (effective_finite_source || effective_finite_target)) {
    throw std::invalid_argument(
        "near-field potential is unsupported for exact finite geometry; "
        "request field only or select point near-field models");
  }
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
  const std::size_t target_count = topology_->sorted_target_positions.size();
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
    if (capture_components_) {
      std::vector<Vec3> sorted_far(target_count);
      cuda_full_plan_->plan->copy_far_fields(sorted_far);
      diagnostic_far_.resize(target_count);
      for (std::size_t i = 0; i < target_count; ++i) {
        diagnostic_far_[topology_->target_permutation[i]] = sorted_far[i];
      }
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
    downward_pass_for_output(output, true);
  }
  const auto target_permutation =
      std::span<const int>(topology_->target_permutation);
  auto phase_start = Clock::now();

  if (capture_components_) {
    diagnostic_far_.resize(target_count);
    for (std::size_t i = 0; i < target_count; ++i) {
      diagnostic_far_[topology_->target_permutation[i]] = sorted_results_[i].H;
    }
  }

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
        apply_static_p2p_signed_tensor_dictionary_plan(
            *p2p_tensor_dictionary_plan_, sorted_dipole_moments_,
            near_fields_);
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
        *topology_, sorted_dipole_moments_, sorted_self_indices_,
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
      topology_->sorted_target_positions.size());
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
  const bool effective_finite_source =
      source_geometry_ != SourceGeometry::PointDipole &&
      near_field_source_model_ == SourceModel::ExactGeometry;
  const bool effective_finite_target =
      target_geometry_ != TargetGeometry::Point &&
      near_field_target_model_ == TargetModel::ExactGeometry;
  if (has_flag(output, OutputFlags::Potential) &&
      (effective_finite_source || effective_finite_target)) {
    throw std::invalid_argument(
        "near-field potential is unsupported for exact finite geometry; "
        "request field only or select point near-field models");
  }
  if (precision_ != StaticPrecision::Float32) {
    throw std::logic_error("evaluate_into_float32 requires an FP32 FMM plan");
  }
  const std::size_t target_count = topology_->sorted_target_positions.size();
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
    if (capture_components_) {
      std::vector<Vec3> sorted_far(target_count);
      cuda_full_plan_->plan->copy_far_fields(sorted_far);
      diagnostic_far_.resize(target_count);
      for (std::size_t i = 0; i < target_count; ++i) {
        diagnostic_far_[topology_->target_permutation[i]] = sorted_far[i];
      }
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
  downward_pass_float_for_output(output, true);

  const auto target_permutation =
      std::span<const int>(topology_->target_permutation);
  auto phase_start = Clock::now();

  if (has_flag(output, OutputFlags::Field)) {
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
        apply_static_p2p_signed_tensor_dictionary_plan(
            *p2p_tensor_dictionary_plan_float_, sorted_dipole_moments_float_,
            near_fields_float_);
      } else {
        apply_static_p2p_compact_plan(p2p_compact_plan_float_,
                                      sorted_dipole_moments_float_,
                                      near_fields_float_, sorted_self_indices_);
      }
    }
    if (capture_components_) {
      diagnostic_far_.resize(target_count);
      for (std::size_t i = 0; i < target_count; ++i) {
        const auto h = sorted_results_float_[i].H;
        diagnostic_far_[topology_->target_permutation[i]] = {h.x, h.y, h.z};
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

UniformFmm::FieldComponents UniformFmm::evaluate_components(
    std::span<const Vec3> moments, std::span<const int> identities) {
  capture_components_ = true;
  std::vector<PotentialField> fields;
  try {
    fields = evaluate(moments, OutputFlags::Field, identities);
  } catch (...) {
    capture_components_ = false;
    throw;
  }
  capture_components_ = false;
  FieldComponents result;
  result.far = diagnostic_far_;
  result.p2p.resize(fields.size());
  result.total.resize(fields.size());
  for (std::size_t i = 0; i < fields.size(); ++i) {
    result.total[i] = fields[i].H;
    result.p2p[i] = result.total[i] - result.far[i];
  }
  return result;
}

} // namespace cdfmm
