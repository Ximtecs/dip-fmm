// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/uniform_fmm.hpp"

#include <algorithm>
#include <cmath>

#ifdef CDFMM_USE_OPENMP
#include <omp.h>
#endif

#include "cdfmm/operators/operators.hpp"
#include "cdfmm/backend/cpu/far_field.hpp"
#include "cdfmm/backend/cpu/m2l.hpp"
#include "cdfmm/backend/cuda/m2l.hpp"
#include "backend/cpu/far_field/internal.hpp"
#include "backend/cpu/far_field/packing.hpp"
#include "backend/cpu/m2l/schedule.hpp"
#include "fmm/internal.hpp"
#include "phase_stopwatch.hpp"
#include "profile.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Far-field evaluation
//------------------------------------------------------------------------------
// This file owns the execution of the expansion hierarchy:
//
//   sources -> P2M -> M2M -> M2L -> L2L -> L2P -> far field
//
// Geometry-plan construction and list1 P2P do not belong here.  The static
// executors consume the CPU far-field packing derived once at construction
// (backend/cpu/far_field/packing.hpp); the reference executors consume the
// canonical operators directly.  Nothing here rebuilds plans.  Near-field work
// may therefore proceed independently while this branch consumes explicit
// topology schedules.
//
// Each stage exists in an FP64 and an FP32 form over the corresponding plan
// and state; the FP32 forms omit the `Reference` executor because the
// dynamic Cartesian reference is FP64 only.  Schedules are level barriers:
// M2M runs deep to shallow, L2L and M2L shallow to deep, and within a level
// every parallel iteration owns a disjoint output node, so no stage needs
// atomics.  Coefficients live in flat node-major arrays
// (`multipole_for_node`, `local_for_node` are views into them).
//
// Every phase clock here is a `Detailed`-level stopwatch: below that level
// `start()`/`record()` are one predictable branch and no clock is read.  The
// clocks bracket whole OpenMP regions, so they report caller wall time.

// One level of the static M2L.  Portable execution prefers the transfer-class
// sorted block schedule (backend/cpu/m2l/schedule.cpp) and falls back to the
// per-target-row executor when the plan is too large for it; oneMKL groups
// interactions by class into GEMMs and reports its gather/multiply/scatter
// split.  Level 0 is only reached by a periodic plan, where the "M2L" of the
// root onto itself is the periodic image operator.
void UniformFmm::static_m2l(const int level) {
  detail::ProfileRange m2l_range{"cdfmm/far_field/m2l"};
  detail::PhaseStopwatch detailed(detailed_timing());
  if (static_matrix_backend_ == StaticMatrixBackend::Portable) {
    detailed.start();
    if (cpu_packing_ && !cpu_packing_->m2l_schedule.empty()) {
      detail::cpu::apply_static_m2l_plan(m2l_plan_, cpu_packing_->m2l_schedule,
                                         level, multipoles_, locals_);
    } else {
      apply_static_m2l_plan(m2l_plan_, level, multipoles_, locals_);
    }
    detailed.record(last_timings_.m2l_multiply);
    return;
  }
  const detail::mkl::M2LApplyTimings timings = mkl_m2l_plan_->apply(
      m2l_plan_, level, multipoles_, locals_, detailed.enabled());
  if (detailed.enabled()) {
    last_timings_.m2l_gather.add(timings.gather_seconds);
    last_timings_.m2l_multiply.add(timings.multiply_seconds);
    last_timings_.m2l_scatter.add(timings.scatter_seconds);
  }
}

void UniformFmm::static_m2l_float(const int level) {
  detail::ProfileRange m2l_range{"cdfmm/far_field/m2l_fp32"};
  detail::PhaseStopwatch detailed(detailed_timing());
  if (static_matrix_backend_ == StaticMatrixBackend::Portable) {
    detailed.start();
    if (cpu_packing_ && !cpu_packing_->m2l_schedule.empty()) {
      detail::cpu::apply_static_m2l_plan(
          m2l_plan_float_, cpu_packing_->m2l_schedule, level,
          multipoles_float_, locals_float_);
    } else {
      apply_static_m2l_plan(m2l_plan_float_, level, multipoles_float_,
                            locals_float_);
    }
    detailed.record(last_timings_.m2l_multiply);
    return;
  }

  const detail::mkl::M2LApplyTimings timings = mkl_m2l_plan_->apply(
      m2l_plan_float_, level, multipoles_float_, locals_float_,
      detailed.enabled());
  if (detailed.enabled()) {
    last_timings_.m2l_gather.add(timings.gather_seconds);
    last_timings_.m2l_multiply.add(timings.multiply_seconds);
    last_timings_.m2l_scatter.add(timings.scatter_seconds);
  }
}

//------------------------------------------------------------------------------
// Upward pass
//------------------------------------------------------------------------------

void UniformFmm::upward_pass(std::span<const Vec3> dipole_moments) {
  if (precision_ == StaticPrecision::Float32) {
    prepare_moments_float(dipole_moments);
    upward_pass_prepared_float();
    return;
  }
  prepare_moments(dipole_moments);
  upward_pass_prepared();
}

// Moment preparation: permute into leaf order and apply the s^-3 length
// scaling that makes the normalised-space field equal the physical field
// (see the evaluation unit).  The FP32 forms convert once here; the
// FP64-input form divides in FP64 before narrowing.
void UniformFmm::prepare_moments_float(
    const std::span<const Vec3> dipole_moments) {
  detail::ProfileRange input_range{"cdfmm/input_preparation_fp32"};
  if (dipole_moments.size() != topology_->sorted_source_positions.size()) {
    throw std::invalid_argument(
        "UniformFmm::upward_pass requires one dipole moment per source "
        "position");
  }
  detail::PhaseStopwatch detailed(detailed_timing());
  detailed.start();
  std::fill(multipoles_float_.begin(), multipoles_float_.end(), 0.0F);
  detailed.record(last_timings_.multipole_reset);

  detailed.start();
  const auto permutation = std::span<const int>(topology_->source_permutation);
#pragma omp parallel for schedule(static) if (permutation.size() >= 256)
  for (std::ptrdiff_t sorted_index = 0;
       sorted_index < static_cast<std::ptrdiff_t>(permutation.size());
       ++sorted_index) {
    const Vec3 moment = dipole_moments[static_cast<std::size_t>(
        permutation[static_cast<std::size_t>(sorted_index)])];
    const double scale = coordinate_scale_;
    sorted_dipole_moments_float_[static_cast<std::size_t>(sorted_index)] = {
        static_cast<float>(moment.x / scale / scale / scale),
        static_cast<float>(moment.y / scale / scale / scale),
        static_cast<float>(moment.z / scale / scale / scale)};
  }
  detailed.record(last_timings_.moment_permutation);
}

void UniformFmm::prepare_moments_float(
    const std::span<const FloatVec3> dipole_moments) {
  detail::ProfileRange input_range{"cdfmm/input_preparation_fp32"};
  if (dipole_moments.size() != topology_->sorted_source_positions.size()) {
    throw std::invalid_argument(
        "UniformFmm::upward_pass requires one dipole moment per source position");
  }
  detail::PhaseStopwatch detailed(detailed_timing());
  detailed.start();
  std::fill(multipoles_float_.begin(), multipoles_float_.end(), 0.0F);
  detailed.record(last_timings_.multipole_reset);
  detailed.start();
  const auto permutation = std::span<const int>(topology_->source_permutation);
  const float scale = static_cast<float>(coordinate_scale_);
#pragma omp parallel for schedule(static) if (permutation.size() >= 256)
  for (std::ptrdiff_t sorted_index = 0;
       sorted_index < static_cast<std::ptrdiff_t>(permutation.size());
       ++sorted_index) {
    const FloatVec3 moment = dipole_moments[static_cast<std::size_t>(
        permutation[static_cast<std::size_t>(sorted_index)])];
    sorted_dipole_moments_float_[static_cast<std::size_t>(sorted_index)] = {
        moment.x / scale / scale / scale,
        moment.y / scale / scale / scale,
        moment.z / scale / scale / scale};
  }
  detailed.record(last_timings_.moment_permutation);
}

void UniformFmm::upward_pass_prepared_float() {
  const auto &nodes = topology_->nodes;
  const auto &occupied_leaves = topology_->source_leaves;
  detail::PhaseStopwatch detailed(detailed_timing());
  detailed.start();
#pragma omp parallel for schedule(static) if (occupied_leaves.size() >= 8)
  for (std::ptrdiff_t occupied_index = 0;
       occupied_index < static_cast<std::ptrdiff_t>(occupied_leaves.size());
       ++occupied_index) {
    const int leaf_index =
        occupied_leaves[static_cast<std::size_t>(occupied_index)].node;
    const StaticLeafRange &leaf_range =
        occupied_leaves[static_cast<std::size_t>(occupied_index)];
    const auto M = multipole_float_for_node(leaf_index);
    const std::span<const FloatVec3> leaf_moments =
        std::span<const FloatVec3>(sorted_dipole_moments_float_)
            .subspan(leaf_range.begin, leaf_range.count);
    if (procedural_p2m_) {
      cpu_packing_->procedural_fp32.apply_p2m(
          nodes[static_cast<std::size_t>(leaf_index)].centre,
          std::span<const Vec3>(topology_->sorted_source_positions)
              .subspan(leaf_range.begin, leaf_range.count),
          leaf_moments, M.data());
    } else {
      detail::cpu::apply_packed_p2m(cpu_packing_->fp32.p2m, leaf_range.begin,
                                    leaf_moments, M.data());
    }
  }
  detailed.record(last_timings_.p2m);

  detailed.start();
#pragma omp parallel if (nodes.size() >= 64)
  {
    for (int level = topology_->maximum_level - 1; level >= 0; --level) {
      const int begin = topology_->m2m_parent_level_offsets[static_cast<std::size_t>(level + 1)];
      const int end = topology_->m2m_parent_level_offsets[static_cast<std::size_t>(level + 2)];
#pragma omp for schedule(static)
      for (int parent_slot = begin; parent_slot < end; ++parent_slot) {
        const int parent_index = topology_->m2m_parent_nodes[static_cast<std::size_t>(parent_slot)];
        const auto &parent = nodes[static_cast<std::size_t>(parent_index)];
        if (parent.source_count() == 0) {
          continue;
        }
        const auto parent_M = multipole_float_for_node(parent_index);
        const int edge_begin = topology_->m2m_parent_edge_offsets[
            static_cast<std::size_t>(parent_slot)];
        const int edge_end = topology_->m2m_parent_edge_offsets[
            static_cast<std::size_t>(parent_slot + 1)];
        for (int edge_slot = edge_begin; edge_slot < edge_end; ++edge_slot) {
          const StaticTranslationEdge &edge = topology_->m2m_edges[
              static_cast<std::size_t>(edge_slot)];
          const int child_index = edge.source_node;
          const auto &child = nodes[static_cast<std::size_t>(child_index)];
          if (child.source_count() == 0) {
            continue;
          }
          detail::cpu::apply_packed_translation(
              cpu_packing_->fp32.m2m, child.level, edge.child_class,
              multipole_float_for_node(child_index).data(), parent_M.data());
        }
      }
    }
  }
  detailed.record(last_timings_.m2m);
}

void UniformFmm::prepare_moments(std::span<const Vec3> dipole_moments) {
  detail::ProfileRange input_range{"cdfmm/input_preparation"};
  if (dipole_moments.size() != topology_->sorted_source_positions.size()) {
    throw std::invalid_argument(
        "UniformFmm::upward_pass requires one dipole moment per source "
        "position");
  }

  detail::PhaseStopwatch detailed(detailed_timing());
  {
    detailed.start();
    detail::ProfileRange reset_range{"cdfmm/far_field/multipole_reset"};
    // Flat node-major storage permits one streaming reset instead of one small
    // fill and one vector-metadata load per tree node.
    std::fill(multipoles_.begin(), multipoles_.end(), 0.0);
    detailed.record(last_timings_.multipole_reset);
  }

  detail::ProfileRange permutation_range{
      "cdfmm/input_preparation/moment_permutation"};
  const auto permutation = std::span<const int>(topology_->source_permutation);
  const double scale = coordinate_scale_;
  const double inverse_volume_scale = 1.0 / (scale * scale * scale);
#pragma omp parallel for schedule(static) if (permutation.size() >= 256)
  for (std::ptrdiff_t sorted_index = 0;
       sorted_index < static_cast<std::ptrdiff_t>(permutation.size());
       ++sorted_index) {
    const int original_index =
        permutation[static_cast<std::size_t>(sorted_index)];
    sorted_dipole_moments_[static_cast<std::size_t>(sorted_index)] =
        dipole_moments[static_cast<std::size_t>(original_index)] *
        inverse_volume_scale;
  }
  detailed.record(last_timings_.moment_permutation);
}

void UniformFmm::upward_pass_prepared() {
  const auto &nodes = topology_->nodes;
  const auto &occupied_leaves = topology_->source_leaves;
  const StaticOperatorExecutor p2m_executor = execution_plan().p2m;
  detail::PhaseStopwatch detailed(detailed_timing());
  detailed.start();
  detail::ProfileRange p2m_range{"cdfmm/far_field/p2m"};
#pragma omp parallel for schedule(static) if (occupied_leaves.size() >= 8)
  for (std::ptrdiff_t occupied_index = 0;
       occupied_index < static_cast<std::ptrdiff_t>(occupied_leaves.size());
       ++occupied_index) {
    // Occupied leaves have disjoint multipole vectors. Each worker therefore
    // owns its output and no synchronisation is needed inside P2M.
    const int leaf_index =
        occupied_leaves[static_cast<std::size_t>(occupied_index)].node;
    const StaticLeafRange &leaf_range =
        occupied_leaves[static_cast<std::size_t>(occupied_index)];
    const auto &leaf = nodes[static_cast<std::size_t>(leaf_index)];
    if (p2m_executor != StaticOperatorExecutor::Reference) {
      const auto M = multipole_for_node(leaf_index);
      const std::span<const Vec3> leaf_moments =
          std::span<const Vec3>(sorted_dipole_moments_)
              .subspan(leaf_range.begin, leaf_range.count);
      if (procedural_p2m_) {
        cpu_packing_->procedural_fp64.apply_p2m(
            leaf.centre,
            std::span<const Vec3>(topology_->sorted_source_positions)
                .subspan(leaf_range.begin, leaf_range.count),
            leaf_moments, M.data());
      } else {
        detail::cpu::apply_packed_p2m(cpu_packing_->fp64.p2m,
                                      leaf_range.begin, leaf_moments,
                                      M.data());
      }
    } else {
      const CoeffVector M = operators::p2m::evaluate(basis_, leaf.centre,
                     std::span<const Vec3>(topology_->sorted_source_positions)
                         .subspan(leaf_range.begin, leaf_range.count),
                     std::span<const Vec3>(sorted_dipole_moments_)
                         .subspan(leaf_range.begin, leaf_range.count));
      std::copy(M.begin(), M.end(), multipole_for_node(leaf_index).begin());
    }
  }
  detailed.record(last_timings_.p2m);
  p2m_range.end();

  detailed.start();
  detail::ProfileRange m2m_range{"cdfmm/far_field/m2m"};
  const StaticOperatorExecutor m2m_executor = execution_plan().m2m;
// One team traverses all dependent levels; the implicit omp-for barrier
// makes each parent level complete before its parent is consumed.
#pragma omp parallel if (nodes.size() >= 64)
  {
    for (int level = topology_->maximum_level - 1; level >= 0; --level) {
      const int begin = topology_->m2m_parent_level_offsets[static_cast<std::size_t>(level + 1)];
      const int end = topology_->m2m_parent_level_offsets[static_cast<std::size_t>(level + 2)];
#pragma omp for schedule(static)
      for (int parent_slot = begin; parent_slot < end; ++parent_slot) {
        const int parent_index = topology_->m2m_parent_nodes[static_cast<std::size_t>(parent_slot)];
        const auto &parent = nodes[static_cast<std::size_t>(parent_index)];
        if (parent.source_count() == 0) {
          continue;
        }
        const auto parent_M = multipole_for_node(parent_index);
        const int edge_begin = topology_->m2m_parent_edge_offsets[
            static_cast<std::size_t>(parent_slot)];
        const int edge_end = topology_->m2m_parent_edge_offsets[
            static_cast<std::size_t>(parent_slot + 1)];
        for (int edge_slot = edge_begin; edge_slot < edge_end; ++edge_slot) {
          const StaticTranslationEdge &edge = topology_->m2m_edges[
              static_cast<std::size_t>(edge_slot)];
          const int child_index = edge.source_node;
          const auto &child = nodes[static_cast<std::size_t>(child_index)];
          if (child.source_count() == 0) {
            continue;
          }
          if (m2m_executor != StaticOperatorExecutor::Reference) {
            detail::cpu::apply_packed_translation(
                cpu_packing_->fp64.m2m, child.level, edge.child_class,
                multipole_for_node(child_index).data(), parent_M.data());
          } else {
            const Vec3 d = parent.centre - child.centre;
            operators::m2m::apply(basis_, d,
                    multipole_for_node(child_index),
                    parent_M);
          }
        }
      }
    }
  }
  detailed.record(last_timings_.m2m);
}

//------------------------------------------------------------------------------
// Downward pass
//------------------------------------------------------------------------------

void UniformFmm::downward_pass() {
  if (precision_ == StaticPrecision::Float32) {
    downward_pass_float_for_output(OutputFlags::None, false);
    return;
  }
  downward_pass_for_output(OutputFlags::None, false);
}

// Downward pass: locals are reset, then per level L2L inherits the parent
// local before that level's M2L adds the list-2 contributions, and finally
// L2P evaluates the leaf locals at the targets.  With the CUDA M2L executor
// the whole M2L is one device call, so the CPU only runs L2L afterwards.
// `evaluate_l2p == false` leaves the locals in place for inspection.
void UniformFmm::downward_pass_for_output(const OutputFlags output,
                                          const bool evaluate_l2p) {
  detail::ProfileRange downward_range{"cdfmm/far_field/downward"};
  detail::PhaseStopwatch detailed(detailed_timing());
  detailed.start();
  detail::ProfileRange reset_range{"cdfmm/far_field/local_reset"};
  std::fill(locals_.begin(), locals_.end(), 0.0);
  detailed.record(last_timings_.local_reset);
  reset_range.end();

  if (execution_plan().m2l == StaticOperatorExecutor::Cuda) {
    cuda_m2l();
    l2l_downward();
  } else {
    if (periodic_.enabled && m2l_backend_ == M2LBackend::Static) {
      detailed.start();
      static_m2l(0);
      detailed.record(last_timings_.m2l);
    }

    const auto &nodes = topology_->nodes;
    const StaticOperatorExecutor l2l_executor = execution_plan().l2l;
    for (int level = 1; level <= topology_->maximum_level; ++level) {
      // Parent locals must be inherited before this level's M2L is added.
      // Advancing levels in order makes the parent-child dependency explicit.
      const int begin = topology_->l2l_level_offsets[static_cast<std::size_t>(level)];
      const int end = topology_->l2l_level_offsets[static_cast<std::size_t>(level + 1)];

      detailed.start();
      detail::ProfileRange l2l_range{"cdfmm/far_field/l2l"};
#pragma omp parallel for schedule(static) if (end - begin >= 8)
      for (int edge_slot = begin; edge_slot < end; ++edge_slot) {
        const StaticTranslationEdge &edge = topology_->l2l_edges[
            static_cast<std::size_t>(edge_slot)];
        const int target_index = edge.target_node;
        const auto &target = nodes[static_cast<std::size_t>(target_index)];
        if (target.target_count() == 0) {
          continue;
        }
        if (l2l_executor != StaticOperatorExecutor::Reference) {
          detail::cpu::apply_packed_translation(
              cpu_packing_->fp64.l2l, target.level, edge.child_class,
              local_for_node(edge.source_node).data(),
              local_for_node(target_index).data());
        } else {
          const Vec3 d = target.centre - nodes[static_cast<std::size_t>(edge.source_node)].centre;
          operators::l2l::apply(basis_, d, local_for_node(edge.source_node),
                  local_for_node(target_index));
        }
      }
      detailed.record(last_timings_.l2l);
      l2l_range.end();

      if (m2l_backend_ == M2LBackend::Static) {
        detailed.start();
        static_m2l(level);
        detailed.record(last_timings_.m2l);
        continue;
      }

      detailed.start();
      detail::ProfileRange m2l_range{"cdfmm/far_field/m2l"};
      // Reference M2L retains list2 order through the canonical interaction
      // records. It is intentionally serial to preserve each target's sum.
      for (const StaticM2LInteraction &interaction : topology_->m2l_interactions) {
        if (interaction.target_level != level) {
          continue;
        }
        const auto &target = nodes[static_cast<std::size_t>(interaction.target_node)];
        if (target.target_count() == 0) {
          continue;
        }
        const auto &source = nodes[static_cast<std::size_t>(interaction.source_node)];
        const Vec3 R = target.centre - source.centre - interaction.source_shift;
        operators::m2l::apply(basis_, R, multipole_for_node(interaction.source_node),
                local_for_node(interaction.target_node));
      }
      detailed.record(last_timings_.m2l);
    }
  }

  // Keep the coarse downward profile scoped to M2L/L2L; L2P has its own
  // hierarchy-stage range and timing below.
  downward_range.end();
  if (evaluate_l2p) {
    const auto &nodes = topology_->nodes;
    const auto targets = std::span<const Vec3>(topology_->sorted_target_positions);
    const auto &occupied_leaves = topology_->target_leaves;
    const StaticOperatorExecutor l2p_executor = execution_plan().l2p;
    const bool want_field = has_flag(output, OutputFlags::Field);
    const bool want_potential = has_flag(output, OutputFlags::Potential);
    detailed.start();
    detail::ProfileRange l2p_range{"cdfmm/far_field/l2p"};
#pragma omp parallel for schedule(static) if (occupied_leaves.size() >= 8)
    for (std::ptrdiff_t occupied_index = 0;
         occupied_index < static_cast<std::ptrdiff_t>(occupied_leaves.size());
         ++occupied_index) {
      const StaticLeafRange &leaf_range =
          occupied_leaves[static_cast<std::size_t>(occupied_index)];
      const int leaf_index = leaf_range.node;
      const auto &leaf = nodes[static_cast<std::size_t>(leaf_index)];
      const double *L = local_for_node(leaf_index).data();
      if (procedural_l2p_ &&
          l2p_executor != StaticOperatorExecutor::Reference) {
        cpu_packing_->procedural_fp64.apply_l2p(
            leaf.centre, targets.subspan(leaf_range.begin, leaf_range.count),
            L,
            std::span<PotentialField>(sorted_results_)
                .subspan(leaf_range.begin, leaf_range.count),
            want_field, want_potential);
        continue;
      }
      for (std::size_t target_index = leaf_range.begin;
           target_index < leaf_range.begin + leaf_range.count; ++target_index) {
        if (l2p_executor != StaticOperatorExecutor::Reference) {
          PotentialField result;
          if (want_field) {
            detail::cpu::apply_packed_l2p_field(
                cpu_packing_->fp64.l2p, target_index, L, result.H.x,
                result.H.y, result.H.z);
          }
          if (want_potential) {
            result.phi = detail::cpu::apply_packed_l2p_potential(
                cpu_packing_->fp64.l2p, target_index, L);
          }
          sorted_results_[target_index] = result;
        } else {
          sorted_results_[target_index] =
              operators::l2p::evaluate(basis_, leaf.centre, targets[target_index],
                       local_for_node(leaf_index), output);
        }
      }
    }
    detailed.record(last_timings_.l2p);
  }
}

// FP32 downward pass; the same sequence as above without the reference
// executor.
void UniformFmm::downward_pass_float_for_output(const OutputFlags output,
                                                const bool evaluate_l2p) {
  detail::PhaseStopwatch detailed(detailed_timing());
  detailed.start();
  std::fill(locals_float_.begin(), locals_float_.end(), 0.0F);
  detailed.record(last_timings_.local_reset);

  const bool cuda_m2l_executor =
      execution_plan().m2l == StaticOperatorExecutor::Cuda;
  if (cuda_m2l_executor) {
    detailed.start();
    cuda_m2l_plan_->plan->evaluate(multipoles_float_, locals_float_);
    if (detailed.enabled()) {
      const CudaEvaluationTimings &device = cuda_m2l_plan_->plan->timings();
      last_timings_.cuda_h2d.add(device.h2d_seconds);
      last_timings_.cuda_m2l_h2d.add(device.h2d_seconds);
      last_timings_.m2l_scale.add(device.scale_seconds);
      last_timings_.m2l_multiply.add(device.multiply_seconds);
      last_timings_.cuda_kernel.add(device.kernel_seconds);
      last_timings_.cuda_d2h.add(device.d2h_seconds);
      last_timings_.cuda_m2l_d2h.add(device.d2h_seconds);
    }
    detailed.record(last_timings_.m2l);
  }

  const auto &nodes = topology_->nodes;
  if (periodic_.enabled && !cuda_m2l_executor) {
    detailed.start();
    static_m2l_float(0);
    detailed.record(last_timings_.m2l);
  }
  for (int level = 1; level <= topology_->maximum_level; ++level) {
    const int begin = topology_->l2l_level_offsets[static_cast<std::size_t>(level)];
    const int end = topology_->l2l_level_offsets[static_cast<std::size_t>(level + 1)];
    detailed.start();
#pragma omp parallel for schedule(static) if (end - begin >= 8)
    for (int edge_slot = begin; edge_slot < end; ++edge_slot) {
      const StaticTranslationEdge &edge = topology_->l2l_edges[
          static_cast<std::size_t>(edge_slot)];
      const int target_index = edge.target_node;
      const auto &target = nodes[static_cast<std::size_t>(target_index)];
      if (target.target_count() == 0) {
        continue;
      }
      detail::cpu::apply_packed_translation(
          cpu_packing_->fp32.l2l, target.level, edge.child_class,
          local_float_for_node(edge.source_node).data(),
          local_float_for_node(target_index).data());
    }
    detailed.record(last_timings_.l2l);

    if (!cuda_m2l_executor) {
      detailed.start();
      static_m2l_float(level);
      detailed.record(last_timings_.m2l);
    }
  }

  if (evaluate_l2p) {
    const auto &occupied_leaves = topology_->target_leaves;
    const bool want_field = has_flag(output, OutputFlags::Field);
    const bool want_potential = has_flag(output, OutputFlags::Potential);
    detailed.start();
#pragma omp parallel for schedule(static) if (occupied_leaves.size() >= 8)
    for (std::ptrdiff_t occupied_index = 0;
         occupied_index < static_cast<std::ptrdiff_t>(occupied_leaves.size());
         ++occupied_index) {
      const StaticLeafRange &leaf_range =
          occupied_leaves[static_cast<std::size_t>(occupied_index)];
      const int leaf_index = leaf_range.node;
      const float *L = local_float_for_node(leaf_index).data();
      if (procedural_l2p_) {
        cpu_packing_->procedural_fp32.apply_l2p(
            nodes[static_cast<std::size_t>(leaf_index)].centre,
            std::span<const Vec3>(topology_->sorted_target_positions)
                .subspan(leaf_range.begin, leaf_range.count),
            L,
            std::span<FloatPotentialField>(sorted_results_float_)
                .subspan(leaf_range.begin, leaf_range.count),
            want_field, want_potential);
        continue;
      }
      for (std::size_t target_index = leaf_range.begin;
           target_index < leaf_range.begin + leaf_range.count; ++target_index) {
        FloatPotentialField result;
        if (want_field) {
          detail::cpu::apply_packed_l2p_field(
              cpu_packing_->fp32.l2p, target_index, L, result.H.x,
              result.H.y, result.H.z);
        }
        if (want_potential) {
          result.phi = detail::cpu::apply_packed_l2p_potential(
              cpu_packing_->fp32.l2p, target_index, L);
        }
        sorted_results_float_[target_index] = result;
      }
    }
    detailed.record(last_timings_.l2p);
  }
}

// Hybrid backend: all levels of M2L in one device call.  The multipoles cross
// to the device and the raw locals come back; the plan's timings are split
// into the public H2D/scale/multiply/D2H lanes.
void UniformFmm::cuda_m2l() {
  detail::ProfileRange m2l_range{"cdfmm/far_field/m2l"};
  detail::PhaseStopwatch detailed(detailed_timing());
  detailed.start();
  cuda_m2l_plan_->plan->evaluate(multipoles_, locals_);
  if (detailed.enabled()) {
    const CudaEvaluationTimings &device = cuda_m2l_plan_->plan->timings();
    last_timings_.cuda_h2d.add(device.h2d_seconds);
    last_timings_.cuda_m2l_h2d.add(device.h2d_seconds);
    last_timings_.m2l_scale.add(device.scale_seconds);
    last_timings_.m2l_multiply.add(device.multiply_seconds);
    last_timings_.cuda_kernel.add(device.kernel_seconds);
    last_timings_.cuda_d2h.add(device.d2h_seconds);
    last_timings_.cuda_m2l_d2h.add(device.d2h_seconds);
  }
  detailed.record(last_timings_.m2l);
}

// L2L over all levels after a device M2L has already filled every local.
void UniformFmm::l2l_downward() {
  detail::ProfileRange l2l_range{"cdfmm/far_field/l2l"};
  detail::PhaseStopwatch detailed(detailed_timing());
  const auto &nodes = topology_->nodes;
  for (int level = 1; level <= topology_->maximum_level; ++level) {
    const int begin = topology_->l2l_level_offsets[static_cast<std::size_t>(level)];
    const int end = topology_->l2l_level_offsets[static_cast<std::size_t>(level + 1)];
    detailed.start();
#pragma omp parallel for schedule(static) if (end - begin >= 8)
    for (int edge_slot = begin; edge_slot < end; ++edge_slot) {
      const StaticTranslationEdge &edge = topology_->l2l_edges[
          static_cast<std::size_t>(edge_slot)];
      const int target_index = edge.target_node;
      const auto &target = nodes[static_cast<std::size_t>(target_index)];
      if (target.target_count() == 0) {
        continue;
      }
      detail::cpu::apply_packed_translation(
          cpu_packing_->fp64.l2l, target.level, edge.child_class,
          local_for_node(edge.source_node).data(),
          local_for_node(target_index).data());
    }
    detailed.record(last_timings_.l2l);
  }
}

} // namespace cdfmm
