// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/uniform_fmm.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#ifdef CDFMM_USE_OPENMP
#include <omp.h>
#endif

#include "cdfmm/operators.hpp"
#include "cdfmm/backend/cpu/far_field.hpp"
#include "cdfmm/backend/cpu/m2l.hpp"
#include "cdfmm/backend/cuda/m2l.hpp"
#include "backend/cpu/far_field/internal.hpp"
#include "fmm/internal.hpp"
#include "profile.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Far-field evaluation
//------------------------------------------------------------------------------
// This file owns the execution of the expansion hierarchy:
//
//   sources -> P2M -> M2M -> M2L -> L2L -> L2P -> far field
//
// Geometry-plan construction and list1 P2P do not belong here.  The routines
// consume the canonical operators owned by UniformFmm; they neither rebuild
// plans nor introduce executor-specific copies.  Near-field work may therefore
// proceed independently while this branch consumes explicit topology schedules.

namespace {
using Clock = std::chrono::steady_clock;

inline double elapsed_seconds(const Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

} // namespace

void UniformFmm::static_m2l(const int level) {
  detail::ProfileRange m2l_range{"cdfmm/far_field/m2l"};
  if (static_matrix_backend_ == StaticMatrixBackend::Portable) {
    const auto phase_start = Clock::now();
    apply_static_m2l_plan(m2l_plan_, level, multipoles_, locals_);
    last_timings_.m2l_multiply.add(elapsed_seconds(phase_start));
    return;
  }
  const detail::mkl::M2LApplyTimings timings = mkl_m2l_plan_->apply(
      m2l_plan_, level, multipoles_, locals_);
  last_timings_.m2l_gather.add(timings.gather_seconds);
  last_timings_.m2l_multiply.add(timings.multiply_seconds);
  last_timings_.m2l_scatter.add(timings.scatter_seconds);
}

void UniformFmm::static_m2l_float(const int level) {
  detail::ProfileRange m2l_range{"cdfmm/far_field/m2l_fp32"};
  if (static_matrix_backend_ == StaticMatrixBackend::Portable) {
    const auto phase_start = Clock::now();
    apply_static_m2l_plan(m2l_plan_float_, level, multipoles_float_,
                          locals_float_);
    last_timings_.m2l_multiply.add(elapsed_seconds(phase_start));
    return;
  }

  const detail::mkl::M2LApplyTimings timings = mkl_m2l_plan_->apply(
      m2l_plan_float_, level, multipoles_float_, locals_float_);
  last_timings_.m2l_gather.add(timings.gather_seconds);
  last_timings_.m2l_multiply.add(timings.multiply_seconds);
  last_timings_.m2l_scatter.add(timings.scatter_seconds);
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

void UniformFmm::prepare_moments_float(
    const std::span<const Vec3> dipole_moments) {
  detail::ProfileRange input_range{"cdfmm/input_preparation_fp32"};
  if (dipole_moments.size() != topology_->sorted_source_positions.size()) {
    throw std::invalid_argument(
        "UniformFmm::upward_pass requires one dipole moment per source "
        "position");
  }
  auto phase_start = Clock::now();
  std::fill(multipoles_float_.begin(), multipoles_float_.end(), 0.0F);
  last_timings_.multipole_reset.add(elapsed_seconds(phase_start));

  phase_start = Clock::now();
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
  last_timings_.moment_permutation.add(elapsed_seconds(phase_start));
}

void UniformFmm::prepare_moments_float(
    const std::span<const FloatVec3> dipole_moments) {
  detail::ProfileRange input_range{"cdfmm/input_preparation_fp32"};
  if (dipole_moments.size() != topology_->sorted_source_positions.size()) {
    throw std::invalid_argument(
        "UniformFmm::upward_pass requires one dipole moment per source position");
  }
  auto phase_start = Clock::now();
  std::fill(multipoles_float_.begin(), multipoles_float_.end(), 0.0F);
  last_timings_.multipole_reset.add(elapsed_seconds(phase_start));
  phase_start = Clock::now();
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
  last_timings_.moment_permutation.add(elapsed_seconds(phase_start));
}

void UniformFmm::upward_pass_prepared_float() {
  const auto &nodes = topology_->nodes;
  const auto &occupied_leaves = topology_->source_leaves;
  auto phase_start = Clock::now();
#pragma omp parallel for schedule(static) if (occupied_leaves.size() >= 8)
  for (std::ptrdiff_t occupied_index = 0;
       occupied_index < static_cast<std::ptrdiff_t>(occupied_leaves.size());
       ++occupied_index) {
    const int leaf_index =
        occupied_leaves[static_cast<std::size_t>(occupied_index)].node;
    const StaticLeafRange &leaf_range =
        occupied_leaves[static_cast<std::size_t>(occupied_index)];
    const FloatStaticCoefficientOperator &operator_map =
        p2m_plans_float_[static_cast<std::size_t>(occupied_index)].operator_map;
    const auto M = multipole_float_for_node(leaf_index);
    detail::cpu::apply_static_p2m(
        operator_map,
        std::span<const FloatVec3>(sorted_dipole_moments_float_)
            .subspan(leaf_range.begin, leaf_range.count), M);
  }
  last_timings_.p2m.add(elapsed_seconds(phase_start));

  phase_start = Clock::now();
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
          detail::cpu::apply_static_translation(
              m2m_operators_float_[edge.child_class],
              multipole_float_for_node(child_index), parent_M, child.level,
              detail::cpu::coefficient_degree_view(
                  expansion_basis_ == ExpansionBasis::Spherical, basis_,
                  spherical_basis_));
        }
      }
    }
  }
  last_timings_.m2m.add(elapsed_seconds(phase_start));
}

void UniformFmm::prepare_moments(std::span<const Vec3> dipole_moments) {
  detail::ProfileRange input_range{"cdfmm/input_preparation"};
  if (dipole_moments.size() != topology_->sorted_source_positions.size()) {
    throw std::invalid_argument(
        "UniformFmm::upward_pass requires one dipole moment per source "
        "position");
  }

  {
    const auto phase_start = Clock::now();
    detail::ProfileRange reset_range{"cdfmm/far_field/multipole_reset"};
    // Flat node-major storage permits one streaming reset instead of one small
    // fill and one vector-metadata load per tree node.
    std::fill(multipoles_.begin(), multipoles_.end(), 0.0);
    last_timings_.multipole_reset.add(elapsed_seconds(phase_start));
  }

  const auto phase_start = Clock::now();
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
  last_timings_.moment_permutation.add(elapsed_seconds(phase_start));
}

void UniformFmm::upward_pass_prepared() {
  const auto &nodes = topology_->nodes;
  const auto &occupied_leaves = topology_->source_leaves;
  const StaticOperatorExecutor p2m_executor = execution_plan().p2m;
  auto phase_start = Clock::now();
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
      const StaticCoefficientOperator &operator_map =
          p2m_plans_[static_cast<std::size_t>(occupied_index)].operator_map;
      const auto M = multipole_for_node(leaf_index);
      detail::cpu::apply_static_p2m(
          operator_map,
          std::span<const Vec3>(sorted_dipole_moments_)
              .subspan(leaf_range.begin, leaf_range.count), M);
    } else {
      const CoeffVector M = p2m_dipole(basis_, leaf.centre,
                     std::span<const Vec3>(topology_->sorted_source_positions)
                         .subspan(leaf_range.begin, leaf_range.count),
                     std::span<const Vec3>(sorted_dipole_moments_)
                         .subspan(leaf_range.begin, leaf_range.count));
      std::copy(M.begin(), M.end(), multipole_for_node(leaf_index).begin());
    }
  }
  last_timings_.p2m.add(elapsed_seconds(phase_start));
  p2m_range.end();

  phase_start = Clock::now();
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
            detail::cpu::apply_static_translation(
                m2m_operators_[edge.child_class],
                multipole_for_node(child_index), parent_M, child.level,
                detail::cpu::coefficient_degree_view(
                    expansion_basis_ == ExpansionBasis::Spherical, basis_,
                    spherical_basis_));
          } else {
            const Vec3 d = parent.centre - child.centre;
            m2m_add(basis_, d,
                    multipole_for_node(child_index),
                    parent_M);
          }
        }
      }
    }
  }
  last_timings_.m2m.add(elapsed_seconds(phase_start));
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

void UniformFmm::downward_pass_for_output(const OutputFlags output,
                                          const bool evaluate_l2p) {
  detail::ProfileRange downward_range{"cdfmm/far_field/downward"};
  auto phase_start = Clock::now();
  detail::ProfileRange reset_range{"cdfmm/far_field/local_reset"};
  std::fill(locals_.begin(), locals_.end(), 0.0);
  last_timings_.local_reset.add(elapsed_seconds(phase_start));
  reset_range.end();

  if (execution_plan().m2l == StaticOperatorExecutor::Cuda) {
    cuda_m2l();
    l2l_downward();
  } else {
    if (periodic_.enabled && m2l_backend_ == M2LBackend::Static) {
      phase_start = Clock::now();
      static_m2l(0);
      last_timings_.m2l.add(elapsed_seconds(phase_start));
    }

    const auto &nodes = topology_->nodes;
    for (int level = 1; level <= topology_->maximum_level; ++level) {
      // Parent locals must be inherited before this level's M2L is added.
      // Advancing levels in order makes the parent-child dependency explicit.
      const int begin = topology_->l2l_level_offsets[static_cast<std::size_t>(level)];
      const int end = topology_->l2l_level_offsets[static_cast<std::size_t>(level + 1)];

      phase_start = Clock::now();
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
        if (execution_plan().l2l != StaticOperatorExecutor::Reference) {
          detail::cpu::apply_static_translation(
              l2l_operators_[edge.child_class], local_for_node(edge.source_node),
              local_for_node(target_index), target.level,
              detail::cpu::coefficient_degree_view(
                  expansion_basis_ == ExpansionBasis::Spherical, basis_,
                  spherical_basis_));
        } else {
          const Vec3 d = target.centre - nodes[static_cast<std::size_t>(edge.source_node)].centre;
          l2l_add(basis_, d, local_for_node(edge.source_node),
                  local_for_node(target_index));
        }
      }
      last_timings_.l2l.add(elapsed_seconds(phase_start));
      l2l_range.end();

      if (m2l_backend_ == M2LBackend::Static) {
        phase_start = Clock::now();
        static_m2l(level);
        last_timings_.m2l.add(elapsed_seconds(phase_start));
        continue;
      }

      phase_start = Clock::now();
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
        m2l_add(basis_, R, multipole_for_node(interaction.source_node),
                local_for_node(interaction.target_node));
      }
      last_timings_.m2l.add(elapsed_seconds(phase_start));
    }
  }

  // Keep the coarse downward profile scoped to M2L/L2L; L2P has its own
  // hierarchy-stage range and timing below.
  downward_range.end();
  if (evaluate_l2p) {
    const auto &nodes = topology_->nodes;
    const auto targets = std::span<const Vec3>(topology_->sorted_target_positions);
    const auto &occupied_leaves = topology_->target_leaves;
    phase_start = Clock::now();
    detail::ProfileRange l2p_range{"cdfmm/far_field/l2p"};
#pragma omp parallel for schedule(static) if (occupied_leaves.size() >= 8)
    for (std::ptrdiff_t occupied_index = 0;
         occupied_index < static_cast<std::ptrdiff_t>(occupied_leaves.size());
         ++occupied_index) {
      const StaticLeafRange &leaf_range =
          occupied_leaves[static_cast<std::size_t>(occupied_index)];
      const int leaf_index = leaf_range.node;
      const auto &leaf = nodes[static_cast<std::size_t>(leaf_index)];
      for (std::size_t target_index = leaf_range.begin;
           target_index < leaf_range.begin + leaf_range.count; ++target_index) {
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
  }
}

void UniformFmm::downward_pass_float_for_output(const OutputFlags output,
                                                const bool evaluate_l2p) {
  auto phase_start = Clock::now();
  std::fill(locals_float_.begin(), locals_float_.end(), 0.0F);
  last_timings_.local_reset.add(elapsed_seconds(phase_start));

  const bool cuda_m2l_executor =
      execution_plan().m2l == StaticOperatorExecutor::Cuda;
  if (cuda_m2l_executor) {
    phase_start = Clock::now();
    cuda_m2l_plan_->plan->evaluate(multipoles_float_, locals_float_);
    const CudaEvaluationTimings &device = cuda_m2l_plan_->plan->timings();
    last_timings_.cuda_h2d.add(device.h2d_seconds);
    last_timings_.cuda_m2l_h2d.add(device.h2d_seconds);
    last_timings_.m2l_scale.add(device.scale_seconds);
    last_timings_.m2l_multiply.add(device.multiply_seconds);
    last_timings_.cuda_kernel.add(device.kernel_seconds);
    last_timings_.cuda_d2h.add(device.d2h_seconds);
    last_timings_.cuda_m2l_d2h.add(device.d2h_seconds);
    last_timings_.m2l.add(elapsed_seconds(phase_start));
  }

  const auto &nodes = topology_->nodes;
  if (periodic_.enabled && !cuda_m2l_executor) {
    phase_start = Clock::now();
    static_m2l_float(0);
    last_timings_.m2l.add(elapsed_seconds(phase_start));
  }
  for (int level = 1; level <= topology_->maximum_level; ++level) {
    const int begin = topology_->l2l_level_offsets[static_cast<std::size_t>(level)];
    const int end = topology_->l2l_level_offsets[static_cast<std::size_t>(level + 1)];
    phase_start = Clock::now();
#pragma omp parallel for schedule(static) if (end - begin >= 8)
    for (int edge_slot = begin; edge_slot < end; ++edge_slot) {
      const StaticTranslationEdge &edge = topology_->l2l_edges[
          static_cast<std::size_t>(edge_slot)];
      const int target_index = edge.target_node;
      const auto &target = nodes[static_cast<std::size_t>(target_index)];
      if (target.target_count() == 0) {
        continue;
      }
      detail::cpu::apply_static_translation(
          l2l_operators_float_[edge.child_class],
          local_float_for_node(edge.source_node),
          local_float_for_node(target_index), target.level,
          detail::cpu::coefficient_degree_view(
              expansion_basis_ == ExpansionBasis::Spherical, basis_,
              spherical_basis_));
    }
    last_timings_.l2l.add(elapsed_seconds(phase_start));

    if (!cuda_m2l_executor) {
      phase_start = Clock::now();
      static_m2l_float(level);
      last_timings_.m2l.add(elapsed_seconds(phase_start));
    }
  }

  if (evaluate_l2p) {
    const auto &occupied_leaves = topology_->target_leaves;
    phase_start = Clock::now();
#pragma omp parallel for schedule(static) if (occupied_leaves.size() >= 8)
    for (std::ptrdiff_t occupied_index = 0;
         occupied_index < static_cast<std::ptrdiff_t>(occupied_leaves.size());
         ++occupied_index) {
      const StaticLeafRange &leaf_range =
          occupied_leaves[static_cast<std::size_t>(occupied_index)];
      const int leaf_index = leaf_range.node;
      for (std::size_t target_index = leaf_range.begin;
           target_index < leaf_range.begin + leaf_range.count; ++target_index) {
        sorted_results_float_[target_index] =
            apply_static_l2p_evaluator(l2p_evaluators_float_[target_index],
                                       local_float_for_node(leaf_index), output);
      }
    }
    last_timings_.l2p.add(elapsed_seconds(phase_start));
  }
}

void UniformFmm::cuda_m2l() {
  detail::ProfileRange m2l_range{"cdfmm/far_field/m2l"};
  const auto phase_start = Clock::now();
  cuda_m2l_plan_->plan->evaluate(multipoles_, locals_);
  const CudaEvaluationTimings &device = cuda_m2l_plan_->plan->timings();
  last_timings_.cuda_h2d.add(device.h2d_seconds);
  last_timings_.cuda_m2l_h2d.add(device.h2d_seconds);
  last_timings_.m2l_scale.add(device.scale_seconds);
  last_timings_.m2l_multiply.add(device.multiply_seconds);
  last_timings_.cuda_kernel.add(device.kernel_seconds);
  last_timings_.cuda_d2h.add(device.d2h_seconds);
  last_timings_.cuda_m2l_d2h.add(device.d2h_seconds);
  last_timings_.m2l.add(elapsed_seconds(phase_start));
}

void UniformFmm::l2l_downward() {
  detail::ProfileRange l2l_range{"cdfmm/far_field/l2l"};
  const auto &nodes = topology_->nodes;
  for (int level = 1; level <= topology_->maximum_level; ++level) {
    const int begin = topology_->l2l_level_offsets[static_cast<std::size_t>(level)];
    const int end = topology_->l2l_level_offsets[static_cast<std::size_t>(level + 1)];
    const auto phase_start = Clock::now();
#pragma omp parallel for schedule(static) if (end - begin >= 8)
    for (int edge_slot = begin; edge_slot < end; ++edge_slot) {
      const StaticTranslationEdge &edge = topology_->l2l_edges[
          static_cast<std::size_t>(edge_slot)];
      const int target_index = edge.target_node;
      const auto &target = nodes[static_cast<std::size_t>(target_index)];
      if (target.target_count() == 0) {
        continue;
      }
      detail::cpu::apply_static_translation(
          l2l_operators_[edge.child_class], local_for_node(edge.source_node),
          local_for_node(target_index), target.level,
          detail::cpu::coefficient_degree_view(
              expansion_basis_ == ExpansionBasis::Spherical, basis_,
              spherical_basis_));
    }
    last_timings_.l2l.add(elapsed_seconds(phase_start));
  }
}

//------------------------------------------------------------------------------
// Complete evaluation
//------------------------------------------------------------------------------

} // namespace cdfmm
