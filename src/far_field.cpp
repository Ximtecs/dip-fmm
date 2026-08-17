// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/uniform_fmm.hpp"

#include <algorithm>
#include <chrono>

#ifdef CDFMM_USE_MKL
#include <mkl.h>
#endif

#ifdef CDFMM_USE_OPENMP
#include <omp.h>
#endif

#include "cdfmm/operators.hpp"
#include "cuda_m2l_plan.hpp"
#include "uniform_fmm_internal.hpp"

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
// proceed independently while this branch traverses the level-ordered tree.

namespace {
using Clock = std::chrono::steady_clock;

inline double elapsed_seconds(const Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
} // namespace

void UniformFmm::static_m2l(const int level) {
  if (static_matrix_backend_ == StaticMatrixBackend::Portable) {
    const auto phase_start = Clock::now();
    apply_static_m2l_plan(m2l_plan_, level, multipoles_, locals_);
    last_timings_.m2l_multiply.add(elapsed_seconds(phase_start));
    return;
  }
  const int n = basis_.size();
  const std::ptrdiff_t group_count =
      static_cast<std::ptrdiff_t>(m2l_groups_.size());
  const double *multipole_scale =
      m2l_plan_.multipole_scaling.data() + static_cast<std::size_t>(level) * n;
  const double *local_scale =
      m2l_plan_.local_scaling.data() + static_cast<std::size_t>(level) * n;

  auto phase_start = Clock::now();
  // oneMKL works most effectively on columns sharing one transfer matrix.
  // Gather applies multipole scaling while preserving the canonical plan.
#pragma omp parallel for schedule(dynamic, 1) if (group_count >= 8)
  for (std::ptrdiff_t group_index = 0; group_index < group_count;
       ++group_index) {
    M2LGroup &group = m2l_groups_[static_cast<std::size_t>(group_index)];
    for (std::size_t column = 0; column < group.sources.size(); ++column) {
      if (group.levels[column] != level) {
        continue;
      }
      const auto M = multipole_for_node(group.sources[column]);
      for (int alpha = 0; alpha < n; ++alpha) {
        group.gathered[static_cast<std::size_t>(alpha) + column * n] =
            multipole_scale[alpha] * M[static_cast<std::size_t>(alpha)];
      }
    }
  }
  last_timings_.m2l_gather.add(elapsed_seconds(phase_start));

  phase_start = Clock::now();
#ifdef CDFMM_USE_MKL
  if (static_matrix_backend_ == StaticMatrixBackend::OneMkl) {
#pragma omp parallel for schedule(dynamic, 1) if (group_count >= 8)
    for (std::ptrdiff_t group_index = 0; group_index < group_count;
         ++group_index) {
      M2LGroup &group = m2l_groups_[static_cast<std::size_t>(group_index)];
      const auto first =
          std::lower_bound(group.levels.begin(), group.levels.end(), level);
      const auto last = std::upper_bound(first, group.levels.end(), level);
      const int columns = static_cast<int>(last - first);
      if (columns == 0) {
        continue;
      }
      const std::size_t column_offset =
          static_cast<std::size_t>(first - group.levels.begin());
      const double *matrix = m2l_plan_.matrices.data() +
                             static_cast<std::size_t>(group.matrix_id) * n * n;
      const int previous_mkl_threads = mkl_set_num_threads_local(1);
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, columns, n, 1.0,
                  matrix, n, group.gathered.data() + column_offset * n, n, 0.0,
                  group.translated.data() + column_offset * n, n);
      mkl_set_num_threads_local(previous_mkl_threads);
    }
  } else
#endif
  {
#pragma omp parallel for schedule(dynamic, 1) if (group_count >= 8)
    for (std::ptrdiff_t group_index = 0; group_index < group_count;
         ++group_index) {
      M2LGroup &group = m2l_groups_[static_cast<std::size_t>(group_index)];
      const double *matrix = m2l_plan_.matrices.data() +
                             static_cast<std::size_t>(group.matrix_id) * n * n;
      for (std::size_t column = 0; column < group.sources.size(); ++column) {
        if (group.levels[column] != level) {
          continue;
        }
        double *translated = group.translated.data() + column * n;
        std::fill(translated, translated + n, 0.0);
        for (int alpha = 0; alpha < n; ++alpha) {
          const double value =
              group.gathered[static_cast<std::size_t>(alpha) + column * n];
          for (int beta = 0; beta < n; ++beta) {
            translated[beta] +=
                matrix[static_cast<std::size_t>(beta + alpha * n)] * value;
          }
        }
      }
    }
  }
  last_timings_.m2l_multiply.add(elapsed_seconds(phase_start));

  phase_start = Clock::now();
  for (M2LGroup &group : m2l_groups_) {
    // Scatter is deliberately serial across groups: different transfer
    // classes can address the same target local, so parallel groups would
    // otherwise require atomics or private reduction buffers.
    for (std::size_t column = 0; column < group.targets.size(); ++column) {
      if (group.levels[column] != level) {
        continue;
      }
      const auto L = local_for_node(group.targets[column]);
      for (int beta = 0; beta < n; ++beta) {
        L[static_cast<std::size_t>(beta)] +=
            local_scale[beta] *
            group.translated[static_cast<std::size_t>(beta) + column * n];
      }
    }
  }
  last_timings_.m2l_scatter.add(elapsed_seconds(phase_start));
}

//------------------------------------------------------------------------------
// Upward pass
//------------------------------------------------------------------------------

void UniformFmm::upward_pass(std::span<const Vec3> dipole_moments) {
  prepare_moments(dipole_moments);
  upward_pass_prepared();
}

void UniformFmm::prepare_moments(std::span<const Vec3> dipole_moments) {
  if (dipole_moments.size() != tree_.sorted_source_positions().size()) {
    throw std::invalid_argument(
        "UniformFmm::upward_pass requires one dipole moment per source "
        "position");
  }

  auto phase_start = Clock::now();
  // Flat node-major storage permits one streaming reset instead of one small
  // fill and one vector-metadata load per tree node.
  std::fill(multipoles_.begin(), multipoles_.end(), 0.0);
  last_timings_.multipole_reset.add(elapsed_seconds(phase_start));

  phase_start = Clock::now();
  const auto permutation = tree_.source_permutation();
#pragma omp parallel for schedule(static) if (permutation.size() >= 256)
  for (std::ptrdiff_t sorted_index = 0;
       sorted_index < static_cast<std::ptrdiff_t>(permutation.size());
       ++sorted_index) {
    const int original_index =
        permutation[static_cast<std::size_t>(sorted_index)];
    sorted_dipole_moments_[static_cast<std::size_t>(sorted_index)] =
        dipole_moments[static_cast<std::size_t>(original_index)];
  }
  last_timings_.moment_permutation.add(elapsed_seconds(phase_start));
}

void UniformFmm::upward_pass_prepared() {
  const auto nodes = tree_.nodes();
  const auto occupied_leaves = tree_.occupied_source_leaves();
  const StaticOperatorExecutor p2m_executor = execution_plan().p2m;
  auto phase_start = Clock::now();
#pragma omp parallel for schedule(static) if (occupied_leaves.size() >= 8)
  for (std::ptrdiff_t occupied_index = 0;
       occupied_index < static_cast<std::ptrdiff_t>(occupied_leaves.size());
       ++occupied_index) {
    // Occupied leaves have disjoint multipole vectors. Each worker therefore
    // owns its output and no synchronisation is needed inside P2M.
    const int leaf_index =
        occupied_leaves[static_cast<std::size_t>(occupied_index)];
    const TreeNode &leaf = nodes[static_cast<std::size_t>(leaf_index)];
    if (p2m_executor != StaticOperatorExecutor::Reference) {
      const StaticCoefficientOperator &operator_map =
          p2m_plans_[static_cast<std::size_t>(occupied_index)].operator_map;
      const auto M = multipole_for_node(leaf_index);
      for (const StaticOperatorEntry &entry : operator_map.entries) {
        const Vec3 &moment =
            sorted_dipole_moments_[leaf.source_begin +
                                   static_cast<std::size_t>(entry.input / 3)];
        const double component =
            entry.input % 3 == 0 ? moment.x
                                 : (entry.input % 3 == 1 ? moment.y : moment.z);
        M[static_cast<std::size_t>(entry.output)] += entry.value * component;
      }
    } else {
      const CoeffVector M = p2m_dipole(basis_, leaf.centre,
                     tree_.sorted_source_positions().subspan(
                         leaf.source_begin, leaf.source_count()),
                     std::span<const Vec3>(sorted_dipole_moments_)
                         .subspan(leaf.source_begin, leaf.source_count()));
      std::copy(M.begin(), M.end(), multipole_for_node(leaf_index).begin());
    }
  }
  last_timings_.p2m.add(elapsed_seconds(phase_start));

  phase_start = Clock::now();
  const StaticOperatorExecutor m2m_executor = execution_plan().m2m;
// One team traverses all dependent levels; the implicit omp-for barrier
// makes each parent level complete before its parent is consumed.
#pragma omp parallel if (nodes.size() >= 64)
  {
    for (int level = tree_.leaf_level() - 1; level >= 0; --level) {
      const int begin = level_offset(level);
      const int end = level_offset(level + 1);
#pragma omp for schedule(static)
      for (int parent_index = begin; parent_index < end; ++parent_index) {
        const TreeNode &parent = nodes[static_cast<std::size_t>(parent_index)];
        if (parent.source_count() == 0) {
          continue;
        }
        const auto parent_M = multipole_for_node(parent_index);
        for (const int child_index : parent.children) {
          const TreeNode &child = nodes[static_cast<std::size_t>(child_index)];
          if (child.source_count() == 0) {
            continue;
          }
          if (m2m_executor != StaticOperatorExecutor::Reference) {
            const int child_class =
                (child.ix & 1) | ((child.iy & 1) << 1) | ((child.iz & 1) << 2);
            apply_static_operator(
                m2m_operators_[static_cast<std::size_t>(child.level)]
                              [child_class],
                multipole_for_node(child_index), parent_M);
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
  auto phase_start = Clock::now();
  std::fill(locals_.begin(), locals_.end(), 0.0);
  last_timings_.local_reset.add(elapsed_seconds(phase_start));

  if (execution_plan().m2l == StaticOperatorExecutor::Cuda) {
    cuda_m2l();
    l2l_downward();
    return;
  }

  const auto nodes = tree_.nodes();
  for (int level = 1; level <= tree_.leaf_level(); ++level) {
    // Parent locals must be inherited before this level's M2L is added.
    // Advancing levels in order makes the parent-child dependency explicit.
    const int begin = level_offset(level);
    const int end = level_offset(level + 1);

    phase_start = Clock::now();
#pragma omp parallel for schedule(static) if (end - begin >= 8)
    for (int target_index = begin; target_index < end; ++target_index) {
      const TreeNode &target = nodes[static_cast<std::size_t>(target_index)];
      if (target.target_count() == 0) {
        continue;
      }
      const TreeNode &parent = nodes[static_cast<std::size_t>(target.parent)];
      if (execution_plan().l2l != StaticOperatorExecutor::Reference) {
        const int child_class =
            (target.ix & 1) | ((target.iy & 1) << 1) | ((target.iz & 1) << 2);
        apply_static_operator(
            l2l_operators_[static_cast<std::size_t>(target.level)][child_class],
            local_for_node(target.parent),
            local_for_node(target_index));
      } else {
        const Vec3 d = target.centre - parent.centre;
        l2l_add(basis_, d, local_for_node(target.parent),
                local_for_node(target_index));
      }
    }
    last_timings_.l2l.add(elapsed_seconds(phase_start));

    if (m2l_backend_ == M2LBackend::Static) {
      phase_start = Clock::now();
      static_m2l(level);
      last_timings_.m2l.add(elapsed_seconds(phase_start));
      continue;
    }

    phase_start = Clock::now();
#pragma omp parallel for schedule(static) if (end - begin >= 8)
    for (int target_index = begin; target_index < end; ++target_index) {
      const TreeNode &target = nodes[static_cast<std::size_t>(target_index)];
      if (target.target_count() == 0) {
        continue;
      }
      const auto target_L = local_for_node(target_index);
      for (const int source_index : target.list2) {
        const TreeNode &source = nodes[static_cast<std::size_t>(source_index)];
        if (source.source_count() == 0) {
          continue;
        }
        const Vec3 R = target.centre - source.centre;
        m2l_add(basis_, R, multipole_for_node(source_index),
                target_L);
      }
    }
    last_timings_.m2l.add(elapsed_seconds(phase_start));
  }
}

void UniformFmm::cuda_m2l() {
  const auto phase_start = Clock::now();
  cuda_m2l_plan_->plan->evaluate(multipoles_, locals_);
  const CudaEvaluationTimings &device = cuda_m2l_plan_->plan->timings();
  last_timings_.cuda_h2d.add(device.h2d_seconds);
  last_timings_.cuda_m2l_h2d.add(device.h2d_seconds);
  last_timings_.m2l_multiply.add(device.multiply_seconds);
  last_timings_.cuda_kernel.add(device.kernel_seconds);
  last_timings_.cuda_d2h.add(device.d2h_seconds);
  last_timings_.cuda_m2l_d2h.add(device.d2h_seconds);
  last_timings_.m2l.add(elapsed_seconds(phase_start));
}

void UniformFmm::l2l_downward() {
  const auto nodes = tree_.nodes();
  for (int level = 1; level <= tree_.leaf_level(); ++level) {
    const int begin = level_offset(level);
    const int end = level_offset(level + 1);
    const auto phase_start = Clock::now();
#pragma omp parallel for schedule(static) if (end - begin >= 8)
    for (int target_index = begin; target_index < end; ++target_index) {
      const TreeNode &target = nodes[static_cast<std::size_t>(target_index)];
      if (target.target_count() == 0) {
        continue;
      }
      const int child_class =
          (target.ix & 1) | ((target.iy & 1) << 1) | ((target.iz & 1) << 2);
      apply_static_operator(
          l2l_operators_[static_cast<std::size_t>(target.level)][child_class],
          local_for_node(target.parent),
          local_for_node(target_index));
    }
    last_timings_.l2l.add(elapsed_seconds(phase_start));
  }
}

//------------------------------------------------------------------------------
// Complete evaluation
//------------------------------------------------------------------------------

} // namespace cdfmm
