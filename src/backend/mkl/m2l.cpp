// SPDX-License-Identifier: Apache-2.0

#include "backend/mkl/m2l.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef CDFMM_USE_MKL
#include <mkl.h>
#endif

namespace cdfmm {

bool one_mkl_available() noexcept {
#ifdef CDFMM_USE_MKL
  return true;
#else
  return false;
#endif
}

namespace detail::mkl {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(const Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

template <typename Scalar>
struct M2LGroup {
  int matrix_id{0};
  std::vector<int> sources{};
  std::vector<int> targets{};
  std::vector<int> source_levels{};
  std::vector<int> levels{};
  std::vector<Scalar> gathered{};
  std::vector<Scalar> translated{};
};

template <typename Plan, typename Scalar>
std::vector<M2LGroup<Scalar>> prepare_groups(
    const Plan& plan, M2LStorageStatistics& statistics) {
  std::vector<M2LGroup<Scalar>> groups(
      static_cast<std::size_t>(plan.matrix_count));
  for (int id = 0; id < plan.matrix_count; ++id) {
    groups[static_cast<std::size_t>(id)].matrix_id = id;
  }

  for (std::size_t target = 0;
       target + 1 < plan.target_row_offsets.size(); ++target) {
    const int begin = plan.target_row_offsets[target];
    const int end = plan.target_row_offsets[target + 1];
    for (int interaction = begin; interaction < end; ++interaction) {
      const std::size_t slot = static_cast<std::size_t>(interaction);
      auto& group = groups[static_cast<std::size_t>(plan.matrix_ids[slot])];
      group.sources.push_back(plan.source_nodes[slot]);
      group.targets.push_back(static_cast<int>(target));
      const int target_level = plan.target_levels.empty()
          ? (plan.interaction_levels.empty()
                 ? plan.node_levels[target]
                 : plan.interaction_levels[slot])
          : plan.target_levels[slot];
      const int source_level = plan.source_levels.empty()
          ? (plan.node_levels.empty()
                 ? target_level
                 : plan.node_levels[static_cast<std::size_t>(
                       plan.source_nodes[slot])])
          : plan.source_levels[slot];
      group.source_levels.push_back(source_level);
      group.levels.push_back(target_level);
    }
  }

  for (auto& group : groups) {
    // oneMKL consumes contiguous columns for one target level. Stable sorting
    // preserves canonical interaction order within each level.
    if (!std::is_sorted(group.levels.begin(), group.levels.end())) {
      std::vector<std::size_t> order(group.levels.size());
      std::iota(order.begin(), order.end(), std::size_t{0});
      std::stable_sort(order.begin(), order.end(),
                       [&group](const std::size_t a, const std::size_t b) {
                         return group.levels[a] < group.levels[b];
                       });
      const auto reorder = [&order](std::vector<int>& values) {
        std::vector<int> sorted;
        sorted.reserve(values.size());
        for (const std::size_t index : order) {
          sorted.push_back(values[index]);
        }
        values = std::move(sorted);
      };
      reorder(group.sources);
      reorder(group.targets);
      reorder(group.source_levels);
      reorder(group.levels);
    }

    const std::size_t values =
        static_cast<std::size_t>(plan.coefficient_count) * group.sources.size();
    group.gathered.resize(values);
    group.translated.resize(values);
    statistics.metadata_bytes +=
        (group.sources.size() + group.targets.size() +
         group.source_levels.size() + group.levels.size()) * sizeof(int);
    statistics.scratch_bytes += 2 * values * sizeof(Scalar);
  }
  return groups;
}

template <typename Plan, typename Scalar>
M2LApplyTimings apply_groups(
    const Plan& plan, std::vector<M2LGroup<Scalar>>& groups, const int level,
    const std::span<const Scalar> multipoles,
    const std::span<Scalar> locals) {
  M2LApplyTimings timings;
  const int n = plan.coefficient_count;
  const std::ptrdiff_t group_count =
      static_cast<std::ptrdiff_t>(groups.size());
  const Scalar* local_scale = plan.local_scaling.data() +
      static_cast<std::size_t>(level) * n;

  auto phase_start = Clock::now();
  // Gather applies source-level multipole scaling while retaining the
  // canonical matrix-column order.
#pragma omp parallel for schedule(dynamic, 1) if (group_count >= 8)
  for (std::ptrdiff_t group_index = 0; group_index < group_count;
       ++group_index) {
    M2LGroup<Scalar>& group =
        groups[static_cast<std::size_t>(group_index)];
    for (std::size_t column = 0; column < group.sources.size(); ++column) {
      if (group.levels[column] != level) {
        continue;
      }
      const Scalar* M = multipoles.data() +
          static_cast<std::size_t>(group.sources[column]) * n;
      const Scalar* source_scale = plan.multipole_scaling.data() +
          static_cast<std::size_t>(group.source_levels.empty()
                                       ? level
                                       : group.source_levels[column]) * n;
      for (int alpha = 0; alpha < n; ++alpha) {
        group.gathered[static_cast<std::size_t>(alpha) + column * n] =
            source_scale[alpha] * M[static_cast<std::size_t>(alpha)];
      }
    }
  }
  timings.gather_seconds = elapsed_seconds(phase_start);

  phase_start = Clock::now();
#ifdef CDFMM_USE_MKL
#pragma omp parallel for schedule(dynamic, 1) if (group_count >= 8)
  for (std::ptrdiff_t group_index = 0; group_index < group_count;
       ++group_index) {
    M2LGroup<Scalar>& group =
        groups[static_cast<std::size_t>(group_index)];
    const auto first =
        std::lower_bound(group.levels.begin(), group.levels.end(), level);
    const auto last = std::upper_bound(first, group.levels.end(), level);
    const int columns = static_cast<int>(last - first);
    if (columns == 0) {
      continue;
    }
    const std::size_t column_offset =
        static_cast<std::size_t>(first - group.levels.begin());
    const Scalar* matrix = plan.matrices.data() +
        static_cast<std::size_t>(group.matrix_id) * n * n;
    const int previous_mkl_threads = mkl_set_num_threads_local(1);
    if constexpr (std::is_same_v<Scalar, float>) {
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                  static_cast<MKL_INT>(n), static_cast<MKL_INT>(columns),
                  static_cast<MKL_INT>(n), 1.0F, matrix,
                  static_cast<MKL_INT>(n),
                  group.gathered.data() + column_offset * n,
                  static_cast<MKL_INT>(n), 0.0F,
                  group.translated.data() + column_offset * n,
                  static_cast<MKL_INT>(n));
    } else {
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                  static_cast<MKL_INT>(n), static_cast<MKL_INT>(columns),
                  static_cast<MKL_INT>(n), 1.0, matrix,
                  static_cast<MKL_INT>(n),
                  group.gathered.data() + column_offset * n,
                  static_cast<MKL_INT>(n), 0.0,
                  group.translated.data() + column_offset * n,
                  static_cast<MKL_INT>(n));
    }
    mkl_set_num_threads_local(previous_mkl_threads);
  }
#else
  static_cast<void>(plan);
  static_cast<void>(groups);
  static_cast<void>(level);
  static_cast<void>(multipoles);
  static_cast<void>(locals);
  throw std::runtime_error("The oneMKL M2L backend is unavailable");
#endif
  timings.multiply_seconds = elapsed_seconds(phase_start);

  phase_start = Clock::now();
  // Scatter remains serial: transfer classes may address the same local and
  // changing that order would require atomics or private reductions.
  for (M2LGroup<Scalar>& group : groups) {
    for (std::size_t column = 0; column < group.targets.size(); ++column) {
      if (group.levels[column] != level) {
        continue;
      }
      Scalar* L = locals.data() +
          static_cast<std::size_t>(group.targets[column]) * n;
      for (int beta = 0; beta < n; ++beta) {
        L[static_cast<std::size_t>(beta)] +=
            local_scale[beta] *
            group.translated[static_cast<std::size_t>(beta) + column * n];
      }
    }
  }
  timings.scatter_seconds = elapsed_seconds(phase_start);
  return timings;
}

} // namespace

struct M2LExecutor::Impl {
  std::vector<M2LGroup<double>> groups{};
  std::vector<M2LGroup<float>> float_groups{};
  M2LStorageStatistics statistics{};
};

M2LExecutor::M2LExecutor(const StaticM2LPlan& plan)
    : impl_(std::make_unique<Impl>()) {
  impl_->groups = prepare_groups<StaticM2LPlan, double>(
      plan, impl_->statistics);
}

M2LExecutor::M2LExecutor(const FloatStaticM2LPlan& plan)
    : impl_(std::make_unique<Impl>()) {
  impl_->float_groups = prepare_groups<FloatStaticM2LPlan, float>(
      plan, impl_->statistics);
}

M2LExecutor::~M2LExecutor() = default;
M2LExecutor::M2LExecutor(M2LExecutor&&) noexcept = default;
M2LExecutor& M2LExecutor::operator=(M2LExecutor&&) noexcept = default;

M2LApplyTimings M2LExecutor::apply(
    const StaticM2LPlan& plan, const int level,
    const std::span<const double> multipoles,
    const std::span<double> locals) {
  return apply_groups(plan, impl_->groups, level, multipoles, locals);
}

M2LApplyTimings M2LExecutor::apply(
    const FloatStaticM2LPlan& plan, const int level,
    const std::span<const float> multipoles,
    const std::span<float> locals) {
  return apply_groups(plan, impl_->float_groups, level, multipoles, locals);
}

M2LStorageStatistics M2LExecutor::statistics() const noexcept {
  return impl_->statistics;
}

} // namespace detail::mkl
} // namespace cdfmm
