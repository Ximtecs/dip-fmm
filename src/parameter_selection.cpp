// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/parameter_selection.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "cdfmm/operators.hpp"

namespace cdfmm {
namespace {

constexpr std::size_t maximum_adviser_nodes = 300000;

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 0) {
    return 0.5 * (values[middle - 1] + values[middle]);
  }
  return values[middle];
}

std::vector<int> default_depths(const std::size_t particle_count) {
  const int upper = particle_count < 32 ? 3 : 5;
  std::vector<int> depths;
  for (int depth = 1; depth <= upper; ++depth) {
    depths.push_back(depth);
  }
  return depths;
}

std::vector<int> default_orders() {
  return {2, 3, 4, 5, 6, 7, 8};
}

bool reasonable_depth(const int depth) {
  if (depth < 1 || depth > 20) {
    return false;
  }
  std::size_t nodes = 0;
  std::size_t level_nodes = 1;
  for (int level = 0; level <= depth; ++level) {
    nodes += level_nodes;
    if (nodes > maximum_adviser_nodes) {
      return false;
    }
    level_nodes *= 8;
  }
  return true;
}

double far_seconds(const EvaluationTimings &timings) {
  return timings.moment_permutation.total_seconds +
         timings.multipole_reset.total_seconds + timings.p2m.total_seconds +
         timings.m2m.total_seconds + timings.local_reset.total_seconds +
         timings.m2l.total_seconds + timings.l2l.total_seconds +
         timings.l2p.total_seconds;
}

PerformanceCandidate benchmark_candidate(
    std::span<const Vec3> source_positions,
    std::span<const Vec3> target_positions,
    std::span<const Vec3> dipole_moments, const int order, const int depth,
    const ExecutionBackend backend, const int repetitions,
    std::span<const int> target_source_indices) {
  PerformanceCandidate candidate;
  candidate.depth = depth;
  if (!reasonable_depth(depth)) {
    candidate.reason = "candidate exceeds the adviser tree-size limit";
    return candidate;
  }

  try {
    UniformFmmOptions options;
    options.expansion_order = order;
    options.tree.max_level = depth;
    options.backend = backend;
    UniformFmm fmm({source_positions.begin(), source_positions.end()},
                   {target_positions.begin(), target_positions.end()}, options);
    (void)fmm.evaluate(dipole_moments, OutputFlags::Field,
                       target_source_indices);

    std::vector<double> near_samples;
    std::vector<double> far_samples;
    std::vector<double> total_samples;
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      (void)fmm.evaluate(dipole_moments, OutputFlags::Field,
                         target_source_indices);
      const EvaluationTimings &timings = fmm.last_timings();
      near_samples.push_back(timings.p2p.total_seconds);
      far_samples.push_back(far_seconds(timings));
      total_samples.push_back(timings.total.total_seconds);
    }
    candidate.near_seconds = median(near_samples);
    candidate.far_seconds = median(far_samples);
    candidate.evaluation_seconds = median(total_samples);
    candidate.estimated_concurrent_seconds =
        std::max(candidate.near_seconds, candidate.far_seconds);
    candidate.balance_ratio = branch_balance_ratio(candidate.near_seconds,
                                                    candidate.far_seconds);
    candidate.succeeded = true;
  } catch (const std::exception &error) {
    candidate.reason = error.what();
  }
  return candidate;
}

} // namespace

std::vector<std::size_t>
deterministic_target_sample(const std::size_t target_count,
                            const std::size_t sample_size) {
  if (sample_size == 0) {
    throw std::invalid_argument("sample_size must be positive");
  }
  const std::size_t count = std::min(target_count, sample_size);
  std::vector<std::size_t> indices;
  indices.reserve(count);
  for (std::size_t sample = 0; sample < count; ++sample) {
    // Midpoints of equal index-space strata avoid a bias towards either end.
    indices.push_back(((2 * sample + 1) * target_count) / (2 * count));
  }
  return indices;
}

double branch_balance_ratio(const double near_seconds,
                            const double far_seconds) {
  const double smaller = std::min(near_seconds, far_seconds);
  const double larger = std::max(near_seconds, far_seconds);
  return smaller > 0.0 ? larger / smaller
                       : std::numeric_limits<double>::infinity();
}

PerformanceSuggestion suggest_depth_for_performance(
    std::span<const Vec3> source_positions,
    std::span<const Vec3> target_positions,
    std::span<const Vec3> dipole_moments, const int order,
    const ExecutionBackend backend, std::span<const int> candidate_depths,
    const int repetitions, std::span<const int> target_source_indices) {
  if (source_positions.size() != dipole_moments.size()) {
    throw std::invalid_argument("source positions and moments must have equal length");
  }
  if (target_positions.empty() || repetitions < 1 || order < 0) {
    throw std::invalid_argument("targets, order, and repetitions are invalid");
  }
  const std::vector<int> defaults = default_depths(
      std::max(source_positions.size(), target_positions.size()));
  const std::span<const int> depths = candidate_depths.empty()
                                          ? std::span<const int>(defaults)
                                          : candidate_depths;
  if (depths.empty()) {
    throw std::invalid_argument("candidate_depths must not be empty");
  }

  PerformanceSuggestion suggestion;
  suggestion.order = order;
  suggestion.branches_concurrent = backend == ExecutionBackend::CudaM2LP2P;
  for (const int depth : depths) {
    suggestion.candidates.push_back(benchmark_candidate(
        source_positions, target_positions, dipole_moments, order, depth,
        backend, repetitions, target_source_indices));
  }
  double fastest_wall = std::numeric_limits<double>::infinity();
  for (const auto &candidate : suggestion.candidates) {
    if (candidate.succeeded) {
      fastest_wall = std::min(fastest_wall, candidate.evaluation_seconds);
    }
  }
  const PerformanceCandidate *best = nullptr;
  for (const auto &candidate : suggestion.candidates) {
    if (!candidate.succeeded) {
      continue;
    }
    // A 25% wall-time guard prevents an attractive balance ratio from
    // recommending a clearly slower overlapping configuration.
    if (suggestion.branches_concurrent &&
        candidate.evaluation_seconds > 1.25 * fastest_wall) {
      continue;
    }
    const double score = suggestion.branches_concurrent
        ? candidate.estimated_concurrent_seconds
        : candidate.evaluation_seconds;
    const double best_score = best == nullptr
        ? std::numeric_limits<double>::infinity()
        : (suggestion.branches_concurrent
               ? best->estimated_concurrent_seconds
               : best->evaluation_seconds);
    if (score < best_score) {
      best = &candidate;
    }
  }
  if (best != nullptr) {
    suggestion.suggested_depth = best->depth;
  }
  return suggestion;
}

AccuracySuggestion suggest_parameters_for_accuracy(
    std::span<const Vec3> source_positions,
    std::span<const Vec3> target_positions,
    std::span<const Vec3> dipole_moments, const double desired_accuracy,
    const ExecutionBackend backend, std::span<const int> candidate_orders,
    std::span<const int> candidate_depths, const std::size_t sample_size,
    const int repetitions, std::span<const int> target_source_indices) {
  if (!(desired_accuracy > 0.0)) {
    throw std::invalid_argument("desired_accuracy must be positive");
  }
  if (source_positions.size() != dipole_moments.size() ||
      target_positions.empty() || repetitions < 1) {
    throw std::invalid_argument("positions, moments, targets, or repetitions are invalid");
  }
  if (!target_source_indices.empty() &&
      target_source_indices.size() != target_positions.size()) {
    throw std::invalid_argument("target_source_indices has incorrect length");
  }
  const std::vector<int> orders_default = default_orders();
  const std::vector<int> depths_default = default_depths(
      std::max(source_positions.size(), target_positions.size()));
  const std::span<const int> orders = candidate_orders.empty()
      ? std::span<const int>(orders_default) : candidate_orders;
  const std::span<const int> depths = candidate_depths.empty()
      ? std::span<const int>(depths_default) : candidate_depths;

  AccuracySuggestion suggestion;
  suggestion.requested_accuracy = desired_accuracy;
  suggestion.reference_target_indices =
      deterministic_target_sample(target_positions.size(), sample_size);
  suggestion.reference_target_count = suggestion.reference_target_indices.size();
  std::vector<Vec3> references;
  references.reserve(suggestion.reference_target_count);
  for (const std::size_t index : suggestion.reference_target_indices) {
    const int self_index = target_source_indices.empty()
        ? -1 : target_source_indices[index];
    references.push_back(p2p_dipole_sum(target_positions[index], source_positions,
                                        dipole_moments, OutputFlags::Field,
                                        self_index).H);
  }

  for (const int order : orders) {
    for (const int depth : depths) {
      AccuracyCandidate candidate;
      candidate.order = order;
      candidate.depth = depth;
      const PerformanceCandidate performance = benchmark_candidate(
          source_positions, target_positions, dipole_moments, order, depth,
          backend, repetitions, target_source_indices);
      candidate.evaluation_seconds = performance.evaluation_seconds;
      candidate.reason = performance.reason;
      if (!performance.succeeded) {
        suggestion.candidates.push_back(candidate);
        continue;
      }
      try {
        UniformFmmOptions options;
        options.expansion_order = order;
        options.tree.max_level = depth;
        options.backend = backend;
        UniformFmm fmm({source_positions.begin(), source_positions.end()},
                       {target_positions.begin(), target_positions.end()}, options);
        const auto values = fmm.evaluate(dipole_moments, OutputFlags::Field,
                                         target_source_indices);
        double relative_sum = 0.0;
        double relative_square_sum = 0.0;
        double absolute_sum = 0.0;
        const double reference_scale = std::sqrt(std::accumulate(
            references.begin(), references.end(), 0.0,
            [](const double sum, const Vec3 &field) { return sum + dot(field, field); }) /
            static_cast<double>(references.size()));
        const double denominator_floor =
            std::max(reference_scale * 1.0e-14, 1.0e-300);
        for (std::size_t sample = 0; sample < references.size(); ++sample) {
          const Vec3 difference =
              values[suggestion.reference_target_indices[sample]].H - references[sample];
          const double absolute = std::sqrt(dot(difference, difference));
          const double relative = absolute /
              std::max(std::sqrt(dot(references[sample], references[sample])),
                       denominator_floor);
          absolute_sum += absolute;
          relative_sum += relative;
          relative_square_sum += relative * relative;
          candidate.maximum_absolute_error =
              std::max(candidate.maximum_absolute_error, absolute);
          candidate.maximum_relative_error =
              std::max(candidate.maximum_relative_error, relative);
        }
        const double count = static_cast<double>(references.size());
        candidate.mean_absolute_error = absolute_sum / count;
        candidate.mean_relative_error = relative_sum / count;
        candidate.rms_relative_error = std::sqrt(relative_square_sum / count);
        candidate.satisfies_accuracy =
            candidate.rms_relative_error <= desired_accuracy;
        candidate.succeeded = true;
      } catch (const std::exception &error) {
        candidate.reason = error.what();
      }
      suggestion.candidates.push_back(candidate);
    }
  }
  const AccuracyCandidate *best = nullptr;
  for (const AccuracyCandidate &candidate : suggestion.candidates) {
    if (candidate.succeeded && candidate.satisfies_accuracy &&
        (best == nullptr || candidate.evaluation_seconds < best->evaluation_seconds)) {
      best = &candidate;
    }
  }
  if (best != nullptr) {
    suggestion.suggested_order = best->order;
    suggestion.suggested_depth = best->depth;
  }
  return suggestion;
}

} // namespace cdfmm
