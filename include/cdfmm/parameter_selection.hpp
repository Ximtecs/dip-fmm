// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "cdfmm/uniform_fmm.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Advisory parameter-selection types
//------------------------------------------------------------------------------

/** @brief Timings and status recorded for one tested tree depth. */
struct PerformanceCandidate {
  int depth{0};
  bool succeeded{false};
  std::string reason{};
  double near_seconds{0.0};
  double far_seconds{0.0};
  double evaluation_seconds{0.0};
  double estimated_concurrent_seconds{0.0};
  double balance_ratio{0.0};
};

/** @brief Result of an empirical fixed-order performance sweep. */
struct PerformanceSuggestion {
  int suggested_depth{-1};
  int order{0};
  bool branches_concurrent{false};
  std::vector<PerformanceCandidate> candidates{};
};

/** @brief Accuracy, timing, and status for one tested order/depth pair. */
struct AccuracyCandidate {
  int order{0};
  int depth{0};
  bool succeeded{false};
  bool satisfies_accuracy{false};
  std::string reason{};
  double evaluation_seconds{0.0};
  double mean_relative_error{0.0};
  double rms_relative_error{0.0};
  double maximum_relative_error{0.0};
  double mean_absolute_error{0.0};
  double maximum_absolute_error{0.0};
};

/** @brief Result of an empirical sampled-accuracy parameter sweep. */
struct AccuracySuggestion {
  int suggested_order{-1};
  int suggested_depth{-1};
  double requested_accuracy{0.0};
  std::size_t reference_target_count{0};
  std::vector<std::size_t> reference_target_indices{};
  std::vector<AccuracyCandidate> candidates{};
};

//------------------------------------------------------------------------------
// Public advisory functions
//------------------------------------------------------------------------------

/** @brief Returns deterministic, approximately uniform target sample indices. */
[[nodiscard]] std::vector<std::size_t>
deterministic_target_sample(std::size_t target_count, std::size_t sample_size);

/** @brief Computes max(near, far) / min(near, far). */
[[nodiscard]] double branch_balance_ratio(double near_seconds,
                                          double far_seconds);

/**
 * @brief Suggests a tree depth from measured repeated-evaluation timings.
 *
 * Every candidate constructs the normal `UniformFmm`, warms it up, and records
 * median branch and wall timings. Only the partial CUDA backend currently
 * overlaps near and far work; sequential backends are ranked by measured wall
 * time rather than by the balance heuristic.
 */
[[nodiscard]] PerformanceSuggestion suggest_depth_for_performance(
    std::span<const Vec3> source_positions,
    std::span<const Vec3> target_positions,
    std::span<const Vec3> dipole_moments, int order = 6,
    ExecutionBackend backend = ExecutionBackend::Auto,
    std::span<const int> candidate_depths = {}, int repetitions = 3,
    std::span<const int> target_source_indices = {});

/**
 * @brief Suggests the fastest tested pair meeting sampled RMS relative error.
 *
 * The direct field is computed once for a deterministic target subset. The
 * result is an empirical estimate for this problem, not a global error bound.
 */
[[nodiscard]] AccuracySuggestion suggest_parameters_for_accuracy(
    std::span<const Vec3> source_positions,
    std::span<const Vec3> target_positions,
    std::span<const Vec3> dipole_moments, double desired_accuracy = 1.0e-4,
    ExecutionBackend backend = ExecutionBackend::Auto,
    std::span<const int> candidate_orders = {},
    std::span<const int> candidate_depths = {}, std::size_t sample_size = 128,
    int repetitions = 3, std::span<const int> target_source_indices = {});

} // namespace cdfmm
