// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "cdfmm/backend/execution.hpp"
#include "cdfmm/math/vec3.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Advisory parameter-selection types
//------------------------------------------------------------------------------

/** @brief Timings and status recorded for one tested tree depth. */
struct PerformanceCandidate {
  int depth{0};                              ///< Tree depth that was constructed and timed.
  bool succeeded{false};                     ///< Whether construction and evaluation completed.
  std::string reason{};                      ///< Failure message when `succeeded` is false.
  double near_seconds{0.0};                  ///< Median near-field (P2P) time per evaluation.
  double far_seconds{0.0};                   ///< Median far-field (P2M..L2P) time per evaluation.
  double evaluation_seconds{0.0};            ///< Median wall time per evaluation.
  double estimated_concurrent_seconds{0.0};  ///< `max(near, far)`, the wall time if the branches overlapped perfectly.
  double balance_ratio{0.0};                 ///< `max(near, far) / min(near, far)`; 1 is perfectly balanced.
};

/** @brief Result of an empirical fixed-order performance sweep. */
struct PerformanceSuggestion {
  int suggested_depth{-1};                   ///< Fastest successful depth, or -1 when none succeeded.
  int order{0};                              ///< Expansion order every candidate used.
  bool branches_concurrent{false};           ///< Whether the backend overlaps near and far work (ranking then uses the concurrent estimate).
  std::vector<PerformanceCandidate> candidates{};  ///< Every tested depth in test order.
};

/** @brief Accuracy, timing, and status for one tested order/depth pair. */
struct AccuracyCandidate {
  int order{0};                              ///< Expansion order tested.
  int depth{0};                              ///< Tree depth tested.
  bool succeeded{false};                     ///< Whether construction and evaluation completed.
  bool satisfies_accuracy{false};            ///< Whether the sampled RMS relative error met the request.
  std::string reason{};                      ///< Failure message when `succeeded` is false.
  double evaluation_seconds{0.0};            ///< Median wall time per evaluation.
  double mean_relative_error{0.0};           ///< Mean of |H - H_ref| / |H_ref| over the sampled targets.
  double rms_relative_error{0.0};            ///< Root-mean-square relative field error over the sampled targets.
  double maximum_relative_error{0.0};        ///< Largest relative field error over the sampled targets.
  double mean_absolute_error{0.0};           ///< Mean of |H - H_ref| over the sampled targets.
  double maximum_absolute_error{0.0};        ///< Largest |H - H_ref| over the sampled targets.
};

/** @brief Result of an empirical sampled-accuracy parameter sweep. */
struct AccuracySuggestion {
  int suggested_order{-1};                   ///< Order of the fastest pair meeting the accuracy, or -1.
  int suggested_depth{-1};                   ///< Depth of the fastest pair meeting the accuracy, or -1.
  double requested_accuracy{0.0};            ///< The RMS relative error that was requested.
  std::size_t reference_target_count{0};     ///< Number of targets whose direct reference field was computed.
  std::vector<std::size_t> reference_target_indices{};  ///< User indices of the sampled reference targets.
  std::vector<AccuracyCandidate> candidates{};  ///< Every tested (order, depth) pair in test order.
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
