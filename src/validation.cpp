// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/validation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "cdfmm/operators.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Validation metrics and helpers
//------------------------------------------------------------------------------

double norm(const Vec3 &v)
{
  return std::sqrt(dot(v, v));
}

double absolute_error(const Vec3 &value, const Vec3 &reference)
{
  return norm(value - reference);
}

double relative_error(const Vec3 &value, const Vec3 &reference, double tiny)
{
  const double denominator = std::max(norm(reference), tiny);
  return absolute_error(value, reference) / denominator;
}

ErrorMetrics compute_error_metrics(std::span<const Vec3> values,
                                   std::span<const Vec3> references,
                                   double tiny)
{
  if (values.size() != references.size()) {
    throw std::invalid_argument(
        "compute_error_metrics requires equal value/reference lengths");
  }

  ErrorMetrics metrics;
  if (values.empty()) {
    return metrics;
  }

  double sum_relative = 0.0;
  double sum_relative_square = 0.0;
  double sum_absolute = 0.0;

  for (size_t i = 0; i < values.size(); ++i) {
    const double abs_err = absolute_error(values[i], references[i]);
    const double rel_err = relative_error(values[i], references[i], tiny);

    sum_absolute += abs_err;
    sum_relative += rel_err;
    sum_relative_square += rel_err * rel_err;

    metrics.max_absolute_error = std::max(metrics.max_absolute_error, abs_err);
    metrics.max_relative_error = std::max(metrics.max_relative_error, rel_err);
  }

  const double n_inverse = 1.0 / static_cast<double>(values.size());
  metrics.mean_absolute_error = sum_absolute * n_inverse;
  metrics.mean_relative_error = sum_relative * n_inverse;
  metrics.rms_relative_error = std::sqrt(sum_relative_square * n_inverse);

  return metrics;
}

std::vector<PotentialField> direct_p2p_reference(
    std::span<const Vec3> target_positions, std::span<const Vec3> source_positions,
    std::span<const Vec3> dipole_moments, OutputFlags output)
{
  std::vector<PotentialField> reference(target_positions.size());

  // Each target owns an independent accumulation, preserving the serial
  // source-loop order while distributing complete targets across threads.
  #pragma omp parallel for schedule(static) if(target_positions.size() >= 8)
  for (std::ptrdiff_t target_index = 0;
       target_index < static_cast<std::ptrdiff_t>(target_positions.size());
       ++target_index) {
    reference[static_cast<std::size_t>(target_index)] = p2p_dipole_sum(
        target_positions[static_cast<std::size_t>(target_index)],
        source_positions, dipole_moments, output
    );
  }

  return reference;
}

} // namespace cdfmm
