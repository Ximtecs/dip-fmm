// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <span>
#include <vector>

#include "cdfmm/output_flags.hpp"
#include "cdfmm/vec3.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// Validation metrics and helpers
//------------------------------------------------------------------------------

/**
 * @brief Aggregated error statistics for vector-valued operator outputs.
 *
 * The metrics are intended for tests and diagnostic examples that compare
 * approximate operators against direct P2P references. They are not part of
 * production FMM traversal logic.
 */
struct ErrorMetrics {
  double mean_relative_error{0.0};
  double rms_relative_error{0.0};
  double max_relative_error{0.0};
  double mean_absolute_error{0.0};
  double max_absolute_error{0.0};
};

/**
 * @brief Computes the Euclidean norm of a 3D vector.
 *
 * @param v Input vector.
 * @return Euclidean magnitude of @p v.
 */
double norm(const Vec3 &v);

/**
 * @brief Computes absolute vector error between value and reference.
 *
 * @param value Approximate vector.
 * @param reference Reference vector.
 * @return Euclidean norm of value-reference.
 */
double absolute_error(const Vec3 &value, const Vec3 &reference);

/**
 * @brief Computes relative vector error against a reference vector.
 *
 * Uses |value-reference| / max(|reference|, tiny) to avoid division by zero.
 *
 * @param value Approximate vector.
 * @param reference Reference vector.
 * @param tiny Small positive floor for the denominator.
 * @return Relative error.
 */
double relative_error(const Vec3 &value, const Vec3 &reference,
                      double tiny = 1e-14);

/**
 * @brief Computes aggregated error statistics for vector arrays.
 *
 * The spans must have identical length and represent pointwise comparisons.
 *
 * @param values Approximate vectors.
 * @param references Reference vectors.
 * @param tiny Small positive floor for relative-error normalisation.
 * @return Error metrics over all entries.
 */
ErrorMetrics compute_error_metrics(std::span<const Vec3> values,
                                   std::span<const Vec3> references,
                                   double tiny = 1e-14);

/**
 * @brief Evaluates direct P2P references for multiple targets.
 *
 * This helper wraps repeated @c p2p_dipole_sum calls to keep tests and
 * examples concise. It is intentionally simple and intended for validation.
 *
 * @param target_positions Target positions.
 * @param source_positions Source positions.
 * @param dipole_moments Source dipole moments.
 * @param output Output selection (field by default).
 * @return Direct per-target potential/field results.
 */
std::vector<PotentialField> direct_p2p_reference(
    std::span<const Vec3> target_positions, std::span<const Vec3> source_positions,
    std::span<const Vec3> dipole_moments,
    OutputFlags output = OutputFlags::Field);

} // namespace cdfmm
