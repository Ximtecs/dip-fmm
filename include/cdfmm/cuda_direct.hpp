// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "cdfmm/operators.hpp"
#include "cdfmm/timings.hpp"

namespace cdfmm {

//------------------------------------------------------------------------------
// CUDA direct reference
//------------------------------------------------------------------------------

/**
 * @brief Owns a persistent device-resident O(N^2) direct-reference plan.
 *
 * Source and target geometry and the optional self-identity map are uploaded
 * during construction. Each evaluation therefore transfers only the changing
 * dipole moments and the requested results.
 */
class CudaDirectPlan {
public:
  /**
   * @brief Constructs a direct CUDA plan for fixed source and target geometry.
   *
   * @param source_positions Source positions in user order.
   * @param target_positions Target positions in user order.
   * @param target_source_indices Optional source identity to omit per target.
   */
  CudaDirectPlan(
      std::span<const Vec3> source_positions,
      std::span<const Vec3> target_positions,
      std::span<const int> target_source_indices = {}
  );

  ~CudaDirectPlan();

  CudaDirectPlan(const CudaDirectPlan &) = delete;
  CudaDirectPlan &operator=(const CudaDirectPlan &) = delete;

  /**
   * @brief Evaluates one dipole-moment state using the fixed device geometry.
   *
   * @param dipole_moments Dipole moments in source order.
   * @param results Output storage with one entry per target.
   * @param output Requested potential and/or magnetic field components.
   */
  void evaluate(
      std::span<const Vec3> dipole_moments,
      std::span<PotentialField> results,
      OutputFlags output = OutputFlags::Field
  );

  /// @brief Returns the number of fixed source positions.
  [[nodiscard]] std::size_t source_count() const noexcept;

  /// @brief Returns the number of fixed target positions.
  [[nodiscard]] std::size_t target_count() const noexcept;

  /// @brief Returns transfer and persistent-storage statistics.
  [[nodiscard]] const CudaPlanStatistics &statistics() const noexcept;

  /// @brief Returns timings for the most recent evaluation.
  [[nodiscard]] const CudaEvaluationTimings &
  evaluation_timings() const noexcept;

private:
  struct Implementation;
  Implementation *implementation_{nullptr};
};

/**
 * @brief Evaluates the O(N^2) direct dipole sum on a CUDA device.
 *
 * This is a numerical and performance reference, not an FMM backend. Geometry
 * is uploaded when this convenience function is called; repeated workloads
 * should use CudaDirectPlan to retain fixed device geometry.
 *
 * @param targets Target positions in user order.
 * @param sources Source positions in user order.
 * @param moments Dipole moments in source order.
 * @param output Requested potential and/or magnetic field components.
 * @param target_source_indices Optional source identity to omit per target.
 * @return Direct values in target order.
 */
[[nodiscard]] std::vector<PotentialField> cuda_direct_p2p_reference(
    std::span<const Vec3> targets,
    std::span<const Vec3> sources,
    std::span<const Vec3> moments,
    OutputFlags output = OutputFlags::Field,
    std::span<const int> target_source_indices = {}
);

} // namespace cdfmm
