// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include "cdfmm/plan/m2l.hpp"

namespace cdfmm::detail::mkl {

/** Timings for one grouped gather/GEMM/scatter application. */
struct M2LApplyTimings {
  double gather_seconds{0.0};
  double multiply_seconds{0.0};
  double scatter_seconds{0.0};
};

/** Persistent storage derived from the canonical M2L plan. */
struct M2LStorageStatistics {
  std::size_t metadata_bytes{0};
  std::size_t scratch_bytes{0};
};

/**
 * oneMKL grouped M2L execution state prepared once for repeated application.
 *
 * The implementation owns only execution metadata and mutable scratch. The
 * canonical matrices and scaling remain in the supplied StaticM2LPlan.
 */
class M2LExecutor {
public:
  explicit M2LExecutor(const StaticM2LPlan& plan);
  explicit M2LExecutor(const FloatStaticM2LPlan& plan);
  ~M2LExecutor();

  M2LExecutor(M2LExecutor&&) noexcept;
  M2LExecutor& operator=(M2LExecutor&&) noexcept;
  M2LExecutor(const M2LExecutor&) = delete;
  M2LExecutor& operator=(const M2LExecutor&) = delete;

  [[nodiscard]] M2LApplyTimings apply(
      const StaticM2LPlan& plan, int level,
      std::span<const double> multipoles, std::span<double> locals);
  [[nodiscard]] M2LApplyTimings apply(
      const FloatStaticM2LPlan& plan, int level,
      std::span<const float> multipoles, std::span<float> locals);

  [[nodiscard]] M2LStorageStatistics statistics() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace cdfmm::detail::mkl
