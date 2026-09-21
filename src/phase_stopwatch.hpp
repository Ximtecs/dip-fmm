// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <chrono>

#include "cdfmm/core/timing.hpp"

namespace cdfmm::detail {

/**
 * @brief Level-gated host clock for the solver's phase accounting.
 *
 * A stopwatch is constructed for one timing level (`Coarse` or `Detailed`)
 * and knows whether the plan's selected `TimingLevel` includes it.  When it
 * does not, `start()` and `record()` are a single predictable branch and no
 * clock is read, so a region that the selected level excludes is never
 * entered by a timer (the MagTense rule: the verbosity gate precedes the
 * clock).  Clocks bracket whole phases or whole OpenMP regions, so a
 * recorded interval is caller wall time, never summed thread time.
 */
class PhaseStopwatch {
public:
  using Clock = std::chrono::steady_clock;

  explicit PhaseStopwatch(const bool enabled) noexcept : enabled_(enabled) {}

  /// @brief Whether this stopwatch reads the clock at all.
  [[nodiscard]] bool enabled() const noexcept { return enabled_; }

  /// @brief Marks the beginning of a phase.
  void start() noexcept {
    if (enabled_) {
      start_ = Clock::now();
    }
  }

  /**
   * @brief Adds the interval since the last `start()` or `record()` to
   *        `phase` and starts the next interval.
   *
   * The restart makes consecutive phases share one clock read at their
   * boundary; a phase that is not immediately followed by another simply
   * calls `start()` again later.
   */
  void record(PhaseTiming &phase) noexcept {
    if (enabled_) {
      const Clock::time_point now = Clock::now();
      phase.add(std::chrono::duration<double>(now - start_).count());
      start_ = now;
    }
  }

  /// @brief Seconds since the last `start()` or `record()`, or zero when disabled.
  [[nodiscard]] double elapsed() const noexcept {
    if (!enabled_) {
      return 0.0;
    }
    return std::chrono::duration<double>(Clock::now() - start_).count();
  }

private:
  bool enabled_{false};
  Clock::time_point start_{};
};

/// @brief True when `selected` collects the regions of `required`.
[[nodiscard]] constexpr bool timing_includes(const TimingLevel selected,
                                             const TimingLevel required) noexcept {
  return static_cast<int>(selected) >= static_cast<int>(required);
}

} // namespace cdfmm::detail
