// SPDX-License-Identifier: Apache-2.0
#pragma once

#ifdef CDFMM_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace cdfmm::detail {

//------------------------------------------------------------------------------
// Profiling support
//------------------------------------------------------------------------------

/**
 * @brief Marks a coarse production phase for an optional timeline profiler.
 *
 * Names are borrowed string literals. The disabled implementation is empty,
 * so production builds make no profiler calls or dynamic allocations.
 */
class ProfileRange {
public:
#ifdef CDFMM_ENABLE_NVTX
  explicit ProfileRange(const char *name) noexcept { nvtxRangePushA(name); }
  ~ProfileRange() { end(); }

  void end() noexcept {
    if (active_) {
      nvtxRangePop();
      active_ = false;
    }
  }
#else
  explicit constexpr ProfileRange(const char *) noexcept {}
  constexpr void end() noexcept {}
#endif

  ProfileRange(const ProfileRange &) = delete;
  ProfileRange &operator=(const ProfileRange &) = delete;

private:
#ifdef CDFMM_ENABLE_NVTX
  bool active_{true};
#endif
};

} // namespace cdfmm::detail
