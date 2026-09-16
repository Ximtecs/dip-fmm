// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace cdfmm {

//------------------------------------------------------------------------------
// oneMKL backend capability query
//------------------------------------------------------------------------------

/** @brief Reports whether this build includes the oneMKL matrix backend. */
[[nodiscard]] bool one_mkl_available() noexcept;

} // namespace cdfmm
