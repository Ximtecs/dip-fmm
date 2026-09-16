// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

namespace cdfmm {

//------------------------------------------------------------------------------
// CUDA backend capability queries
//------------------------------------------------------------------------------

/** @brief Reports whether the library was compiled with CUDA support. */
[[nodiscard]] bool cuda_compiled() noexcept;

/** @brief Reports whether this build can access a CUDA device. */
[[nodiscard]] bool cuda_available() noexcept;

/** @brief Reports whether the O(N^2) CUDA direct reference is available. */
[[nodiscard]] bool cuda_direct_available() noexcept;

/** @brief Reports whether hybrid CUDA static M2L/P2P is available. */
[[nodiscard]] bool cuda_m2l_p2p_available() noexcept;

/** @brief Compatibility alias for `cuda_m2l_p2p_available()`. */
[[nodiscard]] bool cuda_m2l_available() noexcept;

/** @brief Reports whether a complete device-resident CUDA FMM is implemented.
 */
[[nodiscard]] bool cuda_full_available() noexcept;

/** @brief Returns a concise description of the selected CUDA device. */
[[nodiscard]] std::string cuda_device_description();

} // namespace cdfmm
