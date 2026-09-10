// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace cdfmm {

/**
 * @brief Output selection for potential and magnetic field evaluation.
 *
 * Field-only is the default in operator APIs because H is the main quantity
 * of interest. Potential and field can be requested independently so callers
 * can skip unused work.
 */
enum class OutputFlags : unsigned {
  None = 0u,
  Potential = 1u,
  Field = 2u,
  Both = 3u
};

/// @brief Combines independent output selections.
inline OutputFlags operator|(OutputFlags a, OutputFlags b) {
  return static_cast<OutputFlags>(static_cast<unsigned>(a) |
                                  static_cast<unsigned>(b));
}

/// @brief Tests whether an output selection contains a requested flag.
inline bool has_flag(OutputFlags flags, OutputFlags test) {
  return (static_cast<unsigned>(flags) & static_cast<unsigned>(test)) != 0u;
}

} // namespace cdfmm
