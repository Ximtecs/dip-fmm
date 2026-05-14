// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "cdfmm/vec3.hpp"
namespace cdfmm {
enum class OutputFlags : unsigned { None = 0u, Potential = 1u, Field = 2u, Both = 3u };
inline OutputFlags operator|(OutputFlags a, OutputFlags b) { return static_cast<OutputFlags>(static_cast<unsigned>(a) | static_cast<unsigned>(b)); }
inline bool has_flag(OutputFlags flags, OutputFlags test) { return (static_cast<unsigned>(flags) & static_cast<unsigned>(test)) != 0u; }
struct PotentialField { double phi{0.0}; Vec3 H{}; };
}
