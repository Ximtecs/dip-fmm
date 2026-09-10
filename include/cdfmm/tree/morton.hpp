// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>

namespace cdfmm {

/** @brief Interleaves 21 bits from each non-negative box coordinate. */
std::uint64_t morton_encode(int ix, int iy, int iz);

/** @brief Inverts `morton_encode` into `(ix, iy, iz)` box coordinates. */
std::array<int, 3> morton_decode(std::uint64_t morton);

} // namespace cdfmm
