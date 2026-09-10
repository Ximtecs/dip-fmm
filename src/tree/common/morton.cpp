// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/tree/morton.hpp"

namespace cdfmm {

std::uint64_t morton_encode(const int ix, const int iy, const int iz)
{
    std::uint64_t morton = 0;
    // A 64-bit Morton key accommodates 21 bits per coordinate (63 bits total).
    // Each group of three output bits contains x, y, and z respectively.
    for (int bit = 0; bit < 21; ++bit) {
        const std::uint64_t xb = (static_cast<std::uint64_t>(ix) >> bit) & 1ULL;
        const std::uint64_t yb = (static_cast<std::uint64_t>(iy) >> bit) & 1ULL;
        const std::uint64_t zb = (static_cast<std::uint64_t>(iz) >> bit) & 1ULL;
        morton |= (xb << (3 * bit));
        morton |= (yb << (3 * bit + 1));
        morton |= (zb << (3 * bit + 2));
    }
    return morton;
}

std::array<int, 3> morton_decode(const std::uint64_t morton)
{
    int ix = 0;
    int iy = 0;
    int iz = 0;
    for (int bit = 0; bit < 21; ++bit) {
        ix |= static_cast<int>((morton >> (3 * bit)) & 1ULL) << bit;
        iy |= static_cast<int>((morton >> (3 * bit + 1)) & 1ULL) << bit;
        iz |= static_cast<int>((morton >> (3 * bit + 2)) & 1ULL) << bit;
    }
    return {ix, iy, iz};
}

} // namespace cdfmm
