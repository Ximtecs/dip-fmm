// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/tree/indexing.hpp"

#include <cstddef>

#include "cdfmm/tree/morton.hpp"

namespace cdfmm {

int level_offset(const int level)
{
    // Complete levels preceding level l contain sum_{k=0}^{l-1} 8^k nodes.
    std::size_t numerator = 1;
    for (int i = 0; i < level; ++i) {
        numerator *= 8;
    }
    return static_cast<int>((numerator - 1) / 7);
}

int node_index(const int level, const int ix, const int iy, const int iz)
{
    return level_offset(level) + static_cast<int>(morton_encode(ix, iy, iz));
}

} // namespace cdfmm
