// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace cdfmm {

/** @brief Returns the flat offset of a complete octree level. */
int level_offset(int level);

/** @brief Returns a flat node index from level and box coordinates. */
int node_index(int level, int ix, int iy, int iz);

} // namespace cdfmm
