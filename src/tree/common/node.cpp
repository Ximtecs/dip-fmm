// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/tree/node.hpp"

#include <algorithm>

namespace cdfmm {

bool TreeNode::is_leaf() const
{
    return std::all_of(
        children.begin(),
        children.end(),
        [](const int child) {
            return child < 0;
        }
    );
}

std::size_t TreeNode::source_count() const
{
    return source_end - source_begin;
}

std::size_t TreeNode::target_count() const
{
    return target_end - target_begin;
}

} // namespace cdfmm
