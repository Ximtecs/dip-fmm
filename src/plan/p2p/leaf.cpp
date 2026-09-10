// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/plan/p2p/leaf.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cdfmm {

StaticP2PLeafPlan build_static_p2p_leaf_plan(
    const StaticP2POperator& operator_map,
    const std::span<const StaticP2PLeafPair> leaf_pairs)
{
    StaticP2PLeafPlan result;
    result.source_count = operator_map.source_count;
    result.target_count = operator_map.target_count;
    if (leaf_pairs.empty()) {
        result.leaf_row_offsets.push_back(0);
        return result;
    }

    std::vector<unsigned char> covered(operator_map.blocks.size(), 0);
    std::set<std::pair<int, int>> occupied_target_ranges;
    std::set<std::pair<int, int>> occupied_source_ranges;
    int previous_target_begin = -1;
    int previous_target_count = -1;

    // Preserve supplied neighbour order: it is also the canonical per-row
    // interaction order used by all exact validation and execution paths.
    for (const StaticP2PLeafPair& pair : leaf_pairs) {
        if (pair.target_begin < 0 || pair.target_count <= 0 ||
            pair.target_begin + pair.target_count > operator_map.target_count ||
            pair.source_begin < 0 || pair.source_count <= 0 ||
            pair.source_begin + pair.source_count > operator_map.source_count) {
            throw std::invalid_argument("static P2P leaf range is invalid");
        }
        if (pair.target_begin != previous_target_begin) {
            if (previous_target_begin >= 0) {
                result.leaf_row_offsets.push_back(
                    static_cast<int>(result.blocks.size()));
            } else {
                result.leaf_row_offsets.push_back(0);
            }
            result.target_begins.push_back(pair.target_begin);
            result.target_counts.push_back(pair.target_count);
            previous_target_begin = pair.target_begin;
            previous_target_count = pair.target_count;
        } else if (pair.target_count != previous_target_count) {
            throw std::invalid_argument("inconsistent target leaf range");
        }

        StaticP2PLeafBlock leaf_block;
        leaf_block.source_begin = pair.source_begin;
        leaf_block.source_count = pair.source_count;
        leaf_block.tensor_offset = result.tensors[0].size();
        result.blocks.push_back(leaf_block);
        occupied_target_ranges.emplace(pair.target_begin, pair.target_count);
        occupied_source_ranges.emplace(pair.source_begin, pair.source_count);

        for (int target = pair.target_begin;
             target < pair.target_begin + pair.target_count; ++target) {
            int expected_source = pair.source_begin;
            for (int entry = operator_map.row_offsets[
                     static_cast<std::size_t>(target)];
                 entry < operator_map.row_offsets[
                     static_cast<std::size_t>(target) + 1]; ++entry) {
                const StaticDipoleBlock& block =
                    operator_map.blocks[static_cast<std::size_t>(entry)];
                if (block.source < pair.source_begin ||
                    block.source >= pair.source_begin + pair.source_count) {
                    continue;
                }
                if (block.source != expected_source || covered[entry] != 0) {
                    throw std::invalid_argument(
                        "static P2P leaf pairs are not dense and unique");
                }
                covered[entry] = 1;
                ++expected_source;
                result.tensors[0].push_back(block.xx);
                result.tensors[1].push_back(block.xy);
                result.tensors[2].push_back(block.xz);
                result.tensors[3].push_back(block.yy);
                result.tensors[4].push_back(block.yz);
                result.tensors[5].push_back(block.zz);
            }
            if (expected_source != pair.source_begin + pair.source_count) {
                throw std::invalid_argument(
                    "static P2P leaf pair omits canonical interactions");
            }
        }
    }
    result.leaf_row_offsets.push_back(static_cast<int>(result.blocks.size()));
    if (std::find(covered.begin(), covered.end(), 0) != covered.end()) {
        throw std::invalid_argument(
            "static P2P leaf pairs do not cover the canonical operator");
    }

    std::set<int> occupancies;
    std::size_t occupancy_sum = 0;
    result.minimum_occupancy = std::numeric_limits<int>::max();
    const auto record_occupancies = [&](const auto& ranges) {
        for (const auto [begin, count] : ranges) {
            static_cast<void>(begin);
            result.minimum_occupancy =
                std::min(result.minimum_occupancy, count);
            result.maximum_occupancy =
                std::max(result.maximum_occupancy, count);
            occupancy_sum += static_cast<std::size_t>(count);
            occupancies.insert(count);
        }
    };
    record_occupancies(occupied_target_ranges);
    record_occupancies(occupied_source_ranges);
    const std::size_t occupied_range_count =
        occupied_target_ranges.size() + occupied_source_ranges.size();
    result.mean_occupancy = static_cast<double>(occupancy_sum) /
        static_cast<double>(occupied_range_count);
    result.unique_occupancies = static_cast<int>(occupancies.size());
    result.uniform_occupancy = occupancies.size() == 1;
    return result;
}

} // namespace cdfmm
