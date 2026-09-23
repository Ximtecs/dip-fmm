// SPDX-License-Identifier: Apache-2.0

#include "fmm/p2p_construction.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "cdfmm/operators/p2p.hpp"

namespace cdfmm::detail::p2p_construction {
namespace {

// Pairs per chunk. The builder's transient state is about 300 bytes per
// pair, so 2^22 pairs bound a chunk near 1.2 GB. Measured at 4e8 pairs
// against 2^24: lower peak and no slower (the repeated per-chunk exact-reuse
// classification costs less than the larger working set).
constexpr std::size_t kDefaultChunkPairs = std::size_t{1} << 22;

std::atomic<std::size_t> chunk_budget_override{0};

[[nodiscard]] double canonical_zero(const double value) noexcept {
  return value == 0.0 ? 0.0 : value;
}

// A record is identified by its two leaf ranges and its periodic shift. The
// shift is compared bitwise after folding -0.0 onto +0.0, exactly as the
// builder's reciprocity lookup compares shifts.
struct RecordKey {
  std::size_t target_begin{0};
  std::size_t source_begin{0};
  std::uint64_t shift_x{0};
  std::uint64_t shift_y{0};
  std::uint64_t shift_z{0};

  friend bool operator==(const RecordKey &, const RecordKey &) = default;
};

struct RecordKeyHash {
  std::size_t operator()(const RecordKey &key) const noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const std::uint64_t value :
         {static_cast<std::uint64_t>(key.target_begin),
          static_cast<std::uint64_t>(key.source_begin), key.shift_x,
          key.shift_y, key.shift_z}) {
      hash ^= value;
      hash *= 1099511628211ULL;
    }
    return static_cast<std::size_t>(hash);
  }
};

[[nodiscard]] RecordKey record_key(const std::size_t target_begin,
                                   const std::size_t source_begin,
                                   const Vec3 &shift) noexcept {
  return {target_begin, source_begin,
          std::bit_cast<std::uint64_t>(canonical_zero(shift.x)),
          std::bit_cast<std::uint64_t>(canonical_zero(shift.y)),
          std::bit_cast<std::uint64_t>(canonical_zero(shift.z))};
}

[[nodiscard]] bool exactly_equal(const Vec3 &left, const Vec3 &right) noexcept {
  return left.x == right.x && left.y == right.y && left.z == right.z;
}

[[nodiscard]] bool exactly_equal(const Tetrahedron &left,
                                 const Tetrahedron &right) noexcept {
  for (std::size_t vertex = 0; vertex < 4; ++vertex) {
    if (!exactly_equal(left.vertices[vertex], right.vertices[vertex])) {
      return false;
    }
  }
  return true;
}

// The builder's reciprocity condition (src/operators/p2p.cpp): an exact
// tetrahedron <-> tetrahedron plan whose sources and targets are the same
// bodies in the same order.
[[nodiscard]] bool reciprocal_tetrahedron_layout(const CanonicalInputs &inputs) {
  const bool source_tetrahedron =
      inputs.source_geometry == SourceGeometry::Tetrahedron &&
      inputs.source_model == SourceModel::ExactGeometry;
  const bool target_tetrahedron =
      inputs.target_geometry == TargetGeometry::Tetrahedron &&
      inputs.target_model == TargetModel::ExactGeometry;
  if (!source_tetrahedron || !target_tetrahedron ||
      inputs.sources.size() != inputs.targets.size()) {
    return false;
  }
  const auto source_at = [&](const std::size_t index) -> const Tetrahedron & {
    return inputs.source_tetrahedra[inputs.source_tetrahedra.size() == 1 ? 0 : index];
  };
  const auto target_at = [&](const std::size_t index) -> const Tetrahedron & {
    return inputs.target_tetrahedra[inputs.target_tetrahedra.size() == 1 ? 0 : index];
  };
  for (std::size_t index = 0; index < inputs.sources.size(); ++index) {
    if (!exactly_equal(inputs.sources[index], inputs.targets[index]) ||
        !exactly_equal(source_at(index), target_at(index))) {
      return false;
    }
  }
  return true;
}

} // namespace

// A periodic topology lists one record per image of a (target leaf, source
// leaf) pair; the canonical rows order the images of a source by their shift,
// so each record is tagged with its ordinal in that order.
std::vector<StaticP2PLeafPair>
leaf_pairs_from_topology(const StaticFmmTopology &topology) {
  const auto &records = topology.p2p_leaf_records;
  std::vector<StaticP2PLeafPair> leaf_pairs(records.size());
  for (std::size_t row = 0; row < records.size(); ++row) {
    const StaticP2PLeafRecord &record = records[row];
    leaf_pairs[row] = {static_cast<int>(record.target_begin),
                       static_cast<int>(record.target_count),
                       static_cast<int>(record.source_begin),
                       static_cast<int>(record.source_count), 0, 1};
  }
  const auto &offsets = topology.p2p_target_leaf_offsets;
  std::vector<std::size_t> group;
  for (std::size_t leaf = 0; leaf + 1 < offsets.size(); ++leaf) {
    const int begin = offsets[leaf];
    const int end = offsets[leaf + 1];
    for (int first = begin; first < end; ++first) {
      const StaticP2PLeafRecord &lead = records[static_cast<std::size_t>(first)];
      group.clear();
      for (int row = begin; row < end; ++row) {
        const StaticP2PLeafRecord &other = records[static_cast<std::size_t>(row)];
        if (other.source_begin == lead.source_begin &&
            other.source_count == lead.source_count) {
          group.push_back(static_cast<std::size_t>(row));
        }
      }
      if (group.size() <= 1 || group.front() != static_cast<std::size_t>(first)) {
        continue;
      }
      // Same comparison as the canonical builder's interaction sort.
      std::sort(group.begin(), group.end(), [&](const std::size_t left,
                                                const std::size_t right) {
        const Vec3 &a = records[left].source_shift;
        const Vec3 &b = records[right].source_shift;
        return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z);
      });
      for (std::size_t ordinal = 0; ordinal < group.size(); ++ordinal) {
        leaf_pairs[group[ordinal]].image_ordinal = static_cast<int>(ordinal);
        leaf_pairs[group[ordinal]].image_count = static_cast<int>(group.size());
      }
    }
  }
  return leaf_pairs;
}

std::size_t chunk_pair_budget() noexcept {
  const std::size_t override_pairs = chunk_budget_override.load();
  return override_pairs != 0 ? override_pairs : kDefaultChunkPairs;
}

void set_chunk_pair_budget_for_testing(const std::size_t pairs) noexcept {
  chunk_budget_override.store(pairs);
}

std::vector<Chunk> plan_chunks(const StaticFmmTopology &topology,
                               const std::size_t pair_budget) {
  const auto &records = topology.p2p_leaf_records;
  const auto &offsets = topology.p2p_target_leaf_offsets;
  std::vector<Chunk> chunks;
  if (records.empty()) {
    return chunks;
  }
  if (offsets.empty() || static_cast<std::size_t>(offsets.back()) != records.size()) {
    throw std::logic_error("list-1 records are not grouped by target leaf");
  }
  Chunk current;
  bool open = false;
  int previous_target_end = 0;
  for (std::size_t leaf = 0; leaf + 1 < offsets.size(); ++leaf) {
    const std::size_t begin = static_cast<std::size_t>(offsets[leaf]);
    const std::size_t end = static_cast<std::size_t>(offsets[leaf + 1]);
    if (begin == end) {
      continue;
    }
    std::size_t pairs = 0;
    for (std::size_t row = begin; row < end; ++row) {
      pairs += records[row].target_count * records[row].source_count;
    }
    const int target_begin = static_cast<int>(records[begin].target_begin);
    const int target_end =
        static_cast<int>(records[begin].target_begin + records[begin].target_count);
    // Rows are appended in target order, so the leaves must be sorted and
    // disjoint.
    if (target_begin < previous_target_end) {
      throw std::logic_error("list-1 target leaves are not in sorted order");
    }
    previous_target_end = target_end;
    // A leaf is never split, so one leaf above the budget is its own chunk.
    if (open && current.pairs + pairs > pair_budget) {
      chunks.push_back(current);
      open = false;
    }
    if (!open) {
      current = {begin, end, target_begin, target_end, 0};
      open = true;
    }
    current.record_end = end;
    current.target_end = target_end;
    current.pairs += pairs;
  }
  if (open) {
    chunks.push_back(current);
  }
  return chunks;
}

ChunkBuilder::ChunkBuilder(const StaticFmmTopology &topology,
                           const CanonicalInputs &inputs)
    : topology_(topology), inputs_(inputs),
      reciprocal_(reciprocal_tetrahedron_layout(inputs)) {
  if (!reciprocal_) {
    return;
  }
  // Index every record by its key so the reverse of a record -- target and
  // source leaves swapped, shift negated -- is found in constant time.
  const auto &records = topology.p2p_leaf_records;
  std::unordered_map<RecordKey, std::size_t, RecordKeyHash> index;
  index.reserve(records.size());
  for (std::size_t row = 0; row < records.size(); ++row) {
    index.emplace(record_key(records[row].target_begin, records[row].source_begin,
                             records[row].source_shift),
                  row);
  }
  reverse_record_.assign(records.size(), std::numeric_limits<std::size_t>::max());
  for (std::size_t row = 0; row < records.size(); ++row) {
    const StaticP2PLeafRecord &record = records[row];
    const Vec3 negated{-record.source_shift.x, -record.source_shift.y,
                       -record.source_shift.z};
    const auto found =
        index.find(record_key(record.source_begin, record.target_begin, negated));
    if (found != index.end()) {
      reverse_record_[row] = found->second;
    }
  }
}

std::span<const StaticP2PLeafPair>
ChunkBuilder::leaf_pairs(const Chunk &chunk) const {
  if (!leaf_pairs_ready_) {
    leaf_pairs_ = leaf_pairs_from_topology(topology_);
    leaf_pairs_ready_ = true;
  }
  return std::span<const StaticP2PLeafPair>(leaf_pairs_)
      .subspan(chunk.record_begin, chunk.record_end - chunk.record_begin);
}

StaticP2POperator ChunkBuilder::build(const Chunk &chunk,
                                      PhaseTiming &interaction_setup,
                                      PhaseTiming &tensor_build,
                                      const bool timed) const {
  using Clock = std::chrono::steady_clock;
  const auto setup_start = timed ? Clock::now() : Clock::time_point{};
  const auto &records = topology_.p2p_leaf_records;
  // Expand the records exactly as the monolithic build does: free space gives
  // every pair the identity marker and no shift, a periodic plan takes both
  // from its record.
  std::vector<StaticP2PInteraction> interactions;
  interactions.reserve(chunk.pairs);
  const auto expand = [&](const StaticP2PLeafRecord &record, const bool reversed) {
    const bool skip = inputs_.periodic ? record.skip_for_identity : true;
    const Vec3 shift = inputs_.periodic ? record.source_shift : Vec3{};
    for (int target = static_cast<int>(record.target_begin);
         target < static_cast<int>(record.target_begin + record.target_count);
         ++target) {
      for (int source = static_cast<int>(record.source_begin);
           source < static_cast<int>(record.source_begin + record.source_count);
           ++source) {
        if (reversed) {
          interactions.push_back({source, target, {-shift.x, -shift.y, -shift.z}, skip});
        } else {
          interactions.push_back({target, source, shift, skip});
        }
      }
    }
  };
  for (std::size_t row = chunk.record_begin; row < chunk.record_end; ++row) {
    expand(records[row], false);
  }
  // Reciprocity: a pair whose source lies before the chunk takes its tensor
  // from the reverse pair, which the monolithic build owns in the earlier
  // row. Adding that reverse pair lets the builder share it exactly as it
  // would monolithically; its rows are outside the chunk and are dropped.
  if (reciprocal_) {
    for (std::size_t row = chunk.record_begin; row < chunk.record_end; ++row) {
      const StaticP2PLeafRecord &record = records[row];
      const std::size_t reverse = reverse_record_[row];
      if (static_cast<int>(record.source_begin) < chunk.target_begin &&
          reverse != std::numeric_limits<std::size_t>::max()) {
        // The reverse record's own marker is what the monolithic build gives
        // the reverse pairs.
        const StaticP2PLeafRecord &partner = records[reverse];
        const bool skip = inputs_.periodic ? partner.skip_for_identity : true;
        for (int target = static_cast<int>(partner.target_begin);
             target < static_cast<int>(partner.target_begin + partner.target_count);
             ++target) {
          for (int source = static_cast<int>(partner.source_begin);
               source < static_cast<int>(partner.source_begin + partner.source_count);
               ++source) {
            interactions.push_back(
                {target, source,
                 inputs_.periodic ? partner.source_shift : Vec3{}, skip});
          }
        }
      }
    }
  }
  if (timed) {
    interaction_setup.add(
        std::chrono::duration<double>(Clock::now() - setup_start).count());
  }

  const auto tensor_start = timed ? Clock::now() : Clock::time_point{};
  StaticP2POperator built = build_static_p2p_operator(
      inputs_.targets, inputs_.sources, interactions, inputs_.source_geometry,
      inputs_.source_sizes, inputs_.source_tetrahedra, inputs_.target_geometry,
      inputs_.target_sizes, inputs_.target_tetrahedra, inputs_.source_model,
      inputs_.target_model);
  interactions = {};
  // Keep only the chunk's rows: augmented reverse rows lie before it.
  const std::size_t first = static_cast<std::size_t>(
      built.row_offsets[static_cast<std::size_t>(chunk.target_begin)]);
  const std::size_t last = static_cast<std::size_t>(
      built.row_offsets[static_cast<std::size_t>(chunk.target_end)]);
  if (first != 0 || last != built.blocks.size()) {
    std::vector<StaticDipoleBlock> kept(built.blocks.begin() + static_cast<std::ptrdiff_t>(first),
                                        built.blocks.begin() + static_cast<std::ptrdiff_t>(last));
    built.blocks = std::move(kept);
    for (std::size_t target = 0; target < built.row_offsets.size(); ++target) {
      const std::size_t offset = static_cast<std::size_t>(built.row_offsets[target]);
      built.row_offsets[target] =
          static_cast<int>(std::clamp(offset, first, last) - first);
    }
  }
  if (timed) {
    tensor_build.add(
        std::chrono::duration<double>(Clock::now() - tensor_start).count());
  }
  return built;
}

template <typename Rows>
void start_rows(Rows &total, const int source_count, const int target_count,
                const std::size_t pairs, const bool with_tensors) {
  total = {};
  total.source_count = source_count;
  total.target_count = target_count;
  total.row_offsets.reserve(static_cast<std::size_t>(target_count) + 1);
  total.row_offsets.push_back(0);
  if constexpr (requires { total.blocks; }) {
    total.blocks.reserve(pairs);
  } else {
    total.source_indices.reserve(pairs);
    total.skip_for_identity.reserve(pairs);
    for (auto &component : total.potential) {
      component.reserve(pairs);
    }
    if (with_tensors) {
      for (auto &component : total.tensors) {
        component.reserve(pairs);
      }
    }
  }
}

std::vector<int> canonical_row_offsets(const StaticFmmTopology &topology,
                                       const std::size_t target_count) {
  // Every record contributes one pair per source to each of its targets.
  std::vector<int> row_offsets(target_count + 1, 0);
  for (const StaticP2PLeafRecord &record : topology.p2p_leaf_records) {
    for (std::size_t target = record.target_begin;
         target < record.target_begin + record.target_count; ++target) {
      row_offsets[target + 1] += static_cast<int>(record.source_count);
    }
  }
  for (std::size_t target = 1; target < row_offsets.size(); ++target) {
    row_offsets[target] += row_offsets[target - 1];
  }
  return row_offsets;
}

std::size_t total_pairs(const std::vector<Chunk> &chunks) noexcept {
  std::size_t pairs = 0;
  for (const Chunk &chunk : chunks) {
    pairs += chunk.pairs;
  }
  return pairs;
}

namespace {

// Rows are accumulated in target order: `row_offsets` holds the offsets of the
// targets appended so far, and every target before a chunk that no chunk
// covered gets an empty row.
template <typename Rows>
void pad_rows_to(Rows &total, const int target, const std::size_t size) {
  while (static_cast<int>(total.row_offsets.size()) <= target) {
    total.row_offsets.push_back(static_cast<int>(size));
  }
}

template <typename Rows, typename Chunked>
void append_offsets(Rows &total, const Chunked &chunk, const Chunk &range,
                    const std::size_t base) {
  pad_rows_to(total, range.target_begin, base);
  const int first = chunk.row_offsets[static_cast<std::size_t>(range.target_begin)];
  for (int target = range.target_begin; target < range.target_end; ++target) {
    total.row_offsets.push_back(static_cast<int>(
        base + static_cast<std::size_t>(
                   chunk.row_offsets[static_cast<std::size_t>(target) + 1] - first)));
  }
}

} // namespace

template <typename Operator>
void append_rows(Operator &total, const Operator &chunk, const Chunk &range) {
  const std::size_t base = total.blocks.size();
  append_offsets(total, chunk, range, base);
  const std::size_t first = static_cast<std::size_t>(
      chunk.row_offsets[static_cast<std::size_t>(range.target_begin)]);
  const std::size_t last = static_cast<std::size_t>(
      chunk.row_offsets[static_cast<std::size_t>(range.target_end)]);
  total.blocks.insert(total.blocks.end(),
                      chunk.blocks.begin() + static_cast<std::ptrdiff_t>(first),
                      chunk.blocks.begin() + static_cast<std::ptrdiff_t>(last));
}

template <typename Compact>
void append_compact_rows(Compact &total, const Compact &chunk, const Chunk &range,
                         const bool with_tensors) {
  const std::size_t base = total.source_indices.size();
  append_offsets(total, chunk, range, base);
  const auto first = static_cast<std::ptrdiff_t>(
      chunk.row_offsets[static_cast<std::size_t>(range.target_begin)]);
  const auto last = static_cast<std::ptrdiff_t>(
      chunk.row_offsets[static_cast<std::size_t>(range.target_end)]);
  const auto append = [&](auto &destination, const auto &source) {
    destination.insert(destination.end(), source.begin() + first, source.begin() + last);
  };
  append(total.source_indices, chunk.source_indices);
  append(total.skip_for_identity, chunk.skip_for_identity);
  for (std::size_t component = 0; component < 3; ++component) {
    append(total.potential[component], chunk.potential[component]);
  }
  if (with_tensors) {
    for (std::size_t component = 0; component < 6; ++component) {
      append(total.tensors[component], chunk.tensors[component]);
    }
  }
}

template <typename Rows> void finish_rows(Rows &total) {
  std::size_t size = 0;
  if constexpr (requires { total.blocks; }) {
    size = total.blocks.size();
  } else {
    size = total.source_indices.size();
  }
  pad_rows_to(total, total.target_count, size);
}

template <typename Leaf> void append_leaf_blocks(Leaf &total, Leaf &&chunk) {
  if (total.leaf_row_offsets.empty()) {
    total.leaf_row_offsets.push_back(0);
  }
  const std::size_t tensor_base = total.tensors[0].size();
  const int block_base = static_cast<int>(total.blocks.size());
  total.target_begins.insert(total.target_begins.end(), chunk.target_begins.begin(),
                             chunk.target_begins.end());
  total.target_counts.insert(total.target_counts.end(), chunk.target_counts.begin(),
                             chunk.target_counts.end());
  for (std::size_t leaf = 1; leaf < chunk.leaf_row_offsets.size(); ++leaf) {
    total.leaf_row_offsets.push_back(block_base + chunk.leaf_row_offsets[leaf]);
  }
  for (StaticP2PLeafBlock block : chunk.blocks) {
    block.tensor_offset += tensor_base;
    total.blocks.push_back(block);
  }
  for (std::size_t component = 0; component < 6; ++component) {
    total.tensors[component].insert(total.tensors[component].end(),
                                    chunk.tensors[component].begin(),
                                    chunk.tensors[component].end());
  }
  chunk = {};
}

template <typename Leaf> void finish_leaf_plan(Leaf &total) {
  if (total.leaf_row_offsets.empty()) {
    total.leaf_row_offsets.push_back(0);
  }
  // The builder's statistics range over the distinct target ranges and the
  // distinct source ranges of all leaf pairs (src/plan/p2p/leaf.cpp).
  std::set<std::pair<int, int>> target_ranges;
  std::set<std::pair<int, int>> source_ranges;
  for (std::size_t leaf = 0; leaf < total.target_begins.size(); ++leaf) {
    target_ranges.emplace(total.target_begins[leaf], total.target_counts[leaf]);
  }
  for (const StaticP2PLeafBlock &block : total.blocks) {
    source_ranges.emplace(block.source_begin, block.source_count);
  }
  if (target_ranges.empty()) {
    return;
  }
  std::set<int> occupancies;
  std::size_t occupancy_sum = 0;
  total.minimum_occupancy = std::numeric_limits<int>::max();
  total.maximum_occupancy = 0;
  for (const auto *ranges : {&target_ranges, &source_ranges}) {
    for (const auto &[begin, count] : *ranges) {
      static_cast<void>(begin);
      total.minimum_occupancy = std::min(total.minimum_occupancy, count);
      total.maximum_occupancy = std::max(total.maximum_occupancy, count);
      occupancy_sum += static_cast<std::size_t>(count);
      occupancies.insert(count);
    }
  }
  total.mean_occupancy = static_cast<double>(occupancy_sum) /
      static_cast<double>(target_ranges.size() + source_ranges.size());
  total.unique_occupancies = static_cast<int>(occupancies.size());
  total.uniform_occupancy = occupancies.size() == 1;
}

template void start_rows(StaticP2POperator &, int, int, std::size_t, bool);
template void start_rows(FloatStaticP2POperator &, int, int, std::size_t, bool);
template void start_rows(StaticP2PCompactPlan &, int, int, std::size_t, bool);
template void start_rows(FloatStaticP2PCompactPlan &, int, int, std::size_t, bool);
template void append_rows(StaticP2POperator &, const StaticP2POperator &, const Chunk &);
template void append_rows(FloatStaticP2POperator &, const FloatStaticP2POperator &,
                          const Chunk &);
template void append_compact_rows(StaticP2PCompactPlan &, const StaticP2PCompactPlan &,
                                  const Chunk &, bool);
template void append_compact_rows(FloatStaticP2PCompactPlan &,
                                  const FloatStaticP2PCompactPlan &, const Chunk &,
                                  bool);
template void finish_rows(StaticP2POperator &);
template void finish_rows(FloatStaticP2POperator &);
template void finish_rows(StaticP2PCompactPlan &);
template void finish_rows(FloatStaticP2PCompactPlan &);
template void append_leaf_blocks(StaticP2PLeafPlan &, StaticP2PLeafPlan &&);
template void append_leaf_blocks(FloatStaticP2PLeafPlan &, FloatStaticP2PLeafPlan &&);
template void finish_leaf_plan(StaticP2PLeafPlan &);
template void finish_leaf_plan(FloatStaticP2PLeafPlan &);

} // namespace cdfmm::detail::p2p_construction
