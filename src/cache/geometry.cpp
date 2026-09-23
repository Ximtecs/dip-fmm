// SPDX-License-Identifier: Apache-2.0

#include "cache/internal.hpp"
#include "phase_stopwatch.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

// Persistence of the geometry-dependent plan: tree permutations and topology,
// fixed self identities, P2M plans, M2L connectivity and level metadata, L2P
// evaluators, and canonical P2P operators.
//
// The load path re-validates every cached topology and plan invariant against
// the live tree and topology; an invalid or mismatched file is a cache miss,
// never a solver error. FP32 plans are decoded directly into FP32 storage.
//
// NOTE(cdfmm): geometry_cache_loaded_direct_float is set here and consumed by
// src/fmm/plan_preparation.cpp, which skips FP32 quantisation on a direct hit,
// and by src/fmm/execution_setup.cpp, which promotes the FP32 P2P operator back
// to FP64 to build the signed tensor-dictionary plan.
namespace cdfmm::detail::cache {

namespace {

enum class DictionarySectionKind : std::uint8_t {
  CanonicalRecords = 0,
  SignedDictionary = 1,
};

// Field by field: StaticP2PLeafBlock has padding, which must not reach the
// file (it would make equal plans differ in their bytes).
void write_leaf_blocks(Writer& writer,
                       const std::vector<StaticP2PLeafBlock>& blocks) {
  writer.scalar<std::uint64_t>(blocks.size());
  for (const StaticP2PLeafBlock& block : blocks) {
    writer.scalar(block.source_begin);
    writer.scalar(block.source_count);
    writer.scalar<std::uint64_t>(block.tensor_offset);
    writer.scalar(block.skip_for_identity);
  }
}

std::vector<StaticP2PLeafBlock> read_leaf_blocks(Reader& reader) {
  constexpr std::size_t record_bytes =
      3 * sizeof(int) + sizeof(std::uint64_t);
  const auto count = reader.scalar<std::uint64_t>();
  if (count > std::numeric_limits<std::size_t>::max() / record_bytes) {
    throw std::runtime_error("cached dictionary block count too large");
  }
  // The bytes are claimed first, so a corrupt count fails before allocating.
  const unsigned char* record =
      reader.take_bytes(static_cast<std::size_t>(count) * record_bytes);
  std::vector<StaticP2PLeafBlock> blocks(static_cast<std::size_t>(count));
  for (StaticP2PLeafBlock& block : blocks) {
    std::uint64_t tensor_offset = 0;
    std::memcpy(&block.source_begin, record, sizeof(int));
    std::memcpy(&block.source_count, record + sizeof(int), sizeof(int));
    std::memcpy(&tensor_offset, record + 2 * sizeof(int), sizeof(tensor_offset));
    std::memcpy(&block.skip_for_identity,
                record + 2 * sizeof(int) + sizeof(tensor_offset), sizeof(int));
    block.tensor_offset = static_cast<std::size_t>(tensor_offset);
    record += record_bytes;
  }
  return blocks;
}

template <typename Rows> void write_rows(Writer& writer, const Rows& rows) {
  writer.scalar(rows.source_count);
  writer.scalar(rows.target_count);
  writer.vector(rows.row_offsets);
  writer.vector(rows.source_indices);
  writer.vector(rows.skip_for_identity);
  for (const auto& component : rows.potential) {
    writer.vector(component);
  }
  for (const auto& component : rows.tensors) {
    writer.vector(component);
  }
}

// Point potential rows: empty, or one row per target whose pairs index valid
// sources and carry every per-pair array they were built with.
template <typename Rows>
Rows read_rows(Reader& reader, const std::size_t source_count,
               const std::size_t target_count) {
  using Value = typename decltype(Rows{}.potential)::value_type::value_type;
  Rows rows;
  rows.source_count = reader.scalar<int>();
  rows.target_count = reader.scalar<int>();
  rows.row_offsets = reader.vector<int>();
  rows.source_indices = reader.vector<int>();
  rows.skip_for_identity = reader.vector<unsigned char>();
  for (auto& component : rows.potential) {
    component = reader.vector<Value>();
  }
  for (auto& component : rows.tensors) {
    component = reader.vector<Value>();
  }
  if (rows.row_offsets.empty()) {
    if (rows.source_count != 0 || rows.target_count != 0 ||
        !rows.source_indices.empty()) {
      throw std::runtime_error("cached potential rows are inconsistent");
    }
    return rows;
  }
  const std::size_t pairs = rows.source_indices.size();
  if (static_cast<std::size_t>(rows.source_count) != source_count ||
      static_cast<std::size_t>(rows.target_count) != target_count ||
      rows.row_offsets.size() != target_count + 1 ||
      rows.row_offsets.front() != 0 ||
      static_cast<std::size_t>(rows.row_offsets.back()) != pairs ||
      !std::is_sorted(rows.row_offsets.begin(), rows.row_offsets.end()) ||
      rows.skip_for_identity.size() != pairs) {
    throw std::runtime_error("cached potential rows are inconsistent");
  }
  const auto per_pair = [pairs](const auto& component) {
    return component.empty() || component.size() == pairs;
  };
  if (!std::all_of(rows.potential.begin(), rows.potential.end(), per_pair) ||
      !std::all_of(rows.tensors.begin(), rows.tensors.end(), per_pair)) {
    throw std::runtime_error("cached potential rows are inconsistent");
  }
  for (const int source : rows.source_indices) {
    if (source < 0 || static_cast<std::size_t>(source) >= source_count) {
      throw std::runtime_error("cached potential rows are inconsistent");
    }
  }
  return rows;
}

template <typename Plan>
void write_dictionary(Writer& writer, const Plan& plan) {
  writer.scalar(plan.source_count);
  writer.scalar(plan.target_count);
  writer.vector(plan.target_begins);
  writer.vector(plan.target_counts);
  writer.vector(plan.leaf_row_offsets);
  write_leaf_blocks(writer, plan.blocks);
  writer.vector(plan.tile_leaf_indices);
  writer.vector(plan.tile_target_offsets);
  for (const auto& component : plan.tensors) {
    writer.vector(component);
  }
  writer.scalar(plan.token_width_bytes);
  writer.vector(plan.tokens8);
  writer.vector(plan.tokens16);
  writer.vector(plan.tokens32);
  writer.scalar(plan.zero_variant_id);
  writer.scalar(plan.target_tile_size);
}

template <typename Token>
void require_tokens_below(const std::vector<Token>& tokens,
                          const std::size_t variants) {
  for (const Token token : tokens) {
    if (static_cast<std::size_t>(token) >= variants) {
      throw std::runtime_error("cached dictionary token out of range");
    }
  }
}

// Every index the executors follow is bounds-checked, so a corrupt or
// mismatched section is a miss rather than an out-of-range read.
template <typename Plan>
Plan read_dictionary(Reader& reader, const std::size_t source_count,
                     const std::size_t target_count, const int target_tile_size) {
  using Value = typename decltype(Plan{}.tensors)::value_type::value_type;
  Plan plan;
  plan.source_count = reader.scalar<int>();
  plan.target_count = reader.scalar<int>();
  plan.target_begins = reader.vector<int>();
  plan.target_counts = reader.vector<int>();
  plan.leaf_row_offsets = reader.vector<int>();
  plan.blocks = read_leaf_blocks(reader);
  plan.tile_leaf_indices = reader.vector<int>();
  plan.tile_target_offsets = reader.vector<int>();
  for (auto& component : plan.tensors) {
    component = reader.vector<Value>();
  }
  plan.token_width_bytes = reader.scalar<std::uint8_t>();
  plan.tokens8 = reader.vector<std::uint8_t>();
  plan.tokens16 = reader.vector<std::uint16_t>();
  plan.tokens32 = reader.vector<std::uint32_t>();
  plan.zero_variant_id = reader.scalar<std::uint32_t>();
  plan.target_tile_size = reader.scalar<int>();

  const std::size_t variants = plan.variant_count();
  const std::size_t leaves = plan.target_begins.size();
  const bool dimensions =
      static_cast<std::size_t>(plan.source_count) == source_count &&
      static_cast<std::size_t>(plan.target_count) == target_count &&
      plan.target_tile_size == target_tile_size &&
      plan.target_counts.size() == leaves &&
      plan.leaf_row_offsets.size() == leaves + 1 &&
      plan.leaf_row_offsets.front() == 0 &&
      static_cast<std::size_t>(plan.leaf_row_offsets.back()) ==
          plan.blocks.size() &&
      std::is_sorted(plan.leaf_row_offsets.begin(),
                     plan.leaf_row_offsets.end()) &&
      plan.tile_leaf_indices.size() == plan.tile_target_offsets.size() &&
      variants > 0 && plan.zero_variant_id < variants &&
      std::all_of(plan.tensors.begin(), plan.tensors.end(),
                  [variants](const auto& component) {
                    return component.size() == variants;
                  });
  const std::size_t populated = (plan.tokens8.empty() ? 0U : 1U) +
                                (plan.tokens16.empty() ? 0U : 1U) +
                                (plan.tokens32.empty() ? 0U : 1U);
  const bool width =
      populated <= 1 &&
      ((plan.token_width_bytes == 1 && plan.tokens16.empty() &&
        plan.tokens32.empty()) ||
       (plan.token_width_bytes == 2 && plan.tokens8.empty() &&
        plan.tokens32.empty()) ||
       (plan.token_width_bytes == 4 && plan.tokens8.empty() &&
        plan.tokens16.empty()));
  if (!dimensions || !width) {
    throw std::runtime_error("cached dictionary dimensions mismatch");
  }
  const std::size_t tokens = plan.token_count();
  for (std::size_t leaf = 0; leaf < leaves; ++leaf) {
    const int begin = plan.target_begins[leaf];
    const int count = plan.target_counts[leaf];
    if (begin < 0 || count < 0 ||
        static_cast<std::size_t>(begin) + static_cast<std::size_t>(count) >
            target_count) {
      throw std::runtime_error("cached dictionary target leaf out of range");
    }
    for (int block_index = plan.leaf_row_offsets[leaf];
         block_index < plan.leaf_row_offsets[leaf + 1]; ++block_index) {
      const StaticP2PLeafBlock& block =
          plan.blocks[static_cast<std::size_t>(block_index)];
      const std::size_t extent = static_cast<std::size_t>(block.source_count) *
                                 static_cast<std::size_t>(count);
      if (block.source_begin < 0 || block.source_count < 0 ||
          static_cast<std::size_t>(block.source_begin) +
                  static_cast<std::size_t>(block.source_count) >
              source_count ||
          block.tensor_offset > tokens || extent > tokens - block.tensor_offset) {
        throw std::runtime_error("cached dictionary block out of range");
      }
    }
  }
  for (std::size_t tile = 0; tile < plan.tile_leaf_indices.size(); ++tile) {
    const int leaf = plan.tile_leaf_indices[tile];
    if (leaf < 0 || static_cast<std::size_t>(leaf) >= leaves ||
        plan.tile_target_offsets[tile] < 0 ||
        plan.tile_target_offsets[tile] >=
            std::max(plan.target_counts[static_cast<std::size_t>(leaf)], 1)) {
      throw std::runtime_error("cached dictionary tile out of range");
    }
  }
  require_tokens_below(plan.tokens8, variants);
  require_tokens_below(plan.tokens16, variants);
  require_tokens_below(plan.tokens32, variants);
  return plan;
}

// The trailing section of a dictionary-keyed file (see the payload order).
void read_dictionary_section(Reader& reader,
                             const GeometryCacheIdentity& identity,
                             const StaticFmmTopology& topology,
                             const std::uint64_t canonical_blocks,
                             GeometryCachePayload& payload) {
  const auto kind = reader.scalar<std::uint8_t>();
  const std::size_t source_count = topology.sorted_source_positions.size();
  const std::size_t target_count = topology.sorted_target_positions.size();
  if (kind == static_cast<std::uint8_t>(DictionarySectionKind::CanonicalRecords)) {
    payload.p2p_dictionary.reset();
    payload.p2p_dictionary_float.reset();
    return;
  }
  if (kind != static_cast<std::uint8_t>(DictionarySectionKind::SignedDictionary) ||
      canonical_blocks != 0) {
    throw std::runtime_error("cached dictionary section kind mismatch");
  }
  if (identity.precision == StaticPrecision::Float32) {
    payload.p2p_dictionary_float =
        read_dictionary<FloatStaticP2PSignedTensorDictionaryPlan>(
            reader, source_count, target_count,
            identity.dictionary_target_tile_size);
  } else {
    payload.p2p_dictionary =
        read_dictionary<StaticP2PSignedTensorDictionaryPlan>(
            reader, source_count, target_count,
            identity.dictionary_target_tile_size);
  }
  payload.p2p_compact_plan =
      read_rows<StaticP2PCompactPlan>(reader, source_count, target_count);
  payload.p2p_compact_plan_float =
      read_rows<FloatStaticP2PCompactPlan>(reader, source_count, target_count);
}

} // namespace

// Geometry-plan serialisation is implemented below with field-wise arrays.
// It deliberately excludes the universal translation matrices.
//
// Payload order (write_geometry_cache is the authoritative sequence and
// load_geometry_cache reads it back in the same order):
//   tree permutations and inverse permutations, leaf indices, occupied leaves
//   node count, then every TreeNode field by field
//   fixed identity flag and, if set, the fixed identity map
//   P2M plans: leaf, begin, count, coefficient operator
//   M2L plan metadata: counts, scaling tables, CSR rows and level schedule
//   L2P evaluators: potential row and three field rows each
//   canonical P2P operator: counts, row offsets, block count, blocks
//     (empty -- zero counts and blocks -- for a position-based plan, which
//     persists no pair tensors, and for a dictionary plan that kept its
//     dictionary; both are keyed separately for that reason)
//   dictionary-keyed files only: the dictionary section -- a kind byte, then
//     for a kept dictionary the dictionary in the plan's precision and the
//     FP64 and FP32 point potential rows (empty when the plan has none); for
//     a dictionary that fell back to rows, nothing (the canonical records are
//     the near field)
// The tree and identity sections are compared against the live objects on
// load rather than trusted, so a key collision can never yield a wrong plan.
bool load_geometry_cache(
    const GeometryCacheIdentity& identity, const UniformTree& tree,
    const StaticFmmTopology& topology,
    const std::optional<std::vector<int>>& fixed_target_source_indices,
    GeometryCachePayload payload, bool& geometry_cache_loaded_direct_float,
    StaticPlanStatistics& statistics) {
  if (!identity.enabled) {
    return false;
  }
  detail::PhaseStopwatch clock(
      statistics.timing_level == TimingLevel::Detailed);
  clock.start();
  try {
    const auto file = read_cache(
        cache_path(identity.directory, "plans", identity.geometry_key),
        {CacheKind::Plan, identity.basis, identity.order, identity.precision,
         tree.leaf_level(), identity.geometry_key,
         identity.geometry_hash_digest},
        statistics.cache_bytes_read);
    Reader reader(file);
    const auto require_int_span = [&reader](const std::span<const int> expected) {
      if (!reader.equal_span<int>(expected)) {
        throw std::runtime_error("cached tree index metadata mismatch");
      }
    };
    require_int_span(tree.source_permutation());
    require_int_span(tree.source_inverse_permutation());
    require_int_span(tree.target_permutation());
    require_int_span(tree.target_inverse_permutation());
    require_int_span(tree.leaf_indices());
    require_int_span(tree.occupied_source_leaves());
    require_int_span(tree.occupied_target_leaves());
    const auto node_count = reader.scalar<std::uint64_t>();
    const auto nodes = tree.nodes();
    if (node_count != nodes.size()) {
      throw std::runtime_error("cached tree node count mismatch");
    }
    for (const TreeNode& expected : nodes) {
      const int index = reader.scalar<int>();
      const int level = reader.scalar<int>();
      const int parent = reader.scalar<int>();
      std::array<int, 8> children{};
      for (int& child : children) {
        child = reader.scalar<int>();
      }
      const int ix = reader.scalar<int>();
      const int iy = reader.scalar<int>();
      const int iz = reader.scalar<int>();
      const std::uint64_t morton = reader.scalar<std::uint64_t>();
      const Vec3 centre{reader.scalar<double>(), reader.scalar<double>(),
                        reader.scalar<double>()};
      const double half_width = reader.scalar<double>();
      const std::size_t source_begin = reader.scalar<std::size_t>();
      const std::size_t source_end = reader.scalar<std::size_t>();
      const std::size_t target_begin = reader.scalar<std::size_t>();
      const std::size_t target_end = reader.scalar<std::size_t>();
      if (index != expected.index || level != expected.level ||
          parent != expected.parent || children != expected.children ||
          ix != expected.ix || iy != expected.iy || iz != expected.iz ||
          morton != expected.morton_index || centre.x != expected.centre.x ||
          centre.y != expected.centre.y || centre.z != expected.centre.z ||
          half_width != expected.half_width ||
          source_begin != expected.source_begin ||
          source_end != expected.source_end ||
          target_begin != expected.target_begin ||
          target_end != expected.target_end) {
        throw std::runtime_error("cached tree topology mismatch");
      }
    }
    const bool has_fixed_identities = reader.scalar<bool>();
    if (has_fixed_identities != fixed_target_source_indices.has_value()) {
      throw std::runtime_error("cached self-identity metadata mismatch");
    }
    if (has_fixed_identities) {
      const std::vector<int> cached_identities = reader.vector<int>();
      if (cached_identities != *fixed_target_source_indices) {
        throw std::runtime_error("cached self identities mismatch");
      }
    }
    const auto validate_cached_m2l = [&topology](const auto& plan) {
      const std::size_t node_count = topology.nodes.size();
      const std::size_t interaction_count = plan.source_nodes.size();
      if (plan.level_count != topology.maximum_level + 1 ||
          plan.target_row_offsets.size() != node_count + 1 ||
          plan.target_row_offsets.back() !=
              static_cast<int>(interaction_count) ||
          plan.source_nodes.size() != plan.matrix_ids.size() ||
          plan.source_nodes.size() != plan.interaction_levels.size() ||
          plan.source_levels.size() != interaction_count ||
          plan.target_levels.size() != interaction_count ||
          plan.target_level_offsets.size() !=
              static_cast<std::size_t>(plan.level_count) + 1 ||
          plan.target_nodes_by_level.size() != node_count ||
          plan.node_levels.size() != node_count) {
        throw std::runtime_error("cached canonical M2L metadata dimensions mismatch");
      }
      if (!std::equal(plan.node_levels.begin(), plan.node_levels.end(),
                      topology.nodes.begin(),
                      [](const int level, const StaticFmmTopology::Node& node) {
                        return level == node.level;
                      })) {
        throw std::runtime_error("cached canonical M2L node levels mismatch");
      }
      if (plan.target_level_offsets.front() != 0 ||
          plan.target_level_offsets.back() != static_cast<int>(node_count) ||
          !std::is_sorted(plan.target_level_offsets.begin(),
                          plan.target_level_offsets.end())) {
        throw std::runtime_error("cached canonical M2L target schedule mismatch");
      }
      for (int level = 0; level < plan.level_count; ++level) {
        const int begin = plan.target_level_offsets[static_cast<std::size_t>(level)];
        const int end = plan.target_level_offsets[static_cast<std::size_t>(level + 1)];
        for (int slot = begin; slot < end; ++slot) {
          const int node = plan.target_nodes_by_level[static_cast<std::size_t>(slot)];
          if (node < 0 || node >= static_cast<int>(node_count) ||
              topology.nodes[static_cast<std::size_t>(node)].level != level) {
            throw std::runtime_error("cached canonical M2L target schedule mismatch");
          }
        }
      }
      for (std::size_t target = 0; target < node_count; ++target) {
        const int begin = plan.target_row_offsets[target];
        const int end = plan.target_row_offsets[target + 1];
        if (begin < 0 || end < begin || end > static_cast<int>(interaction_count)) {
          throw std::runtime_error("cached canonical M2L row offsets mismatch");
        }
        for (int slot = begin; slot < end; ++slot) {
          const std::size_t index = static_cast<std::size_t>(slot);
          const int source = plan.source_nodes[index];
          const int target_level = topology.nodes[target].level;
          if (source < 0 || source >= static_cast<int>(node_count) ||
              plan.source_levels[index] !=
                  topology.nodes[static_cast<std::size_t>(source)].level ||
              plan.target_levels[index] != target_level ||
              plan.interaction_levels[index] != target_level) {
            throw std::runtime_error("cached canonical M2L endpoint levels mismatch");
          }
        }
      }
    };
    const auto validate_cached_p2m = [&topology](const auto& plans) {
      if (plans.size() != topology.source_leaves.size()) {
        throw std::runtime_error("cached P2M leaf metadata dimensions mismatch");
      }
      for (std::size_t index = 0; index < plans.size(); ++index) {
        const auto& expected = topology.source_leaves[index];
        if (plans[index].leaf != expected.node ||
            plans[index].begin != expected.begin ||
            plans[index].count != expected.count) {
          throw std::runtime_error("cached P2M leaf metadata mismatch");
        }
      }
    };
    const auto p2m_count = reader.scalar<std::uint64_t>();
    if (identity.precision == StaticPrecision::Float32) {
      payload.p2m_plans_float.clear();
      payload.p2m_plans_float.reserve(static_cast<std::size_t>(p2m_count));
      for (std::uint64_t index = 0; index < p2m_count; ++index) {
        payload.p2m_plans_float.push_back(
            {reader.scalar<int>(), reader.scalar<std::size_t>(),
             reader.scalar<std::size_t>(), read_operator_float(reader)});
      }
      validate_cached_p2m(payload.p2m_plans_float);
      payload.m2l_plan_float.coefficient_count = reader.scalar<int>();
      payload.m2l_plan_float.matrix_count = reader.scalar<int>();
      payload.m2l_plan_float.level_count = reader.scalar<int>();
      payload.m2l_plan_float.multipole_scaling = read_values_float(reader);
      payload.m2l_plan_float.local_scaling = read_values_float(reader);
      payload.m2l_plan_float.target_row_offsets = reader.vector<int>();
      payload.m2l_plan_float.source_nodes = reader.vector<int>();
      payload.m2l_plan_float.matrix_ids = reader.vector<int>();
      payload.m2l_plan_float.source_levels = reader.vector<int>();
      payload.m2l_plan_float.target_levels = reader.vector<int>();
      payload.m2l_plan_float.interaction_levels = reader.vector<int>();
      payload.m2l_plan_float.level_target_begin = reader.vector<int>();
      payload.m2l_plan_float.level_target_end = reader.vector<int>();
      payload.m2l_plan_float.target_level_offsets = reader.vector<int>();
      payload.m2l_plan_float.target_nodes_by_level = reader.vector<int>();
      payload.m2l_plan_float.node_levels = reader.vector<int>();

      const auto l2p_count = reader.scalar<std::uint64_t>();
      payload.l2p_evaluators_float.clear();
      payload.l2p_evaluators_float.reserve(static_cast<std::size_t>(l2p_count));
      for (std::uint64_t index = 0; index < l2p_count; ++index) {
        FloatStaticL2PEvaluator evaluator;
        evaluator.potential = read_values_float(reader);
        for (auto& row : evaluator.field) {
          row = read_values_float(reader);
        }
        payload.l2p_evaluators_float.push_back(std::move(evaluator));
      }
      payload.p2p_operator_float.source_count = reader.scalar<int>();
      payload.p2p_operator_float.target_count = reader.scalar<int>();
      payload.p2p_operator_float.row_offsets = reader.vector<int>();
      const auto block_count = reader.scalar<std::uint64_t>();
      if (identity.position_based_p2p && block_count != 0) {
        throw std::runtime_error("tensor-free plan holds pair tensors");
      }
      read_p2p_blocks_float(reader, block_count, payload.p2p_operator_float);
      validate_cached_m2l(payload.m2l_plan_float);
      if (identity.dictionary_p2p) {
        read_dictionary_section(reader, identity, topology, block_count, payload);
      }
      geometry_cache_loaded_direct_float = true;
    } else {
      payload.p2m_plans.clear();
      payload.p2m_plans.reserve(static_cast<std::size_t>(p2m_count));
      for (std::uint64_t index = 0; index < p2m_count; ++index) {
        payload.p2m_plans.push_back(
            {reader.scalar<int>(), reader.scalar<std::size_t>(),
             reader.scalar<std::size_t>(),
             read_operator(reader, identity.precision)});
      }
      validate_cached_p2m(payload.p2m_plans);
      payload.m2l_plan.coefficient_count = reader.scalar<int>();
      payload.m2l_plan.matrix_count = reader.scalar<int>();
      payload.m2l_plan.level_count = reader.scalar<int>();
      payload.m2l_plan.multipole_scaling = read_values(reader, identity.precision);
      payload.m2l_plan.local_scaling = read_values(reader, identity.precision);
      payload.m2l_plan.target_row_offsets = reader.vector<int>();
      payload.m2l_plan.source_nodes = reader.vector<int>();
      payload.m2l_plan.matrix_ids = reader.vector<int>();
      payload.m2l_plan.source_levels = reader.vector<int>();
      payload.m2l_plan.target_levels = reader.vector<int>();
      payload.m2l_plan.interaction_levels = reader.vector<int>();
      payload.m2l_plan.level_target_begin = reader.vector<int>();
      payload.m2l_plan.level_target_end = reader.vector<int>();
      payload.m2l_plan.target_level_offsets = reader.vector<int>();
      payload.m2l_plan.target_nodes_by_level = reader.vector<int>();
      payload.m2l_plan.node_levels = reader.vector<int>();

      const auto l2p_count = reader.scalar<std::uint64_t>();
      payload.l2p_evaluators.clear();
      payload.l2p_evaluators.reserve(static_cast<std::size_t>(l2p_count));
      for (std::uint64_t index = 0; index < l2p_count; ++index) {
        StaticL2PEvaluator evaluator;
        evaluator.potential = read_values(reader, identity.precision);
        for (auto& row : evaluator.field) {
          row = read_values(reader, identity.precision);
        }
        payload.l2p_evaluators.push_back(std::move(evaluator));
      }
      payload.p2p_operator.source_count = reader.scalar<int>();
      payload.p2p_operator.target_count = reader.scalar<int>();
      payload.p2p_operator.row_offsets = reader.vector<int>();
      const auto block_count = reader.scalar<std::uint64_t>();
      if (identity.position_based_p2p && block_count != 0) {
        throw std::runtime_error("tensor-free plan holds pair tensors");
      }
      read_p2p_blocks(reader, block_count, identity.precision,
                      payload.p2p_operator, payload.p2p_compact_plan);
      validate_cached_m2l(payload.m2l_plan);
      if (identity.dictionary_p2p) {
        read_dictionary_section(reader, identity, topology, block_count, payload);
      }
    }
    reader.require_end();
    statistics.geometry_cache_hit = true;
    if (clock.enabled()) {
      statistics.geometry_cache_load.add(clock.elapsed());
    }
    if (clock.enabled()) {
      statistics.geometry_cache_lookup.add(clock.elapsed());
    }
    return true;
  } catch (const std::exception&) {
    // Any failure is a miss.  Only the FP32 members are reset because the
    // FP64 members are overwritten by a cold build regardless, whereas the
    // FP32 members would otherwise be taken as a partial direct load.
    geometry_cache_loaded_direct_float = false;
    payload.p2m_plans_float.clear();
    payload.l2p_evaluators_float.clear();
    payload.p2p_operator_float = {};
    payload.p2p_compact_plan_float = {};
    payload.p2p_bsr_plan_float = {};
    payload.m2l_plan_float = {};
    payload.p2p_dictionary.reset();
    payload.p2p_dictionary_float.reset();
    if (clock.enabled()) {
      statistics.geometry_cache_lookup.add(clock.elapsed());
    }
    return false;
  }
}

GeometryCacheWriter::GeometryCacheWriter(
    const GeometryCacheIdentity& identity, const UniformTree& tree,
    const std::optional<std::vector<int>>& fixed_target_source_indices,
    const std::vector<P2MPlan>& p2m_plans, const StaticM2LPlan& m2l_plan,
    const std::vector<StaticL2PEvaluator>& l2p_evaluators,
    const int p2p_source_count, const int p2p_target_count,
    const std::span<const int> p2p_row_offsets,
    const std::uint64_t p2p_block_count, StaticPlanStatistics& statistics)
    : precision_(identity.precision), dictionary_p2p_(identity.dictionary_p2p),
      expected_blocks_(p2p_block_count),
      timed_(statistics.timing_level == TimingLevel::Detailed),
      statistics_(statistics) {
  if (!identity.enabled) {
    return;
  }
  detail::PhaseStopwatch clock(timed_);
  clock.start();
  stream_ = std::make_unique<CacheFileStream>(
      cache_path(identity.directory, "plans", identity.geometry_key),
      CacheDescriptor{CacheKind::Plan, identity.basis, identity.order,
                      identity.precision, tree.leaf_level(), identity.geometry_key,
                      identity.geometry_hash_digest});
  Writer payload;
  payload.reserve(4 * 1024 * 1024);
  payload.span(tree.source_permutation());
  payload.span(tree.source_inverse_permutation());
  payload.span(tree.target_permutation());
  payload.span(tree.target_inverse_permutation());
  payload.span(tree.leaf_indices());
  payload.span(tree.occupied_source_leaves());
  payload.span(tree.occupied_target_leaves());
  payload.scalar<std::uint64_t>(tree.nodes().size());
  for (const TreeNode& node : tree.nodes()) {
    payload.scalar(node.index);
    payload.scalar(node.level);
    payload.scalar(node.parent);
    for (const int child : node.children) {
      payload.scalar(child);
    }
    payload.scalar(node.ix);
    payload.scalar(node.iy);
    payload.scalar(node.iz);
    payload.scalar(node.morton_index);
    payload.scalar(node.centre.x);
    payload.scalar(node.centre.y);
    payload.scalar(node.centre.z);
    payload.scalar(node.half_width);
    payload.scalar(node.source_begin);
    payload.scalar(node.source_end);
    payload.scalar(node.target_begin);
    payload.scalar(node.target_end);
  }
  payload.scalar(fixed_target_source_indices.has_value());
  if (fixed_target_source_indices) {
    payload.vector(*fixed_target_source_indices);
  }
  payload.scalar<std::uint64_t>(p2m_plans.size());
  for (const P2MPlan& plan : p2m_plans) {
    payload.scalar(plan.leaf);
    payload.scalar(plan.begin);
    payload.scalar(plan.count);
    write_operator(payload, plan.operator_map, identity.precision);
  }
  payload.scalar(m2l_plan.coefficient_count);
  payload.scalar(m2l_plan.matrix_count);
  payload.scalar(m2l_plan.level_count);
  write_values(payload, m2l_plan.multipole_scaling, identity.precision);
  write_values(payload, m2l_plan.local_scaling, identity.precision);
  payload.vector(m2l_plan.target_row_offsets);
  payload.vector(m2l_plan.source_nodes);
  payload.vector(m2l_plan.matrix_ids);
  payload.vector(m2l_plan.source_levels);
  payload.vector(m2l_plan.target_levels);
  payload.vector(m2l_plan.interaction_levels);
  payload.vector(m2l_plan.level_target_begin);
  payload.vector(m2l_plan.level_target_end);
  payload.vector(m2l_plan.target_level_offsets);
  payload.vector(m2l_plan.target_nodes_by_level);
  payload.vector(m2l_plan.node_levels);
  payload.scalar<std::uint64_t>(l2p_evaluators.size());
  for (const StaticL2PEvaluator& evaluator : l2p_evaluators) {
    write_values(payload, evaluator.potential, identity.precision);
    for (const auto& row : evaluator.field) {
      write_values(payload, row, identity.precision);
    }
  }
  payload.scalar(p2p_source_count);
  payload.scalar(p2p_target_count);
  payload.span(p2p_row_offsets);
  payload.scalar<std::uint64_t>(p2p_block_count);
  stream_->append(payload);
  if (clock.enabled()) {
    seconds_ += clock.elapsed();
  }
}

GeometryCacheWriter::~GeometryCacheWriter() = default;

void GeometryCacheWriter::append_p2p_blocks(
    const std::span<const StaticDipoleBlock> blocks) {
  if (!stream_) {
    return;
  }
  detail::PhaseStopwatch clock(timed_);
  clock.start();
  // Encode in bounded pieces so the staging buffer stays small.
  constexpr std::size_t piece = std::size_t{1} << 20;
  for (std::size_t first = 0; first < blocks.size(); first += piece) {
    const std::size_t count = std::min(piece, blocks.size() - first);
    Writer records;
    write_p2p_blocks(records, blocks.subspan(first, count), precision_);
    stream_->append(records);
  }
  written_blocks_ += blocks.size();
  if (clock.enabled()) {
    seconds_ += clock.elapsed();
  }
}

void GeometryCacheWriter::append_dictionary_section(
    const StaticP2PSignedTensorDictionaryPlan* dictionary,
    const FloatStaticP2PSignedTensorDictionaryPlan* dictionary_float,
    const StaticP2PCompactPlan& potential_rows,
    const FloatStaticP2PCompactPlan& potential_rows_float) {
  if (!stream_ || !dictionary_p2p_ || dictionary_section_written_) {
    return;
  }
  detail::PhaseStopwatch clock(timed_);
  clock.start();
  // The stored dictionary is the one the plan's precision executes; a
  // mismatched one would be read back as the wrong type, so it is dropped.
  const bool fp32 = precision_ == StaticPrecision::Float32;
  if ((fp32 && dictionary != nullptr) || (!fp32 && dictionary_float != nullptr)) {
    return;
  }
  Writer section;
  if (dictionary == nullptr && dictionary_float == nullptr) {
    section.scalar(static_cast<std::uint8_t>(DictionarySectionKind::CanonicalRecords));
  } else {
    section.scalar(static_cast<std::uint8_t>(DictionarySectionKind::SignedDictionary));
    if (fp32) {
      write_dictionary(section, *dictionary_float);
    } else {
      write_dictionary(section, *dictionary);
    }
    write_rows(section, potential_rows);
    write_rows(section, potential_rows_float);
  }
  stream_->append(section);
  dictionary_section_written_ = true;
  if (clock.enabled()) {
    seconds_ += clock.elapsed();
  }
}

void GeometryCacheWriter::finish() {
  if (!stream_) {
    return;
  }
  detail::PhaseStopwatch clock(timed_);
  clock.start();
  // A record count that disagrees with the announced one, or a
  // dictionary-keyed file without its section, would write a file the
  // reader rejects; it is dropped instead.
  const bool complete = written_blocks_ == expected_blocks_ &&
                        (!dictionary_p2p_ || dictionary_section_written_);
  const std::size_t bytes = complete ? stream_->commit() : 0;
  stream_.reset();
  statistics_.cache_bytes_written += bytes;
  if (clock.enabled()) {
    seconds_ += clock.elapsed();
  }
  if (timed_) {
    statistics_.geometry_cache_write.add(seconds_);
  }
}

void write_geometry_cache(
    const GeometryCacheIdentity& identity, const UniformTree& tree,
    const std::optional<std::vector<int>>& fixed_target_source_indices,
    const std::vector<P2MPlan>& p2m_plans, const StaticM2LPlan& m2l_plan,
    const std::vector<StaticL2PEvaluator>& l2p_evaluators,
    const StaticP2POperator& p2p_operator, StaticPlanStatistics& statistics) {
  if (!identity.enabled) {
    return;
  }
  GeometryCacheWriter writer(identity, tree, fixed_target_source_indices,
                             p2m_plans, m2l_plan, l2p_evaluators,
                             p2p_operator.source_count, p2p_operator.target_count,
                             p2p_operator.row_offsets, p2p_operator.blocks.size(),
                             statistics);
  writer.append_p2p_blocks(p2p_operator.blocks);
  writer.finish();
}

} // namespace cdfmm::detail::cache
