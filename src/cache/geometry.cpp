// SPDX-License-Identifier: Apache-2.0

#include "cache/internal.hpp"
#include "phase_stopwatch.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
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
//     builds no pair tensors and is keyed separately for that reason)
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
        throw std::runtime_error("position-based plan holds pair tensors");
      }
      read_p2p_blocks_float(reader, block_count, payload.p2p_operator_float);
      validate_cached_m2l(payload.m2l_plan_float);
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
        throw std::runtime_error("position-based plan holds pair tensors");
      }
      read_p2p_blocks(reader, block_count, identity.precision,
                      payload.p2p_operator, payload.p2p_compact_plan);
      validate_cached_m2l(payload.m2l_plan);
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
    if (clock.enabled()) {
      statistics.geometry_cache_lookup.add(clock.elapsed());
    }
    return false;
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
  detail::PhaseStopwatch clock(
      statistics.timing_level == TimingLevel::Detailed);
  clock.start();
  Writer payload;
  const std::size_t p2p_bytes = checked_bytes(
      p2p_operator.blocks.size(), p2p_record_bytes(identity.precision));
  payload.reserve(p2p_bytes + 4 * 1024 * 1024);
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
  payload.scalar(p2p_operator.source_count);
  payload.scalar(p2p_operator.target_count);
  payload.vector(p2p_operator.row_offsets);
  payload.scalar<std::uint64_t>(p2p_operator.blocks.size());
  write_p2p_blocks(payload, p2p_operator.blocks, identity.precision);
  const std::size_t bytes = write_cache(
      cache_path(identity.directory, "plans", identity.geometry_key),
      {CacheKind::Plan, identity.basis, identity.order, identity.precision,
       tree.leaf_level(), identity.geometry_key, identity.geometry_hash_digest},
      payload.bytes());
  statistics.cache_bytes_written += bytes;
  if (clock.enabled()) {
    statistics.geometry_cache_write.add(clock.elapsed());
  }
}

} // namespace cdfmm::detail::cache
