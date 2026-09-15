// SPDX-License-Identifier: Apache-2.0

#include "cache/internal.hpp"

#include <algorithm>
#include <array>
#include <chrono>
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
// NOTE(cdfmm): geometry_cache_loaded_direct_float_ is set here and consumed by
// src/fmm/plan_preparation.cpp, which skips FP32 quantisation on a direct hit,
// and by src/fmm/execution_setup.cpp, which promotes the FP32 P2P operator back
// to FP64 to build the signed tensor-dictionary plan.
namespace cdfmm {

using namespace detail::cache;

// Geometry-plan serialisation is implemented below with field-wise arrays.
// It deliberately excludes the universal translation matrices.
bool UniformFmm::load_geometry_cache() {
  if (!cache_enabled_) {
    return false;
  }
  const auto start = std::chrono::steady_clock::now();
  try {
    const auto payload = read_cache(
        cache_path(cache_directory_, "plans", geometry_cache_key_),
        {CacheKind::Plan, expansion_basis_, expansion_order(), precision_,
         tree_->leaf_level(), geometry_cache_key_, geometry_hash_digest_},
        static_plan_statistics_.cache_bytes_read);
    Reader reader(payload);
    const auto require_int_span = [&reader](const std::span<const int> expected) {
      if (!reader.equal_span<int>(expected)) {
        throw std::runtime_error("cached tree index metadata mismatch");
      }
    };
    require_int_span(tree_->source_permutation());
    require_int_span(tree_->source_inverse_permutation());
    require_int_span(tree_->target_permutation());
    require_int_span(tree_->target_inverse_permutation());
    require_int_span(tree_->leaf_indices());
    require_int_span(tree_->occupied_source_leaves());
    require_int_span(tree_->occupied_target_leaves());
    const auto node_count = reader.scalar<std::uint64_t>();
    const auto nodes = tree_->nodes();
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
    if (has_fixed_identities != fixed_target_source_indices_.has_value()) {
      throw std::runtime_error("cached self-identity metadata mismatch");
    }
    if (has_fixed_identities) {
      const std::vector<int> cached_identities = reader.vector<int>();
      if (cached_identities != *fixed_target_source_indices_) {
        throw std::runtime_error("cached self identities mismatch");
      }
    }
    const auto validate_cached_m2l = [this](const auto& plan) {
      const std::size_t node_count = topology_->nodes.size();
      const std::size_t interaction_count = plan.source_nodes.size();
      if (plan.level_count != topology_->maximum_level + 1 ||
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
                      topology_->nodes.begin(),
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
              topology_->nodes[static_cast<std::size_t>(node)].level != level) {
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
          const int target_level = topology_->nodes[target].level;
          if (source < 0 || source >= static_cast<int>(node_count) ||
              plan.source_levels[index] !=
                  topology_->nodes[static_cast<std::size_t>(source)].level ||
              plan.target_levels[index] != target_level ||
              plan.interaction_levels[index] != target_level) {
            throw std::runtime_error("cached canonical M2L endpoint levels mismatch");
          }
        }
      }
    };
    const auto validate_cached_p2m = [this](const auto& plans) {
      if (plans.size() != topology_->source_leaves.size()) {
        throw std::runtime_error("cached P2M leaf metadata dimensions mismatch");
      }
      for (std::size_t index = 0; index < plans.size(); ++index) {
        const auto& expected = topology_->source_leaves[index];
        if (plans[index].leaf != expected.node ||
            plans[index].begin != expected.begin ||
            plans[index].count != expected.count) {
          throw std::runtime_error("cached P2M leaf metadata mismatch");
        }
      }
    };
    const auto p2m_count = reader.scalar<std::uint64_t>();
    if (precision_ == StaticPrecision::Float32) {
      p2m_plans_float_.clear();
      p2m_plans_float_.reserve(static_cast<std::size_t>(p2m_count));
      for (std::uint64_t index = 0; index < p2m_count; ++index) {
        p2m_plans_float_.push_back(
            {reader.scalar<int>(), reader.scalar<std::size_t>(),
             reader.scalar<std::size_t>(), read_operator_float(reader)});
      }
      validate_cached_p2m(p2m_plans_float_);
      m2l_plan_float_.coefficient_count = reader.scalar<int>();
      m2l_plan_float_.matrix_count = reader.scalar<int>();
      m2l_plan_float_.level_count = reader.scalar<int>();
      m2l_plan_float_.multipole_scaling = read_values_float(reader);
      m2l_plan_float_.local_scaling = read_values_float(reader);
      m2l_plan_float_.target_row_offsets = reader.vector<int>();
      m2l_plan_float_.source_nodes = reader.vector<int>();
      m2l_plan_float_.matrix_ids = reader.vector<int>();
      m2l_plan_float_.source_levels = reader.vector<int>();
      m2l_plan_float_.target_levels = reader.vector<int>();
      m2l_plan_float_.interaction_levels = reader.vector<int>();
      m2l_plan_float_.level_target_begin = reader.vector<int>();
      m2l_plan_float_.level_target_end = reader.vector<int>();
      m2l_plan_float_.target_level_offsets = reader.vector<int>();
      m2l_plan_float_.target_nodes_by_level = reader.vector<int>();
      m2l_plan_float_.node_levels = reader.vector<int>();

      const auto l2p_count = reader.scalar<std::uint64_t>();
      l2p_evaluators_float_.clear();
      l2p_evaluators_float_.reserve(static_cast<std::size_t>(l2p_count));
      for (std::uint64_t index = 0; index < l2p_count; ++index) {
        FloatStaticL2PEvaluator evaluator;
        evaluator.potential = read_values_float(reader);
        for (auto& row : evaluator.field) {
          row = read_values_float(reader);
        }
        l2p_evaluators_float_.push_back(std::move(evaluator));
      }
      p2p_operator_float_.source_count = reader.scalar<int>();
      p2p_operator_float_.target_count = reader.scalar<int>();
      p2p_operator_float_.row_offsets = reader.vector<int>();
      const auto block_count = reader.scalar<std::uint64_t>();
      read_p2p_blocks_float(reader, block_count, p2p_operator_float_);
      validate_cached_m2l(m2l_plan_float_);
      geometry_cache_loaded_direct_float_ = true;
    } else {
      p2m_plans_.clear();
        p2m_plans_.reserve(static_cast<std::size_t>(p2m_count));
      for (std::uint64_t index = 0; index < p2m_count; ++index) {
        p2m_plans_.push_back(
            {reader.scalar<int>(), reader.scalar<std::size_t>(),
             reader.scalar<std::size_t>(), read_operator(reader, precision_)});
      }
      validate_cached_p2m(p2m_plans_);
      m2l_plan_.coefficient_count = reader.scalar<int>();
      m2l_plan_.matrix_count = reader.scalar<int>();
      m2l_plan_.level_count = reader.scalar<int>();
      m2l_plan_.multipole_scaling = read_values(reader, precision_);
      m2l_plan_.local_scaling = read_values(reader, precision_);
      m2l_plan_.target_row_offsets = reader.vector<int>();
      m2l_plan_.source_nodes = reader.vector<int>();
      m2l_plan_.matrix_ids = reader.vector<int>();
      m2l_plan_.source_levels = reader.vector<int>();
      m2l_plan_.target_levels = reader.vector<int>();
      m2l_plan_.interaction_levels = reader.vector<int>();
      m2l_plan_.level_target_begin = reader.vector<int>();
      m2l_plan_.level_target_end = reader.vector<int>();
      m2l_plan_.target_level_offsets = reader.vector<int>();
      m2l_plan_.target_nodes_by_level = reader.vector<int>();
      m2l_plan_.node_levels = reader.vector<int>();

      const auto l2p_count = reader.scalar<std::uint64_t>();
      l2p_evaluators_.clear();
      l2p_evaluators_.reserve(static_cast<std::size_t>(l2p_count));
      for (std::uint64_t index = 0; index < l2p_count; ++index) {
        StaticL2PEvaluator evaluator;
        evaluator.potential = read_values(reader, precision_);
        for (auto& row : evaluator.field) {
          row = read_values(reader, precision_);
        }
        l2p_evaluators_.push_back(std::move(evaluator));
      }
      p2p_operator_.source_count = reader.scalar<int>();
      p2p_operator_.target_count = reader.scalar<int>();
      p2p_operator_.row_offsets = reader.vector<int>();
      const auto block_count = reader.scalar<std::uint64_t>();
      read_p2p_blocks(reader, block_count, precision_, p2p_operator_,
                      p2p_compact_plan_);
      validate_cached_m2l(m2l_plan_);
    }
    reader.require_end();
    static_plan_statistics_.geometry_cache_hit = true;
    static_plan_statistics_.geometry_cache_load.add(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count());
    static_plan_statistics_.geometry_cache_lookup.add(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count());
    return true;
  } catch (const std::exception&) {
    geometry_cache_loaded_direct_float_ = false;
    p2m_plans_float_.clear();
    l2p_evaluators_float_.clear();
    p2p_operator_float_ = {};
    p2p_compact_plan_float_ = {};
    p2p_bsr_plan_float_ = {};
    m2l_plan_float_ = {};
    static_plan_statistics_.geometry_cache_lookup.add(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count());
    return false;
  }
}

void UniformFmm::write_geometry_cache() const {
  if (!cache_enabled_) {
    return;
  }
  const auto start = std::chrono::steady_clock::now();
  Writer payload;
  const std::size_t p2p_bytes = checked_bytes(
      p2p_operator_.blocks.size(), p2p_record_bytes(precision_));
  payload.reserve(p2p_bytes + 4 * 1024 * 1024);
  payload.span(tree_->source_permutation());
  payload.span(tree_->source_inverse_permutation());
  payload.span(tree_->target_permutation());
  payload.span(tree_->target_inverse_permutation());
  payload.span(tree_->leaf_indices());
  payload.span(tree_->occupied_source_leaves());
  payload.span(tree_->occupied_target_leaves());
  payload.scalar<std::uint64_t>(tree_->nodes().size());
  for (const TreeNode& node : tree_->nodes()) {
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
  payload.scalar(fixed_target_source_indices_.has_value());
  if (fixed_target_source_indices_) {
    payload.vector(*fixed_target_source_indices_);
  }
  payload.scalar<std::uint64_t>(p2m_plans_.size());
  for (const P2MPlan& plan : p2m_plans_) {
    payload.scalar(plan.leaf);
    payload.scalar(plan.begin);
    payload.scalar(plan.count);
    write_operator(payload, plan.operator_map, precision_);
  }
  payload.scalar(m2l_plan_.coefficient_count);
  payload.scalar(m2l_plan_.matrix_count);
  payload.scalar(m2l_plan_.level_count);
  write_values(payload, m2l_plan_.multipole_scaling, precision_);
  write_values(payload, m2l_plan_.local_scaling, precision_);
  payload.vector(m2l_plan_.target_row_offsets);
  payload.vector(m2l_plan_.source_nodes);
  payload.vector(m2l_plan_.matrix_ids);
  payload.vector(m2l_plan_.source_levels);
  payload.vector(m2l_plan_.target_levels);
  payload.vector(m2l_plan_.interaction_levels);
  payload.vector(m2l_plan_.level_target_begin);
  payload.vector(m2l_plan_.level_target_end);
  payload.vector(m2l_plan_.target_level_offsets);
  payload.vector(m2l_plan_.target_nodes_by_level);
  payload.vector(m2l_plan_.node_levels);
  payload.scalar<std::uint64_t>(l2p_evaluators_.size());
  for (const StaticL2PEvaluator& evaluator : l2p_evaluators_) {
    write_values(payload, evaluator.potential, precision_);
    for (const auto& row : evaluator.field) {
      write_values(payload, row, precision_);
    }
  }
  payload.scalar(p2p_operator_.source_count);
  payload.scalar(p2p_operator_.target_count);
  payload.vector(p2p_operator_.row_offsets);
  payload.scalar<std::uint64_t>(p2p_operator_.blocks.size());
  write_p2p_blocks(payload, p2p_operator_.blocks, precision_);
  const std::size_t bytes = write_cache(
      cache_path(cache_directory_, "plans", geometry_cache_key_),
      {CacheKind::Plan, expansion_basis_, expansion_order(), precision_,
       tree_->leaf_level(), geometry_cache_key_, geometry_hash_digest_},
      payload.bytes());
  const_cast<StaticPlanStatistics&>(static_plan_statistics_)
      .cache_bytes_written += bytes;
  const_cast<StaticPlanStatistics&>(static_plan_statistics_)
      .geometry_cache_write.add(
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
              .count());
}

} // namespace cdfmm
