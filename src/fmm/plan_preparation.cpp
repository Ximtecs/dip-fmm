// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/uniform_fmm.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <numeric>
#include <tuple>
#include <unordered_map>

#include "cdfmm/operators/operators.hpp"
#include "cdfmm/operators/operators.hpp"
#include "cdfmm/plan/static_plan.hpp"

#include "cache/internal.hpp"

namespace cdfmm {

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(const Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/// @brief The failure a parallel endpoint-operator loop reports.
///
/// The lowest failing index wins, because that is the leaf or target a serial
/// build would have reached first, so the reported cause does not depend on
/// how the iterations were scheduled.  Work above a known failure is skipped:
/// it can no longer win that comparison.
class FirstConstructionFailure {
public:
  [[nodiscard]] bool superseded(const std::size_t index) const noexcept {
    return index > index_.load(std::memory_order_relaxed);
  }

  /// @brief Record the exception currently being handled for @p index.
  void record(const std::size_t index) {
    std::size_t previous = index_.load(std::memory_order_relaxed);
    while (index < previous &&
           !index_.compare_exchange_weak(previous, index,
                                         std::memory_order_relaxed)) {
    }
#pragma omp critical(cdfmm_endpoint_operator_exception)
    {
      if (index_.load(std::memory_order_relaxed) == index) {
        exception_ = std::current_exception();
      }
    }
  }

  void rethrow_any() const {
    if (exception_) {
      std::rethrow_exception(exception_);
    }
  }

private:
  std::atomic<std::size_t> index_{std::numeric_limits<std::size_t>::max()};
  std::exception_ptr exception_{};
};

// Exact endpoint operator reuse.
//
// A finite P2M or L2P operator is a pure function of the expansion basis, the
// body's shape record and its displacement from its leaf centre.  The basis is
// fixed for a whole plan, so the key holds the geometry alone.  Leaf P2M
// entries are indexed by leaf-local source, so two leaves whose bodies sit at
// the same offsets with the same shapes produce identical entries, and a
// target's L2P rows depend on nothing but its own displacement and shape.
//
// Repeated geometry makes this decisive in the same way it does for the near
// field: every leaf of a regular lattice has the same internal layout.  The
// key holds raw bit patterns and is never compared with a tolerance.
using EndpointOperatorKey = std::vector<std::uint64_t>;

struct EndpointOperatorKeyHash {
  [[nodiscard]] std::size_t operator()(
      const EndpointOperatorKey& key) const noexcept {
    std::uint64_t hash = 0x9e3779b97f4a7c15ULL ^ key.size();
    for (const std::uint64_t word : key) {
      hash ^= word;
      hash *= 0x00000100000001b3ULL;
      hash ^= hash >> 29;
    }
    return static_cast<std::size_t>(hash);
  }
};

void push_bits(EndpointOperatorKey& key, const double value) {
  key.push_back(std::bit_cast<std::uint64_t>(value));
}

void push_bits(EndpointOperatorKey& key, const Vec3& value) {
  push_bits(key, value.x);
  push_bits(key, value.y);
  push_bits(key, value.z);
}

void push_bits(EndpointOperatorKey& key, const CuboidSize& size) {
  push_bits(key, size.hx);
  push_bits(key, size.hy);
  push_bits(key, size.hz);
}

void push_bits(EndpointOperatorKey& key, const Tetrahedron& tetrahedron) {
  for (const Vec3& vertex : tetrahedron.vertices) {
    push_bits(key, vertex);
  }
}

/// @brief Items grouped by bitwise-identical endpoint operator inputs.
struct EndpointOperatorClasses {
  std::vector<std::uint32_t> class_of_item;
  std::vector<std::uint32_t> representative;
  bool classified{false};
};

/// @brief Group items whose endpoint operator inputs agree bit for bit.
///
/// Classes are numbered in first-seen order, so the classification and every
/// built value are independent of thread count and of hash iteration order.
/// Classification stops when an initial sample shows too few duplicates to
/// repay it, which only ever changes performance: the caller builds the same
/// operators either way.
template <typename KeyOfItem>
[[nodiscard]] EndpointOperatorClasses classify_endpoint_operators(
    const std::size_t item_count, const KeyOfItem& key_of_item) {
  constexpr std::size_t sample_items = 4096;
  constexpr std::size_t sample_reuse_factor = 2;

  EndpointOperatorClasses result;
  result.class_of_item.resize(item_count);
  std::unordered_map<EndpointOperatorKey, std::uint32_t,
                     EndpointOperatorKeyHash>
      classes;
  for (std::size_t item = 0; item < item_count; ++item) {
    const auto [entry, inserted] = classes.try_emplace(
        key_of_item(item),
        static_cast<std::uint32_t>(result.representative.size()));
    if (inserted) {
      result.representative.push_back(static_cast<std::uint32_t>(item));
    }
    result.class_of_item[item] = entry->second;

    const std::size_t sampled = item + 1;
    if (sampled == sample_items && sampled < item_count &&
        result.representative.size() * sample_reuse_factor > sampled) {
      return {};
    }
  }
  result.classified = true;
  return result;
}

template <typename Operator>
std::size_t estimate_bsr_bytes(const Operator& p2p,
                               const std::size_t scalar_bytes) {
  const std::size_t interactions = p2p.blocks.size();
  const std::size_t tensor_bytes = 9 * interactions * scalar_bytes;
  const std::size_t index_bytes = interactions * sizeof(int);
  // StaticP2PBsrPlan stores one identity entry per target, including -1
  // entries when no self exclusion is requested.
  const std::size_t metadata_bytes =
      (p2p.row_offsets.size() +
       static_cast<std::size_t>(p2p.target_count)) * sizeof(int);
  return tensor_bytes + index_bytes + metadata_bytes;
}

} // namespace

void UniformFmm::build_missing_universal_operators(
    const bool universal_available, const bool periodic_required) {
  constexpr std::size_t class_count =
      StaticPlanStatistics::theoretical_maximum_m2l_classes;
  if (!universal_available) {
    const auto shared_start = Clock::now();
    constexpr double child_half_width = 0.25;
    for (int child_class = 0; child_class < 8; ++child_class) {
      const Vec3 child_offset{
          (child_class & 1) != 0 ? child_half_width : -child_half_width,
          (child_class & 2) != 0 ? child_half_width : -child_half_width,
          (child_class & 4) != 0 ? child_half_width : -child_half_width};
      if (expansion_basis_ == ExpansionBasis::Spherical) {
        m2m_operators_[child_class] =
            build_static_m2m_operator(spherical_basis_, child_offset * -1.0);
        l2l_operators_[child_class] =
            build_static_l2l_operator(spherical_basis_, child_offset);
      } else {
        m2m_operators_[child_class] =
            build_static_m2m_operator(basis_, child_offset * -1.0);
        l2l_operators_[child_class] =
            build_static_l2l_operator(basis_, child_offset);
      }
      const std::size_t m2m_bytes = m2m_operators_[child_class].entries.size() *
          sizeof(StaticOperatorEntry);
      const std::size_t l2l_bytes = l2l_operators_[child_class].entries.size() *
          sizeof(StaticOperatorEntry);
      static_plan_statistics_.m2m_operator_bytes += m2m_bytes;
      static_plan_statistics_.l2l_operator_bytes += l2l_bytes;
      static_plan_statistics_.operator_bytes += m2m_bytes + l2l_bytes;
      ++static_plan_statistics_.m2m_operators;
      ++static_plan_statistics_.l2l_operators;
    }
    const double shared_seconds = elapsed_seconds(shared_start);
    static_plan_statistics_.m2m_plan.add(shared_seconds);
    static_plan_statistics_.l2l_plan.add(shared_seconds);
    static_plan_statistics_.universal_operator_build.add(shared_seconds);

    const std::size_t matrix_values =
        static_cast<std::size_t>(coefficient_count()) * coefficient_count();
    m2l_plan_.matrices.resize(class_count * matrix_values);
    std::vector<std::array<int, 3>> displacements;
    displacements.reserve(class_count);
    for (int dx = -3; dx <= 3; ++dx) {
      for (int dy = -3; dy <= 3; ++dy) {
        for (int dz = -3; dz <= 3; ++dz) {
          if (std::abs(dx) <= 1 && std::abs(dy) <= 1 &&
              std::abs(dz) <= 1) {
            continue;
          }
          displacements.push_back({dx, dy, dz});
        }
      }
    }
    const auto m2l_start = Clock::now();
#pragma omp parallel for schedule(dynamic) if(class_count >= 8)
    for (std::ptrdiff_t id = 0;
         id < static_cast<std::ptrdiff_t>(class_count); ++id) {
      const auto [dx, dy, dz] =
          displacements[static_cast<std::size_t>(id)];
      const Vec3 displacement{static_cast<double>(dx),
                              static_cast<double>(dy),
                              static_cast<double>(dz)};
      const std::vector<double> matrix =
          expansion_basis_ == ExpansionBasis::Spherical
          ? build_static_m2l_matrix(spherical_basis_, displacement)
          : build_static_m2l_matrix(basis_, displacement);
      std::copy(matrix.begin(), matrix.end(),
                m2l_plan_.matrices.begin() + id * matrix_values);
    }
    static_plan_statistics_.universal_operator_build.add(
        elapsed_seconds(m2l_start));
  }

  if (periodic_required && !periodic_operator_available_) {
    const auto periodic_start = Clock::now();
    const std::vector<double> matrix =
        expansion_basis_ == ExpansionBasis::Spherical
        ? build_static_periodic_m2l_matrix(spherical_basis_, periodic_)
        : build_static_periodic_m2l_matrix(basis_, periodic_);
    m2l_plan_.matrices.insert(m2l_plan_.matrices.end(), matrix.begin(),
                              matrix.end());
    periodic_operator_available_ = true;
    static_plan_statistics_.periodic_operator_build.add(
        elapsed_seconds(periodic_start));
  }
}

void UniformFmm::build_static_plan() {
  // This is the geometry-dependent half of the evaluator. None of the data
  // built here depends on dipole moments, so it remains valid for every later
  // evaluate() call; see docs/static-architecture.md.
  const auto total_start = Clock::now();
  static_plan_statistics_.expansion_order = expansion_order();
  static_plan_statistics_.coefficient_count =
      static_cast<std::size_t>(coefficient_count());
  static_plan_statistics_.spherical =
      expansion_basis_ == ExpansionBasis::Spherical;
  static_plan_statistics_.tree_bytes = (tree_ ? tree_->memory_statistics().total_bytes() : 0);
  static_plan_statistics_.topology_bytes = topology_->memory_bytes();
  // Plan preparation owns the cache-vs-build decision: it supplies identity
  // and payload state to the cache boundary and decides whether to build and
  // write on a miss. The cache functions never reach into `this`.
  const detail::cache::UniversalCacheIdentity universal_identity{
      cache_enabled_,        cache_directory_, universal_cache_key_,
      periodic_cache_key_,   expansion_basis_, precision_,
      expansion_order(),     periodic_.enabled};
  bool universal_available = detail::cache::load_universal_cache(
      universal_identity, coefficient_count(),
      {m2m_operators_, l2l_operators_, m2l_plan_.matrices,
       periodic_operator_available_},
      static_plan_statistics_);
  const bool universal_cache_loaded = universal_available;
  const bool periodic_required = periodic_.enabled && !topology_->nodes.empty() &&
      topology_->nodes[static_cast<std::size_t>(topology_->root)].source_count() != 0 &&
      topology_->nodes[static_cast<std::size_t>(topology_->root)].target_count() != 0;
  const bool universal_write_required = !universal_available ||
      (periodic_required && !periodic_operator_available_);
  if (universal_write_required) {
    build_missing_universal_operators(universal_available, periodic_required);
    universal_available = true;
    detail::cache::write_universal_cache(
        universal_identity, coefficient_count(), m2m_operators_,
        l2l_operators_, m2l_plan_.matrices, static_plan_statistics_);
  }
  if (universal_available) {
    // The universal bank always owns exactly eight child-class templates.
    // Record them on both cache hits and analytical construction paths.
    static_plan_statistics_.m2m_operators = 8;
    static_plan_statistics_.l2l_operators = 8;
    if (universal_cache_loaded) {
      // Cache payloads are deserialised into the same retained operator
      // containers as cold construction.  Reconstruct their storage
      // accounting here; the cache-hit path skips the construction loop
      // below, so otherwise these bytes would be silently omitted.
      for (const StaticCoefficientOperator &operator_map : m2m_operators_) {
        const std::size_t bytes =
            operator_map.entries.size() * sizeof(StaticOperatorEntry);
        static_plan_statistics_.m2m_operator_bytes += bytes;
        static_plan_statistics_.operator_bytes += bytes;
      }
      for (const StaticCoefficientOperator &operator_map : l2l_operators_) {
        const std::size_t bytes =
            operator_map.entries.size() * sizeof(StaticOperatorEntry);
        static_plan_statistics_.l2l_operator_bytes += bytes;
        static_plan_statistics_.operator_bytes += bytes;
      }
    }
  }
  const detail::cache::GeometryCacheIdentity geometry_identity{
      cache_enabled_,    cache_directory_, geometry_cache_key_,
      geometry_hash_digest_, expansion_basis_, precision_, expansion_order()};
  if (universal_available &&
      (!periodic_required || periodic_operator_available_) &&
      detail::cache::load_geometry_cache(
          geometry_identity, *tree_, *topology_, fixed_target_source_indices_,
          {p2m_plans_, p2m_plans_float_, m2l_plan_, m2l_plan_float_,
           l2p_evaluators_, l2p_evaluators_float_, p2p_operator_,
           p2p_operator_float_, p2p_compact_plan_, p2p_compact_plan_float_,
           p2p_bsr_plan_float_},
          geometry_cache_loaded_direct_float_, static_plan_statistics_)) {
    static_plan_statistics_.m2l_operators =
        static_cast<std::size_t>(precision_ == StaticPrecision::Float32
                                     ? m2l_plan_float_.matrix_count
                                     : m2l_plan_.matrix_count);
    static_plan_statistics_.transfer_classes =
        StaticPlanStatistics::theoretical_maximum_m2l_classes;
    static_plan_statistics_.interactions =
        precision_ == StaticPrecision::Float32
            ? m2l_plan_float_.source_nodes.size()
            : m2l_plan_.source_nodes.size();
    static_plan_statistics_.p2p_interactions =
        precision_ == StaticPrecision::Float32
            ? p2p_operator_float_.blocks.size()
            : p2p_operator_.blocks.size();
    static_plan_statistics_.m2m_theoretical_interactions =
        topology_->nodes.empty() ? 0 : topology_->nodes.size() - 1;
    static_plan_statistics_.l2l_theoretical_interactions =
        static_plan_statistics_.m2m_theoretical_interactions;

    // Geometry caches contain operator payloads rather than their derived
    // memory statistics. Reconstruct the same accounting recorded by the cold
    // construction path so warm and cold plans report identical storage.
    const auto record_m2l_metadata = [this](const auto& plan) {
      const std::size_t bytes =
          (plan.target_row_offsets.size() + plan.source_nodes.size() +
           plan.matrix_ids.size() + plan.source_levels.size() +
           plan.target_levels.size() + plan.interaction_levels.size() +
           plan.level_target_begin.size() + plan.level_target_end.size() +
           plan.target_level_offsets.size() +
           plan.target_nodes_by_level.size() + plan.node_levels.size()) *
          sizeof(int);
      static_plan_statistics_.interaction_bytes += bytes;
      static_plan_statistics_.m2l_interaction_bytes += bytes;
    };
    if (precision_ == StaticPrecision::Float32) {
      record_m2l_metadata(m2l_plan_float_);
    } else {
      std::size_t p2m_bytes = 0;
      for (const P2MPlan& plan : p2m_plans_) {
        p2m_bytes +=
            plan.operator_map.entries.size() * sizeof(StaticOperatorEntry);
      }
      const std::size_t m2l_bytes =
          (m2l_plan_.matrices.size() +
           m2l_plan_.multipole_scaling.size() +
           m2l_plan_.local_scaling.size()) *
          sizeof(double);
      std::size_t l2p_bytes = 0;
      for (const StaticL2PEvaluator& evaluator : l2p_evaluators_) {
        l2p_bytes += evaluator.potential.size() * sizeof(double);
        for (const std::vector<double>& field : evaluator.field) {
          l2p_bytes += field.size() * sizeof(double);
        }
      }
      const std::size_t near_field_bytes =
          p2p_operator_.memory_bytes() +
          p2p_compact_plan_.memory().total_bytes();
      static_plan_statistics_.operator_bytes +=
          p2m_bytes + m2l_bytes + l2p_bytes + near_field_bytes;
      static_plan_statistics_.p2m_operator_bytes += p2m_bytes;
      static_plan_statistics_.m2l_operator_bytes += m2l_bytes;
      static_plan_statistics_.l2p_operator_bytes += l2p_bytes;
      static_plan_statistics_.near_field_operator_bytes = near_field_bytes;
      record_m2l_metadata(m2l_plan_);
    }
    // A cache hit still derives every execution packing, so the same timers
    // account for it as on a cold build.
    auto warm_stage_start = Clock::now();
    try {
      build_reduced_symmetry_p2p_packing();
    } catch (const std::invalid_argument &error) {
      throw std::runtime_error(
          std::string("reduced-symmetry P2P topology is not representable: ") +
          error.what());
    }
    if (precision_ == StaticPrecision::Float64) {
      static_plan_statistics_.p2p_value_bytes =
          p2p_operator_.blocks.size() * 6 * sizeof(double);
      static_plan_statistics_.p2p_index_bytes =
          p2p_compact_plan_.row_offsets.size() * sizeof(int) +
          p2p_compact_plan_.source_indices.size() * sizeof(int) +
          p2p_compact_plan_.skip_for_identity.size() * sizeof(unsigned char);
      static_plan_statistics_.p2p_canonical_total_bytes =
          p2p_compact_plan_.memory().total_bytes();
      if (p2p_tensor_dictionary_plan_.has_value()) {
        static_plan_statistics_.p2p_unique_tensors =
            p2p_tensor_dictionary_plan_->tensors[0].size();
        static_plan_statistics_.p2p_dictionary_tokens =
            p2p_tensor_dictionary_plan_->token_count();
        static_plan_statistics_.p2p_dictionary_token_width_bytes =
            p2p_tensor_dictionary_plan_->token_width_bytes;
        static_plan_statistics_.p2p_dictionary_token_bytes =
            p2p_tensor_dictionary_plan_->token_count() *
            p2p_tensor_dictionary_plan_->token_width_bytes;
        static_plan_statistics_.p2p_dictionary_tensor_bytes =
            p2p_tensor_dictionary_plan_->tensors[0].size() * 6 * sizeof(double);
        static_plan_statistics_.p2p_dictionary_total_bytes =
            p2p_tensor_dictionary_plan_->memory().total_bytes();
      }
    }
    if (backend_ == ExecutionBackend::CpuStatic) {
      // A cache hit derives the row packing as well; the CPU packing step
      // releases it again when positions replace the stored tensors.
      p2p_execution_packing_ = resolve_cpu_p2p_packing();
    }
    static_plan_statistics_.p2p_derived_packing.add(
        elapsed_seconds(warm_stage_start));
    static_plan_statistics_.p2p_tensor_plan.add(
        elapsed_seconds(warm_stage_start));
    if (precision_ == StaticPrecision::Float32) {
      warm_stage_start = Clock::now();
      quantise_static_plan_to_float();
      static_plan_statistics_.precision_conversion.add(
          elapsed_seconds(warm_stage_start));
    }
    build_backend_packing();
    static_plan_statistics_.total.add(elapsed_seconds(total_start));
    return;
  }
  const bool universal_cache_hit = universal_available;
  const auto &nodes = topology_->nodes;
  using Key = std::tuple<double, double, double, double>;
  using ClassMap = std::map<Key, std::vector<std::pair<int, int>>>;
  ClassMap classes;

  auto phase_start = Clock::now();
  const std::span<const Vec3> sorted_positions = topology_->sorted_source_positions;
  const std::span<const Vec3> sorted_targets = topology_->sorted_target_positions;
  const std::span<const CuboidSize> source_sizes = sorted_source_sizes_;
  const std::span<const CuboidSize> target_sizes = sorted_target_sizes_;
  // One leaf's P2M operator, resolved from the source geometry and the
  // far-field model.
  const auto build_leaf_operator =
      [&](const StaticLeafRange& leaf_range, const Vec3& centre) {
    const auto leaf_positions =
        sorted_positions.subspan(leaf_range.begin, leaf_range.count);
    if (source_geometry_ == SourceGeometry::RectangularPrism &&
        far_field_source_model_ == SourceModel::ExactGeometry) {
      const std::span<const CuboidSize> leaf_sizes =
          source_sizes.size() == 1
              ? source_sizes
              : source_sizes.subspan(leaf_range.begin, leaf_range.count);
      return expansion_basis_ == ExpansionBasis::Spherical
          ? build_static_cuboid_p2m_operator(
                spherical_basis_, centre, leaf_positions, leaf_sizes)
          : build_static_cuboid_p2m_operator(
                basis_, centre, leaf_positions, leaf_sizes);
    }
    if (source_geometry_ == SourceGeometry::Tetrahedron &&
        far_field_source_model_ == SourceModel::ExactGeometry) {
      const std::span<const Tetrahedron> leaf_tetrahedra =
          sorted_source_tetrahedra_.size() == 1
              ? std::span<const Tetrahedron>(sorted_source_tetrahedra_)
              : std::span<const Tetrahedron>(sorted_source_tetrahedra_)
                    .subspan(leaf_range.begin, leaf_range.count);
      return expansion_basis_ == ExpansionBasis::Spherical
          ? build_static_tetrahedron_p2m_operator(
                spherical_basis_, centre, leaf_positions, leaf_tetrahedra)
          : build_static_tetrahedron_p2m_operator(
                basis_, centre, leaf_positions, leaf_tetrahedra);
    }
    return expansion_basis_ == ExpansionBasis::Spherical
        ? build_static_p2m_operator(spherical_basis_, centre, leaf_positions)
        : build_static_p2m_operator(basis_, centre, leaf_positions);
  };

  // Everything the leaf operator depends on, in bits: each body's offset from
  // the leaf centre and its shape record, in leaf-local order.
  const auto leaf_operator_key =
      [&](const std::size_t slot) -> EndpointOperatorKey {
    const StaticLeafRange& leaf_range = topology_->source_leaves[slot];
    const auto& leaf =
        topology_->nodes[static_cast<std::size_t>(leaf_range.node)];
    const bool exact_prism =
        source_geometry_ == SourceGeometry::RectangularPrism &&
        far_field_source_model_ == SourceModel::ExactGeometry;
    const bool exact_tetrahedron =
        source_geometry_ == SourceGeometry::Tetrahedron &&
        far_field_source_model_ == SourceModel::ExactGeometry;
    EndpointOperatorKey key;
    key.reserve(leaf_range.count * (exact_tetrahedron ? 15 : 6));
    for (std::size_t body = 0; body < leaf_range.count; ++body) {
      const std::size_t source = leaf_range.begin + body;
      push_bits(key, sorted_positions[source] - leaf.centre);
      if (exact_prism) {
        push_bits(key, source_sizes[source_sizes.size() == 1 ? 0 : source]);
      } else if (exact_tetrahedron) {
        push_bits(key, sorted_source_tetrahedra_[
            sorted_source_tetrahedra_.size() == 1 ? 0 : source]);
      }
    }
    return key;
  };

  // Leaf P2M operators are independent of one another, so each distinct one is
  // built once and in parallel.  The byte accounting is summed afterwards
  // instead of from inside the loop, which is what previously tied the loop to
  // one thread.  Exact finite sources make each leaf expensive and unequal, so
  // the schedule is dynamic.
  p2m_plans_.assign(topology_->source_leaves.size(), P2MPlan{});
  {
    const std::size_t leaf_total = topology_->source_leaves.size();
    const EndpointOperatorClasses classes =
        classify_endpoint_operators(leaf_total, leaf_operator_key);
    const std::size_t build_total =
        classes.classified ? classes.representative.size() : leaf_total;

    FirstConstructionFailure failure;
    for (std::size_t slot = 0; slot < leaf_total; ++slot) {
      const StaticLeafRange& leaf_range = topology_->source_leaves[slot];
      P2MPlan& plan = p2m_plans_[slot];
      plan.leaf = leaf_range.node;
      plan.begin = leaf_range.begin;
      plan.count = leaf_range.count;
    }

    // Without reuse there is nothing to hold aside, so the operators are built
    // straight into their plans rather than through a second full-size buffer.
    const std::ptrdiff_t build_count =
        static_cast<std::ptrdiff_t>(build_total);
    std::vector<StaticCoefficientOperator> built(
        classes.classified ? build_total : 0);
#pragma omp parallel for schedule(dynamic, 1) if (build_count >= 4)
    for (std::ptrdiff_t raw_build = 0; raw_build < build_count; ++raw_build) {
      const std::size_t entry = static_cast<std::size_t>(raw_build);
      const std::size_t slot = classes.classified
          ? static_cast<std::size_t>(classes.representative[entry])
          : entry;
      if (failure.superseded(slot)) {
        continue;
      }
      try {
        const StaticLeafRange& leaf_range = topology_->source_leaves[slot];
        const auto& leaf =
            topology_->nodes[static_cast<std::size_t>(leaf_range.node)];
        StaticCoefficientOperator operator_map =
            build_leaf_operator(leaf_range, leaf.centre);
        if (classes.classified) {
          built[entry] = std::move(operator_map);
        } else {
          p2m_plans_[slot].operator_map = std::move(operator_map);
        }
      } catch (...) {
        failure.record(slot);
      }
    }
    failure.rethrow_any();

    if (classes.classified) {
      const std::ptrdiff_t scatter_count =
          static_cast<std::ptrdiff_t>(leaf_total);
#pragma omp parallel for schedule(static) if (scatter_count >= 64)
      for (std::ptrdiff_t raw_slot = 0; raw_slot < scatter_count; ++raw_slot) {
        const std::size_t slot = static_cast<std::size_t>(raw_slot);
        p2m_plans_[slot].operator_map = built[classes.class_of_item[slot]];
      }
    }
  }
  for (const P2MPlan& plan : p2m_plans_) {
    const std::size_t bytes =
        plan.operator_map.entries.size() * sizeof(StaticOperatorEntry);
    static_plan_statistics_.operator_bytes += bytes;
    static_plan_statistics_.p2m_operator_bytes += bytes;
  }
  static_plan_statistics_.p2m_plan.add(elapsed_seconds(phase_start));

  phase_start = Clock::now();
  static_plan_statistics_.m2m_theoretical_interactions =
      nodes.empty() ? 0 : nodes.size() - 1;
  static_plan_statistics_.l2l_theoretical_interactions =
      static_plan_statistics_.m2m_theoretical_interactions;
  constexpr double canonical_child_half_width = 0.25;
  for (int child_class = 0;
       child_class < (universal_cache_hit ? 0 : 8); ++child_class) {
    // The eight templates use the level-one child displacement. At level l,
    // each translation monomial is multiplied by 2^(-(l-1) degree).
    const Vec3 child_offset{
        (child_class & 1) != 0 ? canonical_child_half_width
                               : -canonical_child_half_width,
        (child_class & 2) != 0 ? canonical_child_half_width
                               : -canonical_child_half_width,
        (child_class & 4) != 0 ? canonical_child_half_width
                               : -canonical_child_half_width};
    if (expansion_basis_ == ExpansionBasis::Spherical) {
      m2m_operators_[child_class] =
          build_static_m2m_operator(spherical_basis_, child_offset * -1.0);
      l2l_operators_[child_class] =
          build_static_l2l_operator(spherical_basis_, child_offset);
    } else {
      m2m_operators_[child_class] =
          build_static_m2m_operator(basis_, child_offset * -1.0);
      l2l_operators_[child_class] =
          build_static_l2l_operator(basis_, child_offset);
    }
    const std::size_t m2m_bytes =
        m2m_operators_[child_class].entries.size() *
        sizeof(StaticOperatorEntry);
    const std::size_t l2l_bytes =
        l2l_operators_[child_class].entries.size() *
        sizeof(StaticOperatorEntry);
    static_plan_statistics_.m2m_operator_bytes += m2m_bytes;
    static_plan_statistics_.l2l_operator_bytes += l2l_bytes;
    static_plan_statistics_.operator_bytes += m2m_bytes + l2l_bytes;
    ++static_plan_statistics_.m2m_operators;
    ++static_plan_statistics_.l2l_operators;
  }
  const double shared_translation_seconds = elapsed_seconds(phase_start);
  static_plan_statistics_.m2m_plan.add(shared_translation_seconds);
  if (!universal_cache_hit) {
    static_plan_statistics_.universal_operator_build.add(
        shared_translation_seconds);
  }
  // The two triangular families are constructed together from each shared
  // parent-child displacement class.
  static_plan_statistics_.l2l_plan = static_plan_statistics_.m2m_plan;

  phase_start = Clock::now();
  for (const StaticM2LInteraction& interaction : topology_->m2l_interactions) {
    const auto& target = nodes[static_cast<std::size_t>(interaction.target_node)];
    const auto& source = nodes[static_cast<std::size_t>(interaction.source_node)];
    if (target.target_count() == 0 || source.source_count() == 0) {
      continue;
    }
    const double target_width = 2.0 * target.half_width;
    const Vec3 displacement = supplied_topology_
        ? interaction.displacement * (1.0 / target_width)
        : Vec3{static_cast<double>(interaction.transfer_class[0]),
               static_cast<double>(interaction.transfer_class[1]),
               static_cast<double>(interaction.transfer_class[2])};
    classes[Key{displacement.x, displacement.y, displacement.z,
                interaction.source_to_target_width}].emplace_back(
        interaction.source_node, interaction.target_node);
  }
  static_plan_statistics_.transfer_discovery.add(elapsed_seconds(phase_start));

  const int coefficient_count = this->coefficient_count();
  m2l_plan_.coefficient_count = coefficient_count;
  const bool has_periodic_root = periodic_.enabled && !nodes.empty() &&
      nodes[static_cast<std::size_t>(topology_->root)].source_count() != 0 &&
      nodes[static_cast<std::size_t>(topology_->root)].target_count() != 0;
  std::vector<Key> universal_classes;
  universal_classes.reserve(
      StaticPlanStatistics::theoretical_maximum_m2l_classes);
  std::map<Key, int> universal_class_ids;
  for (int dx = -3; dx <= 3; ++dx) {
    for (int dy = -3; dy <= 3; ++dy) {
      for (int dz = -3; dz <= 3; ++dz) {
        if (std::abs(dx) <= 1 && std::abs(dy) <= 1 &&
            std::abs(dz) <= 1) {
          continue;
        }
        const Key key{dx, dy, dz, 1.0};
        universal_class_ids.emplace(
            key, static_cast<int>(universal_classes.size()));
        universal_classes.push_back(key);
      }
    }
  }
  // Additional adaptive classes retain the source/target size ratio.
  for (const auto& [key, interactions] : classes) {
    if (!universal_class_ids.contains(key)) {
      universal_class_ids.emplace(key, static_cast<int>(universal_classes.size()));
      universal_classes.push_back(key);
    }
  }
  m2l_plan_.matrix_count = static_cast<int>(universal_classes.size()) +
      (has_periodic_root ? 1 : 0);
  m2l_plan_.level_count = topology_->maximum_level + 1;
  m2l_plan_.level_target_begin.resize(
      static_cast<std::size_t>(m2l_plan_.level_count));
  m2l_plan_.level_target_end.resize(
      static_cast<std::size_t>(m2l_plan_.level_count));
  m2l_plan_.target_level_offsets.assign(
      static_cast<std::size_t>(m2l_plan_.level_count) + 1, 0);
  m2l_plan_.target_nodes_by_level.clear();
  for (int target_level = 0; target_level < m2l_plan_.level_count;
       ++target_level) {
    m2l_plan_.target_level_offsets[static_cast<std::size_t>(target_level)] =
        static_cast<int>(m2l_plan_.target_nodes_by_level.size());
    for (const auto& node : topology_->nodes) {
      if (node.level == target_level) {
        m2l_plan_.target_nodes_by_level.push_back(node.index);
      }
    }
  }
  m2l_plan_.target_level_offsets.back() =
      static_cast<int>(m2l_plan_.target_nodes_by_level.size());
  m2l_plan_.node_levels.reserve(topology_->nodes.size());
  for (const auto& node : topology_->nodes) {
    m2l_plan_.node_levels.push_back(node.level);
  }

  const std::size_t scaling_size =
      static_cast<std::size_t>(m2l_plan_.level_count) * coefficient_count;
  m2l_plan_.multipole_scaling.resize(scaling_size);
  m2l_plan_.local_scaling.resize(scaling_size);
  std::vector<double> inverse_width_powers(
      static_cast<std::size_t>(expansion_order() + 2), 1.0);

  for (int level = 0; level <= topology_->maximum_level; ++level) {
    m2l_plan_.level_target_begin[static_cast<std::size_t>(level)] =
        m2l_plan_.target_level_offsets[static_cast<std::size_t>(level)];
    m2l_plan_.level_target_end[static_cast<std::size_t>(level)] =
        m2l_plan_.target_level_offsets[static_cast<std::size_t>(level + 1)];
    const int first_target =
        m2l_plan_.target_level_offsets[static_cast<std::size_t>(level)];
    const double box_width = first_target <
            m2l_plan_.target_level_offsets[static_cast<std::size_t>(level + 1)]
        ? 2.0 * nodes[static_cast<std::size_t>(
                         m2l_plan_.target_nodes_by_level[
                             static_cast<std::size_t>(first_target)])]
                  .half_width
        : 0.0;
    const std::size_t scaling_offset =
        static_cast<std::size_t>(level) * coefficient_count;
    // The matrices below are dimensionless and level independent. These two
    // degree-dependent factors restore physical box width; see the M2L
    // normalisation section in docs/math.md.
    inverse_width_powers[0] = 1.0;
    for (int degree = 1; degree <= expansion_order() + 1; ++degree) {
      inverse_width_powers[static_cast<std::size_t>(degree)] =
          inverse_width_powers[static_cast<std::size_t>(degree - 1)] /
          box_width;
    }
    // Each coefficient is filled once from its known degree, rather than
    // rescanning the complete basis for every possible degree.
    for (int index = 0; index < coefficient_count; ++index) {
      const int degree = coefficient_degree(index);
      m2l_plan_.multipole_scaling[scaling_offset + index] =
          inverse_width_powers[static_cast<std::size_t>(degree)];
      m2l_plan_.local_scaling[scaling_offset + index] =
          inverse_width_powers[static_cast<std::size_t>(degree + 1)];
    }
  }

  phase_start = Clock::now();
  using ClassEntry = ClassMap::value_type;
  std::vector<const ClassEntry*> ordered_classes;
  ordered_classes.reserve(classes.size());
  std::size_t interaction_count = 0;
  for (const ClassEntry& entry : classes) {
    ordered_classes.push_back(&entry);
    interaction_count += entry.second.size();
  }
  if (has_periodic_root) {
    ++interaction_count;
  }
  const std::size_t matrix_values =
      static_cast<std::size_t>(coefficient_count) * coefficient_count;
  m2l_plan_.matrices.resize(
      static_cast<std::size_t>(m2l_plan_.matrix_count) * matrix_values);

  // Displacement classes are independent setup work. Constructing them in
  // parallel substantially reduces high-order plan initialisation while
  // preserving the canonical lexicographic matrix-ID ordering.
  const std::ptrdiff_t class_count =
      static_cast<std::ptrdiff_t>(universal_classes.size());
  const auto universal_m2l_start = Clock::now();
#pragma omp parallel for schedule(dynamic) if (class_count >= 8)
  for (std::ptrdiff_t id = 0; id < class_count; ++id) {
    if (universal_cache_hit && id < 316) {
      continue;
    }
    const Key& key = universal_classes[static_cast<std::size_t>(id)];
    const auto [dx, dy, dz, ratio] = key;
    const Vec3 R{static_cast<double>(dx), static_cast<double>(dy),
                 static_cast<double>(dz)};
    std::vector<double> matrix =
        expansion_basis_ == ExpansionBasis::Spherical
            ? build_static_m2l_matrix(spherical_basis_, R)
            : build_static_m2l_matrix(basis_, R);
    // T(R/wt) * (ws/wt)^degree(alpha), followed by ws^-alpha and
    // wt^-(beta+1), equals the physical translation T(R).
    if (ratio != 1.0) {
      for (int alpha = 0; alpha < coefficient_count; ++alpha) {
        const double factor = std::pow(ratio, coefficient_degree(alpha));
        for (int beta = 0; beta < coefficient_count; ++beta) {
          matrix[static_cast<std::size_t>(alpha) * coefficient_count + beta] *= factor;
        }
      }
    }
    std::copy(matrix.begin(), matrix.end(),
              m2l_plan_.matrices.begin() + id * matrix_values);
  }
  if (!universal_cache_hit) {
    static_plan_statistics_.universal_operator_build.add(
        elapsed_seconds(universal_m2l_start));
  }

  if (has_periodic_root && !periodic_operator_available_) {
    // Wrapped traversal resolves the central root and its 26 neighbours. The
    // appended self translation represents every more distant lattice image.
    const auto periodic_build_start = Clock::now();
    const std::vector<double> matrix =
        expansion_basis_ == ExpansionBasis::Spherical
            ? build_static_periodic_m2l_matrix(spherical_basis_, periodic_)
            : build_static_periodic_m2l_matrix(basis_, periodic_);
    std::copy(
        matrix.begin(), matrix.end(),
        m2l_plan_.matrices.begin() +
            static_cast<std::ptrdiff_t>(universal_classes.size() *
                                        matrix_values));
    static_plan_statistics_.periodic_operator_build.add(
        elapsed_seconds(periodic_build_start));
    periodic_operator_available_ = true;
  }

  // The canonical topology already owns target-row boundaries. Static matrix
  // ordering is a derived permutation of those rows, so only the optional
  // periodic root row changes the cumulative offsets.
  m2l_plan_.target_row_offsets = topology_->m2l_target_row_offsets;
  if (has_periodic_root) {
    for (std::size_t row = static_cast<std::size_t>(topology_->root) + 1;
         row < m2l_plan_.target_row_offsets.size(); ++row) {
      ++m2l_plan_.target_row_offsets[row];
    }
  }
  if (m2l_plan_.target_row_offsets.empty() ||
      m2l_plan_.target_row_offsets.back() !=
          static_cast<int>(interaction_count)) {
    throw std::logic_error("canonical M2L row count does not match interactions");
  }
  // Convert discovered interactions to CSR-like target rows. A target's
  // contributions are contiguous, giving the portable and CUDA executors one
  // output owner and deterministic accumulation without atomics on the CPU.
  m2l_plan_.source_nodes.resize(interaction_count);
  m2l_plan_.matrix_ids.resize(interaction_count);
  m2l_plan_.source_levels.resize(interaction_count);
  m2l_plan_.target_levels.resize(interaction_count);
  m2l_plan_.interaction_levels.resize(interaction_count);
  std::vector<int> row_cursors = m2l_plan_.target_row_offsets;
  for (const ClassEntry* entry : ordered_classes) {
    const int matrix_id = universal_class_ids.at(entry->first);
    for (const auto [source, target] : entry->second) {
      const int slot = row_cursors[static_cast<std::size_t>(target)]++;
      m2l_plan_.source_nodes[static_cast<std::size_t>(slot)] = source;
      m2l_plan_.matrix_ids[static_cast<std::size_t>(slot)] = matrix_id;
      m2l_plan_.source_levels[static_cast<std::size_t>(slot)] =
          nodes[static_cast<std::size_t>(source)].level;
      m2l_plan_.target_levels[static_cast<std::size_t>(slot)] =
          nodes[static_cast<std::size_t>(target)].level;
      m2l_plan_.interaction_levels[static_cast<std::size_t>(slot)] =
          nodes[static_cast<std::size_t>(target)].level;
    }
  }
  if (has_periodic_root) {
    const int slot = row_cursors[static_cast<std::size_t>(topology_->root)]++;
    m2l_plan_.source_nodes[static_cast<std::size_t>(slot)] = topology_->root;
    m2l_plan_.matrix_ids[static_cast<std::size_t>(slot)] =
        static_cast<int>(universal_classes.size());
    m2l_plan_.source_levels[static_cast<std::size_t>(slot)] = 0;
    m2l_plan_.target_levels[static_cast<std::size_t>(slot)] = 0;
    m2l_plan_.interaction_levels[static_cast<std::size_t>(slot)] = 0;
  }
  static_plan_statistics_.operator_construction.add(
      elapsed_seconds(phase_start));
  static_plan_statistics_.m2l_plan =
      static_plan_statistics_.operator_construction;

  phase_start = Clock::now();
  const std::size_t matrix_bytes = m2l_plan_.matrices.size() * sizeof(double);
  const std::size_t scaling_bytes =
      (m2l_plan_.multipole_scaling.size() + m2l_plan_.local_scaling.size()) *
      sizeof(double);
  const std::size_t metadata_bytes =
      (m2l_plan_.target_row_offsets.size() + m2l_plan_.source_nodes.size() +
       m2l_plan_.matrix_ids.size() + m2l_plan_.source_levels.size() +
       m2l_plan_.target_levels.size() + m2l_plan_.interaction_levels.size() +
       m2l_plan_.level_target_begin.size() +
       m2l_plan_.level_target_end.size() +
       m2l_plan_.target_level_offsets.size() +
       m2l_plan_.target_nodes_by_level.size() +
       m2l_plan_.node_levels.size()) *
      sizeof(int);
  static_plan_statistics_.operator_bytes += matrix_bytes + scaling_bytes;
  static_plan_statistics_.m2l_operator_bytes += matrix_bytes + scaling_bytes;
  static_plan_statistics_.interaction_bytes += metadata_bytes;
  static_plan_statistics_.m2l_interaction_bytes += metadata_bytes;
  static_plan_statistics_.interactions = interaction_count;

  static_plan_statistics_.transfer_classes = universal_classes.size();
  static_plan_statistics_.m2l_operators =
      static_cast<std::size_t>(m2l_plan_.matrix_count);
  static_plan_statistics_.buffer_allocation.add(elapsed_seconds(phase_start));
  phase_start = Clock::now();
  // Target evaluators are independent, so the leaves are built in parallel.
  // The per-target byte accounting is a closed form, so it no longer has to be
  // summed from inside the loop.
  // One evaluator per target, resolved from the target's geometry and the
  // far-field model.  Naming the choice keeps the parallel loop below readable.
  const auto build_target_evaluator =
      [&](const Vec3& centre, const std::size_t target) {
    const bool exact_prism =
        target_geometry_ == TargetGeometry::RectangularPrism &&
        far_field_target_model_ == TargetModel::ExactGeometry;
    const bool exact_tetrahedron =
        target_geometry_ == TargetGeometry::Tetrahedron &&
        far_field_target_model_ == TargetModel::ExactGeometry;
    const Vec3& position = sorted_targets[target];
    if (expansion_basis_ == ExpansionBasis::Spherical) {
      if (exact_prism) {
        return build_static_cuboid_l2p_evaluator(
            spherical_basis_, centre, position,
            target_sizes[target_sizes.size() == 1 ? 0 : target]);
      }
      if (exact_tetrahedron) {
        return build_static_tetrahedron_l2p_evaluator(
            spherical_basis_, centre, position,
            sorted_target_tetrahedra_[
                sorted_target_tetrahedra_.size() == 1 ? 0 : target]);
      }
      return build_static_l2p_evaluator(spherical_basis_, centre, position);
    }
    if (exact_prism) {
      return build_static_cuboid_l2p_evaluator(
          basis_, centre, position,
          target_sizes[target_sizes.size() == 1 ? 0 : target]);
    }
    if (exact_tetrahedron) {
      return build_static_tetrahedron_l2p_evaluator(
          basis_, centre, position,
          sorted_target_tetrahedra_[
              sorted_target_tetrahedra_.size() == 1 ? 0 : target]);
    }
    return build_static_l2p_evaluator(basis_, centre, position);
  };

  // The targets an occupied leaf covers, and each one's leaf centre, so the
  // evaluator and its reuse key are both resolved from the flat target index.
  // Only covered targets get an evaluator, exactly as before.
  std::vector<Vec3> target_leaf_centre(sorted_targets.size());
  std::vector<std::uint32_t> evaluated_targets;
  evaluated_targets.reserve(sorted_targets.size());
  for (const StaticLeafRange& leaf_range : topology_->target_leaves) {
    const auto& leaf =
        topology_->nodes[static_cast<std::size_t>(leaf_range.node)];
    for (std::size_t target = leaf_range.begin;
         target < leaf_range.begin + leaf_range.count; ++target) {
      target_leaf_centre[target] = leaf.centre;
      evaluated_targets.push_back(static_cast<std::uint32_t>(target));
    }
  }

  // Everything one target's evaluator depends on, in bits: its offset from the
  // leaf centre and its shape record.
  const auto target_operator_key =
      [&](const std::size_t slot) -> EndpointOperatorKey {
    const std::size_t target =
        static_cast<std::size_t>(evaluated_targets[slot]);
    const bool exact_prism =
        target_geometry_ == TargetGeometry::RectangularPrism &&
        far_field_target_model_ == TargetModel::ExactGeometry;
    const bool exact_tetrahedron =
        target_geometry_ == TargetGeometry::Tetrahedron &&
        far_field_target_model_ == TargetModel::ExactGeometry;
    EndpointOperatorKey key;
    key.reserve(exact_tetrahedron ? 15 : 6);
    push_bits(key, sorted_targets[target] - target_leaf_centre[target]);
    if (exact_prism) {
      push_bits(key, target_sizes[target_sizes.size() == 1 ? 0 : target]);
    } else if (exact_tetrahedron) {
      push_bits(key, sorted_target_tetrahedra_[
          sorted_target_tetrahedra_.size() == 1 ? 0 : target]);
    }
    return key;
  };

  // Target evaluators are independent of one another, so each distinct one is
  // built once and in parallel and then copied to the targets that share it.
  // The byte accounting is a closed form, so it no longer has to be summed
  // from inside the loop.  An exact finite target makes each evaluator
  // expensive and unequal, so the schedule is dynamic.
  l2p_evaluators_.resize(sorted_targets.size());
  {
    const std::size_t evaluator_count = evaluated_targets.size();
    const EndpointOperatorClasses classes =
        classify_endpoint_operators(evaluator_count, target_operator_key);
    const std::size_t build_total = classes.classified
        ? classes.representative.size() : evaluator_count;

    FirstConstructionFailure failure;
    // Without reuse there is nothing to hold aside, so the evaluators are
    // built straight into their slots rather than through a second full-size
    // buffer.
    std::vector<StaticL2PEvaluator> built(
        classes.classified ? build_total : 0);
    const std::ptrdiff_t build_count =
        static_cast<std::ptrdiff_t>(build_total);
#pragma omp parallel for schedule(dynamic, 64) if (build_count >= 64)
    for (std::ptrdiff_t raw_build = 0; raw_build < build_count; ++raw_build) {
      const std::size_t entry = static_cast<std::size_t>(raw_build);
      const std::size_t slot = classes.classified
          ? static_cast<std::size_t>(classes.representative[entry])
          : entry;
      if (failure.superseded(slot)) {
        continue;
      }
      try {
        const std::size_t target =
            static_cast<std::size_t>(evaluated_targets[slot]);
        StaticL2PEvaluator evaluator =
            build_target_evaluator(target_leaf_centre[target], target);
        if (classes.classified) {
          built[entry] = std::move(evaluator);
        } else {
          l2p_evaluators_[target] = std::move(evaluator);
        }
      } catch (...) {
        failure.record(slot);
      }
    }
    failure.rethrow_any();

    if (classes.classified) {
      const std::ptrdiff_t scatter_count =
          static_cast<std::ptrdiff_t>(evaluator_count);
#pragma omp parallel for schedule(static) if (scatter_count >= 256)
      for (std::ptrdiff_t raw_slot = 0; raw_slot < scatter_count; ++raw_slot) {
        const std::size_t slot = static_cast<std::size_t>(raw_slot);
        l2p_evaluators_[static_cast<std::size_t>(evaluated_targets[slot])] =
            built[classes.class_of_item[slot]];
      }
    }

    const std::size_t bytes = 4 * static_cast<std::size_t>(coefficient_count) *
        sizeof(double) * evaluator_count;
    static_plan_statistics_.operator_bytes += bytes;
    static_plan_statistics_.l2p_operator_bytes += bytes;
  }
  static_plan_statistics_.l2p_plan.add(elapsed_seconds(phase_start));

  phase_start = Clock::now();
  auto p2p_stage_start = Clock::now();
  if (periodic_.enabled) {
    std::vector<StaticP2PInteraction> near_interactions;
    for (const StaticP2PLeafRecord& record : topology_->p2p_leaf_records) {
      for (int target = static_cast<int>(record.target_begin);
           target < static_cast<int>(record.target_begin + record.target_count);
           ++target) {
        for (int source = static_cast<int>(record.source_begin);
             source < static_cast<int>(record.source_begin + record.source_count);
             ++source) {
          near_interactions.push_back({target, source, record.source_shift,
                                       record.skip_for_identity});
        }
      }
    }
    static_plan_statistics_.p2p_interaction_setup.add(
        elapsed_seconds(p2p_stage_start));
    p2p_stage_start = Clock::now();
    p2p_operator_ = build_static_p2p_operator(
        sorted_targets, sorted_positions, near_interactions,
        source_geometry_, source_sizes, sorted_source_tetrahedra_,
        target_geometry_, target_sizes, sorted_target_tetrahedra_,
        near_field_source_model_, near_field_target_model_);
  } else {
    std::vector<std::array<int, 2>> near_interactions;
    for (const StaticP2PLeafRecord& record : topology_->p2p_leaf_records) {
      for (int target = static_cast<int>(record.target_begin);
           target < static_cast<int>(record.target_begin + record.target_count);
           ++target) {
        for (int source = static_cast<int>(record.source_begin);
             source < static_cast<int>(record.source_begin + record.source_count);
             ++source) {
          near_interactions.push_back({target, source});
        }
      }
    }
    static_plan_statistics_.p2p_interaction_setup.add(
        elapsed_seconds(p2p_stage_start));
    p2p_stage_start = Clock::now();
    p2p_operator_ = build_static_p2p_operator(
        sorted_targets, sorted_positions, near_interactions,
        source_geometry_, source_sizes, sorted_source_tetrahedra_,
        target_geometry_, target_sizes, sorted_target_tetrahedra_,
        near_field_source_model_, near_field_target_model_);
  }
  static_plan_statistics_.p2p_canonical_operator.add(
      elapsed_seconds(p2p_stage_start));
  p2p_stage_start = Clock::now();
  const bool point_geometry_p2p =
      backend_ == ExecutionBackend::CpuStatic && selects_point_geometry_p2p();
  if (!point_geometry_p2p) {
    p2p_compact_plan_ = build_static_p2p_compact_plan(p2p_operator_);
  }
  try {
    build_reduced_symmetry_p2p_packing();
  } catch (const std::invalid_argument &error) {
    throw std::runtime_error(
        std::string("reduced-symmetry P2P topology is not representable: ") +
        error.what());
  }
  if (backend_ == ExecutionBackend::CpuStatic) {
    p2p_execution_packing_ = resolve_cpu_p2p_packing();
  }
  static_plan_statistics_.p2p_derived_packing.add(
      elapsed_seconds(p2p_stage_start));
  static_plan_statistics_.p2p_interactions = p2p_operator_.blocks.size();
  static_plan_statistics_.p2p_value_bytes =
      p2p_operator_.blocks.size() * 6 * sizeof(double);
  static_plan_statistics_.p2p_index_bytes =
      p2p_compact_plan_.row_offsets.size() * sizeof(int) +
      p2p_compact_plan_.source_indices.size() * sizeof(int) +
      p2p_compact_plan_.skip_for_identity.size() * sizeof(unsigned char);
  static_plan_statistics_.p2p_canonical_total_bytes =
      p2p_compact_plan_.memory().total_bytes();
  if (p2p_tensor_dictionary_plan_.has_value()) {
    static_plan_statistics_.p2p_unique_tensors =
        p2p_tensor_dictionary_plan_->tensors[0].size();
    static_plan_statistics_.p2p_dictionary_tokens =
        p2p_tensor_dictionary_plan_->token_count();
    static_plan_statistics_.p2p_dictionary_token_width_bytes =
        p2p_tensor_dictionary_plan_->token_width_bytes;
    static_plan_statistics_.p2p_dictionary_token_bytes =
        p2p_tensor_dictionary_plan_->token_count() *
        p2p_tensor_dictionary_plan_->token_width_bytes;
    static_plan_statistics_.p2p_dictionary_tensor_bytes =
        p2p_tensor_dictionary_plan_->tensors[0].size() * 6 * sizeof(double);
    static_plan_statistics_.p2p_dictionary_total_bytes =
        p2p_tensor_dictionary_plan_->memory().total_bytes();
  }
  static_plan_statistics_.operator_bytes +=
      p2p_operator_.memory_bytes() + p2p_compact_plan_.memory().total_bytes();
  static_plan_statistics_.near_field_operator_bytes =
      p2p_operator_.memory_bytes() + p2p_compact_plan_.memory().total_bytes();
  static_plan_statistics_.p2p_tensor_plan.add(elapsed_seconds(phase_start));
  static_plan_statistics_.total.add(elapsed_seconds(total_start));
  ++static_plan_statistics_.construction_count;

  detail::cache::write_geometry_cache(
      geometry_identity, *tree_, fixed_target_source_indices_, p2m_plans_,
      m2l_plan_, l2p_evaluators_, p2p_operator_, static_plan_statistics_);

  if (precision_ == StaticPrecision::Float32) {
    const auto conversion_start = Clock::now();
    quantise_static_plan_to_float();
    static_plan_statistics_.precision_conversion.add(
        elapsed_seconds(conversion_start));
  }
  // oneMKL derives execution-only gather/GEMM/scatter packing from the same
  // canonical target-row metadata used by portable CPU and CUDA.
  build_backend_packing();
}

void UniformFmm::quantise_static_plan_to_float() {
  const bool effective_point_source =
      source_geometry_ == SourceGeometry::PointDipole ||
      near_field_source_model_ == SourceModel::PointDipole;
  const bool bsr_identity_compatible =
      !effective_point_source || fixed_target_source_indices_.has_value();
  if (!geometry_cache_loaded_direct_float_) {
    p2m_plans_float_.reserve(p2m_plans_.size());
    for (const P2MPlan &plan : p2m_plans_) {
      p2m_plans_float_.push_back(
          {plan.leaf, plan.begin, plan.count,
           quantise_static_operator(plan.operator_map)});
    }
  }

  for (int child_class = 0; child_class < 8; ++child_class) {
    m2m_operators_float_[child_class] =
        quantise_static_operator(m2m_operators_[child_class]);
    l2l_operators_float_[child_class] =
        quantise_static_operator(l2l_operators_[child_class]);
  }

  if (!geometry_cache_loaded_direct_float_) {
    l2p_evaluators_float_.reserve(l2p_evaluators_.size());
    for (const StaticL2PEvaluator &evaluator : l2p_evaluators_) {
      l2p_evaluators_float_.push_back(
          quantise_static_l2p_evaluator(evaluator));
    }
    p2p_operator_float_ = quantise_static_p2p_operator(p2p_operator_);
    m2l_plan_float_ = quantise_static_m2l_plan(m2l_plan_);
  } else {
    // Universal matrices are stored separately from the geometry plan. Only
    // this shared array still needs conversion after a direct FP32 load.
    m2l_plan_float_.matrices.assign(m2l_plan_.matrices.begin(),
                                    m2l_plan_.matrices.end());
  }

  const auto p2p_packing_start = Clock::now();
  const bool uses_cuda_plan = backend_ == ExecutionBackend::CudaM2LP2P ||
      backend_ == ExecutionBackend::CudaFull;
  if (uses_cuda_plan) {
    p2p_compact_plan_float_ = {};
    // The FP64 dictionary exists only when the CUDA execution policy (an
    // explicit option or the regular-grid hint) selected it.
    if (p2p_tensor_dictionary_plan_.has_value()) {
      p2p_tensor_dictionary_plan_float_ =
          quantise_static_p2p_signed_tensor_dictionary_plan(
              *p2p_tensor_dictionary_plan_);
    }
  } else {
    if (p2p_execution_packing_ != P2PExecutionPacking::PointGeometry) {
      p2p_compact_plan_float_ =
          build_static_p2p_compact_plan(p2p_operator_float_);
    }
    if (p2p_tensor_dictionary_plan_.has_value()) {
      auto dictionary = quantise_static_p2p_signed_tensor_dictionary_plan(
          *p2p_tensor_dictionary_plan_);
      p2p_tensor_dictionary_plan_float_ = std::move(dictionary);
      p2p_execution_packing_ = P2PExecutionPacking::TensorDictionary;
    }
  }
  if (bsr_identity_compatible &&
      estimate_bsr_bytes(p2p_operator_float_, sizeof(float)) <=
          cuda_p2p_bsr_max_bytes_) {
    const std::span<const int> bsr_identities =
        fixed_target_source_indices_.has_value()
            ? std::span<const int>(fixed_sorted_self_indices_)
            : std::span<const int>{};
    p2p_bsr_plan_float_ = build_static_p2p_bsr_plan(
        p2p_operator_float_, bsr_identities);
  }
  static_plan_statistics_.backend_packing.add(
      elapsed_seconds(p2p_packing_start));

  // Recalculate scalar-dependent storage from the representation that will
  // remain alive. Integer metadata is unchanged by precision selection.
  std::size_t operator_bytes = 0;
  std::size_t p2m_bytes = 0;
  std::size_t m2m_bytes = 0;
  std::size_t l2l_bytes = 0;
  for (const FloatP2MPlan &plan : p2m_plans_float_) {
    p2m_bytes +=
        plan.operator_map.entries.size() * sizeof(FloatStaticOperatorEntry);
  }
  for (int child_class = 0; child_class < 8; ++child_class) {
    m2m_bytes += m2m_operators_float_[child_class].entries.size() *
                 sizeof(FloatStaticOperatorEntry);
    l2l_bytes += l2l_operators_float_[child_class].entries.size() *
                 sizeof(FloatStaticOperatorEntry);
  }
  const std::size_t m2l_bytes = (m2l_plan_float_.matrices.size() +
       m2l_plan_float_.multipole_scaling.size() +
       m2l_plan_float_.local_scaling.size()) *
      sizeof(float);
  std::size_t l2p_bytes = 0;
  for (const FloatStaticL2PEvaluator &evaluator : l2p_evaluators_float_) {
    l2p_bytes += evaluator.potential.size() * sizeof(float);
    for (const std::vector<float> &field : evaluator.field) {
      l2p_bytes += field.size() * sizeof(float);
    }
  }
  const std::size_t near_field_bytes =
      p2p_operator_float_.memory_bytes() +
      p2p_compact_plan_float_.memory().total_bytes();
  operator_bytes = p2m_bytes + m2m_bytes + l2l_bytes + m2l_bytes + l2p_bytes +
                   near_field_bytes;

  static_plan_statistics_.scalar_bytes = sizeof(float);
  static_plan_statistics_.operator_bytes = operator_bytes;
  static_plan_statistics_.p2m_operator_bytes = p2m_bytes;
  static_plan_statistics_.m2m_operator_bytes = m2m_bytes;
  static_plan_statistics_.m2l_operator_bytes = m2l_bytes;
  static_plan_statistics_.l2l_operator_bytes = l2l_bytes;
  static_plan_statistics_.l2p_operator_bytes = l2p_bytes;
  static_plan_statistics_.near_field_operator_bytes = near_field_bytes;
  static_plan_statistics_.p2p_value_bytes =
      p2p_operator_float_.blocks.size() * 6 * sizeof(float);
  static_plan_statistics_.p2p_index_bytes =
      p2p_operator_float_.row_offsets.size() * sizeof(int) +
      p2p_operator_float_.blocks.size() * 2 * sizeof(int) +
      p2p_compact_plan_float_.source_indices.size() * sizeof(int) +
      p2p_compact_plan_float_.skip_for_identity.size() *
          sizeof(unsigned char);
  if (uses_cuda_plan) {
    // CUDA does not retain the CPU compact plan. Still report its equivalent
    // baseline footprint so reduced-symmetry memory comparisons remain valid
    // without constructing and immediately discarding that packing.
    static_plan_statistics_.p2p_canonical_total_bytes =
        p2p_operator_float_.blocks.size() *
            (9 * sizeof(float) + sizeof(int) + sizeof(unsigned char)) +
        p2p_operator_float_.row_offsets.size() * sizeof(int);
  } else {
    static_plan_statistics_.p2p_canonical_total_bytes =
        p2p_compact_plan_float_.memory().total_bytes();
  }
  if (p2p_tensor_dictionary_plan_float_.has_value()) {
    static_plan_statistics_.p2p_unique_tensors =
        p2p_tensor_dictionary_plan_float_->tensors[0].size();
    static_plan_statistics_.p2p_dictionary_tokens =
        p2p_tensor_dictionary_plan_float_->token_count();
    static_plan_statistics_.p2p_dictionary_token_width_bytes =
        p2p_tensor_dictionary_plan_float_->token_width_bytes;
    static_plan_statistics_.p2p_dictionary_token_bytes =
        p2p_tensor_dictionary_plan_float_->token_count() *
        p2p_tensor_dictionary_plan_float_->token_width_bytes;
    static_plan_statistics_.p2p_dictionary_tensor_bytes =
        p2p_tensor_dictionary_plan_float_->tensors[0].size() * 6 *
        sizeof(float);
    static_plan_statistics_.p2p_dictionary_total_bytes =
        p2p_tensor_dictionary_plan_float_->memory().total_bytes();
  }
  static_plan_statistics_.scratch_bytes = 0;

  // Drop analytical FP64 construction temporaries. An FP32 plan therefore
  // retains no hidden double-precision operator or expansion representation.
  p2m_plans_.clear();
  p2m_plans_.shrink_to_fit();
  m2m_operators_ = {};
  l2l_operators_ = {};
  l2p_evaluators_.clear();
  l2p_evaluators_.shrink_to_fit();
  p2p_operator_ = {};
  p2p_compact_plan_ = {};
  p2p_tensor_dictionary_plan_.reset();
  m2l_plan_ = {};
}

} // namespace cdfmm
