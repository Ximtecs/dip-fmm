// SPDX-License-Identifier: Apache-2.0
//
// UniformFmm plan preparation: the geometry-dependent, moment-independent
// half of the evaluator.  `build_static_plan` owns the cache-versus-build
// decision for the two cached payloads (the depth-independent universal bank
// of M2M/L2L/M2L operators, and the geometry plan of P2M plans, M2L schedule,
// L2P evaluators and the canonical near-field operator), builds whatever is
// missing from the canonical operator constructors, quantises an FP32 plan
// once, and hands the finished canonical data to `build_backend_packing`.
// Construction statistics are recorded per phase so that a warm plan (cache
// hit) and a cold plan report identical resident storage and distinguishable
// timings.

#include "cdfmm/uniform_fmm.hpp"

#include <algorithm>
#include <chrono>
#include <atomic>
#include <bit>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <tuple>
#include <unordered_map>

#include "cdfmm/operators/operators.hpp"
#include "cdfmm/plan/static_plan.hpp"

#include "cache/internal.hpp"
#include "fmm/p2p_construction.hpp"
#include "fmm/internal.hpp"
#include "plan/p2p/signed_dictionary_builder.hpp"
#include "phase_stopwatch.hpp"

// Construction clocks follow the plan's timing level: the static-plan total
// is a `Coarse` stopwatch, every subphase a `Detailed` one, and a stopwatch
// below the selected level reads no clock and records nothing.

namespace cdfmm {

namespace {

/// @brief Number of list-1 (target, source) pairs the leaf records describe.
///
/// This is the pair count of the canonical near-field operator, which holds
/// exactly one block per pair, so a position-based plan that never builds
/// that operator still reports the same interaction count.
[[nodiscard]] std::size_t list1_pair_count(const StaticFmmTopology& topology) {
  std::size_t pairs = 0;
  for (const StaticP2PLeafRecord& record : topology.p2p_leaf_records) {
    pairs += record.target_count * record.source_count;
  }
  return pairs;
}

/// @brief Bytes of the SoA particle rows of `pairs` list-1 pairs: row offsets,
///        source index, identity marker, three potential and six tensor
///        values per pair (StaticP2PCompactPlan). Reported as the baseline of
///        the dictionary comparisons whether or not the plan keeps the rows.
[[nodiscard]] std::size_t baseline_row_bytes(const std::size_t pairs,
                                             const std::size_t targets,
                                             const std::size_t scalar) {
  return pairs * (9 * scalar + sizeof(int) + sizeof(unsigned char)) +
         (targets + 1) * sizeof(int);
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
/// repay it, or when the item indices do not fit the 32-bit words the class
/// map and the representative list store.  Either only ever changes
/// performance: the caller builds the same operators either way.
template <typename KeyOfItem>
[[nodiscard]] EndpointOperatorClasses classify_endpoint_operators(
    const std::size_t item_count, const KeyOfItem& key_of_item) {
  constexpr std::size_t sample_items = 4096;
  constexpr std::size_t sample_reuse_factor = 2;

  // Both vectors hold `std::uint32_t`, and the largest value either can carry
  // is `item_count - 1`.  Abandon rather than narrow silently.  Endpoint items
  // are leaves or targets, so this is bounded by the body count rather than by
  // its square and is far out of reach in practice; the guard keeps the
  // narrowing contract the same as the shared pair classifier's.
  if (item_count != 0 &&
      item_count - 1 > std::numeric_limits<std::uint32_t>::max()) {
    return {};
  }

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

} // namespace

// Cold construction of the universal bank: the eight M2M and eight L2L
// child-class operators at the level-one child offset (+-1/4 in each axis;
// deeper levels rescale them by powers of two), the 316 M2L matrices of the
// complete interaction list (all integer displacements within +-3 excluding
// the 27 nearest), and, for a periodic plan, the periodic root operator
// appended as one more matrix.  All of these are dimensionless functions of
// basis and order alone, which is what makes the bank shareable across every
// geometry and depth.
void UniformFmm::build_missing_universal_operators(
    const bool universal_available, const bool periodic_required) {
  constexpr std::size_t class_count =
      StaticPlanStatistics::theoretical_maximum_m2l_classes;
  if (!universal_available) {
    detail::PhaseStopwatch shared_clock(detailed_timing());
    shared_clock.start();
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
    if (shared_clock.enabled()) {
      const double shared_seconds = shared_clock.elapsed();
      static_plan_statistics_.m2m_plan.add(shared_seconds);
      static_plan_statistics_.l2l_plan.add(shared_seconds);
      static_plan_statistics_.universal_operator_build.add(shared_seconds);
    }

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
    detail::PhaseStopwatch m2l_clock(detailed_timing());
    m2l_clock.start();
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
    m2l_clock.record(static_plan_statistics_.universal_operator_build);
  }

  if (periodic_required && !periodic_operator_available_) {
    detail::PhaseStopwatch periodic_clock(detailed_timing());
    periodic_clock.start();
    const std::vector<double> matrix =
        expansion_basis_ == ExpansionBasis::Spherical
        ? build_static_periodic_m2l_matrix(spherical_basis_, periodic_)
        : build_static_periodic_m2l_matrix(basis_, periodic_);
    m2l_plan_.matrices.insert(m2l_plan_.matrices.end(), matrix.begin(),
                              matrix.end());
    periodic_operator_available_ = true;
    periodic_clock.record(static_plan_statistics_.periodic_operator_build);
  }
}

void UniformFmm::build_static_plan() {
  // This is the geometry-dependent half of the evaluator. None of the data
  // built here depends on dipole moments, so it remains valid for every later
  // evaluate() call; see docs/static-architecture.md.
  detail::PhaseStopwatch total_clock(coarse_timing());
  total_clock.start();
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
      geometry_hash_digest_, expansion_basis_, precision_, expansion_order(),
      position_based_near_field(), dictionary_near_field(),
      signed_p2p_target_tile_size_};
  if (universal_available &&
      (!periodic_required || periodic_operator_available_) &&
      detail::cache::load_geometry_cache(
          geometry_identity, *tree_, *topology_, fixed_target_source_indices_,
          {p2m_plans_, p2m_plans_float_, m2l_plan_, m2l_plan_float_,
           l2p_evaluators_, l2p_evaluators_float_, p2p_operator_,
           p2p_operator_float_, p2p_compact_plan_, p2p_compact_plan_float_,
           p2p_bsr_plan_float_, p2p_tensor_dictionary_plan_,
           p2p_tensor_dictionary_plan_float_},
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
        position_based_near_field() || p2p_tensor_dictionary_plan_.has_value() ||
            p2p_tensor_dictionary_plan_float_.has_value()
        ? list1_pair_count(*topology_)
        : (precision_ == StaticPrecision::Float32
               ? p2p_operator_float_.blocks.size()
               : p2p_operator_.blocks.size());
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
    // account for it as on a cold build. A dictionary plan's file holds the
    // dictionary it executes (or, when the RegularGrid hint's dictionary did
    // not compress, the canonical records it fell back to), so nothing here
    // rebuilds a pair tensor.
    detail::PhaseStopwatch warm_stage_clock(detailed_timing());
    warm_stage_clock.start();
    if (!dictionary_near_field()) {
      try {
        build_reduced_symmetry_p2p_packing();
      } catch (const std::invalid_argument &error) {
        throw std::runtime_error(
            std::string("reduced-symmetry P2P topology is not representable: ") +
            error.what());
      }
    }
    if (precision_ == StaticPrecision::Float64) {
      static_plan_statistics_.p2p_value_bytes =
          static_plan_statistics_.p2p_interactions * 6 * sizeof(double);
      static_plan_statistics_.p2p_index_bytes =
          p2p_compact_plan_.row_offsets.size() * sizeof(int) +
          p2p_compact_plan_.source_indices.size() * sizeof(int) +
          p2p_compact_plan_.skip_for_identity.size() * sizeof(unsigned char);
      static_plan_statistics_.p2p_canonical_total_bytes =
          baseline_row_bytes(static_plan_statistics_.p2p_interactions,
                             topology_->sorted_target_positions.size(),
                             sizeof(double));
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
    // Only the derived-packing timer runs here.  The construction timers,
    // `p2p_tensor_plan` among them, stay at zero calls on a cache hit: that is
    // how a warm plan is distinguished from a rebuilt one.
    warm_stage_clock.record(static_plan_statistics_.p2p_derived_packing);
    if (precision_ == StaticPrecision::Float32) {
      warm_stage_clock.start();
      quantise_static_plan_to_float();
      warm_stage_clock.record(static_plan_statistics_.precision_conversion);
    }
    build_backend_packing();
    total_clock.record(static_plan_statistics_.total);
    return;
  }
  // ---- Cold construction of the geometry plan ------------------------------
  // From here on the geometry cache missed (or was disabled).  The phases
  // below build, in order: leaf P2M operators, the translation templates (if
  // the universal bank also missed), the M2L class discovery and schedule, the
  // per-target L2P evaluators, and the canonical near-field operator with its
  // derived packings; the plan is then written to the cache and, for an FP32
  // plan, quantised.  Each phase updates its own timer and byte counters.
  const bool universal_cache_hit = universal_available;
  const auto &nodes = topology_->nodes;
  // An M2L transfer class is the normalised displacement (in target widths)
  // plus the source/target width ratio; a uniform tree only ever produces
  // ratio 1, an adaptive topology adds cross-level classes.
  using Key = std::tuple<double, double, double, double>;
  using ClassMap = std::map<Key, std::vector<std::pair<int, int>>>;
  ClassMap classes;

  detail::PhaseStopwatch phase_clock(detailed_timing());
  phase_clock.start();
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
  // A procedural P2M stage evaluates the expansion from the positions and
  // never reads these maps. They are still built when the geometry cache
  // persists them (its key does not carry the execution choice) and for the
  // reference backend, which may apply them regardless of the flag.
  const bool endpoint_maps_persisted =
      cache_enabled_ || backend_ == ExecutionBackend::CpuReference;
  p2m_plans_.clear();
  if (!procedural_p2m_ || endpoint_maps_persisted) {
    p2m_plans_.assign(topology_->source_leaves.size(), P2MPlan{});
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
  phase_clock.record(static_plan_statistics_.p2m_plan);

  phase_clock.start();
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
  if (phase_clock.enabled()) {
    const double shared_translation_seconds = phase_clock.elapsed();
    static_plan_statistics_.m2m_plan.add(shared_translation_seconds);
    if (!universal_cache_hit) {
      static_plan_statistics_.universal_operator_build.add(
          shared_translation_seconds);
    }
  }
  // The two triangular families are constructed together from each shared
  // parent-child displacement class.
  static_plan_statistics_.l2l_plan = static_plan_statistics_.m2m_plan;

  phase_clock.start();
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
  phase_clock.record(static_plan_statistics_.transfer_discovery);

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

  phase_clock.start();
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
  detail::PhaseStopwatch universal_m2l_clock(detailed_timing());
  universal_m2l_clock.start();
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
    universal_m2l_clock.record(static_plan_statistics_.universal_operator_build);
  }

  if (has_periodic_root && !periodic_operator_available_) {
    // Wrapped traversal resolves the central root and its 26 neighbours. The
    // appended self translation represents every more distant lattice image.
    detail::PhaseStopwatch periodic_build_clock(detailed_timing());
    periodic_build_clock.start();
    const std::vector<double> matrix =
        expansion_basis_ == ExpansionBasis::Spherical
            ? build_static_periodic_m2l_matrix(spherical_basis_, periodic_)
            : build_static_periodic_m2l_matrix(basis_, periodic_);
    std::copy(
        matrix.begin(), matrix.end(),
        m2l_plan_.matrices.begin() +
            static_cast<std::ptrdiff_t>(universal_classes.size() *
                                        matrix_values));
    periodic_build_clock.record(static_plan_statistics_.periodic_operator_build);
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
    for (const auto &[source, target] : entry->second) {
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
  phase_clock.record(static_plan_statistics_.operator_construction);
  static_plan_statistics_.m2l_plan =
      static_plan_statistics_.operator_construction;

  phase_clock.start();
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
  phase_clock.record(static_plan_statistics_.buffer_allocation);
  phase_clock.start();
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
  // As for P2M: a procedural L2P stage never reads the evaluators.
  l2p_evaluators_.clear();
  if (!procedural_l2p_ || endpoint_maps_persisted) {
    l2p_evaluators_.resize(sorted_targets.size());
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
  phase_clock.record(static_plan_statistics_.l2p_plan);

  // Canonical near field: expand the list-1 leaf records into individual
  // (target, source[, image shift, identity marker]) pairs and build one
  // exact tensor per pair (src/operators/p2p.cpp classifies and reuses them).
  // The three sub-timers separate record expansion, tensor construction and
  // derived packing so a warm plan, which only pays the last, is
  // distinguishable.
  //
  // A position-based near field (`PointGeometry` on the CPU or CUDA) reads
  // only the leaf records and the sorted positions, so neither the pair list
  // nor the canonical operator is built: at N x 27 x occupancy pairs that is
  // what bounded the reachable leaf occupancy. Its sub-timers then record
  // nothing, as nothing ran.
  phase_clock.start();
  detail::PhaseStopwatch p2p_stage_clock(detailed_timing());
  const bool position_based = position_based_near_field();
  // A dictionary plan builds its dictionary chunk by chunk and holds no
  // canonical operator; when the RegularGrid hint's dictionary does not
  // compress it falls back to the general representation.
  bool dictionary_path = false;
  p2p_float_prebuilt_ = false;
  std::unique_ptr<detail::cache::GeometryCacheWriter> canonical_cache_stream;
  std::unique_ptr<detail::cache::GeometryCacheWriter> dictionary_cache_stream;
  // A file that receives the canonical records as the near field is built:
  // their row offsets and count are known from the list-1 records first.
  const auto make_canonical_cache_stream = [&]() {
    return std::make_unique<detail::cache::GeometryCacheWriter>(
        geometry_identity, *tree_, fixed_target_source_indices_, p2m_plans_,
        m2l_plan_, l2p_evaluators_, static_cast<int>(sorted_positions.size()),
        static_cast<int>(sorted_targets.size()),
        detail::p2p_construction::canonical_row_offsets(*topology_,
                                                        sorted_targets.size()),
        list1_pair_count(*topology_), static_plan_statistics_);
  };
  if (position_based) {
    p2p_operator_ = {};
  } else if (dictionary_near_field()) {
    dictionary_path = true;
    if (!build_dictionary_near_field(sorted_targets, sorted_positions,
                                     source_sizes, target_sizes)) {
      // The fallback persists its canonical records as a canonical-keyed
      // plan would, so a warm plan loads rather than rebuilds them.
      if (cache_enabled_) {
        canonical_cache_stream = make_canonical_cache_stream();
      }
      build_general_near_field(sorted_targets, sorted_positions, source_sizes,
                               target_sizes, canonical_cache_stream.get());
    }
  } else if (cache_enabled_ && backend_ != ExecutionBackend::CpuReference) {
    // A canonical-keyed file receives the records as they are built, so the
    // full FP64 operator never has to exist for the cache's sake.
    canonical_cache_stream = make_canonical_cache_stream();
    build_general_near_field(sorted_targets, sorted_positions, source_sizes,
                             target_sizes, canonical_cache_stream.get());
  } else {
    build_general_near_field(sorted_targets, sorted_positions, source_sizes,
                             target_sizes);
  }
  p2p_stage_clock.start();
  if (!dictionary_path) {
    try {
      build_reduced_symmetry_p2p_packing();
    } catch (const std::invalid_argument &error) {
      throw std::runtime_error(
          std::string("reduced-symmetry P2P topology is not representable: ") +
          error.what());
    }
  }
  if (backend_ == ExecutionBackend::CpuStatic) {
    p2p_execution_packing_ = resolve_cpu_p2p_packing();
  }
  p2p_stage_clock.record(static_plan_statistics_.p2p_derived_packing);
  static_plan_statistics_.p2p_interactions = p2p_operator_.blocks.empty()
      ? list1_pair_count(*topology_)
      : p2p_operator_.blocks.size();
  static_plan_statistics_.p2p_value_bytes =
      static_plan_statistics_.p2p_interactions * 6 * sizeof(double);
  static_plan_statistics_.p2p_index_bytes =
      p2p_compact_plan_.row_offsets.size() * sizeof(int) +
      p2p_compact_plan_.source_indices.size() * sizeof(int) +
      p2p_compact_plan_.skip_for_identity.size() * sizeof(unsigned char);
  static_plan_statistics_.p2p_canonical_total_bytes =
      baseline_row_bytes(static_plan_statistics_.p2p_interactions,
                         topology_->sorted_target_positions.size(),
                         sizeof(double));
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
  phase_clock.record(static_plan_statistics_.p2p_tensor_plan);
  total_clock.record(static_plan_statistics_.total);
  ++static_plan_statistics_.construction_count;

  if (canonical_cache_stream) {
    // A dictionary plan that fell back marks its records as the near field.
    canonical_cache_stream->append_dictionary_section(nullptr, nullptr, {}, {});
    canonical_cache_stream->finish();
    canonical_cache_stream.reset();
  } else if (dictionary_near_field()) {
    // The dictionary and the point potential rows built beside it are the
    // whole near field; the canonical section stays empty. The header goes
    // out now, while the FP64 far-field operators exist; the dictionary
    // follows in the plan's precision once an FP32 plan has converted it.
    if (cache_enabled_ && p2p_tensor_dictionary_plan_.has_value()) {
      dictionary_cache_stream = std::make_unique<detail::cache::GeometryCacheWriter>(
          geometry_identity, *tree_, fixed_target_source_indices_, p2m_plans_,
          m2l_plan_, l2p_evaluators_, 0, 0, std::span<const int>{}, 0,
          static_plan_statistics_);
    }
  } else {
    // A position-based plan persists an empty P2P section.
    detail::cache::write_geometry_cache(
        geometry_identity, *tree_, fixed_target_source_indices_, p2m_plans_,
        m2l_plan_, l2p_evaluators_,
        position_based_near_field() ? StaticP2POperator{} : p2p_operator_,
        static_plan_statistics_);
  }
  if (p2p_float_prebuilt_) {
    // The FP32 representations exist already; the FP64 rows were kept only
    // for the cache file.
    p2p_operator_ = {};
  }

  if (precision_ == StaticPrecision::Float32) {
    detail::PhaseStopwatch conversion_clock(detailed_timing());
    conversion_clock.start();
    quantise_static_plan_to_float();
    conversion_clock.record(static_plan_statistics_.precision_conversion);
  }
  if (dictionary_cache_stream) {
    dictionary_cache_stream->append_dictionary_section(
        p2p_tensor_dictionary_plan_ ? &*p2p_tensor_dictionary_plan_ : nullptr,
        p2p_tensor_dictionary_plan_float_ ? &*p2p_tensor_dictionary_plan_float_
                                          : nullptr,
        p2p_compact_plan_, p2p_compact_plan_float_);
    dictionary_cache_stream->finish();
    dictionary_cache_stream.reset();
  }
  // oneMKL derives execution-only gather/GEMM/scatter packing from the same
  // canonical target-row metadata used by portable CPU and CUDA.
  build_backend_packing();
}

// General near field, built chunk by chunk into exactly the representations
// the resolved plan reads: its executor's rows in its own precision (the
// canonical rows, the SoA rows or the CUDA leaf blocks) and, for a point near
// field, the SoA rows its potential output reads (FP32; FP64 on a periodic
// plan; never on CudaFull, which is field-only). The FP64 canonical operator
// is kept in full only when the cache persists it. CpuReference keeps the
// historical FP64 canonical and SoA pair. Every representation is derived
// from the same FP64 chunk rows, so it is bitwise what the one-shot
// derivation from the full operator produced.
void UniformFmm::build_general_near_field(
    const std::span<const Vec3> sorted_targets,
    const std::span<const Vec3> sorted_positions,
    const std::span<const CuboidSize> source_sizes,
    const std::span<const CuboidSize> target_sizes,
    detail::cache::GeometryCacheWriter *const cache_writer) {
  namespace chunked = detail::p2p_construction;
  const bool fp32 = precision_ == StaticPrecision::Float32;
  const bool cpu = backend_ == ExecutionBackend::CpuStatic;
  const bool cuda = backend_ == ExecutionBackend::CudaM2LP2P ||
      backend_ == ExecutionBackend::CudaFull;
  const bool full = backend_ == ExecutionBackend::CudaFull;
  const bool point_near_field =
      (source_geometry_ == SourceGeometry::PointDipole ||
       near_field_source_model_ == SourceModel::PointDipole) &&
      (target_geometry_ == TargetGeometry::Point ||
       near_field_target_model_ == TargetModel::Point);
  const bool point_geometry_p2p = cpu && selects_point_geometry_p2p();

  bool canonical = false;
  bool compact = false;
  bool leaf = false;
  bool historical = !cpu && !cuda;  // CpuReference
  if (cpu && !point_geometry_p2p) {
    if (requested_p2p_packing_ == P2PExecutionPacking::CanonicalAos) {
      canonical = true;
    } else {
      compact = true;
    }
  } else if (cuda) {
    const cuda_policy::CudaP2PPacking packing = cuda_policy_->policy.p2p_packing;
    if (dictionary_near_field()) {
      // A RegularGrid-hint dictionary that did not compress: the device
      // packing is re-resolved later from the General rules.
      canonical = true;
    } else if (packing == cuda_policy::CudaP2PPacking::LeafBlock) {
      leaf = true;
    } else if (packing != cuda_policy::CudaP2PPacking::PointGeometry) {
      canonical = true;
    }
  }
  const bool potential_rows = point_near_field && !full && !point_geometry_p2p &&
      !(cuda && cuda_policy_->policy.p2p_packing ==
                    cuda_policy::CudaP2PPacking::PointGeometry);
  if (potential_rows) {
    if (fp32) {
      compact = compact || !canonical;
    } else if (periodic_.enabled) {
      compact = true;
    }
  }
  if (historical) {
    p2p_operator_ = build_chunked_canonical_operator(
        sorted_targets, sorted_positions, source_sizes, target_sizes);
    p2p_compact_plan_ = build_static_p2p_compact_plan(p2p_operator_);
    return;
  }

  // The cache receives the FP64 records chunk by chunk (`cache_writer`), so
  // the full FP64 operator is kept only when the plan executes it.
  const bool canonical64 = !fp32 && canonical;
  const bool compact64 = !fp32 && compact;
  const bool leaf64 = !fp32 && leaf;
  const bool canonical32 = fp32 && canonical;
  const bool compact32 = fp32 && compact;
  const bool leaf32 = fp32 && leaf;
  // SoA rows kept only for potential output hold no tensor planes.
  const bool compact_tensors =
      cpu && !point_geometry_p2p &&
      requested_p2p_packing_ != P2PExecutionPacking::CanonicalAos;
  p2p_float_prebuilt_ = fp32;

  const chunked::CanonicalInputs inputs{
      sorted_targets,          sorted_positions,  source_geometry_,
      source_sizes,            sorted_source_tetrahedra_, target_geometry_,
      target_sizes,            sorted_target_tetrahedra_, near_field_source_model_,
      near_field_target_model_, periodic_.enabled};
  const std::vector<chunked::Chunk> chunks =
      chunked::plan_chunks(*topology_, chunked::chunk_pair_budget());
  const int source_count = static_cast<int>(sorted_positions.size());
  const int target_count = static_cast<int>(sorted_targets.size());
  const std::size_t pairs = chunked::total_pairs(chunks);
  if (chunks.empty()) {
    // Validate the body records exactly as the one-shot build does.
    static_cast<void>(build_static_p2p_operator(
        sorted_targets, sorted_positions, std::span<const StaticP2PInteraction>{},
        source_geometry_, source_sizes, sorted_source_tetrahedra_,
        target_geometry_, target_sizes, sorted_target_tetrahedra_,
        near_field_source_model_, near_field_target_model_));
  }
  p2p_operator_ = {};
  p2p_compact_plan_ = {};
  p2p_operator_float_ = {};
  p2p_compact_plan_float_ = {};
  p2p_leaf_plan_ = {};
  p2p_leaf_plan_float_ = {};
  if (canonical64) {
    chunked::start_rows(p2p_operator_, source_count, target_count, pairs);
  }
  if (compact64) {
    chunked::start_rows(p2p_compact_plan_, source_count, target_count, pairs,
                        compact_tensors);
  }
  if (canonical32) {
    chunked::start_rows(p2p_operator_float_, source_count, target_count, pairs);
  }
  if (compact32) {
    chunked::start_rows(p2p_compact_plan_float_, source_count, target_count, pairs,
                        compact_tensors);
  }
  p2p_leaf_plan_.source_count = source_count;
  p2p_leaf_plan_.target_count = target_count;
  p2p_leaf_plan_float_.source_count = source_count;
  p2p_leaf_plan_float_.target_count = target_count;

  const chunked::ChunkBuilder builder(*topology_, inputs);
  PhaseTiming interaction_setup{};
  PhaseTiming tensor_build{};
  const bool timed = detailed_timing();
  for (const chunked::Chunk &chunk : chunks) {
    const StaticP2POperator part =
        builder.build(chunk, interaction_setup, tensor_build, timed);
    if (cache_writer != nullptr) {
      cache_writer->append_p2p_blocks(part.blocks);
    }
    if (canonical64) {
      chunked::append_rows(p2p_operator_, part, chunk);
    }
    if (compact64) {
      chunked::append_compact_rows(p2p_compact_plan_,
                                   build_static_p2p_compact_plan(part), chunk,
                                   compact_tensors);
    }
    if (leaf64) {
      chunked::append_leaf_blocks(
          p2p_leaf_plan_, build_static_p2p_leaf_plan(part, builder.leaf_pairs(chunk)));
    }
    if (canonical32 || compact32) {
      const FloatStaticP2POperator part32 = quantise_static_p2p_operator(part);
      if (canonical32) {
        chunked::append_rows(p2p_operator_float_, part32, chunk);
      }
      if (compact32) {
        chunked::append_compact_rows(p2p_compact_plan_float_,
                                     build_static_p2p_compact_plan(part32), chunk,
                                     compact_tensors);
      }
    }
    if (leaf32) {
      chunked::append_leaf_blocks(
          p2p_leaf_plan_float_,
          quantise_static_p2p_leaf_plan(
              build_static_p2p_leaf_plan(part, builder.leaf_pairs(chunk))));
    }
  }
  if (canonical64) {
    chunked::finish_rows(p2p_operator_);
  }
  if (compact64) {
    chunked::finish_rows(p2p_compact_plan_);
  }
  if (canonical32) {
    chunked::finish_rows(p2p_operator_float_);
  }
  if (compact32) {
    chunked::finish_rows(p2p_compact_plan_float_);
  }
  if (leaf64) {
    chunked::finish_leaf_plan(p2p_leaf_plan_);
  }
  if (leaf32) {
    chunked::finish_leaf_plan(p2p_leaf_plan_float_);
  }
  if (timed) {
    static_plan_statistics_.p2p_interaction_setup.add(
        interaction_setup.total_seconds);
    static_plan_statistics_.p2p_canonical_operator.add(tensor_build.total_seconds);
  }
}

// Dictionary near field, built chunk by chunk: each chunk's canonical rows
// become dense leaf blocks, the blocks become tokens, and nothing larger
// than a chunk is ever resident besides the tokens. A point plan also keeps
// the SoA rows its potential output reads (FP32, or FP64 on a periodic
// plan), assembled from the same chunks; CudaFull evaluates field only.
// Returns false, keeping nothing, when a RegularGrid-hint dictionary needs
// wider tokens than the hint allows, so the caller builds the general form.
bool UniformFmm::build_dictionary_near_field(
    const std::span<const Vec3> sorted_targets,
    const std::span<const Vec3> sorted_positions,
    const std::span<const CuboidSize> source_sizes,
    const std::span<const CuboidSize> target_sizes) {
  namespace chunked = detail::p2p_construction;
  using Clock = std::chrono::steady_clock;
  const bool point_near_field =
      (source_geometry_ == SourceGeometry::PointDipole ||
       near_field_source_model_ == SourceModel::PointDipole) &&
      (target_geometry_ == TargetGeometry::Point ||
       near_field_target_model_ == TargetModel::Point);
  const bool potential_rows =
      point_near_field && backend_ != ExecutionBackend::CudaFull;
  const bool rows64 = potential_rows &&
      precision_ == StaticPrecision::Float64 && periodic_.enabled;
  const bool rows32 = potential_rows && precision_ == StaticPrecision::Float32;
  const int source_count = static_cast<int>(sorted_positions.size());
  const int target_count = static_cast<int>(sorted_targets.size());

  const chunked::CanonicalInputs inputs{
      sorted_targets,          sorted_positions,  source_geometry_,
      source_sizes,            sorted_source_tetrahedra_, target_geometry_,
      target_sizes,            sorted_target_tetrahedra_, near_field_source_model_,
      near_field_target_model_, periodic_.enabled};
  const std::vector<chunked::Chunk> chunks =
      chunked::plan_chunks(*topology_, chunked::chunk_pair_budget());
  if (chunks.empty()) {
    // Validate the body records exactly as the general build does.
    static_cast<void>(build_static_p2p_operator(
        sorted_targets, sorted_positions, std::span<const StaticP2PInteraction>{},
        source_geometry_, source_sizes, sorted_source_tetrahedra_,
        target_geometry_, target_sizes, sorted_target_tetrahedra_,
        near_field_source_model_, near_field_target_model_));
  }
  const chunked::ChunkBuilder builder(*topology_, inputs);
  detail::SignedTensorDictionaryBuilder dictionary(
      source_count, target_count, fixed_sorted_self_indices_,
      signed_p2p_target_tile_size_);
  p2p_operator_ = {};
  p2p_compact_plan_ = {};
  p2p_compact_plan_float_ = {};
  // These rows serve potential output only, which never reads the tensors.
  if (rows64) {
    chunked::start_rows(p2p_compact_plan_, source_count, target_count,
                        chunked::total_pairs(chunks), false);
  }
  if (rows32) {
    chunked::start_rows(p2p_compact_plan_float_, source_count, target_count,
                        chunked::total_pairs(chunks), false);
  }
  PhaseTiming interaction_setup{};
  PhaseTiming tensor_build{};
  double packing_seconds = 0.0;
  const bool timed = detailed_timing();
  for (const chunked::Chunk &chunk : chunks) {
    const StaticP2POperator part =
        builder.build(chunk, interaction_setup, tensor_build, timed);
    const auto packing_start = timed ? Clock::now() : Clock::time_point{};
    try {
      dictionary.append(build_static_p2p_leaf_plan(part, builder.leaf_pairs(chunk)));
    } catch (const std::invalid_argument &error) {
      throw std::runtime_error(
          std::string("reduced-symmetry P2P topology is not representable: ") +
          error.what());
    }
    if (rows64) {
      chunked::append_compact_rows(p2p_compact_plan_,
                                   build_static_p2p_compact_plan(part), chunk, false);
    }
    if (rows32) {
      chunked::append_compact_rows(
          p2p_compact_plan_float_,
          build_static_p2p_compact_plan(quantise_static_p2p_operator(part)), chunk,
          false);
    }
    if (timed) {
      packing_seconds +=
          std::chrono::duration<double>(Clock::now() - packing_start).count();
    }
  }
  if (rows64) {
    chunked::finish_rows(p2p_compact_plan_);
  }
  if (rows32) {
    chunked::finish_rows(p2p_compact_plan_float_);
  }
  const auto finish_start = timed ? Clock::now() : Clock::time_point{};
  p2p_tensor_dictionary_plan_ = dictionary.finish();
  if (timed) {
    packing_seconds +=
        std::chrono::duration<double>(Clock::now() - finish_start).count();
    static_plan_statistics_.p2p_interaction_setup.add(
        interaction_setup.total_seconds);
    static_plan_statistics_.p2p_canonical_operator.add(tensor_build.total_seconds);
    static_plan_statistics_.p2p_derived_packing.add(packing_seconds);
  }
  // A dictionary chosen from the layout hint is a prediction; the built plan
  // is the measurement (as in build_reduced_symmetry_p2p_packing).
  if (cuda_policy_->policy.dictionary_from_layout &&
      p2p_tensor_dictionary_plan_->token_width_bytes >
          cuda_policy::dictionary_layout_max_token_width_bytes()) {
    p2p_tensor_dictionary_plan_.reset();
    p2p_compact_plan_ = {};
    p2p_compact_plan_float_ = {};
    return false;
  }
  return true;
}

// Canonical near field, built one chunk of target leaves at a time
// (src/fmm/p2p_construction.hpp): the rows are bitwise the monolithic
// build's, but the pair list, the builder's sorted copy and its
// classification maps exist for one chunk only. The two sub-timers are
// summed over the chunks and recorded once.
StaticP2POperator UniformFmm::build_chunked_canonical_operator(
    const std::span<const Vec3> sorted_targets,
    const std::span<const Vec3> sorted_positions,
    const std::span<const CuboidSize> source_sizes,
    const std::span<const CuboidSize> target_sizes) {
  namespace chunked = detail::p2p_construction;
  const chunked::CanonicalInputs inputs{
      sorted_targets,          sorted_positions,  source_geometry_,
      source_sizes,            sorted_source_tetrahedra_, target_geometry_,
      target_sizes,            sorted_target_tetrahedra_, near_field_source_model_,
      near_field_target_model_, periodic_.enabled};
  const std::vector<chunked::Chunk> chunks =
      chunked::plan_chunks(*topology_, chunked::chunk_pair_budget());
  if (chunks.empty()) {
    // No list-1 pair: one empty build still validates the body records.
    return build_static_p2p_operator(
        sorted_targets, sorted_positions, std::span<const StaticP2PInteraction>{},
        source_geometry_, source_sizes, sorted_source_tetrahedra_,
        target_geometry_, target_sizes, sorted_target_tetrahedra_,
        near_field_source_model_, near_field_target_model_);
  }
  const chunked::ChunkBuilder builder(*topology_, inputs);
  StaticP2POperator canonical;
  chunked::start_rows(canonical, static_cast<int>(sorted_positions.size()),
                      static_cast<int>(sorted_targets.size()),
                      chunked::total_pairs(chunks));
  PhaseTiming interaction_setup{};
  PhaseTiming tensor_build{};
  for (const chunked::Chunk &chunk : chunks) {
    const StaticP2POperator part =
        builder.build(chunk, interaction_setup, tensor_build, detailed_timing());
    chunked::append_rows(canonical, part, chunk);
  }
  chunked::finish_rows(canonical);
  if (detailed_timing()) {
    static_plan_statistics_.p2p_interaction_setup.add(
        interaction_setup.total_seconds);
    static_plan_statistics_.p2p_canonical_operator.add(tensor_build.total_seconds);
  }
  return canonical;
}

// FP32 plans: quantise every completed FP64 operator once into its FP32
// container, derive the FP32 packings the selected backend needs, recompute
// the byte accounting from what stays resident, and release the FP64
// temporaries so the plan retains no hidden double-precision table.  After a
// direct FP32 cache load only the shared universal matrices still need
// conversion.
void UniformFmm::quantise_static_plan_to_float() {
  const bool effective_point_source =
      source_geometry_ == SourceGeometry::PointDipole ||
      near_field_source_model_ == SourceModel::PointDipole;
  const bool bsr_identity_compatible =
      !effective_point_source || fixed_target_source_indices_.has_value();
  // A procedural stage reads neither the FP64 nor the FP32 maps (the
  // reference backend never reaches this conversion with them unused).
  const bool convert_p2m = !procedural_p2m_ || backend_ == ExecutionBackend::CpuReference;
  const bool convert_l2p = !procedural_l2p_ || backend_ == ExecutionBackend::CpuReference;
  // A direct FP32 load brings the maps a shared geometry file holds; a
  // procedural stage reads neither, so they go exactly as a cold plan never
  // converts them.
  if (!convert_p2m) {
    p2m_plans_float_.clear();
    p2m_plans_float_.shrink_to_fit();
  }
  if (!convert_l2p) {
    l2p_evaluators_float_.clear();
    l2p_evaluators_float_.shrink_to_fit();
  }
  if (!geometry_cache_loaded_direct_float_ && convert_p2m) {
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
    if (convert_l2p) {
      l2p_evaluators_float_.reserve(l2p_evaluators_.size());
      for (const StaticL2PEvaluator &evaluator : l2p_evaluators_) {
        l2p_evaluators_float_.push_back(
            quantise_static_l2p_evaluator(evaluator));
      }
    }
    if (!p2p_float_prebuilt_) {
      p2p_operator_float_ = quantise_static_p2p_operator(p2p_operator_);
    }
    m2l_plan_float_ = quantise_static_m2l_plan(m2l_plan_);
  } else {
    // Universal matrices are stored separately from the geometry plan. Only
    // this shared array still needs conversion after a direct FP32 load.
    m2l_plan_float_.matrices.assign(m2l_plan_.matrices.begin(),
                                    m2l_plan_.matrices.end());
  }

  detail::PhaseStopwatch p2p_packing_clock(detailed_timing());
  p2p_packing_clock.start();
  const bool uses_cuda_plan = backend_ == ExecutionBackend::CudaM2LP2P ||
      backend_ == ExecutionBackend::CudaFull;
  // A warm dictionary plan loads the FP32 dictionary it executes; a cold one
  // converts its FP64 dictionary below.
  const bool dictionary_plan = p2p_tensor_dictionary_plan_.has_value() ||
      p2p_tensor_dictionary_plan_float_.has_value();
  if (uses_cuda_plan) {
    // Only a dictionary plan's point potential rows are built before this
    // point (CudaPartial evaluates potential on the host); nothing else here
    // uses the SoA rows.
    if (!dictionary_plan && !p2p_float_prebuilt_) {
      p2p_compact_plan_float_ = {};
    }
    // The FP64 dictionary exists only when the CUDA execution policy (an
    // explicit option or the regular-grid hint) selected it.
    if (p2p_tensor_dictionary_plan_.has_value()) {
      p2p_tensor_dictionary_plan_float_ =
          quantise_static_p2p_signed_tensor_dictionary_plan(
              *p2p_tensor_dictionary_plan_);
    }
  } else {
    // The SoA rows are the FP32 executor of every CPU plan except a
    // position-based, a CanonicalAos (which reads its canonical rows for
    // potential as well) and a dictionary plan (whose point potential rows
    // were built with the dictionary).
    if (p2p_execution_packing_ != P2PExecutionPacking::PointGeometry &&
        p2p_execution_packing_ != P2PExecutionPacking::CanonicalAos &&
        !dictionary_plan && !p2p_float_prebuilt_ &&
        p2p_compact_plan_float_.row_offsets.empty()) {
      p2p_compact_plan_float_ =
          build_static_p2p_compact_plan(p2p_operator_float_);
    }
    if (p2p_tensor_dictionary_plan_.has_value()) {
      auto dictionary = quantise_static_p2p_signed_tensor_dictionary_plan(
          *p2p_tensor_dictionary_plan_);
      p2p_tensor_dictionary_plan_float_ = std::move(dictionary);
    }
    if (dictionary_plan) {
      p2p_execution_packing_ = P2PExecutionPacking::TensorDictionary;
    }
  }
  // BSR(3) is explicit-only and CUDA-only: build it when the plan executes
  // it, whatever `cuda_p2p_bsr_max_bytes` says (docs/backends.md).
  if (bsr_identity_compatible && uses_cuda_plan && cuda_policy_ &&
      cuda_policy_->policy.p2p_packing == cuda_policy::CudaP2PPacking::Bsr3) {
    const std::span<const int> bsr_identities =
        fixed_target_source_indices_.has_value()
            ? std::span<const int>(fixed_sorted_self_indices_)
            : std::span<const int>{};
    p2p_bsr_plan_float_ = build_static_p2p_bsr_plan(
        p2p_operator_float_, bsr_identities);
  }
  p2p_packing_clock.record(static_plan_statistics_.backend_packing);

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
      static_plan_statistics_.p2p_interactions * 6 * sizeof(float);
  static_plan_statistics_.p2p_index_bytes =
      p2p_operator_float_.row_offsets.size() * sizeof(int) +
      p2p_operator_float_.blocks.size() * 2 * sizeof(int) +
      p2p_compact_plan_float_.source_indices.size() * sizeof(int) +
      p2p_compact_plan_float_.skip_for_identity.size() *
          sizeof(unsigned char);
  // The SoA-row footprint of the pairs, whether or not the plan keeps those
  // rows, so the dictionary and stored-tensor memory stays comparable.
  static_plan_statistics_.p2p_canonical_total_bytes =
      baseline_row_bytes(static_plan_statistics_.p2p_interactions,
                         topology_->sorted_target_positions.size(),
                         sizeof(float));
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
