// SPDX-License-Identifier: Apache-2.0
//
// Chunked near-field construction must be invisible in the results.
//
// The canonical operator, and every packing derived from it, is built one
// chunk of target leaves at a time (src/fmm/p2p_construction.hpp). A pair
// tensor is a pure function of its inputs and the rows are ordered by target,
// so the chunked result must equal the one-chunk result bit for bit -- for
// every geometry, including the tetrahedron self-systems whose reciprocal
// pairs share one tensor across chunk boundaries, and for periodic images.
// Each case is built with the default budget (one chunk at these sizes) and
// with a budget of one pair (one target leaf per chunk) and compared bitwise.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include <unistd.h>

#include "cdfmm/uniform_fmm.hpp"
#include "fmm/p2p_construction.hpp"
#include "plan/p2p/signed_dictionary_builder.hpp"

using namespace cdfmm;

namespace {

class ChunkBudget {
public:
  explicit ChunkBudget(const std::size_t pairs) {
    detail::p2p_construction::set_chunk_pair_budget_for_testing(pairs);
  }
  ~ChunkBudget() { detail::p2p_construction::set_chunk_pair_budget_for_testing(0); }
  ChunkBudget(const ChunkBudget &) = delete;
  ChunkBudget &operator=(const ChunkBudget &) = delete;
};

[[nodiscard]] std::vector<Vec3> jittered_lattice(const int side) {
  std::vector<Vec3> positions;
  for (int k = 0; k < side; ++k) {
    for (int j = 0; j < side; ++j) {
      for (int i = 0; i < side; ++i) {
        const double value = static_cast<double>(positions.size());
        positions.push_back({(i + 0.5) / side - 0.5 + 0.01 * std::sin(value),
                             (j + 0.5) / side - 0.5 + 0.01 * std::cos(1.3 * value),
                             (k + 0.5) / side - 0.5 + 0.01 * std::sin(0.7 * value)});
      }
    }
  }
  return positions;
}

[[nodiscard]] Tetrahedron small_tetrahedron(const double scale, const int variant) {
  Tetrahedron tetrahedron;
  const double stretch = 1.0 + 0.1 * (variant % 3);
  tetrahedron.vertices = {{{-0.25 * scale, -0.25 * scale, -0.25 * scale},
                           {0.75 * scale * stretch, -0.25 * scale, -0.25 * scale},
                           {-0.25 * scale, 0.75 * scale, -0.25 * scale},
                           {-0.25 * scale, -0.25 * scale, 0.75 * scale}}};
  return tetrahedron;
}

[[nodiscard]] bool same_bits(const double left, const double right) {
  return std::bit_cast<std::uint64_t>(left) == std::bit_cast<std::uint64_t>(right);
}

void require_identical(const std::vector<PotentialField> &left,
                       const std::vector<PotentialField> &right) {
  REQUIRE(left.size() == right.size());
  for (std::size_t index = 0; index < left.size(); ++index) {
    REQUIRE(same_bits(left[index].H.x, right[index].H.x));
    REQUIRE(same_bits(left[index].H.y, right[index].H.y));
    REQUIRE(same_bits(left[index].H.z, right[index].H.z));
    REQUIRE(same_bits(left[index].phi, right[index].phi));
  }
}

struct Geometry {
  std::string name;
  SourceGeometry source{SourceGeometry::PointDipole};
  TargetGeometry target{TargetGeometry::Point};
  bool per_body_records{false};
};

void configure(UniformFmmOptions &options, const Geometry &geometry,
               const std::size_t count) {
  const double spacing = 1.0 / 5.0;
  options.source_geometry = geometry.source;
  options.target_geometry = geometry.target;
  const std::size_t records = geometry.per_body_records ? count : 1;
  for (std::size_t index = 0; index < records; ++index) {
    const double factor = geometry.per_body_records ? 0.7 + 0.03 * (index % 7) : 0.9;
    if (geometry.source == SourceGeometry::RectangularPrism) {
      options.source_sizes.push_back(
          RectangularPrism{factor * spacing, 0.8 * factor * spacing, 0.9 * factor * spacing});
    }
    if (geometry.target == TargetGeometry::RectangularPrism) {
      options.target_sizes.push_back(
          RectangularPrism{factor * spacing, 0.8 * factor * spacing, 0.9 * factor * spacing});
    }
    if (geometry.source == SourceGeometry::Tetrahedron) {
      options.source_tetrahedra.push_back(
          small_tetrahedron(factor * spacing, static_cast<int>(index)));
    }
    if (geometry.target == TargetGeometry::Tetrahedron) {
      options.target_tetrahedra.push_back(
          small_tetrahedron(factor * spacing, static_cast<int>(index)));
    }
  }
  options.far_field_source_model = SourceModel::PointDipole;
  options.far_field_target_model = TargetModel::Point;
}

} // namespace

TEST_CASE("chunked near-field construction is bitwise the one-chunk build",
          "[uniform_fmm][p2p][chunked]") {
  const std::vector<Vec3> positions = jittered_lattice(5);
  std::vector<Vec3> moments(positions.size());
  for (std::size_t index = 0; index < moments.size(); ++index) {
    const double value = static_cast<double>(index);
    moments[index] = {std::cos(value), std::sin(0.4 * value), std::cos(2.1 * value)};
  }
  std::vector<int> identities(positions.size());
  std::iota(identities.begin(), identities.end(), 0);

  const std::vector<Geometry> geometries{
      {"point", SourceGeometry::PointDipole, TargetGeometry::Point, false},
      {"prism", SourceGeometry::RectangularPrism, TargetGeometry::RectangularPrism, false},
      {"prism per body", SourceGeometry::RectangularPrism,
       TargetGeometry::RectangularPrism, true},
      {"prism to point", SourceGeometry::RectangularPrism, TargetGeometry::Point, false},
      {"tetrahedron self-system", SourceGeometry::Tetrahedron, TargetGeometry::Tetrahedron,
       false},
      {"tetrahedron per body", SourceGeometry::Tetrahedron, TargetGeometry::Tetrahedron,
       true},
  };
  // Every FP32 representation is quantised from the same FP64 chunks, so FP32
  // needs one packing per path (rows and dictionary); periodic images matter
  // for the point and the reciprocal tetrahedron system.
  for (const Geometry &geometry : geometries) {
    const bool periodic_relevant = geometry.name == "point" ||
                                   geometry.name == "tetrahedron self-system";
    for (const bool periodic : {false, true}) {
      if (periodic && !periodic_relevant) {
        continue;
      }
      for (const StaticPrecision precision :
           {StaticPrecision::Float64, StaticPrecision::Float32}) {
        for (const P2PExecutionPacking packing :
             {P2PExecutionPacking::CanonicalAos, P2PExecutionPacking::ParticleRowSoa,
              P2PExecutionPacking::TensorDictionary}) {
          if (precision == StaticPrecision::Float32 &&
              packing == P2PExecutionPacking::CanonicalAos) {
            continue;
          }
          UniformFmmOptions options;
          options.backend = ExecutionBackend::CpuStatic;
          options.precision = precision;
          options.expansion_order = 3;
          options.tree.max_level = 2;
          options.enable_cache = false;
          options.p2p_packing = packing;
          options.fixed_target_source_indices = identities;
          if (periodic) {
            options.periodic.enabled = true;
            options.periodic.centre = Vec3{};
            options.periodic.lengths = Vec3{1.2, 1.2, 1.2};
          }
          configure(options, geometry, positions.size());
          const OutputFlags output = geometry.source == SourceGeometry::PointDipole
                                         ? OutputFlags::Field | OutputFlags::Potential
                                         : OutputFlags::Field;
          INFO(geometry.name << " periodic=" << periodic
                             << " fp32=" << (precision == StaticPrecision::Float32)
                             << " packing=" << static_cast<int>(packing));
          std::vector<PotentialField> one_chunk;
          std::size_t pairs = 0;
          {
            UniformFmm fmm(positions, positions, options);
            REQUIRE(fmm.p2p_execution_packing() == packing);
            one_chunk = fmm.evaluate(moments, output);
            pairs = fmm.static_plan_statistics().p2p_interactions;
          }
          ChunkBudget budget(1);
          UniformFmm chunked(positions, positions, options);
          REQUIRE(chunked.p2p_execution_packing() == packing);
          REQUIRE(chunked.static_plan_statistics().p2p_interactions == pairs);
          require_identical(chunked.evaluate(moments, output), one_chunk);
        }
      }
    }
  }
}

namespace {

class TemporaryCacheDirectory {
public:
  TemporaryCacheDirectory() {
    path_ = std::filesystem::temp_directory_path() /
            ("cdfmm-chunked-" + std::to_string(::getpid()) + "-" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    REQUIRE(::setenv("CDFMM_CACHE_DIR", path_.c_str(), 1) == 0);
    REQUIRE(::unsetenv("CDFMM_DISABLE_CACHE") == 0);
  }
  ~TemporaryCacheDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
    ::unsetenv("CDFMM_CACHE_DIR");
  }
  TemporaryCacheDirectory(const TemporaryCacheDirectory &) = delete;
  TemporaryCacheDirectory &operator=(const TemporaryCacheDirectory &) = delete;

private:
  std::filesystem::path path_{};
};

// A warm plan loads its stored near field: it builds no pair list and no
// tensor, writes nothing, and reports exactly the cold plan's near field.
void require_same_near_field_statistics(const StaticPlanStatistics &cold,
                                        const StaticPlanStatistics &warm) {
  REQUIRE(warm.geometry_cache_hit);
  REQUIRE(warm.cache_bytes_written == 0);
  REQUIRE(warm.p2p_interaction_setup.calls == 0);
  REQUIRE(warm.p2p_canonical_operator.calls == 0);
  REQUIRE(warm.p2p_interactions == cold.p2p_interactions);
  REQUIRE(warm.p2p_value_bytes == cold.p2p_value_bytes);
  REQUIRE(warm.near_field_operator_bytes == cold.near_field_operator_bytes);
  REQUIRE(warm.p2p_unique_tensors == cold.p2p_unique_tensors);
  REQUIRE(warm.p2p_dictionary_tokens == cold.p2p_dictionary_tokens);
  REQUIRE(warm.p2p_dictionary_token_width_bytes ==
          cold.p2p_dictionary_token_width_bytes);
  REQUIRE(warm.p2p_dictionary_total_bytes == cold.p2p_dictionary_total_bytes);
}

} // namespace

TEST_CASE("near-field representations agree with and without the cache",
          "[uniform_fmm][p2p][chunked][cache]") {
  // A plan built without the cache derives only the representations it keeps,
  // straight from the chunks; one built with the cache also streams its
  // stored near field (the canonical rows, or the dictionary) into the file;
  // a warm plan loads that and builds no pair and no tensor. All three must
  // evaluate bit for bit alike and the cold and warm plans must hold the same
  // state.
  const std::vector<Vec3> positions = jittered_lattice(5);
  std::vector<Vec3> moments(positions.size());
  for (std::size_t index = 0; index < moments.size(); ++index) {
    const double value = static_cast<double>(index);
    moments[index] = {std::sin(1.1 * value), std::cos(value), std::sin(0.3 * value)};
  }
  std::vector<int> identities(positions.size());
  std::iota(identities.begin(), identities.end(), 0);
  for (const Geometry &geometry :
       {Geometry{"point", SourceGeometry::PointDipole, TargetGeometry::Point, false},
        Geometry{"prism", SourceGeometry::RectangularPrism,
                 TargetGeometry::RectangularPrism, false}}) {
    for (const StaticPrecision precision :
         {StaticPrecision::Float64, StaticPrecision::Float32}) {
      for (const P2PExecutionPacking packing :
           {P2PExecutionPacking::CanonicalAos, P2PExecutionPacking::ParticleRowSoa,
            P2PExecutionPacking::TensorDictionary}) {
        UniformFmmOptions options;
        options.backend = ExecutionBackend::CpuStatic;
        options.precision = precision;
        options.expansion_order = 3;
        options.tree.max_level = 2;
        options.p2p_packing = packing;
        options.fixed_target_source_indices = identities;
        options.timing_level = TimingLevel::Detailed;
        configure(options, geometry, positions.size());
        const OutputFlags output = geometry.source == SourceGeometry::PointDipole
                                       ? OutputFlags::Field | OutputFlags::Potential
                                       : OutputFlags::Field;
        INFO(geometry.name << " fp32=" << (precision == StaticPrecision::Float32)
                           << " packing=" << static_cast<int>(packing));
        options.enable_cache = false;
        const auto uncached =
            UniformFmm(positions, positions, options).evaluate(moments, output);
        TemporaryCacheDirectory cache;
        options.enable_cache = true;
        UniformFmm cold(positions, positions, options);
        REQUIRE_FALSE(cold.static_plan_statistics().geometry_cache_hit);
        require_identical(cold.evaluate(moments, output), uncached);
        UniformFmm warm(positions, positions, options);
        REQUIRE(warm.static_plan_statistics().geometry_cache_hit);
        REQUIRE(warm.p2p_execution_packing() == packing);
        require_identical(warm.evaluate(moments, output), uncached);
        // Cold and warm plans hold the same state and report the same memory.
        const StaticPlanStatistics &cold_statistics = cold.static_plan_statistics();
        const StaticPlanStatistics &warm_statistics = warm.static_plan_statistics();
        REQUIRE(warm_statistics.p2p_interactions == cold_statistics.p2p_interactions);
        REQUIRE(warm_statistics.near_field_operator_bytes ==
                cold_statistics.near_field_operator_bytes);
        REQUIRE(warm_statistics.operator_bytes == cold_statistics.operator_bytes);
        REQUIRE(warm_statistics.total_bytes() == cold_statistics.total_bytes());
        require_same_near_field_statistics(cold_statistics, warm_statistics);
      }
    }
  }
}

TEST_CASE("CUDA dictionary plans load their near field from the cache",
          "[uniform_fmm][p2p][chunked][cache][cuda]") {
  // The dictionary is the stored near field of these plans: a warm plan
  // must load it (or, where the RegularGrid hint's dictionary did not
  // compress, the canonical records it fell back to) and build no tensor.
  // CUDA accumulation order is not fixed, so results compare to a tolerance.
  if (!cuda_m2l_p2p_available()) {
    SKIP("CUDA M2L/P2P backend unavailable");
  }
  struct Case {
    std::string name;
    ExecutionBackend backend;
    StaticPrecision precision;
    SpatialLayout layout;
    bool jittered;
    bool dictionary_kept;
  };
  const Case cases[] = {
      {"partial fp64 explicit", ExecutionBackend::CudaM2LP2P,
       StaticPrecision::Float64, SpatialLayout::General, false, true},
      {"partial fp32 explicit", ExecutionBackend::CudaM2LP2P,
       StaticPrecision::Float32, SpatialLayout::General, false, true},
      {"full fp32 explicit", ExecutionBackend::CudaFull,
       StaticPrecision::Float32, SpatialLayout::General, false, true},
      {"partial fp64 hint kept", ExecutionBackend::CudaM2LP2P,
       StaticPrecision::Float64, SpatialLayout::RegularGrid, false, true},
      {"partial fp64 hint fallback", ExecutionBackend::CudaM2LP2P,
       StaticPrecision::Float64, SpatialLayout::RegularGrid, true, false},
  };
  for (const Case &test : cases) {
    INFO(test.name);
    // 12^3 points at depth 2 hold about 1.2 M list-1 pairs: on the jittered
    // lattice nearly every tensor differs, so the dictionary needs four-byte
    // tokens and the hint falls back.
    const int side = 12;
    std::vector<Vec3> positions;
    for (int k = 0; k < side; ++k) {
      for (int j = 0; j < side; ++j) {
        for (int i = 0; i < side; ++i) {
          const double value = static_cast<double>(positions.size());
          const double jitter = test.jittered ? 0.01 : 0.0;
          positions.push_back(
              {(i + 0.5) / side - 0.5 + jitter * std::sin(value),
               (j + 0.5) / side - 0.5 + jitter * std::cos(1.3 * value),
               (k + 0.5) / side - 0.5 + jitter * std::sin(0.7 * value)});
        }
      }
    }
    std::vector<Vec3> moments(positions.size());
    for (std::size_t index = 0; index < moments.size(); ++index) {
      const double value = static_cast<double>(index);
      moments[index] = {std::sin(1.1 * value), std::cos(value),
                        std::sin(0.3 * value)};
    }
    std::vector<int> identities(positions.size());
    std::iota(identities.begin(), identities.end(), 0);
    UniformFmmOptions options;
    options.backend = test.backend;
    options.precision = test.precision;
    options.expansion_order = 3;
    options.tree.max_level = 2;
    options.fixed_target_source_indices = identities;
    options.timing_level = TimingLevel::Detailed;
    options.spatial_layout = test.layout;
    if (test.layout == SpatialLayout::General) {
      options.p2p_packing = P2PExecutionPacking::TensorDictionary;
    }
    TemporaryCacheDirectory cache;
    options.enable_cache = true;
    UniformFmm cold(positions, positions, options);
    REQUIRE_FALSE(cold.static_plan_statistics().geometry_cache_hit);
    const bool kept =
        cold.static_plan_statistics().p2p_dictionary_token_width_bytes > 0;
    REQUIRE(kept == test.dictionary_kept);
    const auto cold_result = cold.evaluate(moments, OutputFlags::Field);
    UniformFmm warm(positions, positions, options);
    REQUIRE(warm.static_plan_statistics().geometry_cache_hit);
    REQUIRE(warm.p2p_execution_packing() == cold.p2p_execution_packing());
    const auto warm_result = warm.evaluate(moments, OutputFlags::Field);
    const double tolerance =
        test.precision == StaticPrecision::Float32 ? 1e-5 : 1e-12;
    double scale = 0.0;
    double error = 0.0;
    for (std::size_t index = 0; index < cold_result.size(); ++index) {
      const Vec3 &left = cold_result[index].H;
      const Vec3 &right = warm_result[index].H;
      scale = std::max({scale, std::abs(left.x), std::abs(left.y), std::abs(left.z)});
      error = std::max({error, std::abs(left.x - right.x),
                        std::abs(left.y - right.y), std::abs(left.z - right.z)});
    }
    REQUIRE(error <= tolerance * scale);
    require_same_near_field_statistics(cold.static_plan_statistics(),
                                       warm.static_plan_statistics());
    REQUIRE(warm.static_plan_statistics().total_bytes() ==
            cold.static_plan_statistics().total_bytes());
  }
}

namespace {

constexpr int kLeafTargets = 64;
constexpr int kLeafSources = 64;
constexpr std::size_t kLeafPairs =
    static_cast<std::size_t>(kLeafTargets) * kLeafSources;

// The xx value of global pair `pair`; the other components are zero. `pair`
// counts target-major within each leaf, so `distinct` sets the number of
// non-zero variants the dictionary must find.
double synthetic_value(const std::size_t pair, const std::size_t distinct) {
  return 1.0 + static_cast<double>(pair % distinct);
}

// Target leaves [first_leaf, first_leaf + leaf_count), each one dense block
// against the same source leaf, in the layout `build_static_p2p_leaf_plan`
// produces for one chunk.
StaticP2PLeafPlan synthetic_leaves(const int first_leaf, const int leaf_count,
                                   const int total_leaves,
                                   const std::size_t distinct) {
  StaticP2PLeafPlan leaf;
  leaf.source_count = kLeafSources;
  leaf.target_count = total_leaves * kLeafTargets;
  leaf.leaf_row_offsets.push_back(0);
  for (int local_leaf = 0; local_leaf < leaf_count; ++local_leaf) {
    const int target_leaf = first_leaf + local_leaf;
    leaf.target_begins.push_back(target_leaf * kLeafTargets);
    leaf.target_counts.push_back(kLeafTargets);
    StaticP2PLeafBlock block;
    block.source_begin = 0;
    block.source_count = kLeafSources;
    block.tensor_offset = static_cast<std::size_t>(local_leaf) * kLeafPairs;
    leaf.blocks.push_back(block);
    leaf.leaf_row_offsets.push_back(local_leaf + 1);
    for (std::size_t local_pair = 0; local_pair < kLeafPairs; ++local_pair) {
      const std::size_t pair =
          static_cast<std::size_t>(target_leaf) * kLeafPairs + local_pair;
      leaf.tensors[0].push_back(synthetic_value(pair, distinct));
      for (std::size_t component = 1; component < 6; ++component) {
        leaf.tensors[component].push_back(0.0);
      }
    }
  }
  return leaf;
}

std::uint32_t token_at(const StaticP2PSignedTensorDictionaryPlan &plan,
                       const std::size_t index) {
  switch (plan.token_width_bytes) {
  case 1:
    return plan.tokens8[index];
  case 2:
    return plan.tokens16[index];
  default:
    return plan.tokens32[index];
  }
}

} // namespace

TEST_CASE("signed dictionary construction widens its tokens only on demand",
          "[p2p][dictionary][memory]") {
  // The builder holds two-byte tokens while every variant id fits and widens
  // once when one does not. Each case checks that the stored width follows
  // the variant count, that every token decodes to the input tensor, and that
  // appending the leaves in two chunks gives the one-append plan exactly --
  // including when the widening happens inside the second chunk.
  struct Case {
    std::size_t distinct;
    int leaves;
    std::uint8_t width;
  };
  // Variant counts include the zero variant: 1001, 65536 (ids still fit in
  // two bytes while building, stored at four) and 70001 (widens mid-build).
  const Case cases[] = {{1000, 2, 2}, {65535, 17, 4}, {70000, 18, 4}};
  for (const Case &test : cases) {
    INFO("distinct=" << test.distinct);
    // Ten leaves hold 40960 pairs, so the 70000-variant case widens inside
    // the second append.
    const int split = std::min(10, test.leaves / 2);
    detail::SignedTensorDictionaryBuilder whole(
        kLeafSources, test.leaves * kLeafTargets, {}, 32);
    whole.append(synthetic_leaves(0, test.leaves, test.leaves, test.distinct));
    const StaticP2PSignedTensorDictionaryPlan one = whole.finish();

    detail::SignedTensorDictionaryBuilder chunked(
        kLeafSources, test.leaves * kLeafTargets, {}, 32);
    chunked.append(synthetic_leaves(0, split, test.leaves, test.distinct));
    chunked.append(synthetic_leaves(split, test.leaves - split, test.leaves,
                                    test.distinct));
    const StaticP2PSignedTensorDictionaryPlan two = chunked.finish();

    REQUIRE(one.variant_count() == test.distinct + 1);
    REQUIRE(one.token_width_bytes == test.width);
    REQUIRE(one.token_count() ==
            static_cast<std::size_t>(test.leaves) * kLeafPairs);

    // Tokens are source-major within a block.
    for (std::size_t target_leaf = 0; target_leaf < one.blocks.size();
         ++target_leaf) {
      const StaticP2PLeafBlock &block = one.blocks[target_leaf];
      for (int local_source = 0; local_source < kLeafSources; ++local_source) {
        for (int local_target = 0; local_target < kLeafTargets; ++local_target) {
          const std::size_t pair = target_leaf * kLeafPairs +
                                   static_cast<std::size_t>(local_target) *
                                       kLeafSources +
                                   static_cast<std::size_t>(local_source);
          const std::uint32_t token = token_at(
              one, block.tensor_offset +
                       static_cast<std::size_t>(local_source) * kLeafTargets +
                       static_cast<std::size_t>(local_target));
          REQUIRE(one.tensors[0][token] ==
                  synthetic_value(pair, test.distinct));
        }
      }
    }

    REQUIRE(two.token_width_bytes == one.token_width_bytes);
    REQUIRE(two.tokens8 == one.tokens8);
    REQUIRE(two.tokens16 == one.tokens16);
    REQUIRE(two.tokens32 == one.tokens32);
    REQUIRE(two.tensors == one.tensors);
    REQUIRE(two.zero_variant_id == one.zero_variant_id);
    REQUIRE(two.leaf_row_offsets == one.leaf_row_offsets);
    REQUIRE(two.tile_leaf_indices == one.tile_leaf_indices);
    REQUIRE(two.tile_target_offsets == one.tile_target_offsets);
    REQUIRE(two.blocks.size() == one.blocks.size());
    for (std::size_t block = 0; block < one.blocks.size(); ++block) {
      REQUIRE(two.blocks[block].tensor_offset == one.blocks[block].tensor_offset);
      REQUIRE(two.blocks[block].source_begin == one.blocks[block].source_begin);
    }
  }
}
