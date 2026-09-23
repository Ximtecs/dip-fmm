// SPDX-License-Identifier: Apache-2.0
#pragma once

// Chunked construction of the canonical near field.
//
// The canonical operator holds one record per list-1 pair, and every derived
// packing (SoA rows, the signed dictionary, CUDA leaf blocks, FP32 copies) is
// a per-row or per-leaf transformation of it. Building it in one call needs
// the full pair list, the builder's sorted copy and its classification maps
// at the same time as the output: about 100 bytes per pair of transient state
// on top of the 88-byte records. Here the pairs are built one chunk of
// consecutive target leaves at a time and each chunk is handed to whichever
// representation the plan keeps, so the transient state is bounded by the
// chunk and a representation the plan does not keep is never materialised.
//
// Chunking never changes a value. A pair tensor is a pure function of its
// displacement bits, the two body records and the identity marker
// (src/operators/exact_operator_reuse.hpp), and the builder orders rows by
// target, so a chunk's rows are bitwise the monolithic build's rows for the
// same targets. The one cross-row dependency -- tetrahedron self-systems
// share each tensor with its reciprocal pair, computed by the lower-indexed
// partner -- is preserved by adding to each chunk the reverse pairs it would
// copy from and dropping their rows afterwards.

#include <cstddef>
#include <span>
#include <vector>

#include "cdfmm/core/timing.hpp"
#include "cdfmm/geometry/models.hpp"
#include "cdfmm/geometry/primitives/rectangular_prism.hpp"
#include "cdfmm/geometry/primitives/tetrahedron.hpp"
#include "cdfmm/math/vec3.hpp"
#include "cdfmm/plan/p2p/canonical.hpp"
#include "cdfmm/plan/p2p/compact.hpp"
#include "cdfmm/plan/p2p/leaf.hpp"
#include "cdfmm/tree/static_topology.hpp"

namespace cdfmm::detail::p2p_construction {

/// @brief Consecutive target leaves whose list-1 pairs are built together.
struct Chunk {
  std::size_t record_begin{0};  ///< First `p2p_leaf_records` entry.
  std::size_t record_end{0};    ///< One past the last record.
  int target_begin{0};          ///< First sorted target of the chunk.
  int target_end{0};            ///< One past the last sorted target.
  std::size_t pairs{0};         ///< List-1 pairs of the chunk.
};

/// @brief The plan's dense leaf pairs in canonical order, one per list-1
///        record, each tagged with its periodic image ordinal. The dictionary
///        and leaf packings consume these alongside the canonical rows.
[[nodiscard]] std::vector<StaticP2PLeafPair>
leaf_pairs_from_topology(const StaticFmmTopology &topology);

/// @brief Pairs per chunk: 2^22, about 1.2 GB of transient chunk state.
[[nodiscard]] std::size_t chunk_pair_budget() noexcept;

/// @brief Overrides the budget so tests can force many chunks; 0 restores it.
void set_chunk_pair_budget_for_testing(std::size_t pairs) noexcept;

/// @brief Splits the list-1 records into chunks at target-leaf boundaries.
[[nodiscard]] std::vector<Chunk> plan_chunks(const StaticFmmTopology &topology,
                                             std::size_t pair_budget);

/// @brief The geometry and model inputs of the canonical builder.
struct CanonicalInputs {
  std::span<const Vec3> targets{};
  std::span<const Vec3> sources{};
  SourceGeometry source_geometry{SourceGeometry::PointDipole};
  std::span<const RectangularPrism> source_sizes{};
  std::span<const Tetrahedron> source_tetrahedra{};
  TargetGeometry target_geometry{TargetGeometry::Point};
  std::span<const RectangularPrism> target_sizes{};
  std::span<const Tetrahedron> target_tetrahedra{};
  SourceModel source_model{SourceModel::ExactGeometry};
  TargetModel target_model{TargetModel::ExactGeometry};
  bool periodic{false};
};

/// @brief Builds chunk operators for one plan.
///
/// Holds the reciprocal-record lookup, built once, when the plan is a
/// tetrahedron self-system; otherwise it is empty.
class ChunkBuilder {
public:
  ChunkBuilder(const StaticFmmTopology &topology, const CanonicalInputs &inputs);

  /// The chunk's canonical operator: `row_offsets` spans every target, only
  /// the chunk's rows are populated, and they are bitwise the monolithic
  /// build's rows. Record expansion and tensor time are added to the timers.
  [[nodiscard]] StaticP2POperator build(const Chunk &chunk,
                                        PhaseTiming &interaction_setup,
                                        PhaseTiming &tensor_build,
                                        bool timed) const;

  /// The chunk's leaf pairs, sliced from the plan's leaf pairs (derived on
  /// first use: only the leaf-block packings need them).
  [[nodiscard]] std::span<const StaticP2PLeafPair>
  leaf_pairs(const Chunk &chunk) const;

private:
  const StaticFmmTopology &topology_;
  CanonicalInputs inputs_;
  bool reciprocal_{false};
  mutable std::vector<StaticP2PLeafPair> leaf_pairs_{};
  mutable bool leaf_pairs_ready_{false};
  std::vector<std::size_t> reverse_record_{};  ///< Reverse record or npos.
};

/// @brief Appends a chunk's rows to an operator accumulated in target order.
template <typename Operator>
void append_rows(Operator &total, const Operator &chunk, const Chunk &range);

/// @brief Appends a chunk's rows to a compact plan accumulated in target order.
///        `with_tensors = false` keeps only what potential output reads (source
///        indices, identity markers, potential rows) and no tensor planes.
template <typename Compact>
void append_compact_rows(Compact &total, const Compact &chunk, const Chunk &range,
                         bool with_tensors = true);

/// @brief Closes the row offsets of an accumulated operator or compact plan.
template <typename Rows> void finish_rows(Rows &total);

/// @brief Starts an accumulator with the plan's dimensions and no rows yet,
///        reserving room for `pairs` so appending never reallocates.
template <typename Rows>
void start_rows(Rows &total, int source_count, int target_count,
                std::size_t pairs = 0, bool with_tensors = true);

/// @brief The canonical operator's row offsets (`target_count + 1` entries),
///        known from the list-1 records before any pair is built.
[[nodiscard]] std::vector<int>
canonical_row_offsets(const StaticFmmTopology &topology, std::size_t target_count);

/// @brief Total list-1 pairs of a chunk list.
[[nodiscard]] std::size_t total_pairs(const std::vector<Chunk> &chunks) noexcept;

/// @brief Appends a chunk's leaf blocks to an accumulated leaf plan.
template <typename Leaf> void append_leaf_blocks(Leaf &total, Leaf &&chunk);

/// @brief Recomputes the occupancy statistics of an accumulated leaf plan.
template <typename Leaf> void finish_leaf_plan(Leaf &total);

} // namespace cdfmm::detail::p2p_construction
