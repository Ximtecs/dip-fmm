// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cdfmm/geometry/models.hpp"
#include "cdfmm/geometry/primitives/rectangular_prism.hpp"
#include "cdfmm/geometry/primitives/tetrahedron.hpp"
#include "cdfmm/periodic.hpp"
#include "cdfmm/plan/static_plan.hpp"
#include "cdfmm/precision.hpp"
#include "cdfmm/timings.hpp"
#include "cdfmm/tree/static_topology.hpp"
#include "cdfmm/tree/uniform_tree.hpp"

#include <sys/mman.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Implementation-only interface shared by the cache translation units. The
// cache subsystem persists solver data defined elsewhere: nothing here
// constructs operators, plans, topology, or execution packings, and nothing
// here depends on `UniformFmm` itself. Every cache entry point below takes
// explicit identity/payload records naming the specific solver/plan state it
// persists; `src/fmm/execution_setup.cpp` and `src/fmm/plan_preparation.cpp`
// assemble those records from `UniformFmm`'s private state and remain the
// sole callers, preserving plan preparation as the cache-vs-build owner.
//
// Ownership inside this subsystem:
//   io.cpp        cache root/environment policy and the validated file
//                 container (magic, header fields, checksum, atomic write)
//   format.cpp    field-wise records for the solver types held in a payload
//   keys.cpp      cache identity: digests, canonicalisation, key strings
//   universal.cpp depth-independent translation bank and periodic root payload
//   geometry.cpp  geometry-dependent plan payload
namespace cdfmm::detail::cache {

// WARNING(cdfmm): every constant below is part of the on-disk contract.
// Changing one invalidates existing cache files and requires an explicit
// schema review, not an incidental refactor.
// Version 4 adds independent near/far source/target model selectors and
// representative-relative tetrahedron records to the geometry identity.
inline constexpr std::uint32_t kCacheSchemaVersion = 4;
// Version 2 invalidates plans built before the analytical tetrahedron pair
// kernel and its rank-deficient reduction fixes.
inline constexpr std::uint32_t kOperatorVersion = 2;
inline constexpr std::uint32_t kEndianMarker = 0x01020304U;
// Fast 64-bit payload checksum.
inline constexpr std::uint32_t kChecksumAlgorithm = 2U;

enum class CacheKind : std::uint32_t { Universal = 1, Periodic = 2, Plan = 3 };

struct CacheDescriptor {
  CacheKind kind{};
  ExpansionBasis basis{};
  int order{0};
  StaticPrecision precision{};
  int depth{-1};
  std::string key{};
  std::string geometry_hash{};
};

// Read-only mapping of a complete cache file plus the selected payload range.
class CachePayload {
public:
  CachePayload() = default;

  CachePayload(void* mapping, const std::size_t mapping_size)
      : mapping_(mapping), mapping_size_(mapping_size), payload_size_(mapping_size) {}

  CachePayload(const CachePayload&) = delete;
  CachePayload& operator=(const CachePayload&) = delete;

  CachePayload(CachePayload&& other) noexcept { move_from(other); }

  CachePayload& operator=(CachePayload&& other) noexcept {
    if (this != &other) {
      reset();
      move_from(other);
    }
    return *this;
  }

  ~CachePayload() { reset(); }

  [[nodiscard]] std::span<const unsigned char> full_bytes() const noexcept {
    if (mapping_ == MAP_FAILED || mapping_ == nullptr) {
      return {};
    }
    return {static_cast<const unsigned char*>(mapping_), mapping_size_};
  }

  [[nodiscard]] std::span<const unsigned char> bytes() const noexcept {
    const auto full = full_bytes();
    return full.subspan(payload_offset_, payload_size_);
  }

  void select_payload(const std::size_t offset, const std::size_t size) {
    if (offset > mapping_size_ || size > mapping_size_ - offset) {
      throw std::runtime_error("cache payload range is invalid");
    }
    payload_offset_ = offset;
    payload_size_ = size;
  }

private:
  void reset() noexcept {
    if (mapping_ != MAP_FAILED && mapping_ != nullptr && mapping_size_ != 0) {
      (void)::munmap(mapping_, mapping_size_);
    }
    mapping_ = MAP_FAILED;
    mapping_size_ = 0;
    payload_offset_ = 0;
    payload_size_ = 0;
  }

  void move_from(CachePayload& other) noexcept {
    mapping_ = other.mapping_;
    mapping_size_ = other.mapping_size_;
    payload_offset_ = other.payload_offset_;
    payload_size_ = other.payload_size_;
    other.mapping_ = MAP_FAILED;
    other.mapping_size_ = 0;
    other.payload_offset_ = 0;
    other.payload_size_ = 0;
  }

  void* mapping_{MAP_FAILED};
  std::size_t mapping_size_{0};
  std::size_t payload_offset_{0};
  std::size_t payload_size_{0};
};

class Writer {
public:
  void reserve(const std::size_t bytes) { bytes_.reserve(bytes); }

  unsigned char* append_uninitialized(const std::size_t size) {
    const std::size_t old_size = bytes_.size();
    if (size > std::numeric_limits<std::size_t>::max() - old_size) {
      throw std::runtime_error("cache payload size overflow");
    }
    bytes_.resize(old_size + size);
    return bytes_.data() + old_size;
  }

  void append_raw(const void* data, const std::size_t size) {
    if (size == 0) {
      return;
    }
    unsigned char* destination = append_uninitialized(size);
    std::memcpy(destination, data, size);
  }

  template <typename T> void scalar(const T value) {
    static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T>);
    append_raw(&value, sizeof(value));
  }

  template <typename T> void vector(const std::vector<T>& values) {
    static_assert(std::is_trivially_copyable_v<T>);
    span<T>(values);
  }

  template <typename T> void span(const std::span<const T> values) {
    static_assert(std::is_trivially_copyable_v<T>);
    scalar<std::uint64_t>(values.size());
    append_raw(values.data(), values.size_bytes());
  }

  void string(const std::string& value) {
    scalar<std::uint64_t>(value.size());
    append_raw(value.data(), value.size());
  }

  [[nodiscard]] const std::vector<unsigned char>& bytes() const {
    return bytes_;
  }

private:
  std::vector<unsigned char> bytes_{};
};

class Reader {
public:
  explicit Reader(const std::vector<unsigned char>& bytes)
      : bytes_(bytes.data(), bytes.size()) {}
  explicit Reader(const std::span<const unsigned char> bytes) : bytes_(bytes) {}
  explicit Reader(const CachePayload& payload) : bytes_(payload.bytes()) {}

  [[nodiscard]] const unsigned char* take_bytes(const std::size_t size) {
    if (size > bytes_.size() - offset_) {
      throw std::runtime_error("truncated cache payload");
    }
    const unsigned char* result = bytes_.data() + offset_;
    offset_ += size;
    return result;
  }

  template <typename T> T scalar() {
    static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T>);
    T value;
    std::memcpy(&value, take_bytes(sizeof(T)), sizeof(T));
    return value;
  }

  template <typename T> std::vector<T> vector() {
    static_assert(std::is_trivially_copyable_v<T>);
    const std::uint64_t count = scalar<std::uint64_t>();
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::runtime_error("invalid cache vector length");
    }
    std::vector<T> values(static_cast<std::size_t>(count));
    const std::size_t bytes = values.size() * sizeof(T);
    if (bytes != 0) {
      std::memcpy(values.data(), take_bytes(bytes), bytes);
    }
    return values;
  }

  template <typename T>
  bool equal_span(const std::span<const T> expected) {
    static_assert(std::is_trivially_copyable_v<T>);
    const std::uint64_t count = scalar<std::uint64_t>();
    if (count != expected.size()) {
      return false;
    }
    const std::size_t bytes = expected.size_bytes();
    const unsigned char* cached = take_bytes(bytes);
    return bytes == 0 || std::memcmp(cached, expected.data(), bytes) == 0;
  }

  [[nodiscard]] std::size_t offset() const noexcept { return offset_; }

  std::string string() {
    const std::uint64_t count = scalar<std::uint64_t>();
    if (count > std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error("invalid cache string length");
    }
    const std::size_t size = static_cast<std::size_t>(count);
    const unsigned char* raw = take_bytes(size);
    return std::string(reinterpret_cast<const char*>(raw), size);
  }

  void require_end() const {
    if (offset_ != bytes_.size()) {
      throw std::runtime_error("unexpected trailing cache data");
    }
  }

private:
  std::span<const unsigned char> bytes_;
  std::size_t offset_{0};
};

[[nodiscard]] inline std::size_t checked_bytes(
    const std::uint64_t count, const std::size_t element_size) {
  if (element_size != 0 &&
      count > std::numeric_limits<std::size_t>::max() / element_size) {
    throw std::runtime_error("cache array size overflow");
  }
  return static_cast<std::size_t>(count) * element_size;
}

template <typename T>
T load_unaligned(const unsigned char* source) noexcept {
  static_assert(std::is_trivially_copyable_v<T>);
  T value;
  std::memcpy(&value, source, sizeof(T));
  return value;
}

template <typename T>
void store_unaligned(unsigned char* destination, const T value) noexcept {
  static_assert(std::is_trivially_copyable_v<T>);
  std::memcpy(destination, &value, sizeof(T));
}

[[nodiscard]] inline std::size_t p2p_record_bytes(
    const StaticPrecision precision) {
  return 3 * sizeof(int) + 9 *
      (precision == StaticPrecision::Float32 ? sizeof(float) : sizeof(double));
}

// Cache location policy and the validated file container, in io.cpp.
std::filesystem::path cache_root();

bool environment_disables_cache();

std::filesystem::path cache_path(const std::string& root,
                                 const std::string& category,
                                 const std::string& key);

CachePayload read_cache(const std::filesystem::path& path,
                        const CacheDescriptor& expected,
                        std::size_t& bytes_read);

std::size_t write_cache(const std::filesystem::path& path,
                        const CacheDescriptor& descriptor,
                        const std::vector<unsigned char>& payload);

// Payload records for the persisted solver types, in format.cpp. These are the
// facilities shared by the universal and geometry payloads; the file container
// itself is owned by io.cpp.
void write_operator(Writer& writer, const StaticCoefficientOperator& value,
                    StaticPrecision precision);

StaticCoefficientOperator read_operator(Reader& reader,
                                        StaticPrecision precision);

FloatStaticCoefficientOperator read_operator_float(Reader& reader);

void write_values(Writer& writer, std::span<const double> values,
                  StaticPrecision precision);

void write_values(Writer& writer, const std::vector<double>& values,
                  StaticPrecision precision);

std::vector<double> read_values(Reader& reader, StaticPrecision precision);

std::vector<float> read_values_float(Reader& reader);

void write_p2p_blocks(Writer& writer,
                      std::span<const StaticDipoleBlock> blocks,
                      StaticPrecision precision);

// NOTE(cdfmm): the compact plan is seeded from operator_map.source_count,
// target_count, and row_offsets, so the caller must populate those before
// calling. Decoding and compact packing share one pass deliberately.
void read_p2p_blocks(Reader& reader, std::uint64_t block_count64,
                     StaticPrecision precision,
                     StaticP2POperator& operator_map,
                     StaticP2PCompactPlan& compact_plan);

void read_p2p_blocks_float(Reader& reader, std::uint64_t block_count64,
                           FloatStaticP2POperator& operator_map);

//------------------------------------------------------------------------------
// Cache identity: geometry/option facts in, keys and digest out. keys.cpp.
//------------------------------------------------------------------------------

// Explicit geometry/option facts that determine cache identity. This is the
// same information `initialise_cache_keys` used to read directly from
// `UniformFmm`; `execution_setup.cpp` now assembles it from the solver's
// state and hands it over instead of identity logic reaching into the solver
// object itself. References are only read for the duration of the call.
struct CacheIdentityInputs {
  ExpansionBasis expansion_basis;
  StaticPrecision precision;
  int expansion_order;
  SourceGeometry source_geometry;
  TargetGeometry target_geometry;
  SourceModel near_field_source_model;
  TargetModel near_field_target_model;
  SourceModel far_field_source_model;
  TargetModel far_field_target_model;
  bool use_reduced_symmetry_p2p;
  const PeriodicCellOptions& periodic;
  const UniformTree& tree;
  std::span<const CuboidSize> sorted_source_sizes;
  std::span<const CuboidSize> sorted_target_sizes;
  std::span<const Tetrahedron> sorted_source_tetrahedra;
  std::span<const Tetrahedron> sorted_target_tetrahedra;
  const std::optional<std::vector<int>>& fixed_target_source_indices;
};

// Resolved cache identity, ready to store back on the solver object. Every
// key/digest is empty and `enabled` is false whenever the enclosing option
// disables caching, the environment disables it, or the topology was
// supplied externally (see `compute_cache_identity`'s `supplied_topology`
// parameter).
struct CacheIdentity {
  bool enabled{false};
  std::string directory{};
  std::string universal_key{};
  std::string periodic_key{};
  std::string geometry_key{};
  std::string geometry_hash_digest{};
};

// Computes cache identity from explicit geometry/option facts. `statistics`
// receives the geometry-hash phase timing, matching the timing this used to
// record as a side effect of `initialise_cache_keys`. Caching is disabled
// unconditionally, with an otherwise-default-constructed identity, whenever
// `supplied_topology` is set: supplied topologies do not yet participate in
// cache keys (see `UniformFmm`'s topology-supplying constructor).
[[nodiscard]] CacheIdentity compute_cache_identity(
    bool supplied_topology, bool option_enable_cache,
    const CacheIdentityInputs& inputs, StaticPlanStatistics& statistics);

//------------------------------------------------------------------------------
// Depth-independent translation bank and periodic root payload. universal.cpp.
//------------------------------------------------------------------------------

struct UniversalCacheIdentity {
  bool enabled{false};
  std::string directory{};
  std::string universal_key{};
  std::string periodic_key{};
  ExpansionBasis basis{};
  StaticPrecision precision{};
  int order{0};
  bool periodic_enabled{false};
};

// Mutable references to the depth-independent operator bank this payload
// persists. `m2l_matrices` is `StaticM2LPlan::matrices`: the universal bank is
// always built and cached in FP64, then widened once into the FP32 plan by
// `quantise_static_plan_to_float`, so only the FP64 array is a payload field
// here.
struct UniversalCachePayload {
  std::array<StaticCoefficientOperator, 8>& m2m_operators;
  std::array<StaticCoefficientOperator, 8>& l2l_operators;
  std::vector<double>& m2l_matrices;
  bool& periodic_operator_available;
};

// Loads the universal translation bank and, if `identity.periodic_enabled`,
// the periodic root operator appended to its tail. Returns whether the
// universal bank itself was hit; the periodic root is loaded independently
// and best-effort (a periodic miss does not fail this call, matching prior
// behaviour). A missing, truncated, or incompatible file is a safe miss.
[[nodiscard]] bool load_universal_cache(const UniversalCacheIdentity& identity,
                                        int coefficient_count,
                                        UniversalCachePayload payload,
                                        StaticPlanStatistics& statistics);

// Writes the universal translation bank, and the periodic root tail when
// `identity.periodic_enabled` and the tail is present in `m2l_matrices`.
void write_universal_cache(
    const UniversalCacheIdentity& identity, int coefficient_count,
    const std::array<StaticCoefficientOperator, 8>& m2m_operators,
    const std::array<StaticCoefficientOperator, 8>& l2l_operators,
    const std::vector<double>& m2l_matrices, StaticPlanStatistics& statistics);

//------------------------------------------------------------------------------
// Geometry-dependent plan payload. geometry.cpp.
//------------------------------------------------------------------------------

struct GeometryCacheIdentity {
  bool enabled{false};
  std::string directory{};
  std::string geometry_key{};
  std::string geometry_hash_digest{};
  ExpansionBasis basis{};
  StaticPrecision precision{};
  int order{0};
};

// Mutable references to exactly the geometry-dependent plan state this
// payload persists or restores. FP32 and FP64 fields both appear because a
// warm load populates only the side selected by `identity.precision`; a
// direct FP32 hit leaves the FP64 fields untouched (see
// `geometry_cache_loaded_direct_float`).
struct GeometryCachePayload {
  std::vector<P2MPlan>& p2m_plans;
  std::vector<FloatP2MPlan>& p2m_plans_float;
  StaticM2LPlan& m2l_plan;
  FloatStaticM2LPlan& m2l_plan_float;
  std::vector<StaticL2PEvaluator>& l2p_evaluators;
  std::vector<FloatStaticL2PEvaluator>& l2p_evaluators_float;
  StaticP2POperator& p2p_operator;
  FloatStaticP2POperator& p2p_operator_float;
  StaticP2PCompactPlan& p2p_compact_plan;
  FloatStaticP2PCompactPlan& p2p_compact_plan_float;
  FloatStaticP2PBsrPlan& p2p_bsr_plan_float;
};

// Loads a geometry plan, validating every cached tree/topology invariant
// against `tree`/`topology`. A missing, truncated, corrupt, or mismatched file
// is a safe miss (returns false) and never throws past this call. On a direct
// FP32 hit, `geometry_cache_loaded_direct_float` is set true and only the
// `_float` payload fields are populated. On any miss, the FP32 payload fields
// and `geometry_cache_loaded_direct_float` are reset exactly as before: the
// FP64 fields are deliberately left untouched (`cache/AGENTS.md` warns against
// symmetrising this FP32-only failure cleanup).
[[nodiscard]] bool load_geometry_cache(
    const GeometryCacheIdentity& identity, const UniformTree& tree,
    const StaticFmmTopology& topology,
    const std::optional<std::vector<int>>& fixed_target_source_indices,
    GeometryCachePayload payload, bool& geometry_cache_loaded_direct_float,
    StaticPlanStatistics& statistics);

// Writes the FP64 geometry plan built by cold construction. Always called
// with the FP64 payload populated; the FP32 twin and the compact/BSR P2P
// packings are derived state, not part of the persisted geometry payload, and
// are not parameters here.
void write_geometry_cache(
    const GeometryCacheIdentity& identity, const UniformTree& tree,
    const std::optional<std::vector<int>>& fixed_target_source_indices,
    const std::vector<P2MPlan>& p2m_plans, const StaticM2LPlan& m2l_plan,
    const std::vector<StaticL2PEvaluator>& l2p_evaluators,
    const StaticP2POperator& p2p_operator, StaticPlanStatistics& statistics);

} // namespace cdfmm::detail::cache
