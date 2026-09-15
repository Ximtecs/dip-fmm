// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cdfmm/uniform_fmm.hpp"

#include <sys/mman.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Implementation-only interface shared by the cache translation units. The
// cache subsystem persists solver data defined elsewhere: nothing here
// constructs operators, plans, topology, or execution packings.
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

} // namespace cdfmm::detail::cache
