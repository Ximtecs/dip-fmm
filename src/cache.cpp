// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/uniform_fmm.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <sys/mman.h>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>

namespace cdfmm {
namespace {

constexpr std::uint32_t kCacheSchemaVersion = 2;
constexpr std::uint32_t kOperatorVersion = 1;
constexpr std::uint32_t kEndianMarker = 0x01020304U;
constexpr std::uint32_t kChecksumAlgorithm = 2U; // fast 64-bit payload checksum
constexpr std::size_t kUniversalClassCount =
    StaticPlanStatistics::theoretical_maximum_m2l_classes;

class Sha256 {
public:
  void update(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    total_bytes_ += size;
    while (size != 0) {
      const std::size_t copied = std::min(size, block_.size() - used_);
      std::memcpy(block_.data() + used_, bytes, copied);
      used_ += copied;
      bytes += copied;
      size -= copied;
      if (used_ == block_.size()) {
        transform(block_.data());
        used_ = 0;
      }
    }
  }

  template <typename T> void value(const T& input) {
    static_assert(std::is_trivially_copyable_v<T>);
    update(&input, sizeof(input));
  }

  [[nodiscard]] std::array<unsigned char, 32> finish() {
    const std::uint64_t bit_count = total_bytes_ * 8;
    const unsigned char marker = 0x80;
    update(&marker, 1);
    const unsigned char zero = 0;
    while (used_ != 56) {
      update(&zero, 1);
    }
    std::array<unsigned char, 8> length{};
    for (int index = 0; index < 8; ++index) {
      length[static_cast<std::size_t>(7 - index)] =
          static_cast<unsigned char>(bit_count >> (8 * index));
    }
    update(length.data(), length.size());
    std::array<unsigned char, 32> digest{};
    for (std::size_t index = 0; index < state_.size(); ++index) {
      for (int byte = 0; byte < 4; ++byte) {
        digest[index * 4 + static_cast<std::size_t>(byte)] =
            static_cast<unsigned char>(state_[index] >> (24 - 8 * byte));
      }
    }
    return digest;
  }

private:
  static constexpr std::array<std::uint32_t, 64> constants_{
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
      0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
      0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
      0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
      0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
      0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
      0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
      0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
      0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
      0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
      0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
      0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

  static std::uint32_t rotate(const std::uint32_t value,
                              const unsigned amount) {
    return (value >> amount) | (value << (32 - amount));
  }

  void transform(const unsigned char* data) {
    std::array<std::uint32_t, 64> words{};
    for (int index = 0; index < 16; ++index) {
      const std::size_t offset = static_cast<std::size_t>(index) * 4;
      words[static_cast<std::size_t>(index)] =
          (static_cast<std::uint32_t>(data[offset]) << 24) |
          (static_cast<std::uint32_t>(data[offset + 1]) << 16) |
          (static_cast<std::uint32_t>(data[offset + 2]) << 8) |
          static_cast<std::uint32_t>(data[offset + 3]);
    }
    for (int index = 16; index < 64; ++index) {
      const std::uint32_t a = words[static_cast<std::size_t>(index - 15)];
      const std::uint32_t b = words[static_cast<std::size_t>(index - 2)];
      const std::uint32_t s0 = rotate(a, 7) ^ rotate(a, 18) ^ (a >> 3);
      const std::uint32_t s1 = rotate(b, 17) ^ rotate(b, 19) ^ (b >> 10);
      words[static_cast<std::size_t>(index)] =
          words[static_cast<std::size_t>(index - 16)] + s0 +
          words[static_cast<std::size_t>(index - 7)] + s1;
    }
    auto working = state_;
    for (int index = 0; index < 64; ++index) {
      const std::uint32_t s1 = rotate(working[4], 6) ^
          rotate(working[4], 11) ^ rotate(working[4], 25);
      const std::uint32_t choose =
          (working[4] & working[5]) ^ (~working[4] & working[6]);
      const std::uint32_t temp1 = working[7] + s1 + choose +
          constants_[static_cast<std::size_t>(index)] +
          words[static_cast<std::size_t>(index)];
      const std::uint32_t s0 = rotate(working[0], 2) ^
          rotate(working[0], 13) ^ rotate(working[0], 22);
      const std::uint32_t majority = (working[0] & working[1]) ^
          (working[0] & working[2]) ^ (working[1] & working[2]);
      const std::uint32_t temp2 = s0 + majority;
      for (int lane = 7; lane > 0; --lane) {
        working[static_cast<std::size_t>(lane)] =
            working[static_cast<std::size_t>(lane - 1)];
      }
      working[4] += temp1;
      working[0] = temp1 + temp2;
    }
    for (std::size_t index = 0; index < state_.size(); ++index) {
      state_[index] += working[index];
    }
  }

  std::array<std::uint32_t, 8> state_{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<unsigned char, 64> block_{};
  std::size_t used_{0};
  std::uint64_t total_bytes_{0};
};

std::string hexadecimal(const std::array<unsigned char, 32>& digest) {
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (const unsigned char byte : digest) {
    stream << std::setw(2) << static_cast<unsigned>(byte);
  }
  return stream.str();
}

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

CachePayload map_cache_file(const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("cache file unavailable");
  }

  struct stat metadata {};
  if (::fstat(fd, &metadata) != 0 || metadata.st_size <= 0) {
    (void)::close(fd);
    throw std::runtime_error("cache file metadata unavailable");
  }
  if (static_cast<std::uintmax_t>(metadata.st_size) >
      std::numeric_limits<std::size_t>::max()) {
    (void)::close(fd);
    throw std::runtime_error("cache file too large for this process");
  }

  const std::size_t size = static_cast<std::size_t>(metadata.st_size);
  void* mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  const int saved_errno = errno;
  (void)::close(fd);
  if (mapping == MAP_FAILED) {
    errno = saved_errno;
    throw std::runtime_error("failed to memory-map cache file");
  }
  return CachePayload(mapping, size);
}

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

[[nodiscard]] std::size_t checked_bytes(const std::uint64_t count,
                                        const std::size_t element_size) {
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

[[nodiscard]] std::uint64_t checksum_avalanche(std::uint64_t value) noexcept {
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdULL;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ULL;
  value ^= value >> 33;
  return value;
}

// Cache corruption detection only; geometry identity continues to use SHA-256.
// Four independent streaming accumulators keep this close to memory bandwidth
// while still detecting ordinary truncation/bit corruption robustly.
[[nodiscard]] std::uint64_t fast_checksum64(const void* data,
                                            const std::size_t size) noexcept {
  const auto* bytes = static_cast<const unsigned char*>(data);
  std::uint64_t a = 0x9e3779b97f4a7c15ULL;
  std::uint64_t b = 0xd1b54a32d192ed03ULL;
  std::uint64_t c = 0x94d049bb133111ebULL;
  std::uint64_t d = 0x243f6a8885a308d3ULL;
  std::size_t offset = 0;

  while (offset + 32 <= size) {
    const std::uint64_t x0 = load_unaligned<std::uint64_t>(bytes + offset);
    const std::uint64_t x1 = load_unaligned<std::uint64_t>(bytes + offset + 8);
    const std::uint64_t x2 = load_unaligned<std::uint64_t>(bytes + offset + 16);
    const std::uint64_t x3 = load_unaligned<std::uint64_t>(bytes + offset + 24);
    a += x0;
    b ^= std::rotl(x1, 17);
    c += x2;
    d ^= std::rotl(x3, 31);
    b += a;
    d += c;
    offset += 32;
  }
  while (offset + 8 <= size) {
    a += load_unaligned<std::uint64_t>(bytes + offset);
    b += a;
    offset += 8;
  }
  if (offset != size) {
    std::uint64_t tail = 0;
    std::memcpy(&tail, bytes + offset, size - offset);
    c += tail;
    d += c;
  }

  return checksum_avalanche(a ^ std::rotl(b, 13) ^ std::rotl(c, 29) ^
                            std::rotl(d, 47) ^ size);
}

bool write_all_fd(const int fd, const void* data, std::size_t bytes) noexcept {
  const auto* input = static_cast<const unsigned char*>(data);
  while (bytes != 0) {
    const ssize_t count = ::write(fd, input, bytes);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      return false;
    }
    input += static_cast<std::size_t>(count);
    bytes -= static_cast<std::size_t>(count);
  }
  return true;
}

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

std::filesystem::path cache_root() {
  if (const char* override_path = std::getenv("CDFMM_CACHE_DIR");
      override_path != nullptr && *override_path != '\0') {
    return std::filesystem::path(override_path) / "v1";
  }
#ifdef CDFMM_DEFAULT_CACHE_DIR
  return std::filesystem::path(CDFMM_DEFAULT_CACHE_DIR) / "v1";
#else
  return std::filesystem::current_path() / "caches" / "v1";
#endif
}

bool environment_disables_cache() {
  const char* value = std::getenv("CDFMM_DISABLE_CACHE");
  return value != nullptr && std::string_view(value) != "0";
}

CachePayload read_cache(const std::filesystem::path& path,
                        const CacheDescriptor& expected,
                        std::size_t& bytes_read) {
  CachePayload file = map_cache_file(path);
  Reader reader(file.full_bytes());

  const std::array<unsigned char, 8> expected_magic{
      'C', 'D', 'F', 'M', 'M', 'C', '2', '\0'};
  std::array<unsigned char, 8> magic{};
  std::memcpy(magic.data(), reader.take_bytes(magic.size()), magic.size());
  if (magic != expected_magic) {
    throw std::runtime_error("cache magic mismatch");
  }

  if (reader.scalar<std::uint32_t>() != kCacheSchemaVersion ||
      reader.scalar<std::uint32_t>() != kOperatorVersion ||
      reader.scalar<std::uint32_t>() != kEndianMarker ||
      reader.scalar<std::uint32_t>() != sizeof(std::size_t) ||
      reader.scalar<std::uint32_t>() != static_cast<std::uint32_t>(expected.kind) ||
      reader.scalar<std::uint32_t>() != static_cast<std::uint32_t>(expected.basis) ||
      reader.scalar<int>() != expected.order ||
      reader.scalar<std::uint32_t>() != static_cast<std::uint32_t>(expected.precision) ||
      reader.scalar<int>() != expected.depth ||
      reader.scalar<std::uint32_t>() != kChecksumAlgorithm) {
    throw std::runtime_error("cache version mismatch");
  }

  if (reader.string() != expected.key) {
    throw std::runtime_error("cache key mismatch");
  }
  if (reader.string() != expected.geometry_hash) {
    throw std::runtime_error("cache geometry hash mismatch");
  }

  const std::uint64_t payload_offset64 = reader.scalar<std::uint64_t>();
  const std::uint64_t payload_size64 = reader.scalar<std::uint64_t>();
  const std::uint64_t expected_checksum = reader.scalar<std::uint64_t>();
  if (payload_offset64 > std::numeric_limits<std::size_t>::max() ||
      payload_size64 > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("cache payload is too large");
  }
  const std::size_t payload_offset = static_cast<std::size_t>(payload_offset64);
  const std::size_t payload_size = static_cast<std::size_t>(payload_size64);
  const std::size_t file_size = file.full_bytes().size();
  if (reader.offset() != payload_offset || payload_offset > file_size ||
      payload_size != file_size - payload_offset) {
    throw std::runtime_error("cache section bounds mismatch");
  }

  file.select_payload(payload_offset, payload_size);
  const auto payload = file.bytes();
  if (fast_checksum64(payload.data(), payload.size()) != expected_checksum) {
    throw std::runtime_error("cache checksum mismatch");
  }

  bytes_read += file_size;
  return file;
}

std::size_t write_cache(const std::filesystem::path& path,
                        const CacheDescriptor& descriptor,
                        const std::vector<unsigned char>& payload) {
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error) {
    return 0;
  }

  Writer header;
  header.reserve(256 + descriptor.key.size() + descriptor.geometry_hash.size());
  const std::array<unsigned char, 8> magic{
      'C', 'D', 'F', 'M', 'M', 'C', '2', '\0'};
  header.append_raw(magic.data(), magic.size());
  header.scalar(kCacheSchemaVersion);
  header.scalar(kOperatorVersion);
  header.scalar(kEndianMarker);
  header.scalar<std::uint32_t>(sizeof(std::size_t));
  header.scalar(static_cast<std::uint32_t>(descriptor.kind));
  header.scalar(static_cast<std::uint32_t>(descriptor.basis));
  header.scalar(descriptor.order);
  header.scalar(static_cast<std::uint32_t>(descriptor.precision));
  header.scalar(descriptor.depth);
  header.scalar(kChecksumAlgorithm);
  header.string(descriptor.key);
  header.string(descriptor.geometry_hash);

  constexpr std::size_t fixed_header_bytes =
      8 + 10 * sizeof(std::uint32_t) + 4 * sizeof(std::uint64_t) +
      sizeof(std::uint64_t);
  const std::uint64_t payload_offset = fixed_header_bytes +
      descriptor.key.size() + descriptor.geometry_hash.size();
  header.scalar(payload_offset);
  header.scalar<std::uint64_t>(payload.size());
  header.scalar(fast_checksum64(payload.data(), payload.size()));
  if (header.bytes().size() != payload_offset) {
    throw std::runtime_error("internal cache header size mismatch");
  }

  static std::atomic<std::uint64_t> temporary_counter{0};
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::uint64_t unique = temporary_counter.fetch_add(
      1, std::memory_order_relaxed);
  const std::filesystem::path temporary =
      path.string() + ".tmp." + std::to_string(::getpid()) + "." +
      std::to_string(stamp) + "." + std::to_string(unique);
  const int descriptor_fd =
      ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  if (descriptor_fd < 0) {
    return 0;
  }

  const bool write_ok =
      write_all_fd(descriptor_fd, header.bytes().data(), header.bytes().size()) &&
      write_all_fd(descriptor_fd, payload.data(), payload.size());
  const bool fsync_ok = write_ok && ::fsync(descriptor_fd) == 0;
  const bool close_ok = ::close(descriptor_fd) == 0;
  if (!write_ok || !fsync_ok || !close_ok) {
    std::filesystem::remove(temporary, error);
    return 0;
  }

  std::filesystem::rename(temporary, path, error);
  if (error) {
    // Another process may have completed the same valid cache first.
    std::filesystem::remove(temporary, error);
    return std::filesystem::exists(path)
        ? header.bytes().size() + payload.size()
        : 0;
  }
  const int directory_fd = ::open(path.parent_path().c_str(), O_RDONLY);
  if (directory_fd >= 0) {
    (void)::fsync(directory_fd);
    (void)::close(directory_fd);
  }
  return header.bytes().size() + payload.size();
}

std::string basis_name(const ExpansionBasis basis) {
  return basis == ExpansionBasis::Spherical ? "spherical" : "cartesian";
}

std::string precision_name(const StaticPrecision precision) {
  return precision == StaticPrecision::Float32 ? "f32" : "f64";
}

std::string cache_number(const double value) {
  std::array<char, 64> buffer{};
  const auto [end, error] = std::to_chars(
      buffer.data(), buffer.data() + buffer.size(), value,
      std::chars_format::general);
  if (error != std::errc{}) {
    throw std::runtime_error("failed to format cache-key number");
  }
  return std::string(buffer.data(), end);
}

void write_operator(Writer& writer, const StaticCoefficientOperator& value,
                    const StaticPrecision precision) {
  writer.scalar(value.input_size);
  writer.scalar(value.output_size);
  writer.scalar<std::uint64_t>(value.entries.size());
  const std::size_t record_bytes =
      2 * sizeof(int) +
      (precision == StaticPrecision::Float32 ? sizeof(float) : sizeof(double));
  unsigned char* raw = writer.append_uninitialized(
      checked_bytes(value.entries.size(), record_bytes));
  const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(value.entries.size());
#pragma omp parallel for schedule(static) if (count >= 100000)
  for (std::ptrdiff_t index = 0; index < count; ++index) {
    const StaticOperatorEntry& entry = value.entries[static_cast<std::size_t>(index)];
    unsigned char* record = raw + static_cast<std::size_t>(index) * record_bytes;
    store_unaligned<int>(record, entry.output);
    store_unaligned<int>(record + sizeof(int), entry.input);
    if (precision == StaticPrecision::Float32) {
      store_unaligned<float>(record + 2 * sizeof(int),
                             static_cast<float>(entry.value));
    } else {
      store_unaligned<double>(record + 2 * sizeof(int), entry.value);
    }
  }
}

StaticCoefficientOperator read_operator(Reader& reader,
                                        const StaticPrecision precision) {
  StaticCoefficientOperator value;
  value.input_size = reader.scalar<int>();
  value.output_size = reader.scalar<int>();
  const std::uint64_t count64 = reader.scalar<std::uint64_t>();
  if (count64 > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    throw std::runtime_error("operator cache entry count too large");
  }
  const std::size_t record_bytes =
      2 * sizeof(int) +
      (precision == StaticPrecision::Float32 ? sizeof(float) : sizeof(double));
  const unsigned char* raw = reader.take_bytes(checked_bytes(count64, record_bytes));
  value.entries.resize(static_cast<std::size_t>(count64));
  const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(count64);
#pragma omp parallel for schedule(static) if (count >= 100000)
  for (std::ptrdiff_t index = 0; index < count; ++index) {
    const unsigned char* record =
        raw + static_cast<std::size_t>(index) * record_bytes;
    StaticOperatorEntry& entry = value.entries[static_cast<std::size_t>(index)];
    entry.output = load_unaligned<int>(record);
    entry.input = load_unaligned<int>(record + sizeof(int));
    entry.value = precision == StaticPrecision::Float32
        ? static_cast<double>(load_unaligned<float>(record + 2 * sizeof(int)))
        : load_unaligned<double>(record + 2 * sizeof(int));
  }
  return value;
}

FloatStaticCoefficientOperator read_operator_float(Reader& reader) {
  FloatStaticCoefficientOperator value;
  value.input_size = reader.scalar<int>();
  value.output_size = reader.scalar<int>();
  const std::uint64_t count64 = reader.scalar<std::uint64_t>();
  if (count64 >
      static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    throw std::runtime_error("operator cache entry count too large");
  }
  constexpr std::size_t record_bytes = 2 * sizeof(int) + sizeof(float);
  const unsigned char* raw =
      reader.take_bytes(checked_bytes(count64, record_bytes));
  value.entries.resize(static_cast<std::size_t>(count64));
  const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(count64);
#pragma omp parallel for schedule(static) if (count >= 100000)
  for (std::ptrdiff_t index = 0; index < count; ++index) {
    const unsigned char* record =
        raw + static_cast<std::size_t>(index) * record_bytes;
    FloatStaticOperatorEntry& entry =
        value.entries[static_cast<std::size_t>(index)];
    entry.output = load_unaligned<int>(record);
    entry.input = load_unaligned<int>(record + sizeof(int));
    entry.value = load_unaligned<float>(record + 2 * sizeof(int));
  }
  return value;
}

void write_values(Writer& writer, const std::span<const double> values,
                  const StaticPrecision precision) {
  writer.scalar<std::uint64_t>(values.size());
  if (precision == StaticPrecision::Float64) {
    writer.append_raw(values.data(), values.size_bytes());
    return;
  }
  unsigned char* raw =
      writer.append_uninitialized(checked_bytes(values.size(), sizeof(float)));
  const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(values.size());
#pragma omp parallel for schedule(static) if (count >= 100000)
  for (std::ptrdiff_t index = 0; index < count; ++index) {
    store_unaligned<float>(raw + static_cast<std::size_t>(index) * sizeof(float),
                           static_cast<float>(values[static_cast<std::size_t>(index)]));
  }
}

void write_values(Writer& writer, const std::vector<double>& values,
                  const StaticPrecision precision) {
  write_values(writer, std::span<const double>(values), precision);
}

std::vector<double> read_values(Reader& reader,
                                const StaticPrecision precision) {
  const std::uint64_t count64 = reader.scalar<std::uint64_t>();
  if (count64 > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    throw std::runtime_error("cache value count too large");
  }
  const std::size_t count = static_cast<std::size_t>(count64);
  std::vector<double> values(count);
  if (precision == StaticPrecision::Float64) {
    const std::size_t bytes = checked_bytes(count64, sizeof(double));
    if (bytes != 0) {
      std::memcpy(values.data(), reader.take_bytes(bytes), bytes);
    }
    return values;
  }

  const unsigned char* raw =
      reader.take_bytes(checked_bytes(count64, sizeof(float)));
  const std::ptrdiff_t parallel_count = static_cast<std::ptrdiff_t>(count64);
#pragma omp parallel for schedule(static) if (parallel_count >= 100000)
  for (std::ptrdiff_t index = 0; index < parallel_count; ++index) {
    values[static_cast<std::size_t>(index)] = static_cast<double>(
        load_unaligned<float>(raw + static_cast<std::size_t>(index) * sizeof(float)));
  }
  return values;
}

std::vector<float> read_values_float(Reader& reader) {
  const std::uint64_t count64 = reader.scalar<std::uint64_t>();
  const std::size_t bytes = checked_bytes(count64, sizeof(float));
  std::vector<float> values(static_cast<std::size_t>(count64));
  if (bytes != 0) {
    std::memcpy(values.data(), reader.take_bytes(bytes), bytes);
  }
  return values;
}

[[nodiscard]] std::size_t p2p_record_bytes(const StaticPrecision precision) {
  return 3 * sizeof(int) + 9 *
      (precision == StaticPrecision::Float32 ? sizeof(float) : sizeof(double));
}

void write_p2p_blocks(Writer& writer,
                      const std::span<const StaticDipoleBlock> blocks,
                      const StaticPrecision precision) {
  const std::size_t record_bytes = p2p_record_bytes(precision);
  unsigned char* raw =
      writer.append_uninitialized(checked_bytes(blocks.size(), record_bytes));
  if (blocks.size() > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    throw std::runtime_error("P2P cache block count too large");
  }
  const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(blocks.size());
#pragma omp parallel for schedule(static) if (count >= 100000)
  for (std::ptrdiff_t index = 0; index < count; ++index) {
    const StaticDipoleBlock& block = blocks[static_cast<std::size_t>(index)];
    unsigned char* record = raw + static_cast<std::size_t>(index) * record_bytes;
    store_unaligned<int>(record, block.target);
    store_unaligned<int>(record + sizeof(int), block.source);
    const std::array<double, 9> values{
        block.px, block.py, block.pz, block.xx, block.xy,
        block.xz, block.yy, block.yz, block.zz};
    std::size_t offset = 2 * sizeof(int);
    if (precision == StaticPrecision::Float32) {
      for (const double value : values) {
        store_unaligned<float>(record + offset, static_cast<float>(value));
        offset += sizeof(float);
      }
    } else {
      for (const double value : values) {
        store_unaligned<double>(record + offset, value);
        offset += sizeof(double);
      }
    }
    store_unaligned<int>(record + offset, block.skip_for_identity);
  }
}

void read_p2p_blocks(Reader& reader,
                     const std::uint64_t block_count64,
                     const StaticPrecision precision,
                     StaticP2POperator& operator_map,
                     StaticP2PCompactPlan& compact_plan) {
  if (block_count64 >
      static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    throw std::runtime_error("P2P cache block count too large");
  }
  const std::size_t block_count = static_cast<std::size_t>(block_count64);
  const std::size_t record_bytes = p2p_record_bytes(precision);
  const unsigned char* raw =
      reader.take_bytes(checked_bytes(block_count64, record_bytes));

  operator_map.blocks.resize(block_count);

  compact_plan.source_count = operator_map.source_count;
  compact_plan.target_count = operator_map.target_count;
  compact_plan.row_offsets = operator_map.row_offsets;
  compact_plan.source_indices.resize(block_count);
  compact_plan.skip_for_identity.resize(block_count);
  for (auto& row : compact_plan.potential) {
    row.resize(block_count);
  }
  for (auto& row : compact_plan.tensors) {
    row.resize(block_count);
  }

  const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(block_count64);
#pragma omp parallel for schedule(static) if (count >= 100000)
  for (std::ptrdiff_t index = 0; index < count; ++index) {
    const std::size_t slot = static_cast<std::size_t>(index);
    const unsigned char* record = raw + slot * record_bytes;
    StaticDipoleBlock& block = operator_map.blocks[slot];
    block.target = load_unaligned<int>(record);
    block.source = load_unaligned<int>(record + sizeof(int));
    std::size_t offset = 2 * sizeof(int);
    std::array<double, 9> values{};
    if (precision == StaticPrecision::Float32) {
      for (double& value : values) {
        value = static_cast<double>(load_unaligned<float>(record + offset));
        offset += sizeof(float);
      }
    } else {
      for (double& value : values) {
        value = load_unaligned<double>(record + offset);
        offset += sizeof(double);
      }
    }
    block.px = values[0];
    block.py = values[1];
    block.pz = values[2];
    block.xx = values[3];
    block.xy = values[4];
    block.xz = values[5];
    block.yy = values[6];
    block.yz = values[7];
    block.zz = values[8];
    block.skip_for_identity = load_unaligned<int>(record + offset);

    compact_plan.source_indices[slot] = block.source;
    compact_plan.skip_for_identity[slot] =
        static_cast<unsigned char>(block.skip_for_identity != 0);
    compact_plan.potential[0][slot] = block.px;
    compact_plan.potential[1][slot] = block.py;
    compact_plan.potential[2][slot] = block.pz;
    compact_plan.tensors[0][slot] = block.xx;
    compact_plan.tensors[1][slot] = block.xy;
    compact_plan.tensors[2][slot] = block.xz;
    compact_plan.tensors[3][slot] = block.yy;
    compact_plan.tensors[4][slot] = block.yz;
    compact_plan.tensors[5][slot] = block.zz;
  }
}

void read_p2p_blocks_float(Reader& reader,
                           const std::uint64_t block_count64,
                           FloatStaticP2POperator& operator_map) {
  if (block_count64 >
      static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    throw std::runtime_error("P2P cache block count too large");
  }
  const std::size_t block_count = static_cast<std::size_t>(block_count64);
  const std::size_t record_bytes = p2p_record_bytes(StaticPrecision::Float32);
  const unsigned char* raw =
      reader.take_bytes(checked_bytes(block_count64, record_bytes));
  operator_map.blocks.resize(block_count);

  const std::ptrdiff_t count = static_cast<std::ptrdiff_t>(block_count64);
#pragma omp parallel for schedule(static) if (count >= 100000)
  for (std::ptrdiff_t index = 0; index < count; ++index) {
    const std::size_t slot = static_cast<std::size_t>(index);
    const unsigned char* record = raw + slot * record_bytes;
    FloatStaticDipoleBlock& block = operator_map.blocks[slot];
    block.target = load_unaligned<int>(record);
    block.source = load_unaligned<int>(record + sizeof(int));
    std::size_t offset = 2 * sizeof(int);
    block.px = load_unaligned<float>(record + offset);
    offset += sizeof(float);
    block.py = load_unaligned<float>(record + offset);
    offset += sizeof(float);
    block.pz = load_unaligned<float>(record + offset);
    offset += sizeof(float);
    block.xx = load_unaligned<float>(record + offset);
    offset += sizeof(float);
    block.xy = load_unaligned<float>(record + offset);
    offset += sizeof(float);
    block.xz = load_unaligned<float>(record + offset);
    offset += sizeof(float);
    block.yy = load_unaligned<float>(record + offset);
    offset += sizeof(float);
    block.yz = load_unaligned<float>(record + offset);
    offset += sizeof(float);
    block.zz = load_unaligned<float>(record + offset);
    offset += sizeof(float);
    block.skip_for_identity = load_unaligned<int>(record + offset);
  }
}

std::filesystem::path cache_path(const std::string& root,
                                 const std::string& category,
                                 const std::string& key) {
  return std::filesystem::path(root) / category / key;
}

template <typename T> void hash_value(Sha256& hash, const T value) {
  hash.value(value);
}

std::int64_t canonical_coordinate(const double value) {
  constexpr double resolution = 1.0e9;
  return static_cast<std::int64_t>(std::llround(value * resolution));
}

struct UniformGridDescriptor {
  std::array<std::int64_t, 3> first{};
  std::array<std::int64_t, 3> step{};
  std::array<std::size_t, 3> count{};
};

std::optional<UniformGridDescriptor> detect_uniform_grid(
    const std::span<const Vec3> positions) {
  if (positions.empty()) {
    return std::nullopt;
  }
  std::array<std::vector<std::int64_t>, 3> axes;
  for (auto& axis : axes) {
    axis.reserve(positions.size());
  }
  for (const Vec3& position : positions) {
    axes[0].push_back(canonical_coordinate(position.x));
    axes[1].push_back(canonical_coordinate(position.y));
    axes[2].push_back(canonical_coordinate(position.z));
  }
  UniformGridDescriptor descriptor;
  std::size_t product = 1;
  for (int dimension = 0; dimension < 3; ++dimension) {
    auto& axis = axes[static_cast<std::size_t>(dimension)];
    std::sort(axis.begin(), axis.end());
    axis.erase(std::unique(axis.begin(), axis.end()), axis.end());
    descriptor.first[static_cast<std::size_t>(dimension)] = axis.front();
    descriptor.count[static_cast<std::size_t>(dimension)] = axis.size();
    descriptor.step[static_cast<std::size_t>(dimension)] =
        axis.size() > 1 ? axis[1] - axis[0] : 0;
    if (axis.size() > 1 &&
        !std::equal(axis.begin() + 1, axis.end(), axis.begin(),
                    [step = descriptor.step[static_cast<std::size_t>(dimension)]](
                        const std::int64_t next, const std::int64_t previous) {
                      return next - previous == step;
                    })) {
      return std::nullopt;
    }
    if (axis.size() > positions.size() / product) {
      return std::nullopt;
    }
    product *= axis.size();
  }
  if (product != positions.size()) {
    return std::nullopt;
  }
  std::vector<unsigned char> occupied(product, 0);
  for (const Vec3& position : positions) {
    const std::array<std::int64_t, 3> coordinate{
        canonical_coordinate(position.x), canonical_coordinate(position.y),
        canonical_coordinate(position.z)};
    std::array<std::size_t, 3> index{};
    for (int dimension = 0; dimension < 3; ++dimension) {
      const auto& axis = axes[static_cast<std::size_t>(dimension)];
      const auto found = std::lower_bound(
          axis.begin(), axis.end(), coordinate[static_cast<std::size_t>(dimension)]);
      if (found == axis.end() || *found != coordinate[static_cast<std::size_t>(dimension)]) {
        return std::nullopt;
      }
      index[static_cast<std::size_t>(dimension)] =
          static_cast<std::size_t>(found - axis.begin());
    }
    const std::size_t flat = index[0] + descriptor.count[0] *
        (index[1] + descriptor.count[1] * index[2]);
    if (occupied[flat] != 0) {
      return std::nullopt;
    }
    occupied[flat] = 1;
  }
  return descriptor;
}

void hash_grid_descriptor(Sha256& hash,
                          const UniformGridDescriptor& descriptor) {
  hash.value<std::uint32_t>(0x47524944U); // "GRID"
  for (int dimension = 0; dimension < 3; ++dimension) {
    hash.value<std::uint64_t>(
        descriptor.count[static_cast<std::size_t>(dimension)]);
    hash.value(descriptor.first[static_cast<std::size_t>(dimension)]);
    hash.value(descriptor.step[static_cast<std::size_t>(dimension)]);
  }
}

void hash_permutation(Sha256& hash, const std::span<const int> permutation,
                      const std::span<const Vec3> sorted_positions,
                      const std::optional<UniformGridDescriptor>& grid) {
  hash.value<std::uint64_t>(permutation.size());
  if (grid && permutation.size() == sorted_positions.size()) {
    constexpr std::array<std::array<int, 3>, 6> axis_orders{{
        {{0, 1, 2}}, {{0, 2, 1}}, {{1, 0, 2}},
        {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}}}};
    for (std::size_t order = 0; order < axis_orders.size(); ++order) {
      for (unsigned reversals = 0; reversals < 8; ++reversals) {
        bool matches = true;
        for (std::size_t sorted = 0; sorted < permutation.size(); ++sorted) {
          const Vec3& position = sorted_positions[sorted];
          const std::array<std::int64_t, 3> coordinate{
              canonical_coordinate(position.x), canonical_coordinate(position.y),
              canonical_coordinate(position.z)};
          std::array<std::size_t, 3> index{};
          for (int dimension = 0; dimension < 3; ++dimension) {
            const std::size_t slot = static_cast<std::size_t>(dimension);
            index[slot] = grid->count[slot] == 1 ? 0 :
                static_cast<std::size_t>((coordinate[slot] - grid->first[slot]) /
                                         grid->step[slot]);
            if ((reversals & (1U << dimension)) != 0) {
              index[slot] = grid->count[slot] - 1 - index[slot];
            }
          }
          const auto& axes = axis_orders[order];
          const std::size_t expected =
              index[static_cast<std::size_t>(axes[0])] +
              grid->count[static_cast<std::size_t>(axes[0])] *
                  (index[static_cast<std::size_t>(axes[1])] +
                   grid->count[static_cast<std::size_t>(axes[1])] *
                       index[static_cast<std::size_t>(axes[2])]);
          if (permutation[sorted] != static_cast<int>(expected)) {
            matches = false;
            break;
          }
        }
        if (matches) {
          hash.value<std::uint32_t>(0x4c41594fU); // "LAYO"
          hash.value<std::uint32_t>(static_cast<std::uint32_t>(order));
          hash.value<std::uint32_t>(reversals);
          return;
        }
      }
    }
  }
  hash.value<std::uint32_t>(0x5045524dU); // "PERM"
  for (const int value : permutation) {
    hash.value(value);
  }
}

} // namespace

void UniformFmm::initialise_cache_keys(const UniformFmmOptions& options) {
  const auto start = std::chrono::steady_clock::now();
  cache_enabled_ = options.enable_cache && !environment_disables_cache();
  cache_directory_ = cache_root().string();
  std::ostringstream universal;
  universal << "operators_" << basis_name(expansion_basis_) << "_p"
            << std::setw(2) << std::setfill('0') << expansion_order() << '_'
            << precision_name(precision_) << "_m2m-m2l-l2l_v02.bin";
  universal_cache_key_ = universal.str();

  if (periodic_.enabled) {
    std::ostringstream periodic;
    periodic << "periodic_" << basis_name(expansion_basis_) << "_p"
             << std::setw(2) << std::setfill('0') << expansion_order() << '_'
             << precision_name(precision_) << "_zerok0_tol"
             << cache_number(periodic_.setup_tolerance)
             << "_v02.bin";
    periodic_cache_key_ = periodic.str();
  }

  Sha256 hash;
  hash_value(hash, kCacheSchemaVersion);
  hash_value(hash, kOperatorVersion);
  hash_value(hash, static_cast<std::uint32_t>(expansion_basis_));
  hash_value(hash, static_cast<std::uint32_t>(precision_));
  hash_value(hash, expansion_order());
  hash_value(hash, tree_.leaf_level());
  hash_value(hash, static_cast<std::uint32_t>(source_geometry_));
  hash_value(hash, static_cast<std::uint32_t>(target_geometry_));
  hash_value(hash, use_cuboid_p2m_);
  hash_value(hash, use_cuboid_l2p_);
  hash_value(hash, periodic_.enabled);
  hash_value(hash, periodic_.axes);
  hash_value(hash, static_cast<std::uint32_t>(periodic_.convention));
  hash_value(hash, periodic_.setup_tolerance);
  const auto hash_positions = [&hash](const std::span<const Vec3> positions) {
    hash_value(hash, static_cast<std::uint64_t>(positions.size()));
    const auto grid = detect_uniform_grid(positions);
    if (grid) {
      hash_grid_descriptor(hash, *grid);
      return grid;
    }
    hash_value(hash, std::uint32_t{0x504f494eU}); // "POIN"
    for (const Vec3& position : positions) {
      hash_value(hash, canonical_coordinate(position.x));
      hash_value(hash, canonical_coordinate(position.y));
      hash_value(hash, canonical_coordinate(position.z));
    }
    return grid;
  };
  const auto source_grid = hash_positions(tree_.sorted_source_positions());
  const auto target_grid = hash_positions(tree_.sorted_target_positions());
  const auto hash_sizes = [&hash](const std::vector<CuboidSize>& sizes) {
    hash_value(hash, static_cast<std::uint64_t>(sizes.size()));
    if (!sizes.empty() && std::all_of(
            sizes.begin() + 1, sizes.end(), [&sizes](const CuboidSize& size) {
              return canonical_coordinate(size.hx) ==
                         canonical_coordinate(sizes.front().hx) &&
                     canonical_coordinate(size.hy) ==
                         canonical_coordinate(sizes.front().hy) &&
                     canonical_coordinate(size.hz) ==
                         canonical_coordinate(sizes.front().hz);
            })) {
      hash_value(hash, std::uint32_t{0x53414d45U}); // "SAME"
      hash_value(hash, canonical_coordinate(sizes.front().hx));
      hash_value(hash, canonical_coordinate(sizes.front().hy));
      hash_value(hash, canonical_coordinate(sizes.front().hz));
      return;
    }
    hash_value(hash, std::uint32_t{0x53495a45U}); // "SIZE"
    for (const CuboidSize& size : sizes) {
      hash_value(hash, canonical_coordinate(size.hx));
      hash_value(hash, canonical_coordinate(size.hy));
      hash_value(hash, canonical_coordinate(size.hz));
    }
  };
  hash_sizes(sorted_source_sizes_);
  hash_sizes(sorted_target_sizes_);
  hash_permutation(hash, tree_.source_permutation(),
                   tree_.sorted_source_positions(), source_grid);
  hash_permutation(hash, tree_.target_permutation(),
                   tree_.sorted_target_positions(), target_grid);
  hash_value(hash, fixed_target_source_indices_.has_value());
  if (fixed_target_source_indices_) {
    for (const int value : *fixed_target_source_indices_) {
      hash_value(hash, value);
    }
  }
  const std::string digest = hexadecimal(hash.finish());
  geometry_hash_digest_ = digest;
  std::ostringstream plan;
  plan << "plan_" << basis_name(expansion_basis_) << "_p" << std::setw(2)
       << std::setfill('0') << expansion_order() << "_d" << std::setw(2)
       << tree_.leaf_level() << '_' << precision_name(precision_) << "_N_"
       << tree_.sorted_source_positions().size() << '_' << digest << "_v02.bin";
  geometry_cache_key_ = plan.str();
  static_plan_statistics_.geometry_hash.add(
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count());
}

bool UniformFmm::load_universal_cache() {
  if (!cache_enabled_) {
    return false;
  }
  const auto start = std::chrono::steady_clock::now();
  try {
    const auto payload = read_cache(
        cache_path(cache_directory_, "universal", universal_cache_key_),
        {CacheKind::Universal, expansion_basis_, expansion_order(), precision_,
         -1, universal_cache_key_, {}},
        static_plan_statistics_.cache_bytes_read);
    Reader reader(payload);
    for (int child = 0; child < 8; ++child) {
      m2m_operators_[child] = read_operator(reader, precision_);
    }
    for (int child = 0; child < 8; ++child) {
      l2l_operators_[child] = read_operator(reader, precision_);
    }
    m2l_plan_.matrices = read_values(reader, precision_);
    reader.require_end();
    const std::size_t expected = kUniversalClassCount *
        static_cast<std::size_t>(coefficient_count()) * coefficient_count();
    if (m2l_plan_.matrices.size() != expected) {
      throw std::runtime_error("universal M2L bank size mismatch");
    }
    static_plan_statistics_.universal_cache_hit = true;
    static_plan_statistics_.universal_cache_load.add(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count());
  } catch (const std::exception&) {
    static_plan_statistics_.universal_cache_lookup.add(
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count());
    return false;
  }
  static_plan_statistics_.universal_cache_lookup.add(
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count());

  if (!periodic_.enabled) {
    return true;
  }
  const auto periodic_start = std::chrono::steady_clock::now();
  try {
    const auto payload = read_cache(
        cache_path(cache_directory_, "periodic", periodic_cache_key_),
        {CacheKind::Periodic, expansion_basis_, expansion_order(), precision_,
         -1, periodic_cache_key_, {}},
        static_plan_statistics_.cache_bytes_read);
    Reader reader(payload);
    std::vector<double> periodic = read_values(reader, precision_);
    reader.require_end();
    const std::size_t expected =
        static_cast<std::size_t>(coefficient_count()) * coefficient_count();
    if (periodic.size() != expected) {
      throw std::runtime_error("periodic matrix size mismatch");
    }
    m2l_plan_.matrices.insert(m2l_plan_.matrices.end(), periodic.begin(),
                              periodic.end());
    static_plan_statistics_.periodic_cache_hit = true;
    periodic_operator_available_ = true;
    static_plan_statistics_.periodic_cache_load.add(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                     periodic_start)
            .count());
  } catch (const std::exception&) {
    static_plan_statistics_.periodic_cache_lookup.add(
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                     periodic_start)
            .count());
    return true;
  }
  static_plan_statistics_.periodic_cache_lookup.add(
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                   periodic_start)
          .count());
  return true;
}

void UniformFmm::write_universal_cache() const {
  if (!cache_enabled_) {
    return;
  }
  const auto start = std::chrono::steady_clock::now();
  Writer payload;
  for (const auto& value : m2m_operators_) {
    write_operator(payload, value, precision_);
  }
  for (const auto& value : l2l_operators_) {
    write_operator(payload, value, precision_);
  }
  const std::size_t matrix_values =
      static_cast<std::size_t>(coefficient_count()) * coefficient_count();
  const std::size_t universal_values = kUniversalClassCount * matrix_values;
  write_values(payload,
               std::span<const double>(m2l_plan_.matrices.data(), universal_values),
               precision_);
  const std::size_t bytes = write_cache(
      cache_path(cache_directory_, "universal", universal_cache_key_),
      {CacheKind::Universal, expansion_basis_, expansion_order(), precision_,
       -1, universal_cache_key_, {}},
      payload.bytes());
  const_cast<StaticPlanStatistics&>(static_plan_statistics_)
      .cache_bytes_written += bytes;

  if (periodic_.enabled && m2l_plan_.matrices.size() >=
                               universal_values + matrix_values) {
    Writer periodic_payload;
    write_values(periodic_payload,
                 std::span<const double>(m2l_plan_.matrices.data() + universal_values,
                                         matrix_values),
                 precision_);
    const std::size_t periodic_bytes = write_cache(
        cache_path(cache_directory_, "periodic", periodic_cache_key_),
        {CacheKind::Periodic, expansion_basis_, expansion_order(), precision_,
         -1, periodic_cache_key_, {}},
        periodic_payload.bytes());
    const_cast<StaticPlanStatistics&>(static_plan_statistics_)
        .cache_bytes_written += periodic_bytes;
  }
  const_cast<StaticPlanStatistics&>(static_plan_statistics_)
      .universal_cache_write.add(
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
              .count());
}

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
         tree_.leaf_level(), geometry_cache_key_, geometry_hash_digest_},
        static_plan_statistics_.cache_bytes_read);
    Reader reader(payload);
    const auto require_int_span = [&reader](const std::span<const int> expected) {
      if (!reader.equal_span<int>(expected)) {
        throw std::runtime_error("cached tree index metadata mismatch");
      }
    };
    require_int_span(tree_.source_permutation());
    require_int_span(tree_.source_inverse_permutation());
    require_int_span(tree_.target_permutation());
    require_int_span(tree_.target_inverse_permutation());
    require_int_span(tree_.leaf_indices());
    require_int_span(tree_.occupied_source_leaves());
    require_int_span(tree_.occupied_target_leaves());
    const auto node_count = reader.scalar<std::uint64_t>();
    const auto nodes = tree_.nodes();
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
    const auto p2m_count = reader.scalar<std::uint64_t>();
    if (precision_ == StaticPrecision::Float32) {
      p2m_plans_float_.clear();
      p2m_plans_float_.reserve(static_cast<std::size_t>(p2m_count));
      for (std::uint64_t index = 0; index < p2m_count; ++index) {
        p2m_plans_float_.push_back(
            {reader.scalar<int>(), read_operator_float(reader)});
      }
      m2l_plan_float_.coefficient_count = reader.scalar<int>();
      m2l_plan_float_.matrix_count = reader.scalar<int>();
      m2l_plan_float_.level_count = reader.scalar<int>();
      m2l_plan_float_.multipole_scaling = read_values_float(reader);
      m2l_plan_float_.local_scaling = read_values_float(reader);
      m2l_plan_float_.target_row_offsets = reader.vector<int>();
      m2l_plan_float_.source_nodes = reader.vector<int>();
      m2l_plan_float_.matrix_ids = reader.vector<int>();
      m2l_plan_float_.interaction_levels = reader.vector<int>();
      m2l_plan_float_.level_target_begin = reader.vector<int>();
      m2l_plan_float_.level_target_end = reader.vector<int>();

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
      geometry_cache_loaded_direct_float_ = true;
    } else {
      p2m_plans_.clear();
      p2m_plans_.reserve(static_cast<std::size_t>(p2m_count));
      for (std::uint64_t index = 0; index < p2m_count; ++index) {
        p2m_plans_.push_back(
            {reader.scalar<int>(), read_operator(reader, precision_)});
      }
      m2l_plan_.coefficient_count = reader.scalar<int>();
      m2l_plan_.matrix_count = reader.scalar<int>();
      m2l_plan_.level_count = reader.scalar<int>();
      m2l_plan_.multipole_scaling = read_values(reader, precision_);
      m2l_plan_.local_scaling = read_values(reader, precision_);
      m2l_plan_.target_row_offsets = reader.vector<int>();
      m2l_plan_.source_nodes = reader.vector<int>();
      m2l_plan_.matrix_ids = reader.vector<int>();
      m2l_plan_.interaction_levels = reader.vector<int>();
      m2l_plan_.level_target_begin = reader.vector<int>();
      m2l_plan_.level_target_end = reader.vector<int>();

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
  payload.span(tree_.source_permutation());
  payload.span(tree_.source_inverse_permutation());
  payload.span(tree_.target_permutation());
  payload.span(tree_.target_inverse_permutation());
  payload.span(tree_.leaf_indices());
  payload.span(tree_.occupied_source_leaves());
  payload.span(tree_.occupied_target_leaves());
  payload.scalar<std::uint64_t>(tree_.nodes().size());
  for (const TreeNode& node : tree_.nodes()) {
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
  payload.vector(m2l_plan_.interaction_levels);
  payload.vector(m2l_plan_.level_target_begin);
  payload.vector(m2l_plan_.level_target_end);
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
       tree_.leaf_level(), geometry_cache_key_, geometry_hash_digest_},
      payload.bytes());
  const_cast<StaticPlanStatistics&>(static_plan_statistics_)
      .cache_bytes_written += bytes;
  const_cast<StaticPlanStatistics&>(static_plan_statistics_)
      .geometry_cache_write.add(
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
              .count());
}

const std::string& UniformFmm::universal_cache_key() const noexcept {
  return universal_cache_key_;
}

const std::string& UniformFmm::geometry_cache_key() const noexcept {
  return geometry_cache_key_;
}

const std::string& UniformFmm::periodic_cache_key() const noexcept {
  return periodic_cache_key_;
}

} // namespace cdfmm
