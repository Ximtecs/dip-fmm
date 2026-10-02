// SPDX-License-Identifier: Apache-2.0

#include "cache/internal.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <string_view>
#include <system_error>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// The cache file container: where cache files live, the header this unit
// writes and validates, and how a file is materialised atomically so
// concurrent writers may race safely. The container format is owned here
// because it cannot be separated from the read/write path; format.cpp owns the
// records inside the payload, which this unit treats as an opaque byte range.
namespace cdfmm::detail::cache {
namespace {

// Read-only private mapping of a whole cache file.  The payload is decoded
// straight out of the mapping, so nothing is copied until a record is built;
// the CachePayload owns the mapping and unmaps it on destruction.
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

} // namespace

// The root is resolved at every construction, so a test or a caller may
// change `CDFMM_CACHE_DIR` between plans.  The `v1` subdirectory versions the
// directory layout independently of the file schema, so a future layout can
// coexist with files an older installation still reads.
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

std::filesystem::path cache_path(const std::string& root,
                                 const std::string& category,
                                 const std::string& key) {
  return std::filesystem::path(root) / category / key;
}

// Container layout (all integers little-endian as written by this process):
//
//   magic "CDFMMC2\0"
//   u32 schema version, u32 operator version, u32 endian marker,
//   u32 sizeof(size_t), u32 kind, u32 basis, i32 order, u32 precision,
//   i32 depth, u32 checksum algorithm
//   length-prefixed key string, length-prefixed geometry-hash string
//   u64 payload offset, u64 payload size, u64 payload checksum
//   payload bytes to end of file
//
// Every header field is checked against what the caller expects; any mismatch
// throws and the caller treats the file as a rebuildable miss.
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

// Atomic publication: write the complete file under a unique temporary name,
// fsync it, rename it into place, then fsync the directory.  A reader can
// therefore only ever see a complete file, two processes building the same
// plan may race harmlessly (the loser removes its temporary and reports the
// winner's bytes), and any failure returns 0 bytes rather than throwing,
// because a failed cache write must never fail a construction.
void StreamingChecksum::block(const unsigned char* bytes) noexcept {
  const std::uint64_t x0 = load_unaligned<std::uint64_t>(bytes);
  const std::uint64_t x1 = load_unaligned<std::uint64_t>(bytes + 8);
  const std::uint64_t x2 = load_unaligned<std::uint64_t>(bytes + 16);
  const std::uint64_t x3 = load_unaligned<std::uint64_t>(bytes + 24);
  a_ += x0;
  b_ ^= std::rotl(x1, 17);
  c_ += x2;
  d_ ^= std::rotl(x3, 31);
  b_ += a_;
  d_ += c_;
}

// Full 32-byte blocks are consumed as they arrive; fewer than 32 trailing
// bytes wait in `carry_`, and `finish` treats them exactly as the one-shot
// checksum treats its tail.
void StreamingChecksum::update(const void* data, std::size_t size) noexcept {
  const auto* bytes = static_cast<const unsigned char*>(data);
  total_ += size;
  if (carry_size_ != 0) {
    const std::size_t take = std::min<std::size_t>(32 - carry_size_, size);
    std::memcpy(carry_ + carry_size_, bytes, take);
    carry_size_ += take;
    bytes += take;
    size -= take;
    if (carry_size_ < 32) {
      return;
    }
    block(carry_);
    carry_size_ = 0;
  }
  while (size >= 32) {
    block(bytes);
    bytes += 32;
    size -= 32;
  }
  if (size != 0) {
    std::memcpy(carry_, bytes, size);
    carry_size_ = size;
  }
}

std::uint64_t StreamingChecksum::finish() const noexcept {
  std::uint64_t a = a_;
  std::uint64_t b = b_;
  std::uint64_t c = c_;
  std::uint64_t d = d_;
  std::size_t offset = 0;
  while (offset + 8 <= carry_size_) {
    a += load_unaligned<std::uint64_t>(carry_ + offset);
    b += a;
    offset += 8;
  }
  if (offset != carry_size_) {
    std::uint64_t tail = 0;
    std::memcpy(&tail, carry_ + offset, carry_size_ - offset);
    c += tail;
    d += c;
  }
  return checksum_avalanche(a ^ std::rotl(b, 13) ^ std::rotl(c, 29) ^
                            std::rotl(d, 47) ^ total_);
}

std::uint64_t payload_checksum(const void* data, const std::size_t size) noexcept {
  return fast_checksum64(data, size);
}

struct CacheFileStream::State {
  std::filesystem::path path{};
  std::filesystem::path temporary{};
  CacheDescriptor descriptor{};
  std::uint64_t payload_offset{0};
  std::uint64_t payload_size{0};
  StreamingChecksum checksum{};
  int fd{-1};
  bool failed{false};
  bool committed{false};
};

namespace {

// The container header in front of the payload; its size depends only on the
// descriptor, so the payload can be written first at that offset.
[[nodiscard]] Writer container_header(const CacheDescriptor& descriptor,
                                      const std::uint64_t payload_size,
                                      const std::uint64_t checksum) {
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
  header.scalar<std::uint64_t>(payload_size);
  header.scalar(checksum);
  if (header.bytes().size() != payload_offset) {
    throw std::runtime_error("internal cache header size mismatch");
  }
  return header;
}

bool pwrite_all_fd(const int fd, const void* data, std::size_t bytes,
                   off_t offset) noexcept {
  const auto* input = static_cast<const unsigned char*>(data);
  while (bytes != 0) {
    const ssize_t count = ::pwrite(fd, input, bytes, offset);
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    input += count;
    bytes -= static_cast<std::size_t>(count);
    offset += count;
  }
  return true;
}

} // namespace

CacheFileStream::CacheFileStream(const std::filesystem::path& path,
                                 CacheDescriptor descriptor)
    : state_(std::make_unique<State>()) {
  State& state = *state_;
  state.path = path;
  state.descriptor = std::move(descriptor);
  state.payload_offset = container_header(state.descriptor, 0, 0).bytes().size();
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error) {
    state.failed = true;
    return;
  }
  static std::atomic<std::uint64_t> temporary_counter{0};
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::uint64_t unique =
      temporary_counter.fetch_add(1, std::memory_order_relaxed);
  state.temporary = path.string() + ".tmp." + std::to_string(::getpid()) + "." +
      std::to_string(stamp) + "." + std::to_string(unique);
  state.fd = ::open(state.temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
  if (state.fd < 0) {
    state.failed = true;
    return;
  }
  // The payload follows the header, which is written once its size and
  // checksum are known.
  if (::lseek(state.fd, static_cast<off_t>(state.payload_offset), SEEK_SET) < 0) {
    state.failed = true;
  }
}

CacheFileStream::~CacheFileStream() {
  State& state = *state_;
  if (state.fd >= 0) {
    (void)::close(state.fd);
  }
  if (!state.committed && !state.temporary.empty()) {
    std::error_code error;
    std::filesystem::remove(state.temporary, error);
  }
}

void CacheFileStream::append(const void* data, const std::size_t bytes) noexcept {
  State& state = *state_;
  if (state.failed || bytes == 0) {
    return;
  }
  if (!write_all_fd(state.fd, data, bytes)) {
    state.failed = true;
    return;
  }
  state.checksum.update(data, bytes);
  state.payload_size += bytes;
}

void CacheFileStream::append(const Writer& writer) noexcept {
  append(writer.bytes().data(), writer.bytes().size());
}

std::size_t CacheFileStream::commit() noexcept {
  State& state = *state_;
  if (state.failed || state.fd < 0) {
    return 0;
  }
  std::error_code error;
  try {
    const Writer header = container_header(state.descriptor, state.payload_size,
                                           state.checksum.finish());
    if (!pwrite_all_fd(state.fd, header.bytes().data(), header.bytes().size(), 0)) {
      return 0;
    }
  } catch (...) {
    return 0;
  }
  const bool fsync_ok = ::fsync(state.fd) == 0;
  const bool close_ok = ::close(state.fd) == 0;
  state.fd = -1;
  if (!fsync_ok || !close_ok) {
    return 0;
  }
  const std::size_t total = state.payload_offset + state.payload_size;
  std::filesystem::rename(state.temporary, state.path, error);
  if (error) {
    // Another process may have completed the same valid cache first.
    return std::filesystem::exists(state.path) ? total : 0;
  }
  state.committed = true;
  const int directory_fd = ::open(state.path.parent_path().c_str(), O_RDONLY);
  if (directory_fd >= 0) {
    (void)::fsync(directory_fd);
    (void)::close(directory_fd);
  }
  return total;
}

// Atomicity and concurrency: see CacheFileStream. A failed write returns 0
// bytes rather than throwing, because a failed cache write must never fail a
// construction.
std::size_t write_cache(const std::filesystem::path& path,
                        const CacheDescriptor& descriptor,
                        const std::vector<unsigned char>& payload) {
  CacheFileStream stream(path, descriptor);
  stream.append(payload.data(), payload.size());
  return stream.commit();
}

} // namespace cdfmm::detail::cache
