// SPDX-License-Identifier: Apache-2.0

#include "cache/internal.hpp"
#include "phase_stopwatch.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Cache identity: the digest primitive, canonical coordinate representation,
// compact grid/permutation recognition, and the key strings that name a cache
// file. Identity is a compatibility surface, not an implementation detail.
//
// WARNING(cdfmm): every hash input, its order, its marker, and the digest
// representation are part of the on-disk contract. Equivalent physical
// geometry must keep producing the same key, and geometry that differs today
// must keep producing a different one.
namespace cdfmm::detail::cache {

namespace {

// A self-contained SHA-256 so that the geometry digest depends on no external
// library and is reproducible byte for byte across builds.  Only the digest
// matters here; it is a stable fingerprint, not a security primitive.
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

template <typename T> void hash_value(Sha256& hash, const T value) {
  hash.value(value);
}

// Coordinates enter the hash as integers on the same 1e-9 grid that
// construction snaps normalised geometry to, so the hash sees exactly the
// values the operators were built from and is insensitive to representation
// noise below that grid.
std::int64_t canonical_coordinate(const double value) {
  constexpr double resolution = 1.0e9;
  return static_cast<std::int64_t>(std::llround(value * resolution));
}

// A complete, duplicate-free lattice of positions is hashed as this compact
// descriptor instead of the full coordinate list.  The two encodings are
// distinguished by their markers, and the descriptor is used only when it
// reproduces the position set exactly, so equal keys still mean equal
// geometry.
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
  // Each axis must be an arithmetic progression, the axis counts must multiply
  // to the position count, and every lattice site must be occupied exactly
  // once; anything else falls back to the explicit coordinate list.
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

// The user-order permutation is part of the identity because the plan's
// result ordering depends on it.  For a lattice whose user order is a
// lexicographic sweep in some axis order, possibly with reversed axes, the
// permutation is recovered exactly from the grid descriptor and hashed as the
// (axis order, reversal mask) pair; any other ordering is hashed in full.
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

// The three keys of one plan.  The universal key names the depth-independent
// translation bank and depends only on basis, order and precision; the
// periodic key adds the zero-k0 convention and its setup tolerance; the
// geometry key names the complete static plan and carries a digest of
// everything the plan's operators depend on: the geometry models, the
// canonical positions and finite records, both permutations and the fixed
// identity map.  A plan built from a supplied topology is never cached, so it
// gets an empty identity.
CacheIdentity compute_cache_identity(const bool supplied_topology,
                                     const bool option_enable_cache,
                                     const CacheIdentityInputs& inputs,
                                     StaticPlanStatistics& statistics) {
  if (supplied_topology) {
    return {};
  }
  detail::PhaseStopwatch clock(
      statistics.timing_level == TimingLevel::Detailed);
  clock.start();
  CacheIdentity identity;
  identity.enabled = option_enable_cache && !environment_disables_cache();
  identity.directory = cache_root().string();
  std::ostringstream universal;
  universal << "operators_" << basis_name(inputs.expansion_basis) << "_p"
            << std::setw(2) << std::setfill('0') << inputs.expansion_order
            << '_' << precision_name(inputs.precision)
            << "_m2m-m2l-l2l_v02.bin";
  identity.universal_key = universal.str();

  if (inputs.periodic.enabled) {
    std::ostringstream periodic;
    periodic << "periodic_" << basis_name(inputs.expansion_basis) << "_p"
             << std::setw(2) << std::setfill('0') << inputs.expansion_order
             << '_' << precision_name(inputs.precision) << "_zerok0_tol"
             << cache_number(inputs.periodic.setup_tolerance)
             << "_v02.bin";
    identity.periodic_key = periodic.str();
  }

  Sha256 hash;
  hash_value(hash, kCacheSchemaVersion);
  hash_value(hash, kOperatorVersion);
  hash_value(hash, static_cast<std::uint32_t>(inputs.expansion_basis));
  hash_value(hash, static_cast<std::uint32_t>(inputs.precision));
  hash_value(hash, inputs.expansion_order);
  hash_value(hash, inputs.tree.leaf_level());
  hash_value(hash, static_cast<std::uint32_t>(inputs.source_geometry));
  hash_value(hash, static_cast<std::uint32_t>(inputs.target_geometry));
  hash_value(hash, static_cast<std::uint32_t>(inputs.near_field_source_model));
  hash_value(hash, static_cast<std::uint32_t>(inputs.near_field_target_model));
  hash_value(hash, static_cast<std::uint32_t>(inputs.far_field_source_model));
  hash_value(hash, static_cast<std::uint32_t>(inputs.far_field_target_model));
  // Derived P2P execution packing is part of the plan identity. The cached
  // canonical operator remains reusable, while the cache key prevents a
  // reduced-symmetry request from being reported as a default packing.
  hash_value(hash, inputs.use_reduced_symmetry_p2p);
  // A position-based near field persists no pair tensors. The marker is
  // hashed only in that case, so every stored-tensor key -- and therefore
  // every file an existing installation already holds -- is unchanged.
  if (inputs.position_based_p2p) {
    hash_value(hash, std::uint32_t{0x504f5347U}); // "POSG"
  }
  hash_value(hash, inputs.periodic.enabled);
  hash_value(hash, inputs.periodic.axes);
  hash_value(hash, static_cast<std::uint32_t>(inputs.periodic.convention));
  hash_value(hash, inputs.periodic.setup_tolerance);
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
  const auto source_grid = hash_positions(inputs.tree.sorted_source_positions());
  const auto target_grid = hash_positions(inputs.tree.sorted_target_positions());
  // Finite records: a single common record is hashed once under "SAME" so a
  // regular lattice of identical bodies keys compactly; otherwise every
  // record is hashed in sorted order.
  const auto hash_sizes = [&hash](const std::span<const CuboidSize> sizes) {
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
  const auto hash_tetrahedra = [&hash](
                                  const std::span<const Tetrahedron> tetrahedra) {
    hash_value(hash, static_cast<std::uint64_t>(tetrahedra.size()));
    // Keep a common record compact while retaining one-per-particle geometry
    // exactly. Coordinates are representative-relative and already in the
    // topology-normalised coordinate system at this point.
    if (!tetrahedra.empty() && std::all_of(
            tetrahedra.begin() + 1, tetrahedra.end(),
            [&tetrahedra](const Tetrahedron& tetrahedron) {
              for (std::size_t vertex = 0; vertex < 4; ++vertex) {
                const Vec3& lhs = tetrahedron.vertices[vertex];
                const Vec3& rhs = tetrahedra.front().vertices[vertex];
                if (lhs.x != rhs.x || lhs.y != rhs.y || lhs.z != rhs.z) {
                  return false;
                }
              }
              return true;
            })) {
      hash_value(hash, std::uint32_t{0x53414d45U}); // "SAME"
      for (const Vec3& vertex : tetrahedra.front().vertices) {
        hash_value(hash, canonical_coordinate(vertex.x));
        hash_value(hash, canonical_coordinate(vertex.y));
        hash_value(hash, canonical_coordinate(vertex.z));
      }
      return;
    }
    hash_value(hash, std::uint32_t{0x54455452U}); // "TETR"
    for (const Tetrahedron& tetrahedron : tetrahedra) {
      for (const Vec3& vertex : tetrahedron.vertices) {
        hash_value(hash, canonical_coordinate(vertex.x));
        hash_value(hash, canonical_coordinate(vertex.y));
        hash_value(hash, canonical_coordinate(vertex.z));
      }
    }
  };
  hash_sizes(inputs.sorted_source_sizes);
  hash_sizes(inputs.sorted_target_sizes);
  hash_tetrahedra(inputs.sorted_source_tetrahedra);
  hash_tetrahedra(inputs.sorted_target_tetrahedra);
  hash_permutation(hash, inputs.tree.source_permutation(),
                   inputs.tree.sorted_source_positions(), source_grid);
  hash_permutation(hash, inputs.tree.target_permutation(),
                   inputs.tree.sorted_target_positions(), target_grid);
  hash_value(hash, inputs.fixed_target_source_indices.has_value());
  if (inputs.fixed_target_source_indices) {
    for (const int value : *inputs.fixed_target_source_indices) {
      hash_value(hash, value);
    }
  }
  const std::string digest = hexadecimal(hash.finish());
  identity.geometry_hash_digest = digest;
  std::ostringstream plan;
  plan << "plan_" << basis_name(inputs.expansion_basis) << "_p"
       << std::setw(2) << std::setfill('0') << inputs.expansion_order << "_d"
       << std::setw(2) << inputs.tree.leaf_level() << '_'
       << precision_name(inputs.precision) << "_N_"
       << inputs.tree.sorted_source_positions().size() << "_p2p_"
       << (inputs.position_based_p2p
               ? "positions"
               : (inputs.use_reduced_symmetry_p2p ? "reduced_symmetry"
                                                  : "canonical"))
       << '_' << digest << "_v04.bin";
  identity.geometry_key = plan.str();
  if (clock.enabled()) {
    statistics.geometry_hash.add(clock.elapsed());
  }
  return identity;
}

} // namespace cdfmm::detail::cache
