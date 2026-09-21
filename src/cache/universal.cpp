// SPDX-License-Identifier: Apache-2.0

#include "cache/internal.hpp"
#include "phase_stopwatch.hpp"

#include <cstddef>
#include <exception>
#include <span>
#include <stdexcept>
#include <vector>

// Persistence of the depth-independent translation bank and of the periodic
// root operator. The two payloads are separate cache files with independent
// keys, but they are loaded and written in lockstep because the periodic
// matrix is stored as the tail of the same M2L matrix bank.
//
// This unit serialises and deserialises already-defined solver data. Cold
// construction of the translation mathematics stays in the operator and
// plan-preparation layers.
namespace cdfmm::detail::cache {

namespace {

// The bank always holds one matrix per possible M2L transfer class (316 for
// the complete interaction list); a periodic plan appends one more matrix,
// the periodic root operator, after these.
constexpr std::size_t kUniversalClassCount =
    StaticPlanStatistics::theoretical_maximum_m2l_classes;

} // namespace

// Payload: the eight M2M child operators, the eight L2L child operators, then
// the flat column-major bank of `kUniversalClassCount` M2L matrices.  A
// universal miss returns false so the caller rebuilds the bank; a periodic
// miss after a universal hit still returns true, with
// `periodic_operator_available` left false so only the periodic matrix is
// rebuilt.
bool load_universal_cache(const UniversalCacheIdentity& identity,
                          const int coefficient_count,
                          UniversalCachePayload payload,
                          StaticPlanStatistics& statistics) {
  if (!identity.enabled) {
    return false;
  }
  detail::PhaseStopwatch clock(
      statistics.timing_level == TimingLevel::Detailed);
  clock.start();
  try {
    const auto file = read_cache(
        cache_path(identity.directory, "universal", identity.universal_key),
        {CacheKind::Universal, identity.basis, identity.order,
         identity.precision, -1, identity.universal_key, {}},
        statistics.cache_bytes_read);
    Reader reader(file);
    for (int child = 0; child < 8; ++child) {
      payload.m2m_operators[static_cast<std::size_t>(child)] =
          read_operator(reader, identity.precision);
    }
    for (int child = 0; child < 8; ++child) {
      payload.l2l_operators[static_cast<std::size_t>(child)] =
          read_operator(reader, identity.precision);
    }
    payload.m2l_matrices = read_values(reader, identity.precision);
    reader.require_end();
    const std::size_t expected = kUniversalClassCount *
        static_cast<std::size_t>(coefficient_count) * coefficient_count;
    if (payload.m2l_matrices.size() != expected) {
      throw std::runtime_error("universal M2L bank size mismatch");
    }
    statistics.universal_cache_hit = true;
    if (clock.enabled()) {
      statistics.universal_cache_load.add(clock.elapsed());
    }
  } catch (const std::exception&) {
    if (clock.enabled()) {
      statistics.universal_cache_lookup.add(clock.elapsed());
    }
    return false;
  }
  if (clock.enabled()) {
    statistics.universal_cache_lookup.add(clock.elapsed());
  }

  if (!identity.periodic_enabled) {
    return true;
  }
  detail::PhaseStopwatch periodic_clock(
      statistics.timing_level == TimingLevel::Detailed);
  periodic_clock.start();
  try {
    const auto file = read_cache(
        cache_path(identity.directory, "periodic", identity.periodic_key),
        {CacheKind::Periodic, identity.basis, identity.order,
         identity.precision, -1, identity.periodic_key, {}},
        statistics.cache_bytes_read);
    Reader reader(file);
    std::vector<double> periodic = read_values(reader, identity.precision);
    reader.require_end();
    const std::size_t expected =
        static_cast<std::size_t>(coefficient_count) * coefficient_count;
    if (periodic.size() != expected) {
      throw std::runtime_error("periodic matrix size mismatch");
    }
    payload.m2l_matrices.insert(payload.m2l_matrices.end(), periodic.begin(),
                                periodic.end());
    statistics.periodic_cache_hit = true;
    payload.periodic_operator_available = true;
    if (periodic_clock.enabled()) {
      statistics.periodic_cache_load.add(periodic_clock.elapsed());
    }
  } catch (const std::exception&) {
    if (periodic_clock.enabled()) {
      statistics.periodic_cache_lookup.add(periodic_clock.elapsed());
    }
    return true;
  }
  if (periodic_clock.enabled()) {
    statistics.periodic_cache_lookup.add(periodic_clock.elapsed());
  }
  return true;
}

void write_universal_cache(
    const UniversalCacheIdentity& identity, const int coefficient_count,
    const std::array<StaticCoefficientOperator, 8>& m2m_operators,
    const std::array<StaticCoefficientOperator, 8>& l2l_operators,
    const std::vector<double>& m2l_matrices, StaticPlanStatistics& statistics) {
  if (!identity.enabled) {
    return;
  }
  detail::PhaseStopwatch clock(
      statistics.timing_level == TimingLevel::Detailed);
  clock.start();
  Writer payload;
  for (const auto& value : m2m_operators) {
    write_operator(payload, value, identity.precision);
  }
  for (const auto& value : l2l_operators) {
    write_operator(payload, value, identity.precision);
  }
  const std::size_t matrix_values =
      static_cast<std::size_t>(coefficient_count) * coefficient_count;
  const std::size_t universal_values = kUniversalClassCount * matrix_values;
  write_values(payload,
               std::span<const double>(m2l_matrices.data(), universal_values),
               identity.precision);
  const std::size_t bytes = write_cache(
      cache_path(identity.directory, "universal", identity.universal_key),
      {CacheKind::Universal, identity.basis, identity.order,
       identity.precision, -1, identity.universal_key, {}},
      payload.bytes());
  statistics.cache_bytes_written += bytes;

  if (identity.periodic_enabled &&
      m2l_matrices.size() >= universal_values + matrix_values) {
    Writer periodic_payload;
    write_values(periodic_payload,
                 std::span<const double>(m2l_matrices.data() + universal_values,
                                         matrix_values),
                 identity.precision);
    const std::size_t periodic_bytes = write_cache(
        cache_path(identity.directory, "periodic", identity.periodic_key),
        {CacheKind::Periodic, identity.basis, identity.order,
         identity.precision, -1, identity.periodic_key, {}},
        periodic_payload.bytes());
    statistics.cache_bytes_written += periodic_bytes;
  }
  if (clock.enabled()) {
    statistics.universal_cache_write.add(clock.elapsed());
  }
}

} // namespace cdfmm::detail::cache
