// SPDX-License-Identifier: Apache-2.0

#include "cache/internal.hpp"

#include <chrono>
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
namespace cdfmm {

using namespace detail::cache;

namespace {

constexpr std::size_t kUniversalClassCount =
    StaticPlanStatistics::theoretical_maximum_m2l_classes;

} // namespace

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

} // namespace cdfmm
