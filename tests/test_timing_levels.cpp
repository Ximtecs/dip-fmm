// SPDX-License-Identifier: Apache-2.0
//
// The timing levels (TimingLevel::Off / Coarse / Detailed) are an accounting
// switch and nothing else: they decide which PhaseTiming fields a plan
// collects, and must never change results, the resolved execution policy,
// cache keys or persisted cache contents.  Wall-clock values are never
// asserted; only whether a field was collected (its call count) and the
// invariants between collected fields.

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include <unistd.h>

#include "cdfmm/backend/cuda/direct.hpp"
#include "cdfmm/uniform_fmm.hpp"

using namespace cdfmm;

namespace {

std::vector<Vec3> make_positions(const int count) {
  std::vector<Vec3> positions;
  positions.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    positions.push_back({
        -0.95 + 1.9 * static_cast<double>((index * 17) % 67) / 66.0,
        -0.95 + 1.9 * static_cast<double>((index * 29) % 71) / 70.0,
        -0.95 + 1.9 * static_cast<double>((index * 43) % 73) / 72.0,
    });
  }
  return positions;
}

std::vector<Vec3> make_moments(const int count) {
  std::vector<Vec3> moments;
  moments.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    const double value = static_cast<double>(index);
    moments.push_back({std::sin(value), std::cos(value), std::sin(0.25 * value)});
  }
  return moments;
}

std::vector<int> identity_map(const int count) {
  std::vector<int> identities(static_cast<std::size_t>(count));
  std::iota(identities.begin(), identities.end(), 0);
  return identities;
}

UniformFmmOptions base_options(const ExecutionBackend backend,
                               const StaticPrecision precision,
                               const TimingLevel level) {
  UniformFmmOptions options;
  options.expansion_order = 4;
  options.tree.max_level = 3;
  options.tree.root_centre = Vec3{};
  options.tree.root_half_width = 1.0;
  options.precision = precision;
  options.backend = backend;
  options.timing_level = level;
  options.enable_cache = false;
  return options;
}

constexpr int point_count = 200;

bool same_fields(const std::vector<PotentialField> &left,
                 const std::vector<PotentialField> &right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.size(); ++index) {
    if (left[index].H.x != right[index].H.x ||
        left[index].H.y != right[index].H.y ||
        left[index].H.z != right[index].H.z ||
        left[index].phi != right[index].phi) {
      return false;
    }
  }
  return true;
}

// The fields only TimingLevel::Detailed collects on a CPU plan.
bool detailed_host_fields_collected(const EvaluationTimings &timings) {
  return timings.moment_permutation.calls > 0 &&
         timings.multipole_reset.calls > 0 && timings.p2m.calls > 0 &&
         timings.m2m.calls > 0 && timings.local_reset.calls > 0 &&
         timings.l2l.calls > 0 && timings.m2l.calls > 0 &&
         timings.l2p.calls > 0 && timings.result_unpermutation.calls > 0;
}

bool detailed_host_fields_empty(const EvaluationTimings &timings) {
  return timings.moment_permutation.calls == 0 &&
         timings.multipole_reset.calls == 0 && timings.p2m.calls == 0 &&
         timings.m2m.calls == 0 && timings.local_reset.calls == 0 &&
         timings.l2l.calls == 0 && timings.m2l.calls == 0 &&
         timings.m2l_multiply.calls == 0 && timings.l2p.calls == 0 &&
         timings.result_unpermutation.calls == 0;
}

bool cuda_lanes_empty(const EvaluationTimings &timings) {
  return timings.cuda_h2d.calls == 0 && timings.cuda_kernel.calls == 0 &&
         timings.cuda_d2h.calls == 0 && timings.cuda_m2l_h2d.calls == 0 &&
         timings.cuda_m2l_d2h.calls == 0 && timings.cuda_p2p_h2d.calls == 0 &&
         timings.cuda_p2p_kernel.calls == 0 &&
         timings.cuda_p2p_d2h.calls == 0;
}

bool construction_subphases_empty(const StaticPlanStatistics &statistics) {
  return statistics.normalisation.calls == 0 &&
         statistics.tree_construction.calls == 0 &&
         statistics.topology_construction.calls == 0 &&
         statistics.p2m_plan.calls == 0 && statistics.m2m_plan.calls == 0 &&
         statistics.m2l_plan.calls == 0 && statistics.l2l_plan.calls == 0 &&
         statistics.l2p_plan.calls == 0 &&
         statistics.p2p_tensor_plan.calls == 0 &&
         statistics.p2p_derived_packing.calls == 0 &&
         statistics.far_field_packing.calls == 0 &&
         statistics.universal_operator_build.calls == 0 &&
         statistics.transfer_discovery.calls == 0 &&
         statistics.buffer_allocation.calls == 0;
}

class TemporaryCache {
public:
  TemporaryCache() {
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
        ("cdfmm-timing-" + std::to_string(::getpid()) + "-" +
         std::to_string(stamp));
    REQUIRE(::setenv("CDFMM_CACHE_DIR", path_.c_str(), 1) == 0);
    REQUIRE(::unsetenv("CDFMM_DISABLE_CACHE") == 0);
  }

  ~TemporaryCache() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
    ::unsetenv("CDFMM_CACHE_DIR");
  }

  // Every persisted file, by its path relative to the cache root, with its
  // complete contents.
  [[nodiscard]] std::map<std::string, std::string> contents() const {
    std::map<std::string, std::string> files;
    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(path_)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      std::ifstream stream(entry.path(), std::ios::binary);
      files[entry.path().lexically_relative(path_).string()] =
          std::string(std::istreambuf_iterator<char>(stream), {});
    }
    return files;
  }

private:
  std::filesystem::path path_{};
};

} // namespace

TEST_CASE("TimingLevel::Off is the default and collects nothing", "[timing]") {
  REQUIRE(UniformFmmOptions{}.timing_level == TimingLevel::Off);
  const auto positions = make_positions(point_count);
  const auto moments = make_moments(point_count);
  const auto identities = identity_map(point_count);
  UniformFmm plan(positions, positions,
                  base_options(ExecutionBackend::CpuStatic,
                               StaticPrecision::Float64, TimingLevel::Off));
  REQUIRE(plan.timing_level() == TimingLevel::Off);

  const StaticPlanStatistics &statistics = plan.static_plan_statistics();
  REQUIRE(statistics.timing_level == TimingLevel::Off);
  REQUIRE(statistics.total_setup.calls == 0);
  REQUIRE(statistics.total.calls == 0);
  REQUIRE(construction_subphases_empty(statistics));
  // The byte, count and policy statistics do not depend on the level.
  REQUIRE(statistics.operator_bytes > 0);
  REQUIRE(statistics.p2p_interactions > 0);
  REQUIRE(statistics.construction_count == 1);

  (void)plan.evaluate(moments, OutputFlags::Field, identities);
  const EvaluationTimings &last = plan.last_timings();
  REQUIRE(last.timing_level == TimingLevel::Off);
  REQUIRE(last.evaluations == 1);
  REQUIRE(last.total.calls == 0);
  REQUIRE(last.far_field.calls == 0);
  REQUIRE(last.p2p.calls == 0);
  REQUIRE(detailed_host_fields_empty(last));
  REQUIRE(cuda_lanes_empty(last));
  const EvaluationTimings &aggregate = plan.aggregate_timings();
  REQUIRE(aggregate.timing_level == TimingLevel::Off);
  REQUIRE(aggregate.evaluations == 1);
  REQUIRE(aggregate.total.calls == 0);
}

TEST_CASE("TimingLevel::Coarse populates only the coarse fields", "[timing]") {
  const auto positions = make_positions(point_count);
  const auto moments = make_moments(point_count);
  const auto identities = identity_map(point_count);
  UniformFmm plan(positions, positions,
                  base_options(ExecutionBackend::CpuStatic,
                               StaticPrecision::Float32, TimingLevel::Coarse));
  REQUIRE(plan.timing_level() == TimingLevel::Coarse);

  const StaticPlanStatistics &statistics = plan.static_plan_statistics();
  REQUIRE(statistics.timing_level == TimingLevel::Coarse);
  REQUIRE(statistics.total_setup.calls == 1);
  REQUIRE(statistics.total.calls == 1);
  REQUIRE(statistics.total_setup.total_seconds >= 0.0);
  REQUIRE(statistics.total_setup.total_seconds >=
          statistics.total.total_seconds);
  REQUIRE(construction_subphases_empty(statistics));

  (void)plan.evaluate(moments, OutputFlags::Field, identities);
  const EvaluationTimings &last = plan.last_timings();
  REQUIRE(last.timing_level == TimingLevel::Coarse);
  REQUIRE(last.total.calls == 1);
  REQUIRE(last.far_field.calls == 1);
  REQUIRE(last.p2p.calls == 1);
  REQUIRE(last.total.total_seconds >= 0.0);
  // On the CPU the branches are serial and inside the evaluation.
  REQUIRE(last.total.total_seconds >= last.far_field.total_seconds);
  REQUIRE(last.total.total_seconds >= last.p2p.total_seconds);
  REQUIRE(detailed_host_fields_empty(last));
  REQUIRE(cuda_lanes_empty(last));
  REQUIRE(last.cuda_p2p_wait.calls == 0);
}

TEST_CASE("TimingLevel::Detailed populates every host phase", "[timing]") {
  const auto positions = make_positions(point_count);
  const auto moments = make_moments(point_count);
  const auto identities = identity_map(point_count);
  UniformFmm plan(positions, positions,
                  base_options(ExecutionBackend::CpuStatic,
                               StaticPrecision::Float64,
                               TimingLevel::Detailed));
  const StaticPlanStatistics &statistics = plan.static_plan_statistics();
  REQUIRE(statistics.timing_level == TimingLevel::Detailed);
  REQUIRE(statistics.total_setup.calls == 1);
  REQUIRE(statistics.total.calls == 1);
  REQUIRE(statistics.topology_construction.calls == 1);
  REQUIRE(statistics.normalisation.calls == 1);
  REQUIRE(statistics.tree_construction.calls >= 1);
  REQUIRE(statistics.p2m_plan.calls == 1);
  REQUIRE(statistics.l2p_plan.calls == 1);
  REQUIRE(statistics.p2p_tensor_plan.calls == 1);
  REQUIRE(statistics.transfer_discovery.calls == 1);
  REQUIRE(statistics.far_field_packing.calls == 1);

  (void)plan.evaluate(moments, OutputFlags::Field, identities);
  const EvaluationTimings &last = plan.last_timings();
  REQUIRE(last.timing_level == TimingLevel::Detailed);
  REQUIRE(last.total.calls == 1);
  REQUIRE(last.far_field.calls == 1);
  REQUIRE(last.p2p.calls == 1);
  REQUIRE(detailed_host_fields_collected(last));
  REQUIRE(last.m2l_multiply.calls > 0);
  REQUIRE(cuda_lanes_empty(last));
  // Non-negative and nested: the detailed phases partition the far field.
  const double phases = last.moment_permutation.total_seconds +
      last.multipole_reset.total_seconds + last.p2m.total_seconds +
      last.m2m.total_seconds + last.local_reset.total_seconds +
      last.l2l.total_seconds + last.m2l.total_seconds +
      last.l2p.total_seconds;
  REQUIRE(phases >= 0.0);
  REQUIRE(last.far_field.total_seconds >= 0.0);
  REQUIRE(last.total.total_seconds >= last.far_field.total_seconds);
}

TEST_CASE("aggregate timings accumulate and reset_timings clears only them",
          "[timing]") {
  const auto positions = make_positions(point_count);
  const auto moments = make_moments(point_count);
  const auto identities = identity_map(point_count);
  UniformFmm plan(positions, positions,
                  base_options(ExecutionBackend::CpuStatic,
                               StaticPrecision::Float64,
                               TimingLevel::Detailed));
  std::vector<PotentialField> first;
  for (int repetition = 0; repetition < 3; ++repetition) {
    const auto result = plan.evaluate(moments, OutputFlags::Field, identities);
    if (repetition == 0) {
      first = result;
    } else {
      REQUIRE(same_fields(first, result));
    }
  }
  const EvaluationTimings &aggregate = plan.aggregate_timings();
  REQUIRE(aggregate.evaluations == 3);
  REQUIRE(aggregate.total.calls == 3);
  REQUIRE(aggregate.p2m.calls == 3 * plan.last_timings().p2m.calls);
  REQUIRE(aggregate.total.total_seconds >=
          plan.last_timings().total.total_seconds);

  const std::size_t operator_bytes =
      plan.static_plan_statistics().operator_bytes;
  const std::uint64_t setup_calls =
      plan.static_plan_statistics().total_setup.calls;
  plan.reset_timings();
  REQUIRE(plan.aggregate_timings().evaluations == 0);
  REQUIRE(plan.aggregate_timings().total.calls == 0);
  REQUIRE(plan.aggregate_timings().timing_level == TimingLevel::Detailed);
  // The last evaluation, the construction record and the results survive.
  REQUIRE(plan.last_timings().total.calls == 1);
  REQUIRE(plan.static_plan_statistics().operator_bytes == operator_bytes);
  REQUIRE(plan.static_plan_statistics().total_setup.calls == setup_calls);
  REQUIRE(same_fields(first,
                      plan.evaluate(moments, OutputFlags::Field, identities)));
  REQUIRE(plan.aggregate_timings().evaluations == 1);
}

TEST_CASE("changing the timing level at run time changes only accounting",
          "[timing]") {
  const auto positions = make_positions(point_count);
  const auto moments = make_moments(point_count);
  const auto identities = identity_map(point_count);
  UniformFmm plan(positions, positions,
                  base_options(ExecutionBackend::CpuStatic,
                               StaticPrecision::Float32, TimingLevel::Off));
  const auto reference = plan.evaluate(moments, OutputFlags::Field, identities);
  const StaticExecutionPlan executors = plan.execution_plan();
  const P2PExecutionPacking packing = plan.p2p_execution_packing();
  const PointExpansionExecution p2m = plan.p2m_execution();
  const PointExpansionExecution l2p = plan.l2p_execution();

  for (const TimingLevel level : {TimingLevel::Detailed, TimingLevel::Coarse,
                                  TimingLevel::Off, TimingLevel::Detailed}) {
    plan.set_timing_level(level);
    REQUIRE(plan.timing_level() == level);
    // A level change resets the aggregate so it never mixes levels.
    REQUIRE(plan.aggregate_timings().evaluations == 0);
    REQUIRE(plan.aggregate_timings().timing_level == level);
    const auto result = plan.evaluate(moments, OutputFlags::Field, identities);
    REQUIRE(same_fields(reference, result));
    REQUIRE(plan.last_timings().timing_level == level);
    REQUIRE(plan.execution_plan().p2m == executors.p2m);
    REQUIRE(plan.execution_plan().m2l == executors.m2l);
    REQUIRE(plan.execution_plan().p2p == executors.p2p);
    REQUIRE(plan.p2p_execution_packing() == packing);
    REQUIRE(plan.p2m_execution() == p2m);
    REQUIRE(plan.l2p_execution() == l2p);
    if (level == TimingLevel::Off) {
      REQUIRE(plan.last_timings().total.calls == 0);
    } else {
      REQUIRE(plan.last_timings().total.calls == 1);
    }
    REQUIRE(detailed_host_fields_collected(plan.last_timings()) ==
            (level == TimingLevel::Detailed));
  }
}

TEST_CASE("timing levels leave cache keys and persisted plans identical",
          "[timing][cache]") {
  const auto positions = make_positions(point_count);
  const auto moments = make_moments(point_count);
  const auto identities = identity_map(point_count);
  std::string universal_key;
  std::string geometry_key;
  std::map<std::string, std::string> off_files;
  std::vector<PotentialField> off_result;
  {
    TemporaryCache cache;
    UniformFmmOptions options = base_options(
        ExecutionBackend::CpuStatic, StaticPrecision::Float64,
        TimingLevel::Off);
    options.enable_cache = true;
    UniformFmm plan(positions, positions, options);
    REQUIRE_FALSE(plan.static_plan_statistics().geometry_cache_hit);
    REQUIRE(plan.static_plan_statistics().cache_bytes_written > 0);
    universal_key = plan.universal_cache_key();
    geometry_key = plan.geometry_cache_key();
    off_result = plan.evaluate(moments, OutputFlags::Field, identities);
    off_files = cache.contents();
  }
  REQUIRE_FALSE(off_files.empty());
  {
    TemporaryCache cache;
    UniformFmmOptions options = base_options(
        ExecutionBackend::CpuStatic, StaticPrecision::Float64,
        TimingLevel::Detailed);
    options.enable_cache = true;
    UniformFmm plan(positions, positions, options);
    REQUIRE(plan.universal_cache_key() == universal_key);
    REQUIRE(plan.geometry_cache_key() == geometry_key);
    REQUIRE(plan.static_plan_statistics().geometry_cache_write.calls == 1);
    REQUIRE(same_fields(off_result, plan.evaluate(moments, OutputFlags::Field,
                                                  identities)));
    REQUIRE(cache.contents() == off_files);

    // A warm load at yet another level reads the same files and agrees.
    UniformFmmOptions warm_options = options;
    warm_options.timing_level = TimingLevel::Coarse;
    UniformFmm warm(positions, positions, warm_options);
    REQUIRE(warm.static_plan_statistics().geometry_cache_hit);
    REQUIRE(warm.static_plan_statistics().timing_level == TimingLevel::Coarse);
    REQUIRE(warm.static_plan_statistics().geometry_cache_load.calls == 0);
    REQUIRE(warm.static_plan_statistics().total_setup.calls == 1);
    REQUIRE(same_fields(off_result, warm.evaluate(moments, OutputFlags::Field,
                                                  identities)));
  }
}

TEST_CASE("CUDA backends keep results across timing levels and gate the lanes",
          "[timing][cuda]") {
  if (!cuda_full_available() || !cuda_m2l_p2p_available()) {
    SUCCEED("CUDA backends are unavailable");
    return;
  }
  const auto positions = make_positions(point_count);
  const auto moments = make_moments(point_count);
  const auto identities = identity_map(point_count);
  UniformFmm cpu(positions, positions,
                 base_options(ExecutionBackend::CpuStatic,
                              StaticPrecision::Float64, TimingLevel::Off));
  const auto expected = cpu.evaluate(moments, OutputFlags::Field, identities);

  for (const ExecutionBackend backend :
       {ExecutionBackend::CudaFull, ExecutionBackend::CudaM2LP2P}) {
    for (const StaticPrecision precision :
         {StaticPrecision::Float64, StaticPrecision::Float32}) {
      UniformFmm plan(positions, positions,
                      base_options(backend, precision, TimingLevel::Off));
      const auto off = plan.evaluate(moments, OutputFlags::Field, identities);
      REQUIRE(plan.last_timings().total.calls == 0);
      REQUIRE(cuda_lanes_empty(plan.last_timings()));
      REQUIRE(plan.last_timings().cuda_p2p_wait.calls == 0);
      // A device kernel that accumulates with atomics is not bitwise
      // reproducible from one evaluation to the next.  The level must add no
      // difference beyond that: bitwise equality when a repeat is bitwise
      // equal, otherwise agreement to the backend's own precision.
      const auto repeat = plan.evaluate(moments, OutputFlags::Field, identities);
      const bool deterministic = same_fields(off, repeat);
      const double tolerance =
          precision == StaticPrecision::Float64 ? 1.0e-9 : 2.0e-3;
      const auto agrees = [&](const std::vector<PotentialField> &other) {
        if (deterministic) {
          return same_fields(off, other);
        }
        for (std::size_t index = 0; index < off.size(); ++index) {
          const double scale = std::max(
              1.0, std::abs(off[index].H.x) + std::abs(off[index].H.y) +
                       std::abs(off[index].H.z));
          if (std::abs(other[index].H.x - off[index].H.x) > tolerance * scale ||
              std::abs(other[index].H.y - off[index].H.y) > tolerance * scale ||
              std::abs(other[index].H.z - off[index].H.z) > tolerance * scale) {
            return false;
          }
        }
        return true;
      };

      plan.set_timing_level(TimingLevel::Coarse);
      const auto coarse =
          plan.evaluate(moments, OutputFlags::Field, identities);
      REQUIRE(agrees(coarse));
      REQUIRE(plan.last_timings().total.calls == 1);
      REQUIRE(cuda_lanes_empty(plan.last_timings()));
      if (backend == ExecutionBackend::CudaM2LP2P) {
        // The host wait for the device near field is a coarse host clock.
        REQUIRE(plan.last_timings().cuda_p2p_wait.calls == 1);
        REQUIRE(plan.last_timings().far_field.calls == 1);
      } else {
        REQUIRE(plan.last_timings().far_field.calls == 0);
      }

      plan.set_timing_level(TimingLevel::Detailed);
      const auto detailed =
          plan.evaluate(moments, OutputFlags::Field, identities);
      REQUIRE(agrees(detailed));
      REQUIRE(plan.last_timings().total.calls == 1);
      REQUIRE(plan.last_timings().cuda_p2p_kernel.calls == 1);
      REQUIRE(plan.last_timings().cuda_p2p_kernel.total_seconds >= 0.0);
      if (backend == ExecutionBackend::CudaFull) {
        REQUIRE(plan.last_timings().p2m.calls == 1);
        REQUIRE(plan.last_timings().m2l.calls == 1);
        REQUIRE(plan.last_timings().l2p.calls == 1);
        REQUIRE(plan.last_timings().cuda_d2h.calls == 1);
      } else {
        REQUIRE(plan.last_timings().cuda_p2p_wait.calls == 1);
        REQUIRE(plan.last_timings().m2l_scale.calls == 1);
        REQUIRE(plan.last_timings().cuda_m2l_h2d.calls == 1);
      }

      plan.set_timing_level(TimingLevel::Off);
      REQUIRE(agrees(plan.evaluate(moments, OutputFlags::Field, identities)));
      REQUIRE(cuda_lanes_empty(plan.last_timings()));

      // Every level agrees with the CPU reference to the backend's precision.
      for (std::size_t index = 0; index < expected.size(); ++index) {
        const double scale =
            std::max(1.0, std::abs(expected[index].H.x) +
                              std::abs(expected[index].H.y) +
                              std::abs(expected[index].H.z));
        REQUIRE(std::abs(off[index].H.x - expected[index].H.x) <=
                tolerance * scale);
        REQUIRE(std::abs(off[index].H.y - expected[index].H.y) <=
                tolerance * scale);
        REQUIRE(std::abs(off[index].H.z - expected[index].H.z) <=
                tolerance * scale);
      }
    }
  }
}

TEST_CASE("the CUDA direct plan records device lanes only when detailed",
          "[timing][cuda]") {
  if (!cuda_direct_available()) {
    SUCCEED("CUDA direct execution is unavailable");
    return;
  }
  const auto positions = make_positions(64);
  const auto moments = make_moments(64);
  const auto identities = identity_map(64);
  CudaDirectPlan plan(positions, positions, identities);
  REQUIRE(plan.timing_level() == TimingLevel::Off);
  std::vector<PotentialField> off(positions.size());
  plan.evaluate(moments, off, OutputFlags::Field);
  REQUIRE(plan.evaluation_timings().timing_level == TimingLevel::Off);
  REQUIRE(plan.evaluation_timings().kernel_seconds == 0.0);
  REQUIRE(plan.evaluation_timings().total_seconds == 0.0);

  plan.set_timing_level(TimingLevel::Detailed);
  std::vector<PotentialField> detailed(positions.size());
  plan.evaluate(moments, detailed, OutputFlags::Field);
  REQUIRE(same_fields(off, detailed));
  REQUIRE(plan.evaluation_timings().timing_level == TimingLevel::Detailed);
  REQUIRE(plan.evaluation_timings().kernel_seconds >= 0.0);
  REQUIRE(plan.evaluation_timings().total_seconds >=
          plan.evaluation_timings().kernel_seconds);
}
