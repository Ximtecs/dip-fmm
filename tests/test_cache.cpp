// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>
#include <unistd.h>

#include "cdfmm/uniform_fmm.hpp"

using namespace cdfmm;

namespace {

class TemporaryCache {
public:
  TemporaryCache() {
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
        ("cdfmm-test-" + std::to_string(::getpid()) + "-" +
         std::to_string(stamp));
    REQUIRE(::setenv("CDFMM_CACHE_DIR", path_.c_str(), 1) == 0);
    REQUIRE(::unsetenv("CDFMM_DISABLE_CACHE") == 0);
  }

  ~TemporaryCache() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
    ::unsetenv("CDFMM_CACHE_DIR");
  }

  [[nodiscard]] std::filesystem::path path() const { return path_; }

private:
  std::filesystem::path path_{};
};

const std::vector<Vec3> positions{
    {-0.42, -0.31, -0.22}, {0.37, -0.27, -0.18},
    {-0.34, 0.32, -0.16},  {0.38, 0.33, 0.28},
    {-0.07, 0.1, 0.36},    {0.12, -0.06, 0.08}};
const std::vector<Vec3> targets{
    {-0.36, 0.26, 0.31}, {0.3, -0.21, 0.3}, {0.27, 0.26, -0.29}};
const std::vector<Vec3> moments{
    {0.7, -0.2, 0.1}, {-0.4, 0.8, 0.3}, {0.2, 0.1, -0.6},
    {-0.3, -0.5, 0.9}, {0.6, 0.4, -0.2}, {-0.1, 0.3, 0.5}};

UniformFmmOptions cache_options() {
  UniformFmmOptions options;
  options.expansion_basis = ExpansionBasis::Cartesian;
  options.expansion_order = 3;
  options.precision = StaticPrecision::Float64;
  options.tree.max_level = 2;
  options.enable_cache = true;
  return options;
}

std::vector<Vec3> transform(const std::vector<Vec3>& input,
                            const double scale, const Vec3 translation) {
  std::vector<Vec3> result;
  result.reserve(input.size());
  for (const Vec3 value : input) {
    result.push_back(scale * value + translation);
  }
  return result;
}

void require_same_fields(const std::vector<PotentialField>& first,
                         const std::vector<PotentialField>& second) {
  REQUIRE(first.size() == second.size());
  for (std::size_t index = 0; index < first.size(); ++index) {
    REQUIRE(second[index].H.x == Catch::Approx(first[index].H.x).margin(1e-13));
    REQUIRE(second[index].H.y == Catch::Approx(first[index].H.y).margin(1e-13));
    REQUIRE(second[index].H.z == Catch::Approx(first[index].H.z).margin(1e-13));
  }
}

void require_identical_fields(
    const std::vector<FloatPotentialField>& first,
    const std::vector<FloatPotentialField>& second) {
  REQUIRE(first.size() == second.size());
  for (std::size_t index = 0; index < first.size(); ++index) {
    REQUIRE(second[index].phi == first[index].phi);
    REQUIRE(second[index].H.x == first[index].H.x);
    REQUIRE(second[index].H.y == first[index].H.y);
    REQUIRE(second[index].H.z == first[index].H.z);
  }
}

void require_close_fields(const std::vector<FloatPotentialField>& first,
                          const std::vector<FloatPotentialField>& second) {
  REQUIRE(first.size() == second.size());
  for (std::size_t index = 0; index < first.size(); ++index) {
    REQUIRE(second[index].H.x ==
            Catch::Approx(first[index].H.x).margin(2.0e-5));
    REQUIRE(second[index].H.y ==
            Catch::Approx(first[index].H.y).margin(2.0e-5));
    REQUIRE(second[index].H.z ==
            Catch::Approx(first[index].H.z).margin(2.0e-5));
  }
}

} // namespace

TEST_CASE("cold and warm binary caches preserve complete plan results",
          "[cache]") {
  TemporaryCache cache;
  const UniformFmmOptions options = cache_options();
  UniformFmm cold(positions, targets, options);
  const auto cold_result = cold.evaluate_float64(moments);
  REQUIRE_FALSE(cold.static_plan_statistics().universal_cache_hit);
  REQUIRE_FALSE(cold.static_plan_statistics().geometry_cache_hit);
  REQUIRE(cold.static_plan_statistics().cache_bytes_written > 0);

  UniformFmm warm(positions, targets, options);
  const auto warm_result = warm.evaluate_float64(moments);
  REQUIRE(warm.static_plan_statistics().universal_cache_hit);
  REQUIRE(warm.static_plan_statistics().geometry_cache_hit);
  REQUIRE(warm.static_plan_statistics().cache_bytes_read > 0);
  REQUIRE(warm.static_plan_statistics().p2m_plan.calls == 0);
  REQUIRE(warm.static_plan_statistics().m2m_plan.calls == 0);
  REQUIRE(warm.static_plan_statistics().m2l_plan.calls == 0);
  REQUIRE(warm.static_plan_statistics().l2l_plan.calls == 0);
  REQUIRE(warm.static_plan_statistics().l2p_plan.calls == 0);
  REQUIRE(warm.static_plan_statistics().p2p_tensor_plan.calls == 0);
  REQUIRE(warm.universal_cache_key() == cold.universal_cache_key());
  REQUIRE(warm.geometry_cache_key() == cold.geometry_cache_key());
  require_same_fields(cold_result, warm_result);
}

TEST_CASE("warm FP32 caches preserve the canonical plan exactly", "[cache]") {
  TemporaryCache cache;
  UniformFmmOptions options = cache_options();
  options.precision = StaticPrecision::Float32;

  UniformFmm cold(positions, targets, options);
  const auto cold_result = cold.evaluate_float32(moments, OutputFlags::Both);
  REQUIRE_FALSE(cold.static_plan_statistics().geometry_cache_hit);

  UniformFmm warm(positions, targets, options);
  const auto warm_result = warm.evaluate_float32(moments, OutputFlags::Both);
  REQUIRE(warm.static_plan_statistics().universal_cache_hit);
  REQUIRE(warm.static_plan_statistics().geometry_cache_hit);
  REQUIRE(warm.p2p_execution_packing() == P2PExecutionPacking::ParticleRowSoa);
  require_identical_fields(cold_result, warm_result);
}

TEST_CASE("one canonical cache plan serves portable and oneMKL executors",
          "[cache]") {
  if (!one_mkl_available()) {
    SKIP("oneMKL is unavailable");
  }
  TemporaryCache cache;
  UniformFmmOptions portable_options = cache_options();
  portable_options.precision = StaticPrecision::Float32;
  UniformFmm portable(positions, targets, portable_options);
  const auto expected = portable.evaluate_float32(moments);

  UniformFmmOptions mkl_options = portable_options;
  mkl_options.static_matrix_backend = StaticMatrixBackend::OneMkl;
  UniformFmm mkl(positions, targets, mkl_options);
  REQUIRE(mkl.geometry_cache_key() == portable.geometry_cache_key());
  REQUIRE(mkl.static_plan_statistics().geometry_cache_hit);
  REQUIRE(mkl.static_plan_statistics().backend_packing.calls == 2);
  require_close_fields(expected, mkl.evaluate_float32(moments));
}

TEST_CASE("cache keys separate universal mathematics from normalised geometry",
          "[cache]") {
  TemporaryCache cache;
  UniformFmmOptions options = cache_options();
  options.enable_cache = false;
  UniformFmm reference(positions, targets, options);

  const double scale = 1.0e-3;
  const Vec3 shift{2.4, -1.7, 0.6};
  UniformFmm transformed(transform(positions, scale, shift),
                         transform(targets, scale, shift), options);
  REQUIRE(transformed.geometry_cache_key() == reference.geometry_cache_key());
  REQUIRE(transformed.universal_cache_key() == reference.universal_cache_key());

  options.tree.max_level = 3;
  UniformFmm deeper(positions, targets, options);
  REQUIRE(deeper.geometry_cache_key() != reference.geometry_cache_key());
  REQUIRE(deeper.universal_cache_key() == reference.universal_cache_key());

  options = cache_options();
  options.enable_cache = false;
  options.expansion_order = 4;
  UniformFmm different_order(positions, targets, options);
  REQUIRE(different_order.universal_cache_key() !=
          reference.universal_cache_key());
  REQUIRE(different_order.geometry_cache_key() != reference.geometry_cache_key());

  options = cache_options();
  options.enable_cache = false;
  options.expansion_basis = ExpansionBasis::Spherical;
  UniformFmm different_basis(positions, targets, options);
  REQUIRE(different_basis.universal_cache_key() !=
          reference.universal_cache_key());

  options = cache_options();
  options.enable_cache = false;
  options.precision = StaticPrecision::Float32;
  UniformFmm different_precision(positions, targets, options);
  REQUIRE(different_precision.universal_cache_key() !=
          reference.universal_cache_key());
}

TEST_CASE("uniform grids use scale-independent compact cache descriptors",
          "[cache][grid]") {
  TemporaryCache cache;
  std::vector<Vec3> grid;
  for (int z = 0; z < 2; ++z) {
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 4; ++x) {
        grid.push_back({-0.3 + 0.2 * x, -0.25 + 0.25 * y,
                        -0.1 + 0.2 * z});
      }
    }
  }
  UniformFmmOptions options = cache_options();
  options.enable_cache = false;
  UniformFmm reference(grid, grid, options);
  const auto shifted = transform(grid, 1.0e-3, {2.7, -1.9, 0.8});
  UniformFmm transformed(shifted, shifted, options);
  REQUIRE(transformed.geometry_cache_key() == reference.geometry_cache_key());
}

TEST_CASE("geometry and periodic physics participate in cache keys",
          "[cache]") {
  TemporaryCache cache;
  UniformFmmOptions point_options = cache_options();
  point_options.enable_cache = false;
  UniformFmm point(positions, targets, point_options);

  UniformFmmOptions cuboid_options = point_options;
  cuboid_options.source_geometry = SourceGeometry::UniformCuboid;
  cuboid_options.source_sizes = {{0.05, 0.06, 0.07}};
  UniformFmm cuboid(positions, targets, cuboid_options);
  REQUIRE(cuboid.geometry_cache_key() != point.geometry_cache_key());

  cuboid_options.use_cuboid_p2m = false;
  UniformFmm point_far_field(positions, targets, cuboid_options);
  REQUIRE(point_far_field.geometry_cache_key() != cuboid.geometry_cache_key());

  UniformFmmOptions periodic_options = point_options;
  periodic_options.periodic.enabled = true;
  periodic_options.periodic.lengths = {1.2, 1.2, 1.2};
  UniformFmm periodic(positions, targets, periodic_options);
  periodic_options.periodic.setup_tolerance = 1.1e-12;
  UniformFmm nearby_tolerance(positions, targets, periodic_options);
  REQUIRE(nearby_tolerance.periodic_cache_key() !=
          periodic.periodic_cache_key());
  periodic_options.periodic.setup_tolerance = 1.0e-9;
  UniformFmm changed_tolerance(positions, targets, periodic_options);
  REQUIRE(changed_tolerance.periodic_cache_key() != periodic.periodic_cache_key());
  REQUIRE(changed_tolerance.geometry_cache_key() != periodic.geometry_cache_key());
}

TEST_CASE("corrupt and incompatible caches rebuild safely", "[cache]") {
  TemporaryCache cache;
  const UniformFmmOptions options = cache_options();
  UniformFmm cold(positions, targets, options);
  const auto reference = cold.evaluate_float64(moments);
  const auto universal_path = cache.path() / "v1" / "universal" /
      cold.universal_cache_key();

  {
    std::ofstream corrupt(universal_path, std::ios::binary | std::ios::trunc);
    corrupt << "truncated";
  }
  UniformFmm rebuilt(positions, targets, options);
  REQUIRE_FALSE(rebuilt.static_plan_statistics().universal_cache_hit);
  REQUIRE(rebuilt.static_plan_statistics().geometry_cache_hit);
  require_same_fields(reference, rebuilt.evaluate_float64(moments));

  {
    std::fstream incompatible(universal_path,
                              std::ios::binary | std::ios::in |
                                  std::ios::out);
    REQUIRE(incompatible.good());
    incompatible.seekp(8);
    const std::uint32_t incompatible_version = 999;
    incompatible.write(reinterpret_cast<const char*>(&incompatible_version),
                       sizeof(incompatible_version));
  }
  UniformFmm version_rebuilt(positions, targets, options);
  REQUIRE_FALSE(version_rebuilt.static_plan_statistics().universal_cache_hit);
  REQUIRE(version_rebuilt.static_plan_statistics().geometry_cache_hit);
  require_same_fields(reference, version_rebuilt.evaluate_float64(moments));
}

TEST_CASE("simultaneous cache creation leaves valid atomic files", "[cache]") {
  TemporaryCache cache;
  const UniformFmmOptions options = cache_options();
  std::array<std::vector<PotentialField>, 4> results;
  std::array<std::thread, 4> workers;
  for (std::size_t index = 0; index < workers.size(); ++index) {
    workers[index] = std::thread([&, index] {
      UniformFmm plan(positions, targets, options);
      results[index] = plan.evaluate_float64(moments);
    });
  }
  for (std::thread& worker : workers) {
    worker.join();
  }
  for (std::size_t index = 1; index < results.size(); ++index) {
    require_same_fields(results[0], results[index]);
  }
  UniformFmm warm(positions, targets, options);
  REQUIRE(warm.static_plan_statistics().universal_cache_hit);
  REQUIRE(warm.static_plan_statistics().geometry_cache_hit);
  require_same_fields(results[0], warm.evaluate_float64(moments));
}

TEST_CASE("cache and no-cache CPU paths agree", "[cache]") {
  TemporaryCache cache;
  UniformFmmOptions uncached_options = cache_options();
  uncached_options.enable_cache = false;
  UniformFmm uncached(positions, targets, uncached_options);
  const auto reference = uncached.evaluate_float64(moments);

  UniformFmmOptions cached_options = cache_options();
  UniformFmm cold(positions, targets, cached_options);
  UniformFmm warm(positions, targets, cached_options);
  require_same_fields(reference, cold.evaluate_float64(moments));
  require_same_fields(reference, warm.evaluate_float64(moments));
}

TEST_CASE("cache and no-cache CUDA-full paths agree", "[cache][cuda]") {
  if (!cuda_full_available()) {
    SKIP("CUDA full is unavailable");
  }
  TemporaryCache cache;
  UniformFmmOptions options = cache_options();
  options.expansion_basis = ExpansionBasis::Spherical;
  options.backend = ExecutionBackend::CudaFull;
  options.enable_cache = false;
  UniformFmm uncached(positions, targets, options);
  const auto reference = uncached.evaluate_float64(moments);

  options.enable_cache = true;
  UniformFmm cold(positions, targets, options);
  UniformFmm warm(positions, targets, options);
  REQUIRE(warm.static_plan_statistics().universal_cache_hit);
  REQUIRE(warm.static_plan_statistics().geometry_cache_hit);
  require_same_fields(reference, cold.evaluate_float64(moments));
  require_same_fields(reference, warm.evaluate_float64(moments));
}
