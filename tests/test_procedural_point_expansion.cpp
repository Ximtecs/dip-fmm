// SPDX-License-Identifier: Apache-2.0
//
// Procedural point P2M / L2P.  Recomputing the point-source expansion
// contribution and the point-target local evaluation from the positions
// during every evaluation must reproduce the precomputed coefficient rows on
// every static backend, in both precisions and at every compiled order, for
// the field and (on the CPU) the potential; the selection rules must reject
// what the executors cannot do and leave finite far-field models precomputed.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

#include <unistd.h>

#include "cdfmm/rectangular_prism.hpp"
#include "cdfmm/uniform_fmm.hpp"

using namespace cdfmm;

namespace {

struct Scene {
    std::vector<Vec3> positions{};
    std::vector<Vec3> moments{};
    std::vector<int> identities{};
};

Scene make_scene(const int count)
{
    Scene scene;
    std::uint64_t state = 0x2545F4914F6CDD1DULL;
    const auto next = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return -0.9 + 1.8 * static_cast<double>(state >> 11) /
            static_cast<double>(1ULL << 53);
    };
    // UniformFmm canonicalises normalised geometry to a 1e-9 grid of the
    // root side; placing the scene on that grid keeps every plan identical.
    const auto canonical = [](const double value) {
        return 2.0 * std::nearbyint(value * 0.5e9) / 1.0e9;
    };
    for (int index = 0; index < count; ++index) {
        scene.positions.push_back(
            {canonical(next()), canonical(next()), canonical(next())});
        const double value = static_cast<double>(index);
        scene.moments.push_back({std::sin(1.3 * value + 0.2),
                                 std::cos(0.9 * value - 0.4),
                                 std::sin(0.5 * value + 1.7)});
    }
    scene.identities.resize(scene.positions.size());
    std::iota(scene.identities.begin(), scene.identities.end(), 0);
    return scene;
}

UniformFmmOptions plan_options(const ExecutionBackend backend,
                               const StaticPrecision precision,
                               const int order,
                               const PointExpansionExecution execution)
{
    UniformFmmOptions options;
    options.backend = backend;
    options.precision = precision;
    options.expansion_order = order;
    options.tree.max_level = 2;
    options.tree.root_centre = Vec3{};
    options.tree.root_half_width = 1.0;
    options.point_expansion_execution = execution;
    options.enable_cache = false;
    return options;
}

std::vector<ExecutionBackend> available_backends()
{
    std::vector<ExecutionBackend> result{ExecutionBackend::CpuStatic};
    if (cuda_m2l_p2p_available()) {
        result.push_back(ExecutionBackend::CudaM2LP2P);
    }
    if (cuda_full_available()) {
        result.push_back(ExecutionBackend::CudaFull);
    }
    return result;
}

const char* name(const ExecutionBackend backend)
{
    switch (backend) {
    case ExecutionBackend::CpuStatic: return "CpuStatic";
    case ExecutionBackend::CudaM2LP2P: return "CudaPartial";
    case ExecutionBackend::CudaFull: return "CudaFull";
    default: return "?";
    }
}

double field_scale(const std::vector<PotentialField>& values)
{
    double scale = 0.0;
    for (const PotentialField& value : values) {
        scale = std::max({scale, std::abs(value.H.x), std::abs(value.H.y),
                          std::abs(value.H.z)});
    }
    return scale;
}

double maximum_field_difference(const std::vector<PotentialField>& a,
                                const std::vector<PotentialField>& b)
{
    REQUIRE(a.size() == b.size());
    double difference = 0.0;
    for (std::size_t index = 0; index < a.size(); ++index) {
        difference = std::max({difference, std::abs(a[index].H.x - b[index].H.x),
                               std::abs(a[index].H.y - b[index].H.y),
                               std::abs(a[index].H.z - b[index].H.z)});
    }
    return difference;
}

double maximum_potential_difference(const std::vector<PotentialField>& a,
                                    const std::vector<PotentialField>& b)
{
    double difference = 0.0;
    double scale = 0.0;
    for (std::size_t index = 0; index < a.size(); ++index) {
        difference = std::max(difference, std::abs(a[index].phi - b[index].phi));
        scale = std::max(scale, std::abs(b[index].phi));
    }
    REQUIRE(scale > 0.0);
    return difference / scale;
}

// FP64 executors differ by rounding only; FP32 plans quantise the stored
// rows once while the procedural path rounds every recurrence step, so their
// far fields differ at the single-precision level of the field scale.
double tolerance_for(const StaticPrecision precision)
{
    return precision == StaticPrecision::Float32 ? 5.0e-5 : 1.0e-11;
}

} // namespace

TEST_CASE("procedural point P2M and L2P reproduce the precomputed rows",
          "[far-field][procedural]")
{
    const Scene scene = make_scene(2000);
    for (const ExecutionBackend backend : available_backends()) {
        for (const StaticPrecision precision :
             {StaticPrecision::Float64, StaticPrecision::Float32}) {
            for (const int order : {1, 3, 6, 10}) {
                INFO(name(backend) << " "
                     << (precision == StaticPrecision::Float32 ? "fp32" : "fp64")
                     << " order " << order);
                UniformFmm precomputed(
                    scene.positions, scene.positions,
                    plan_options(backend, precision, order,
                                 PointExpansionExecution::Precomputed));
                UniformFmm procedural(
                    scene.positions, scene.positions,
                    plan_options(backend, precision, order,
                                 PointExpansionExecution::Procedural));
                REQUIRE(precomputed.p2m_execution() ==
                        PointExpansionExecution::Precomputed);
                REQUIRE(precomputed.l2p_execution() ==
                        PointExpansionExecution::Precomputed);
                REQUIRE(procedural.p2m_execution() ==
                        PointExpansionExecution::Procedural);
                REQUIRE(procedural.l2p_execution() ==
                        PointExpansionExecution::Procedural);
                REQUIRE(procedural.requested_point_expansion_execution() ==
                        PointExpansionExecution::Procedural);
                const auto expected = precomputed.evaluate(
                    scene.moments, OutputFlags::Field, scene.identities);
                const auto actual = procedural.evaluate(
                    scene.moments, OutputFlags::Field, scene.identities);
                const double scale = field_scale(expected);
                REQUIRE(scale > 0.0);
                const double difference =
                    maximum_field_difference(actual, expected);
                CAPTURE(difference / scale);
                REQUIRE(difference <= tolerance_for(precision) * scale);
                if (backend != ExecutionBackend::CudaFull) {
                    // The CPU hierarchy keeps only three small factor tables
                    // instead of 3 C values per source and per target.
                    const StaticPlanStatistics& lean =
                        procedural.static_plan_statistics();
                    const StaticPlanStatistics& rows =
                        precomputed.static_plan_statistics();
                    REQUIRE(lean.p2m_operator_bytes < rows.p2m_operator_bytes);
                    REQUIRE(lean.l2p_operator_bytes < rows.l2p_operator_bytes);
                }
            }
        }
    }
}

TEST_CASE("procedural point L2P reproduces the precomputed potential",
          "[far-field][procedural]")
{
    const Scene scene = make_scene(2000);
    for (const StaticPrecision precision :
         {StaticPrecision::Float64, StaticPrecision::Float32}) {
        INFO((precision == StaticPrecision::Float32 ? "fp32" : "fp64"));
        UniformFmm precomputed(
            scene.positions, scene.positions,
            plan_options(ExecutionBackend::CpuStatic, precision, 6,
                         PointExpansionExecution::Precomputed));
        UniformFmm procedural(
            scene.positions, scene.positions,
            plan_options(ExecutionBackend::CpuStatic, precision, 6,
                         PointExpansionExecution::Procedural));
        const auto expected = precomputed.evaluate(
            scene.moments, OutputFlags::Both, scene.identities);
        const auto actual = procedural.evaluate(
            scene.moments, OutputFlags::Both, scene.identities);
        const double scale = field_scale(expected);
        REQUIRE(maximum_field_difference(actual, expected) <=
                tolerance_for(precision) * scale);
        REQUIRE(maximum_potential_difference(actual, expected) <=
                tolerance_for(precision));
    }
}

TEST_CASE("procedural point expansion requests are validated",
          "[far-field][procedural]")
{
    const Scene scene = make_scene(500);

    // The Cartesian basis has no procedural executor.
    UniformFmmOptions cartesian = plan_options(
        ExecutionBackend::CpuStatic, StaticPrecision::Float64, 4,
        PointExpansionExecution::Procedural);
    cartesian.expansion_basis = ExpansionBasis::Cartesian;
    REQUIRE_THROWS_AS(UniformFmm(scene.positions, scene.positions, cartesian),
                      std::invalid_argument);

    // Orders above the compiled range keep the precomputed rows.
    REQUIRE_THROWS_AS(
        UniformFmm(scene.positions, scene.positions,
                   plan_options(ExecutionBackend::CpuStatic,
                                StaticPrecision::Float64, 11,
                                PointExpansionExecution::Procedural)),
        std::invalid_argument);

    // A finite far-field source model keeps its exact precomputed P2M rows
    // while the point-target L2P becomes procedural.
    UniformFmmOptions prism = plan_options(
        ExecutionBackend::CpuStatic, StaticPrecision::Float64, 4,
        PointExpansionExecution::Procedural);
    prism.source_geometry = SourceGeometry::RectangularPrism;
    prism.source_sizes = {RectangularPrism{0.02, 0.015, 0.01}};
    UniformFmm mixed(scene.positions, scene.positions, prism);
    REQUIRE(mixed.p2m_execution() == PointExpansionExecution::Precomputed);
    REQUIRE(mixed.l2p_execution() == PointExpansionExecution::Procedural);
    prism.point_expansion_execution = PointExpansionExecution::Precomputed;
    UniformFmm reference(scene.positions, scene.positions, prism);
    const auto expected =
        reference.evaluate(scene.moments, OutputFlags::Field, scene.identities);
    const auto actual =
        mixed.evaluate(scene.moments, OutputFlags::Field, scene.identities);
    REQUIRE(maximum_field_difference(actual, expected) <=
            1.0e-11 * field_scale(expected));

    // A finite far-field model on both sides has nothing to recompute.
    prism.point_expansion_execution = PointExpansionExecution::Procedural;
    prism.target_geometry = TargetGeometry::RectangularPrism;
    prism.target_sizes = prism.source_sizes;
    REQUIRE_THROWS_AS(UniformFmm(scene.positions, scene.positions, prism),
                      std::invalid_argument);

    // The default request resolves to a concrete choice per stage.
    UniformFmm automatic(scene.positions, scene.positions,
                         plan_options(ExecutionBackend::CpuStatic,
                                      StaticPrecision::Float64, 4,
                                      PointExpansionExecution::Auto));
    REQUIRE(automatic.requested_point_expansion_execution() ==
            PointExpansionExecution::Auto);
    REQUIRE(automatic.p2m_execution() != PointExpansionExecution::Auto);
    REQUIRE(automatic.l2p_execution() != PointExpansionExecution::Auto);
}

TEST_CASE("procedural point expansion agrees between cold and warm caches",
          "[far-field][procedural][cache]")
{
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        ("cdfmm-procedural-" + std::to_string(::getpid()));
    std::filesystem::create_directories(directory);
    ::setenv("CDFMM_CACHE_DIR", directory.c_str(), 1);
    const Scene scene = make_scene(1500);
    UniformFmmOptions options = plan_options(
        ExecutionBackend::CpuStatic, StaticPrecision::Float32, 6,
        PointExpansionExecution::Procedural);
    options.enable_cache = true;
    std::vector<PotentialField> cold;
    {
        UniformFmm fmm(scene.positions, scene.positions, options);
        cold = fmm.evaluate(scene.moments, OutputFlags::Field,
                            scene.identities);
    }
    UniformFmm warm(scene.positions, scene.positions, options);
    REQUIRE(warm.p2m_execution() == PointExpansionExecution::Procedural);
    const auto hot =
        warm.evaluate(scene.moments, OutputFlags::Field, scene.identities);
    REQUIRE(maximum_field_difference(hot, cold) == 0.0);
    ::unsetenv("CDFMM_CACHE_DIR");
    std::error_code error;
    std::filesystem::remove_all(directory, error);
}
