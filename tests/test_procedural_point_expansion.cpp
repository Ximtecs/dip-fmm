// SPDX-License-Identifier: Apache-2.0
//
// Procedural point P2M / L2P.  Recomputing the point-source expansion
// contribution and the point-target local evaluation from the positions
// during every evaluation must reproduce the precomputed coefficient rows.
// The CPU kernel is checked against the canonical rows at every compiled
// order (1..15) on one leaf, which needs no plan and no operator bank; the
// complete plans are then compared on every static backend, in both
// precisions and at a low, a middle and a high order (the order-10 universal
// M2L bank alone costs over a minute to build cold, so the maximum-order
// boundary is the kernel test's job). The selection rules must reject what
// the executors cannot do and leave finite far-field models precomputed.

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

#include "cdfmm/math/spherical_harmonics.hpp"
#include "cdfmm/operators/l2p.hpp"
#include "cdfmm/operators/p2m.hpp"
#include "cdfmm/rectangular_prism.hpp"
#include "cdfmm/uniform_fmm.hpp"

#include "backend/cpu/far_field/procedural.hpp"

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

// These are reproduction checks between two executions of one operator, not
// convergence studies: the scene only has to populate every leaf of the
// depth-2 tree so that P2M, M2M, M2L, L2L and L2P all run. A few hundred
// points do that; more points multiply the cost without adding coverage.
constexpr int scene_size = 256;

} // namespace

TEST_CASE("procedural point P2M and L2P kernels reproduce the canonical rows "
          "at every compiled order",
          "[far-field][procedural]")
{
    using detail::cpu::ProceduralPointExpansion;
    // One leaf of eleven points: not a multiple of either SIMD pack width
    // (four FP64 or eight FP32 lanes), so the padded tail pack is exercised.
    const Scene scene = make_scene(11);
    const Vec3 centre{0.05, -0.03, 0.02};
    std::vector<FloatVec3> float_moments;
    std::vector<double> flat_moments;
    for (const Vec3& moment : scene.moments) {
        float_moments.push_back({static_cast<float>(moment.x),
                                 static_cast<float>(moment.y),
                                 static_cast<float>(moment.z)});
        flat_moments.insert(flat_moments.end(), {moment.x, moment.y, moment.z});
    }
    std::uint64_t state = 0x9E3779B97F4A7C15ULL;
    const auto next_local = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return -1.0 + 2.0 * static_cast<double>(state >> 11) /
            static_cast<double>(1ULL << 53);
    };

    for (int order = 1; order <= ProceduralPointExpansion<double>::max_order;
         ++order) {
        INFO("order " << order);
        const SphericalHarmonicBasis basis(order);
        const auto modes = static_cast<std::size_t>(basis.size());

        // P2M: the canonical sparse map applied to the flattened moments.
        const StaticCoefficientOperator rows =
            build_static_p2m_operator(basis, centre, scene.positions);
        std::vector<double> expected_M(modes, 0.0);
        for (const StaticOperatorEntry& entry : rows.entries) {
            expected_M[static_cast<std::size_t>(entry.output)] +=
                entry.value * flat_moments[static_cast<std::size_t>(entry.input)];
        }
        const double M_scale = *std::max_element(
            expected_M.begin(), expected_M.end(),
            [](const double a, const double b) {
                return std::abs(a) < std::abs(b);
            });
        REQUIRE(std::abs(M_scale) > 0.0);

        const ProceduralPointExpansion<double> fp64(order);
        std::vector<double> M(modes, 0.0);
        fp64.apply_p2m<Vec3>(centre, scene.positions, scene.moments, M.data());
        const ProceduralPointExpansion<float> fp32(order);
        std::vector<float> M_float(modes, 0.0F);
        fp32.apply_p2m<FloatVec3>(centre, scene.positions, float_moments,
                                  M_float.data());
        for (std::size_t mode = 0; mode < modes; ++mode) {
            REQUIRE(std::abs(M[mode] - expected_M[mode]) <=
                    tolerance_for(StaticPrecision::Float64) *
                        std::abs(M_scale));
            REQUIRE(std::abs(static_cast<double>(M_float[mode]) -
                             expected_M[mode]) <=
                    tolerance_for(StaticPrecision::Float32) *
                        std::abs(M_scale));
        }

        // L2P: the canonical potential and field rows of every point applied
        // to one local expansion with entries of order one.
        std::vector<double> L(modes);
        std::vector<float> L_float(modes);
        for (std::size_t mode = 0; mode < modes; ++mode) {
            L[mode] = next_local();
            L_float[mode] = static_cast<float>(L[mode]);
        }
        std::vector<PotentialField> expected(scene.positions.size());
        double field_scale = 0.0;
        double potential_scale = 0.0;
        for (std::size_t target = 0; target < scene.positions.size(); ++target) {
            const StaticL2PEvaluator evaluator = build_static_l2p_evaluator(
                basis, centre, scene.positions[target]);
            PotentialField& value = expected[target];
            for (std::size_t mode = 0; mode < modes; ++mode) {
                value.phi += evaluator.potential[mode] * L[mode];
                value.H.x += evaluator.field[0][mode] * L[mode];
                value.H.y += evaluator.field[1][mode] * L[mode];
                value.H.z += evaluator.field[2][mode] * L[mode];
            }
            field_scale = std::max({field_scale, std::abs(value.H.x),
                                    std::abs(value.H.y), std::abs(value.H.z)});
            potential_scale = std::max(potential_scale, std::abs(value.phi));
        }
        REQUIRE(field_scale > 0.0);
        REQUIRE(potential_scale > 0.0);

        std::vector<PotentialField> actual(scene.positions.size());
        fp64.apply_l2p<PotentialField>(centre, scene.positions, L.data(),
                                       actual, true, true);
        std::vector<FloatPotentialField> actual_float(scene.positions.size());
        fp32.apply_l2p<FloatPotentialField>(centre, scene.positions,
                                            L_float.data(), actual_float, true,
                                            true);
        for (std::size_t target = 0; target < scene.positions.size(); ++target) {
            const PotentialField& reference = expected[target];
            const double fp64_tolerance = tolerance_for(StaticPrecision::Float64);
            const double fp32_tolerance = tolerance_for(StaticPrecision::Float32);
            REQUIRE(std::abs(actual[target].H.x - reference.H.x) <=
                    fp64_tolerance * field_scale);
            REQUIRE(std::abs(actual[target].H.y - reference.H.y) <=
                    fp64_tolerance * field_scale);
            REQUIRE(std::abs(actual[target].H.z - reference.H.z) <=
                    fp64_tolerance * field_scale);
            REQUIRE(std::abs(actual[target].phi - reference.phi) <=
                    fp64_tolerance * potential_scale);
            REQUIRE(std::abs(actual_float[target].H.x - reference.H.x) <=
                    fp32_tolerance * field_scale);
            REQUIRE(std::abs(actual_float[target].H.y - reference.H.y) <=
                    fp32_tolerance * field_scale);
            REQUIRE(std::abs(actual_float[target].H.z - reference.H.z) <=
                    fp32_tolerance * field_scale);
            REQUIRE(std::abs(actual_float[target].phi - reference.phi) <=
                    fp32_tolerance * potential_scale);
        }
    }
}

TEST_CASE("procedural point P2M and L2P reproduce the precomputed rows",
          "[far-field][procedural]")
{
    const Scene scene = make_scene(scene_size);
    for (const ExecutionBackend backend : available_backends()) {
        // CudaFull is field-only; the CPU hierarchy (CpuStatic and the CPU
        // stages of CudaPartial) also evaluates the potential, so the
        // procedural L2P potential row is checked there.
        const OutputFlags output = backend == ExecutionBackend::CudaFull
            ? OutputFlags::Field : OutputFlags::Both;
        for (const StaticPrecision precision :
             {StaticPrecision::Float64, StaticPrecision::Float32}) {
            // Orders 1, 3 and 6 exercise the smallest, an odd and the
            // production expansion; the compiled maximum is covered by the
            // kernel test above without an order-10 operator bank.
            for (const int order : {1, 3, 6}) {
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
                    scene.moments, output, scene.identities);
                const auto actual = procedural.evaluate(
                    scene.moments, output, scene.identities);
                const double scale = field_scale(expected);
                REQUIRE(scale > 0.0);
                const double difference =
                    maximum_field_difference(actual, expected);
                CAPTURE(difference / scale);
                REQUIRE(difference <= tolerance_for(precision) * scale);
                if (output == OutputFlags::Both) {
                    REQUIRE(maximum_potential_difference(actual, expected) <=
                            tolerance_for(precision));
                }
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

TEST_CASE("procedural point expansion requests are validated",
          "[far-field][procedural]")
{
    const Scene scene = make_scene(scene_size / 2);

    // The Cartesian basis has no procedural executor.
    UniformFmmOptions cartesian = plan_options(
        ExecutionBackend::CpuStatic, StaticPrecision::Float64, 4,
        PointExpansionExecution::Procedural);
    cartesian.expansion_basis = ExpansionBasis::Cartesian;
    REQUIRE_THROWS_AS(UniformFmm(scene.positions, scene.positions, cartesian),
                      std::invalid_argument);

    // Orders above the compiled range (1..15) are rejected before the
    // operator bank is built, so this costs nothing.
    REQUIRE(detail::cpu::ProceduralPointExpansion<double>::max_order == 15);
    REQUIRE_THROWS_AS(
        UniformFmm(scene.positions, scene.positions,
                   plan_options(ExecutionBackend::CpuStatic,
                                StaticPrecision::Float64, 16,
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
    const Scene scene = make_scene(scene_size);
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
