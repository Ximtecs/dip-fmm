// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>

#include "cdfmm/c_api.h"

TEST_CASE("C ABI creates, reuses, diagnoses, and destroys a point plan") {
    REQUIRE(cdfmm_abi_version() == CDFMM_ABI_VERSION);
    REQUIRE((cdfmm_one_mkl_available() == 0 ||
             cdfmm_one_mkl_available() == 1));
    cdfmm_options options{};
    cdfmm_default_options(&options);
    options.precision = CDFMM_PRECISION_FLOAT64;
    options.expansion_basis = CDFMM_BASIS_CARTESIAN;
    const double x[] = {0.0, 0.0};
    const double y[] = {0.0, 0.0};
    const double z[] = {0.0, 1.0};
    cdfmm_plan* plan = nullptr;
    REQUIRE(cdfmm_plan_create_same_points(2, x, y, z, &options, &plan) ==
            CDFMM_SUCCESS);
    const double mx[] = {0.0, 0.0};
    const double my[] = {0.0, 0.0};
    const double mz[] = {1.0, 0.0};
    double hx[2]{}, hy[2]{}, hz[2]{};
    REQUIRE(cdfmm_plan_evaluate_f64(plan, mx, my, mz, hx, hy, hz) ==
            CDFMM_SUCCESS);
    REQUIRE(hz[1] == Catch::Approx(1.0 / (2.0 * std::acos(-1.0))));
    cdfmm_plan_stats stats{};
    stats.struct_size = sizeof(stats);
    REQUIRE(cdfmm_plan_get_stats(plan, &stats) == CDFMM_SUCCESS);
    REQUIRE(stats.source_count == 2);
    REQUIRE(stats.host_persistent_bytes > 0);
    // Timing is off by default, so the wall time is unavailable rather than
    // a misleading zero; selecting a level makes it a measurement.
    double seconds = -1.0;
    REQUIRE(cdfmm_plan_get_last_evaluation_seconds(plan, &seconds) ==
            CDFMM_ERROR_UNSUPPORTED);
    REQUIRE(cdfmm_get_last_error()[0] != '\0');
    REQUIRE(cdfmm_plan_set_timing_level(plan, 7) ==
            CDFMM_ERROR_INVALID_ARGUMENT);
    REQUIRE(cdfmm_plan_set_timing_level(plan, CDFMM_TIMING_COARSE) ==
            CDFMM_SUCCESS);
    REQUIRE(cdfmm_plan_evaluate_f64(plan, mx, my, mz, hx, hy, hz) ==
            CDFMM_SUCCESS);
    REQUIRE(hz[1] == Catch::Approx(1.0 / (2.0 * std::acos(-1.0))));
    REQUIRE(cdfmm_plan_get_last_evaluation_seconds(plan, &seconds) ==
            CDFMM_SUCCESS);
    REQUIRE(seconds >= 0.0);
    // Detailed is a measurement too, and switching back to Off makes the
    // accessor unsupported again rather than replaying a stale value: the
    // v0.2 contract is that a C caller selects a level before reading time.
    REQUIRE(cdfmm_plan_set_timing_level(plan, CDFMM_TIMING_DETAILED) ==
            CDFMM_SUCCESS);
    REQUIRE(cdfmm_plan_evaluate_f64(plan, mx, my, mz, hx, hy, hz) ==
            CDFMM_SUCCESS);
    seconds = -1.0;
    REQUIRE(cdfmm_plan_get_last_evaluation_seconds(plan, &seconds) ==
            CDFMM_SUCCESS);
    REQUIRE(seconds >= 0.0);
    REQUIRE(cdfmm_plan_set_timing_level(plan, CDFMM_TIMING_OFF) ==
            CDFMM_SUCCESS);
    REQUIRE(cdfmm_plan_evaluate_f64(plan, mx, my, mz, hx, hy, hz) ==
            CDFMM_SUCCESS);
    REQUIRE(hz[1] == Catch::Approx(1.0 / (2.0 * std::acos(-1.0))));
    REQUIRE(cdfmm_plan_get_last_evaluation_seconds(plan, &seconds) ==
            CDFMM_ERROR_UNSUPPORTED);
    cdfmm_plan_destroy(plan);
    cdfmm_plan_destroy(nullptr);
}

TEST_CASE("C ABI reports errors and supports native FP32 calls") {
  REQUIRE(cdfmm_plan_evaluate_f32(nullptr, nullptr, nullptr, nullptr, nullptr,
                                  nullptr,
                                  nullptr) == CDFMM_ERROR_INVALID_ARGUMENT);
    REQUIRE(cdfmm_get_last_error()[0] != '\0');
    cdfmm_options options{};
    cdfmm_default_options(&options);
    const double x[] = {0.0, 0.0};
    const double y[] = {0.0, 0.0};
    const double z[] = {0.0, 1.0};
    cdfmm_plan* plan = nullptr;
    REQUIRE(cdfmm_plan_create_same_points(2, x, y, z, &options, &plan) == 0);
    const float zero[] = {0.0F, 0.0F};
    const float mz[] = {1.0F, 0.0F};
    float hx[2]{}, hy[2]{}, hz[2]{};
    REQUIRE(cdfmm_plan_evaluate_f32(plan, zero, zero, mz, hx, hy, hz) == 0);
    REQUIRE(hz[1] == Catch::Approx(1.0F / (2.0F * std::acos(-1.0F))).epsilon(1e-5));
    REQUIRE(cdfmm_plan_evaluate_f32(plan, zero, zero, mz, hx, hy, hz) == 0);
    cdfmm_plan_destroy(plan);
}

TEST_CASE("C ABI exposes finite cuboids for spherical and Cartesian bases")
{
    cdfmm_options options{};
    cdfmm_default_options(&options);
    options.precision = CDFMM_PRECISION_FLOAT64;
    const double coordinate[] = {0.0};
    cdfmm_plan* plan = nullptr;
    const int32_t identity[] = {0};
    REQUIRE(cdfmm_plan_create_uniform_cuboid_sources(
                1, coordinate, coordinate, coordinate, 1, coordinate,
                coordinate, coordinate, 1.0, 1.0, 1.0, identity, &options,
                &plan) == CDFMM_SUCCESS);
    const double zero[] = {0.0};
    const double mz[] = {1.0};
    double hx[1]{}, hy[1]{}, hz[1]{};
    REQUIRE(cdfmm_plan_evaluate_f64(plan, zero, zero, mz, hx, hy, hz) == 0);
    REQUIRE(hz[0] == Catch::Approx(-1.0 / 3.0));
    cdfmm_plan_destroy(plan);

    options.expansion_basis = CDFMM_BASIS_CARTESIAN;
    plan = nullptr;
    REQUIRE(cdfmm_plan_create_uniform_cuboid_sources(
                1, coordinate, coordinate, coordinate, 1, coordinate,
                coordinate, coordinate, 1.0, 1.0, 1.0, identity, &options,
                &plan) == CDFMM_SUCCESS);
    cdfmm_plan_destroy(plan);
}

TEST_CASE("C ABI same cuboids include finite volume-averaged self field") {
  cdfmm_options options{};
  cdfmm_default_options(&options);
  options.precision = CDFMM_PRECISION_FLOAT64;
  options.expansion_basis = CDFMM_BASIS_CARTESIAN;
  options.execution_backend = CDFMM_BACKEND_CPU_STATIC;
  const double coordinate[] = {0.0};
  cdfmm_plan *plan = nullptr;
  REQUIRE(cdfmm_plan_create_same_uniform_cuboids(
              1, coordinate, coordinate, coordinate, 2.0, 2.0, 2.0, &options,
              &plan) == CDFMM_SUCCESS);

  // Runtime inputs are total moments: m=V*M=8*(1,0,0).
  const double mx[] = {8.0};
  const double zero[] = {0.0};
  double hx[1]{}, hy[1]{}, hz[1]{};
  REQUIRE(cdfmm_plan_evaluate_f64(plan, mx, zero, zero, hx, hy, hz) ==
          CDFMM_SUCCESS);
  REQUIRE(hx[0] == Catch::Approx(-1.0 / 3.0).margin(2.0e-14));
  REQUIRE(hy[0] == Catch::Approx(0.0).margin(2.0e-14));
  REQUIRE(hz[0] == Catch::Approx(0.0).margin(2.0e-14));

  // The immutable plan is reusable for an independent magnetisation state.
  REQUIRE(cdfmm_plan_evaluate_f64(plan, zero, mx, zero, hx, hy, hz) ==
          CDFMM_SUCCESS);
  REQUIRE(hy[0] == Catch::Approx(-1.0 / 3.0).margin(2.0e-14));
  cdfmm_plan_destroy(plan);
}

TEST_CASE("C ABI creates fully periodic same-cuboid plans")
{
    cdfmm_options options{};
    cdfmm_default_options(&options);
    options.precision = CDFMM_PRECISION_FLOAT64;
    options.expansion_order = 4;
    options.tree_depth = 1;
    options.execution_backend = CDFMM_BACKEND_CPU_STATIC;

    const double coordinate[] = {0.0};
    const double cell_centre[] = {0.0, 0.0, 0.0};
    const double cell_lengths[] = {1.0, 1.0, 1.0};
    cdfmm_plan* plan = nullptr;
    REQUIRE(cdfmm_plan_create_same_uniform_cuboids_periodic(
                1, coordinate, coordinate, coordinate, 0.2, 0.2, 0.2,
                cell_centre, cell_lengths, 1.0e-12, &options, &plan) ==
            CDFMM_SUCCESS);

    const double zero[] = {0.0};
    const double mz[] = {1.0};
    double hx[1]{}, hy[1]{}, hz[1]{};
    REQUIRE(cdfmm_plan_evaluate_f64(plan, zero, zero, mz, hx, hy, hz) ==
            CDFMM_SUCCESS);
    REQUIRE(std::isfinite(hx[0]));
    REQUIRE(std::isfinite(hy[0]));
    REQUIRE(std::isfinite(hz[0]));
    cdfmm_plan_destroy(plan);

    const double non_cubic_lengths[] = {1.0, 2.0, 1.0};
    plan = nullptr;
    REQUIRE(cdfmm_plan_create_same_uniform_cuboids_periodic(
                1, coordinate, coordinate, coordinate, 0.2, 0.2, 0.2,
                cell_centre, non_cubic_lengths, 1.0e-12, &options, &plan) ==
            CDFMM_ERROR_INVALID_ARGUMENT);
    REQUIRE(plan == nullptr);
}

// ---------------------------------------------------------------------------
// Per-body prism sizes through the C ABI (uniform and adaptive trees)
// ---------------------------------------------------------------------------

#include <array>
#include <vector>

#include "cdfmm/plan/direct/dense.hpp"
#include "cdfmm/uniform_fmm.hpp"

namespace {

// Eight cubes of side 2 at (+-1, +-1, +-1), with the (+,+,+) cube replaced by
// its eight unit-cube children: a two-level octree mesh, 15 prisms in a
// 4 x 4 x 4 block centred on the origin.
struct GradedScene {
    std::vector<double> x, y, z, hx, hy, hz, mx, my, mz;
    std::vector<cdfmm::Vec3> positions;
    std::vector<cdfmm::CuboidSize> sizes;
    std::vector<cdfmm::Vec3> moments;
    void add(double px, double py, double pz, double side) {
        const std::size_t i = x.size();
        x.push_back(px); y.push_back(py); z.push_back(pz);
        hx.push_back(side); hy.push_back(side); hz.push_back(side);
        const double volume = side * side * side;
        mx.push_back(volume * (0.3 + 0.05 * i));
        my.push_back(volume * (-0.2 + 0.03 * i));
        mz.push_back(volume * (0.9 - 0.04 * i));
        positions.push_back({px, py, pz});
        sizes.push_back({side, side, side});
        moments.push_back({mx.back(), my.back(), mz.back()});
    }
};

GradedScene graded_scene() {
    GradedScene scene;
    for (double sx : {-1.0, 1.0}) for (double sy : {-1.0, 1.0}) for (double sz : {-1.0, 1.0}) {
        if (sx > 0 && sy > 0 && sz > 0) continue;
        scene.add(sx, sy, sz, 2.0);
    }
    for (double sx : {0.5, 1.5}) for (double sy : {0.5, 1.5}) for (double sz : {0.5, 1.5}) {
        scene.add(sx, sy, sz, 1.0);
    }
    return scene;
}

std::vector<cdfmm::Vec3> dense_reference(const GradedScene& scene) {
    const cdfmm::DenseDirectPlan dense(
        scene.positions, scene.positions, cdfmm::SourceGeometry::RectangularPrism,
        cdfmm::TargetGeometry::RectangularPrism, scene.sizes, scene.sizes, {},
        cdfmm::StaticPrecision::Float64);
    return dense.evaluate(scene.moments, cdfmm::DenseDirectBackend::Portable);
}

double relative_l2(const std::vector<cdfmm::Vec3>& reference,
                   const std::vector<double>& hx, const std::vector<double>& hy,
                   const std::vector<double>& hz) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double dx = hx[i] - reference[i].x, dy = hy[i] - reference[i].y,
                     dz = hz[i] - reference[i].z;
        num += dx * dx + dy * dy + dz * dz;
        den += reference[i].x * reference[i].x + reference[i].y * reference[i].y +
               reference[i].z * reference[i].z;
    }
    return std::sqrt(num / den);
}

cdfmm_options fp64_options(int order, int depth) {
    cdfmm_options options{};
    cdfmm_default_options(&options);
    options.precision = CDFMM_PRECISION_FLOAT64;
    options.expansion_order = order;
    options.tree_depth = depth;
    options.execution_backend = CDFMM_BACKEND_CPU_STATIC;
    return options;
}

} // namespace

TEST_CASE("C ABI per-body cuboids with equal sizes reproduce the uniform-size plan") {
    const cdfmm_options options = fp64_options(4, 1);
    const double x[] = {-0.5, 0.5, -0.5, 0.5};
    const double y[] = {-0.5, -0.5, 0.5, 0.5};
    const double z[] = {0.0, 0.0, 0.0, 0.0};
    const double h[] = {1.0, 1.0, 1.0, 1.0};
    cdfmm_plan *uniform = nullptr, *variable = nullptr;
    REQUIRE(cdfmm_plan_create_same_uniform_cuboids(4, x, y, z, 1.0, 1.0, 1.0, &options, &uniform) == CDFMM_SUCCESS);
    REQUIRE(cdfmm_plan_create_same_variable_cuboids(4, x, y, z, h, h, h, &options, &variable) == CDFMM_SUCCESS);
    const double mx[] = {1.0, -0.5, 0.25, 0.75}, my[] = {0.2, 0.4, -0.6, 0.1}, mz[] = {-0.3, 0.9, 0.5, -0.8};
    double ux[4]{}, uy[4]{}, uz[4]{}, vx[4]{}, vy[4]{}, vz[4]{};
    REQUIRE(cdfmm_plan_evaluate_f64(uniform, mx, my, mz, ux, uy, uz) == CDFMM_SUCCESS);
    REQUIRE(cdfmm_plan_evaluate_f64(variable, mx, my, mz, vx, vy, vz) == CDFMM_SUCCESS);
    for (int i = 0; i < 4; ++i) {
        REQUIRE(vx[i] == ux[i]);
        REQUIRE(vy[i] == uy[i]);
        REQUIRE(vz[i] == uz[i]);
    }
    cdfmm_plan_destroy(uniform);
    cdfmm_plan_destroy(variable);
}

TEST_CASE("C ABI per-body cuboids match the dense reference on a graded mesh") {
    const GradedScene scene = graded_scene();
    const auto reference = dense_reference(scene);
    const std::size_t n = scene.x.size();
    REQUIRE(n == 15);
    std::vector<double> hx(n), hy(n), hz(n);

    SECTION("uniform tree, leaves as large as the coarse tiles: exact near field") {
        const cdfmm_options options = fp64_options(8, 1);
        cdfmm_plan* plan = nullptr;
        REQUIRE(cdfmm_plan_create_same_variable_cuboids(
                    n, scene.x.data(), scene.y.data(), scene.z.data(), scene.hx.data(),
                    scene.hy.data(), scene.hz.data(), &options, &plan) == CDFMM_SUCCESS);
        REQUIRE(cdfmm_plan_evaluate_f64(plan, scene.mx.data(), scene.my.data(), scene.mz.data(),
                                        hx.data(), hy.data(), hz.data()) == CDFMM_SUCCESS);
        REQUIRE(relative_l2(reference, hx, hy, hz) < 1.0e-10);
        cdfmm_plan_destroy(plan);
    }
    SECTION("adaptive tree keeps every tile inside its leaf and converges with the order") {
        double previous = 1.0;
        for (const int order : {4, 8}) {
            const cdfmm_options options = fp64_options(order, 0);
            cdfmm_plan* plan = nullptr;
            REQUIRE(cdfmm_plan_create_adaptive_variable_cuboids(
                        n, scene.x.data(), scene.y.data(), scene.z.data(), scene.hx.data(),
                        scene.hy.data(), scene.hz.data(), /*capacity*/ 4, /*max_depth*/ 3,
                        nullptr, 0.0, &options, &plan) == CDFMM_SUCCESS);
            REQUIRE(cdfmm_plan_evaluate_f64(plan, scene.mx.data(), scene.my.data(), scene.mz.data(),
                                            hx.data(), hy.data(), hz.data()) == CDFMM_SUCCESS);
            const double error = relative_l2(reference, hx, hy, hz);
            REQUIRE(error < 1.0e-2);
            REQUIRE(error <= previous);
            previous = error;
            cdfmm_plan_destroy(plan);
        }
    }
    SECTION("an explicit root is honoured and a root that cuts a body is rejected") {
        const cdfmm_options options = fp64_options(4, 0);
        const double centre[3] = {0.0, 0.0, 0.0};
        cdfmm_plan* plan = nullptr;
        REQUIRE(cdfmm_plan_create_adaptive_variable_cuboids(
                    n, scene.x.data(), scene.y.data(), scene.z.data(), scene.hx.data(),
                    scene.hy.data(), scene.hz.data(), 4, 3, centre, 2.5, &options, &plan) == CDFMM_SUCCESS);
        cdfmm_plan_destroy(plan);
        REQUIRE(cdfmm_plan_create_adaptive_variable_cuboids(
                    n, scene.x.data(), scene.y.data(), scene.z.data(), scene.hx.data(),
                    scene.hy.data(), scene.hz.data(), 4, 3, centre, 1.5, &options, &plan) ==
                CDFMM_ERROR_INVALID_ARGUMENT);
        REQUIRE(plan == nullptr);
    }
}

TEST_CASE("C ABI per-body cuboids reject invalid sizes and adaptive controls") {
    const cdfmm_options options = fp64_options(4, 1);
    const double x[] = {0.0, 1.0}, h[] = {0.5, 0.5}, bad[] = {0.5, -0.5};
    cdfmm_plan* plan = nullptr;
    REQUIRE(cdfmm_plan_create_same_variable_cuboids(2, x, x, x, nullptr, h, h, &options, &plan) == CDFMM_ERROR_INVALID_ARGUMENT);
    REQUIRE(cdfmm_plan_create_same_variable_cuboids(2, x, x, x, h, bad, h, &options, &plan) == CDFMM_ERROR_INVALID_ARGUMENT);
    REQUIRE(cdfmm_plan_create_same_variable_cuboids(0, x, x, x, h, h, h, &options, &plan) == CDFMM_ERROR_INVALID_ARGUMENT);
    REQUIRE(cdfmm_plan_create_adaptive_variable_cuboids(2, x, x, x, h, h, h, 4, 9, nullptr, 0.0, &options, &plan) == CDFMM_ERROR_INVALID_ARGUMENT);
    REQUIRE(cdfmm_plan_create_adaptive_variable_cuboids(2, x, x, x, h, h, h, 0, 3, nullptr, 0.0, &options, &plan) == CDFMM_ERROR_INVALID_ARGUMENT);
    REQUIRE(plan == nullptr);
}
