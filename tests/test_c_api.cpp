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
