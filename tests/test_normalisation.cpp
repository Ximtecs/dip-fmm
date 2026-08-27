// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <vector>

#include "cdfmm/uniform_fmm.hpp"

using namespace cdfmm;

namespace {

std::vector<Vec3> transform_positions(const std::vector<Vec3>& positions,
                                      const double scale,
                                      const Vec3& translation) {
  std::vector<Vec3> transformed;
  transformed.reserve(positions.size());
  for (const Vec3& position : positions) {
    transformed.push_back(scale * position + translation);
  }
  return transformed;
}

std::vector<Vec3> transform_moments(const std::vector<Vec3>& moments,
                                    const double scale) {
  std::vector<Vec3> transformed;
  transformed.reserve(moments.size());
  const double volume_scale = scale * scale * scale;
  for (const Vec3& moment : moments) {
    transformed.push_back(moment * volume_scale);
  }
  return transformed;
}

std::vector<CuboidSize> transform_sizes(const std::vector<CuboidSize>& sizes,
                                        const double scale) {
  std::vector<CuboidSize> transformed = sizes;
  for (CuboidSize& size : transformed) {
    size.hx *= scale;
    size.hy *= scale;
    size.hz *= scale;
  }
  return transformed;
}

std::vector<PotentialField> evaluate_plan(UniformFmm& plan,
                                          const std::vector<Vec3>& state,
                                          const OutputFlags output) {
  if (plan.precision() == StaticPrecision::Float64) {
    return plan.evaluate_float64(state, output);
  }
  const auto float_results = plan.evaluate_float32(state, output);
  std::vector<PotentialField> results;
  results.reserve(float_results.size());
  for (const FloatPotentialField& value : float_results) {
    results.push_back({static_cast<double>(value.phi),
                       {static_cast<double>(value.H.x),
                        static_cast<double>(value.H.y),
                        static_cast<double>(value.H.z)}});
  }
  return results;
}

void require_invariant(const std::vector<PotentialField>& reference,
                       const std::vector<PotentialField>& transformed,
                       const double scale,
                       const double tolerance) {
  REQUIRE(transformed.size() == reference.size());
  for (std::size_t index = 0; index < reference.size(); ++index) {
    const double field_norm = std::sqrt(dot(reference[index].H,
                                            reference[index].H));
    const double field_margin = tolerance * std::max(1.0, field_norm);
    REQUIRE(transformed[index].H.x ==
            Catch::Approx(reference[index].H.x).margin(field_margin));
    REQUIRE(transformed[index].H.y ==
            Catch::Approx(reference[index].H.y).margin(field_margin));
    REQUIRE(transformed[index].H.z ==
            Catch::Approx(reference[index].H.z).margin(field_margin));
    const double potential_margin =
        tolerance * std::max(1.0, std::abs(scale * reference[index].phi));
    REQUIRE(transformed[index].phi ==
            Catch::Approx(scale * reference[index].phi)
                .margin(potential_margin));
  }
}

const std::vector<Vec3> source_positions{
    {-0.43, -0.31, -0.27}, {0.37, -0.29, -0.21},
    {-0.35, 0.33, -0.19},  {0.39, 0.35, 0.29},
    {-0.08, 0.11, 0.38},   {0.13, -0.07, 0.09},
    {-0.22, 0.04, -0.02},  {0.06, 0.25, -0.34}};

const std::vector<Vec3> target_positions{
    {-0.38, 0.27, 0.34}, {0.31, -0.23, 0.32},
    {0.29, 0.28, -0.31}, {-0.27, -0.25, 0.24},
    {0.02, 0.03, -0.08}};

const std::vector<Vec3> moments{
    {0.7, -0.2, 0.1}, {-0.4, 0.8, 0.3}, {0.2, 0.1, -0.6},
    {-0.3, -0.5, 0.9}, {0.6, 0.4, -0.2}, {-0.1, 0.3, 0.5},
    {0.9, -0.7, 0.2}, {-0.5, 0.2, -0.4}};

} // namespace

TEST_CASE("FMM fields are invariant under translation and uniform scaling",
          "[normalisation]") {
  for (const ExpansionBasis basis : {ExpansionBasis::Cartesian,
                                     ExpansionBasis::Spherical}) {
    for (const StaticPrecision precision : {StaticPrecision::Float32,
                                            StaticPrecision::Float64}) {
      CAPTURE(basis, precision);
      UniformFmmOptions options;
      options.expansion_basis = basis;
      options.precision = precision;
      options.expansion_order = 5;
      options.tree.max_level = 2;
      UniformFmm reference_plan(source_positions, target_positions, options);
      const auto reference = evaluate_plan(
          reference_plan, moments, OutputFlags::Both);

      for (const double scale : {1.0e-3, 1.0, 1.0e3}) {
        const Vec3 translation{2.7, -1.9, 0.8};
        UniformFmm transformed_plan(
            transform_positions(source_positions, scale, translation),
            transform_positions(target_positions, scale, translation),
            options);
        REQUIRE(transformed_plan.geometry_cache_key() ==
                reference_plan.geometry_cache_key());
        const auto transformed = evaluate_plan(
            transformed_plan, transform_moments(moments, scale),
            OutputFlags::Both);
        require_invariant(reference, transformed, scale,
                          precision == StaticPrecision::Float32 ? 8.0e-5
                                                                : 2.0e-11);
      }
    }
  }
}

TEST_CASE("cuboid P2P and full FMM use complete scale-independent geometry",
          "[normalisation][cuboid]") {
  const std::vector<CuboidSize> source_sizes{{0.08, 0.11, 0.07}};
  const std::vector<CuboidSize> target_sizes{{0.06, 0.09, 0.05}};
  for (const int depth : {0, 2}) {
    CAPTURE(depth);
    UniformFmmOptions options;
    options.expansion_basis = ExpansionBasis::Spherical;
    options.precision = StaticPrecision::Float64;
    options.expansion_order = 6;
    options.tree.max_level = depth;
    options.source_geometry = SourceGeometry::UniformCuboid;
    options.target_geometry = TargetGeometry::VolumeAveragedCuboid;
    options.source_sizes = source_sizes;
    options.target_sizes = target_sizes;
    UniformFmm reference_plan(source_positions, target_positions, options);
    const auto reference = reference_plan.evaluate_float64(
        moments, OutputFlags::Both);

    for (const double scale : {1.0e-3, 1.0e3}) {
      UniformFmmOptions transformed_options = options;
      transformed_options.source_sizes = transform_sizes(source_sizes, scale);
      transformed_options.target_sizes = transform_sizes(target_sizes, scale);
      const Vec3 translation{-3.1, 0.7, 2.2};
      UniformFmm transformed_plan(
          transform_positions(source_positions, scale, translation),
          transform_positions(target_positions, scale, translation),
          transformed_options);
      REQUIRE(transformed_plan.geometry_cache_key() ==
              reference_plan.geometry_cache_key());
      const auto transformed = transformed_plan.evaluate_float64(
          transform_moments(moments, scale),
          OutputFlags::Both);
      require_invariant(reference, transformed, scale, 3.0e-10);
    }
  }
}

TEST_CASE("periodic normalisation follows the supplied physical cube",
          "[normalisation][periodic]") {
  UniformFmmOptions options;
  options.expansion_basis = ExpansionBasis::Spherical;
  options.precision = StaticPrecision::Float64;
  options.expansion_order = 5;
  options.tree.max_level = 2;
  options.periodic.enabled = true;
  options.periodic.centre = {0.1, -0.2, 0.3};
  options.periodic.lengths = {1.4, 1.4, 1.4};
  UniformFmm reference_plan(source_positions, target_positions, options);
  const auto reference = reference_plan.evaluate_float64(
      moments, OutputFlags::Both);

  for (const double scale : {1.0e-3, 1.0e3}) {
    const Vec3 translation{1.7, -2.4, 0.6};
    UniformFmmOptions transformed_options = options;
    transformed_options.periodic.centre =
        scale * options.periodic.centre + translation;
    transformed_options.periodic.lengths =
        options.periodic.lengths * scale;
    UniformFmm transformed_plan(
        transform_positions(source_positions, scale, translation),
        transform_positions(target_positions, scale, translation),
        transformed_options);
    REQUIRE(transformed_plan.geometry_cache_key() ==
            reference_plan.geometry_cache_key());
    REQUIRE(transformed_plan.periodic_cache_key() ==
            reference_plan.periodic_cache_key());
    const auto transformed = transformed_plan.evaluate_float64(
        transform_moments(moments, scale),
        OutputFlags::Both);
    require_invariant(reference, transformed, scale, 8.0e-10);
  }
}

TEST_CASE("CUDA normalisation preserves physical fields",
          "[normalisation][cuda]") {
  if (!cuda_available()) {
    SKIP("CUDA is unavailable");
  }
  for (const ExecutionBackend backend : {ExecutionBackend::CudaPartial,
                                         ExecutionBackend::CudaFull}) {
    for (const StaticPrecision precision : {StaticPrecision::Float32,
                                            StaticPrecision::Float64}) {
      CAPTURE(backend, precision);
      UniformFmmOptions options;
      options.backend = backend;
      options.precision = precision;
      options.expansion_basis = ExpansionBasis::Spherical;
      options.expansion_order = 5;
      options.tree.max_level = 2;
      UniformFmm reference_plan(source_positions, target_positions, options);
      const auto reference = evaluate_plan(
          reference_plan, moments, OutputFlags::Field);
      const double scale = 1.0e3;
      const Vec3 translation{2.0, -1.0, 0.5};
      UniformFmm transformed_plan(
          transform_positions(source_positions, scale, translation),
          transform_positions(target_positions, scale, translation), options);
      REQUIRE(transformed_plan.geometry_cache_key() ==
              reference_plan.geometry_cache_key());
      const auto transformed = evaluate_plan(
          transformed_plan, transform_moments(moments, scale),
          OutputFlags::Field);
      require_invariant(reference, transformed, scale,
                        precision == StaticPrecision::Float32 ? 1.0e-4
                                                              : 5.0e-11);
    }
  }
}
