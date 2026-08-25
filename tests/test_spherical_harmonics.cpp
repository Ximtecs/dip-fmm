// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numeric>
#include <numbers>
#include <random>
#include <vector>

#include "cdfmm/operators.hpp"
#include "cdfmm/spherical_harmonics.hpp"
#include "cdfmm/static_operators.hpp"
#include "cdfmm/uniform_fmm.hpp"
#include "cdfmm/validation.hpp"

using namespace cdfmm;

namespace {

std::vector<double> flattened_moments(const std::vector<Vec3>& moments)
{
  std::vector<double> result;
  result.reserve(3 * moments.size());
  for (const Vec3 moment : moments) {
    result.push_back(moment.x);
    result.push_back(moment.y);
    result.push_back(moment.z);
  }
  return result;
}

PotentialField evaluate_spherical_multipole(
    const SphericalHarmonicBasis& basis, const std::span<const double> M,
    const Vec3& centre, const Vec3& target)
{
  const SolidHarmonicValues irregular =
      irregular_solid_harmonics(basis, target - centre);
  PotentialField result;
  for (int mode = 0; mode < basis.size(); ++mode) {
    const double coefficient = M[static_cast<std::size_t>(mode)];
    result.phi += coefficient * irregular.values[static_cast<std::size_t>(mode)];
    result.H = result.H -
        irregular.gradients[static_cast<std::size_t>(mode)] * coefficient;
  }
  return result;
}

double relative_field_error(const std::span<const PotentialField> actual,
                            const std::span<const PotentialField> expected)
{
  double numerator = 0.0;
  double denominator = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const Vec3 difference = actual[index].H - expected[index].H;
    numerator += dot(difference, difference);
    denominator += dot(expected[index].H, expected[index].H);
  }
  return std::sqrt(numerator / denominator);
}

} // namespace

TEST_CASE("real spherical modes have the documented normalisation")
{
  REQUIRE(UniformFmmOptions{}.expansion_basis == ExpansionBasis::Spherical);

  const SphericalHarmonicBasis basis(3);
  REQUIRE(basis.size() == 16);
  REQUIRE(basis.index(0, 0) == 0);
  REQUIRE(basis.index(1, -1) == 1);
  REQUIRE(basis.index(1, 0) == 2);
  REQUIRE(basis.index(1, 1) == 3);

  const Vec3 r{0.3, -0.4, 0.5};
  const SolidHarmonicValues regular = regular_solid_harmonics(basis, r);
  REQUIRE(regular.values[0] == Catch::Approx(1.0).margin(1.0e-15));
  REQUIRE(regular.values[1] == Catch::Approx(r.y).margin(1.0e-15));
  REQUIRE(regular.values[2] == Catch::Approx(r.z).margin(1.0e-15));
  REQUIRE(regular.values[3] == Catch::Approx(r.x).margin(1.0e-15));
  REQUIRE(regular.gradients[1].y == Catch::Approx(1.0).margin(1.0e-15));
  REQUIRE(regular.gradients[2].z == Catch::Approx(1.0).margin(1.0e-15));
  REQUIRE(regular.gradients[3].x == Catch::Approx(1.0).margin(1.0e-15));

  const SolidHarmonicValues irregular = irregular_solid_harmonics(basis, r);
  REQUIRE(irregular.values[0] ==
          Catch::Approx(1.0 / std::sqrt(dot(r, r))).epsilon(2.0e-14));
}

TEST_CASE("spherical point-dipole P2M converges in the far field")
{
  const std::vector<Vec3> sources{{-0.2, 0.1, 0.05},
                                  {0.15, -0.12, 0.08},
                                  {0.04, 0.18, -0.16}};
  const std::vector<Vec3> moments{{0.7, -0.2, 0.1},
                                  {-0.3, 0.6, 0.25},
                                  {0.2, 0.15, -0.5}};
  const Vec3 target{3.0, -2.0, 4.0};
  const PotentialField exact =
      p2p_dipole_sum(target, sources, moments, OutputFlags::Both);
  double previous_error = 1.0;
  for (const int order : {1, 2, 3, 4, 6, 8}) {
    const SphericalHarmonicBasis basis(order);
    const StaticCoefficientOperator p2m =
        build_static_p2m_operator(basis, Vec3{}, sources);
    std::vector<double> M(static_cast<std::size_t>(basis.size()), 0.0);
    const std::vector<double> inputs = flattened_moments(moments);
    apply_static_operator(p2m, inputs, M);
    const PotentialField actual =
        evaluate_spherical_multipole(basis, M, Vec3{}, target);
    const double error = norm(actual.H - exact.H) / norm(exact.H);
    CAPTURE(order, error, previous_error);
    REQUIRE(error < previous_error);
    previous_error = error;
  }
  REQUIRE(previous_error < 1.0e-9);
}

TEST_CASE("spherical M2M and L2L preserve their represented expansions")
{
  const SphericalHarmonicBasis basis(5);
  const Vec3 child_centre{0.15, -0.1, 0.2};
  const std::vector<Vec3> sources{{0.12, -0.08, 0.19},
                                  {0.18, -0.13, 0.24}};
  const std::vector<Vec3> moments{{0.7, -0.2, 0.1}, {-0.3, 0.6, 0.25}};
  const auto inputs = flattened_moments(moments);
  std::vector<double> child(static_cast<std::size_t>(basis.size()), 0.0);
  std::vector<double> parent(static_cast<std::size_t>(basis.size()), 0.0);
  std::vector<double> direct_parent(static_cast<std::size_t>(basis.size()), 0.0);
  apply_static_operator(build_static_p2m_operator(basis, child_centre, sources),
                        inputs, child);
  apply_static_operator(build_static_m2m_operator(basis, Vec3{} - child_centre),
                        child, parent);
  apply_static_operator(build_static_p2m_operator(basis, Vec3{}, sources),
                        inputs, direct_parent);
  for (std::size_t index = 0; index < parent.size(); ++index) {
    REQUIRE(parent[index] ==
            Catch::Approx(direct_parent[index]).margin(2.0e-12));
  }

  std::vector<double> parent_local(static_cast<std::size_t>(basis.size()));
  for (std::size_t index = 0; index < parent_local.size(); ++index) {
    parent_local[index] = std::sin(0.3 * static_cast<double>(index + 1));
  }
  std::vector<double> child_local(static_cast<std::size_t>(basis.size()), 0.0);
  apply_static_operator(build_static_l2l_operator(basis, child_centre),
                        parent_local, child_local);
  const Vec3 target = child_centre + Vec3{0.02, -0.01, 0.03};
  const auto parent_eval = apply_static_l2p_evaluator(
      build_static_l2p_evaluator(basis, Vec3{}, target), parent_local,
      OutputFlags::Both);
  const auto child_eval = apply_static_l2p_evaluator(
      build_static_l2p_evaluator(basis, child_centre, target), child_local,
      OutputFlags::Both);
  REQUIRE(child_eval.phi == Catch::Approx(parent_eval.phi).margin(3.0e-11));
  REQUIRE(child_eval.H.x == Catch::Approx(parent_eval.H.x).margin(3.0e-10));
  REQUIRE(child_eval.H.y == Catch::Approx(parent_eval.H.y).margin(3.0e-10));
  REQUIRE(child_eval.H.z == Catch::Approx(parent_eval.H.z).margin(3.0e-10));
}

TEST_CASE("spherical dense M2L converges for one fixed displacement class")
{
  const std::vector<Vec3> sources{{-0.12, 0.06, 0.03},
                                  {0.09, -0.08, 0.05},
                                  {0.03, 0.11, -0.09}};
  const std::vector<Vec3> moments{{0.7, -0.2, 0.1},
                                  {-0.3, 0.6, 0.25},
                                  {0.2, 0.15, -0.5}};
  const Vec3 source_centre{};
  const Vec3 target_centre{2.5, -1.5, 3.0};
  const Vec3 target = target_centre + Vec3{0.08, -0.04, 0.06};
  const PotentialField exact =
      p2p_dipole_sum(target, sources, moments, OutputFlags::Both);
  const std::vector<double> inputs = flattened_moments(moments);
  double previous_error = 1.0;

  for (const int order : {2, 4, 6, 8}) {
    const SphericalHarmonicBasis basis(order);
    std::vector<double> M(static_cast<std::size_t>(basis.size()), 0.0);
    std::vector<double> L(static_cast<std::size_t>(basis.size()), 0.0);
    apply_static_operator(
        build_static_p2m_operator(basis, source_centre, sources), inputs, M);
    const std::vector<double> matrix =
        build_static_m2l_matrix(basis, target_centre - source_centre);
    apply_static_coefficient_matrix(matrix, M, L);
    const PotentialField actual = apply_static_l2p_evaluator(
        build_static_l2p_evaluator(basis, target_centre, target), L,
        OutputFlags::Both);
    const double error = norm(actual.H - exact.H) / norm(exact.H);
    CAPTURE(order, error, previous_error);
    REQUIRE(error < previous_error);
    previous_error = error;
  }

  REQUIRE(previous_error < 1.0e-9);
}

TEST_CASE("spherical L2P has analytic signs and honours output modes")
{
  const SphericalHarmonicBasis basis(1);
  const Vec3 target{0.2, -0.1, 0.3};
  const std::vector<double> L{2.0, 3.0, 5.0, 7.0};
  const StaticL2PEvaluator evaluator =
      build_static_l2p_evaluator(basis, Vec3{}, target);

  const PotentialField both =
      apply_static_l2p_evaluator(evaluator, L, OutputFlags::Both);
  REQUIRE(both.phi == Catch::Approx(4.6).margin(1.0e-14));
  REQUIRE(both.H.x == Catch::Approx(-7.0).margin(1.0e-14));
  REQUIRE(both.H.y == Catch::Approx(-3.0).margin(1.0e-14));
  REQUIRE(both.H.z == Catch::Approx(-5.0).margin(1.0e-14));

  const PotentialField field =
      apply_static_l2p_evaluator(evaluator, L, OutputFlags::Field);
  REQUIRE(field.phi == 0.0);
  REQUIRE(field.H.x == Catch::Approx(-7.0));

  const PotentialField potential =
      apply_static_l2p_evaluator(evaluator, L, OutputFlags::Potential);
  REQUIRE(potential.phi == Catch::Approx(4.6));
  REQUIRE(norm(potential.H) == 0.0);
}

TEST_CASE("complete spherical FMM converges against direct P2P")
{
  std::mt19937 generator(2718);
  std::uniform_real_distribution<double> position(-0.95, 0.95);
  std::uniform_real_distribution<double> moment(-1.0, 1.0);
  std::vector<Vec3> positions(96);
  std::vector<Vec3> moments(96);
  for (std::size_t index = 0; index < positions.size(); ++index) {
    positions[index] = {position(generator), position(generator),
                        position(generator)};
    moments[index] = {moment(generator), moment(generator), moment(generator)};
  }
  std::vector<int> identities(positions.size());
  std::iota(identities.begin(), identities.end(), 0);
  const auto exact = direct_p2p_reference(
      positions, positions, moments, OutputFlags::Field, identities);
  double first_error = 0.0;
  double final_error = 0.0;
  for (const int order : {2, 4, 6}) {
    UniformFmmOptions options;
    options.precision = StaticPrecision::Float64;
    options.expansion_basis = ExpansionBasis::Spherical;
    options.expansion_order = order;
    options.tree.max_level = 3;
    options.backend = ExecutionBackend::CpuStatic;
    UniformFmm fmm(positions, positions, options);
    const auto actual = fmm.evaluate(moments, OutputFlags::Field, identities);
    const double error = relative_field_error(actual, exact);
    CAPTURE(order, error);
    if (order == 2) {
      first_error = error;
    }
    final_error = error;
    REQUIRE(fmm.coefficient_count() == (order + 1) * (order + 1));
  }
  REQUIRE(final_error < first_error);
  REQUIRE(final_error < 2.0e-3);
}

TEST_CASE("spherical cuboid P2M and L2P equal Gaussian volume averages")
{
  const SphericalHarmonicBasis basis(4);
  const Vec3 centre{-0.2, 0.1, -0.3};
  const Vec3 point{0.35, -0.25, 0.45};
  const CuboidSize size{0.6, 0.4, 0.8};
  const std::vector<Vec3> moment{{0.7, -0.2, 0.4}};
  const auto inputs = flattened_moments(moment);
  const auto cuboid_p2m = build_static_cuboid_p2m_operator(
      basis, centre, std::span<const Vec3>(&point, 1),
      std::span<const CuboidSize>(&size, 1));
  std::vector<double> analytical_M(static_cast<std::size_t>(basis.size()));
  apply_static_operator(cuboid_p2m, inputs, analytical_M);

  std::vector<double> numerical_M(static_cast<std::size_t>(basis.size()));
  std::vector<double> locals(static_cast<std::size_t>(basis.size()));
  for (int mode = 0; mode < basis.size(); ++mode) {
    locals[static_cast<std::size_t>(mode)] =
        0.03 * static_cast<double>(mode + 1);
  }
  PotentialField numerical_target;
  constexpr double node = 0.77459666924148337704;
  constexpr double nodes[3] = {-node, 0.0, node};
  constexpr double weights[3] = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
  for (int x_index = 0; x_index < 3; ++x_index) {
    for (int y_index = 0; y_index < 3; ++y_index) {
      for (int z_index = 0; z_index < 3; ++z_index) {
        const double x = nodes[x_index];
        const double y = nodes[y_index];
        const double z = nodes[z_index];
        const double weight =
            weights[x_index] * weights[y_index] * weights[z_index] / 8.0;
        const Vec3 sample{point.x + 0.5 * size.hx * x,
                          point.y + 0.5 * size.hy * y,
                          point.z + 0.5 * size.hz * z};
        const auto point_p2m = build_static_p2m_operator(
            basis, centre, std::span<const Vec3>(&sample, 1));
        std::vector<double> sample_M(static_cast<std::size_t>(basis.size()));
        apply_static_operator(point_p2m, inputs, sample_M);
        for (int mode = 0; mode < basis.size(); ++mode) {
          numerical_M[static_cast<std::size_t>(mode)] +=
              weight * sample_M[static_cast<std::size_t>(mode)];
        }
        const PotentialField sample_value = apply_static_l2p_evaluator(
            build_static_l2p_evaluator(basis, centre, sample), locals,
            OutputFlags::Both);
        numerical_target.phi += weight * sample_value.phi;
        numerical_target.H = numerical_target.H + weight * sample_value.H;
      }
    }
  }
  for (int mode = 0; mode < basis.size(); ++mode) {
    REQUIRE(analytical_M[static_cast<std::size_t>(mode)] ==
            Catch::Approx(numerical_M[static_cast<std::size_t>(mode)])
                .margin(2.0e-14));
  }

  const PotentialField analytical_target = apply_static_l2p_evaluator(
      build_static_cuboid_l2p_evaluator(basis, centre, point, size), locals,
      OutputFlags::Both);
  REQUIRE(analytical_target.phi ==
          Catch::Approx(numerical_target.phi).margin(2.0e-13));
  REQUIRE(analytical_target.H.x ==
          Catch::Approx(numerical_target.H.x).margin(2.0e-13));
  REQUIRE(analytical_target.H.y ==
          Catch::Approx(numerical_target.H.y).margin(2.0e-13));
  REQUIRE(analytical_target.H.z ==
          Catch::Approx(numerical_target.H.z).margin(2.0e-13));
}

TEST_CASE("spherical plans accept finite cuboid source and target geometry")
{
  const std::vector<Vec3> positions{{0.0, 0.0, 0.0}};
  UniformFmmOptions options;
  options.expansion_basis = ExpansionBasis::Spherical;
  options.precision = StaticPrecision::Float64;
  options.source_geometry = SourceGeometry::UniformCuboid;
  options.source_sizes = {{1.0, 1.0, 1.0}};
  options.target_geometry = TargetGeometry::VolumeAveragedCuboid;
  options.target_sizes = options.source_sizes;
  options.backend = ExecutionBackend::CpuStatic;
  options.tree.root_centre = Vec3{};
  options.tree.root_half_width = 1.0;
  UniformFmm cuboid_fmm(positions, positions, options);
  REQUIRE(cuboid_fmm.coefficient_count() == 25);
  const auto self = cuboid_fmm.evaluate(
      std::vector<Vec3>{{0.0, 0.0, 1.0}}, OutputFlags::Field);
  REQUIRE(self[0].H.z == Catch::Approx(-1.0 / 3.0).epsilon(1.0e-12));

  options.source_geometry = SourceGeometry::PointDipole;
  options.source_sizes.clear();
  options.target_geometry = TargetGeometry::Point;
  options.target_sizes.clear();
  options.backend = ExecutionBackend::CpuReference;
  REQUIRE_THROWS_AS(UniformFmm(positions, positions, options),
                    std::invalid_argument);
  options.backend = ExecutionBackend::CpuStatic;
  options.spherical_m2l_backend =
      static_cast<SphericalM2LBackend>(-1);
  REQUIRE_THROWS_AS(UniformFmm(positions, positions, options),
                    std::invalid_argument);
}

TEST_CASE("spherical FP32 state replaces cleanly and accepts empty geometry")
{
  UniformFmmOptions options;
  options.expansion_basis = ExpansionBasis::Spherical;
  options.expansion_order = 4;
  options.precision = StaticPrecision::Float32;
  options.tree.max_level = 2;

  UniformFmm empty({}, {}, options);
  REQUIRE(empty.evaluate_float32({}, OutputFlags::Both).empty());

  std::vector<Vec3> positions;
  std::vector<Vec3> moments;
  for (int z = 0; z < 3; ++z) {
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 3; ++x) {
        positions.push_back({1.0e-9 * (-0.75 + 0.75 * x),
                             1.0e-9 * (-0.75 + 0.75 * y),
                             1.0e-9 * (-0.75 + 0.75 * z)});
        moments.push_back({0.2 + 0.03 * x, -0.1 + 0.02 * y,
                           0.15 - 0.01 * z});
      }
    }
  }
  std::vector<int> identities(positions.size());
  std::iota(identities.begin(), identities.end(), 0);
  options.tree.root_centre = Vec3{};
  options.tree.root_half_width = 1.0e-9;
  UniformFmm fmm(positions, positions, options);
  const auto first =
      fmm.evaluate_float32(moments, OutputFlags::Both, identities);
  REQUIRE(first.size() == positions.size());
  for (const FloatPotentialField value : first) {
    REQUIRE(std::isfinite(value.phi));
    REQUIRE(std::isfinite(value.H.x));
    REQUIRE(std::isfinite(value.H.y));
    REQUIRE(std::isfinite(value.H.z));
  }
  UniformFmmOptions double_options = options;
  double_options.precision = StaticPrecision::Float64;
  UniformFmm double_fmm(positions, positions, double_options);
  const auto double_values =
      double_fmm.evaluate(moments, OutputFlags::Field, identities);
  double difference_norm_squared = 0.0;
  double reference_norm_squared = 0.0;
  for (std::size_t index = 0; index < first.size(); ++index) {
    const Vec3 float_field{first[index].H.x, first[index].H.y,
                           first[index].H.z};
    const Vec3 difference = float_field - double_values[index].H;
    difference_norm_squared += dot(difference, difference);
    reference_norm_squared += dot(double_values[index].H,
                                  double_values[index].H);
  }
  REQUIRE(std::sqrt(difference_norm_squared / reference_norm_squared) <
          2.0e-4);

  const std::vector<Vec3> zero_moments(positions.size());
  const auto second =
      fmm.evaluate_float32(zero_moments, OutputFlags::Both, identities);
  for (const FloatPotentialField value : second) {
    REQUIRE(value.phi == 0.0F);
    REQUIRE(value.H.x == 0.0F);
    REQUIRE(value.H.y == 0.0F);
    REQUIRE(value.H.z == 0.0F);
  }

  const StaticPlanStatistics& statistics = fmm.static_plan_statistics();
  REQUIRE(statistics.scalar_bytes == sizeof(float));
  REQUIRE(statistics.coefficient_count == 25);
  REQUIRE(statistics.multipole_state_bytes > 0);
  REQUIRE(statistics.local_state_bytes > 0);
}

TEST_CASE("spherical oneMKL and portable static execution agree")
{
  if (!one_mkl_available()) {
    SUCCEED("This build does not include oneMKL");
    return;
  }

  const std::vector<Vec3> positions{{-0.8, -0.6, -0.4},
                                    {0.7, -0.5, 0.3},
                                    {-0.3, 0.8, 0.6},
                                    {0.6, 0.4, -0.7},
                                    {-0.55, 0.25, -0.15},
                                    {0.35, -0.2, 0.75}};
  const std::vector<Vec3> moments{{0.2, -0.5, 0.7},
                                  {-0.6, 0.1, 0.4},
                                  {0.8, 0.3, -0.2},
                                  {-0.1, -0.7, 0.5},
                                  {0.45, -0.25, 0.15},
                                  {-0.2, 0.55, -0.35}};
  std::vector<int> identities(positions.size());
  std::iota(identities.begin(), identities.end(), 0);

  UniformFmmOptions portable_options;
  portable_options.precision = StaticPrecision::Float64;
  portable_options.expansion_basis = ExpansionBasis::Spherical;
  portable_options.expansion_order = 4;
  portable_options.tree.max_level = 2;
  portable_options.backend = ExecutionBackend::CpuStatic;
  UniformFmmOptions mkl_options = portable_options;
  mkl_options.static_matrix_backend = StaticMatrixBackend::OneMkl;

  UniformFmm portable(positions, positions, portable_options);
  UniformFmm mkl(positions, positions, mkl_options);
  const auto portable_values =
      portable.evaluate(moments, OutputFlags::Both, identities);
  const auto mkl_values = mkl.evaluate(moments, OutputFlags::Both, identities);

  for (std::size_t index = 0; index < positions.size(); ++index) {
    REQUIRE(mkl_values[index].phi ==
            Catch::Approx(portable_values[index].phi).epsilon(2.0e-13));
    REQUIRE(mkl_values[index].H.x ==
            Catch::Approx(portable_values[index].H.x).epsilon(2.0e-13));
    REQUIRE(mkl_values[index].H.y ==
            Catch::Approx(portable_values[index].H.y).epsilon(2.0e-13));
    REQUIRE(mkl_values[index].H.z ==
            Catch::Approx(portable_values[index].H.z).epsilon(2.0e-13));
  }
}

TEST_CASE("spherical CUDA partial and full agree with CPU static", "[cuda]")
{
  if (!cuda_m2l_p2p_available() || !cuda_full_available()) {
    SKIP("No usable CUDA device is available for spherical runtime checks");
  }

  std::mt19937 generator(1618);
  std::uniform_real_distribution<double> position(-0.9, 0.9);
  std::uniform_real_distribution<double> moment(-1.0, 1.0);
  std::vector<Vec3> positions(32);
  std::vector<Vec3> moments(32);
  for (std::size_t index = 0; index < positions.size(); ++index) {
    positions[index] = {position(generator), position(generator),
                        position(generator)};
    moments[index] = {moment(generator), moment(generator), moment(generator)};
  }
  std::vector<int> identities(positions.size());
  std::iota(identities.begin(), identities.end(), 0);

  UniformFmmOptions options;
  options.precision = StaticPrecision::Float64;
  options.expansion_basis = ExpansionBasis::Spherical;
  options.expansion_order = 3;
  options.tree.max_level = 2;
  options.fixed_target_source_indices = identities;
  options.backend = ExecutionBackend::CpuStatic;
  UniformFmm cpu(positions, positions, options);
  const auto expected = cpu.evaluate(moments, OutputFlags::Field, identities);

  for (const ExecutionBackend backend :
       {ExecutionBackend::CudaPartial, ExecutionBackend::CudaFull}) {
    options.backend = backend;
    UniformFmm cuda(positions, positions, options);
    const auto actual = cuda.evaluate(moments, OutputFlags::Field, identities);
    CAPTURE(backend);
    for (std::size_t index = 0; index < actual.size(); ++index) {
      REQUIRE(actual[index].H.x ==
              Catch::Approx(expected[index].H.x).margin(3.0e-11));
      REQUIRE(actual[index].H.y ==
              Catch::Approx(expected[index].H.y).margin(3.0e-11));
      REQUIRE(actual[index].H.z ==
              Catch::Approx(expected[index].H.z).margin(3.0e-11));
    }
    REQUIRE(cuda.cuda_plan_statistics().m2l_unique_matrix_count ==
            cuda.static_plan_statistics().m2l_operators);
  }
}
