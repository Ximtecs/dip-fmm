// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "cdfmm/operators.hpp"
#include "cdfmm/static_operators.hpp"
#include "cdfmm/uniform_fmm.hpp"
#include "cdfmm/validation.hpp"

using namespace cdfmm;

namespace {

double relative_field_error(const std::span<const PotentialField> actual,
                            const std::span<const PotentialField> expected) {
  double error_squared = 0.0;
  double reference_squared = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const Vec3 difference = actual[index].H - expected[index].H;
    error_squared += dot(difference, difference);
    reference_squared += dot(expected[index].H, expected[index].H);
  }
  return std::sqrt(error_squared / reference_squared);
}

double relative_field_error(
    const std::span<const FloatPotentialField> actual,
    const std::span<const PotentialField> expected) {
  double error_squared = 0.0;
  double reference_squared = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const Vec3 value{actual[index].H.x, actual[index].H.y,
                     actual[index].H.z};
    const Vec3 difference = value - expected[index].H;
    error_squared += dot(difference, difference);
    reference_squared += dot(expected[index].H, expected[index].H);
  }
  return std::sqrt(error_squared / reference_squared);
}

} // namespace

TEST_CASE("FP32 static operators quantise storage and arithmetic") {
  REQUIRE(UniformFmmOptions{}.precision == StaticPrecision::Float32);
  REQUIRE(UniformFmmOptions{}.expansion_basis == ExpansionBasis::Spherical);

  const MultiIndexSet basis(4);
  const std::vector<Vec3> positions{
      {-0.4, 0.1, 0.2}, {0.3, -0.2, 0.5}, {0.8, 0.4, -0.1}};
  const std::vector<Vec3> moments{
      {0.2, -0.4, 0.7}, {-0.3, 0.8, 0.1}, {0.6, 0.2, -0.5}};
  std::vector<FloatVec3> moments_float;
  for (const Vec3 moment : moments) {
    moments_float.push_back({static_cast<float>(moment.x),
                             static_cast<float>(moment.y),
                             static_cast<float>(moment.z)});
  }

  const StaticCoefficientOperator p2m =
      build_static_p2m_operator(basis, Vec3{}, positions);
  const FloatStaticCoefficientOperator p2m_float =
      quantise_static_operator(p2m);
  REQUIRE(sizeof(p2m_float.entries.front().value) == 4);
  std::vector<double> input(moments.size() * 3);
  std::vector<float> input_float(moments.size() * 3);
  for (std::size_t index = 0; index < moments.size(); ++index) {
    for (int component = 0; component < 3; ++component) {
      input[index * 3 + component] = moments[index][component];
      input_float[index * 3 + component] =
          static_cast<float>(moments[index][component]);
    }
  }
  std::vector<double> M(static_cast<std::size_t>(basis.size()));
  std::vector<float> M_float(static_cast<std::size_t>(basis.size()));
  apply_static_operator(p2m, input, M);
  apply_static_operator(p2m_float, input_float, M_float);
  for (int coefficient = 0; coefficient < basis.size(); ++coefficient) {
    REQUIRE(M_float[static_cast<std::size_t>(coefficient)] ==
            Catch::Approx(M[static_cast<std::size_t>(coefficient)])
                .margin(3.0e-6));
  }

  const StaticL2PEvaluator l2p =
      build_static_l2p_evaluator(basis, Vec3{}, {0.2, -0.1, 0.3});
  const auto l2p_float = quantise_static_l2p_evaluator(l2p);
  const PotentialField field = apply_static_l2p_evaluator(l2p, M);
  const FloatPotentialField field_float =
      apply_static_l2p_evaluator(l2p_float, M_float);
  REQUIRE(field_float.H.x == Catch::Approx(field.H.x).margin(2.0e-5));
  REQUIRE(field_float.H.y == Catch::Approx(field.H.y).margin(2.0e-5));
  REQUIRE(field_float.H.z == Catch::Approx(field.H.z).margin(2.0e-5));
}

TEST_CASE("FP32 M2M L2L and M2L use quantised coefficient arithmetic") {
  const MultiIndexSet basis(4);
  std::vector<double> input(static_cast<std::size_t>(basis.size()));
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = 0.03 * static_cast<double>(index + 1);
  }
  const std::vector<float> input_float(input.begin(), input.end());
  for (const StaticCoefficientOperator operator_map : {
           build_static_m2m_operator(basis, {0.1, -0.2, 0.3}),
           build_static_l2l_operator(basis, {-0.15, 0.05, 0.25})}) {
    const auto operator_float = quantise_static_operator(operator_map);
    std::vector<double> expected(input.size());
    std::vector<float> actual(input.size());
    apply_static_operator(operator_map, input, expected);
    apply_static_operator(operator_float, input_float, actual);
    for (std::size_t index = 0; index < input.size(); ++index) {
      REQUIRE(actual[index] == Catch::Approx(expected[index]).margin(3.0e-6));
    }
  }

  StaticM2LPlan m2l;
  m2l.coefficient_count = basis.size();
  m2l.matrix_count = 1;
  m2l.level_count = 2;
  m2l.matrices = build_static_m2l_matrix(basis, {2.0, -3.0, 4.0});
  m2l.multipole_scaling.assign(static_cast<std::size_t>(2 * basis.size()),
                               1.0);
  m2l.local_scaling.assign(static_cast<std::size_t>(2 * basis.size()), 1.0);
  m2l.target_row_offsets = {0, 0, 1};
  m2l.source_nodes = {0};
  m2l.matrix_ids = {0};
  m2l.interaction_levels = {1};
  m2l.level_target_begin = {0, 1};
  m2l.level_target_end = {1, 2};
  const FloatStaticM2LPlan m2l_float = quantise_static_m2l_plan(m2l);
  std::vector<double> multipoles(input.size() * 2);
  std::copy(input.begin(), input.end(), multipoles.begin());
  std::vector<float> multipoles_float(multipoles.begin(), multipoles.end());
  std::vector<double> locals(input.size() * 2);
  std::vector<float> locals_float(input.size() * 2);
  apply_static_m2l_plan(m2l, 1, multipoles, locals);
  apply_static_m2l_plan(m2l_float, 1, multipoles_float, locals_float);
  for (std::size_t index = input.size(); index < locals.size(); ++index) {
    REQUIRE(locals_float[index] ==
            Catch::Approx(locals[index]).margin(3.0e-6));
  }
}

TEST_CASE("FP32 canonical compact leaf and BSR P2P agree") {
  const std::vector<Vec3> sources{
      {-0.4, 0.1, 0.2}, {0.3, -0.2, 0.5}, {0.8, 0.4, -0.1}};
  const std::vector<Vec3> targets{{0.1, 0.6, 0.3}, {-0.7, -0.3, 0.2}};
  const std::vector<std::array<int, 2>> interactions{
      {0, 0}, {0, 2}, {1, 0}, {1, 1}};
  const StaticP2POperator canonical =
      build_static_p2p_operator(targets, sources, interactions);
  const StaticP2PCompactPlan compact =
      build_static_p2p_compact_plan(canonical);
  const std::vector<StaticP2PLeafPair> pairs{
      {0, 1, 0, 1}, {0, 1, 2, 1}, {1, 1, 0, 2}};
  const StaticP2PLeafPlan leaf = build_static_p2p_leaf_plan(canonical, pairs);
  const std::vector<int> identities{-1, 1};
  const StaticP2PBsrPlan bsr =
      build_static_p2p_bsr_plan(canonical, identities);
  const auto canonical_float = quantise_static_p2p_operator(canonical);
  const auto compact_float = quantise_static_p2p_compact_plan(compact);
  const auto leaf_float = quantise_static_p2p_leaf_plan(leaf);
  const auto bsr_float = quantise_static_p2p_bsr_plan(bsr);
  const std::vector<FloatVec3> moments{
      {0.2F, -0.4F, 0.7F}, {-0.3F, 0.8F, 0.1F}, {0.6F, 0.2F, -0.5F}};
  std::vector<FloatVec3> expected(targets.size());
  std::vector<FloatVec3> compact_result(targets.size());
  std::vector<FloatVec3> leaf_result(targets.size());
  std::vector<FloatVec3> bsr_result(targets.size());
  apply_static_p2p_operator(canonical_float, moments, expected, identities);
  apply_static_p2p_compact_plan(compact_float, moments, compact_result,
                                identities);
  apply_static_p2p_leaf_plan(leaf_float, moments, leaf_result, identities);
  apply_static_p2p_bsr_plan(bsr_float, moments, bsr_result, identities);
  for (std::size_t target = 0; target < targets.size(); ++target) {
    for (int component = 0; component < 3; ++component) {
      REQUIRE(compact_result[target][component] ==
              Catch::Approx(expected[target][component]).margin(2.0e-6));
      REQUIRE(leaf_result[target][component] ==
              Catch::Approx(expected[target][component]).margin(2.0e-6));
      REQUIRE(bsr_result[target][component] ==
              Catch::Approx(expected[target][component]).margin(2.0e-6));
    }
  }
  REQUIRE(compact_float.memory().tensor_bytes * 2 ==
          compact.memory().tensor_bytes);
  REQUIRE(bsr_float.memory().tensor_bytes * 2 == bsr.memory().tensor_bytes);
}

TEST_CASE("FMM precision controls state output and convergence") {
  std::mt19937 generator(1701);
  std::uniform_real_distribution<double> position(-0.95, 0.95);
  std::uniform_real_distribution<double> moment(-1.0, 1.0);
  std::vector<Vec3> positions(48);
  std::vector<Vec3> moments(48);
  for (std::size_t index = 0; index < positions.size(); ++index) {
    positions[index] = {position(generator), position(generator),
                        position(generator)};
    moments[index] = {moment(generator), moment(generator), moment(generator)};
  }
  std::vector<int> identities(positions.size());
  for (std::size_t index = 0; index < identities.size(); ++index) {
    identities[index] = static_cast<int>(index);
  }
  const std::vector<PotentialField> direct = direct_p2p_reference(
      positions, positions, moments, OutputFlags::Field, identities);

  for (const ExecutionBackend backend :
       {ExecutionBackend::CpuReference, ExecutionBackend::CpuStatic}) {
    const std::vector<std::array<int, 2>> configurations =
        backend == ExecutionBackend::CpuReference
            ? std::vector<std::array<int, 2>>{{4, 3}}
            : std::vector<std::array<int, 2>>{
                  {2, 2}, {4, 3}, {6, 4}};
    for (const auto [order, depth] : configurations) {
        UniformFmmOptions fp64_options;
        fp64_options.expansion_basis = ExpansionBasis::Cartesian;
        fp64_options.expansion_order = order;
        fp64_options.tree.max_level = depth;
        fp64_options.backend = backend;
        fp64_options.precision = StaticPrecision::Float64;
        UniformFmmOptions fp32_options = fp64_options;
        fp32_options.precision = StaticPrecision::Float32;
        UniformFmm fp64(positions, positions, fp64_options);
        UniformFmm fp32(positions, positions, fp32_options);
        const auto result64 = fp64.evaluate_float64(
            moments, OutputFlags::Field, identities);
        const auto result32 = fp32.evaluate_float32(
            moments, OutputFlags::Field, identities);
        const double error64 = relative_field_error(result64, direct);
        const double error32 = relative_field_error(result32, direct);
        CAPTURE(backend, order, depth, error64, error32);
        REQUIRE(error32 <= error64 + 3.0e-5);
        REQUIRE(fp32.root_multipole_float32().size() ==
                fp64.root_multipole_float64().size());
        REQUIRE(fp32.static_plan_statistics().scalar_bytes == 4);
        REQUIRE(fp64.static_plan_statistics().scalar_bytes == 8);
        REQUIRE(fp32.static_plan_statistics().state_bytes * 2 ==
                fp64.static_plan_statistics().state_bytes +
                    identities.size() * sizeof(int));
        REQUIRE_THROWS(fp32.root_multipole_float64());
        REQUIRE_THROWS(fp64.root_multipole_float32());
    }
  }
}

TEST_CASE("FP32 FMM accepts empty geometry and replaces repeated state") {
  UniformFmmOptions options;
  options.expansion_basis = ExpansionBasis::Cartesian;
  options.precision = StaticPrecision::Float32;
  UniformFmm empty({}, {}, options);
  REQUIRE(empty.evaluate_float32({}).empty());

  const std::vector<Vec3> positions{{-0.25, 0.0, 0.0},
                                    {0.25, 0.0, 0.0}};
  UniformFmm fmm(positions, positions, options);
  const std::vector<int> identities{0, 1};
  const std::vector<Vec3> first{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
  const std::vector<Vec3> second{{0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}};
  (void)fmm.evaluate_float32(first, OutputFlags::Field, identities);
  const auto repeated =
      fmm.evaluate_float32(second, OutputFlags::Field, identities);
  const auto widened = fmm.evaluate(second, OutputFlags::Field, identities);
  UniformFmm fresh(positions, positions, options);
  const auto expected =
      fresh.evaluate_float32(second, OutputFlags::Field, identities);
  for (std::size_t target = 0; target < positions.size(); ++target) {
    REQUIRE(repeated[target].H.x == expected[target].H.x);
    REQUIRE(repeated[target].H.y == expected[target].H.y);
    REQUIRE(repeated[target].H.z == expected[target].H.z);
    REQUIRE(widened[target].H.x ==
            static_cast<double>(repeated[target].H.x));
    REQUIRE(widened[target].H.y ==
            static_cast<double>(repeated[target].H.y));
    REQUIRE(widened[target].H.z ==
            static_cast<double>(repeated[target].H.z));
  }
  const std::span<const float> root_values = fmm.root_multipole_float32();
  const std::vector<float> root_float(root_values.begin(), root_values.end());
  const std::span<const double> root_widened = fmm.root_multipole();
  for (std::size_t index = 0; index < root_float.size(); ++index) {
    REQUIRE(root_widened[index] == static_cast<double>(root_float[index]));
  }
}

TEST_CASE("FP32 FMM normalises nanometre-scale expansion state") {
  std::vector<Vec3> positions;
  std::vector<Vec3> moments;
  for (int iz = 0; iz < 4; ++iz) {
    for (int iy = 0; iy < 4; ++iy) {
      for (int ix = 0; ix < 4; ++ix) {
        positions.push_back({(static_cast<double>(ix) - 1.5) * 3.0e-8,
                             (static_cast<double>(iy) - 1.5) * 3.0e-8,
                             (static_cast<double>(iz) - 1.5) * 3.0e-8});
        const double index = static_cast<double>(positions.size());
        moments.push_back({1.0e-21 * (0.7 + 0.2 * std::sin(0.17 * index)),
                           1.0e-21 * (-0.3 + 0.2 * std::cos(0.11 * index)),
                           4.0e-22 * std::sin(0.07 * index + 0.3)});
      }
    }
  }
  std::vector<int> identities(positions.size());
  std::iota(identities.begin(), identities.end(), 0);

  UniformFmmOptions options64;
  options64.expansion_basis = ExpansionBasis::Cartesian;
  options64.expansion_order = 4;
  options64.tree.max_level = 4;
  options64.backend = ExecutionBackend::CpuStatic;
  options64.precision = StaticPrecision::Float64;
  UniformFmmOptions options32 = options64;
  options32.precision = StaticPrecision::Float32;
  UniformFmm fp64(positions, positions, options64);
  UniformFmm fp32(positions, positions, options32);
  const auto expected = fp64.evaluate_float64(
      moments, OutputFlags::Both, identities);
  const auto actual = fp32.evaluate_float32(
      moments, OutputFlags::Both, identities);

  REQUIRE(relative_field_error(actual, expected) < 2.0e-5);
  for (std::size_t target = 0; target < actual.size(); ++target) {
    REQUIRE(std::isfinite(actual[target].phi));
    REQUIRE(std::isfinite(actual[target].H.x));
    REQUIRE(std::isfinite(actual[target].H.y));
    REQUIRE(std::isfinite(actual[target].H.z));
    REQUIRE(actual[target].phi ==
            Catch::Approx(expected[target].phi).epsilon(3.0e-5));
  }
}

TEST_CASE("CUDA partial and full execute FP32 and FP64 plans") {
  if (!cuda_available()) {
    SUCCEED("CUDA is unavailable");
    return;
  }
  const std::vector<Vec3> positions{
      {-0.7, -0.2, 0.1}, {-0.1, 0.4, -0.3}, {0.5, -0.4, 0.2},
      {0.8, 0.3, -0.6},  {0.2, 0.7, 0.5},   {-0.5, 0.1, 0.6}};
  const std::vector<Vec3> moments{
      {0.2, -0.4, 0.7}, {-0.3, 0.8, 0.1}, {0.6, 0.2, -0.5},
      {-0.1, 0.9, 0.3}, {0.4, -0.2, 0.5}, {0.7, 0.1, -0.6}};
  std::vector<int> identities(positions.size());
  std::iota(identities.begin(), identities.end(), 0);
  for (const ExecutionBackend backend :
       {ExecutionBackend::CudaPartial, ExecutionBackend::CudaFull}) {
    UniformFmmOptions fp64_options;
    fp64_options.expansion_basis = ExpansionBasis::Cartesian;
    fp64_options.expansion_order = 4;
    fp64_options.tree.max_level = 2;
    fp64_options.backend = backend;
    fp64_options.precision = StaticPrecision::Float64;
    fp64_options.fixed_target_source_indices = identities;
    UniformFmmOptions fp32_options = fp64_options;
    fp32_options.precision = StaticPrecision::Float32;
    UniformFmm fp64(positions, positions, fp64_options);
    UniformFmm fp32(positions, positions, fp32_options);
    const auto expected =
        fp64.evaluate_float64(moments, OutputFlags::Field, identities);
    const auto actual =
        fp32.evaluate_float32(moments, OutputFlags::Field, identities);
    CAPTURE(backend);
    REQUIRE(relative_field_error(actual, expected) < 3.0e-5);
    REQUIRE(fp32.cuda_plan_statistics().scalar_bytes == sizeof(float));
    REQUIRE(fp64.cuda_plan_statistics().scalar_bytes == sizeof(double));
    REQUIRE(fp32.cuda_plan_statistics().evaluation_h2d_bytes <
            fp64.cuda_plan_statistics().evaluation_h2d_bytes);
  }
}
