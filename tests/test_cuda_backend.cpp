// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <array>
#include <cmath>
#include <numeric>
#include <vector>

#include "cdfmm/cuda_direct.hpp"
#include "cdfmm/cuda_cuboid.hpp"
#include "cdfmm/cuda_p2p.hpp"
#include "cdfmm/static_operators.hpp"
#include "cdfmm/uniform_fmm.hpp"

#include "backend/cuda/execution_policy.hpp"
#include "cdfmm/validation.hpp"

using namespace cdfmm;

TEST_CASE("CUDA direct stubs preserve disabled-build behaviour", "[cuda]")
{
    if (cuda_compiled()) {
        SUCCEED("CUDA is compiled in; disabled-build stubs are not active");
        return;
    }

    REQUIRE_FALSE(cuda_direct_available());
    REQUIRE_FALSE(cuda_dense_direct_available());

    REQUIRE_THROWS_WITH(
        CudaDirectPlan(std::span<const Vec3>{}, std::span<const Vec3>{}),
        "CUDA backend requested, but CDFMM_ENABLE_CUDA is OFF");
    REQUIRE_THROWS_WITH(
        CudaDenseDirectPlan(
            std::span<const Vec3>{}, std::span<const Vec3>{}),
        "CUDA dense direct backend requested, but CDFMM_ENABLE_CUDA is OFF");
    REQUIRE_THROWS_WITH(
        cuda_direct_p2p_reference(
            std::span<const Vec3>{}, std::span<const Vec3>{},
            std::span<const Vec3>{}),
        "CUDA direct P2P requested, but CDFMM_ENABLE_CUDA is OFF");
}

TEST_CASE("CUDA M2L/P2P compatibility names resolve to one backend", "[cuda]")
{
    REQUIRE(ExecutionBackend::CudaM2L == ExecutionBackend::CudaM2LP2P);
    REQUIRE(ExecutionBackend::CudaM2LStaticP2P ==
            ExecutionBackend::CudaM2LP2P);
    REQUIRE(cuda_m2l_available() == cuda_m2l_p2p_available());
    REQUIRE(ExecutionBackend::CudaPartial == ExecutionBackend::CudaM2LP2P);
}

TEST_CASE("production backends dispatch canonical operators per stage", "[cuda]")
{
    const std::vector<Vec3> positions{{-0.25, 0.0, 0.0},
                                      {0.25, 0.0, 0.0}};
    UniformFmmOptions options;
    options.expansion_basis = ExpansionBasis::Cartesian;
    options.tree.max_level = 1;
    options.backend = ExecutionBackend::CpuStatic;
    UniformFmm cpu(positions, positions, options);
    const StaticExecutionPlan cpu_plan = cpu.execution_plan();
    REQUIRE(cpu_plan.p2m == StaticOperatorExecutor::Portable);
    REQUIRE(cpu_plan.m2m == StaticOperatorExecutor::Portable);
    REQUIRE(cpu_plan.m2l == StaticOperatorExecutor::Portable);
    REQUIRE(cpu_plan.l2l == StaticOperatorExecutor::Portable);
    REQUIRE(cpu_plan.l2p == StaticOperatorExecutor::Portable);
    REQUIRE(cpu_plan.p2p == StaticOperatorExecutor::Portable);

    if (cuda_m2l_p2p_available()) {
        options.backend = ExecutionBackend::CudaPartial;
        UniformFmm partial(positions, positions, options);
        const StaticExecutionPlan partial_plan = partial.execution_plan();
        REQUIRE(partial_plan.p2m == cpu_plan.p2m);
        REQUIRE(partial_plan.m2m == cpu_plan.m2m);
        REQUIRE(partial_plan.m2l == StaticOperatorExecutor::Cuda);
        REQUIRE(partial_plan.l2l == cpu_plan.l2l);
        REQUIRE(partial_plan.l2p == cpu_plan.l2p);
        REQUIRE(partial_plan.p2p == StaticOperatorExecutor::Cuda);
    }
}

TEST_CASE("full CUDA FMM is device resident across evaluations", "[cuda][manual]")
{
    if (!cuda_full_available()) {
        SUCCEED("full CUDA FMM is unavailable");
        return;
    }
    std::vector<Vec3> positions;
    std::vector<Vec3> moments;
    for (int index = 0; index < 64; ++index) {
        const double value = static_cast<double>(index);
        positions.push_back({
            -0.95 + 1.9 * static_cast<double>((index * 17) % 67) / 66.0,
            -0.95 + 1.9 * static_cast<double>((index * 29) % 71) / 70.0,
            -0.95 + 1.9 * static_cast<double>((index * 43) % 73) / 72.0
        });
        moments.push_back({std::sin(value), std::cos(value),
                           std::sin(0.25 * value)});
    }
    std::vector<int> identities(positions.size());
    std::iota(identities.begin(), identities.end(), 0);
    UniformFmmOptions cpu_options;
    cpu_options.expansion_basis = ExpansionBasis::Cartesian;
    cpu_options.precision = StaticPrecision::Float64;
    cpu_options.expansion_order = 4;
    cpu_options.tree.max_level = 3;
    cpu_options.tree.root_centre = Vec3{};
    cpu_options.tree.root_half_width = 1.0;
    cpu_options.backend = ExecutionBackend::CpuStatic;
    UniformFmmOptions cuda_options = cpu_options;
    cuda_options.backend = ExecutionBackend::CudaFull;
    UniformFmm cpu(positions, positions, cpu_options);
    UniformFmm cuda(positions, positions, cuda_options);
    const auto expected = cpu.evaluate(moments, OutputFlags::Field, identities);
    const auto actual = cuda.evaluate(moments, OutputFlags::Field, identities);
    for (std::size_t index = 0; index < actual.size(); ++index) {
        REQUIRE(actual[index].H.x == Catch::Approx(expected[index].H.x).margin(3.0e-11));
        REQUIRE(actual[index].H.y == Catch::Approx(expected[index].H.y).margin(3.0e-11));
        REQUIRE(actual[index].H.z == Catch::Approx(expected[index].H.z).margin(3.0e-11));
    }
    const CudaPlanStatistics first = cuda.cuda_plan_statistics();
    REQUIRE(first.m2l_unique_matrix_count ==
            cuda.static_plan_statistics().m2l_operators);
    REQUIRE(first.p2p_interaction_count ==
            cuda.static_plan_statistics().p2p_interactions);
    REQUIRE(first.m2l_unique_matrix_count <=
            StaticPlanStatistics::theoretical_maximum_m2l_classes);
    REQUIRE(first.evaluation_h2d_bytes == positions.size() * 3 * sizeof(double));
    REQUIRE(first.evaluation_d2h_bytes == positions.size() * 3 * sizeof(double));
    REQUIRE(first.evaluation_h2d_calls == 1);
    REQUIRE(first.evaluation_d2h_calls == 1);
    REQUIRE(cuda.last_timings().cuda_p2p_kernel.calls == 1);
    REQUIRE(cuda.last_timings().cuda_p2p_kernel.total_seconds > 0.0);
    const auto second = cuda.evaluate(moments, OutputFlags::Field, identities);
    REQUIRE(second.size() == positions.size());
    const CudaPlanStatistics repeated = cuda.cuda_plan_statistics();
    REQUIRE(repeated.static_upload_count == first.static_upload_count);
    REQUIRE(repeated.geometry_upload_count == first.geometry_upload_count);
    REQUIRE(repeated.setup_h2d_bytes == first.setup_h2d_bytes);
    REQUIRE(repeated.evaluation_h2d_calls == 2);
    REQUIRE(repeated.evaluation_d2h_calls == 2);
}

TEST_CASE("CUDA direct P2P agrees with the CPU direct reference", "[cuda]")
{
    if (!cuda_available()) {
        SUCCEED("No CUDA-capable device is available");
        return;
    }

    const std::vector<Vec3> sources{
        {-0.75, -0.20, 0.10},
        {0.60, -0.35, -0.25},
        {-0.30, 0.70, 0.40},
        {0.45, 0.55, -0.60}
    };
    const std::vector<Vec3> targets{
        {0.20, -0.10, 0.80},
        {-0.50, 0.25, -0.70},
        {0.80, 0.40, 0.30}
    };
    const std::vector<Vec3> moments{
        {0.70, -0.20, 0.10},
        {-0.40, 0.80, 0.30},
        {0.20, 0.10, -0.60},
        {-0.30, -0.50, 0.90}
    };
    const auto direct = direct_p2p_reference(
        targets,
        sources,
        moments,
        OutputFlags::Both
    );

    const auto actual = cuda_direct_p2p_reference(
        targets, sources, moments, OutputFlags::Both
    );
    CudaDirectPlan plan(sources, targets);
    std::vector<PotentialField> persistent_actual(targets.size());
    plan.evaluate(moments, persistent_actual, OutputFlags::Both);

    REQUIRE(cuda_direct_available());
    REQUIRE(plan.source_count() == sources.size());
    REQUIRE(plan.target_count() == targets.size());
    for (std::size_t index = 0; index < targets.size(); ++index) {
            REQUIRE(actual[index].phi ==
                    Catch::Approx(direct[index].phi).epsilon(2.0e-13));
            REQUIRE(actual[index].H.x ==
                    Catch::Approx(direct[index].H.x).epsilon(2.0e-13));
            REQUIRE(actual[index].H.y ==
                    Catch::Approx(direct[index].H.y).epsilon(2.0e-13));
            REQUIRE(actual[index].H.z ==
                    Catch::Approx(direct[index].H.z).epsilon(2.0e-13));
            REQUIRE(persistent_actual[index].phi ==
                    Catch::Approx(direct[index].phi).epsilon(2.0e-13));
            REQUIRE(persistent_actual[index].H.x ==
                    Catch::Approx(direct[index].H.x).epsilon(2.0e-13));
            REQUIRE(persistent_actual[index].H.y ==
                    Catch::Approx(direct[index].H.y).epsilon(2.0e-13));
            REQUIRE(persistent_actual[index].H.z ==
                    Catch::Approx(direct[index].H.z).epsilon(2.0e-13));
    }
}

TEST_CASE("CUDA dense cuboid direct plan agrees with portable CPU",
          "[cuda][manual]")
{
    if (!cuda_dense_direct_available()) {
        SUCCEED("CUDA dense cuboid backend is unavailable");
        return;
    }

    const std::vector<Vec3> positions{
        {-1.0, -1.0, 0.0},
        {1.0, -1.0, 0.0},
        {-1.0, 1.0, 0.0},
        {1.0, 1.0, 0.0}
    };
    const std::vector<Vec3> moments{
        {0.7, -0.2, 0.1},
        {-0.4, 0.8, 0.3},
        {0.2, 0.1, -0.6},
        {-0.3, -0.5, 0.9}
    };
    const std::array<CuboidSize, 1> cube{{{0.5, 0.5, 0.5}}};
    for (const StaticPrecision precision : {
             StaticPrecision::Float32, StaticPrecision::Float64}) {
        const DenseDirectPlan cpu(
            positions, positions, SourceGeometry::RectangularPrism,
            TargetGeometry::Point, cube, {}, {}, precision);
        CudaDenseDirectPlan cuda(
            positions, positions, SourceGeometry::RectangularPrism,
            TargetGeometry::Point, cube, {}, {}, precision);

        const auto expected = cpu.evaluate(
            moments, DenseDirectBackend::Portable);
        const auto actual = cuda.evaluate(moments);
        const double tolerance = precision == StaticPrecision::Float32
            ? 2.0e-5 : 2.0e-13;

        REQUIRE(cuda.source_count() == positions.size());
        REQUIRE(cuda.target_count() == positions.size());
        REQUIRE(cuda.static_precision() == precision);
        REQUIRE(cuda.tensor_memory_bytes() == cpu.tensor_memory_bytes());
        REQUIRE(cuda.persistent_device_bytes() >= cuda.tensor_memory_bytes());
        for (std::size_t index = 0; index < positions.size(); ++index) {
            REQUIRE(actual[index].x ==
                    Catch::Approx(expected[index].x).epsilon(tolerance));
            REQUIRE(actual[index].y ==
                    Catch::Approx(expected[index].y).epsilon(tolerance));
            REQUIRE(actual[index].z ==
                    Catch::Approx(expected[index].z).epsilon(tolerance));
        }
    }
}

TEST_CASE("CUDA static P2P packings agree with canonical CPU rows",
          "[cuda][manual]") {
  if (!cuda_m2l_p2p_available()) {
    SUCCEED("CUDA static P2P is unavailable");
    return;
  }

  std::vector<Vec3> positions;
  for (int index = 0; index < 24; ++index) {
    const double value = static_cast<double>(index);
    positions.push_back(
        {-0.9 + 1.8 * static_cast<double>((index * 17) % 29) / 28.0,
         -0.9 + 1.8 * static_cast<double>((index * 11) % 31) / 30.0,
         -0.9 + 1.8 * static_cast<double>((index * 7) % 37) / 36.0});
  }
  UniformTreeOptions tree_options;
  tree_options.max_level = 2;
  tree_options.root_centre = Vec3{};
  tree_options.root_half_width = 1.0;
  const UniformTree tree(positions, positions, tree_options);
  const auto nodes = tree.nodes();
  std::vector<std::array<int, 2>> interactions;
  std::vector<StaticP2PLeafPair> leaf_pairs;
  for (const int leaf_index : tree.occupied_target_leaves()) {
    const TreeNode &leaf_node = nodes[static_cast<std::size_t>(leaf_index)];
    for (const int neighbour_index : leaf_node.list1) {
      const TreeNode &neighbour =
          nodes[static_cast<std::size_t>(neighbour_index)];
      if (neighbour.source_count() == 0) {
        continue;
      }
      leaf_pairs.push_back({static_cast<int>(leaf_node.target_begin),
                            static_cast<int>(leaf_node.target_count()),
                            static_cast<int>(neighbour.source_begin),
                            static_cast<int>(neighbour.source_count())});
      for (std::size_t target = leaf_node.target_begin;
           target < leaf_node.target_end; ++target) {
        for (std::size_t source = neighbour.source_begin;
             source < neighbour.source_end; ++source) {
          interactions.push_back(
              {static_cast<int>(target), static_cast<int>(source)});
        }
      }
    }
  }

  const StaticP2POperator canonical =
      build_static_p2p_operator(tree.sorted_target_positions(),
                                tree.sorted_source_positions(), interactions);
  const StaticP2PCompactPlan compact = build_static_p2p_compact_plan(canonical);
  const StaticP2PLeafPlan leaf =
      build_static_p2p_leaf_plan(canonical, leaf_pairs);
  std::vector<int> identities(positions.size());
  std::iota(identities.begin(), identities.end(), 0);
  const StaticP2PSignedTensorDictionaryPlan dictionary =
      build_static_p2p_signed_tensor_dictionary_plan(
          canonical, leaf_pairs, identities);
  const StaticP2PBsrPlan bsr = build_static_p2p_bsr_plan(canonical, identities);
  std::vector<Vec3> moments(positions.size());
  for (std::size_t index = 0; index < moments.size(); ++index) {
    const double value = static_cast<double>(index);
    moments[index] = {std::sin(value), std::cos(value), std::sin(0.3 * value)};
  }
  std::vector<Vec3> expected(positions.size());
  apply_static_p2p_operator(canonical, moments, expected, identities);

  const auto verify = [&](CudaP2PPlan &plan) {
    std::vector<Vec3> actual(positions.size());
    plan.evaluate(moments, identities, actual);
    for (std::size_t target = 0; target < actual.size(); ++target) {
      REQUIRE(actual[target].x ==
              Catch::Approx(expected[target].x).margin(3.0e-11));
      REQUIRE(actual[target].y ==
              Catch::Approx(expected[target].y).margin(3.0e-11));
      REQUIRE(actual[target].z ==
              Catch::Approx(expected[target].z).margin(3.0e-11));
    }
    REQUIRE(plan.timings().kernel_seconds > 0.0);
    REQUIRE(plan.statistics().p2p_interaction_count == canonical.blocks.size());
    REQUIRE(plan.statistics().evaluation_h2d_bytes ==
            moments.size() * sizeof(Vec3));
  };

  CudaP2PPlan canonical_cuda(canonical, identities);
  verify(canonical_cuda);
  CudaP2PPlan compact_cuda(compact, identities);
  verify(compact_cuda);
  CudaP2PPlan leaf_cuda(leaf, identities);
  verify(leaf_cuda);
  // The warp-per-block leaf kernel stages nothing in shared memory.
  REQUIRE(leaf_cuda.statistics().p2p_scratch_bytes == 0);
  REQUIRE(leaf_cuda.statistics().p2p_leaf_metadata_bytes > 0);
  CudaP2PPlan bsr_cuda(bsr);
  verify(bsr_cuda);
  const auto verify_dictionary = [&](CudaP2PPlan &plan) {
    std::vector<Vec3> actual(positions.size());
    plan.evaluate(moments, identities, actual);
    for (std::size_t target = 0; target < actual.size(); ++target) {
      REQUIRE(actual[target].x ==
              Catch::Approx(expected[target].x).margin(3.0e-11));
      REQUIRE(actual[target].y ==
              Catch::Approx(expected[target].y).margin(3.0e-11));
      REQUIRE(actual[target].z ==
              Catch::Approx(expected[target].z).margin(3.0e-11));
    }
    REQUIRE(plan.timings().kernel_seconds > 0.0);
    REQUIRE(plan.statistics().p2p_interaction_count ==
            dictionary.token_count());
  };
  CudaP2PPlan dictionary_source_warp(dictionary);
  verify_dictionary(dictionary_source_warp);
  CudaP2PPlan dictionary_target_owned(dictionary, true, false);
  verify_dictionary(dictionary_target_owned);
  REQUIRE(dictionary_target_owned.statistics().p2p_scratch_bytes == 0);
  CudaP2PPlan dictionary_power2(dictionary, false, true);
  verify_dictionary(dictionary_power2);
  REQUIRE(dictionary_power2.statistics().p2p_threads_per_block == 128);
  CudaP2PPlan dictionary_both(dictionary, true, true);
  verify_dictionary(dictionary_both);
  REQUIRE(dictionary_both.statistics().p2p_threads_per_block == 256);
  std::vector<int> changed_identities(identities.size(), -1);
  std::vector<Vec3> rejected(positions.size());
  REQUIRE_THROWS_AS(bsr_cuda.evaluate(moments, changed_identities, rejected),
                    std::invalid_argument);
}

TEST_CASE("CUDA M2L/P2P hybrid agrees with CPU static", "[cuda][manual]")
{
    if (!cuda_m2l_p2p_available()) {
        SUCCEED("CUDA M2L/P2P is unavailable");
        return;
    }

    std::vector<Vec3> positions;
    std::vector<Vec3> moments;
    for (int index = 0; index < 48; ++index) {
        const double value = static_cast<double>(index);
        positions.push_back({
            -0.9 + 1.8 * static_cast<double>((index * 17) % 97) / 96.0,
            -0.9 + 1.8 * static_cast<double>((index * 31) % 89) / 88.0,
            -0.9 + 1.8 * static_cast<double>((index * 43) % 83) / 82.0
        });
        moments.push_back({std::sin(value), std::cos(0.7 * value),
                           std::sin(0.3 * value)});
    }
    std::vector<int> source_identities(positions.size());
    std::iota(source_identities.begin(), source_identities.end(), 0);

    UniformFmmOptions cpu_options;
    cpu_options.expansion_basis = ExpansionBasis::Cartesian;
    cpu_options.precision = StaticPrecision::Float64;
    cpu_options.expansion_order = 3;
    cpu_options.tree.max_level = 2;
    cpu_options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
    cpu_options.tree.root_half_width = 1.0;
    cpu_options.backend = ExecutionBackend::CpuStatic;
    UniformFmmOptions cuda_options = cpu_options;
    cuda_options.backend = ExecutionBackend::CudaM2LP2P;

    UniformFmm cpu(positions, positions, cpu_options);
    UniformFmm cuda(positions, positions, cuda_options);
    REQUIRE(cuda.static_plan_statistics().interactions > 0);
    REQUIRE(cuda.static_plan_statistics().p2p_interactions > 0);

    const auto expected = cpu.evaluate(
        moments,
        OutputFlags::Both,
        source_identities
    );
    const auto actual = cuda.evaluate(
        moments,
        OutputFlags::Both,
        source_identities
    );
    REQUIRE(cuda.cuda_plan_statistics().static_m2l_upload_count == 1);
    REQUIRE(cuda.cuda_plan_statistics().static_p2p_upload_count == 1);
    REQUIRE(cuda.cuda_plan_statistics().m2l_unique_matrix_count ==
            cuda.static_plan_statistics().m2l_operators);
    REQUIRE(cuda.cuda_plan_statistics().p2p_interaction_count ==
            cuda.static_plan_statistics().p2p_interactions);
    REQUIRE(cuda.last_timings().cuda_p2p_wait.calls == 1);
    REQUIRE(cuda.last_timings().cuda_p2p_kernel.calls == 1);
    for (std::size_t index = 0; index < actual.size(); ++index) {
        REQUIRE(actual[index].phi ==
                Catch::Approx(expected[index].phi).margin(2.0e-12));
        REQUIRE(actual[index].H.x ==
                Catch::Approx(expected[index].H.x).margin(2.0e-12));
        REQUIRE(actual[index].H.y ==
                Catch::Approx(expected[index].H.y).margin(2.0e-12));
        REQUIRE(actual[index].H.z ==
                Catch::Approx(expected[index].H.z).margin(2.0e-12));
    }

    const auto potential_expected = cpu.evaluate(
        moments, OutputFlags::Potential, source_identities
    );
    const auto potential_actual = cuda.evaluate(
        moments, OutputFlags::Potential, source_identities
    );
    REQUIRE(cuda.last_timings().cuda_p2p_wait.calls == 0);
    for (std::size_t index = 0; index < potential_actual.size(); ++index) {
        REQUIRE(potential_actual[index].phi == Catch::Approx(
            potential_expected[index].phi
        ).margin(2.0e-12));
    }

    moments.front().x += 0.25;
    const auto repeated_expected = cpu.evaluate(
        moments, OutputFlags::Field, source_identities
    );
    const auto repeated = cuda.evaluate(
        moments, OutputFlags::Field, source_identities
    );
    REQUIRE(repeated.size() == positions.size());
    REQUIRE(repeated.front().H.x == Catch::Approx(
        repeated_expected.front().H.x
    ).margin(2.0e-12));
    REQUIRE(cuda.cuda_plan_statistics().static_m2l_upload_count == 1);
    REQUIRE(cuda.cuda_plan_statistics().static_p2p_upload_count == 1);
}

TEST_CASE("CUDA power-of-two dictionary covers every microtile class",
          "[cuda][manual]")
{
    if (!cuda_m2l_p2p_available()) {
        SUCCEED("CUDA signed tensor-dictionary P2P is unavailable");
        return;
    }

    // These leaves exercise the requested irregular occupancies, including
    // 31 = 16 + 8 + 4 + 2 + 1 and 33 = 32 + 1.
    std::vector<Vec3> sources;
    std::vector<Vec3> targets;
    for (int source = 0; source < 9; ++source) {
        sources.push_back({-0.8 + 0.03 * source,
                           -0.4 + 0.02 * source,
                           0.1 + 0.01 * source});
    }
    constexpr std::array<int, 9> occupancies{{1, 2, 3, 5, 7,
                                                10, 17, 31, 33}};
    int target_count = 0;
    for (const int occupancy : occupancies) {
        target_count += occupancy;
    }
    for (int target = 0; target < target_count; ++target) {
        targets.push_back({0.2 + 0.01 * target,
                           -0.7 + 0.005 * target,
                           -0.3 + 0.004 * target});
    }

    std::vector<std::array<int, 2>> interactions;
    interactions.reserve(targets.size() * sources.size());
    for (int target = 0; target < static_cast<int>(targets.size()); ++target) {
        for (int source = 0; source < static_cast<int>(sources.size());
             ++source) {
            interactions.push_back({target, source});
        }
    }
    std::vector<StaticP2PLeafPair> leaf_pairs;
    leaf_pairs.reserve(occupancies.size());
    int target_begin = 0;
    for (const int occupancy : occupancies) {
        leaf_pairs.push_back({target_begin, occupancy, 0, 9});
        target_begin += occupancy;
    }
    const StaticP2POperator canonical = build_static_p2p_operator(
        targets, sources, interactions);
    const StaticP2PSignedTensorDictionaryPlan dictionary =
        build_static_p2p_signed_tensor_dictionary_plan(
            canonical, leaf_pairs);

    std::vector<Vec3> moments(sources.size());
    for (std::size_t source = 0; source < moments.size(); ++source) {
        const double value = static_cast<double>(source);
        moments[source] = {std::sin(value), std::cos(value),
                           std::sin(0.3 * value)};
    }
    const std::vector<int> no_identities(targets.size(), -1);
    std::vector<Vec3> expected(targets.size());
    apply_static_p2p_operator(canonical, moments, expected, no_identities);

    CudaP2PPlan source_warp(dictionary);
    CudaP2PPlan power2(dictionary, false, true);
    std::vector<Vec3> source_values(targets.size());
    std::vector<Vec3> power2_values(targets.size());
    source_warp.evaluate(moments, no_identities, source_values);
    power2.evaluate(moments, no_identities, power2_values);
    for (std::size_t target = 0; target < targets.size(); ++target) {
        REQUIRE(power2_values[target].x ==
                Catch::Approx(expected[target].x).margin(3.0e-11));
        REQUIRE(power2_values[target].y ==
                Catch::Approx(expected[target].y).margin(3.0e-11));
        REQUIRE(power2_values[target].z ==
                Catch::Approx(expected[target].z).margin(3.0e-11));
        REQUIRE(power2_values[target].x ==
                Catch::Approx(source_values[target].x).margin(3.0e-11));
        REQUIRE(power2_values[target].y ==
                Catch::Approx(source_values[target].y).margin(3.0e-11));
        REQUIRE(power2_values[target].z ==
                Catch::Approx(source_values[target].z).margin(3.0e-11));
    }
    REQUIRE(power2.statistics().p2p_threads_per_block == 128);
    REQUIRE(power2.statistics().p2p_scratch_bytes == 0);

    const FloatStaticP2PSignedTensorDictionaryPlan dictionary_float =
        quantise_static_p2p_signed_tensor_dictionary_plan(dictionary);
    CudaP2PPlan source_warp_float(dictionary_float);
    CudaP2PPlan power2_float(dictionary_float, false, true);
    std::vector<FloatVec3> float_moments(moments.size());
    for (std::size_t source = 0; source < moments.size(); ++source) {
        float_moments[source] = {
            static_cast<float>(moments[source].x),
            static_cast<float>(moments[source].y),
            static_cast<float>(moments[source].z)};
    }
    std::vector<FloatVec3> source_float_values(targets.size());
    std::vector<FloatVec3> power2_float_values(targets.size());
    source_warp_float.evaluate(float_moments, no_identities,
                               source_float_values);
    power2_float.evaluate(float_moments, no_identities,
                          power2_float_values);
    for (std::size_t target = 0; target < targets.size(); ++target) {
        REQUIRE(power2_float_values[target].x == Catch::Approx(
            source_float_values[target].x).margin(4.0e-5));
        REQUIRE(power2_float_values[target].y == Catch::Approx(
            source_float_values[target].y).margin(4.0e-5));
        REQUIRE(power2_float_values[target].z == Catch::Approx(
            source_float_values[target].z).margin(4.0e-5));
    }
}

TEST_CASE("CUDA partial and full share canonical static plan behaviour",
          "[cuda][manual]")
{
    if (!cuda_m2l_p2p_available() || !cuda_full_available()) {
        SUCCEED("Both CUDA FMM modes are required");
        return;
    }

    struct Scenario {
        int order;
        int depth;
        int particle_count;
        bool separate_targets;
    };
    const std::array<Scenario, 4> scenarios{{
        {2, 2, 24, false},
        {3, 3, 48, true},
        {3, 5, 64, false},
        {4, 2, 32, true},
    }};

    for (const Scenario scenario : scenarios) {
        CAPTURE(scenario.order, scenario.depth, scenario.particle_count,
                scenario.separate_targets);
        std::vector<Vec3> sources;
        std::vector<Vec3> targets;
        std::vector<Vec3> moments;
        std::vector<int> identities;
        for (int index = 0; index < scenario.particle_count; ++index) {
            const double value = static_cast<double>(index);
            const Vec3 source{
                -0.95 + 1.9 * static_cast<double>((index * 17) % 67) / 66.0,
                -0.95 + 1.9 * static_cast<double>((index * 29) % 71) / 70.0,
                -0.95 + 1.9 * static_cast<double>((index * 43) % 73) / 72.0
            };
            sources.push_back(source);
            targets.push_back(scenario.separate_targets
                                  ? Vec3{0.8 * source.x + 0.03,
                                         0.8 * source.y - 0.02,
                                         0.8 * source.z + 0.01}
                                  : source);
            moments.push_back({std::sin(value), std::cos(0.7 * value),
                               std::sin(0.3 * value)});
            identities.push_back(scenario.separate_targets ? -1 : index);
        }

        UniformFmmOptions options;
        options.expansion_basis = ExpansionBasis::Cartesian;
        options.precision = StaticPrecision::Float64;
        options.expansion_order = scenario.order;
        options.tree.max_level = scenario.depth;
        options.tree.root_centre = Vec3{};
        options.tree.root_half_width = 1.0;
        options.backend = ExecutionBackend::CpuStatic;
        UniformFmm cpu(sources, targets, options);
        options.fixed_target_source_indices = identities;
        options.backend = ExecutionBackend::CudaPartial;
        UniformFmm partial(sources, targets, options);
        options.backend = ExecutionBackend::CudaFull;
        UniformFmm full(sources, targets, options);

        // Point sources select the dense leaf-block packing, which keeps the
        // fixed identity map resident on the device.
        REQUIRE(partial.p2p_execution_packing() ==
                P2PExecutionPacking::LeafBlock);
        REQUIRE(full.p2p_execution_packing() == P2PExecutionPacking::LeafBlock);
        REQUIRE(partial.cuda_plan_statistics().p2p_identity_bytes ==
                targets.size() * sizeof(int));
        REQUIRE(full.cuda_plan_statistics().p2p_identity_bytes ==
                targets.size() * sizeof(int));

        const auto expected =
            cpu.evaluate(moments, OutputFlags::Field, identities);
        const auto partial_values =
            partial.evaluate(moments, OutputFlags::Field, identities);
        const auto full_values =
            full.evaluate(moments, OutputFlags::Field, identities);
        for (std::size_t target = 0; target < targets.size(); ++target) {
            REQUIRE(partial_values[target].H.x ==
                    Catch::Approx(expected[target].H.x).margin(3.0e-11));
            REQUIRE(partial_values[target].H.y ==
                    Catch::Approx(expected[target].H.y).margin(3.0e-11));
            REQUIRE(partial_values[target].H.z ==
                    Catch::Approx(expected[target].H.z).margin(3.0e-11));
            REQUIRE(full_values[target].H.x ==
                    Catch::Approx(expected[target].H.x).margin(3.0e-11));
            REQUIRE(full_values[target].H.y ==
                    Catch::Approx(expected[target].H.y).margin(3.0e-11));
            REQUIRE(full_values[target].H.z ==
                    Catch::Approx(expected[target].H.z).margin(3.0e-11));
        }

        const CudaPlanStatistics partial_statistics =
            partial.cuda_plan_statistics();
        const CudaPlanStatistics full_statistics = full.cuda_plan_statistics();
        REQUIRE(partial_statistics.m2l_unique_matrix_count ==
                full_statistics.m2l_unique_matrix_count);
        REQUIRE(partial_statistics.m2l_matrix_bytes ==
                full_statistics.m2l_matrix_bytes);
        REQUIRE(partial_statistics.m2l_interaction_metadata_bytes ==
                full_statistics.m2l_interaction_metadata_bytes);
        REQUIRE(partial_statistics.m2l_interaction_count ==
                full_statistics.m2l_interaction_count);
        REQUIRE(partial_statistics.m2l_active_row_count ==
                full_statistics.m2l_active_row_count);
        REQUIRE(partial_statistics.m2l_threads_per_block ==
                full_statistics.m2l_threads_per_block);
        REQUIRE(partial_statistics.p2p_interaction_count ==
                full_statistics.p2p_interaction_count);
        REQUIRE(partial_statistics.static_m2l_upload_count == 1);
        REQUIRE(partial_statistics.static_p2p_upload_count == 1);
        REQUIRE(full_statistics.static_m2l_upload_count == 1);
        REQUIRE(full_statistics.static_p2p_upload_count == 1);
        REQUIRE(full_statistics.evaluation_h2d_bytes ==
                moments.size() * sizeof(Vec3));
        REQUIRE(full_statistics.evaluation_d2h_bytes ==
                targets.size() * sizeof(Vec3));

        const std::size_t coefficient_bytes =
            partial.tree().nodes().size() *
            static_cast<std::size_t>(partial.basis().size()) * sizeof(double);
        const std::size_t scaling_bytes =
            2 * static_cast<std::size_t>(scenario.depth + 1) *
            partial.basis().size() * sizeof(double);
        // The selected CUDA BSR packing expands each symmetric six-value
        // tensor to a full 3 x 3 block and stores source indices separately.
        // Derive the upload size from the reported packing categories rather
        // than assuming the canonical StaticDipoleBlock representation.
        const std::size_t p2p_static_bytes =
            partial_statistics.p2p_tensor_bytes +
            partial_statistics.p2p_index_bytes +
            partial_statistics.p2p_row_metadata_bytes +
            partial_statistics.p2p_leaf_metadata_bytes +
            partial_statistics.p2p_identity_bytes;
        const std::size_t canonical_m2l_bytes =
            partial_statistics.m2l_matrix_bytes +
            partial_statistics.m2l_interaction_metadata_bytes + scaling_bytes;
        REQUIRE(partial_statistics.setup_h2d_bytes ==
                canonical_m2l_bytes + p2p_static_bytes);
        REQUIRE(partial_statistics.evaluation_h2d_bytes ==
                coefficient_bytes + moments.size() * sizeof(Vec3));
        REQUIRE(partial_statistics.evaluation_d2h_bytes ==
                coefficient_bytes + targets.size() * sizeof(Vec3));
        REQUIRE(partial_statistics.persistent_device_bytes ==
                partial_statistics.setup_h2d_bytes + 2 * coefficient_bytes +
                    sources.size() * sizeof(Vec3) +
                    targets.size() * sizeof(Vec3) +
                    partial_statistics.m2l_scratch_bytes);
        REQUIRE(full_statistics.persistent_device_bytes ==
                full_statistics.setup_h2d_bytes + 2 * coefficient_bytes +
                    2 * sources.size() * sizeof(Vec3) +
                    3 * targets.size() * sizeof(Vec3) +
                    full_statistics.m2l_scratch_bytes);
        REQUIRE(partial.last_timings().m2l_scale.calls == 1);
        REQUIRE(partial.last_timings().m2l_multiply.calls == 1);
        REQUIRE(partial.last_timings().m2l_gather.calls == 0);
        REQUIRE(partial.last_timings().m2l_scatter.calls == 0);
        REQUIRE(full.last_timings().m2l_multiply.calls == 1);
        REQUIRE(full.last_timings().m2l_scale.calls == 1);
        REQUIRE(full.last_timings().m2l_gather.calls == 0);
        REQUIRE(full.last_timings().m2l_scatter.calls == 0);

        moments.front().x += 0.125;
        const auto partial_repeated =
            partial.evaluate(moments, OutputFlags::Field, identities);
        const auto full_repeated =
            full.evaluate(moments, OutputFlags::Field, identities);
        REQUIRE(partial_repeated.size() == targets.size());
        REQUIRE(full_repeated.size() == targets.size());
        REQUIRE(partial.cuda_plan_statistics().static_m2l_upload_count == 1);
        REQUIRE(partial.cuda_plan_statistics().static_p2p_upload_count == 1);
        REQUIRE(full.cuda_plan_statistics().static_m2l_upload_count == 1);
        REQUIRE(full.cuda_plan_statistics().static_p2p_upload_count == 1);
        REQUIRE(partial.cuda_plan_statistics().evaluation_h2d_calls == 4);
        REQUIRE(partial.cuda_plan_statistics().evaluation_d2h_calls == 4);
        REQUIRE(full.cuda_plan_statistics().evaluation_h2d_calls == 2);
        REQUIRE(full.cuda_plan_statistics().evaluation_d2h_calls == 2);
    }
}

namespace {

// 8 x 8 x 8 lattice inside the unit cube: a regular grid whose leaves hold
// 8 points at depth 2 and 64 points at depth 1.
std::vector<Vec3> lattice_positions() {
  std::vector<Vec3> positions;
  for (int z = 0; z < 8; ++z) {
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 8; ++x) {
        positions.push_back({-0.875 + 0.25 * x, -0.875 + 0.25 * y,
                             -0.875 + 0.25 * z});
      }
    }
  }
  return positions;
}

std::vector<Vec3> lattice_moments(const std::size_t count) {
  std::vector<Vec3> moments;
  for (std::size_t index = 0; index < count; ++index) {
    const double value = static_cast<double>(index);
    moments.push_back({std::sin(0.3 * value), std::cos(0.7 * value),
                       std::sin(0.11 * value + 0.5)});
  }
  return moments;
}

UniformFmmOptions lattice_cuda_options(const ExecutionBackend backend,
                                       const int depth,
                                       const std::vector<int> &identities) {
  UniformFmmOptions options;
  options.expansion_basis = ExpansionBasis::Spherical;
  options.expansion_order = 4;
  options.tree.max_level = depth;
  options.tree.root_centre = Vec3{};
  options.tree.root_half_width = 1.0;
  options.backend = backend;
  options.precision = StaticPrecision::Float64;
  options.fixed_target_source_indices = identities;
  options.enable_cache = false;
  return options;
}

void require_fields_match(const std::vector<PotentialField> &actual,
                          const std::vector<PotentialField> &expected,
                          const double margin) {
  REQUIRE(actual.size() == expected.size());
  for (std::size_t target = 0; target < actual.size(); ++target) {
    REQUIRE(actual[target].H.x == Catch::Approx(expected[target].H.x).margin(margin));
    REQUIRE(actual[target].H.y == Catch::Approx(expected[target].H.y).margin(margin));
    REQUIRE(actual[target].H.z == Catch::Approx(expected[target].H.z).margin(margin));
  }
}

} // namespace

TEST_CASE("CUDA execution policy resolves the P2P packing from layout and options",
          "[cuda][policy]") {
  using cdfmm::cuda_policy::CudaDictionaryExecutor;
  using cdfmm::cuda_policy::CudaExecutionPolicyInputs;
  using cdfmm::cuda_policy::CudaP2PPacking;
  using cdfmm::cuda_policy::resolve_cuda_execution_policy;

  REQUIRE(UniformFmmOptions{}.spatial_layout == SpatialLayout::General);

  CudaExecutionPolicyInputs inputs;
  inputs.cuda_backend = true;
  inputs.effective_point_source = true;
  inputs.fixed_identity_available = true;
  inputs.target_count = 4096;
  inputs.occupied_target_leaf_count = 512;
  inputs.mean_leaf_occupancy = 8.0;
  inputs.bsr_budget_bytes = 1ULL << 40;

  SECTION("General point sources keep the leaf-block default") {
    REQUIRE(resolve_cuda_execution_policy(inputs).p2p_packing ==
            CudaP2PPacking::LeafBlock);
    inputs.fixed_identity_available = false;
    REQUIRE(resolve_cuda_execution_policy(inputs).p2p_packing ==
            CudaP2PPacking::LeafBlock);
  }
  SECTION("RegularGrid selects the dictionary and an occupancy-matched executor") {
    inputs.spatial_layout = SpatialLayout::RegularGrid;
    const auto low = resolve_cuda_execution_policy(inputs);
    REQUIRE(low.p2p_packing == CudaP2PPacking::SignedDictionary);
    REQUIRE(low.dictionary_from_layout);
    REQUIRE(low.dictionary_executor ==
            CudaDictionaryExecutor::PowerOfTwoMicrotiles);
    inputs.mean_leaf_occupancy = 64.0;
    const auto medium = resolve_cuda_execution_policy(inputs);
    REQUIRE(medium.p2p_packing == CudaP2PPacking::SignedDictionary);
    REQUIRE(medium.dictionary_executor == CudaDictionaryExecutor::TargetOwned);
    inputs.mean_leaf_occupancy = 128.0;
    const auto high = resolve_cuda_execution_policy(inputs);
    REQUIRE(high.p2p_packing == CudaP2PPacking::SignedDictionary);
    REQUIRE(high.dictionary_from_layout);
    REQUIRE(high.dictionary_executor == CudaDictionaryExecutor::SourceWarp);
    // The regime boundaries are the calibrated occupancy limits.
    inputs.mean_leaf_occupancy =
        cdfmm::cuda_policy::dictionary_microtile_occupancy_limit();
    REQUIRE(resolve_cuda_execution_policy(inputs).dictionary_executor ==
            CudaDictionaryExecutor::TargetOwned);
    inputs.mean_leaf_occupancy =
        cdfmm::cuda_policy::dictionary_source_warp_occupancy_limit();
    REQUIRE(resolve_cuda_execution_policy(inputs).dictionary_executor ==
            CudaDictionaryExecutor::SourceWarp);
    REQUIRE(cdfmm::cuda_policy::dictionary_microtile_occupancy_limit() <
            cdfmm::cuda_policy::dictionary_source_warp_occupancy_limit());
  }
  SECTION("RegularGrid without identity keeps the defaults; CPU backends share the lattice rule") {
    inputs.spatial_layout = SpatialLayout::RegularGrid;
    inputs.fixed_identity_available = false;
    REQUIRE(resolve_cuda_execution_policy(inputs).p2p_packing ==
            CudaP2PPacking::LeafBlock);
    inputs.fixed_identity_available = true;
    inputs.cuda_backend = false;
    REQUIRE(resolve_cuda_execution_policy(inputs).p2p_packing ==
            CudaP2PPacking::SignedDictionary);
  }
  SECTION("periodicity does not restrict any stored-tensor packing") {
    // Image records are ordinary dense leaf pairs or merged sparse blocks,
    // so periodic plans follow the free-space rules.
    inputs.periodic = true;
    REQUIRE(resolve_cuda_execution_policy(inputs).p2p_packing ==
            CudaP2PPacking::LeafBlock);
    inputs.spatial_layout = SpatialLayout::RegularGrid;
    REQUIRE(resolve_cuda_execution_policy(inputs).p2p_packing ==
            CudaP2PPacking::SignedDictionary);
    inputs.spatial_layout = SpatialLayout::General;
    inputs.effective_point_source = false;
    inputs.bsr_estimate_bytes = 1024;
    REQUIRE(resolve_cuda_execution_policy(inputs).p2p_packing ==
            CudaP2PPacking::LeafBlock);
    for (const CudaP2PPacking packing :
         {CudaP2PPacking::CanonicalRows, CudaP2PPacking::LeafBlock,
          CudaP2PPacking::Bsr3, CudaP2PPacking::SignedDictionary}) {
      REQUIRE(cdfmm::cuda_policy::explicit_packing_rejection(inputs, packing) ==
              nullptr);
    }
    inputs.effective_point_source = true;
    inputs.fixed_identity_available = false;
    REQUIRE(cdfmm::cuda_policy::explicit_packing_rejection(
                inputs, CudaP2PPacking::Bsr3) != nullptr);
    REQUIRE(cdfmm::cuda_policy::explicit_packing_rejection(
                inputs, CudaP2PPacking::LeafBlock) == nullptr);
  }
  SECTION("finite sources follow the same rules as points") {
    // Leaf blocks are the general default for every geometry (measured
    // faster than BSR(3) on finite bodies once the leaf packing carried the
    // identity metadata); the BSR budget no longer steers the policy, and the
    // lattice hint selects the dictionary for finite bodies too.
    inputs.effective_point_source = false;
    inputs.fixed_identity_available = false;
    inputs.bsr_estimate_bytes = 1024;
    REQUIRE(resolve_cuda_execution_policy(inputs).p2p_packing ==
            CudaP2PPacking::LeafBlock);
    inputs.bsr_budget_bytes = 0;
    REQUIRE(resolve_cuda_execution_policy(inputs).p2p_packing ==
            CudaP2PPacking::LeafBlock);
    inputs.spatial_layout = SpatialLayout::RegularGrid;
    const auto lattice = resolve_cuda_execution_policy(inputs);
    REQUIRE(lattice.p2p_packing == CudaP2PPacking::SignedDictionary);
    REQUIRE(lattice.dictionary_from_layout);
    REQUIRE(cdfmm::cuda_policy::dictionary_layout_max_token_width_bytes() == 2);
  }
  SECTION("explicit options win over the layout hint and keep their meaning") {
    inputs.spatial_layout = SpatialLayout::RegularGrid;
    inputs.mean_leaf_occupancy = 128.0;
    inputs.explicit_dictionary_power2_microtiles = true;
    REQUIRE(resolve_cuda_execution_policy(inputs).dictionary_executor ==
            CudaDictionaryExecutor::PowerOfTwoMicrotiles);
    inputs.explicit_dictionary_target_owned = true;
    REQUIRE(resolve_cuda_execution_policy(inputs).dictionary_executor ==
            CudaDictionaryExecutor::TargetOwned);
    // Explicit reduced symmetry on a general layout: dictionary with the
    // documented source-warp default when no executor flag is set.
    inputs.spatial_layout = SpatialLayout::General;
    inputs.explicit_dictionary_target_owned = false;
    inputs.explicit_dictionary_power2_microtiles = false;
    inputs.explicit_reduced_symmetry = true;
    const auto explicit_policy = resolve_cuda_execution_policy(inputs);
    REQUIRE(explicit_policy.p2p_packing == CudaP2PPacking::SignedDictionary);
    REQUIRE(!explicit_policy.dictionary_from_layout);
    REQUIRE(explicit_policy.dictionary_executor ==
            CudaDictionaryExecutor::SourceWarp);
  }
  SECTION("tuning rules are the measured Phase-3A thresholds") {
    REQUIRE(cdfmm::cuda_policy::m2l_pairs_per_thread(StaticPrecision::Float32,
                                                      1000000) == 16);
    REQUIRE(cdfmm::cuda_policy::m2l_pairs_per_thread(StaticPrecision::Float32,
                                                      1000) == 8);
    REQUIRE(cdfmm::cuda_policy::m2l_pairs_per_thread(StaticPrecision::Float64,
                                                      1000000) == 8);
    REQUIRE(cdfmm::cuda_policy::translation_lanes_for_outputs(1000) == 32);
    REQUIRE(cdfmm::cuda_policy::translation_lanes_for_outputs(1000000) == 4);
  }
}

TEST_CASE("regular-grid layout hint selects the dictionary on CUDA plans",
          "[cuda][policy][manual]") {
  if (!cuda_m2l_p2p_available() || !cuda_full_available()) {
    SUCCEED("Both CUDA FMM modes are required");
    return;
  }
  const std::vector<Vec3> positions = lattice_positions();
  const std::vector<Vec3> moments = lattice_moments(positions.size());
  std::vector<int> identities(positions.size());
  std::iota(identities.begin(), identities.end(), 0);

  for (const int depth : {1, 2}) {
    UniformFmmOptions cpu_options =
        lattice_cuda_options(ExecutionBackend::CpuStatic, depth, identities);
    UniformFmm reference(positions, positions, cpu_options);
    const auto expected =
        reference.evaluate(moments, OutputFlags::Field, identities);

    for (const ExecutionBackend backend :
         {ExecutionBackend::CudaPartial, ExecutionBackend::CudaFull}) {
      CAPTURE(depth, backend);
      UniformFmmOptions options =
          lattice_cuda_options(backend, depth, identities);

      UniformFmm general(positions, positions, options);
      REQUIRE(general.spatial_layout() == SpatialLayout::General);
      REQUIRE(general.p2p_execution_packing() ==
              P2PExecutionPacking::LeafBlock);
      require_fields_match(
          general.evaluate(moments, OutputFlags::Field, identities), expected,
          3.0e-11);

      options.spatial_layout = SpatialLayout::RegularGrid;
      UniformFmm regular(positions, positions, options);
      REQUIRE(regular.spatial_layout() == SpatialLayout::RegularGrid);
      REQUIRE(regular.p2p_execution_packing() ==
              P2PExecutionPacking::TensorDictionary);
      // Automatic executors: 64 targets per leaf at depth 1 select the
      // target-owned kernel (256 threads per block), 8 per leaf at depth 2
      // the power-of-two microtiles (128 threads per block).
      const std::size_t automatic_threads =
          regular.cuda_plan_statistics().p2p_threads_per_block;
      REQUIRE(automatic_threads == (depth == 1 ? 256U : 128U));
      require_fields_match(
          regular.evaluate(moments, OutputFlags::Field, identities), expected,
          3.0e-11);

      // An explicit executor request wins over the automatic choice.
      options.cuda_dictionary_target_owned = true;
      UniformFmm owned(positions, positions, options);
      REQUIRE(owned.p2p_execution_packing() ==
              P2PExecutionPacking::TensorDictionary);
      REQUIRE(owned.cuda_plan_statistics().p2p_threads_per_block == 256);
      options.cuda_dictionary_target_owned = false;
      options.cuda_dictionary_power2_microtiles = true;
      UniformFmm power2(positions, positions, options);
      REQUIRE(power2.cuda_plan_statistics().p2p_threads_per_block == 128);
      require_fields_match(
          power2.evaluate(moments, OutputFlags::Field, identities), expected,
          3.0e-11);

      // The explicit reduced-symmetry option keeps its source-warp default
      // whatever the layout says.
      UniformFmmOptions explicit_options =
          lattice_cuda_options(backend, depth, identities);
      explicit_options.use_reduced_symmetry_p2p = true;
      UniformFmm explicit_dictionary(positions, positions, explicit_options);
      REQUIRE(explicit_dictionary.p2p_execution_packing() ==
              P2PExecutionPacking::TensorDictionary);
      REQUIRE(explicit_dictionary.cuda_plan_statistics().p2p_threads_per_block ==
              (depth == 2 ? 32 : 64));

      // Without a fixed identity map the dictionary is not valid: leaf block.
      UniformFmmOptions dynamic_options = options;
      dynamic_options.fixed_target_source_indices.reset();
      dynamic_options.cuda_dictionary_power2_microtiles = false;
      UniformFmm dynamic(positions, positions, dynamic_options);
      REQUIRE(dynamic.p2p_execution_packing() == P2PExecutionPacking::LeafBlock);
    }
  }

  // FP32 through the regular-grid dictionary agrees with the FP64 reference.
  UniformFmmOptions fp32_options =
      lattice_cuda_options(ExecutionBackend::CudaFull, 2, identities);
  fp32_options.spatial_layout = SpatialLayout::RegularGrid;
  fp32_options.precision = StaticPrecision::Float32;
  UniformFmm fp32(positions, positions, fp32_options);
  REQUIRE(fp32.p2p_execution_packing() == P2PExecutionPacking::TensorDictionary);
  UniformFmmOptions fp64_options =
      lattice_cuda_options(ExecutionBackend::CpuStatic, 2, identities);
  UniformFmm fp64(positions, positions, fp64_options);
  const auto expected = fp64.evaluate(moments, OutputFlags::Field, identities);
  const auto actual = fp32.evaluate(moments, OutputFlags::Field, identities);
  double numerator = 0.0;
  double denominator = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const Vec3 difference = actual[index].H - expected[index].H;
    numerator += difference.x * difference.x + difference.y * difference.y +
                 difference.z * difference.z;
    numerator += 0.0;
    const Vec3 reference = expected[index].H;
    denominator += reference.x * reference.x + reference.y * reference.y +
                   reference.z * reference.z;
  }
  REQUIRE(std::sqrt(numerator / denominator) < 3.0e-5);
}

TEST_CASE("regular-grid layout hint keeps finite and periodic CUDA policies",
          "[cuda][policy][manual]") {
  if (!cuda_m2l_p2p_available()) {
    SUCCEED("CUDA M2L/P2P is unavailable");
    return;
  }
  const std::vector<Vec3> positions = lattice_positions();
  std::vector<int> identities(positions.size());
  std::iota(identities.begin(), identities.end(), 0);

  // Finite (prism) sources on a lattice: the hint selects the dictionary for
  // finite bodies too (identical prisms compress like a point lattice).
  UniformFmmOptions finite;
  finite.expansion_basis = ExpansionBasis::Cartesian;
  finite.expansion_order = 2;
  finite.tree.max_level = 1;
  finite.tree.root_centre = Vec3{};
  finite.tree.root_half_width = 1.0;
  finite.backend = ExecutionBackend::CudaPartial;
  finite.precision = StaticPrecision::Float64;
  finite.source_geometry = SourceGeometry::RectangularPrism;
  finite.source_sizes = {CuboidSize{0.1, 0.08, 0.06}};
  finite.spatial_layout = SpatialLayout::RegularGrid;
  finite.enable_cache = false;
  UniformFmm finite_plan(positions, positions, finite);
  REQUIRE(finite_plan.p2p_execution_packing() ==
          P2PExecutionPacking::TensorDictionary);
  finite.spatial_layout = SpatialLayout::General;
  UniformFmm finite_general(positions, positions, finite);
  REQUIRE(finite_general.p2p_execution_packing() ==
          P2PExecutionPacking::LeafBlock);

  // Periodic point sources follow the free-space rules: leaf blocks on the
  // general layout, the dictionary on a regular lattice with a fixed identity
  // map. Image records are dense leaf pairs like any other, so both must
  // agree with the CPU rows.
  UniformFmmOptions periodic =
      lattice_cuda_options(ExecutionBackend::CudaPartial, 2, identities);
  periodic.periodic.enabled = true;
  periodic.periodic.centre = Vec3{};
  periodic.periodic.lengths = {2.0, 2.0, 2.0};
  UniformFmmOptions periodic_general = periodic;
  periodic.spatial_layout = SpatialLayout::RegularGrid;
  UniformFmm periodic_regular(positions, positions, periodic);
  UniformFmm periodic_plain(positions, positions, periodic_general);
  REQUIRE(periodic_plain.p2p_execution_packing() ==
          P2PExecutionPacking::LeafBlock);
  REQUIRE(periodic_regular.p2p_execution_packing() ==
          P2PExecutionPacking::TensorDictionary);
  UniformFmmOptions periodic_cpu = periodic_general;
  periodic_cpu.backend = ExecutionBackend::CpuStatic;
  UniformFmm periodic_rows(positions, positions, periodic_cpu);
  std::vector<Vec3> moments(positions.size());
  for (std::size_t index = 0; index < moments.size(); ++index) {
    const double value = static_cast<double>(index);
    moments[index] = {std::sin(value), std::cos(value), std::sin(0.3 * value)};
  }
  const auto expected = periodic_rows.evaluate(moments, OutputFlags::Field,
                                               identities);
  for (UniformFmm *plan : {&periodic_regular, &periodic_plain}) {
    const auto actual = plan->evaluate(moments, OutputFlags::Field, identities);
    for (std::size_t target = 0; target < actual.size(); ++target) {
      REQUIRE(actual[target].H.x ==
              Catch::Approx(expected[target].H.x).margin(1.0e-9));
      REQUIRE(actual[target].H.y ==
              Catch::Approx(expected[target].H.y).margin(1.0e-9));
      REQUIRE(actual[target].H.z ==
              Catch::Approx(expected[target].H.z).margin(1.0e-9));
    }
  }
}

TEST_CASE("CUDA tensor packings retain finite self fields under an identity map",
          "[cuda][manual][identity]") {
  if (!cuda_m2l_p2p_available()) {
    SUCCEED("CUDA static P2P is unavailable");
    return;
  }
  // Two rectangular prisms that are both sources and targets, evaluated with
  // an identity map. Finite sources carry no identity marker, so every packed
  // executor must keep the physical self tensor; the same packings built from
  // point sources must omit the singular self pair. Only the canonical
  // metadata distinguishes the two cases.
  const std::vector<Vec3> positions{{-0.15, 0.02, 0.01}, {0.17, -0.03, 0.02}};
  const std::array<CuboidSize, 1> sizes{{{0.2, 0.18, 0.16}}};
  const std::vector<std::array<int, 2>> interactions{
      {0, 0}, {0, 1}, {1, 0}, {1, 1}};
  const std::vector<StaticP2PLeafPair> leaf_pairs{{0, 2, 0, 2}};
  const std::vector<int> identities{0, 1};
  const std::vector<Vec3> moments{{0.7, -0.4, 0.2}, {-0.3, 0.5, 0.9}};

  for (const SourceGeometry source_geometry :
       {SourceGeometry::PointDipole, SourceGeometry::RectangularPrism}) {
    const std::span<const CuboidSize> source_sizes =
        source_geometry == SourceGeometry::PointDipole
            ? std::span<const CuboidSize>{}
            : std::span<const CuboidSize>(sizes);
    const StaticP2POperator canonical = build_static_p2p_operator(
        positions, positions, interactions, source_geometry, source_sizes,
        TargetGeometry::RectangularPrism, sizes);
    std::vector<Vec3> expected(positions.size());
    apply_static_p2p_operator(canonical, moments, expected, identities);

    const auto verify = [&](CudaP2PPlan &plan) {
      std::vector<Vec3> actual(positions.size());
      plan.evaluate(moments, identities, actual);
      for (std::size_t target = 0; target < actual.size(); ++target) {
        REQUIRE(actual[target].x ==
                Catch::Approx(expected[target].x).margin(2.0e-12));
        REQUIRE(actual[target].y ==
                Catch::Approx(expected[target].y).margin(2.0e-12));
        REQUIRE(actual[target].z ==
                Catch::Approx(expected[target].z).margin(2.0e-12));
      }
    };
    CudaP2PPlan canonical_cuda(canonical, identities);
    verify(canonical_cuda);
    CudaP2PPlan compact_cuda(build_static_p2p_compact_plan(canonical),
                             identities);
    verify(compact_cuda);
    CudaP2PPlan leaf_cuda(build_static_p2p_leaf_plan(canonical, leaf_pairs),
                          identities);
    verify(leaf_cuda);
    CudaP2PPlan bsr_cuda(build_static_p2p_bsr_plan(canonical, identities));
    verify(bsr_cuda);
    const StaticP2PSignedTensorDictionaryPlan dictionary =
        build_static_p2p_signed_tensor_dictionary_plan(canonical, leaf_pairs,
                                                       identities);
    CudaP2PPlan source_warp(dictionary);
    verify(source_warp);
    CudaP2PPlan target_owned(dictionary, true, false);
    verify(target_owned);
    CudaP2PPlan microtiles(dictionary, false, true);
    verify(microtiles);
    // Dynamic identities on the canonical, compact and leaf plans behave
    // the same way when the map arrives per evaluation.
    CudaP2PPlan dynamic_leaf(build_static_p2p_leaf_plan(canonical, leaf_pairs));
    verify(dynamic_leaf);
  }
}

TEST_CASE("explicit P2P packing requests are honoured by both CUDA backends",
          "[cuda][manual][packing]") {
  if (!cuda_m2l_p2p_available() || !cuda_full_available()) {
    SUCCEED("CUDA backends are unavailable");
    return;
  }
  std::vector<Vec3> positions;
  for (int index = 0; index < 96; ++index) {
    positions.push_back(
        {-0.9 + 1.8 * static_cast<double>((index * 17) % 29) / 28.0,
         -0.9 + 1.8 * static_cast<double>((index * 11) % 31) / 30.0,
         -0.9 + 1.8 * static_cast<double>((index * 7) % 37) / 36.0});
  }
  std::vector<Vec3> moments(positions.size());
  for (std::size_t index = 0; index < moments.size(); ++index) {
    const double value = static_cast<double>(index);
    moments[index] = {std::sin(value), std::cos(1.3 * value),
                      std::sin(0.7 * value)};
  }
  std::vector<int> identities(positions.size());
  std::iota(identities.begin(), identities.end(), 0);

  const auto require_close = [&](const std::vector<PotentialField> &actual,
                                 const std::vector<PotentialField> &expected,
                                 const double tolerance) {
    double scale = 0.0;
    for (const PotentialField &value : expected) {
      scale = std::max({scale, std::abs(value.H.x), std::abs(value.H.y),
                        std::abs(value.H.z)});
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
      REQUIRE(std::abs(actual[index].H.x - expected[index].H.x) <=
              tolerance * scale);
      REQUIRE(std::abs(actual[index].H.y - expected[index].H.y) <=
              tolerance * scale);
      REQUIRE(std::abs(actual[index].H.z - expected[index].H.z) <=
              tolerance * scale);
    }
  };

  for (const SourceGeometry geometry :
       {SourceGeometry::PointDipole, SourceGeometry::RectangularPrism}) {
    UniformFmmOptions options;
    options.backend = ExecutionBackend::CpuStatic;
    options.precision = StaticPrecision::Float64;
    options.expansion_order = 3;
    options.tree.max_level = 2;
    options.enable_cache = false;
    options.source_geometry = geometry;
    if (geometry == SourceGeometry::RectangularPrism) {
      options.source_sizes = {RectangularPrism{0.04, 0.03, 0.05}};
      options.far_field_source_model = SourceModel::PointDipole;
    } else {
      options.fixed_target_source_indices = identities;
    }
    UniformFmm reference(positions, positions, options);
    const auto expected = reference.evaluate(moments, OutputFlags::Field);

    for (const ExecutionBackend backend :
         {ExecutionBackend::CudaPartial, ExecutionBackend::CudaFull}) {
      for (const P2PExecutionPacking packing :
           {P2PExecutionPacking::CanonicalAos, P2PExecutionPacking::LeafBlock,
            P2PExecutionPacking::CudaBsr3,
            P2PExecutionPacking::TensorDictionary}) {
        options.backend = backend;
        options.p2p_packing = packing;
        options.cuda_p2p_bsr_max_bytes = 0; // explicit BSR ignores the budget
        UniformFmm forced(positions, positions, options);
        REQUIRE(forced.requested_p2p_packing() == packing);
        REQUIRE(forced.p2p_execution_packing() == packing);
        require_close(forced.evaluate(moments, OutputFlags::Field), expected,
                      1.0e-11);
      }
    }
  }

  // CPU-only packings and identity-dependent packings without a fixed map
  // are rejected with the reason.
  const auto require_rejection = [&](const UniformFmmOptions &options,
                                     const char *fragment) {
    try {
      UniformFmm fmm(positions, positions, options);
      FAIL("construction should have rejected the packing request");
    } catch (const std::invalid_argument &error) {
      const std::string message = error.what();
      REQUIRE(message.find(fragment) != std::string::npos);
    }
  };
  UniformFmmOptions options;
  options.backend = ExecutionBackend::CudaPartial;
  options.enable_cache = false;
  options.p2p_packing = P2PExecutionPacking::PointGeometry;
  require_rejection(options, "CPU position-based executor");
  options.p2p_packing = P2PExecutionPacking::ParticleRowSoa;
  require_rejection(options, "CPU row packing");
  options.p2p_packing = P2PExecutionPacking::CudaBsr3;
  require_rejection(options, "fixed_target_source_indices");
  options.p2p_packing = P2PExecutionPacking::TensorDictionary;
  require_rejection(options, "fixed_target_source_indices");
}

TEST_CASE("CUDA BSR memory budget selects the canonical fallback",
          "[cuda][manual]") {
  if (!cuda_m2l_p2p_available()) {
    SUCCEED("CUDA M2L/P2P is unavailable");
    return;
  }

  // Leaf blocks are the general default for finite and point sources alike;
  // the BSR memory budget no longer steers the automatic policy, and BSR(3)
  // remains available as an explicit packing (its finite self fields are
  // physical, so no identity map is needed).
  const std::vector<Vec3> positions{
      {-0.5, 0.0, 0.0}, {0.25, 0.1, -0.2}, {0.4, -0.3, 0.2}};
  UniformFmmOptions options;
  options.expansion_basis = ExpansionBasis::Cartesian;
  options.expansion_order = 2;
  options.backend = ExecutionBackend::CudaPartial;
  options.tree.max_level = 0;
  options.source_geometry = SourceGeometry::RectangularPrism;
  options.source_sizes = {CuboidSize{0.05, 0.04, 0.03}};
  options.cuda_p2p_bsr_max_bytes = 0;

  UniformFmm fmm(positions, positions, options);

  REQUIRE(fmm.p2p_execution_packing() == P2PExecutionPacking::LeafBlock);
  REQUIRE(fmm.cuda_plan_statistics().p2p_identity_bytes ==
          positions.size() * sizeof(int));

  options.cuda_p2p_bsr_max_bytes = 20ULL * 1024ULL * 1024ULL * 1024ULL;
  UniformFmm within_budget(positions, positions, options);
  REQUIRE(within_budget.p2p_execution_packing() ==
          P2PExecutionPacking::LeafBlock);
  options.p2p_packing = P2PExecutionPacking::CudaBsr3;
  options.cuda_p2p_bsr_max_bytes = 0;
  UniformFmm explicit_bsr(positions, positions, options);
  REQUIRE(explicit_bsr.p2p_execution_packing() ==
          P2PExecutionPacking::CudaBsr3);
  REQUIRE(explicit_bsr.cuda_plan_statistics().p2p_identity_bytes == 0);

  const std::vector<int> identities{0, 1, 2};
  UniformFmmOptions point_options;
  point_options.expansion_basis = ExpansionBasis::Cartesian;
  point_options.backend = ExecutionBackend::CudaPartial;
  point_options.tree.max_level = 0;
  point_options.fixed_target_source_indices = identities;
  point_options.cuda_p2p_bsr_max_bytes = 0;
  UniformFmm point(positions, positions, point_options);
  REQUIRE(point.p2p_execution_packing() == P2PExecutionPacking::LeafBlock);
}

TEST_CASE("CUDA BSR supports finite cuboid point and cuboid self fields",
          "[cuda][manual][cuboid]") {
  if (!cuda_m2l_p2p_available()) {
    SUCCEED("CUDA M2L/P2P is unavailable");
    return;
  }

  const std::vector<Vec3> positions{{-0.2, 0.0, 0.0}, {0.2, 0.0, 0.0}};
  const std::vector<Vec3> moments{{0.7, -0.4, 0.2}, {-0.3, 0.5, -0.1}};
  const std::array<CuboidSize, 1> source_sizes{{{0.16, 0.12, 0.10}}};
  const std::array<CuboidSize, 1> target_sizes{{{0.11, 0.14, 0.09}}};

  for (const TargetGeometry target_geometry :
       {TargetGeometry::Point, TargetGeometry::RectangularPrism}) {
    const std::span<const CuboidSize> target_geometry_sizes =
        target_geometry == TargetGeometry::Point
            ? std::span<const CuboidSize>{}
            : std::span<const CuboidSize>(target_sizes);
    const DenseDirectPlan direct(
        positions, positions, SourceGeometry::RectangularPrism, target_geometry,
        source_sizes, target_geometry_sizes, {}, StaticPrecision::Float64);
    const auto expected = direct.evaluate(moments, DenseDirectBackend::Portable);

    UniformFmmOptions options;
    options.backend = ExecutionBackend::CudaPartial;
    options.precision = StaticPrecision::Float64;
    options.expansion_basis = ExpansionBasis::Cartesian;
    options.expansion_order = 2;
    options.tree.max_level = 0;
    options.tree.root_centre = Vec3{};
    options.tree.root_half_width = 0.5;
    options.source_geometry = SourceGeometry::RectangularPrism;
    options.source_sizes = {source_sizes.front()};
    options.target_geometry = target_geometry;
    if (target_geometry == TargetGeometry::RectangularPrism) {
      options.target_sizes = {target_sizes.front()};
    }
    options.p2p_packing = P2PExecutionPacking::CudaBsr3;

    UniformFmm fmm(positions, positions, options);
    REQUIRE(fmm.p2p_execution_packing() == P2PExecutionPacking::CudaBsr3);
    REQUIRE(fmm.cuda_plan_statistics().p2p_identity_bytes == 0);
    const auto actual = fmm.evaluate(moments, OutputFlags::Field);
    for (std::size_t target = 0; target < actual.size(); ++target) {
      REQUIRE(actual[target].H.x ==
              Catch::Approx(expected[target].x).margin(3.0e-11));
      REQUIRE(actual[target].H.y ==
              Catch::Approx(expected[target].y).margin(3.0e-11));
      REQUIRE(actual[target].H.z ==
              Catch::Approx(expected[target].z).margin(3.0e-11));
    }
    REQUIRE(std::abs(actual[0].H.x) + std::abs(actual[0].H.y) +
                std::abs(actual[0].H.z) >
            1.0e-12);
  }
}

TEST_CASE("CUDA M2L/P2P accepts empty geometry", "[cuda][manual]")
{
    if (!cuda_m2l_p2p_available()) {
        SUCCEED("CUDA M2L/P2P is unavailable");
        return;
    }

    UniformFmmOptions options;
    options.expansion_basis = ExpansionBasis::Cartesian;
    options.precision = StaticPrecision::Float64;
    options.backend = ExecutionBackend::CudaM2LP2P;
    UniformFmm fmm(std::vector<Vec3>{}, std::vector<Vec3>{}, options);
    const auto result = fmm.evaluate({}, OutputFlags::Field);

    REQUIRE(result.empty());
    REQUIRE(fmm.last_timings().cuda_p2p_wait.calls == 1);
}

TEST_CASE("full CUDA accepts empty geometry", "[cuda][manual]")
{
    if (!cuda_full_available()) {
        SUCCEED("full CUDA FMM is unavailable");
        return;
    }

    UniformFmmOptions options;
    options.expansion_basis = ExpansionBasis::Cartesian;
    options.backend = ExecutionBackend::CudaFull;
    UniformFmm fmm(std::vector<Vec3>{}, std::vector<Vec3>{}, options);
    const auto result = fmm.evaluate({}, OutputFlags::Field);

    REQUIRE(result.empty());
    REQUIRE(fmm.last_timings().cuda_p2p_kernel.calls == 1);
}
