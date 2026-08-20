// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <numeric>
#include <vector>

#include "cdfmm/cuda_direct.hpp"
#include "cdfmm/cuda_cuboid.hpp"
#include "cdfmm/cuda_p2p.hpp"
#include "cdfmm/uniform_fmm.hpp"
#include "cdfmm/validation.hpp"

using namespace cdfmm;

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
    const DenseDirectPlan cpu(
        positions, positions, SourceGeometry::UniformCuboid,
        TargetGeometry::Point, cube);
    CudaDenseDirectPlan cuda(
        positions, positions, SourceGeometry::UniformCuboid,
        TargetGeometry::Point, cube);

    const auto expected = cpu.evaluate(
        moments, DenseDirectBackend::Portable);
    const auto actual = cuda.evaluate(moments);

    REQUIRE(cuda.source_count() == positions.size());
    REQUIRE(cuda.target_count() == positions.size());
    REQUIRE(cuda.tensor_memory_bytes() == cpu.tensor_memory_bytes());
    REQUIRE(cuda.persistent_device_bytes() >= cuda.tensor_memory_bytes());
    for (std::size_t index = 0; index < positions.size(); ++index) {
        REQUIRE(actual[index].x ==
                Catch::Approx(expected[index].x).epsilon(2.0e-13));
        REQUIRE(actual[index].y ==
                Catch::Approx(expected[index].y).epsilon(2.0e-13));
        REQUIRE(actual[index].z ==
                Catch::Approx(expected[index].z).epsilon(2.0e-13));
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
  REQUIRE(leaf_cuda.statistics().p2p_scratch_bytes > 0);
  REQUIRE(leaf_cuda.statistics().p2p_leaf_metadata_bytes > 0);
  CudaP2PPlan bsr_cuda(bsr);
  verify(bsr_cuda);
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

        REQUIRE(partial.p2p_execution_packing() ==
                P2PExecutionPacking::CudaBsr3);
        REQUIRE(full.p2p_execution_packing() == P2PExecutionPacking::CudaBsr3);
        REQUIRE(partial.cuda_plan_statistics().p2p_identity_bytes == 0);
        REQUIRE(full.cuda_plan_statistics().p2p_identity_bytes == 0);

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
        const std::size_t p2p_static_bytes =
            partial_statistics.p2p_interaction_count *
                sizeof(StaticDipoleBlock) +
            (targets.size() + 1) * sizeof(int);
        const std::size_t canonical_m2l_bytes =
            partial_statistics.m2l_matrix_bytes +
            partial_statistics.m2l_interaction_metadata_bytes + scaling_bytes;
        REQUIRE(partial_statistics.setup_h2d_bytes ==
                canonical_m2l_bytes + p2p_static_bytes);
        REQUIRE(partial_statistics.evaluation_h2d_bytes ==
                coefficient_bytes + moments.size() * sizeof(Vec3) +
                    identities.size() * sizeof(int));
        REQUIRE(partial_statistics.evaluation_d2h_bytes ==
                coefficient_bytes + targets.size() * sizeof(Vec3));
        REQUIRE(partial_statistics.persistent_device_bytes ==
                partial_statistics.setup_h2d_bytes + 2 * coefficient_bytes +
                    sources.size() * sizeof(Vec3) +
                    targets.size() * (sizeof(Vec3) + sizeof(int)) +
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

TEST_CASE("CUDA BSR memory budget selects the canonical fallback",
          "[cuda][manual]") {
  if (!cuda_m2l_p2p_available()) {
    SUCCEED("CUDA M2L/P2P is unavailable");
    return;
  }

  const std::vector<Vec3> positions{
      {-0.5, 0.0, 0.0}, {0.25, 0.1, -0.2}, {0.4, -0.3, 0.2}};
  const std::vector<int> identities{0, 1, 2};
  UniformFmmOptions options;
  options.backend = ExecutionBackend::CudaPartial;
  options.tree.max_level = 0;
  options.fixed_target_source_indices = identities;
  options.cuda_p2p_bsr_max_bytes = 0;

  UniformFmm fmm(positions, positions, options);

  REQUIRE(fmm.p2p_execution_packing() == P2PExecutionPacking::CanonicalAos);
  REQUIRE(fmm.cuda_plan_statistics().p2p_identity_bytes ==
          positions.size() * sizeof(int));
}

TEST_CASE("CUDA M2L/P2P accepts empty geometry", "[cuda][manual]")
{
    if (!cuda_m2l_p2p_available()) {
        SUCCEED("CUDA M2L/P2P is unavailable");
        return;
    }

    UniformFmmOptions options;
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
    options.backend = ExecutionBackend::CudaFull;
    UniformFmm fmm(std::vector<Vec3>{}, std::vector<Vec3>{}, options);
    const auto result = fmm.evaluate({}, OutputFlags::Field);

    REQUIRE(result.empty());
    REQUIRE(fmm.last_timings().cuda_p2p_kernel.calls == 1);
}
