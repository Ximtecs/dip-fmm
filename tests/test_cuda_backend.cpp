// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <numeric>
#include <vector>

#include "cdfmm/uniform_fmm.hpp"
#include "cdfmm/cuda_direct.hpp"
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
    REQUIRE(cuda_direct_available());
    for (std::size_t index = 0; index < targets.size(); ++index) {
            REQUIRE(actual[index].phi ==
                    Catch::Approx(direct[index].phi).epsilon(2.0e-13));
            REQUIRE(actual[index].H.x ==
                    Catch::Approx(direct[index].H.x).epsilon(2.0e-13));
            REQUIRE(actual[index].H.y ==
                    Catch::Approx(direct[index].H.y).epsilon(2.0e-13));
            REQUIRE(actual[index].H.z ==
                    Catch::Approx(direct[index].H.z).epsilon(2.0e-13));
    }
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
        options.backend = ExecutionBackend::CudaPartial;
        UniformFmm partial(sources, targets, options);
        options.backend = ExecutionBackend::CudaFull;
        UniformFmm full(sources, targets, options);

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
