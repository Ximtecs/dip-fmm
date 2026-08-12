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
    REQUIRE(first.evaluation_h2d_bytes == positions.size() * 3 * sizeof(double));
    REQUIRE(first.evaluation_d2h_bytes == positions.size() * 3 * sizeof(double));
    REQUIRE(first.evaluation_h2d_calls == 1);
    REQUIRE(first.evaluation_d2h_calls == 1);
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
    REQUIRE_FALSE(cuda_full_available());
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
    for (int index = 0; index < 96; ++index) {
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

    for (const int order : {2, 4, 6}) {
        for (const int depth : {2, 3}) {
            UniformFmmOptions cpu_options;
            cpu_options.expansion_order = order;
            cpu_options.tree.max_level = depth;
            cpu_options.tree.root_centre = Vec3{0.0, 0.0, 0.0};
            cpu_options.tree.root_half_width = 1.0;
            cpu_options.backend = ExecutionBackend::CpuStatic;
            UniformFmmOptions cuda_options = cpu_options;
            cuda_options.backend = ExecutionBackend::CudaM2LP2P;

            UniformFmm cpu(positions, positions, cpu_options);
            UniformFmm cuda(positions, positions, cuda_options);
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

            const auto repeated = cuda.evaluate(
                moments, OutputFlags::Field, source_identities
            );
            REQUIRE(repeated.size() == positions.size());
            REQUIRE(cuda.cuda_plan_statistics().static_m2l_upload_count == 1);
            REQUIRE(cuda.cuda_plan_statistics().static_p2p_upload_count == 1);
        }
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
