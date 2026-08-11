// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "cdfmm/uniform_fmm.hpp"
#include "cdfmm/operators.hpp"
#include "cdfmm/static_operators.hpp"

using namespace cdfmm;

TEST_CASE("canonical static M2L matrices match independent m2l_add")
{
    std::mt19937 generator(9182);
    std::uniform_real_distribution<double> coefficient(-1.0, 1.0);

    for (const int order : {1, 2, 3, 4, 6}) {
        const MultiIndexSet basis(order);
        for (const Vec3 R : {
                 Vec3{2.0, -3.0, 1.0},
                 Vec3{-0.75, 1.25, 1.5},
                 Vec3{4.0, 3.0, -2.0}
             }) {
            CoeffVector M(static_cast<std::size_t>(basis.size()));
            for (double& value : M) {
                value = coefficient(generator);
            }
            CoeffVector expected(M.size(), 0.0);
            CoeffVector actual(M.size(), 0.0);
            m2l_add(basis, R, M, expected);
            const auto matrix = build_static_m2l_matrix(basis, R);
            apply_static_coefficient_matrix(matrix, M, actual);

            for (std::size_t index = 0; index < actual.size(); ++index) {
                REQUIRE(actual[index] == Catch::Approx(expected[index])
                    .margin(2.0e-14)
                    .epsilon(4.0e-13));
            }
        }
    }
}

TEST_CASE("static grouped M2L matches the independent reference traversal")
{
    std::mt19937 generator(417);
    std::uniform_real_distribution<double> coordinate(-0.9, 0.9);
    std::uniform_real_distribution<double> moment(-1.0, 1.0);

    std::vector<Vec3> positions(96);
    std::vector<Vec3> moments(96);
    std::vector<int> target_source_indices(positions.size());
    std::iota(target_source_indices.begin(), target_source_indices.end(), 0);
    for (std::size_t index = 0; index < positions.size(); ++index) {
        positions[index] = {coordinate(generator), coordinate(generator),
                            coordinate(generator)};
        moments[index] = {moment(generator), moment(generator),
                          moment(generator)};
    }

    for (const int order : {2, 3, 4}) {
        UniformFmmOptions static_options;
        static_options.expansion_order = order;
        static_options.tree.max_level = 3;
        static_options.backend = ExecutionBackend::CpuStatic;
        UniformFmmOptions reference_options = static_options;
        reference_options.m2l_backend = M2LBackend::Reference;
        reference_options.backend = ExecutionBackend::CpuReference;

        UniformFmm static_fmm(positions, positions, static_options);
        UniformFmm reference_fmm(positions, positions, reference_options);
        const auto static_values = static_fmm.evaluate(
            moments,
            OutputFlags::Field,
            target_source_indices
        );
        const auto reference_values = reference_fmm.evaluate(
            moments,
            OutputFlags::Field,
            target_source_indices
        );

        REQUIRE(static_fmm.m2l_backend() == M2LBackend::Static);
        REQUIRE(static_fmm.static_plan_statistics().transfer_classes > 0);
        for (std::size_t index = 0; index < positions.size(); ++index) {
            REQUIRE(std::isfinite(static_values[index].H.x));
            REQUIRE(std::isfinite(static_values[index].H.y));
            REQUIRE(std::isfinite(static_values[index].H.z));
            REQUIRE(static_values[index].H.x ==
                    Catch::Approx(reference_values[index].H.x).epsilon(2.0e-12));
            REQUIRE(static_values[index].H.y ==
                    Catch::Approx(reference_values[index].H.y).epsilon(2.0e-12));
            REQUIRE(static_values[index].H.z ==
                    Catch::Approx(reference_values[index].H.z).epsilon(2.0e-12));
        }

        const auto classes = static_fmm.static_plan_statistics().transfer_classes;
        moments.front().x += 0.25;
        const auto repeated_values = static_fmm.evaluate(
            moments,
            OutputFlags::Field,
            target_source_indices
        );
        REQUIRE(repeated_values.size() == positions.size());
        REQUIRE(static_fmm.static_plan_statistics().transfer_classes == classes);
    }
}
