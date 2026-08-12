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

TEST_CASE("compact static P2P matches list1-style direct pairs")
{
    const std::vector<Vec3> sources{
        {-0.4, 0.1, 0.2}, {0.3, -0.2, 0.5}, {0.8, 0.4, -0.1}
    };
    const std::vector<Vec3> targets{
        {0.1, 0.6, 0.3}, {-0.7, -0.3, 0.2}, {1.5, 1.5, 1.5}
    };
    const std::vector<Vec3> moments{
        {0.2, -0.4, 0.7}, {-0.3, 0.8, 0.1}, {0.6, 0.2, -0.5}
    };
    // The last target deliberately represents an empty near-field row.
    const std::vector<std::array<int, 2>> interactions{
        {0, 0}, {0, 2}, {1, 0}, {1, 1}
    };
    const StaticP2POperator operator_map = build_static_p2p_operator(
        targets, sources, interactions
    );
    std::vector<Vec3> actual(targets.size());
    apply_static_p2p_operator(operator_map, moments, actual);

    for (std::size_t target = 0; target < targets.size(); ++target) {
        PotentialField expected;
        for (const auto pair : interactions) {
            if (pair[0] == static_cast<int>(target)) {
                expected.H += p2p_dipole_pair(
                    targets[target], sources[static_cast<std::size_t>(pair[1])],
                    moments[static_cast<std::size_t>(pair[1])]
                ).H;
            }
        }
        REQUIRE(actual[target].x == Catch::Approx(expected.H.x).margin(2.0e-15));
        REQUIRE(actual[target].y == Catch::Approx(expected.H.y).margin(2.0e-15));
        REQUIRE(actual[target].z == Catch::Approx(expected.H.z).margin(2.0e-15));
    }
    REQUIRE(operator_map.blocks.size() == interactions.size());
    REQUIRE(operator_map.memory_bytes() ==
            operator_map.blocks.size() * sizeof(StaticDipoleBlock) +
            operator_map.row_offsets.size() * sizeof(int));
}

TEST_CASE("compact static P2P honours explicit self identity")
{
    const std::vector<Vec3> positions{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    const std::vector<Vec3> moments{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    const std::vector<std::array<int, 2>> interactions{
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    const auto operator_map = build_static_p2p_operator(
        positions, positions, interactions
    );
    std::vector<Vec3> actual(positions.size());
    const std::vector<int> identities{0, 1};
    apply_static_p2p_operator(operator_map, moments, actual, identities);
    for (std::size_t target = 0; target < positions.size(); ++target) {
        const auto expected = p2p_dipole_sum(
            positions[target], positions, moments, OutputFlags::Field,
            static_cast<int>(target)
        );
        REQUIRE(actual[target].x == Catch::Approx(expected.H.x));
        REQUIRE(actual[target].y == Catch::Approx(expected.H.y));
        REQUIRE(actual[target].z == Catch::Approx(expected.H.z));
    }
}

TEST_CASE("static P2M matches the independent dipole operator")
{
    std::mt19937 generator(731);
    std::uniform_real_distribution<double> random(-0.8, 0.8);
    const Vec3 centre{0.1, -0.2, 0.3};
    const std::vector<Vec3> positions{
        {-0.4, 0.2, 0.5}, {0.3, -0.6, 0.1}, {0.7, 0.4, -0.2}
    };
    for (const int order : {1, 2, 4, 6}) {
        const MultiIndexSet basis(order);
        const auto operator_map = build_static_p2m_operator(
            basis, centre, positions
        );
        for (int state = 0; state < 4; ++state) {
            std::vector<Vec3> moments(positions.size());
            std::vector<double> flat(3 * positions.size());
            for (std::size_t source = 0; source < moments.size(); ++source) {
                moments[source] = {random(generator), random(generator),
                                   random(generator)};
                flat[3 * source] = moments[source].x;
                flat[3 * source + 1] = moments[source].y;
                flat[3 * source + 2] = moments[source].z;
            }
            const CoeffVector expected = p2m_dipole(
                basis, centre, positions, moments
            );
            CoeffVector actual(expected.size(), 0.0);
            apply_static_operator(operator_map, flat, actual);
            for (std::size_t index = 0; index < actual.size(); ++index) {
                REQUIRE(actual[index] == Catch::Approx(expected[index])
                    .margin(2.0e-14));
            }
        }
    }
}

TEST_CASE("static triangular translations match M2M and L2L references")
{
    std::mt19937 generator(882);
    std::uniform_real_distribution<double> random(-1.0, 1.0);
    for (const int order : {1, 3, 5}) {
        const MultiIndexSet basis(order);
        for (int child_class = 0; child_class < 8; ++child_class) {
            const Vec3 child_offset{
                (child_class & 1) != 0 ? 0.25 : -0.25,
                (child_class & 2) != 0 ? 0.25 : -0.25,
                (child_class & 4) != 0 ? 0.25 : -0.25
            };
            CoeffVector input(static_cast<std::size_t>(basis.size()));
            for (double& value : input) {
                value = random(generator);
            }
            CoeffVector expected(input.size(), 0.0);
            CoeffVector actual(input.size(), 0.0);
            m2m_add(basis, child_offset * -1.0, input, expected);
            apply_static_operator(
                build_static_m2m_operator(basis, child_offset * -1.0),
                input, actual
            );
            REQUIRE(actual == expected);

            std::fill(expected.begin(), expected.end(), 0.0);
            std::fill(actual.begin(), actual.end(), 0.0);
            l2l_add(basis, child_offset, input, expected);
            apply_static_operator(
                build_static_l2l_operator(basis, child_offset), input, actual
            );
            REQUIRE(actual == expected);
        }
    }
}

TEST_CASE("static L2P matches every reference output mode")
{
    std::mt19937 generator(192);
    std::uniform_real_distribution<double> random(-1.0, 1.0);
    const Vec3 centre{-0.2, 0.3, 0.1};
    for (const int order : {1, 2, 4, 6}) {
        const MultiIndexSet basis(order);
        CoeffVector L(static_cast<std::size_t>(basis.size()));
        for (double& value : L) {
            value = random(generator);
        }
        for (const Vec3 target : {Vec3{0.1, 0.2, -0.1},
                                  Vec3{-0.4, 0.7, 0.3}}) {
            const auto evaluator = build_static_l2p_evaluator(
                basis, centre, target
            );
            for (const OutputFlags output : {OutputFlags::Field,
                                             OutputFlags::Potential,
                                             OutputFlags::Both}) {
                const PotentialField expected = l2p_eval(
                    basis, centre, target, L, output
                );
                const PotentialField actual = apply_static_l2p_evaluator(
                    evaluator, L, output
                );
                REQUIRE(actual.phi == Catch::Approx(expected.phi));
                REQUIRE(actual.H.x == Catch::Approx(expected.H.x));
                REQUIRE(actual.H.y == Catch::Approx(expected.H.y));
                REQUIRE(actual.H.z == Catch::Approx(expected.H.z));
            }
        }
    }
}

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
        REQUIRE(static_fmm.static_matrix_backend() ==
                StaticMatrixBackend::Portable);
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
        const auto construction_count =
            static_fmm.static_plan_statistics().construction_count;
        const auto plan_bytes = static_fmm.static_plan_statistics().total_bytes();
        moments.front().x += 0.25;
        const auto repeated_values = static_fmm.evaluate(
            moments,
            OutputFlags::Field,
            target_source_indices
        );
        REQUIRE(repeated_values.size() == positions.size());
        REQUIRE(static_fmm.static_plan_statistics().transfer_classes == classes);
        REQUIRE(static_fmm.static_plan_statistics().construction_count ==
                construction_count);
        REQUIRE(static_fmm.static_plan_statistics().total_bytes() == plan_bytes);
    }
}

TEST_CASE("oneMKL and portable static matrices agree when oneMKL is enabled")
{
    if (!one_mkl_available()) {
        SUCCEED("This build does not include oneMKL");
        return;
    }

    const std::vector<Vec3> positions{
        {-0.8, -0.6, -0.4},
        {0.7, -0.5, 0.3},
        {-0.3, 0.8, 0.6},
        {0.6, 0.4, -0.7}
    };
    const std::vector<Vec3> moments{
        {0.2, -0.5, 0.7},
        {-0.6, 0.1, 0.4},
        {0.8, 0.3, -0.2},
        {-0.1, -0.7, 0.5}
    };
    const std::vector<int> identities{0, 1, 2, 3};

    UniformFmmOptions portable_options;
    portable_options.expansion_order = 3;
    portable_options.tree.max_level = 2;
    portable_options.backend = ExecutionBackend::CpuStatic;
    UniformFmmOptions mkl_options = portable_options;
    mkl_options.static_matrix_backend = StaticMatrixBackend::OneMkl;

    UniformFmm portable(positions, positions, portable_options);
    UniformFmm mkl(positions, positions, mkl_options);
    const auto portable_values = portable.evaluate(
        moments,
        OutputFlags::Field,
        identities
    );
    const auto mkl_values = mkl.evaluate(
        moments,
        OutputFlags::Field,
        identities
    );

    REQUIRE(mkl.static_matrix_backend() == StaticMatrixBackend::OneMkl);
    for (std::size_t index = 0; index < positions.size(); ++index) {
        REQUIRE(mkl_values[index].H.x ==
                Catch::Approx(portable_values[index].H.x).epsilon(2.0e-13));
        REQUIRE(mkl_values[index].H.y ==
                Catch::Approx(portable_values[index].H.y).epsilon(2.0e-13));
        REQUIRE(mkl_values[index].H.z ==
                Catch::Approx(portable_values[index].H.z).epsilon(2.0e-13));
    }
}
