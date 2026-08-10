// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include <random>
#include <vector>

#include "cdfmm/uniform_fmm.hpp"

using namespace cdfmm;

TEST_CASE("static grouped M2L matches the independent reference traversal")
{
    std::mt19937 generator(417);
    std::uniform_real_distribution<double> coordinate(-0.9, 0.9);
    std::uniform_real_distribution<double> moment(-1.0, 1.0);

    std::vector<Vec3> positions(96);
    std::vector<Vec3> moments(96);
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
        UniformFmmOptions reference_options = static_options;
        reference_options.m2l_backend = M2LBackend::Reference;

        UniformFmm static_fmm(positions, positions, static_options);
        UniformFmm reference_fmm(positions, positions, reference_options);
        const auto static_values = static_fmm.evaluate(moments);
        const auto reference_values = reference_fmm.evaluate(moments);

        REQUIRE(static_fmm.m2l_backend() == M2LBackend::Static);
        REQUIRE(static_fmm.static_plan_statistics().transfer_classes > 0);
        for (std::size_t index = 0; index < positions.size(); ++index) {
            REQUIRE(static_values[index].H.x ==
                    Catch::Approx(reference_values[index].H.x).epsilon(2.0e-12));
            REQUIRE(static_values[index].H.y ==
                    Catch::Approx(reference_values[index].H.y).epsilon(2.0e-12));
            REQUIRE(static_values[index].H.z ==
                    Catch::Approx(reference_values[index].H.z).epsilon(2.0e-12));
        }

        const auto classes = static_fmm.static_plan_statistics().transfer_classes;
        moments.front().x += 0.25;
        static_fmm.evaluate(moments);
        REQUIRE(static_fmm.static_plan_statistics().transfer_classes == classes);
    }
}
