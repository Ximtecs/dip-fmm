// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "cdfmm/multi_index.hpp"
#include "cdfmm/static_operators.hpp"
#include "cdfmm/uniform_fmm.hpp"
#include "cuda_m2l_plan.hpp"

using namespace cdfmm;

namespace {

StaticM2LPlan make_m2l_plan(const int order, const int depth) {
  const MultiIndexSet basis(order);
  const int coefficient_count = basis.size();
  const int level_count = depth + 1;
  const int node_count = 2 * level_count;

  StaticM2LPlan plan;
  plan.coefficient_count = coefficient_count;
  plan.matrix_count = 2;
  plan.level_count = level_count;
  const std::array<Vec3, 2> transfers{
      Vec3{3.0, 0.0, 0.0},
      Vec3{-3.0, 3.0, 0.0},
  };
  for (const Vec3 transfer : transfers) {
    const std::vector<double> matrix =
        build_static_m2l_matrix(basis, transfer);
    plan.matrices.insert(plan.matrices.end(), matrix.begin(), matrix.end());
  }

  plan.multipole_scaling.resize(
      static_cast<std::size_t>(level_count) * coefficient_count);
  plan.local_scaling.resize(
      static_cast<std::size_t>(level_count) * coefficient_count);
  for (int level = 0; level < level_count; ++level) {
    for (int coefficient = 0; coefficient < coefficient_count;
         ++coefficient) {
      const std::size_t index =
          static_cast<std::size_t>(level) * coefficient_count + coefficient;
      plan.multipole_scaling[index] =
          1.0 + 0.01 * static_cast<double>(level + coefficient);
      plan.local_scaling[index] =
          1.0 / (1.0 + 0.02 * static_cast<double>(level + coefficient));
    }
  }

  plan.target_row_offsets.push_back(0);
  for (int target = 0; target < node_count; ++target) {
    const int level = target / 2;
    if (level > 0) {
      const int source = 2 * level + (target + 1) % 2;
      // Repeated transfer classes exercise stable deterministic accumulation.
      plan.source_nodes.push_back(source);
      plan.matrix_ids.push_back(target % 2);
      plan.interaction_levels.push_back(level);
      plan.source_nodes.push_back(source);
      plan.matrix_ids.push_back((target + 1) % 2);
      plan.interaction_levels.push_back(level);
    }
    plan.target_row_offsets.push_back(
        static_cast<int>(plan.source_nodes.size()));
  }
  for (int level = 0; level < level_count; ++level) {
    plan.level_target_begin.push_back(2 * level);
    plan.level_target_end.push_back(2 * level + 2);
  }
  return plan;
}

} // namespace

TEST_CASE("shared CUDA M2L executor agrees with the canonical CPU plan",
          "[cuda][manual][m2l]") {
  if (!cuda_m2l_p2p_available()) {
    SUCCEED("No CUDA-capable device is available");
    return;
  }

  constexpr std::array<int, 4> orders{2, 4, 6, 8};
  constexpr std::array<int, 3> depths{2, 3, 5};
  for (const int order : orders) {
    for (const int depth : depths) {
      CAPTURE(order, depth);
      StaticM2LPlan plan = make_m2l_plan(order, depth);
      const std::vector<double> original_matrices = plan.matrices;
      const std::vector<int> original_matrix_ids = plan.matrix_ids;
      const std::vector<int> original_sources = plan.source_nodes;
      const std::vector<int> original_levels = plan.interaction_levels;
      const std::size_t value_count =
          static_cast<std::size_t>(plan.target_row_offsets.size() - 1) *
          plan.coefficient_count;
      std::vector<double> multipoles(value_count);
      for (std::size_t index = 0; index < multipoles.size(); ++index) {
        multipoles[index] =
            std::sin(0.17 * static_cast<double>(index + 1)) * 0.125;
      }
      std::vector<double> expected(value_count, 0.0);
      for (int level = 0; level < plan.level_count; ++level) {
        apply_static_m2l_plan(plan, level, multipoles, expected);
      }

      CudaM2LPlan cuda(plan);
      std::vector<double> actual(value_count, 0.0);
      cuda.evaluate(multipoles, actual);
      for (std::size_t index = 0; index < value_count; ++index) {
        REQUIRE(actual[index] == Catch::Approx(expected[index])
                                     .margin(2.0e-11)
                                     .epsilon(5.0e-13));
      }

      // Deriving execution metadata must not alter canonical mathematics.
      REQUIRE(plan.matrices == original_matrices);
      REQUIRE(plan.matrix_ids == original_matrix_ids);
      REQUIRE(plan.source_nodes == original_sources);
      REQUIRE(plan.interaction_levels == original_levels);
      const CudaPlanStatistics &statistics = cuda.statistics();
      REQUIRE(statistics.m2l_unique_matrix_count == 2);
      REQUIRE(statistics.m2l_interaction_count ==
              plan.source_nodes.size());
      REQUIRE(statistics.m2l_active_row_count ==
              static_cast<std::size_t>(2 * depth));
      REQUIRE(statistics.m2l_scratch_bytes <=
              value_count * sizeof(double));
      REQUIRE(statistics.m2l_threads_per_block >= 64);
      REQUIRE(cuda.timings().scale_seconds >= 0.0);
      REQUIRE(cuda.timings().multiply_seconds >= 0.0);
    }
  }
}
