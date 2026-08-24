// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "cdfmm/operators.hpp"
#include "cdfmm/static_operators.hpp"
#include "cdfmm/uniform_fmm.hpp"

using namespace cdfmm;

TEST_CASE("compact static P2P matches list1-style direct pairs") {
  const std::vector<Vec3> sources{
      {-0.4, 0.1, 0.2}, {0.3, -0.2, 0.5}, {0.8, 0.4, -0.1}};
  const std::vector<Vec3> targets{
      {0.1, 0.6, 0.3}, {-0.7, -0.3, 0.2}, {1.5, 1.5, 1.5}};
  const std::vector<Vec3> moments{
      {0.2, -0.4, 0.7}, {-0.3, 0.8, 0.1}, {0.6, 0.2, -0.5}};
  // The last target deliberately represents an empty near-field row.
  const std::vector<std::array<int, 2>> interactions{
      {0, 0}, {0, 2}, {1, 0}, {1, 1}};
  const StaticP2POperator operator_map =
      build_static_p2p_operator(targets, sources, interactions);
  std::vector<Vec3> actual(targets.size());
  apply_static_p2p_operator(operator_map, moments, actual);

    for (std::size_t target = 0; target < targets.size(); ++target) {
    PotentialField expected;
    for (const auto pair : interactions) {
      if (pair[0] == static_cast<int>(target)) {
        expected.H +=
            p2p_dipole_pair(targets[target],
                            sources[static_cast<std::size_t>(pair[1])],
                            moments[static_cast<std::size_t>(pair[1])])
                .H;
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

TEST_CASE("compact static P2P honours explicit self identity") {
  const std::vector<Vec3> positions{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  const std::vector<Vec3> moments{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
  const std::vector<std::array<int, 2>> interactions{
      {0, 0}, {0, 1}, {1, 0}, {1, 1}};
  const auto operator_map =
      build_static_p2p_operator(positions, positions, interactions);
  std::vector<Vec3> actual(positions.size());
  const std::vector<int> identities{0, 1};
  apply_static_p2p_operator(operator_map, moments, actual, identities);
  for (std::size_t target = 0; target < positions.size(); ++target) {
    const auto expected =
        p2p_dipole_sum(positions[target], positions, moments,
                       OutputFlags::Field, static_cast<int>(target));
    REQUIRE(actual[target].x == Catch::Approx(expected.H.x));
    REQUIRE(actual[target].y == Catch::Approx(expected.H.y));
    REQUIRE(actual[target].z == Catch::Approx(expected.H.z));
  }
}

TEST_CASE("P2P execution packings preserve canonical rows and self identity") {
  const std::vector<Vec3> sources{
      {-0.4, 0.1, 0.2}, {0.3, -0.2, 0.5}, {0.8, 0.4, -0.1}};
  const std::vector<Vec3> targets{
      {0.1, 0.6, 0.3}, {-0.7, -0.3, 0.2}, {1.5, 1.5, 1.5}};
  const std::vector<std::array<int, 2>> interactions{
      {0, 0}, {0, 2}, {1, 0}, {1, 1}};
  const StaticP2POperator canonical =
      build_static_p2p_operator(targets, sources, interactions);
  const StaticP2PCompactPlan compact = build_static_p2p_compact_plan(canonical);
  const std::vector<StaticP2PLeafPair> leaf_pairs{
      {0, 1, 0, 1}, {0, 1, 2, 1}, {1, 1, 0, 2}};
  const StaticP2PLeafPlan leaf =
      build_static_p2p_leaf_plan(canonical, leaf_pairs);
  const std::vector<int> self_indices{-1, 1, -1};
  const StaticP2PBsrPlan bsr =
      build_static_p2p_bsr_plan(canonical, self_indices);

  std::mt19937 generator(8123);
  std::uniform_real_distribution<double> random(-1.0, 1.0);
  for (int state = 0; state < 5; ++state) {
    std::vector<Vec3> moments(sources.size());
    for (Vec3 &moment : moments) {
      moment = {random(generator), random(generator), random(generator)};
    }
    std::vector<Vec3> expected(targets.size());
    std::vector<Vec3> compact_result(targets.size());
    std::vector<Vec3> leaf_result(targets.size());
    std::vector<Vec3> bsr_result(targets.size());
    apply_static_p2p_operator(canonical, moments, expected, self_indices);
    apply_static_p2p_compact_plan(compact, moments, compact_result,
                                  self_indices);
    apply_static_p2p_leaf_plan(leaf, moments, leaf_result, self_indices);
    apply_static_p2p_bsr_plan(bsr, moments, bsr_result, self_indices);
    for (std::size_t target = 0; target < targets.size(); ++target) {
      REQUIRE(compact_result[target].x ==
              Catch::Approx(expected[target].x).margin(3.0e-15));
      REQUIRE(compact_result[target].y ==
              Catch::Approx(expected[target].y).margin(3.0e-15));
      REQUIRE(compact_result[target].z ==
              Catch::Approx(expected[target].z).margin(3.0e-15));
      REQUIRE(leaf_result[target].x ==
              Catch::Approx(expected[target].x).margin(3.0e-15));
      REQUIRE(leaf_result[target].y ==
              Catch::Approx(expected[target].y).margin(3.0e-15));
      REQUIRE(leaf_result[target].z ==
              Catch::Approx(expected[target].z).margin(3.0e-15));
      REQUIRE(bsr_result[target].x ==
              Catch::Approx(expected[target].x).margin(3.0e-15));
      REQUIRE(bsr_result[target].y ==
              Catch::Approx(expected[target].y).margin(3.0e-15));
      REQUIRE(bsr_result[target].z ==
              Catch::Approx(expected[target].z).margin(3.0e-15));
    }
  }

  REQUIRE(compact.memory().tensor_bytes ==
          interactions.size() * 9 * sizeof(double));
  REQUIRE(compact.memory().index_bytes == interactions.size() * sizeof(int));
  REQUIRE(leaf.memory().tensor_bytes ==
          interactions.size() * 6 * sizeof(double));
  REQUIRE(leaf.memory().index_bytes == 0);
  REQUIRE(leaf.blocks.size() == leaf_pairs.size());
  REQUIRE(bsr.memory().tensor_bytes ==
          interactions.size() * 9 * sizeof(double));
  const std::vector<int> changed_self_indices{-1, -1, -1};
  const std::vector<Vec3> changed_moments(sources.size(), Vec3{1.0, 0.0, 0.0});
  std::vector<Vec3> rejected(targets.size());
  REQUIRE_THROWS_AS(apply_static_p2p_bsr_plan(bsr, changed_moments, rejected,
                                              changed_self_indices),
                    std::invalid_argument);
}

TEST_CASE("P2P packings preserve singular distinct coincident particles") {
  const std::vector<Vec3> sources{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  const std::vector<Vec3> targets{{0.0, 0.0, 0.0}};
  const std::vector<Vec3> moments{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
  const std::vector<std::array<int, 2>> interactions{{0, 0}, {0, 1}};
  const StaticP2POperator canonical =
      build_static_p2p_operator(targets, sources, interactions);
  const StaticP2PCompactPlan compact = build_static_p2p_compact_plan(canonical);
  const StaticP2PLeafPlan leaf = build_static_p2p_leaf_plan(
      canonical, std::array<StaticP2PLeafPair, 1>{{{0, 1, 0, 2}}});
  const std::array<int, 1> identities{0};
  const StaticP2PBsrPlan bsr = build_static_p2p_bsr_plan(canonical, identities);
  std::array<Vec3, 1> compact_result{};
  std::array<Vec3, 1> leaf_result{};
  std::array<Vec3, 1> bsr_result{};
  apply_static_p2p_compact_plan(compact, moments, compact_result, identities);
  apply_static_p2p_leaf_plan(leaf, moments, leaf_result, identities);
  apply_static_p2p_bsr_plan(bsr, moments, bsr_result, identities);
  REQUIRE(std::isnan(compact_result[0].x));
  REQUIRE(std::isnan(leaf_result[0].x));
  REQUIRE(std::isnan(bsr_result[0].x));
}

TEST_CASE("leaf P2P packing handles irregular occupancies across depths") {
  std::vector<Vec3> positions{{-0.99, -0.99, -0.99},
                              {0.99, 0.99, 0.99},
                              {-0.99, 0.99, -0.99},
                              {0.99, -0.99, 0.99}};
  std::mt19937 generator(7129);
  std::uniform_real_distribution<double> coordinate(-0.98, 0.98);
  for (int particle = 0; particle < 28; ++particle) {
    positions.push_back(
        {coordinate(generator), coordinate(generator), coordinate(generator)});
  }

  for (const int depth : {1, 2, 3}) {
    UniformTreeOptions options;
    options.max_level = depth;
    options.root_centre = Vec3{};
    options.root_half_width = 1.0;
    const UniformTree tree(positions, positions, options);
    const auto nodes = tree.nodes();
    std::vector<std::array<int, 2>> interactions;
    std::vector<StaticP2PLeafPair> leaf_pairs;
    for (const int leaf_index : tree.occupied_target_leaves()) {
      const TreeNode &leaf = nodes[static_cast<std::size_t>(leaf_index)];
      for (const int neighbour_index : leaf.list1) {
        const TreeNode &neighbour =
            nodes[static_cast<std::size_t>(neighbour_index)];
        if (neighbour.source_count() == 0) {
          continue;
        }
        leaf_pairs.push_back({static_cast<int>(leaf.target_begin),
                              static_cast<int>(leaf.target_count()),
                              static_cast<int>(neighbour.source_begin),
                              static_cast<int>(neighbour.source_count())});
        for (std::size_t target = leaf.target_begin; target < leaf.target_end;
             ++target) {
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
    const StaticP2PLeafPlan leaf =
        build_static_p2p_leaf_plan(canonical, leaf_pairs);
    std::vector<Vec3> moments(positions.size());
    std::vector<int> identities(positions.size());
    for (std::size_t index = 0; index < positions.size(); ++index) {
      moments[index] = {coordinate(generator), coordinate(generator),
                        coordinate(generator)};
      identities[index] = static_cast<int>(index);
    }
    std::vector<Vec3> expected(positions.size());
    std::vector<Vec3> actual(positions.size());
    apply_static_p2p_operator(canonical, moments, expected, identities);
    apply_static_p2p_leaf_plan(leaf, moments, actual, identities);
    for (std::size_t target = 0; target < positions.size(); ++target) {
      REQUIRE(actual[target].x ==
              Catch::Approx(expected[target].x).epsilon(2.0e-13));
      REQUIRE(actual[target].y ==
              Catch::Approx(expected[target].y).epsilon(2.0e-13));
      REQUIRE(actual[target].z ==
              Catch::Approx(expected[target].z).epsilon(2.0e-13));
    }
    REQUIRE(leaf.maximum_occupancy >= leaf.minimum_occupancy);
    REQUIRE(leaf.unique_occupancies >= 1);
  }
}

TEST_CASE("static P2M matches the independent dipole operator") {
  std::mt19937 generator(731);
  std::uniform_real_distribution<double> random(-0.8, 0.8);
  const Vec3 centre{0.1, -0.2, 0.3};
  const std::vector<Vec3> positions{
      {-0.4, 0.2, 0.5}, {0.3, -0.6, 0.1}, {0.7, 0.4, -0.2}};
  for (const int order : {1, 2, 4, 6}) {
    const MultiIndexSet basis(order);
    const auto operator_map =
        build_static_p2m_operator(basis, centre, positions);
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
      const CoeffVector expected =
          p2m_dipole(basis, centre, positions, moments);
      CoeffVector actual(expected.size(), 0.0);
      apply_static_operator(operator_map, flat, actual);
      for (std::size_t index = 0; index < actual.size(); ++index) {
        REQUIRE(actual[index] ==
                Catch::Approx(expected[index]).margin(2.0e-14));
      }
    }
  }
}

TEST_CASE("static triangular translations match M2M and L2L references") {
  std::mt19937 generator(882);
  std::uniform_real_distribution<double> random(-1.0, 1.0);
  for (const int order : {1, 3, 5}) {
    const MultiIndexSet basis(order);
    for (int child_class = 0; child_class < 8; ++child_class) {
      const Vec3 child_offset{(child_class & 1) != 0 ? 0.25 : -0.25,
                              (child_class & 2) != 0 ? 0.25 : -0.25,
                              (child_class & 4) != 0 ? 0.25 : -0.25};
      CoeffVector input(static_cast<std::size_t>(basis.size()));
      for (double &value : input) {
        value = random(generator);
            }
            CoeffVector expected(input.size(), 0.0);
      CoeffVector actual(input.size(), 0.0);
      m2m_add(basis, child_offset * -1.0, input, expected);
      apply_static_operator(
          build_static_m2m_operator(basis, child_offset * -1.0), input, actual);
      REQUIRE(actual == expected);

      std::fill(expected.begin(), expected.end(), 0.0);
      std::fill(actual.begin(), actual.end(), 0.0);
      l2l_add(basis, child_offset, input, expected);
      apply_static_operator(build_static_l2l_operator(basis, child_offset),
                            input, actual);
      REQUIRE(actual == expected);
    }
  }
}

TEST_CASE("static L2P matches every reference output mode") {
  std::mt19937 generator(192);
  std::uniform_real_distribution<double> random(-1.0, 1.0);
  const Vec3 centre{-0.2, 0.3, 0.1};
    for (const int order : {1, 2, 4, 6}) {
        const MultiIndexSet basis(order);
        CoeffVector L(static_cast<std::size_t>(basis.size()));
    for (double &value : L) {
      value = random(generator);
    }
    for (const Vec3 target : {Vec3{0.1, 0.2, -0.1}, Vec3{-0.4, 0.7, 0.3}}) {
      const auto evaluator = build_static_l2p_evaluator(basis, centre, target);
      for (const OutputFlags output :
           {OutputFlags::Field, OutputFlags::Potential, OutputFlags::Both}) {
        const PotentialField expected =
            l2p_eval(basis, centre, target, L, output);
        const PotentialField actual =
            apply_static_l2p_evaluator(evaluator, L, output);
        REQUIRE(actual.phi == Catch::Approx(expected.phi));
        REQUIRE(actual.H.x == Catch::Approx(expected.H.x));
        REQUIRE(actual.H.y == Catch::Approx(expected.H.y));
                REQUIRE(actual.H.z == Catch::Approx(expected.H.z));
            }
        }
  }
}

TEST_CASE("canonical static M2L matrices match independent m2l_add") {
  std::mt19937 generator(9182);
  std::uniform_real_distribution<double> coefficient(-1.0, 1.0);

  for (const int order : {1, 2, 3, 4, 6}) {
    const MultiIndexSet basis(order);
    for (const Vec3 R :
         {Vec3{2.0, -3.0, 1.0}, Vec3{-0.75, 1.25, 1.5}, Vec3{4.0, 3.0, -2.0}}) {
      CoeffVector M(static_cast<std::size_t>(basis.size()));
      for (double &value : M) {
        value = coefficient(generator);
            }
            CoeffVector expected(M.size(), 0.0);
            CoeffVector actual(M.size(), 0.0);
            m2l_add(basis, R, M, expected);
            const auto matrix = build_static_m2l_matrix(basis, R);
      apply_static_coefficient_matrix(matrix, M, actual);

      for (std::size_t index = 0; index < actual.size(); ++index) {
        REQUIRE(
            actual[index] ==
            Catch::Approx(expected[index]).margin(2.0e-14).epsilon(4.0e-13));
      }
    }
  }
}

TEST_CASE("normalised M2L transfer classes scale exactly across levels") {
  const MultiIndexSet basis(5);
  const Vec3 transfer{3.0, -2.0, 1.0};
  const auto normalised = build_static_m2l_matrix(basis, transfer);

  for (const double box_width : {0.25, 0.125}) {
    const auto physical = build_static_m2l_matrix(basis, transfer * box_width);
    for (int alpha = 0; alpha < basis.size(); ++alpha) {
      for (int beta = 0; beta < basis.size(); ++beta) {
        const std::size_t entry =
            static_cast<std::size_t>(beta) +
            static_cast<std::size_t>(basis.size()) * alpha;
        const int homogeneous_degree =
            basis[alpha].degree() + basis[beta].degree() + 1;
        const double scaled =
            normalised[entry] * std::pow(box_width, -homogeneous_degree);
        REQUIRE(
            scaled ==
            Catch::Approx(physical[entry]).margin(2.0e-11).epsilon(2.0e-13));
      }
    }
  }
}

TEST_CASE("canonical target-row M2L plan handles levels and repeated rows") {
  const MultiIndexSet basis(3);
  const int n = basis.size();
  const std::array<Vec3, 2> normalised_transfers{
      Vec3{2.0, -3.0, 1.0}, Vec3{-3.0, 2.0, 2.0}};

  StaticM2LPlan plan;
  plan.coefficient_count = n;
  plan.matrix_count = static_cast<int>(normalised_transfers.size());
  plan.level_count = 3;
  for (const Vec3 transfer : normalised_transfers) {
    const auto matrix = build_static_m2l_matrix(basis, transfer);
    plan.matrices.insert(plan.matrices.end(), matrix.begin(), matrix.end());
  }
  for (const double width : {1.0, 0.5, 0.25}) {
    for (int coefficient = 0; coefficient < n; ++coefficient) {
      const int degree = basis[coefficient].degree();
      plan.multipole_scaling.push_back(std::pow(width, -degree));
      plan.local_scaling.push_back(std::pow(width, -(degree + 1)));
    }
  }

  // Targets one and three have two and one interactions, respectively.
  plan.target_row_offsets = {0, 0, 2, 2, 3};
  plan.source_nodes = {0, 2, 1};
  plan.matrix_ids = {0, 1, 0};
  plan.interaction_levels = {1, 1, 2};

  std::vector<CoeffVector> multipoles(
      4, CoeffVector(static_cast<std::size_t>(n), 0.0));
  for (std::size_t node = 0; node < multipoles.size(); ++node) {
    for (int coefficient = 0; coefficient < n; ++coefficient) {
      multipoles[node][static_cast<std::size_t>(coefficient)] =
          0.01 * static_cast<double>((node + 1) * (coefficient + 2));
    }
  }
  std::vector<CoeffVector> actual(
      4, CoeffVector(static_cast<std::size_t>(n), 0.0));
  std::vector<CoeffVector> expected = actual;

  apply_static_m2l_plan(plan, 1, multipoles, actual);
  apply_static_m2l_plan(plan, 2, multipoles, actual);
  m2l_add(basis, normalised_transfers[0] * 0.5, multipoles[0], expected[1]);
  m2l_add(basis, normalised_transfers[1] * 0.5, multipoles[2], expected[1]);
  m2l_add(basis, normalised_transfers[0] * 0.25, multipoles[1], expected[3]);

  for (std::size_t node = 0; node < actual.size(); ++node) {
    for (int coefficient = 0; coefficient < n; ++coefficient) {
      REQUIRE(actual[node][static_cast<std::size_t>(coefficient)] ==
              Catch::Approx(
                  expected[node][static_cast<std::size_t>(coefficient)])
                  .margin(2.0e-11)
                  .epsilon(5.0e-13));
    }
  }
}

TEST_CASE("canonical target-row M2L plan accepts empty geometry") {
  StaticM2LPlan plan;
  plan.coefficient_count = 1;
  plan.level_count = 1;
  plan.target_row_offsets = {0};
  std::vector<CoeffVector> multipoles;
  std::vector<CoeffVector> locals;

  REQUIRE_NOTHROW(apply_static_m2l_plan(plan, 0, multipoles, locals));
}

TEST_CASE("static grouped M2L matches the independent reference traversal") {
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
    moments[index] = {moment(generator), moment(generator), moment(generator)};
  }

  for (const int order : {2, 3, 4}) {
        UniformFmmOptions static_options;
        static_options.expansion_basis = ExpansionBasis::Cartesian;
        static_options.precision = StaticPrecision::Float64;
        static_options.expansion_order = order;
        static_options.tree.max_level = 3;
        static_options.backend = ExecutionBackend::CpuStatic;
        UniformFmmOptions reference_options = static_options;
        reference_options.m2l_backend = M2LBackend::Reference;
        reference_options.backend = ExecutionBackend::CpuReference;

    UniformFmm static_fmm(positions, positions, static_options);
    UniformFmm reference_fmm(positions, positions, reference_options);
    const auto static_values =
        static_fmm.evaluate(moments, OutputFlags::Field, target_source_indices);
    const auto reference_values = reference_fmm.evaluate(
        moments, OutputFlags::Field, target_source_indices);

    REQUIRE(static_fmm.m2l_backend() == M2LBackend::Static);
    REQUIRE(static_fmm.static_matrix_backend() ==
                StaticMatrixBackend::Portable);
        REQUIRE(static_fmm.static_plan_statistics().transfer_classes > 0);
        const auto& plan = static_fmm.static_plan_statistics();
        REQUIRE(plan.m2m_operators == static_cast<std::size_t>(8 * 3));
        REQUIRE(plan.l2l_operators == static_cast<std::size_t>(8 * 3));
        REQUIRE(plan.m2m_operators < plan.m2m_theoretical_interactions);
    REQUIRE(plan.l2l_operators < plan.l2l_theoretical_interactions);
    REQUIRE(plan.m2l_operators == plan.transfer_classes);
    REQUIRE(plan.m2l_operators <=
            StaticPlanStatistics::theoretical_maximum_m2l_classes);
    REQUIRE(plan.translation_operator_bytes() == plan.m2m_operator_bytes +
                                                     plan.m2l_operator_bytes +
                                                     plan.l2l_operator_bytes);
    REQUIRE(plan.dense);
    REQUIRE_FALSE(plan.sparse);
        REQUIRE_FALSE(plan.numerically_pruned);
        REQUIRE_FALSE(plan.symmetry_compressed);
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
    const auto repeated_values =
        static_fmm.evaluate(moments, OutputFlags::Field, target_source_indices);
    REQUIRE(repeated_values.size() == positions.size());
    REQUIRE(static_fmm.static_plan_statistics().transfer_classes == classes);
    REQUIRE(static_fmm.static_plan_statistics().construction_count ==
                construction_count);
        REQUIRE(static_fmm.static_plan_statistics().total_bytes() == plan_bytes);
  }
}

TEST_CASE("oneMKL and portable static matrices agree when oneMKL is enabled") {
  if (!one_mkl_available()) {
    SUCCEED("This build does not include oneMKL");
    return;
  }

  const std::vector<Vec3> positions{
      {-0.8, -0.6, -0.4}, {0.7, -0.5, 0.3}, {-0.3, 0.8, 0.6}, {0.6, 0.4, -0.7}};
  const std::vector<Vec3> moments{
      {0.2, -0.5, 0.7}, {-0.6, 0.1, 0.4}, {0.8, 0.3, -0.2}, {-0.1, -0.7, 0.5}};
  const std::vector<int> identities{0, 1, 2, 3};

  UniformFmmOptions portable_options;
    portable_options.expansion_basis = ExpansionBasis::Cartesian;
    portable_options.precision = StaticPrecision::Float64;
    portable_options.expansion_order = 3;
    portable_options.tree.max_level = 2;
    portable_options.backend = ExecutionBackend::CpuStatic;
    UniformFmmOptions mkl_options = portable_options;
    mkl_options.static_matrix_backend = StaticMatrixBackend::OneMkl;

  UniformFmm portable(positions, positions, portable_options);
  UniformFmm mkl(positions, positions, mkl_options);
  const auto portable_values =
      portable.evaluate(moments, OutputFlags::Field, identities);
  const auto mkl_values = mkl.evaluate(moments, OutputFlags::Field, identities);

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
