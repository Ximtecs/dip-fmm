// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <array>
#include <cmath>
#include <vector>

// Canonical foundational interfaces.
#include "cdfmm/core/output_flags.hpp"
#include "cdfmm/core/precision.hpp"
#include "cdfmm/core/timing.hpp"
#include "cdfmm/geometry/models.hpp"
#include "cdfmm/geometry/primitives/rectangular_prism.hpp"
#include "cdfmm/geometry/primitives/tetrahedron.hpp"
#include "cdfmm/math/coefficients.hpp"
#include "cdfmm/math/laplace_derivatives.hpp"
#include "cdfmm/math/multi_index.hpp"
#include "cdfmm/math/pair_tensor.hpp"
#include "cdfmm/math/potential_field.hpp"
#include "cdfmm/math/spherical_harmonics.hpp"
#include "cdfmm/math/taylor_jet.hpp"
#include "cdfmm/math/vec3.hpp"
#include "cdfmm/operators/l2l.hpp"
#include "cdfmm/operators/l2p.hpp"
#include "cdfmm/operators/m2p.hpp"
#include "cdfmm/operators/m2l.hpp"
#include "cdfmm/operators/m2m.hpp"
#include "cdfmm/operators/operators.hpp"
#include "cdfmm/operators/p2m.hpp"
#include "cdfmm/operators/p2p.hpp"
#include "cdfmm/plan/p2p/bsr.hpp"
#include "cdfmm/plan/p2p/compact.hpp"
#include "cdfmm/plan/p2p/dictionary.hpp"
#include "cdfmm/plan/p2p/leaf.hpp"
#include "cdfmm/plan/p2p/signed_dictionary.hpp"
#include "cdfmm/plan/static_coefficient.hpp"
#include "cdfmm/plan/static_plan.hpp"
#include "cdfmm/tree/adaptive_tree.hpp"
#include "cdfmm/tree/indexing.hpp"
#include "cdfmm/tree/morton.hpp"
#include "cdfmm/tree/node.hpp"
#include "cdfmm/tree/statistics.hpp"
#include "cdfmm/tree/static_topology.hpp"
#include "cdfmm/tree/uniform_tree.hpp"
#include "cdfmm/tree/uniform_topology.hpp"

// Pre-v0.2 compatibility include paths.
#include "cdfmm/adaptive_tree.hpp"
#include "cdfmm/coefficients.hpp"
#include "cdfmm/geometry.hpp"
#include "cdfmm/laplace_derivatives.hpp"
#include "cdfmm/multi_index.hpp"
#include "cdfmm/output_flags.hpp"
#include "cdfmm/precision.hpp"
#include "cdfmm/rectangular_prism.hpp"
#include "cdfmm/spherical_harmonics.hpp"
#include "cdfmm/taylor_jet.hpp"
#include "cdfmm/tetrahedron.hpp"
#include "cdfmm/timings.hpp"
#include "cdfmm/tree_node.hpp"
#include "cdfmm/uniform_tree.hpp"
#include "cdfmm/vec3.hpp"
#include "cdfmm/operators.hpp"
#include "cdfmm/static_operators.hpp"
#include "cdfmm/static_topology.hpp"

TEST_CASE("Canonical foundational headers coexist with compatibility shims")
{
    cdfmm::PhaseTiming timing;
    timing.add(0.25);

    const cdfmm::Vec3 displacement{1.0, 2.0, 3.0};
    const cdfmm::RectangularPrism prism{2.0, 3.0, 4.0};
    const cdfmm::PairTensor tensor{};

    CHECK(timing.calls == 1);
    CHECK(prism.volume() == 24.0);
    CHECK(cdfmm::dot(displacement, displacement) == 14.0);
    CHECK(tensor.xx == 0.0);
    CHECK(cdfmm::has_flag(cdfmm::OutputFlags::Both,
                          cdfmm::OutputFlags::Field));
}

TEST_CASE("Canonical operator and plan headers expose public boundaries")
{
    const cdfmm::MultiIndexSet basis(1);
    const cdfmm::Vec3 centre{};
    const std::array<cdfmm::Vec3, 1> source_positions{
        cdfmm::Vec3{0.25, -0.1, 0.2}};
    const std::array<cdfmm::Vec3, 1> moments{
        cdfmm::Vec3{1.0, 2.0, -0.5}};

    const cdfmm::StaticCoefficientOperator p2m =
        cdfmm::operators::p2m::build(basis, centre, source_positions);
    const cdfmm::CoeffVector multipole = cdfmm::operators::p2m::evaluate(
        basis, centre, source_positions, moments);
    CHECK(p2m.input_size == 3);
    CHECK(p2m.output_size == basis.size());
    CHECK(multipole.size() == static_cast<std::size_t>(basis.size()));

    const cdfmm::StaticCoefficientOperator m2m =
        cdfmm::operators::m2m::build(basis, cdfmm::Vec3{0.5, 0.0, 0.0});
    const cdfmm::StaticCoefficientOperator l2l =
        cdfmm::operators::l2l::build(basis, cdfmm::Vec3{0.0, 0.5, 0.0});
    CHECK(m2m.entries.size() > 0);
    CHECK(l2l.entries.size() > 0);

    const std::vector<double> m2l_matrix =
        cdfmm::operators::m2l::build_matrix(basis, cdfmm::Vec3{2.0, 0.0, 0.0});
    CHECK(m2l_matrix.size() ==
          static_cast<std::size_t>(basis.size() * basis.size()));

    const cdfmm::StaticL2PEvaluator l2p = cdfmm::operators::l2p::build(
        basis, centre, cdfmm::Vec3{0.5, 0.25, -0.75});
    const cdfmm::PotentialField local_field = cdfmm::operators::l2p::evaluate(
        basis, centre, cdfmm::Vec3{0.5, 0.25, -0.75}, multipole);
    CHECK(l2p.potential.size() == static_cast<std::size_t>(basis.size()));
    CHECK(std::isfinite(local_field.H.x));

    const cdfmm::Vec3 target{1.0, 0.0, 0.0};
    const cdfmm::Vec3 source{};
    const cdfmm::PairTensor pair = cdfmm::operators::p2p::build_pair(
        target, source);
    const cdfmm::PotentialField pair_field =
        cdfmm::operators::p2p::evaluate_pair(target, source, moments[0]);
    CHECK(std::isfinite(pair.xx));
    CHECK(std::isfinite(pair_field.H.x));

    const std::array<std::array<int, 2>, 1> interactions{{{{0, 0}}}};
    const std::array<cdfmm::Vec3, 1> targets{target};
    const std::array<cdfmm::Vec3, 1> sources{source};
    const cdfmm::StaticP2POperator canonical =
        cdfmm::operators::p2p::build(targets, sources, interactions);
    CHECK(canonical.target_count == 1);
    CHECK(canonical.source_count == 1);
    CHECK(canonical.blocks.size() == 1);

    // These names remain available through the old compatibility umbrella,
    // while their definitions now come from the structured plan headers.
    cdfmm::StaticP2PCompactPlan compatibility_plan;
    cdfmm::StaticM2LPlan compatibility_m2l;
    CHECK(compatibility_plan.target_count == 0);
    CHECK(compatibility_m2l.coefficient_count == 0);
}

TEST_CASE("Namespaced dynamic operators match flat compatibility functions")
{
    const cdfmm::MultiIndexSet basis(2);
    const cdfmm::Vec3 centre{0.1, -0.2, 0.3};
    const std::array<cdfmm::Vec3, 2> source_positions{
        cdfmm::Vec3{-0.35, 0.25, 0.45},
        cdfmm::Vec3{0.55, -0.15, -0.4}};
    const std::array<cdfmm::Vec3, 2> moments{
        cdfmm::Vec3{0.8, -0.4, 0.6},
        cdfmm::Vec3{-0.2, 0.7, 0.1}};

    const auto check_field = [](const cdfmm::PotentialField& actual,
                                const cdfmm::PotentialField& expected) {
        CHECK(actual.phi == Catch::Approx(expected.phi).epsilon(1.0e-12));
        CHECK(actual.H.x == Catch::Approx(expected.H.x).epsilon(1.0e-12));
        CHECK(actual.H.y == Catch::Approx(expected.H.y).epsilon(1.0e-12));
        CHECK(actual.H.z == Catch::Approx(expected.H.z).epsilon(1.0e-12));
    };

    const cdfmm::CoeffVector namespaced_multipole =
        cdfmm::operators::p2m::evaluate(
            basis, centre, source_positions, moments);
    const cdfmm::CoeffVector compatibility_multipole = cdfmm::p2m_dipole(
        basis, centre, source_positions, moments);
    REQUIRE(namespaced_multipole.size() == compatibility_multipole.size());
    for (std::size_t coefficient = 0;
         coefficient < namespaced_multipole.size(); ++coefficient) {
        CHECK(namespaced_multipole[coefficient] ==
              Catch::Approx(compatibility_multipole[coefficient])
                  .epsilon(1.0e-12));
    }

    const cdfmm::Vec3 m2m_displacement{0.17, -0.23, 0.11};
    cdfmm::CoeffVector namespaced_parent(basis.size(), 0.25);
    cdfmm::CoeffVector compatibility_parent(basis.size(), 0.25);
    cdfmm::operators::m2m::apply(
        basis, m2m_displacement, namespaced_multipole, namespaced_parent);
    cdfmm::m2m_add(
        basis, m2m_displacement, compatibility_multipole, compatibility_parent);
    REQUIRE(namespaced_parent.size() == compatibility_parent.size());
    for (std::size_t coefficient = 0; coefficient < namespaced_parent.size();
         ++coefficient) {
        CHECK(namespaced_parent[coefficient] ==
              Catch::Approx(compatibility_parent[coefficient])
                  .epsilon(1.0e-12));
    }

    const cdfmm::Vec3 m2l_displacement{1.7, -1.1, 0.9};
    cdfmm::CoeffVector namespaced_local(basis.size(), -0.15);
    cdfmm::CoeffVector compatibility_local(basis.size(), -0.15);
    cdfmm::operators::m2l::apply(
        basis, m2l_displacement, namespaced_parent, namespaced_local);
    cdfmm::m2l_add(
        basis, m2l_displacement, compatibility_parent, compatibility_local);
    REQUIRE(namespaced_local.size() == compatibility_local.size());
    for (std::size_t coefficient = 0; coefficient < namespaced_local.size();
         ++coefficient) {
        CHECK(namespaced_local[coefficient] ==
              Catch::Approx(compatibility_local[coefficient])
                  .epsilon(1.0e-12));
    }

    const cdfmm::Vec3 l2l_displacement{-0.12, 0.21, -0.08};
    cdfmm::CoeffVector namespaced_child(basis.size(), 0.4);
    cdfmm::CoeffVector compatibility_child(basis.size(), 0.4);
    cdfmm::operators::l2l::apply(
        basis, l2l_displacement, namespaced_local, namespaced_child);
    cdfmm::l2l_add(
        basis, l2l_displacement, compatibility_local, compatibility_child);
    REQUIRE(namespaced_child.size() == compatibility_child.size());
    for (std::size_t coefficient = 0; coefficient < namespaced_child.size();
         ++coefficient) {
        CHECK(namespaced_child[coefficient] ==
              Catch::Approx(compatibility_child[coefficient])
                  .epsilon(1.0e-12));
    }

    const cdfmm::Vec3 target{1.2, -0.7, 0.4};
    const cdfmm::PotentialField namespaced_local_field =
        cdfmm::operators::l2p::evaluate(
            basis, centre, target, namespaced_child,
            cdfmm::OutputFlags::Both);
    const cdfmm::PotentialField compatibility_local_field = cdfmm::l2p_eval(
        basis, centre, target, compatibility_child, cdfmm::OutputFlags::Both);
    check_field(namespaced_local_field, compatibility_local_field);

    const cdfmm::PotentialField namespaced_multipole_field =
        cdfmm::operators::m2p::evaluate(
            basis, compatibility_multipole, centre, target,
            cdfmm::OutputFlags::Both);
    const cdfmm::PotentialField compatibility_multipole_field =
        cdfmm::m2p_eval(basis, compatibility_multipole, centre, target,
                        cdfmm::OutputFlags::Both);
    check_field(namespaced_multipole_field, compatibility_multipole_field);

    const cdfmm::Vec3 pair_target{1.2, -0.7, 0.4};
    const cdfmm::Vec3 pair_source{0.1, 0.2, -0.3};
    const cdfmm::Vec3 pair_moment{0.8, -0.4, 0.6};
    const cdfmm::PotentialField namespaced_pair =
        cdfmm::operators::p2p::evaluate_pair(
            pair_target, pair_source, pair_moment, cdfmm::OutputFlags::Both);
    const cdfmm::PotentialField compatibility_pair = cdfmm::p2p_dipole_pair(
        pair_target, pair_source, pair_moment, cdfmm::OutputFlags::Both);
    check_field(namespaced_pair, compatibility_pair);

    const std::array<cdfmm::Vec3, 2> sum_sources{
        pair_target, pair_source};
    const std::array<cdfmm::Vec3, 2> sum_moments{
        cdfmm::Vec3{2.0, 3.0, 4.0}, pair_moment};
    const cdfmm::PotentialField namespaced_sum =
        cdfmm::operators::p2p::evaluate_sum(
            pair_target, sum_sources, sum_moments, cdfmm::OutputFlags::Both, 0);
    const cdfmm::PotentialField compatibility_sum = cdfmm::p2p_dipole_sum(
        pair_target, sum_sources, sum_moments, cdfmm::OutputFlags::Both, 0);
    check_field(namespaced_sum, compatibility_sum);
    check_field(namespaced_sum, namespaced_pair);

    const cdfmm::PotentialField potential_only =
        cdfmm::operators::p2p::evaluate_pair(
            pair_target, pair_source, pair_moment,
            cdfmm::OutputFlags::Potential);
    CHECK(potential_only.phi ==
          Catch::Approx(namespaced_pair.phi).epsilon(1.0e-12));
    CHECK(potential_only.H.x == 0.0);
    CHECK(potential_only.H.y == 0.0);
    CHECK(potential_only.H.z == 0.0);
}
