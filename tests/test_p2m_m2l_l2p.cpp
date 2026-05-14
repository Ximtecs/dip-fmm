#include <catch2/catch_test_macros.hpp>

#include "cdfmm/operators.hpp"

using namespace cdfmm;

TEST_CASE("P2M M2L L2P convergence") {
    std::vector<Vec3> xs{{0.1, 0.0, 0.0}, {-0.05, 0.07, 0.0}, {0.02, -0.04, 0.03}};
    std::vector<Vec3> ms{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    Vec3 ct{2, 0, 0};
    Vec3 target{2.05, 0.02, -0.01};

    const auto direct = p2p_dipole_sum(target, xs, ms, OutputFlags::Field);
    double prev = 1e9;

    for (int p = 1; p <= 4; ++p) {
        MultiIndexSet b(p);
        auto M = p2m_dipole(b, {0, 0, 0}, xs, ms);
        std::vector<double> L(b.size(), 0.0);
        m2l_add(b, ct - Vec3{0, 0, 0}, M, L);

        const auto far = l2p_eval(b, ct, target, L, OutputFlags::Field);
        const double err = std::sqrt(dot(far.H - direct.H, far.H - direct.H));

        REQUIRE(err < prev);
        prev = err;
    }
}
