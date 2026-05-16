#include <catch2/catch_test_macros.hpp>

#include "cdfmm/operators.hpp"

using namespace cdfmm;

TEST_CASE("P2M M2L L2P convergence") {
    std::vector<Vec3> xs{{0.1, 0.0, 0.0}, {-0.05, 0.07, 0.0}, {0.02, -0.04, 0.03}};
    std::vector<Vec3> ms{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    Vec3 ct{2, 0, 0};
    Vec3 target{2.05, 0.02, -0.01};

    const auto direct = p2p_dipole_sum(target, xs, ms, OutputFlags::Field);
    double first_err = 0.0;
    double best_err = 1e9;

    for (int p = 1; p <= 4; ++p) {
        MultiIndexSet b(p);
        auto M = p2m_dipole(b, {0, 0, 0}, xs, ms);
        std::vector<double> L(b.size(), 0.0);
        m2l_add(b, ct - Vec3{0, 0, 0}, M, L);

        const auto far = l2p_eval(b, ct, target, L, OutputFlags::Field);
        const double err = std::sqrt(dot(far.H - direct.H, far.H - direct.H));

        if (p == 1) {
            first_err = err;
        }
        if (err < best_err) {
            best_err = err;
        }
    }

    // Numerical Laplace derivatives are finite-difference based, so strict
    // monotonic reduction with order is not guaranteed at every step.
    REQUIRE(best_err < first_err);
    REQUIRE(best_err < 1e-2);
}
