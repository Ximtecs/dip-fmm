// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/operators.hpp"

#include <cmath>
#include <numbers>

#include "cdfmm/laplace_derivatives.hpp"

namespace cdfmm {

CoeffVector p2m_dipole(
    const MultiIndexSet& basis,
    const Vec3& centre,
    std::span<const Vec3> source_positions,
    std::span<const Vec3> dipole_moments) {
    CoeffVector M(basis.size(), 0.0);

    for (size_t j = 0; j < source_positions.size(); ++j) {
        const Vec3 dx = source_positions[j] - centre;

        for (int i = 0; i < basis.size(); ++i) {
            const MultiIndex alpha = basis[i];
            double value = 0.0;

            if (alpha.ax > 0) {
                value += dipole_moments[j].x * MultiIndexSet::monomial_over_factorial(
                                                  dx, {alpha.ax - 1, alpha.ay, alpha.az});
            }
            if (alpha.ay > 0) {
                value += dipole_moments[j].y * MultiIndexSet::monomial_over_factorial(
                                                  dx, {alpha.ax, alpha.ay - 1, alpha.az});
            }
            if (alpha.az > 0) {
                value += dipole_moments[j].z * MultiIndexSet::monomial_over_factorial(
                                                  dx, {alpha.ax, alpha.ay, alpha.az - 1});
            }

            const double sign = (alpha.degree() % 2 == 0) ? 1.0 : -1.0;
            M[i] += sign * value;
        }
    }

    return M;
}

void m2m_add(const MultiIndexSet& basis, const Vec3& d, std::span<const double> child,
             std::span<double> parent) {
    for (int ia = 0; ia < basis.size(); ++ia) {
        const MultiIndex alpha = basis[ia];

        for (int ig = 0; ig < basis.size(); ++ig) {
            const MultiIndex gamma = basis[ig];
            if (!leq(gamma, alpha)) {
                continue;
            }

            parent[ia] += MultiIndexSet::monomial_over_factorial(d, gamma) *
                          child[basis.index(sub(alpha, gamma))];
        }
    }
}

void m2l_add(const MultiIndexSet& basis, const Vec3& R, std::span<const double> M,
             std::span<double> L) {
    MultiIndexSet deriv_basis(2 * basis.order());
    const auto D = laplace_derivatives_raw(deriv_basis, R);

    for (int ib = 0; ib < basis.size(); ++ib) {
        const MultiIndex beta = basis[ib];

        for (int ia = 0; ia < basis.size(); ++ia) {
            const MultiIndex alpha = basis[ia];
            L[ib] += M[ia] * D[deriv_basis.index(add(alpha, beta))];
        }
    }
}

void l2l_add(const MultiIndexSet& basis, const Vec3& d, std::span<const double> parent,
             std::span<double> child) {
    for (int ib = 0; ib < basis.size(); ++ib) {
        const MultiIndex beta = basis[ib];

        for (int ig = 0; ig < basis.size(); ++ig) {
            const MultiIndex gamma = basis[ig];
            const MultiIndex sum = add(beta, gamma);
            if (sum.degree() > basis.order()) {
                continue;
            }

            child[ib] += MultiIndexSet::monomial_over_factorial(d, gamma) *
                         parent[basis.index(sum)];
        }
    }
}

PotentialField l2p_eval(const MultiIndexSet& basis, const Vec3& centre, const Vec3& target,
                        std::span<const double> L, OutputFlags output) {
    PotentialField result;
    const Vec3 dx = target - centre;

    for (int ib = 0; ib < basis.size(); ++ib) {
        const MultiIndex beta = basis[ib];

        if (has_flag(output, OutputFlags::Potential)) {
            result.phi += L[ib] * MultiIndexSet::monomial_over_factorial(dx, beta);
        }

        if (has_flag(output, OutputFlags::Field)) {
            if (beta.ax > 0) {
                result.H.x -= L[ib] * MultiIndexSet::monomial_over_factorial(
                                          dx, {beta.ax - 1, beta.ay, beta.az});
            }
            if (beta.ay > 0) {
                result.H.y -= L[ib] * MultiIndexSet::monomial_over_factorial(
                                          dx, {beta.ax, beta.ay - 1, beta.az});
            }
            if (beta.az > 0) {
                result.H.z -= L[ib] * MultiIndexSet::monomial_over_factorial(
                                          dx, {beta.ax, beta.ay, beta.az - 1});
            }
        }
    }

    return result;
}

PotentialField p2p_dipole_pair(const Vec3& target, const Vec3& source, const Vec3& moment,
                               OutputFlags output) {
    PotentialField result;
    const Vec3 r = target - source;
    const double r2 = dot(r, r);
    const double rinv = 1.0 / std::sqrt(r2);
    const double rinv3 = rinv * rinv * rinv;
    const double c = 1.0 / (4.0 * std::numbers::pi);
    const double m_dot_r = dot(moment, r);

    if (has_flag(output, OutputFlags::Potential)) {
        result.phi = c * m_dot_r * rinv3;
    }

    if (has_flag(output, OutputFlags::Field)) {
        const double rinv5 = rinv3 * rinv * rinv;
        result.H = (r * (3.0 * m_dot_r * rinv5) - moment * rinv3) * c;
    }

    return result;
}

PotentialField p2p_dipole_sum(const Vec3& target, std::span<const Vec3> sources,
                              std::span<const Vec3> moments, OutputFlags output,
                              int self_index) {
    PotentialField result;

    for (size_t i = 0; i < sources.size(); ++i) {
        if (static_cast<int>(i) == self_index) {
            continue;
        }

        const PotentialField pair =
            p2p_dipole_pair(target, sources[i], moments[i], output);
        result.phi += pair.phi;
        result.H += pair.H;
    }

    return result;
}

} // namespace cdfmm
