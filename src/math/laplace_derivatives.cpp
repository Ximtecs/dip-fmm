// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/math/laplace_derivatives.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace cdfmm {

static constexpr double k_inv_four_pi = 1.0 / (4.0 * std::numbers::pi);

std::vector<double> laplace_derivatives_raw(const MultiIndexSet& basis,
                                           const Vec3& r)
{
    // The previous coordinate-jet construction required degree-one entries,
    // even when only the constant term was requested. Preserve its exception.
    if (basis.order() < 1) {
        throw std::out_of_range("multi-index not found");
    }
    const double r2 = r.x * r.x + r.y * r.y + r.z * r.z;
    if (r2 <= 0.0) {
        throw std::invalid_argument(
            "TaylorJet::invsqrt requires positive constant coefficient");
    }

    // For g(h) = 1/|r+h|, contract |r+h|^2 grad(g) = -(r+h)g with h
    // and equate coefficients. The Taylor coefficients c_alpha satisfy
    //
    // |r|^2 n c_alpha = -(2n-1) sum_i r_i c_(alpha-e_i)
    //                  -(n-1) sum_i c_(alpha-2e_i),   n = |alpha|.
    //
    // All dependencies precede alpha in total-degree order. The recurrence
    // takes constant work per coefficient instead of composing whole jets.
    const MultiIndex directions[3]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    std::vector<double> coefficients(static_cast<std::size_t>(basis.size()));
    coefficients[0] = 1.0 / std::sqrt(r2);
    for (int index = 1; index < basis.size(); ++index) {
        const MultiIndex alpha = basis[index];
        const int degree = alpha.degree();
        double first = 0.0;
        double second = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            if (alpha[axis] == 0) {
                continue;
            }
            const MultiIndex previous = sub(alpha, directions[axis]);
            first += r[axis] *
                coefficients[static_cast<std::size_t>(basis.index(previous))];
            if (alpha[axis] >= 2) {
                const MultiIndex twice_previous = sub(previous, directions[axis]);
                second += coefficients[
                    static_cast<std::size_t>(basis.index(twice_previous))];
            }
        }
        coefficients[static_cast<std::size_t>(index)] =
            -((2.0 * degree - 1.0) * first + (degree - 1.0) * second) /
            (r2 * degree);
    }

    // M2L and M2P consume raw derivatives D_alpha G, with
    // G = g/(4*pi) and c_alpha = D_alpha g / alpha!.
    for (int index = 0; index < basis.size(); ++index) {
        coefficients[static_cast<std::size_t>(index)] *= k_inv_four_pi;
        coefficients[static_cast<std::size_t>(index)] *=
            MultiIndexSet::multi_factorial(basis[index]);
    }
    return coefficients;
}

} // namespace cdfmm
