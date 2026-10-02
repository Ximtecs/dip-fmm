// SPDX-License-Identifier: Apache-2.0

#include "geometry/primitives/rectangular_prism_moments.hpp"

#include <array>

namespace cdfmm::detail {

std::vector<double> rectangular_prism_averaged_monomials(
    const MultiIndexSet& basis, const Vec3& displacement,
    const RectangularPrism& prism)
{
    std::vector<double> averages(static_cast<std::size_t>(basis.size()));
    if (averages.empty()) {
        return averages;
    }

    // Keep the scalar path's prism validation and historical exception text.
    static_cast<void>(rectangular_prism_averaged_monomial(
        {0, 0, 0}, displacement, prism));

    const double offsets[3]{displacement.x, displacement.y, displacement.z};
    const double lengths[3]{prism.hx, prism.hy, prism.hz};
    std::array<std::vector<double>, 3> axis_averages;
    for (auto& values : axis_averages) {
        values.resize(static_cast<std::size_t>(basis.order() + 1));
    }
    for (int axis = 0; axis < 3; ++axis) {
        for (int power = 0; power <= basis.order(); ++power) {
            double axis_sum = 0.0;
            for (int gamma = 0; gamma <= power; gamma += 2) {
                axis_sum += std::pow(offsets[axis], power - gamma) /
                    MultiIndexSet::factorial(power - gamma) *
                    std::pow(lengths[axis], gamma) /
                    (std::pow(2.0, gamma) *
                     MultiIndexSet::factorial(gamma + 1));
            }
            axis_averages[static_cast<std::size_t>(axis)]
                         [static_cast<std::size_t>(power)] = axis_sum;
        }
    }

    for (int index = 0; index < basis.size(); ++index) {
        const MultiIndex alpha = basis[index];
        double value = 1.0;
        value *= axis_averages[0][static_cast<std::size_t>(alpha.ax)];
        value *= axis_averages[1][static_cast<std::size_t>(alpha.ay)];
        value *= axis_averages[2][static_cast<std::size_t>(alpha.az)];
        averages[static_cast<std::size_t>(index)] = value;
    }
    return averages;
}

} // namespace cdfmm::detail
