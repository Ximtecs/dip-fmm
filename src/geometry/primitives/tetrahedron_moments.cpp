// SPDX-License-Identifier: Apache-2.0
#include "geometry/primitives/tetrahedron_moments.hpp"

namespace cdfmm::detail {

std::vector<double> tetrahedron_averaged_monomials(
    const MultiIndexSet& basis, const Vec3& d, const Tetrahedron& tetrahedron)
{
    (void)tetrahedron_volume(tetrahedron);
    std::vector<long double> coefficients(static_cast<std::size_t>(basis.size()));
    if (coefficients.empty()) {
        return {};
    }
    coefficients[0] = 1.0L;

    // Dirichlet integration gives E[x^alpha]/alpha! = 6/(|alpha|+3)!
    // times the t^alpha coefficient of product_v (1 - (d+v).t)^-1.
    // For one factor, (1 - w.t) H_new = H_old, so total-degree order
    // permits an in-place recurrence using three already updated neighbours.
    // Four factors therefore cost O(number of monomials), rather than the
    // scalar formula's Cartesian product of barycentric partitions.
    for (const Vec3& vertex : tetrahedron.vertices) {
        const long double coordinate[3]{
            static_cast<long double>(d.x) + vertex.x,
            static_cast<long double>(d.y) + vertex.y,
            static_cast<long double>(d.z) + vertex.z};
        for (int index = 1; index < basis.size(); ++index) {
            const MultiIndex alpha = basis[index];
            long double value = coefficients[static_cast<std::size_t>(index)];
            for (int axis = 0; axis < 3; ++axis) {
                if (alpha[axis] == 0) {
                    continue;
                }
                MultiIndex previous = alpha;
                if (axis == 0) {
                    --previous.ax;
                } else if (axis == 1) {
                    --previous.ay;
                } else {
                    --previous.az;
                }
                value += coordinate[axis] *
                    coefficients[tetrahedron_monomial_index(previous)];
            }
            coefficients[static_cast<std::size_t>(index)] = value;
        }
    }

    // Accumulate in extended precision: opposing vertices can strongly cancel
    // in centred tetrahedra. Cast only the final factorial-normalised average.
    std::vector<double> averages(coefficients.size());
    long double normalisation = 1.0L;
    int degree = 0;
    for (int index = 0; index < basis.size(); ++index) {
        const int next_degree = basis[index].degree();
        while (degree < next_degree) {
            ++degree;
            normalisation /= degree + 3;
        }
        averages[static_cast<std::size_t>(index)] = static_cast<double>(
            coefficients[static_cast<std::size_t>(index)] * normalisation);
    }
    return averages;
}

} // namespace cdfmm::detail
