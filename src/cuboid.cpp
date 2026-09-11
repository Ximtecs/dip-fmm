// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/cuboid.hpp"
#include "cdfmm/rectangular_prism.hpp"

#include <cmath>
#include <numbers>
#include <string>
#include <stdexcept>

namespace cdfmm {
namespace {

double factorial(const int n)
{
    double result = 1.0;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

void validate_size(const CuboidSize& h, const char* name)
{
    if (!(std::isfinite(h.hx) && std::isfinite(h.hy) &&
          std::isfinite(h.hz) && h.hx > 0.0 && h.hy > 0.0 && h.hz > 0.0)) {
        throw std::invalid_argument(std::string(name) +
                                    " dimensions must be finite and positive");
    }
}

} // namespace

double cuboid_averaged_monomial(const MultiIndex& beta, const Vec3& d,
                                const CuboidSize& h)
{
    validate_size(h, "cuboid");
    const int powers[3] = {beta.ax, beta.ay, beta.az};
    const double offsets[3] = {d.x, d.y, d.z};
    const double lengths[3] = {h.hx, h.hy, h.hz};
    double result = 1.0;
    for (int axis = 0; axis < 3; ++axis) {
        double axis_sum = 0.0;
        for (int gamma = 0; gamma <= powers[axis]; gamma += 2) {
            axis_sum += std::pow(offsets[axis], powers[axis] - gamma) /
                factorial(powers[axis] - gamma) *
                std::pow(lengths[axis], gamma) /
                (std::pow(2.0, gamma) * factorial(gamma + 1));
        }
        result *= axis_sum;
    }
    return result;
}

double rectangular_prism_averaged_monomial(const MultiIndex& beta,
                                           const Vec3& d,
                                           const RectangularPrism& prism)
{
    return cuboid_averaged_monomial(beta, d, prism);
}

PairTensor build_pair_tensor(const Vec3& target_position,
                             const Vec3& source_position,
                             const SourceGeometry source_geometry,
                             const TargetGeometry target_geometry,
                             const CuboidSize& source_size,
                             const CuboidSize& target_size,
                             const bool omit_singular_point_pair)
{
    if (source_geometry == SourceGeometry::Tetrahedron ||
        target_geometry == TargetGeometry::Tetrahedron) {
        throw std::invalid_argument(
            "build_pair_tensor does not accept tetrahedral geometry; "
            "use the tetrahedron P2P operator with tetrahedron records");
    }
    const Vec3 r = target_position - source_position;
    // An explicit identity map marks a point-source self interaction.  The
    // singular point field is omitted even when the target is a finite volume;
    // finite sources are handled by their analytical self-limit below.
    if (omit_singular_point_pair &&
        source_geometry == SourceGeometry::PointDipole) {
        return {};
    }
    if (source_geometry == SourceGeometry::RectangularPrism &&
        target_geometry == TargetGeometry::RectangularPrism) {
        return rectangular_prism_rectangular_prism_tensor(
            r, source_size, target_size);
    }

    // Reciprocity converts a point-to-volume average into the corresponding
    // finite-source field, including the required total-moment normalisation.
    if (source_geometry == SourceGeometry::PointDipole &&
        target_geometry == TargetGeometry::RectangularPrism) {
        return rectangular_prism_point_tensor(r, target_size);
    }

    if (source_geometry == SourceGeometry::RectangularPrism) {
        return rectangular_prism_point_tensor(r, source_size);
    }

    const double r2 = dot(r, r);
    if (r2 == 0.0) {
        if (omit_singular_point_pair) {
            return {};
        }
        throw std::domain_error("coincident point dipole and point target");
    }
    const double inverse_r = 1.0 / std::sqrt(r2);
    const double inverse_r3 = inverse_r / r2;
    const double diagonal = inverse_r3 / (4.0 * std::numbers::pi);
    const double common = 3.0 * diagonal / r2;
    return {common * r.x * r.x - diagonal,
            common * r.x * r.y, common * r.x * r.z,
            common * r.y * r.y - diagonal, common * r.y * r.z,
            common * r.z * r.z - diagonal};
}

} // namespace cdfmm
