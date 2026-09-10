// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/operators/l2p.hpp"

#include <cstddef>

namespace cdfmm {

StaticL2PEvaluator build_static_l2p_evaluator(
    const MultiIndexSet& basis,
    const Vec3& centre,
    const Vec3& target)
{
    StaticL2PEvaluator result;
    result.potential.resize(static_cast<std::size_t>(basis.size()));
    for (auto& row : result.field) {
        row.resize(static_cast<std::size_t>(basis.size()));
    }
    const Vec3 displacement = target - centre;
    for (int beta_index = 0; beta_index < basis.size(); ++beta_index) {
        const MultiIndex beta = basis[beta_index];
        result.potential[static_cast<std::size_t>(beta_index)] =
            MultiIndexSet::monomial_over_factorial(displacement, beta);
        if (beta.ax > 0) {
            result.field[0][static_cast<std::size_t>(beta_index)] =
                -MultiIndexSet::monomial_over_factorial(
                    displacement, {beta.ax - 1, beta.ay, beta.az});
        }
        if (beta.ay > 0) {
            result.field[1][static_cast<std::size_t>(beta_index)] =
                -MultiIndexSet::monomial_over_factorial(
                    displacement, {beta.ax, beta.ay - 1, beta.az});
        }
        if (beta.az > 0) {
            result.field[2][static_cast<std::size_t>(beta_index)] =
                -MultiIndexSet::monomial_over_factorial(
                    displacement, {beta.ax, beta.ay, beta.az - 1});
        }
    }
    return result;
}

StaticL2PEvaluator build_static_l2p_evaluator(
    const SphericalHarmonicBasis& basis,
    const Vec3& centre,
    const Vec3& target)
{
    const SolidHarmonicValues regular =
        regular_solid_harmonics(basis, target - centre);
    StaticL2PEvaluator result;
    result.potential = regular.values;
    for (auto& row : result.field) {
        row.resize(static_cast<std::size_t>(basis.size()));
    }
    for (int mode = 0; mode < basis.size(); ++mode) {
        const Vec3 gradient =
            regular.gradients[static_cast<std::size_t>(mode)];
        result.field[0][static_cast<std::size_t>(mode)] = -gradient.x;
        result.field[1][static_cast<std::size_t>(mode)] = -gradient.y;
        result.field[2][static_cast<std::size_t>(mode)] = -gradient.z;
    }
    return result;
}

StaticL2PEvaluator build_static_cuboid_l2p_evaluator(
    const MultiIndexSet& basis,
    const Vec3& centre,
    const Vec3& target,
    const CuboidSize& target_size)
{
    StaticL2PEvaluator result;
    result.potential.resize(static_cast<std::size_t>(basis.size()));
    for (auto& row : result.field) {
        row.resize(static_cast<std::size_t>(basis.size()));
    }
    const Vec3 displacement = target - centre;
    for (int beta_index = 0; beta_index < basis.size(); ++beta_index) {
        const MultiIndex beta = basis[beta_index];
        result.potential[static_cast<std::size_t>(beta_index)] =
            cuboid_averaged_monomial(beta, displacement, target_size);
        if (beta.ax > 0) {
            result.field[0][static_cast<std::size_t>(beta_index)] =
                -cuboid_averaged_monomial(
                    {beta.ax - 1, beta.ay, beta.az}, displacement,
                    target_size);
        }
        if (beta.ay > 0) {
            result.field[1][static_cast<std::size_t>(beta_index)] =
                -cuboid_averaged_monomial(
                    {beta.ax, beta.ay - 1, beta.az}, displacement,
                    target_size);
        }
        if (beta.az > 0) {
            result.field[2][static_cast<std::size_t>(beta_index)] =
                -cuboid_averaged_monomial(
                    {beta.ax, beta.ay, beta.az - 1}, displacement,
                    target_size);
        }
    }
    return result;
}

StaticL2PEvaluator build_static_cuboid_l2p_evaluator(
    const SphericalHarmonicBasis& basis,
    const Vec3& centre,
    const Vec3& target,
    const CuboidSize& target_size)
{
    StaticL2PEvaluator result;
    result.potential.resize(static_cast<std::size_t>(basis.size()));
    for (auto& row : result.field) {
        row.resize(static_cast<std::size_t>(basis.size()));
    }
    const Vec3 displacement = target - centre;
    for (int mode = 0; mode < basis.size(); ++mode) {
        for (const SolidHarmonicTerm& term : basis.polynomial(mode)) {
            result.potential[static_cast<std::size_t>(mode)] +=
                term.coefficient *
                MultiIndexSet::multi_factorial(term.power) *
                cuboid_averaged_monomial(
                    term.power, displacement, target_size);
            const int powers[3] = {
                term.power.ax, term.power.ay, term.power.az};
            for (int component = 0; component < 3; ++component) {
                if (powers[component] == 0) {
                    continue;
                }
                MultiIndex derivative = term.power;
                if (component == 0) {
                    --derivative.ax;
                } else if (component == 1) {
                    --derivative.ay;
                } else {
                    --derivative.az;
                }
                result.field[static_cast<std::size_t>(component)]
                            [static_cast<std::size_t>(mode)] -=
                    term.coefficient * powers[component] *
                    MultiIndexSet::multi_factorial(derivative) *
                    cuboid_averaged_monomial(
                        derivative, displacement, target_size);
            }
        }
    }
    return result;
}

StaticL2PEvaluator build_static_tetrahedron_l2p_evaluator(
    const MultiIndexSet& basis,
    const Vec3& centre,
    const Vec3& target,
    const Tetrahedron& target_tetrahedron)
{
    static_cast<void>(tetrahedron_volume(target_tetrahedron));
    StaticL2PEvaluator result;
    result.potential.resize(static_cast<std::size_t>(basis.size()));
    for (auto& row : result.field) {
        row.resize(static_cast<std::size_t>(basis.size()));
    }
    const Vec3 displacement = target - centre;
    for (int beta_index = 0; beta_index < basis.size(); ++beta_index) {
        const MultiIndex beta = basis[beta_index];
        result.potential[static_cast<std::size_t>(beta_index)] =
            tetrahedron_averaged_monomial(
                beta, displacement, target_tetrahedron);
        if (beta.ax > 0) {
            result.field[0][static_cast<std::size_t>(beta_index)] =
                -tetrahedron_averaged_monomial(
                    {beta.ax - 1, beta.ay, beta.az}, displacement,
                    target_tetrahedron);
        }
        if (beta.ay > 0) {
            result.field[1][static_cast<std::size_t>(beta_index)] =
                -tetrahedron_averaged_monomial(
                    {beta.ax, beta.ay - 1, beta.az}, displacement,
                    target_tetrahedron);
        }
        if (beta.az > 0) {
            result.field[2][static_cast<std::size_t>(beta_index)] =
                -tetrahedron_averaged_monomial(
                    {beta.ax, beta.ay, beta.az - 1}, displacement,
                    target_tetrahedron);
        }
    }
    return result;
}

StaticL2PEvaluator build_static_tetrahedron_l2p_evaluator(
    const SphericalHarmonicBasis& basis,
    const Vec3& centre,
    const Vec3& target,
    const Tetrahedron& target_tetrahedron)
{
    static_cast<void>(tetrahedron_volume(target_tetrahedron));
    StaticL2PEvaluator result;
    result.potential.resize(static_cast<std::size_t>(basis.size()));
    for (auto& row : result.field) {
        row.resize(static_cast<std::size_t>(basis.size()));
    }
    const Vec3 displacement = target - centre;
    for (int mode = 0; mode < basis.size(); ++mode) {
        for (const SolidHarmonicTerm& term : basis.polynomial(mode)) {
            result.potential[static_cast<std::size_t>(mode)] +=
                term.coefficient *
                MultiIndexSet::multi_factorial(term.power) *
                tetrahedron_averaged_monomial(
                    term.power, displacement, target_tetrahedron);
            const int powers[3] = {
                term.power.ax, term.power.ay, term.power.az};
            for (int component = 0; component < 3; ++component) {
                if (powers[component] == 0) {
                    continue;
                }
                MultiIndex derivative = term.power;
                if (component == 0) {
                    --derivative.ax;
                } else if (component == 1) {
                    --derivative.ay;
                } else {
                    --derivative.az;
                }
                result.field[static_cast<std::size_t>(component)]
                            [static_cast<std::size_t>(mode)] -=
                    term.coefficient * powers[component] *
                    MultiIndexSet::multi_factorial(derivative) *
                    tetrahedron_averaged_monomial(
                        derivative, displacement, target_tetrahedron);
            }
        }
    }
    return result;
}

} // namespace cdfmm

namespace cdfmm::operators::l2p {

StaticL2PEvaluator build(
    const MultiIndexSet& basis, const Vec3& centre, const Vec3& target)
{
    return build_static_l2p_evaluator(basis, centre, target);
}

StaticL2PEvaluator build(
    const SphericalHarmonicBasis& basis, const Vec3& centre,
    const Vec3& target)
{
    return build_static_l2p_evaluator(basis, centre, target);
}

StaticL2PEvaluator build_cuboid(
    const MultiIndexSet& basis, const Vec3& centre, const Vec3& target,
    const CuboidSize& target_size)
{
    return build_static_cuboid_l2p_evaluator(
        basis, centre, target, target_size);
}

StaticL2PEvaluator build_cuboid(
    const SphericalHarmonicBasis& basis, const Vec3& centre,
    const Vec3& target, const CuboidSize& target_size)
{
    return build_static_cuboid_l2p_evaluator(
        basis, centre, target, target_size);
}

StaticL2PEvaluator build_tetrahedron(
    const MultiIndexSet& basis, const Vec3& centre, const Vec3& target,
    const Tetrahedron& target_tetrahedron)
{
    return build_static_tetrahedron_l2p_evaluator(
        basis, centre, target, target_tetrahedron);
}

StaticL2PEvaluator build_tetrahedron(
    const SphericalHarmonicBasis& basis, const Vec3& centre,
    const Vec3& target, const Tetrahedron& target_tetrahedron)
{
    return build_static_tetrahedron_l2p_evaluator(
        basis, centre, target, target_tetrahedron);
}

PotentialField evaluate(
    const MultiIndexSet& basis, const Vec3& centre, const Vec3& target,
    const std::span<const double> local, const OutputFlags output)
{
    PotentialField result;
    const Vec3 dx = target - centre;

    for (int ib = 0; ib < basis.size(); ++ib) {
        const MultiIndex beta = basis[ib];

        if (has_flag(output, OutputFlags::Potential)) {
            result.phi += local[ib] * MultiIndexSet::monomial_over_factorial(dx, beta);
        }

        if (has_flag(output, OutputFlags::Field)) {
            // Field is H = -grad(phi), hence the explicit minus signs.
            if (beta.ax > 0) {
                result.H.x -= local[ib] * MultiIndexSet::monomial_over_factorial(
                    dx, {beta.ax - 1, beta.ay, beta.az});
            }
            if (beta.ay > 0) {
                result.H.y -= local[ib] * MultiIndexSet::monomial_over_factorial(
                    dx, {beta.ax, beta.ay - 1, beta.az});
            }
            if (beta.az > 0) {
                result.H.z -= local[ib] * MultiIndexSet::monomial_over_factorial(
                    dx, {beta.ax, beta.ay, beta.az - 1});
            }
        }
    }

    return result;
}

} // namespace cdfmm::operators::l2p
