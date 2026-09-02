// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/static_operators.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <numbers>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include "cdfmm/laplace_derivatives.hpp"

namespace cdfmm {

void apply_static_m2l_plan(const StaticM2LPlan &plan, const int level,
    const std::span<const double> multipoles,
                           const std::span<double> locals) {
    const int n = plan.coefficient_count;
    const int target_begin =
        plan.level_target_begin[static_cast<std::size_t>(level)];
  const int target_end = plan.level_target_end[static_cast<std::size_t>(level)];
  const double *multipole_scale =
      plan.multipole_scaling.data() + static_cast<std::size_t>(level) * n;
  const double *local_scale =
      plan.local_scaling.data() + static_cast<std::size_t>(level) * n;

    // Uniform-tree nodes are level ordered. Iterating only this level avoids
    // revisiting every other node for each downward-pass level; one iteration
    // owns one target coefficient and therefore needs no atomic accumulation.
    const std::ptrdiff_t output_count =
        static_cast<std::ptrdiff_t>(target_end - target_begin) * n;
#pragma omp parallel for schedule(static) if (output_count >= 256)
    for (std::ptrdiff_t output = 0; output < output_count; ++output) {
        const int target = target_begin + static_cast<int>(output / n);
        const int beta = static_cast<int>(output % n);
        double value = 0.0;
        const int row_begin = plan.target_row_offsets[target];
        const int row_end = plan.target_row_offsets[target + 1];
    for (int interaction = row_begin; interaction < row_end; ++interaction) {
            const int source = plan.source_nodes[interaction];
            const int matrix_id = plan.matrix_ids[interaction];
      const double *matrix_column =
          plan.matrices.data() + static_cast<std::size_t>(matrix_id) * n * n +
          beta;
      const double *source_M =
          multipoles.data() + static_cast<std::size_t>(source) * n;
            for (int alpha = 0; alpha < n; ++alpha) {
                value += matrix_column[static_cast<std::size_t>(alpha) * n] *
                    multipole_scale[alpha] * source_M[alpha];
            }
        }
        locals[static_cast<std::size_t>(target) * n + beta] +=
            local_scale[beta] * value;
    }
}

void apply_static_m2l_plan(const FloatStaticM2LPlan &plan, const int level,
    const std::span<const float> multipoles,
                           const std::span<float> locals) {
    const int n = plan.coefficient_count;
    const int target_begin =
        plan.level_target_begin[static_cast<std::size_t>(level)];
  const int target_end = plan.level_target_end[static_cast<std::size_t>(level)];
  const float *multipole_scale =
      plan.multipole_scaling.data() + static_cast<std::size_t>(level) * n;
  const float *local_scale =
      plan.local_scaling.data() + static_cast<std::size_t>(level) * n;
    const std::ptrdiff_t output_count =
        static_cast<std::ptrdiff_t>(target_end - target_begin) * n;
#pragma omp parallel for schedule(static) if (output_count >= 256)
    for (std::ptrdiff_t output = 0; output < output_count; ++output) {
        const int target = target_begin + static_cast<int>(output / n);
        const int beta = static_cast<int>(output % n);
        float value = 0.0F;
        const int row_begin = plan.target_row_offsets[target];
        const int row_end = plan.target_row_offsets[target + 1];
        for (int interaction = row_begin; interaction < row_end;
             ++interaction) {
            const int source = plan.source_nodes[interaction];
            const int matrix_id = plan.matrix_ids[interaction];
            const float* matrix_column = plan.matrices.data() +
                static_cast<std::size_t>(matrix_id) * n * n + beta;
            const float* source_M = multipoles.data() +
                static_cast<std::size_t>(source) * n;
            for (int alpha = 0; alpha < n; ++alpha) {
                value += matrix_column[static_cast<std::size_t>(alpha) * n] *
                    multipole_scale[alpha] * source_M[alpha];
            }
        }
        locals[static_cast<std::size_t>(target) * n + beta] +=
            local_scale[beta] * value;
    }
}

void apply_static_m2l_plan(
    const StaticM2LPlan &plan, const int level,
    const std::span<const std::vector<double>> multipoles,
    const std::span<std::vector<double>> locals) {
    const int n = plan.coefficient_count;
    const std::ptrdiff_t output_count =
        static_cast<std::ptrdiff_t>(locals.size()) * n;
#pragma omp parallel for schedule(static) if (output_count >= 256)
    for (std::ptrdiff_t output = 0; output < output_count; ++output) {
        const int target = static_cast<int>(output / n);
        const int beta = static_cast<int>(output % n);
        double value = 0.0;
        for (int interaction = plan.target_row_offsets[target];
         interaction < plan.target_row_offsets[target + 1]; ++interaction) {
            if (plan.interaction_levels[interaction] != level) {
                continue;
            }
            const int source = plan.source_nodes[interaction];
            const int matrix_id = plan.matrix_ids[interaction];
            const double local_scale =
                plan.local_scaling[static_cast<std::size_t>(level) * n + beta];
            for (int alpha = 0; alpha < n; ++alpha) {
                const std::size_t matrix_index =
                    (static_cast<std::size_t>(matrix_id) * n + alpha) * n + beta;
                value += local_scale * plan.matrices[matrix_index] *
                 plan.multipole_scaling[static_cast<std::size_t>(level) * n +
                                        alpha] *
                    multipoles[source][alpha];
            }
        }
        locals[target][beta] += value;
    }
}

namespace {

double odd_double_factorial(const int l) {
    double result = 1.0;
    for (int value = 1; value <= 2 * l - 1; value += 2) {
        result *= static_cast<double>(value);
    }
    return result;
}

struct SphericalCartesianMaps {
    int cartesian_count{0};
    int spherical_count{0};
    std::vector<double> multipole_embedding{};
    std::vector<double> multipole_projection{};
    std::vector<double> local_embedding{};
    std::vector<double> local_projection{};
};

SphericalCartesianMaps
make_spherical_cartesian_maps(const SphericalHarmonicBasis &spherical,
                              const MultiIndexSet &cartesian) {
    SphericalCartesianMaps maps;
    maps.cartesian_count = cartesian.size();
    maps.spherical_count = spherical.size();
  const std::size_t values =
      static_cast<std::size_t>(cartesian.size()) * spherical.size();
    maps.multipole_embedding.assign(values, 0.0);
    maps.multipole_projection.assign(values, 0.0);
    maps.local_embedding.assign(values, 0.0);
    maps.local_projection.assign(values, 0.0);
    for (int mode = 0; mode < spherical.size(); ++mode) {
        const int l = spherical[mode].l;
        const double sign = l % 2 == 0 ? 1.0 : -1.0;
        const double double_factorial = odd_double_factorial(l);
        for (const SolidHarmonicTerm& term : spherical.polynomial(mode)) {
            const int alpha = cartesian.index(term.power);
            const double alpha_factorial =
                MultiIndexSet::multi_factorial(term.power);
            const std::size_t embedding_index =
                static_cast<std::size_t>(alpha) * spherical.size() + mode;
            const std::size_t projection_index =
                static_cast<std::size_t>(mode) * cartesian.size() + alpha;
            maps.multipole_embedding[embedding_index] =
                4.0 * std::numbers::pi * sign * term.coefficient /
                double_factorial;
            maps.multipole_projection[projection_index] =
                sign * term.coefficient * alpha_factorial /
                (4.0 * std::numbers::pi);
            maps.local_embedding[embedding_index] =
                alpha_factorial * term.coefficient;
            maps.local_projection[projection_index] =
                term.coefficient / double_factorial;
        }
    }
    return maps;
}

std::vector<double> compose_spherical_translation(
    const StaticCoefficientOperator& cartesian_operator,
    const std::span<const double> input_embedding,
    const std::span<const double> output_projection,
    const int cartesian_count,
    const int spherical_count)
{
    std::vector<double> intermediate(
        static_cast<std::size_t>(cartesian_count) * spherical_count, 0.0);
    for (const StaticOperatorEntry& entry : cartesian_operator.entries) {
        for (int input_mode = 0; input_mode < spherical_count; ++input_mode) {
            intermediate[static_cast<std::size_t>(entry.output) *
                             spherical_count + input_mode] +=
                entry.value *
                input_embedding[static_cast<std::size_t>(entry.input) *
                                    spherical_count + input_mode];
        }
    }
    std::vector<double> result(
        static_cast<std::size_t>(spherical_count) * spherical_count, 0.0);
    for (int output_mode = 0; output_mode < spherical_count; ++output_mode) {
        for (int cartesian = 0; cartesian < cartesian_count; ++cartesian) {
            const double projection =
                output_projection[static_cast<std::size_t>(output_mode) *
                                cartesian_count +
                            cartesian];
            if (projection == 0.0) {
                continue;
            }
      for (int input_mode = 0; input_mode < spherical_count; ++input_mode) {
                result[static_cast<std::size_t>(output_mode) +
                       static_cast<std::size_t>(spherical_count) * input_mode] +=
                    projection *
            intermediate[static_cast<std::size_t>(cartesian) * spherical_count +
                         input_mode];
            }
        }
    }
    return result;
}

StaticCoefficientOperator
pack_spherical_translation(const SphericalHarmonicBasis &basis,
    const std::span<const double> matrix,
                           const bool multipole_translation) {
    StaticCoefficientOperator result;
    result.input_size = basis.size();
    result.output_size = basis.size();
    for (int input = 0; input < basis.size(); ++input) {
        for (int output = 0; output < basis.size(); ++output) {
            const bool structurally_valid = multipole_translation
                ? basis[output].l >= basis[input].l
                : basis[output].l <= basis[input].l;
            if (!structurally_valid) {
                continue;
            }
            result.entries.push_back(
                {output, input,
                 matrix[static_cast<std::size_t>(output) +
                        static_cast<std::size_t>(basis.size()) * input]});
        }
    }
    return result;
}

void validate_operator_dimensions(const StaticCoefficientOperator &operator_map,
    const std::span<const double> input,
                                  const std::span<double> output) {
    if (input.size() != static_cast<std::size_t>(operator_map.input_size) ||
        output.size() != static_cast<std::size_t>(operator_map.output_size)) {
        throw std::invalid_argument("static operator dimensions are inconsistent");
    }
}

} // namespace

//------------------------------------------------------------------------------
// Canonical static mathematical operators
//------------------------------------------------------------------------------

std::vector<double> build_static_m2l_matrix(const MultiIndexSet &basis,
                                            const Vec3 &R) {
    const int coefficient_count = basis.size();
    const MultiIndexSet derivative_basis(2 * basis.order());
  const CoeffVector derivatives = laplace_derivatives_raw(derivative_basis, R);
  std::vector<double> matrix(static_cast<std::size_t>(coefficient_count) *
                             coefficient_count);
    for (int alpha_index = 0; alpha_index < coefficient_count; ++alpha_index) {
        for (int beta_index = 0; beta_index < coefficient_count; ++beta_index) {
      const MultiIndex gamma = add(basis[alpha_index], basis[beta_index]);
            matrix[static_cast<std::size_t>(beta_index) +
                   static_cast<std::size_t>(coefficient_count) * alpha_index] =
          derivatives[static_cast<std::size_t>(derivative_basis.index(gamma))];
        }
    }
    return matrix;
}

std::vector<double> build_static_m2l_matrix(const SphericalHarmonicBasis &basis,
                                            const Vec3 &R) {
    const MultiIndexSet derivatives_basis(2 * basis.order());
  const CoeffVector derivatives = laplace_derivatives_raw(derivatives_basis, R);
    std::vector<double> matrix(
        static_cast<std::size_t>(basis.size()) * basis.size(), 0.0);
    for (int input = 0; input < basis.size(); ++input) {
        const int input_degree = basis[input].l;
    const double input_factor = 4.0 * std::numbers::pi *
            (input_degree % 2 == 0 ? 1.0 : -1.0) /
            odd_double_factorial(input_degree);
        for (int output = 0; output < basis.size(); ++output) {
      const double factor =
          input_factor / odd_double_factorial(basis[output].l);
            double value = 0.0;
      for (const SolidHarmonicTerm &source_term : basis.polynomial(input)) {
        for (const SolidHarmonicTerm &target_term : basis.polynomial(output)) {
                    const MultiIndex derivative =
                        add(source_term.power, target_term.power);
          value += source_term.coefficient * target_term.coefficient *
                             derivatives[static_cast<std::size_t>(
                                 derivatives_basis.index(derivative))];
                }
            }
            matrix[static_cast<std::size_t>(output) +
             static_cast<std::size_t>(basis.size()) * input] = factor * value;
        }
    }
    return matrix;
}

std::vector<double> build_static_periodic_m2l_matrix(
    const MultiIndexSet& basis,
    const PeriodicCellOptions& options)
{
    PeriodicCellOptions normalised = options;
    normalised.centre = {};
    normalised.lengths = {1.0, 1.0, 1.0};
    const MultiIndexSet derivative_basis(2 * basis.order());
    const std::vector<double> derivatives =
        periodic_laplace_derivatives_raw(derivative_basis, normalised);
    const int coefficient_count = basis.size();
    std::vector<double> matrix(
        static_cast<std::size_t>(coefficient_count) * coefficient_count, 0.0);
    for (int alpha_index = 0; alpha_index < coefficient_count;
         ++alpha_index) {
        for (int beta_index = 0; beta_index < coefficient_count;
             ++beta_index) {
            const MultiIndex gamma =
                add(basis[alpha_index], basis[beta_index]);
            matrix[static_cast<std::size_t>(beta_index) +
                   static_cast<std::size_t>(coefficient_count) * alpha_index] =
                derivatives[static_cast<std::size_t>(
                    derivative_basis.index(gamma))];
        }
    }
    return matrix;
}

std::vector<double> build_static_periodic_m2l_matrix(
    const SphericalHarmonicBasis& basis,
    const PeriodicCellOptions& options)
{
    PeriodicCellOptions normalised = options;
    normalised.centre = {};
    normalised.lengths = {1.0, 1.0, 1.0};
    const MultiIndexSet derivative_basis(2 * basis.order());
    const std::vector<double> derivatives =
        periodic_laplace_derivatives_raw(derivative_basis, normalised);
    std::vector<double> matrix(
        static_cast<std::size_t>(basis.size()) * basis.size(), 0.0);
    for (int input = 0; input < basis.size(); ++input) {
        const int input_degree = basis[input].l;
        const double input_factor =
            4.0 * std::numbers::pi *
            (input_degree % 2 == 0 ? 1.0 : -1.0) /
            odd_double_factorial(input_degree);
        for (int output = 0; output < basis.size(); ++output) {
            const double factor =
                input_factor / odd_double_factorial(basis[output].l);
            double value = 0.0;
            for (const SolidHarmonicTerm& source_term :
                 basis.polynomial(input)) {
                for (const SolidHarmonicTerm& target_term :
                     basis.polynomial(output)) {
                    const MultiIndex derivative =
                        add(source_term.power, target_term.power);
                    value += source_term.coefficient * target_term.coefficient *
                        derivatives[static_cast<std::size_t>(
                            derivative_basis.index(derivative))];
                }
            }
            matrix[static_cast<std::size_t>(output) +
                   static_cast<std::size_t>(basis.size()) * input] =
                factor * value;
        }
    }
    return matrix;
}

StaticCoefficientOperator
build_static_p2m_operator(const MultiIndexSet &basis, const Vec3 &centre,
                          const std::span<const Vec3> source_positions) {
    StaticCoefficientOperator result;
    result.input_size = static_cast<int>(3 * source_positions.size());
    result.output_size = basis.size();
    for (std::size_t source = 0; source < source_positions.size(); ++source) {
        const Vec3 dx = source_positions[source] - centre;
        for (int alpha_index = 0; alpha_index < basis.size(); ++alpha_index) {
            const MultiIndex alpha = basis[alpha_index];
            const double sign = alpha.degree() % 2 == 0 ? 1.0 : -1.0;
            const MultiIndex shifted[3] = {
                {alpha.ax - 1, alpha.ay, alpha.az},
                {alpha.ax, alpha.ay - 1, alpha.az},
                {alpha.ax, alpha.ay, alpha.az - 1}
            };
            const int components[3] = {alpha.ax, alpha.ay, alpha.az};
            for (int component = 0; component < 3; ++component) {
                if (components[component] == 0) {
                    continue;
                }
                result.entries.push_back({
                    alpha_index,
                    static_cast<int>(3 * source) + component,
                    sign * MultiIndexSet::monomial_over_factorial(
                        dx, shifted[component]
                    )
                });
            }
        }
    }
    return result;
}

StaticCoefficientOperator build_static_p2m_operator(
    const SphericalHarmonicBasis& basis,
    const Vec3& centre,
    const std::span<const Vec3> source_positions)
{
    StaticCoefficientOperator result;
    result.input_size = static_cast<int>(3 * source_positions.size());
    result.output_size = basis.size();
    const double green_factor = 1.0 / (4.0 * std::numbers::pi);
    for (std::size_t source = 0; source < source_positions.size(); ++source) {
        const SolidHarmonicValues regular = regular_solid_harmonics(
            basis, source_positions[source] - centre);
        for (int mode = 0; mode < basis.size(); ++mode) {
            const Vec3 gradient =
                regular.gradients[static_cast<std::size_t>(mode)];
            for (int component = 0; component < 3; ++component) {
                const double value = green_factor * gradient[component];
                if (value != 0.0) {
                    result.entries.push_back(
                        {mode, static_cast<int>(3 * source) + component, value});
                }
            }
        }
    }
    return result;
}

StaticCoefficientOperator build_static_cuboid_p2m_operator(
    const MultiIndexSet &basis, const Vec3 &centre,
    const std::span<const Vec3> source_positions,
    const std::span<const CuboidSize> source_sizes) {
  if (source_sizes.size() != 1 &&
      source_sizes.size() != source_positions.size()) {
    throw std::invalid_argument(
        "cuboid P2M sizes must be common or per source");
    }
    StaticCoefficientOperator result;
    result.input_size = static_cast<int>(3 * source_positions.size());
    result.output_size = basis.size();
    for (std::size_t source = 0; source < source_positions.size(); ++source) {
        const Vec3 d = source_positions[source] - centre;
        const CuboidSize h = source_sizes[source_sizes.size() == 1 ? 0 : source];
        for (int alpha_index = 0; alpha_index < basis.size(); ++alpha_index) {
            const MultiIndex alpha = basis[alpha_index];
            const double sign = alpha.degree() % 2 == 0 ? 1.0 : -1.0;
            const MultiIndex shifted[3] = {{alpha.ax - 1, alpha.ay, alpha.az},
                                           {alpha.ax, alpha.ay - 1, alpha.az},
                                           {alpha.ax, alpha.ay, alpha.az - 1}};
            const int components[3] = {alpha.ax, alpha.ay, alpha.az};
            for (int component = 0; component < 3; ++component) {
                if (components[component] > 0) {
          result.entries.push_back(
              {alpha_index, static_cast<int>(3 * source) + component,
               sign * cuboid_averaged_monomial(shifted[component], d, h)});
                }
            }
        }
    }
    return result;
}

StaticCoefficientOperator build_static_cuboid_p2m_operator(
    const SphericalHarmonicBasis& basis, const Vec3& centre,
    const std::span<const Vec3> source_positions,
    const std::span<const CuboidSize> source_sizes)
{
    if (source_sizes.size() != 1 &&
        source_sizes.size() != source_positions.size()) {
        throw std::invalid_argument(
            "spherical cuboid P2M sizes must be common or per source");
    }

    StaticCoefficientOperator result;
    result.input_size = static_cast<int>(3 * source_positions.size());
    result.output_size = basis.size();
    const double green_factor = 1.0 / (4.0 * std::numbers::pi);
    for (std::size_t source = 0; source < source_positions.size(); ++source) {
        const Vec3 d = source_positions[source] - centre;
        const CuboidSize size =
            source_sizes[source_sizes.size() == 1 ? 0 : source];
        for (int mode = 0; mode < basis.size(); ++mode) {
            double averaged_gradient[3]{0.0, 0.0, 0.0};
            for (const SolidHarmonicTerm& term : basis.polynomial(mode)) {
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
                    averaged_gradient[component] +=
                        term.coefficient * powers[component] *
                        MultiIndexSet::multi_factorial(derivative) *
                        cuboid_averaged_monomial(derivative, d, size);
                }
            }
            for (int component = 0; component < 3; ++component) {
                const double value =
                    green_factor * averaged_gradient[component];
                if (value != 0.0) {
                    result.entries.push_back(
                        {mode, static_cast<int>(3 * source) + component,
                         value});
                }
            }
        }
    }
    return result;
}

StaticCoefficientOperator build_static_m2m_operator(const MultiIndexSet &basis,
                                                    const Vec3 &d) {
    StaticCoefficientOperator result{basis.size(), basis.size(), {}};
    for (int alpha_index = 0; alpha_index < basis.size(); ++alpha_index) {
        const MultiIndex alpha = basis[alpha_index];
        for (int gamma_index = 0; gamma_index < basis.size(); ++gamma_index) {
            const MultiIndex gamma = basis[gamma_index];
            if (leq(gamma, alpha)) {
        result.entries.push_back(
            {alpha_index, basis.index(sub(alpha, gamma)),
             MultiIndexSet::monomial_over_factorial(d, gamma)});
            }
        }
    }
    return result;
}

StaticCoefficientOperator
build_static_m2m_operator(const SphericalHarmonicBasis &basis, const Vec3 &d) {
    const MultiIndexSet cartesian(basis.order());
    const SphericalCartesianMaps maps =
        make_spherical_cartesian_maps(basis, cartesian);
    const StaticCoefficientOperator cartesian_operator =
        build_static_m2m_operator(cartesian, d);
    const std::vector<double> matrix = compose_spherical_translation(
      cartesian_operator, maps.multipole_embedding, maps.multipole_projection,
      maps.cartesian_count, maps.spherical_count);
    return pack_spherical_translation(basis, matrix, true);
}

StaticCoefficientOperator build_static_l2l_operator(const MultiIndexSet &basis,
                                                    const Vec3 &d) {
    StaticCoefficientOperator result{basis.size(), basis.size(), {}};
    for (int beta_index = 0; beta_index < basis.size(); ++beta_index) {
        const MultiIndex beta = basis[beta_index];
        for (int gamma_index = 0; gamma_index < basis.size(); ++gamma_index) {
            const MultiIndex gamma = basis[gamma_index];
            const MultiIndex sum = add(beta, gamma);
            if (sum.degree() <= basis.order()) {
        result.entries.push_back(
            {beta_index, basis.index(sum),
             MultiIndexSet::monomial_over_factorial(d, gamma)});
            }
        }
    }
    return result;
}

StaticCoefficientOperator
build_static_l2l_operator(const SphericalHarmonicBasis &basis, const Vec3 &d) {
    const MultiIndexSet cartesian(basis.order());
    const SphericalCartesianMaps maps =
        make_spherical_cartesian_maps(basis, cartesian);
    const StaticCoefficientOperator cartesian_operator =
        build_static_l2l_operator(cartesian, d);
    const std::vector<double> matrix = compose_spherical_translation(
        cartesian_operator, maps.local_embedding, maps.local_projection,
        maps.cartesian_count, maps.spherical_count);
    return pack_spherical_translation(basis, matrix, false);
}

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
    const Vec3 dx = target - centre;
    for (int beta_index = 0; beta_index < basis.size(); ++beta_index) {
        const MultiIndex beta = basis[beta_index];
        result.potential[static_cast<std::size_t>(beta_index)] =
            MultiIndexSet::monomial_over_factorial(dx, beta);
        if (beta.ax > 0) {
            result.field[0][static_cast<std::size_t>(beta_index)] =
                -MultiIndexSet::monomial_over_factorial(
              dx, {beta.ax - 1, beta.ay, beta.az});
        }
        if (beta.ay > 0) {
            result.field[1][static_cast<std::size_t>(beta_index)] =
                -MultiIndexSet::monomial_over_factorial(
              dx, {beta.ax, beta.ay - 1, beta.az});
        }
        if (beta.az > 0) {
            result.field[2][static_cast<std::size_t>(beta_index)] =
                -MultiIndexSet::monomial_over_factorial(
              dx, {beta.ax, beta.ay, beta.az - 1});
        }
    }
    return result;
}

StaticL2PEvaluator
build_static_l2p_evaluator(const SphericalHarmonicBasis &basis,
                           const Vec3 &centre, const Vec3 &target) {
    const SolidHarmonicValues regular =
        regular_solid_harmonics(basis, target - centre);
    StaticL2PEvaluator result;
    result.potential = regular.values;
    for (auto& row : result.field) {
        row.resize(static_cast<std::size_t>(basis.size()));
    }
    for (int mode = 0; mode < basis.size(); ++mode) {
        const Vec3 gradient = regular.gradients[static_cast<std::size_t>(mode)];
        result.field[0][static_cast<std::size_t>(mode)] = -gradient.x;
        result.field[1][static_cast<std::size_t>(mode)] = -gradient.y;
        result.field[2][static_cast<std::size_t>(mode)] = -gradient.z;
    }
    return result;
}

StaticL2PEvaluator build_static_cuboid_l2p_evaluator(
    const MultiIndexSet& basis, const Vec3& centre, const Vec3& target,
    const CuboidSize& target_size)
{
    StaticL2PEvaluator result;
    result.potential.resize(static_cast<std::size_t>(basis.size()));
    for (auto& row : result.field) {
        row.resize(static_cast<std::size_t>(basis.size()));
    }
    const Vec3 dx = target - centre;
    for (int beta_index = 0; beta_index < basis.size(); ++beta_index) {
        const MultiIndex beta = basis[beta_index];
    result.potential[beta_index] =
        cuboid_averaged_monomial(beta, dx, target_size);
        if (beta.ax > 0) {
            result.field[0][beta_index] = -cuboid_averaged_monomial(
                {beta.ax - 1, beta.ay, beta.az}, dx, target_size);
        }
        if (beta.ay > 0) {
            result.field[1][beta_index] = -cuboid_averaged_monomial(
                {beta.ax, beta.ay - 1, beta.az}, dx, target_size);
        }
        if (beta.az > 0) {
            result.field[2][beta_index] = -cuboid_averaged_monomial(
                {beta.ax, beta.ay, beta.az - 1}, dx, target_size);
        }
    }
    return result;
}

StaticL2PEvaluator build_static_cuboid_l2p_evaluator(
    const SphericalHarmonicBasis& basis, const Vec3& centre,
    const Vec3& target, const CuboidSize& target_size)
{
    StaticL2PEvaluator result;
    result.potential.resize(static_cast<std::size_t>(basis.size()));
    for (auto& row : result.field) {
        row.resize(static_cast<std::size_t>(basis.size()));
    }
    const Vec3 dx = target - centre;
    for (int mode = 0; mode < basis.size(); ++mode) {
        for (const SolidHarmonicTerm& term : basis.polynomial(mode)) {
            result.potential[static_cast<std::size_t>(mode)] +=
                term.coefficient *
                MultiIndexSet::multi_factorial(term.power) *
                cuboid_averaged_monomial(term.power, dx, target_size);
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
                    cuboid_averaged_monomial(derivative, dx, target_size);
            }
        }
    }
    return result;
}

std::size_t StaticP2POperator::memory_bytes() const noexcept {
    return row_offsets.size() * sizeof(int) +
        blocks.size() * sizeof(StaticDipoleBlock);
}

std::size_t StaticP2PMemory::total_bytes() const noexcept {
  return tensor_bytes + index_bytes + row_metadata_bytes + leaf_metadata_bytes +
         scratch_bytes;
}

StaticP2PMemory StaticP2PCompactPlan::memory() const noexcept {
  StaticP2PMemory result;
  result.tensor_bytes = tensors[0].size() * 9 * sizeof(double);
  result.index_bytes = source_indices.size() * sizeof(int) +
      skip_for_identity.size() * sizeof(unsigned char);
  result.row_metadata_bytes = row_offsets.size() * sizeof(int);
  return result;
}

StaticP2PMemory StaticP2PLeafPlan::memory() const noexcept {
  StaticP2PMemory result;
  result.tensor_bytes = tensors[0].size() * 6 * sizeof(double);
  result.row_metadata_bytes = leaf_row_offsets.size() * sizeof(int);
  result.leaf_metadata_bytes =
      (target_begins.size() + target_counts.size()) * sizeof(int) +
      blocks.size() * sizeof(StaticP2PLeafBlock);
  return result;
}

StaticP2PMemory StaticP2PTensorDictionaryPlan::memory() const noexcept {
  StaticP2PMemory result;
  result.tensor_bytes = tensors[0].size() * 6 * sizeof(double);
  result.index_bytes = tokens.size() * sizeof(std::uint32_t);
  result.row_metadata_bytes = leaf_row_offsets.size() * sizeof(int);
  result.leaf_metadata_bytes =
      (target_begins.size() + target_counts.size()) * sizeof(int) +
      blocks.size() * sizeof(StaticP2PLeafBlock);
  return result;
}

StaticP2PMemory StaticP2PSignedTensorDictionaryPlan::memory() const noexcept {
  StaticP2PMemory result;
  result.tensor_bytes = tensors[0].size() * 6 * sizeof(double);
  result.index_bytes = tokens8.size() * sizeof(std::uint8_t) +
      tokens16.size() * sizeof(std::uint16_t) +
      tokens32.size() * sizeof(std::uint32_t);
  result.row_metadata_bytes = leaf_row_offsets.size() * sizeof(int);
  result.leaf_metadata_bytes =
      (target_begins.size() + target_counts.size() + tile_leaf_indices.size() +
       tile_target_offsets.size()) * sizeof(int) +
      blocks.size() * sizeof(StaticP2PLeafBlock);
  return result;
}

StaticP2PMemory StaticP2PBsrPlan::memory() const noexcept {
  StaticP2PMemory result;
  result.tensor_bytes = values.size() * sizeof(double);
  result.index_bytes = source_indices.size() * sizeof(int);
  result.row_metadata_bytes =
      (row_offsets.size() + target_source_indices.size()) * sizeof(int);
  return result;
}

StaticP2POperator build_static_p2p_operator(
    const std::span<const Vec3> target_positions,
    const std::span<const Vec3> source_positions,
    const std::span<const std::array<int, 2>> interactions,
    const SourceGeometry source_geometry,
    const std::span<const CuboidSize> source_sizes,
    const TargetGeometry target_geometry,
    const std::span<const CuboidSize> target_sizes)
{
    std::vector<StaticP2PInteraction> image_interactions;
    image_interactions.reserve(interactions.size());
    for (const std::array<int, 2> pair : interactions) {
        image_interactions.push_back({pair[0], pair[1], {}, true});
    }
    return build_static_p2p_operator(
        target_positions, source_positions, image_interactions,
        source_geometry, source_sizes, target_geometry, target_sizes);
}

StaticP2POperator build_static_p2p_operator(
    const std::span<const Vec3> target_positions,
    const std::span<const Vec3> source_positions,
    const std::span<const StaticP2PInteraction> interactions,
    const SourceGeometry source_geometry,
    const std::span<const CuboidSize> source_sizes,
    const TargetGeometry target_geometry,
    const std::span<const CuboidSize> target_sizes)
{
    if (source_geometry == SourceGeometry::UniformCuboid &&
        source_sizes.size() != 1 &&
        source_sizes.size() != source_positions.size()) {
        throw std::invalid_argument(
            "static cuboid P2P sizes must be common or per source");
    }
    if (target_geometry == TargetGeometry::VolumeAveragedCuboid &&
        target_sizes.size() != 1 &&
        target_sizes.size() != target_positions.size()) {
        throw std::invalid_argument(
            "static cuboid P2P target sizes must be common or per target");
    }
    if (source_geometry == SourceGeometry::UniformCuboid) {
        for (const CuboidSize& size : source_sizes) {
            if (!(size.hx > 0.0 && size.hy > 0.0 && size.hz > 0.0)) {
                throw std::invalid_argument(
                    "source cuboid dimensions must be positive");
            }
        }
    }
    if (target_geometry == TargetGeometry::VolumeAveragedCuboid) {
        for (const CuboidSize& size : target_sizes) {
            if (!(size.hx > 0.0 && size.hy > 0.0 && size.hz > 0.0)) {
                throw std::invalid_argument(
                    "target cuboid dimensions must be positive");
            }
        }
    }
    StaticP2POperator result;
    result.source_count = static_cast<int>(source_positions.size());
    result.target_count = static_cast<int>(target_positions.size());
    result.row_offsets.assign(target_positions.size() + 1, 0);

    std::vector<StaticP2PInteraction> sorted(
        interactions.begin(), interactions.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& left,
                                                const auto& right) {
        return std::tie(left.target, left.source, left.source_shift.x,
                        left.source_shift.y, left.source_shift.z) <
            std::tie(right.target, right.source, right.source_shift.x,
                     right.source_shift.y, right.source_shift.z);
    });

    // Validate indices and construct the target-row metadata before entering
    // the parallel region. This keeps exception handling outside OpenMP and
    // makes every subsequent iteration own exactly one output block.
    for (const StaticP2PInteraction& interaction : sorted) {
        const int target = interaction.target;
        const int source = interaction.source;
        if (target < 0 || target >= result.target_count || source < 0 ||
            source >= result.source_count) {
            throw std::invalid_argument(
                "static P2P interaction index is invalid");
        }
        ++result.row_offsets[static_cast<std::size_t>(target) + 1];
    }
    for (std::size_t row = 1; row < result.row_offsets.size(); ++row) {
        result.row_offsets[row] += result.row_offsets[row - 1];
    }

    result.blocks.resize(sorted.size());
    const std::ptrdiff_t interaction_count =
        static_cast<std::ptrdiff_t>(sorted.size());
#pragma omp parallel for schedule(static) if (interaction_count >= 256)
    for (std::ptrdiff_t index = 0; index < interaction_count; ++index) {
        const StaticP2PInteraction& interaction =
            sorted[static_cast<std::size_t>(index)];
        const int target = interaction.target;
        const int source = interaction.source;
        const Vec3 shifted_source =
            source_positions[static_cast<std::size_t>(source)] +
            interaction.source_shift;
        const Vec3 r = target_positions[static_cast<std::size_t>(target)] -
            shifted_source;
        if (source_geometry == SourceGeometry::PointDipole &&
            target_geometry == TargetGeometry::Point && dot(r, r) == 0.0) {
            // Preserve the block so an explicit identity map can skip it. If
            // it is not a self pair, NaNs deliberately expose the same
            // undefined point-dipole singularity as the reference operator.
            const double undefined = std::numeric_limits<double>::quiet_NaN();
            result.blocks[static_cast<std::size_t>(index)] =
                {target, source, undefined, undefined, undefined,
                 undefined, undefined, undefined, undefined, undefined,
                 undefined, interaction.skip_for_identity ? 1 : 0};
            continue;
        }
        const CuboidSize source_size =
            source_geometry == SourceGeometry::UniformCuboid
                ? source_sizes[source_sizes.size() == 1 ? 0 : source]
                : CuboidSize{};
        const PairTensor tensor = build_pair_tensor(
            target_positions[target], shifted_source, source_geometry,
            target_geometry, source_size,
            target_geometry == TargetGeometry::VolumeAveragedCuboid
                ? target_sizes[target_sizes.size() == 1 ? 0 : target]
                : CuboidSize{});
        const double radius_squared = dot(r, r);
        const double potential_scale =
            radius_squared == 0.0
                ? 0.0
                : 1.0 / (4.0 * std::numbers::pi * radius_squared *
                         std::sqrt(radius_squared));
        result.blocks[static_cast<std::size_t>(index)] =
            {target, source, potential_scale * r.x, potential_scale * r.y,
             potential_scale * r.z, tensor.xx, tensor.xy, tensor.xz,
             tensor.yy, tensor.yz, tensor.zz,
             interaction.skip_for_identity ? 1 : 0};
    }
    return result;
}

StaticP2PCompactPlan
build_static_p2p_compact_plan(const StaticP2POperator &operator_map) {
  StaticP2PCompactPlan result;
  result.source_count = operator_map.source_count;
  result.target_count = operator_map.target_count;
  result.row_offsets = operator_map.row_offsets;
  result.source_indices.reserve(operator_map.blocks.size());
  result.skip_for_identity.reserve(operator_map.blocks.size());
  for (auto &coefficient : result.potential) {
    coefficient.reserve(operator_map.blocks.size());
  }
  for (auto &tensor : result.tensors) {
    tensor.reserve(operator_map.blocks.size());
  }

  for (const StaticDipoleBlock &block : operator_map.blocks) {
    result.source_indices.push_back(block.source);
    result.skip_for_identity.push_back(
        static_cast<unsigned char>(block.skip_for_identity != 0));
    result.potential[0].push_back(block.px);
    result.potential[1].push_back(block.py);
    result.potential[2].push_back(block.pz);
    result.tensors[0].push_back(block.xx);
    result.tensors[1].push_back(block.xy);
    result.tensors[2].push_back(block.xz);
    result.tensors[3].push_back(block.yy);
    result.tensors[4].push_back(block.yz);
    result.tensors[5].push_back(block.zz);
  }
  return result;
}

FloatStaticP2PCompactPlan
build_static_p2p_compact_plan(const FloatStaticP2POperator &operator_map) {
  FloatStaticP2PCompactPlan result;
  result.source_count = operator_map.source_count;
  result.target_count = operator_map.target_count;
  result.row_offsets = operator_map.row_offsets;
  result.source_indices.reserve(operator_map.blocks.size());
  result.skip_for_identity.reserve(operator_map.blocks.size());
  for (auto &coefficient : result.potential) {
    coefficient.reserve(operator_map.blocks.size());
  }
  for (auto &tensor : result.tensors) {
    tensor.reserve(operator_map.blocks.size());
  }

  for (const FloatStaticDipoleBlock &block : operator_map.blocks) {
    result.source_indices.push_back(block.source);
    result.skip_for_identity.push_back(
        static_cast<unsigned char>(block.skip_for_identity != 0));
    result.potential[0].push_back(block.px);
    result.potential[1].push_back(block.py);
    result.potential[2].push_back(block.pz);
    result.tensors[0].push_back(block.xx);
    result.tensors[1].push_back(block.xy);
    result.tensors[2].push_back(block.xz);
    result.tensors[3].push_back(block.yy);
    result.tensors[4].push_back(block.yz);
    result.tensors[5].push_back(block.zz);
  }
  return result;
}

StaticP2PLeafPlan build_static_p2p_leaf_plan(
    const StaticP2POperator &operator_map,
    const std::span<const StaticP2PLeafPair> leaf_pairs) {
  StaticP2PLeafPlan result;
  result.source_count = operator_map.source_count;
  result.target_count = operator_map.target_count;
  if (leaf_pairs.empty()) {
    result.leaf_row_offsets.push_back(0);
    return result;
  }

  std::vector<unsigned char> covered(operator_map.blocks.size(), 0);
  std::set<std::pair<int, int>> occupied_target_ranges;
  std::set<std::pair<int, int>> occupied_source_ranges;
  int previous_target_begin = -1;
  int previous_target_count = -1;
  // The canonical operator retains `list1` neighbour order in each target
  // row. Preserve the supplied leaf-pair order so dense rectangles validate
  // against that exact interaction order rather than a reordered topology.
  for (const StaticP2PLeafPair &pair : leaf_pairs) {
    if (pair.target_begin < 0 || pair.target_count <= 0 ||
        pair.target_begin + pair.target_count > operator_map.target_count ||
        pair.source_begin < 0 || pair.source_count <= 0 ||
        pair.source_begin + pair.source_count > operator_map.source_count) {
      throw std::invalid_argument("static P2P leaf range is invalid");
    }
    if (pair.target_begin != previous_target_begin) {
      if (previous_target_begin >= 0) {
        result.leaf_row_offsets.push_back(
            static_cast<int>(result.blocks.size()));
      } else {
        result.leaf_row_offsets.push_back(0);
      }
      result.target_begins.push_back(pair.target_begin);
      result.target_counts.push_back(pair.target_count);
      previous_target_begin = pair.target_begin;
      previous_target_count = pair.target_count;
    } else if (pair.target_count != previous_target_count) {
      throw std::invalid_argument("inconsistent target leaf range");
    }

    StaticP2PLeafBlock leaf_block;
    leaf_block.source_begin = pair.source_begin;
    leaf_block.source_count = pair.source_count;
    leaf_block.tensor_offset = result.tensors[0].size();
    result.blocks.push_back(leaf_block);
    occupied_target_ranges.emplace(pair.target_begin, pair.target_count);
    occupied_source_ranges.emplace(pair.source_begin, pair.source_count);

    for (int target = pair.target_begin;
         target < pair.target_begin + pair.target_count; ++target) {
      int expected_source = pair.source_begin;
      for (int entry =
               operator_map.row_offsets[static_cast<std::size_t>(target)];
           entry <
           operator_map.row_offsets[static_cast<std::size_t>(target) + 1];
           ++entry) {
        const StaticDipoleBlock &block =
            operator_map.blocks[static_cast<std::size_t>(entry)];
        if (block.source < pair.source_begin ||
            block.source >= pair.source_begin + pair.source_count) {
          continue;
        }
        if (block.source != expected_source || covered[entry] != 0) {
          throw std::invalid_argument(
              "static P2P leaf pairs are not dense and unique");
        }
        covered[entry] = 1;
        ++expected_source;
        result.tensors[0].push_back(block.xx);
        result.tensors[1].push_back(block.xy);
        result.tensors[2].push_back(block.xz);
        result.tensors[3].push_back(block.yy);
        result.tensors[4].push_back(block.yz);
        result.tensors[5].push_back(block.zz);
      }
      if (expected_source != pair.source_begin + pair.source_count) {
        throw std::invalid_argument(
            "static P2P leaf pair omits canonical interactions");
      }
    }
  }
  result.leaf_row_offsets.push_back(static_cast<int>(result.blocks.size()));
  if (std::find(covered.begin(), covered.end(), 0) != covered.end()) {
    throw std::invalid_argument(
        "static P2P leaf pairs do not cover the canonical operator");
  }

  std::set<int> occupancies;
  std::size_t occupancy_sum = 0;
  result.minimum_occupancy = std::numeric_limits<int>::max();
  const auto record_occupancies = [&](const auto &ranges) {
    for (const auto [begin, count] : ranges) {
      static_cast<void>(begin);
      result.minimum_occupancy = std::min(result.minimum_occupancy, count);
      result.maximum_occupancy = std::max(result.maximum_occupancy, count);
      occupancy_sum += static_cast<std::size_t>(count);
      occupancies.insert(count);
    }
  };
  record_occupancies(occupied_target_ranges);
  record_occupancies(occupied_source_ranges);
  const std::size_t occupied_range_count =
      occupied_target_ranges.size() + occupied_source_ranges.size();
  result.mean_occupancy = static_cast<double>(occupancy_sum) /
                          static_cast<double>(occupied_range_count);
  result.unique_occupancies = static_cast<int>(occupancies.size());
  result.uniform_occupancy = occupancies.size() == 1;
  return result;
}

namespace {

template <typename Scalar> struct Tensor6BitKeyHash {
  std::size_t operator()(const Tensor6BitKey<Scalar> &key) const noexcept {
    std::size_t hash = 0;
    for (const auto value : key.values) {
      hash ^= std::hash<typename Tensor6BitKey<Scalar>::Bits>{}(value) +
          0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
    }
    return hash;
  }
};

template <typename Scalar, typename DictionaryPlan, typename LeafPlan>
DictionaryPlan build_tensor_dictionary_from_leaf(const LeafPlan &leaf,
                                                 const bool skip_for_identity) {
  DictionaryPlan result;
  result.source_count = leaf.source_count;
  result.target_count = leaf.target_count;
  result.target_begins = leaf.target_begins;
  result.target_counts = leaf.target_counts;
  result.leaf_row_offsets = leaf.leaf_row_offsets;
  result.blocks = leaf.blocks;
  result.skip_for_identity = skip_for_identity;
  result.tokens.reserve(leaf.tensors[0].size());

  std::unordered_map<Tensor6BitKey<Scalar>, std::uint32_t,
                     Tensor6BitKeyHash<Scalar>> dictionary_ids;
  dictionary_ids.reserve(leaf.tensors[0].size());
  std::vector<std::size_t> frequencies;
  for (std::size_t interaction = 0; interaction < leaf.tensors[0].size();
       ++interaction) {
    const std::array<Scalar, 6> tensor{
        leaf.tensors[0][interaction], leaf.tensors[1][interaction],
        leaf.tensors[2][interaction], leaf.tensors[3][interaction],
        leaf.tensors[4][interaction], leaf.tensors[5][interaction]};
    const auto canonical = canonicalise_tensor6(tensor);
    const auto key = tensor6_bit_key(canonical.values);
    const auto [iterator, inserted] = dictionary_ids.try_emplace(
        key, static_cast<std::uint32_t>(dictionary_ids.size()));
    if (inserted) {
      if (iterator->second >= (1U << 26U)) {
        throw std::overflow_error("Tensor6 dictionary exceeds packed token IDs");
      }
      for (int component = 0; component < 6; ++component) {
        result.tensors[static_cast<std::size_t>(component)].push_back(
            canonical.values[static_cast<std::size_t>(component)]);
      }
      frequencies.push_back(0);
    }
    ++frequencies[iterator->second];
    result.tokens.push_back(pack_tensor6_token(iterator->second,
                                                canonical.sign_mask));
  }

  // Hot dictionary entries receive low IDs, improving cache locality without
  // changing the interaction order or floating-point accumulation order.
  std::vector<std::uint32_t> order(frequencies.size());
  std::iota(order.begin(), order.end(), 0U);
  std::stable_sort(order.begin(), order.end(), [&](const auto left,
                                                   const auto right) {
    return frequencies[left] > frequencies[right];
  });
  std::vector<std::uint32_t> remap(order.size());
  std::array<std::vector<Scalar>, 6> reordered;
  for (std::uint32_t new_id = 0; new_id < order.size(); ++new_id) {
    const std::uint32_t old_id = order[new_id];
    remap[old_id] = new_id;
    for (int component = 0; component < 6; ++component) {
      reordered[static_cast<std::size_t>(component)].push_back(
          result.tensors[static_cast<std::size_t>(component)][old_id]);
    }
  }
  result.tensors = std::move(reordered);
  for (std::uint32_t &token : result.tokens) {
    token = pack_tensor6_token(remap[tensor6_token_id(token)],
                               tensor6_token_sign_mask(token));
  }
  return result;
}

} // namespace

StaticP2PTensorDictionaryPlan build_static_p2p_tensor_dictionary_plan(
    const StaticP2POperator &operator_map,
    const std::span<const StaticP2PLeafPair> leaf_pairs) {
  const StaticP2PLeafPlan leaf =
      build_static_p2p_leaf_plan(operator_map, leaf_pairs);
  bool skip_for_identity = false;
  bool first = true;
  for (const StaticDipoleBlock &block : operator_map.blocks) {
    const bool current = block.skip_for_identity != 0;
    if (!first && current != skip_for_identity) {
      throw std::invalid_argument(
          "Tensor6 dictionary requires a uniform self-identity policy");
    }
    skip_for_identity = current;
    first = false;
  }
  return build_tensor_dictionary_from_leaf<double,
                                           StaticP2PTensorDictionaryPlan>(
      leaf, skip_for_identity);
}

StaticP2PSignedTensorDictionaryPlan
build_static_p2p_signed_tensor_dictionary_plan(
    const StaticP2POperator &operator_map,
    const std::span<const StaticP2PLeafPair> leaf_pairs,
    const std::span<const int> target_source_indices, const int target_tile_size) {
  if (target_tile_size <= 0 || target_tile_size > 128) {
    throw std::invalid_argument("signed Tensor6 target tile must be in [1, 128]");
  }
  if (!target_source_indices.empty() &&
      target_source_indices.size() != static_cast<std::size_t>(operator_map.target_count)) {
    throw std::invalid_argument("signed Tensor6 dictionary identity dimensions are inconsistent");
  }
  const StaticP2PLeafPlan leaf =
      build_static_p2p_leaf_plan(operator_map, leaf_pairs);
  StaticP2PSignedTensorDictionaryPlan result;
  result.source_count = leaf.source_count;
  result.target_count = leaf.target_count;
  result.target_begins = leaf.target_begins;
  result.target_counts = leaf.target_counts;
  result.leaf_row_offsets = leaf.leaf_row_offsets;
  result.blocks = leaf.blocks;
  result.target_tile_size = target_tile_size;

  // Variant zero is deliberately exact zero: fixed point self interactions
  // become ordinary dictionary lookups and need no execution-time branch.
  for (auto &component : result.tensors) {
    component.push_back(0.0);
  }
  std::unordered_map<Tensor6BitKey<double>, std::uint32_t,
                     Tensor6BitKeyHash<double>> variant_ids;
  variant_ids.reserve(leaf.tensors[0].size());
  variant_ids.emplace(tensor6_bit_key(std::array<double, 6>{}), 0U);
  std::vector<std::uint32_t> wide_tokens;
  wide_tokens.reserve(leaf.tensors[0].size());

  for (int target_leaf = 0;
       target_leaf < static_cast<int>(leaf.target_begins.size()); ++target_leaf) {
    const int target_begin = leaf.target_begins[static_cast<std::size_t>(target_leaf)];
    const int target_count = leaf.target_counts[static_cast<std::size_t>(target_leaf)];
    for (int local_begin = 0; local_begin < target_count;
         local_begin += target_tile_size) {
      result.tile_leaf_indices.push_back(target_leaf);
      result.tile_target_offsets.push_back(local_begin);
    }
    for (int block_index = leaf.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
         block_index < leaf.leaf_row_offsets[static_cast<std::size_t>(target_leaf) + 1];
         ++block_index) {
      StaticP2PLeafBlock &block = result.blocks[static_cast<std::size_t>(block_index)];
      const StaticP2PLeafBlock &source_block =
          leaf.blocks[static_cast<std::size_t>(block_index)];
      block.tensor_offset = wide_tokens.size();
      for (int local_source = 0; local_source < block.source_count; ++local_source) {
        const int source = block.source_begin + local_source;
        for (int local_target = 0; local_target < target_count; ++local_target) {
          const int target = target_begin + local_target;
          const bool is_self = !target_source_indices.empty() &&
              source == target_source_indices[static_cast<std::size_t>(target)];
          if (is_self) {
            wide_tokens.push_back(0U);
            continue;
          }
          const std::size_t index = source_block.tensor_offset +
              static_cast<std::size_t>(local_target) * block.source_count +
              static_cast<std::size_t>(local_source);
          std::array<double, 6> tensor{};
          for (int component = 0; component < 6; ++component) {
            const double value = leaf.tensors[static_cast<std::size_t>(component)][index];
            tensor[static_cast<std::size_t>(component)] = value == 0.0 ? 0.0 : value;
          }
          const auto key = tensor6_bit_key(tensor);
          const auto [iterator, inserted] = variant_ids.try_emplace(
              key, static_cast<std::uint32_t>(variant_ids.size()));
          if (inserted) {
            for (int component = 0; component < 6; ++component) {
              result.tensors[static_cast<std::size_t>(component)].push_back(
                  tensor[static_cast<std::size_t>(component)]);
            }
          }
          wide_tokens.push_back(iterator->second);
        }
      }
    }
  }
  if (result.variant_count() <= 255U) {
    result.token_width_bytes = 1;
    result.tokens8.assign(wide_tokens.begin(), wide_tokens.end());
  } else if (result.variant_count() <= 65535U) {
    result.token_width_bytes = 2;
    result.tokens16.assign(wide_tokens.begin(), wide_tokens.end());
  } else {
    result.token_width_bytes = 4;
    result.tokens32 = std::move(wide_tokens);
  }
  return result;
}

StaticP2PBsrPlan
build_static_p2p_bsr_plan(const StaticP2POperator &operator_map,
                          const std::span<const int> target_source_indices) {
  if (!target_source_indices.empty() &&
      target_source_indices.size() !=
          static_cast<std::size_t>(operator_map.target_count)) {
    throw std::invalid_argument("BSR P2P identity dimensions are inconsistent");
  }

  StaticP2PBsrPlan result;
  result.source_count = operator_map.source_count;
  result.target_count = operator_map.target_count;
  result.row_offsets = operator_map.row_offsets;
  result.target_source_indices.assign(target_source_indices.begin(),
                                      target_source_indices.end());
  if (result.target_source_indices.empty()) {
    result.target_source_indices.assign(
        static_cast<std::size_t>(operator_map.target_count), -1);
  }
  result.source_indices.reserve(operator_map.blocks.size());
  result.values.reserve(operator_map.blocks.size() * 9);

  for (const StaticDipoleBlock &block : operator_map.blocks) {
    result.source_indices.push_back(block.source);
    const bool self =
        block.skip_for_identity != 0 &&
        block.source ==
            result
                .target_source_indices[static_cast<std::size_t>(block.target)];
    const double xx = self ? 0.0 : block.xx;
    const double xy = self ? 0.0 : block.xy;
    const double xz = self ? 0.0 : block.xz;
    const double yy = self ? 0.0 : block.yy;
    const double yz = self ? 0.0 : block.yz;
    const double zz = self ? 0.0 : block.zz;
    result.values.insert(result.values.end(),
                         {xx, xy, xz, xy, yy, yz, xz, yz, zz});
  }
  return result;
}

FloatStaticP2PBsrPlan
build_static_p2p_bsr_plan(
    const FloatStaticP2POperator &operator_map,
    const std::span<const int> target_source_indices) {
  if (!target_source_indices.empty() &&
      target_source_indices.size() !=
          static_cast<std::size_t>(operator_map.target_count)) {
    throw std::invalid_argument("BSR P2P identity dimensions are inconsistent");
  }

  FloatStaticP2PBsrPlan result;
  result.source_count = operator_map.source_count;
  result.target_count = operator_map.target_count;
  result.row_offsets = operator_map.row_offsets;
  result.target_source_indices.assign(target_source_indices.begin(),
                                      target_source_indices.end());
  if (result.target_source_indices.empty()) {
    result.target_source_indices.assign(
        static_cast<std::size_t>(operator_map.target_count), -1);
  }
  result.source_indices.reserve(operator_map.blocks.size());
  result.values.reserve(operator_map.blocks.size() * 9);

  for (const FloatStaticDipoleBlock &block : operator_map.blocks) {
    result.source_indices.push_back(block.source);
    const bool self =
        block.skip_for_identity != 0 &&
        block.source ==
            result
                .target_source_indices[static_cast<std::size_t>(block.target)];
    const float xx = self ? 0.0F : block.xx;
    const float xy = self ? 0.0F : block.xy;
    const float xz = self ? 0.0F : block.xz;
    const float yy = self ? 0.0F : block.yy;
    const float yz = self ? 0.0F : block.yz;
    const float zz = self ? 0.0F : block.zz;
    result.values.insert(result.values.end(),
                         {xx, xy, xz, xy, yy, yz, xz, yz, zz});
  }
  return result;
}

void apply_static_p2p_operator(
    const StaticP2POperator& operator_map,
    const std::span<const Vec3> dipole_moments,
    const std::span<Vec3> H,
    const std::span<const int> target_source_indices)
{
    if (dipole_moments.size() != static_cast<std::size_t>(operator_map.source_count) ||
        H.size() != static_cast<std::size_t>(operator_map.target_count) ||
        (!target_source_indices.empty() && target_source_indices.size() != H.size())) {
        throw std::invalid_argument("static P2P dimensions are inconsistent");
    }
    #pragma omp parallel for schedule(static) if(operator_map.target_count >= 64)
    for (int target = 0; target < operator_map.target_count; ++target) {
        Vec3 field{};
        const int self = target_source_indices.empty()
            ? -1 : target_source_indices[static_cast<std::size_t>(target)];
        for (int entry = operator_map.row_offsets[static_cast<std::size_t>(target)];
             entry < operator_map.row_offsets[static_cast<std::size_t>(target) + 1];
             ++entry) {
            const StaticDipoleBlock& block = operator_map.blocks[
                static_cast<std::size_t>(entry)];
            if (block.skip_for_identity != 0 && block.source == self) {
                continue;
            }
            const Vec3 m = dipole_moments[static_cast<std::size_t>(block.source)];
            accumulate_static_dipole_block(block, m, field);
        }
        H[static_cast<std::size_t>(target)] += field;
    }
}

void apply_static_p2p_compact_plan(
    const StaticP2PCompactPlan &plan,
    const std::span<const Vec3> dipole_moments, const std::span<Vec3> H,
    const std::span<const int> target_source_indices) {
  if (dipole_moments.size() != static_cast<std::size_t>(plan.source_count) ||
      H.size() != static_cast<std::size_t>(plan.target_count) ||
      plan.skip_for_identity.size() != plan.source_indices.size() ||
      (!target_source_indices.empty() &&
       target_source_indices.size() != H.size())) {
    throw std::invalid_argument("compact P2P dimensions are inconsistent");
  }

#pragma omp parallel for schedule(static) if (plan.target_count >= 64)
  for (int target = 0; target < plan.target_count; ++target) {
    double Hx = 0.0;
    double Hy = 0.0;
    double Hz = 0.0;
    const int self =
        target_source_indices.empty()
            ? -1
            : target_source_indices[static_cast<std::size_t>(target)];
    const int begin = plan.row_offsets[static_cast<std::size_t>(target)];
    const int end = plan.row_offsets[static_cast<std::size_t>(target) + 1];
#pragma omp simd reduction(+ : Hx, Hy, Hz)
    for (int entry = begin; entry < end; ++entry) {
      const int source = plan.source_indices[static_cast<std::size_t>(entry)];
      if (plan.skip_for_identity[static_cast<std::size_t>(entry)] != 0 &&
          source == self) {
        continue;
      }
      const Vec3 m = dipole_moments[static_cast<std::size_t>(source)];
      const std::size_t index = static_cast<std::size_t>(entry);
      Hx += plan.tensors[0][index] * m.x + plan.tensors[1][index] * m.y +
            plan.tensors[2][index] * m.z;
      Hy += plan.tensors[1][index] * m.x + plan.tensors[3][index] * m.y +
            plan.tensors[4][index] * m.z;
      Hz += plan.tensors[2][index] * m.x + plan.tensors[4][index] * m.y +
            plan.tensors[5][index] * m.z;
    }
    H[static_cast<std::size_t>(target)] += Vec3{Hx, Hy, Hz};
  }
}

void apply_static_p2p_operator(
    const FloatStaticP2POperator& operator_map,
    const std::span<const FloatVec3> dipole_moments,
    const std::span<FloatVec3> H,
    const std::span<const int> target_source_indices)
{
    if (dipole_moments.size() !=
            static_cast<std::size_t>(operator_map.source_count) ||
        H.size() != static_cast<std::size_t>(operator_map.target_count) ||
        (!target_source_indices.empty() &&
         target_source_indices.size() != H.size())) {
        throw std::invalid_argument("static FP32 P2P dimensions are inconsistent");
    }
#pragma omp parallel for schedule(static) if(operator_map.target_count >= 64)
    for (int target = 0; target < operator_map.target_count; ++target) {
        FloatVec3 field{};
        const int self = target_source_indices.empty()
            ? -1 : target_source_indices[static_cast<std::size_t>(target)];
        for (int entry = operator_map.row_offsets[static_cast<std::size_t>(target)];
             entry < operator_map.row_offsets[static_cast<std::size_t>(target) + 1];
             ++entry) {
            const FloatStaticDipoleBlock& block =
                operator_map.blocks[static_cast<std::size_t>(entry)];
            if (block.skip_for_identity != 0 && block.source == self) {
                continue;
            }
            accumulate_static_dipole_block(
          block, dipole_moments[static_cast<std::size_t>(block.source)], field);
        }
        H[static_cast<std::size_t>(target)] += field;
    }
}

void apply_static_p2p_leaf_plan(
    const FloatStaticP2PLeafPlan& plan,
    const std::span<const FloatVec3> dipole_moments,
    const std::span<FloatVec3> H,
    const std::span<const int> target_source_indices)
{
    if (dipole_moments.size() != static_cast<std::size_t>(plan.source_count) ||
        H.size() != static_cast<std::size_t>(plan.target_count) ||
        (!target_source_indices.empty() &&
         target_source_indices.size() != H.size())) {
        throw std::invalid_argument("leaf FP32 P2P dimensions are inconsistent");
    }
    const int target_leaf_count = static_cast<int>(plan.target_begins.size());
#pragma omp parallel for schedule(static) if(target_leaf_count >= 8)
    for (int target_leaf = 0; target_leaf < target_leaf_count; ++target_leaf) {
        const int target_begin =
            plan.target_begins[static_cast<std::size_t>(target_leaf)];
        const int target_count =
            plan.target_counts[static_cast<std::size_t>(target_leaf)];
        for (int local_target = 0; local_target < target_count; ++local_target) {
            const int target = target_begin + local_target;
            const int self = target_source_indices.empty()
                ? -1 : target_source_indices[static_cast<std::size_t>(target)];
            FloatVec3 field{};
            for (int block_index =
                     plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
                 block_index < plan.leaf_row_offsets[
                     static_cast<std::size_t>(target_leaf) + 1];
                 ++block_index) {
                const StaticP2PLeafBlock& block =
                    plan.blocks[static_cast<std::size_t>(block_index)];
                const std::size_t tensor_begin = block.tensor_offset +
                    static_cast<std::size_t>(local_target) * block.source_count;
                for (int local_source = 0; local_source < block.source_count;
                     ++local_source) {
                    const int source = block.source_begin + local_source;
                    if (source == self) {
                        continue;
                    }
                    const std::size_t index = tensor_begin + local_source;
                    const FloatVec3 moment =
                        dipole_moments[static_cast<std::size_t>(source)];
                    field.x += plan.tensors[0][index] * moment.x +
                        plan.tensors[1][index] * moment.y +
                        plan.tensors[2][index] * moment.z;
                    field.y += plan.tensors[1][index] * moment.x +
                        plan.tensors[3][index] * moment.y +
                        plan.tensors[4][index] * moment.z;
                    field.z += plan.tensors[2][index] * moment.x +
                        plan.tensors[4][index] * moment.y +
                        plan.tensors[5][index] * moment.z;
                }
            }
            H[static_cast<std::size_t>(target)] += field;
        }
    }
}

void apply_static_p2p_tensor_dictionary_plan(
    const FloatStaticP2PTensorDictionaryPlan &plan,
    const std::span<const FloatVec3> dipole_moments,
    const std::span<FloatVec3> H,
    const std::span<const int> target_source_indices) {
  if (dipole_moments.size() != static_cast<std::size_t>(plan.source_count) ||
      H.size() != static_cast<std::size_t>(plan.target_count) ||
      (!target_source_indices.empty() && target_source_indices.size() != H.size())) {
    throw std::invalid_argument("FP32 Tensor6 dictionary P2P dimensions are inconsistent");
  }
#pragma omp parallel for schedule(static) if (plan.target_begins.size() >= 8)
  for (int target_leaf = 0;
       target_leaf < static_cast<int>(plan.target_begins.size()); ++target_leaf) {
    const int target_begin = plan.target_begins[static_cast<std::size_t>(target_leaf)];
    const int target_count = plan.target_counts[static_cast<std::size_t>(target_leaf)];
    for (int local_target = 0; local_target < target_count; ++local_target) {
      const int target = target_begin + local_target;
      const int self = target_source_indices.empty() ? -1 : target_source_indices[static_cast<std::size_t>(target)];
      FloatVec3 field{};
      for (int block_index = plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
           block_index < plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf) + 1]; ++block_index) {
        const StaticP2PLeafBlock &block = plan.blocks[static_cast<std::size_t>(block_index)];
        const std::size_t begin = block.tensor_offset + static_cast<std::size_t>(local_target) * block.source_count;
        for (int local_source = 0; local_source < block.source_count; ++local_source) {
          const int source = block.source_begin + local_source;
          if (plan.skip_for_identity && source == self) continue;
          const std::uint32_t token = plan.tokens[begin + local_source];
          const auto id = tensor6_token_id(token);
          const auto signs = tensor6_token_sign_mask(token);
          const auto value = [&](int component) {
            const float coefficient = plan.tensors[static_cast<std::size_t>(component)][id];
            return (signs & (1U << component)) != 0U ? -coefficient : coefficient;
          };
          const auto moment = dipole_moments[static_cast<std::size_t>(source)];
          const float xx = value(0), xy = value(1), xz = value(2),
                      yy = value(3), yz = value(4), zz = value(5);
          field.x += xx * moment.x + xy * moment.y + xz * moment.z;
          field.y += xy * moment.x + yy * moment.y + yz * moment.z;
          field.z += xz * moment.x + yz * moment.y + zz * moment.z;
        }
      }
      H[static_cast<std::size_t>(target)] += field;
    }
  }
}

void apply_static_p2p_bsr_plan(
    const FloatStaticP2PBsrPlan& plan,
    const std::span<const FloatVec3> dipole_moments,
    const std::span<FloatVec3> H,
    const std::span<const int> target_source_indices)
{
    if (dipole_moments.size() != static_cast<std::size_t>(plan.source_count) ||
        H.size() != static_cast<std::size_t>(plan.target_count) ||
        (!target_source_indices.empty() &&
         !std::equal(target_source_indices.begin(), target_source_indices.end(),
                     plan.target_source_indices.begin(),
                     plan.target_source_indices.end()))) {
        throw std::invalid_argument(
            "FP32 BSR P2P dimensions or identities are inconsistent");
    }
#pragma omp parallel for schedule(static) if(plan.target_count >= 64)
    for (int target = 0; target < plan.target_count; ++target) {
        FloatVec3 field{};
        for (int entry = plan.row_offsets[static_cast<std::size_t>(target)];
             entry < plan.row_offsets[static_cast<std::size_t>(target) + 1];
             ++entry) {
            const FloatVec3 moment = dipole_moments[static_cast<std::size_t>(
                plan.source_indices[static_cast<std::size_t>(entry)])];
            const float* block =
                plan.values.data() + static_cast<std::size_t>(entry) * 9;
            field.x += block[0] * moment.x + block[1] * moment.y +
                block[2] * moment.z;
            field.y += block[3] * moment.x + block[4] * moment.y +
                block[5] * moment.z;
            field.z += block[6] * moment.x + block[7] * moment.y +
                block[8] * moment.z;
        }
        H[static_cast<std::size_t>(target)] += field;
    }
}

void apply_static_p2p_leaf_plan(
    const StaticP2PLeafPlan &plan, const std::span<const Vec3> dipole_moments,
    const std::span<Vec3> H, const std::span<const int> target_source_indices) {
  if (dipole_moments.size() != static_cast<std::size_t>(plan.source_count) ||
      H.size() != static_cast<std::size_t>(plan.target_count) ||
      (!target_source_indices.empty() &&
       target_source_indices.size() != H.size())) {
    throw std::invalid_argument("leaf P2P dimensions are inconsistent");
  }

  const int target_leaf_count = static_cast<int>(plan.target_begins.size());
#pragma omp parallel for schedule(static) if (target_leaf_count >= 8)
  for (int target_leaf = 0; target_leaf < target_leaf_count; ++target_leaf) {
    const int target_begin =
        plan.target_begins[static_cast<std::size_t>(target_leaf)];
    const int target_count =
        plan.target_counts[static_cast<std::size_t>(target_leaf)];
    for (int local_target = 0; local_target < target_count; ++local_target) {
      const int target = target_begin + local_target;
      const int self =
          target_source_indices.empty()
              ? -1
              : target_source_indices[static_cast<std::size_t>(target)];
      double Hx = 0.0;
      double Hy = 0.0;
      double Hz = 0.0;
      for (int block_index =
               plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
           block_index <
           plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf) + 1];
           ++block_index) {
        const StaticP2PLeafBlock &block =
            plan.blocks[static_cast<std::size_t>(block_index)];
        const std::size_t tensor_begin =
            block.tensor_offset +
            static_cast<std::size_t>(local_target) * block.source_count;
#pragma omp simd reduction(+ : Hx, Hy, Hz)
        for (int local_source = 0; local_source < block.source_count;
             ++local_source) {
          const int source = block.source_begin + local_source;
          if (source == self) {
            continue;
          }
          const std::size_t index = tensor_begin + local_source;
          const Vec3 m = dipole_moments[static_cast<std::size_t>(source)];
          Hx += plan.tensors[0][index] * m.x + plan.tensors[1][index] * m.y +
                plan.tensors[2][index] * m.z;
          Hy += plan.tensors[1][index] * m.x + plan.tensors[3][index] * m.y +
                plan.tensors[4][index] * m.z;
          Hz += plan.tensors[2][index] * m.x + plan.tensors[4][index] * m.y +
                plan.tensors[5][index] * m.z;
        }
      }
      H[static_cast<std::size_t>(target)] += Vec3{Hx, Hy, Hz};
    }
  }
}

void apply_static_p2p_tensor_dictionary_plan(
    const StaticP2PTensorDictionaryPlan &plan,
    const std::span<const Vec3> dipole_moments, const std::span<Vec3> H,
    const std::span<const int> target_source_indices) {
  if (dipole_moments.size() != static_cast<std::size_t>(plan.source_count) ||
      H.size() != static_cast<std::size_t>(plan.target_count) ||
      (!target_source_indices.empty() && target_source_indices.size() != H.size())) {
    throw std::invalid_argument("Tensor6 dictionary P2P dimensions are inconsistent");
  }
#pragma omp parallel for schedule(static) if (plan.target_begins.size() >= 8)
  for (int target_leaf = 0;
       target_leaf < static_cast<int>(plan.target_begins.size()); ++target_leaf) {
    const int target_begin = plan.target_begins[static_cast<std::size_t>(target_leaf)];
    const int target_count = plan.target_counts[static_cast<std::size_t>(target_leaf)];
    for (int local_target = 0; local_target < target_count; ++local_target) {
      const int target = target_begin + local_target;
      const int self = target_source_indices.empty() ? -1 :
          target_source_indices[static_cast<std::size_t>(target)];
      Vec3 field{};
      for (int block_index = plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
           block_index < plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf) + 1];
           ++block_index) {
        const StaticP2PLeafBlock &block = plan.blocks[static_cast<std::size_t>(block_index)];
        const std::size_t begin = block.tensor_offset +
            static_cast<std::size_t>(local_target) * block.source_count;
        for (int local_source = 0; local_source < block.source_count; ++local_source) {
          const int source = block.source_begin + local_source;
          if (plan.skip_for_identity && source == self) {
            continue;
          }
          const std::uint32_t token = plan.tokens[begin + local_source];
          const std::uint32_t id = tensor6_token_id(token);
          const std::uint8_t signs = tensor6_token_sign_mask(token);
          const auto signed_value = [&](const int component) {
            const double value = plan.tensors[static_cast<std::size_t>(component)][id];
            return (signs & (1U << component)) != 0U ? -value : value;
          };
          const Vec3 moment = dipole_moments[static_cast<std::size_t>(source)];
          const double xx = signed_value(0), xy = signed_value(1),
                       xz = signed_value(2), yy = signed_value(3),
                       yz = signed_value(4), zz = signed_value(5);
          field.x += xx * moment.x + xy * moment.y + xz * moment.z;
          field.y += xy * moment.x + yy * moment.y + yz * moment.z;
          field.z += xz * moment.x + yz * moment.y + zz * moment.z;
        }
      }
      H[static_cast<std::size_t>(target)] += field;
    }
  }
}

namespace {

template <typename Scalar, typename Vector, typename Plan, typename Token>
void apply_signed_tensor_dictionary_impl(const Plan &plan,
                                         const std::span<const Vector> moments,
                                         const std::span<Vector> fields,
                                         const std::span<const Token> tokens) {
  constexpr int max_tile_size = 128;
  const int work_count = static_cast<int>(plan.tile_leaf_indices.size());
  const Scalar *const xx = plan.tensors[0].data();
  const Scalar *const xy = plan.tensors[1].data();
  const Scalar *const xz = plan.tensors[2].data();
  const Scalar *const yy = plan.tensors[3].data();
  const Scalar *const yz = plan.tensors[4].data();
  const Scalar *const zz = plan.tensors[5].data();
  const Token *const token_data = tokens.data();
#pragma omp parallel for schedule(static) if (work_count >= 8)
  for (int work = 0; work < work_count; ++work) {
    const int target_leaf = plan.tile_leaf_indices[static_cast<std::size_t>(work)];
    const int local_begin = plan.tile_target_offsets[static_cast<std::size_t>(work)];
    const int target_begin = plan.target_begins[static_cast<std::size_t>(target_leaf)];
    const int target_count = plan.target_counts[static_cast<std::size_t>(target_leaf)];
    const int lanes = std::min(plan.target_tile_size, target_count - local_begin);
    std::array<Scalar, max_tile_size> Hx{};
    std::array<Scalar, max_tile_size> Hy{};
    std::array<Scalar, max_tile_size> Hz{};
    for (int block_index = plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf)];
         block_index < plan.leaf_row_offsets[static_cast<std::size_t>(target_leaf) + 1];
         ++block_index) {
      const StaticP2PLeafBlock &block = plan.blocks[static_cast<std::size_t>(block_index)];
      for (int local_source = 0; local_source < block.source_count; ++local_source) {
        const Vector moment = moments[static_cast<std::size_t>(block.source_begin + local_source)];
        const Token *const block_tokens = token_data + block.tensor_offset +
            static_cast<std::size_t>(local_source) * target_count + local_begin;
#pragma omp simd
        for (int lane = 0; lane < lanes; ++lane) {
          const std::size_t variant = block_tokens[lane];
          Hx[static_cast<std::size_t>(lane)] += xx[variant] * moment.x +
              xy[variant] * moment.y + xz[variant] * moment.z;
          Hy[static_cast<std::size_t>(lane)] += xy[variant] * moment.x +
              yy[variant] * moment.y + yz[variant] * moment.z;
          Hz[static_cast<std::size_t>(lane)] += xz[variant] * moment.x +
              yz[variant] * moment.y + zz[variant] * moment.z;
        }
      }
    }
    for (int lane = 0; lane < lanes; ++lane) {
      fields[static_cast<std::size_t>(target_begin + local_begin + lane)] +=
          Vector{Hx[static_cast<std::size_t>(lane)], Hy[static_cast<std::size_t>(lane)],
                 Hz[static_cast<std::size_t>(lane)]};
    }
  }
}

template <typename Scalar, typename Vector, typename Plan>
void apply_signed_tensor_dictionary(const Plan &plan,
                                    const std::span<const Vector> moments,
                                    const std::span<Vector> fields) {
  if (moments.size() != static_cast<std::size_t>(plan.source_count) ||
      fields.size() != static_cast<std::size_t>(plan.target_count)) {
    throw std::invalid_argument("signed Tensor6 dictionary P2P dimensions are inconsistent");
  }
  if (plan.target_tile_size <= 0 || plan.target_tile_size > 128 ||
      plan.tile_leaf_indices.size() != plan.tile_target_offsets.size()) {
    throw std::invalid_argument("signed Tensor6 dictionary plan is malformed");
  }
  if (plan.token_width_bytes == 1) {
    apply_signed_tensor_dictionary_impl<Scalar>(plan, moments, fields,
                                                std::span{plan.tokens8});
  } else if (plan.token_width_bytes == 2) {
    apply_signed_tensor_dictionary_impl<Scalar>(plan, moments, fields,
                                                std::span{plan.tokens16});
  } else if (plan.token_width_bytes == 4) {
    apply_signed_tensor_dictionary_impl<Scalar>(plan, moments, fields,
                                                std::span{plan.tokens32});
  } else {
    throw std::invalid_argument("signed Tensor6 dictionary token width is invalid");
  }
}

} // namespace

void apply_static_p2p_signed_tensor_dictionary_plan(
    const StaticP2PSignedTensorDictionaryPlan &plan,
    const std::span<const Vec3> dipole_moments, const std::span<Vec3> H) {
  apply_signed_tensor_dictionary<double>(plan, dipole_moments, H);
}

void apply_static_p2p_signed_tensor_dictionary_plan(
    const FloatStaticP2PSignedTensorDictionaryPlan &plan,
    const std::span<const FloatVec3> dipole_moments,
    const std::span<FloatVec3> H) {
  apply_signed_tensor_dictionary<float>(plan, dipole_moments, H);
}

void apply_static_p2p_bsr_plan(
    const StaticP2PBsrPlan &plan, const std::span<const Vec3> dipole_moments,
    const std::span<Vec3> H, const std::span<const int> target_source_indices) {
  if (dipole_moments.size() != static_cast<std::size_t>(plan.source_count) ||
      H.size() != static_cast<std::size_t>(plan.target_count) ||
      (!target_source_indices.empty() &&
       !std::equal(target_source_indices.begin(), target_source_indices.end(),
                   plan.target_source_indices.begin(),
                   plan.target_source_indices.end()))) {
    throw std::invalid_argument(
        "BSR P2P dimensions or fixed identities are inconsistent");
  }

#pragma omp parallel for schedule(static) if (plan.target_count >= 64)
  for (int target = 0; target < plan.target_count; ++target) {
    Vec3 field{};
    for (int entry = plan.row_offsets[static_cast<std::size_t>(target)];
         entry < plan.row_offsets[static_cast<std::size_t>(target) + 1];
         ++entry) {
      const Vec3 moment = dipole_moments[static_cast<std::size_t>(
          plan.source_indices[static_cast<std::size_t>(entry)])];
      const double *block =
          plan.values.data() + static_cast<std::size_t>(entry) * 9;
      field.x +=
          block[0] * moment.x + block[1] * moment.y + block[2] * moment.z;
      field.y +=
          block[3] * moment.x + block[4] * moment.y + block[5] * moment.z;
      field.z +=
          block[6] * moment.x + block[7] * moment.y + block[8] * moment.z;
    }
    H[static_cast<std::size_t>(target)] += field;
  }
}

void apply_static_operator(
    const StaticCoefficientOperator& operator_map,
    const std::span<const double> input,
    const std::span<double> output)
{
    validate_operator_dimensions(operator_map, input, output);
    for (const StaticOperatorEntry& entry : operator_map.entries) {
        output[static_cast<std::size_t>(entry.output)] += entry.value *
            input[static_cast<std::size_t>(entry.input)];
    }
}

PotentialField apply_static_l2p_evaluator(
    const StaticL2PEvaluator& evaluator,
    const std::span<const double> L,
    const OutputFlags output)
{
    if (evaluator.potential.size() != L.size()) {
        throw std::invalid_argument("static L2P dimensions are inconsistent");
    }
    PotentialField result;
    for (std::size_t index = 0; index < L.size(); ++index) {
        if (has_flag(output, OutputFlags::Potential)) {
            result.phi += evaluator.potential[index] * L[index];
        }
        if (has_flag(output, OutputFlags::Field)) {
            result.H.x += evaluator.field[0][index] * L[index];
            result.H.y += evaluator.field[1][index] * L[index];
            result.H.z += evaluator.field[2][index] * L[index];
        }
    }
    return result;
}

void apply_static_coefficient_matrix(const std::span<const double> matrix,
    const std::span<const double> input,
                                     const std::span<double> output) {
    const std::size_t coefficient_count = input.size();
    if (output.size() != coefficient_count ||
        matrix.size() != coefficient_count * coefficient_count) {
        throw std::invalid_argument(
        "static coefficient matrix dimensions are inconsistent");
    }
    for (std::size_t alpha = 0; alpha < coefficient_count; ++alpha) {
        for (std::size_t beta = 0; beta < coefficient_count; ++beta) {
      output[beta] += matrix[beta + coefficient_count * alpha] * input[alpha];
        }
    }
}

//------------------------------------------------------------------------------
// FP32 plan conversion and execution
//------------------------------------------------------------------------------

FloatStaticCoefficientOperator
quantise_static_operator(const StaticCoefficientOperator &source) {
    FloatStaticCoefficientOperator result;
    result.input_size = source.input_size;
    result.output_size = source.output_size;
    result.entries.reserve(source.entries.size());
    for (const StaticOperatorEntry& entry : source.entries) {
        result.entries.push_back(
            {entry.output, entry.input, static_cast<float>(entry.value)});
    }
    return result;
}

FloatStaticL2PEvaluator
quantise_static_l2p_evaluator(const StaticL2PEvaluator &source) {
    FloatStaticL2PEvaluator result;
    result.potential.assign(source.potential.begin(), source.potential.end());
    for (std::size_t component = 0; component < 3; ++component) {
        result.field[component].assign(source.field[component].begin(),
                                       source.field[component].end());
    }
    return result;
}

FloatStaticP2POperator quantise_static_p2p_operator(
    const StaticP2POperator& source)
{
    FloatStaticP2POperator result;
    result.source_count = source.source_count;
    result.target_count = source.target_count;
    result.row_offsets = source.row_offsets;
    result.blocks.reserve(source.blocks.size());
    for (const StaticDipoleBlock& block : source.blocks) {
        result.blocks.push_back({
            block.target, block.source, static_cast<float>(block.px),
            static_cast<float>(block.py), static_cast<float>(block.pz),
            static_cast<float>(block.xx),
            static_cast<float>(block.xy), static_cast<float>(block.xz),
            static_cast<float>(block.yy), static_cast<float>(block.yz),
            static_cast<float>(block.zz), block.skip_for_identity});
    }
    return result;
}

FloatStaticP2PCompactPlan quantise_static_p2p_compact_plan(
    const StaticP2PCompactPlan& source)
{
    FloatStaticP2PCompactPlan result;
    result.source_count = source.source_count;
    result.target_count = source.target_count;
    result.row_offsets = source.row_offsets;
    result.source_indices = source.source_indices;
    result.skip_for_identity = source.skip_for_identity;
    for (std::size_t component = 0; component < 3; ++component) {
        result.potential[component].assign(
            source.potential[component].begin(),
            source.potential[component].end());
    }
    for (std::size_t component = 0; component < 6; ++component) {
        result.tensors[component].assign(source.tensors[component].begin(),
                                         source.tensors[component].end());
    }
    return result;
}

FloatStaticP2PLeafPlan
quantise_static_p2p_leaf_plan(const StaticP2PLeafPlan &source) {
    FloatStaticP2PLeafPlan result;
    result.source_count = source.source_count;
    result.target_count = source.target_count;
    result.target_begins = source.target_begins;
    result.target_counts = source.target_counts;
    result.leaf_row_offsets = source.leaf_row_offsets;
    result.blocks = source.blocks;
    for (std::size_t component = 0; component < 6; ++component) {
        result.tensors[component].assign(source.tensors[component].begin(),
                                         source.tensors[component].end());
    }
    result.minimum_occupancy = source.minimum_occupancy;
    result.maximum_occupancy = source.maximum_occupancy;
    result.mean_occupancy = source.mean_occupancy;
    result.unique_occupancies = source.unique_occupancies;
    result.uniform_occupancy = source.uniform_occupancy;
    return result;
}

FloatStaticP2PTensorDictionaryPlan
quantise_static_p2p_tensor_dictionary_plan(
    const StaticP2PTensorDictionaryPlan &source) {
  FloatStaticP2PTensorDictionaryPlan result;
  result.source_count = source.source_count;
  result.target_count = source.target_count;
  result.target_begins = source.target_begins;
  result.target_counts = source.target_counts;
  result.leaf_row_offsets = source.leaf_row_offsets;
  result.blocks = source.blocks;
  result.skip_for_identity = source.skip_for_identity;
  result.tokens.reserve(source.tokens.size());
  std::unordered_map<Tensor6BitKey<float>, std::uint32_t,
                     Tensor6BitKeyHash<float>> dictionary_ids;
  dictionary_ids.reserve(source.tokens.size());
  for (const std::uint32_t source_token : source.tokens) {
    const std::uint32_t source_id = tensor6_token_id(source_token);
    const std::uint8_t source_signs = tensor6_token_sign_mask(source_token);
    std::array<float, 6> tensor{};
    for (int component = 0; component < 6; ++component) {
      const float magnitude = static_cast<float>(
          source.tensors[static_cast<std::size_t>(component)][source_id]);
      tensor[static_cast<std::size_t>(component)] =
          (source_signs & (1U << component)) != 0U ? -magnitude : magnitude;
    }
    const auto canonical = canonicalise_tensor6(tensor);
    const auto key = tensor6_bit_key(canonical.values);
    const auto [iterator, inserted] = dictionary_ids.try_emplace(
        key, static_cast<std::uint32_t>(dictionary_ids.size()));
    if (inserted) {
      if (iterator->second >= (1U << 26U)) {
        throw std::overflow_error("FP32 Tensor6 dictionary exceeds packed token IDs");
      }
      for (int component = 0; component < 6; ++component) {
        result.tensors[static_cast<std::size_t>(component)].push_back(
            canonical.values[static_cast<std::size_t>(component)]);
      }
    }
    result.tokens.push_back(pack_tensor6_token(iterator->second,
                                                canonical.sign_mask));
  }
  return result;
}

FloatStaticP2PSignedTensorDictionaryPlan
quantise_static_p2p_signed_tensor_dictionary_plan(
    const StaticP2PSignedTensorDictionaryPlan &source) {
  FloatStaticP2PSignedTensorDictionaryPlan result;
  result.source_count = source.source_count;
  result.target_count = source.target_count;
  result.target_begins = source.target_begins;
  result.target_counts = source.target_counts;
  result.leaf_row_offsets = source.leaf_row_offsets;
  result.blocks = source.blocks;
  result.tile_leaf_indices = source.tile_leaf_indices;
  result.tile_target_offsets = source.tile_target_offsets;
  result.target_tile_size = source.target_tile_size;
  for (auto &component : result.tensors) {
    component.push_back(0.0F);
  }
  std::unordered_map<Tensor6BitKey<float>, std::uint32_t,
                     Tensor6BitKeyHash<float>> variant_ids;
  variant_ids.emplace(tensor6_bit_key(std::array<float, 6>{}), 0U);
  std::vector<std::uint32_t> wide_tokens;
  wide_tokens.reserve(source.token_count());
  const auto append_token = [&](const std::uint32_t source_variant) {
    std::array<float, 6> tensor{};
    for (int component = 0; component < 6; ++component) {
      const float value = static_cast<float>(
          source.tensors[static_cast<std::size_t>(component)][source_variant]);
      tensor[static_cast<std::size_t>(component)] = value == 0.0F ? 0.0F : value;
    }
    const auto [iterator, inserted] = variant_ids.try_emplace(
        tensor6_bit_key(tensor), static_cast<std::uint32_t>(variant_ids.size()));
    if (inserted) {
      for (int component = 0; component < 6; ++component) {
        result.tensors[static_cast<std::size_t>(component)].push_back(
            tensor[static_cast<std::size_t>(component)]);
      }
    }
    wide_tokens.push_back(iterator->second);
  };
  if (source.token_width_bytes == 1) {
    for (const std::uint8_t token : source.tokens8) append_token(token);
  } else if (source.token_width_bytes == 2) {
    for (const std::uint16_t token : source.tokens16) append_token(token);
  } else if (source.token_width_bytes == 4) {
    for (const std::uint32_t token : source.tokens32) append_token(token);
  } else {
    throw std::invalid_argument("signed Tensor6 dictionary token width is invalid");
  }
  if (result.variant_count() <= 255U) {
    result.token_width_bytes = 1;
    result.tokens8.assign(wide_tokens.begin(), wide_tokens.end());
  } else if (result.variant_count() <= 65535U) {
    result.token_width_bytes = 2;
    result.tokens16.assign(wide_tokens.begin(), wide_tokens.end());
  } else {
    result.token_width_bytes = 4;
    result.tokens32 = std::move(wide_tokens);
  }
  return result;
}

FloatStaticP2PBsrPlan
quantise_static_p2p_bsr_plan(const StaticP2PBsrPlan &source) {
    FloatStaticP2PBsrPlan result;
    result.source_count = source.source_count;
    result.target_count = source.target_count;
    result.row_offsets = source.row_offsets;
    result.source_indices = source.source_indices;
    result.values.assign(source.values.begin(), source.values.end());
    result.target_source_indices = source.target_source_indices;
    return result;
}

FloatStaticM2LPlan quantise_static_m2l_plan(const StaticM2LPlan &source) {
    FloatStaticM2LPlan result;
    result.coefficient_count = source.coefficient_count;
    result.matrix_count = source.matrix_count;
    result.level_count = source.level_count;
    result.matrices.assign(source.matrices.begin(), source.matrices.end());
    result.multipole_scaling.assign(source.multipole_scaling.begin(),
                                    source.multipole_scaling.end());
    result.local_scaling.assign(source.local_scaling.begin(),
                                source.local_scaling.end());
    result.target_row_offsets = source.target_row_offsets;
    result.source_nodes = source.source_nodes;
    result.matrix_ids = source.matrix_ids;
    result.interaction_levels = source.interaction_levels;
    result.level_target_begin = source.level_target_begin;
    result.level_target_end = source.level_target_end;
    return result;
}

std::size_t FloatStaticP2POperator::memory_bytes() const noexcept {
    return row_offsets.size() * sizeof(int) +
           blocks.size() * sizeof(FloatStaticDipoleBlock);
}

StaticP2PMemory FloatStaticP2PCompactPlan::memory() const noexcept {
    StaticP2PMemory result;
    result.tensor_bytes = tensors[0].size() * 9 * sizeof(float);
    result.index_bytes = source_indices.size() * sizeof(int) +
        skip_for_identity.size() * sizeof(unsigned char);
    result.row_metadata_bytes = row_offsets.size() * sizeof(int);
    return result;
}

StaticP2PMemory FloatStaticP2PLeafPlan::memory() const noexcept {
    StaticP2PMemory result;
    result.tensor_bytes = tensors[0].size() * 6 * sizeof(float);
    result.row_metadata_bytes = leaf_row_offsets.size() * sizeof(int);
    result.leaf_metadata_bytes =
        (target_begins.size() + target_counts.size()) * sizeof(int) +
        blocks.size() * sizeof(StaticP2PLeafBlock);
    return result;
}

StaticP2PMemory FloatStaticP2PTensorDictionaryPlan::memory() const noexcept {
  StaticP2PMemory result;
  result.tensor_bytes = tensors[0].size() * 6 * sizeof(float);
  result.index_bytes = tokens.size() * sizeof(std::uint32_t);
  result.row_metadata_bytes = leaf_row_offsets.size() * sizeof(int);
  result.leaf_metadata_bytes =
      (target_begins.size() + target_counts.size()) * sizeof(int) +
      blocks.size() * sizeof(StaticP2PLeafBlock);
  return result;
}

StaticP2PMemory FloatStaticP2PSignedTensorDictionaryPlan::memory() const noexcept {
  StaticP2PMemory result;
  result.tensor_bytes = tensors[0].size() * 6 * sizeof(float);
  result.index_bytes = tokens8.size() * sizeof(std::uint8_t) +
      tokens16.size() * sizeof(std::uint16_t) +
      tokens32.size() * sizeof(std::uint32_t);
  result.row_metadata_bytes = leaf_row_offsets.size() * sizeof(int);
  result.leaf_metadata_bytes =
      (target_begins.size() + target_counts.size() + tile_leaf_indices.size() +
       tile_target_offsets.size()) * sizeof(int) +
      blocks.size() * sizeof(StaticP2PLeafBlock);
  return result;
}

StaticP2PMemory FloatStaticP2PBsrPlan::memory() const noexcept {
    StaticP2PMemory result;
    result.tensor_bytes = values.size() * sizeof(float);
    result.index_bytes = source_indices.size() * sizeof(int);
    result.row_metadata_bytes =
        (row_offsets.size() + target_source_indices.size()) * sizeof(int);
    return result;
}

void apply_static_operator(
    const FloatStaticCoefficientOperator& operator_map,
    const std::span<const float> input,
    const std::span<float> output)
{
    if (input.size() != static_cast<std::size_t>(operator_map.input_size) ||
        output.size() != static_cast<std::size_t>(operator_map.output_size)) {
        throw std::invalid_argument("static FP32 operator dimensions are inconsistent");
    }
    for (const FloatStaticOperatorEntry& entry : operator_map.entries) {
        output[static_cast<std::size_t>(entry.output)] +=
            entry.value * input[static_cast<std::size_t>(entry.input)];
    }
}

FloatPotentialField
apply_static_l2p_evaluator(const FloatStaticL2PEvaluator &evaluator,
    const std::span<const float> L,
                           const OutputFlags output) {
    if (evaluator.potential.size() != L.size()) {
        throw std::invalid_argument("static FP32 L2P dimensions are inconsistent");
    }
    FloatPotentialField result;
    for (std::size_t index = 0; index < L.size(); ++index) {
        if (has_flag(output, OutputFlags::Potential)) {
            result.phi += evaluator.potential[index] * L[index];
        }
        if (has_flag(output, OutputFlags::Field)) {
            result.H.x += evaluator.field[0][index] * L[index];
            result.H.y += evaluator.field[1][index] * L[index];
            result.H.z += evaluator.field[2][index] * L[index];
        }
    }
    return result;
}

void apply_static_p2p_compact_plan(
    const FloatStaticP2PCompactPlan& plan,
    const std::span<const FloatVec3> dipole_moments,
    const std::span<FloatVec3> H,
    const std::span<const int> target_source_indices)
{
    if (dipole_moments.size() != static_cast<std::size_t>(plan.source_count) ||
        H.size() != static_cast<std::size_t>(plan.target_count) ||
        plan.skip_for_identity.size() != plan.source_indices.size() ||
        (!target_source_indices.empty() && target_source_indices.size() != H.size())) {
        throw std::invalid_argument("compact FP32 P2P dimensions are inconsistent");
    }
#pragma omp parallel for schedule(static) if(plan.target_count >= 64)
    for (int target = 0; target < plan.target_count; ++target) {
        FloatVec3 field{};
        const int self = target_source_indices.empty()
            ? -1 : target_source_indices[static_cast<std::size_t>(target)];
        for (int entry = plan.row_offsets[static_cast<std::size_t>(target)];
             entry < plan.row_offsets[static_cast<std::size_t>(target) + 1];
             ++entry) {
            const std::size_t index = static_cast<std::size_t>(entry);
            const int source = plan.source_indices[index];
            if (plan.skip_for_identity[index] != 0 && source == self) {
                continue;
            }
            const FloatVec3 moment = dipole_moments[static_cast<std::size_t>(source)];
            field.x += plan.tensors[0][index] * moment.x +
                       plan.tensors[1][index] * moment.y +
                       plan.tensors[2][index] * moment.z;
            field.y += plan.tensors[1][index] * moment.x +
                       plan.tensors[3][index] * moment.y +
                       plan.tensors[4][index] * moment.z;
            field.z += plan.tensors[2][index] * moment.x +
                       plan.tensors[4][index] * moment.y +
                       plan.tensors[5][index] * moment.z;
        }
        H[static_cast<std::size_t>(target)] += field;
    }
}

} // namespace cdfmm
