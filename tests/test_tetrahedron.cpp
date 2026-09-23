// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cdfmm/rectangular_prism.hpp"
#include "cdfmm/tetrahedron.hpp"
#include "geometry/primitives/tetrahedron_detail.hpp"

using namespace cdfmm;

namespace {

Tetrahedron reference_tetrahedron()
{
    // Unit right tetrahedron, shifted so that its representative is its
    // centroid.  This gives volume 1/6 and a particularly simple moment
    // reference.
    return {{{{-0.25, -0.25, -0.25},
              {0.75, -0.25, -0.25},
              {-0.25, 0.75, -0.25},
              {-0.25, -0.25, 0.75}}}};
}

PairTensor point_tensor(const Vec3& r)
{
    const double r2 = dot(r, r);
    const double inverse_r3 = 1.0 / (std::sqrt(r2) * r2);
    const double diagonal = inverse_r3 / (4.0 * std::numbers::pi);
    const double common = 3.0 * diagonal / r2;
    return {common * r.x * r.x - diagonal,
            common * r.x * r.y,
            common * r.x * r.z,
            common * r.y * r.y - diagonal,
            common * r.y * r.z,
            common * r.z * r.z - diagonal};
}

Tetrahedron scale_tetrahedron(const Tetrahedron& tetrahedron,
                              const double factor)
{
    Tetrahedron result = tetrahedron;
    for (Vec3& vertex : result.vertices) {
        vertex = vertex * factor;
    }
    return result;
}

Tetrahedron centred_physical_tetrahedron(const std::array<Vec3, 4>& vertices)
{
    Vec3 centroid{};
    for (const Vec3& vertex : vertices) {
        centroid += vertex;
    }
    centroid = centroid * 0.25;
    Tetrahedron result;
    for (int index = 0; index < 4; ++index) {
        result.vertices[static_cast<std::size_t>(index)] =
            vertices[static_cast<std::size_t>(index)] - centroid;
    }
    return result;
}

void require_finite(const PairTensor& tensor)
{
    REQUIRE(std::isfinite(tensor.xx));
    REQUIRE(std::isfinite(tensor.xy));
    REQUIRE(std::isfinite(tensor.xz));
    REQUIRE(std::isfinite(tensor.yy));
    REQUIRE(std::isfinite(tensor.yz));
    REQUIRE(std::isfinite(tensor.zz));
}

double maximum_component_difference(const PairTensor& left,
                                   const PairTensor& right)
{
    return std::max({
        std::abs(left.xx - right.xx), std::abs(left.xy - right.xy),
        std::abs(left.xz - right.xz), std::abs(left.yy - right.yy),
        std::abs(left.yz - right.yz), std::abs(left.zz - right.zz)});
}

PairTensor rotate_z_quarter_turn(const PairTensor& tensor)
{
    // Q = [[0,-1,0],[1,0,0],[0,0,1]], so K' = Q K Q^T.
    return {tensor.yy, -tensor.xy, -tensor.yz, tensor.xx, tensor.xz,
            tensor.zz};
}

PairTensor average_point_tensor_over_target(const Vec3& displacement,
                                            const Tetrahedron& source,
                                            const Tetrahedron& target)
{
    constexpr std::array<double, 8> node{{
        0.0198550717512319, 0.1016667612931866,
        0.2372337950418355, 0.4082826787521751,
        0.5917173212478249, 0.7627662049581645,
        0.8983332387068134, 0.9801449282487681}};
    constexpr std::array<double, 8> weight{{
        0.0506142681451881, 0.1111905172266872,
        0.1568533229389436, 0.1813418916891809,
        0.1813418916891809, 0.1568533229389436,
        0.1111905172266872, 0.0506142681451881}};
    const Vec3 edge_b = target.vertices[1] - target.vertices[0];
    const Vec3 edge_c = target.vertices[2] - target.vertices[0];
    const Vec3 edge_d = target.vertices[3] - target.vertices[0];
    PairTensor result{};
    for (int i = 0; i < 8; ++i) {
        const double u = node[static_cast<std::size_t>(i)];
        for (int j = 0; j < 8; ++j) {
            const double v = node[static_cast<std::size_t>(j)];
            for (int k = 0; k < 8; ++k) {
                const double w = node[static_cast<std::size_t>(k)];
                const Vec3 offset = target.vertices[0] +
                    edge_b * u + edge_c * ((1.0 - u) * v) +
                    edge_d * ((1.0 - u) * (1.0 - v) * w);
                const PairTensor value = tetrahedron_point_tensor(
                    displacement + offset, source);
                const double normalised_weight = 6.0 * weight[
                    static_cast<std::size_t>(i)] * weight[
                        static_cast<std::size_t>(j)] * weight[
                            static_cast<std::size_t>(k)] * (1.0 - u) *
                    (1.0 - u) * (1.0 - v);
                result.xx += normalised_weight * value.xx;
                result.xy += normalised_weight * value.xy;
                result.xz += normalised_weight * value.xz;
                result.yy += normalised_weight * value.yy;
                result.yz += normalised_weight * value.yz;
                result.zz += normalised_weight * value.zz;
            }
        }
    }
    return result;
}

using Triangle = std::array<Vec3, 3>;

Triangle translate_triangle(const Triangle& triangle, const Vec3& shift)
{
    Triangle result = triangle;
    for (Vec3& vertex : result) {
        vertex += shift;
    }
    return result;
}

Triangle rotate_triangle_z_quarter_turn(const Triangle& triangle)
{
    Triangle result = triangle;
    for (Vec3& vertex : result) {
        vertex = {-vertex.y, vertex.x, vertex.z};
    }
    return result;
}

void require_close(const PairTensor& actual, const PairTensor& expected,
                   const double epsilon = 2.0e-11)
{
    const auto close = [epsilon](const double value, const double reference) {
        return value == Catch::Approx(reference)
            .epsilon(epsilon)
            .margin(2.0e-9);
    };
    REQUIRE(close(actual.xx, expected.xx));
    REQUIRE(close(actual.xy, expected.xy));
    REQUIRE(close(actual.xz, expected.xz));
    REQUIRE(close(actual.yy, expected.yy));
    REQUIRE(close(actual.yz, expected.yz));
    REQUIRE(close(actual.zz, expected.zz));
}

void require_tight_close(const PairTensor& actual, const PairTensor& expected,
                         const double epsilon = 5.0e-12,
                         const double margin = 5.0e-13)
{
    const auto close = [epsilon, margin](const double value,
                                         const double reference) {
        return value == Catch::Approx(reference)
            .epsilon(epsilon)
            .margin(margin);
    };
    REQUIRE(close(actual.xx, expected.xx));
    REQUIRE(close(actual.xy, expected.xy));
    REQUIRE(close(actual.xz, expected.xz));
    REQUIRE(close(actual.yy, expected.yy));
    REQUIRE(close(actual.yz, expected.yz));
    REQUIRE(close(actual.zz, expected.zz));
}

} // namespace

TEST_CASE("tetrahedron volume validation and centroided moments")
{
    const Tetrahedron tetrahedron = reference_tetrahedron();
    REQUIRE(tetrahedron_volume(tetrahedron) == Catch::Approx(1.0 / 6.0));
    REQUIRE(tetrahedron.centroid_offset().x == Catch::Approx(0.0));
    REQUIRE(tetrahedron.centroid_offset().y == Catch::Approx(0.0));
    REQUIRE(tetrahedron.centroid_offset().z == Catch::Approx(0.0));

    REQUIRE(tetrahedron_averaged_monomial({0, 0, 0}, {}, tetrahedron) ==
            Catch::Approx(1.0));
    REQUIRE(tetrahedron_averaged_monomial({1, 0, 0}, {2.0, -1.0, 0.5},
                                          tetrahedron) == Catch::Approx(2.0));
    // E[(x-E[x])^2]/2 = (3/80)/2 for the unit reference tetrahedron.
    REQUIRE(tetrahedron_averaged_monomial({2, 0, 0}, {}, tetrahedron) ==
            Catch::Approx(3.0 / 160.0));

    Tetrahedron degenerate = tetrahedron;
    degenerate.vertices[3] = degenerate.vertices[2];
    REQUIRE_THROWS_AS(tetrahedron_volume(degenerate), std::invalid_argument);
}

TEST_CASE("analytical triangle pair integral has reference invariances")
{
    const Triangle first{{{0.0, 0.0, 0.0},
                          {1.0, 0.0, 0.0},
                          {0.0, 1.0, 0.0}}};
    const Triangle second{{{0.2, -0.3, 1.4},
                           {1.1, -0.3, 1.4},
                           {0.2, 0.5, 1.4}}};
    const double expected = detail::triangle_triangle_laplace_integral(
        first, second);
    REQUIRE(detail::triangle_triangle_laplace_integral(second, first) ==
            Catch::Approx(expected).epsilon(2.0e-12));
    REQUIRE(detail::triangle_triangle_laplace_integral(
                translate_triangle(first, {4.0, -2.0, 7.0}),
                translate_triangle(second, {4.0, -2.0, 7.0})) ==
            Catch::Approx(expected).epsilon(2.0e-12));

    // Recentring must happen before scale normalization.  At this magnitude
    // the common translation is exactly representable, while the old
    // absolute-coordinate rescaling could repeatedly perturb the edge scale.
    const Vec3 large_shift{1.0e8, -1.0e8, 1.0e8};
    REQUIRE(detail::triangle_triangle_laplace_integral(
                translate_triangle(first, large_shift),
                translate_triangle(second, large_shift)) ==
            Catch::Approx(expected).epsilon(1.0e-7).margin(1.0e-10));

    const Triangle rotated_first = rotate_triangle_z_quarter_turn(first);
    const Triangle rotated_second = rotate_triangle_z_quarter_turn(second);
    REQUIRE(detail::triangle_triangle_laplace_integral(
                rotated_first, rotated_second) ==
            Catch::Approx(expected).epsilon(2.0e-12));

    const double factor = 1.7;
    Triangle scaled_first = first;
    Triangle scaled_second = second;
    for (Vec3& vertex : scaled_first) {
        vertex = vertex * factor;
    }
    for (Vec3& vertex : scaled_second) {
        vertex = vertex * factor;
    }
    REQUIRE(detail::triangle_triangle_laplace_integral(
                scaled_first, scaled_second) ==
            Catch::Approx(expected * std::pow(factor, 3.0))
                .epsilon(2.0e-11));
}

TEST_CASE("analytical triangle pair integral preserves singular configurations")
{
    const Triangle shared_edge_first{{{0.0, 0.0, 0.0},
                                      {0.0, 1.0, 0.0},
                                      {1.0, 0.0, 0.0}}};
    const Triangle shared_edge_second{{{0.0, 0.0, 0.0},
                                       {0.0, 1.0, 0.0},
                                       {-1.0, 0.0, 0.0}}};
    REQUIRE(detail::triangle_triangle_laplace_integral(
                shared_edge_first, shared_edge_second) ==
            Catch::Approx(0.4154834934268203).epsilon(2.0e-12));

    const Triangle separated_edge_second = translate_triangle(
        shared_edge_second, {0.0, 0.0, 1.0});
    REQUIRE(detail::triangle_triangle_laplace_integral(
                shared_edge_first, separated_edge_second) ==
            Catch::Approx(0.1994877345160992).epsilon(2.0e-12));

    const Triangle nearly_touching_edge_second = translate_triangle(
        shared_edge_second, {0.0, 0.0, 1.0e-4});
    REQUIRE(detail::triangle_triangle_laplace_integral(
                shared_edge_first, nearly_touching_edge_second) ==
            Catch::Approx(0.4154834087866362).epsilon(2.0e-12));

    const Triangle equilateral{{{0.0, 0.0, 0.0},
                               {1.0, 0.0, 0.0},
                               {0.5, std::sqrt(3.0) / 2.0, 0.0}}};
    REQUIRE(detail::triangle_triangle_laplace_integral(equilateral,
                                                       equilateral) ==
            Catch::Approx(0.75 * std::log(3.0)).epsilon(2.0e-12));

    const Triangle shared_vertex{{{0.0, 0.0, 0.0},
                                  {0.0, 0.0, 1.0},
                                  {0.0, -1.0, 0.0}}};
    REQUIRE(std::isfinite(detail::triangle_triangle_laplace_integral(
        shared_edge_first, shared_vertex)));

    const Triangle nearly_touching = translate_triangle(
        shared_edge_first, {0.0, 0.0, 1.0e-6});
    REQUIRE(std::isfinite(detail::triangle_triangle_laplace_integral(
        shared_edge_first, nearly_touching)));
}

TEST_CASE("analytical triangle pair integral is symmetric at contact")
{
    const Triangle separated_first{{{0.0, 0.0, 0.0},
                                    {1.0, 0.0, 0.0},
                                    {0.0, 1.0, 0.0}}};
    const Triangle separated_second{{{0.2, -0.3, 1.4},
                                     {1.1, -0.3, 1.4},
                                     {0.2, 0.5, 1.4}}};
    const Triangle near_second = translate_triangle(
        separated_second, {0.0, 0.0, -1.3999});
    const Triangle shared_edge_first{{{0.0, 0.0, 0.0},
                                      {0.0, 1.0, 0.0},
                                      {1.0, 0.0, 0.0}}};
    const Triangle shared_edge_second{{{0.0, 0.0, 0.0},
                                       {0.0, 1.0, 0.0},
                                       {-1.0, 0.0, 0.0}}};
    const Triangle shared_vertex{{{0.0, 0.0, 0.0},
                                  {0.0, 0.0, 1.0},
                                  {0.0, -1.0, 0.0}}};
    const Triangle cases_first[] = {separated_first, shared_edge_first,
                                    shared_edge_first, shared_edge_first,
                                    separated_first};
    const Triangle cases_second[] = {separated_second, near_second,
                                     shared_vertex, shared_edge_second,
                                     separated_first};

    for (std::size_t index = 0; index < std::size(cases_first); ++index) {
        const double forward = detail::triangle_triangle_laplace_integral(
            cases_first[index], cases_second[index]);
        const double reverse = detail::triangle_triangle_laplace_integral(
            cases_second[index], cases_first[index]);
        REQUIRE(std::isfinite(forward));
        REQUIRE(reverse == Catch::Approx(forward)
                             .epsilon(5.0e-12)
                             .margin(2.0e-13));
    }
}

TEST_CASE("tetrahedron point tensor has far-field and symmetry limits")
{
    const Tetrahedron tetrahedron = reference_tetrahedron();
    const PairTensor expected = point_tensor({0.0, 0.0, 30.0});
    const PairTensor actual = tetrahedron_point_tensor({0.0, 0.0, 30.0},
                                                       tetrahedron);
    require_close(actual, expected, 2.0e-5);

}

TEST_CASE("tetrahedron face orientation is permutation independent")
{
    const Tetrahedron tetrahedron = reference_tetrahedron();
    const PairTensor expected = tetrahedron_point_tensor({2.0, -1.5, 3.0},
                                                         tetrahedron);
    const std::array<int, 4> permutations[3]{
        {1, 0, 2, 3}, {2, 3, 0, 1}, {3, 1, 0, 2}};
    for (const auto& permutation : permutations) {
        Tetrahedron reordered;
        for (int i = 0; i < 4; ++i) {
            reordered.vertices[static_cast<std::size_t>(i)] =
                tetrahedron.vertices[static_cast<std::size_t>(
                    permutation[static_cast<std::size_t>(i)])];
        }
        require_close(tetrahedron_point_tensor({2.0, -1.5, 3.0}, reordered),
                      expected, 2.0e-11);
    }
}

TEST_CASE("point to tetrahedron uses reciprocal exact tensor")
{
    const Tetrahedron tetrahedron = reference_tetrahedron();
    const Vec3 displacement{1.3, -0.9, 2.4};
    require_close(point_tetrahedron_tensor(displacement, tetrahedron),
                  tetrahedron_point_tensor(displacement * -1.0, tetrahedron),
                  1.0e-13);
}

TEST_CASE("exact tetrahedron pair retains finite self interaction and trace")
{
    const Tetrahedron tetrahedron = reference_tetrahedron();
    const PairTensor tensor = tetrahedron_tetrahedron_tensor(
        {}, tetrahedron, tetrahedron);
    REQUIRE(std::isfinite(tensor.xx));
    REQUIRE(std::isfinite(tensor.yy));
    REQUIRE(std::isfinite(tensor.zz));
    REQUIRE(tensor.xx + tensor.yy + tensor.zz == Catch::Approx(-6.0)
                .margin(2.0e-10));
}

TEST_CASE("exact tetrahedron pair obeys reciprocity, rotation, and scaling")
{
    const Tetrahedron source = reference_tetrahedron();
    const Tetrahedron target = scale_tetrahedron(source, 0.8);
    const Vec3 displacement{2.2, -1.4, 1.7};
    const PairTensor forward = tetrahedron_tetrahedron_tensor(
        displacement, source, target);
    const PairTensor reverse = tetrahedron_tetrahedron_tensor(
        displacement * -1.0, target, source);
    require_close(forward, reverse, 2.0e-11);

    const Vec3 rotated_displacement{-displacement.y, displacement.x,
                                    displacement.z};
    Tetrahedron rotated_source = scale_tetrahedron(source, 1.0);
    Tetrahedron rotated_target = scale_tetrahedron(target, 1.0);
    for (int vertex = 0; vertex < 4; ++vertex) {
        const Vec3 source_vertex = rotated_source.vertices[
            static_cast<std::size_t>(vertex)];
        rotated_source.vertices[static_cast<std::size_t>(vertex)] = {
            -source_vertex.y, source_vertex.x, source_vertex.z};
        const Vec3 target_vertex = rotated_target.vertices[
            static_cast<std::size_t>(vertex)];
        rotated_target.vertices[static_cast<std::size_t>(vertex)] = {
            -target_vertex.y, target_vertex.x, target_vertex.z};
    }
    const PairTensor rotated = tetrahedron_tetrahedron_tensor(
        rotated_displacement, rotated_source, rotated_target);
    require_close(rotated, rotate_z_quarter_turn(forward), 2.0e-11);

    const double factor = 1.7;
    const PairTensor scaled = tetrahedron_tetrahedron_tensor(
        displacement * factor, scale_tetrahedron(source, factor),
        scale_tetrahedron(target, factor));
    const PairTensor expected{
        forward.xx / std::pow(factor, 3.0),
        forward.xy / std::pow(factor, 3.0),
        forward.xz / std::pow(factor, 3.0),
        forward.yy / std::pow(factor, 3.0),
        forward.yz / std::pow(factor, 3.0),
        forward.zz / std::pow(factor, 3.0)};
    require_close(scaled, expected, 2.0e-10);
}

TEST_CASE("exact tetrahedron pair obeys reciprocity for irregular geometry")
{
    const Tetrahedron source = centred_physical_tetrahedron({{
        {-0.3, 0.2, 0.4},
        {1.7, -0.5, 0.1},
        {0.4, 1.2, -0.6},
        {0.1, 0.3, 2.1},
    }});
    const Tetrahedron target = centred_physical_tetrahedron({{
        {-1.1, 0.4, 0.2},
        {0.8, -0.7, 0.6},
        {0.2, 1.9, -0.3},
        {1.4, 0.1, 1.7},
    }});
    const Vec3 displacement{2.35, -1.15, 1.8};

    const PairTensor forward = tetrahedron_tetrahedron_tensor(
        displacement, source, target);
    const PairTensor reverse = tetrahedron_tetrahedron_tensor(
        displacement * -1.0, target, source);
    require_tight_close(forward, reverse);
}

TEST_CASE("prepared coincident tetrahedron path matches generic face pairs")
{
    const Tetrahedron regular = centred_physical_tetrahedron({{
        {1.0, 1.0, 1.0},
        {1.0, -1.0, -1.0},
        {-1.0, 1.0, -1.0},
        {-1.0, -1.0, 1.0},
    }});
    const Tetrahedron irregular = centred_physical_tetrahedron({{
        {-0.3, 0.2, 0.4},
        {1.7, -0.5, 0.1},
        {0.4, 1.2, -0.6},
        {0.1, 0.3, 2.1},
    }});
    const Tetrahedron anisotropic = centred_physical_tetrahedron({{
        {1.7, 0.8, 2.3},
        {1.7, -0.8, -2.3},
        {-1.7, 0.8, -2.3},
        {-1.7, -0.8, 2.3},
    }});

    const std::array<Tetrahedron, 3> tetrahedra{{
        regular, irregular, anisotropic}};
    for (const Tetrahedron& tetrahedron : tetrahedra) {
        const detail::PreparedTetrahedron prepared =
            detail::prepare_tetrahedron(tetrahedron);
        const PairTensor ten_integrals =
            detail::tetrahedron_tetrahedron_tensor_prepared(
                {}, prepared, prepared, true);
        const PairTensor sixteen_integrals =
            detail::tetrahedron_tetrahedron_tensor_prepared(
                {}, prepared, prepared, false);
        require_tight_close(ten_integrals, sixteen_integrals);
        require_tight_close(ten_integrals,
                            tetrahedron_tetrahedron_tensor(
                                {}, tetrahedron, tetrahedron));
    }
}

TEST_CASE("exact tetrahedron pair handles touching and shrinking limits")
{
    const Tetrahedron source = centred_physical_tetrahedron({{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }});
    const Tetrahedron shared_face = centred_physical_tetrahedron({{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, -1.0},
    }});
    const Vec3 face_displacement{0.0, 0.0, -0.5};
    const PairTensor touching_face = tetrahedron_tetrahedron_tensor(
        face_displacement, source, shared_face);
    require_finite(touching_face);

    const Vec3 common_shift{7.0, -3.0, 5.0};
    Tetrahedron shifted_source = source;
    Tetrahedron shifted_face = shared_face;
    for (int vertex = 0; vertex < 4; ++vertex) {
        shifted_source.vertices[static_cast<std::size_t>(vertex)] +=
            common_shift;
        shifted_face.vertices[static_cast<std::size_t>(vertex)] +=
            common_shift;
    }
    require_close(tetrahedron_tetrahedron_tensor(
                      face_displacement, shifted_source, shifted_face),
                  touching_face, 2.0e-11);

    const Tetrahedron shared_edge = centred_physical_tetrahedron({{
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, -1.0, 0.0},
        {0.0, 0.0, -1.0},
    }});
    const Vec3 edge_displacement{0.25, -0.5, -0.5};
    require_finite(tetrahedron_tetrahedron_tensor(
        edge_displacement, source, shared_edge));

    const Tetrahedron shared_vertex = centred_physical_tetrahedron({{
        {0.0, 0.0, 0.0},
        {-1.0, 0.0, 0.0},
        {0.0, -1.0, 0.0},
        {0.0, 0.0, -1.0},
    }});
    const Vec3 vertex_displacement{-0.5, -0.5, -0.5};
    require_finite(tetrahedron_tetrahedron_tensor(
        vertex_displacement, source, shared_vertex));

    const PairTensor disjoint = tetrahedron_tetrahedron_tensor(
        {0.0, 0.0, 4.0}, source, source);
    REQUIRE(disjoint.xx + disjoint.yy + disjoint.zz == Catch::Approx(0.0)
                .margin(2.0e-6));

    const Vec3 separated{1.7, -0.8, 2.1};
    const PairTensor target_limit = tetrahedron_point_tensor(
        separated, source);
    double previous_target_error = std::numeric_limits<double>::infinity();
    for (const double factor : {0.5, 0.25, 0.125}) {
        const PairTensor shrinking_target = tetrahedron_tetrahedron_tensor(
            separated, source, scale_tetrahedron(source, factor));
        const double error = maximum_component_difference(
            shrinking_target, target_limit);
        REQUIRE(error < previous_target_error);
        previous_target_error = error;
    }

    const PairTensor source_limit = point_tetrahedron_tensor(
        separated, source);
    double previous_source_error = std::numeric_limits<double>::infinity();
    for (const double factor : {0.5, 0.25, 0.125}) {
        const PairTensor shrinking_source = tetrahedron_tetrahedron_tensor(
            separated, scale_tetrahedron(source, factor), source);
        const double error = maximum_component_difference(
            shrinking_source, source_limit);
        REQUIRE(error < previous_source_error);
        previous_source_error = error;
    }
}

TEST_CASE("exact tetrahedron pair agrees with independent target quadrature")
{
    const Tetrahedron tetrahedron = reference_tetrahedron();
    const Vec3 displacement{2.8, -1.9, 3.4};
    const PairTensor exact = tetrahedron_tetrahedron_tensor(
        displacement, tetrahedron, tetrahedron);
    const PairTensor quadrature = average_point_tensor_over_target(
        displacement, tetrahedron, tetrahedron);
    require_close(exact, quadrature, 2.0e-7);
}

namespace {

// Eight-point Gauss-Legendre rule on [0, 1], shared by the independent
// target-average references below.
constexpr std::array<double, 8> gauss8_node{{
    0.0198550717512319, 0.1016667612931866,
    0.2372337950418355, 0.4082826787521751,
    0.5917173212478249, 0.7627662049581645,
    0.8983332387068134, 0.9801449282487681}};
constexpr std::array<double, 8> gauss8_weight{{
    0.0506142681451881, 0.1111905172266872,
    0.1568533229389436, 0.1813418916891809,
    0.1813418916891809, 0.1568533229389436,
    0.1111905172266872, 0.0506142681451881}};

void accumulate_scaled(PairTensor& sum, const PairTensor& value,
                       const double weight)
{
    sum.xx += weight * value.xx;
    sum.xy += weight * value.xy;
    sum.xz += weight * value.xz;
    sum.yy += weight * value.yy;
    sum.yz += weight * value.yz;
    sum.zz += weight * value.zz;
}

// Volume average of a point-evaluated source tensor over a tetrahedron
// target, using the Duffy-type collapsed cube.
template <typename PointTensor>
PairTensor average_over_tetrahedron(const Vec3& displacement,
                                    const Tetrahedron& target,
                                    PointTensor&& point_tensor)
{
    const Vec3 edge_b = target.vertices[1] - target.vertices[0];
    const Vec3 edge_c = target.vertices[2] - target.vertices[0];
    const Vec3 edge_d = target.vertices[3] - target.vertices[0];
    PairTensor result{};
    for (int i = 0; i < 8; ++i) {
        const double u = gauss8_node[static_cast<std::size_t>(i)];
        for (int j = 0; j < 8; ++j) {
            const double v = gauss8_node[static_cast<std::size_t>(j)];
            for (int k = 0; k < 8; ++k) {
                const double w = gauss8_node[static_cast<std::size_t>(k)];
                const Vec3 offset = target.vertices[0] + edge_b * u +
                    edge_c * ((1.0 - u) * v) +
                    edge_d * ((1.0 - u) * (1.0 - v) * w);
                const double weight = 6.0 *
                    gauss8_weight[static_cast<std::size_t>(i)] *
                    gauss8_weight[static_cast<std::size_t>(j)] *
                    gauss8_weight[static_cast<std::size_t>(k)] *
                    (1.0 - u) * (1.0 - u) * (1.0 - v);
                accumulate_scaled(result, point_tensor(displacement + offset),
                                  weight);
            }
        }
    }
    return result;
}

// Volume average of a point-evaluated source tensor over a prism target.
template <typename PointTensor>
PairTensor average_over_prism(const Vec3& displacement,
                              const RectangularPrism& target,
                              PointTensor&& point_tensor)
{
    PairTensor result{};
    for (int i = 0; i < 8; ++i) {
        const double x = (gauss8_node[static_cast<std::size_t>(i)] - 0.5) *
            target.hx;
        for (int j = 0; j < 8; ++j) {
            const double y = (gauss8_node[static_cast<std::size_t>(j)] - 0.5) *
                target.hy;
            for (int k = 0; k < 8; ++k) {
                const double z =
                    (gauss8_node[static_cast<std::size_t>(k)] - 0.5) *
                    target.hz;
                const double weight =
                    gauss8_weight[static_cast<std::size_t>(i)] *
                    gauss8_weight[static_cast<std::size_t>(j)] *
                    gauss8_weight[static_cast<std::size_t>(k)];
                accumulate_scaled(
                    result, point_tensor(displacement + Vec3{x, y, z}),
                    weight);
            }
        }
    }
    return result;
}

// Kuhn decomposition of an axis-aligned prism into six tetrahedra of equal
// volume, each returned relative to its own centroid together with that
// centroid.
struct PrismTetrahedron {
    Tetrahedron tetrahedron{};
    Vec3 centroid{};
};

std::array<PrismTetrahedron, 6> kuhn_decomposition(
    const RectangularPrism& prism)
{
    const std::array<double, 3> half{{0.5 * prism.hx, 0.5 * prism.hy,
                                      0.5 * prism.hz}};
    const std::array<std::array<int, 3>, 6> permutations{{
        {{0, 1, 2}}, {{0, 2, 1}}, {{1, 0, 2}},
        {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}}}};
    std::array<PrismTetrahedron, 6> result{};
    for (std::size_t index = 0; index < 6; ++index) {
        std::array<std::array<double, 3>, 4> corners{};
        corners[0] = {{-half[0], -half[1], -half[2]}};
        std::array<double, 3> current = corners[0];
        for (std::size_t step = 0; step < 2; ++step) {
            const std::size_t axis =
                static_cast<std::size_t>(permutations[index][step]);
            current[axis] += 2.0 * half[axis];
            corners[step + 1] = current;
        }
        corners[3] = {{half[0], half[1], half[2]}};
        std::array<Vec3, 4> vertices{};
        Vec3 centroid{};
        for (std::size_t vertex = 0; vertex < 4; ++vertex) {
            vertices[vertex] = {corners[vertex][0], corners[vertex][1],
                                corners[vertex][2]};
            centroid += vertices[vertex];
        }
        centroid = centroid * 0.25;
        for (Vec3& vertex : vertices) {
            vertex = vertex - centroid;
        }
        result[index] = {Tetrahedron{vertices}, centroid};
    }
    return result;
}

} // namespace

TEST_CASE("exact prism tetrahedron pair agrees with independent target quadrature")
{
    const RectangularPrism prism{0.6, 0.4, 0.5};
    const Tetrahedron tetrahedron = scale_tetrahedron(reference_tetrahedron(),
                                                      0.8);
    const Vec3 displacement{1.3, -0.9, 1.1};
    const PairTensor exact = rectangular_prism_tetrahedron_tensor(
        displacement, prism, tetrahedron);
    const PairTensor quadrature = average_over_tetrahedron(
        displacement, tetrahedron, [&](const Vec3& point) {
            return rectangular_prism_point_tensor(point, prism);
        });
    require_close(exact, quadrature, 2.0e-7);
}

TEST_CASE("exact tetrahedron prism pair agrees with independent target quadrature")
{
    const RectangularPrism prism{0.5, 0.7, 0.35};
    const Tetrahedron tetrahedron = centred_physical_tetrahedron({{
        {-0.3, 0.2, 0.4},
        {0.7, -0.5, 0.1},
        {0.4, 0.9, -0.6},
        {0.1, 0.3, 0.8},
    }});
    const Vec3 displacement{-1.2, 1.4, 0.9};
    const PairTensor exact = tetrahedron_rectangular_prism_tensor(
        displacement, tetrahedron, prism);
    const PairTensor quadrature = average_over_prism(
        displacement, prism, [&](const Vec3& point) {
            return tetrahedron_point_tensor(point, tetrahedron);
        });
    require_close(exact, quadrature, 2.0e-7);
}

TEST_CASE("prism tetrahedron pair obeys reciprocity and the prism decomposition")
{
    const RectangularPrism prism{0.6, 0.4, 0.5};
    const Tetrahedron tetrahedron = centred_physical_tetrahedron({{
        {-0.3, 0.2, 0.4},
        {0.7, -0.5, 0.1},
        {0.4, 0.9, -0.6},
        {0.1, 0.3, 0.8},
    }});
    const Vec3 displacement{0.9, -0.7, 0.6};
    const PairTensor forward = rectangular_prism_tetrahedron_tensor(
        displacement, prism, tetrahedron);
    const PairTensor reverse = tetrahedron_rectangular_prism_tensor(
        displacement * -1.0, tetrahedron, prism);
    require_tight_close(forward, reverse);

    // The prism is the union of six tetrahedra carrying one sixth of its
    // total moment each, so the prism pair is the equal-weight sum of the
    // already validated tetrahedron pairs.
    PairTensor decomposed{};
    for (const PrismTetrahedron& piece : kuhn_decomposition(prism)) {
        REQUIRE(tetrahedron_volume(piece.tetrahedron) ==
                Catch::Approx(prism.volume() / 6.0));
        accumulate_scaled(
            decomposed,
            tetrahedron_tetrahedron_tensor(displacement - piece.centroid,
                                           piece.tetrahedron, tetrahedron),
            1.0 / 6.0);
    }
    require_close(forward, decomposed, 2.0e-10);

    // Touching bodies (a tetrahedron vertex on the prism face) stay finite.
    const Vec3 touching{0.5 * prism.hx - tetrahedron.vertices[1].x, 0.0, 0.0};
    require_finite(rectangular_prism_tetrahedron_tensor(
        touching, prism, tetrahedron));
}

TEST_CASE("polyhedron surface formulation reproduces the MagTense prism pair")
{
    const RectangularPrism source{0.6, 0.4, 0.5};
    const RectangularPrism target{0.3, 0.45, 0.25};
    const detail::PolyhedronSurface source_surface =
        detail::prepare_rectangular_prism_surface(source);
    const detail::PolyhedronSurface target_surface =
        detail::prepare_rectangular_prism_surface(target);
    REQUIRE(source_surface.faces.size() == 12);
    REQUIRE(source_surface.volume == Catch::Approx(source.volume()));
    const auto surface_pair = [&](const Vec3& displacement,
                                  const detail::PolyhedronSurface& s,
                                  const detail::PolyhedronSurface& t) {
        return detail::polyhedron_polyhedron_tensor(
            displacement, s.faces, s.outward_normals, s.volume, t.faces,
            t.outward_normals, t.volume);
    };

    const Vec3 separated{1.3, -0.7, 0.4};
    require_close(surface_pair(separated, source_surface, target_surface),
                  rectangular_prism_rectangular_prism_tensor(
                      separated, source, target), 2.0e-10);

    // Shared face between the two prisms.
    const Vec3 touching{0.5 * (source.hx + target.hx), 0.05, -0.02};
    require_close(surface_pair(touching, source_surface, target_surface),
                  rectangular_prism_rectangular_prism_tensor(
                      touching, source, target), 2.0e-10);

    // Coincident identical prisms: the finite self-demagnetisation tensor,
    // whose trace is -1/V in the total-moment normalisation (tr N = 1).
    const PairTensor self = surface_pair({}, source_surface, source_surface);
    require_close(self, rectangular_prism_rectangular_prism_tensor(
                      {}, source, source), 2.0e-9);
    REQUIRE(self.xx + self.yy + self.zz ==
            Catch::Approx(-1.0 / source.volume()).epsilon(1.0e-9));
}

TEST_CASE("prism tetrahedron pair has far-field and shrinking limits")
{
    const RectangularPrism prism{0.6, 0.4, 0.5};
    const Tetrahedron tetrahedron = scale_tetrahedron(reference_tetrahedron(),
                                                      0.8);
    const Vec3 far{60.0, 40.0, -50.0};
    const PairTensor far_tensor = rectangular_prism_tetrahedron_tensor(
        far, prism, tetrahedron);
    const PairTensor point = point_tensor(far);
    const double scale = std::abs(point.xx) + std::abs(point.yy) +
        std::abs(point.zz);
    // Both bodies are within a fraction of a unit of their representatives,
    // so the finite-size correction at |r| ~ 88 is O((size/r)^2) ~ 1e-4.
    REQUIRE(maximum_component_difference(far_tensor, point) < 1.0e-3 * scale);
    REQUIRE(maximum_component_difference(far_tensor, point) > 1.0e-7 * scale);

    // A shrinking prism source converges to the point-to-tetrahedron tensor.
    const Vec3 separated{1.4, -0.9, 1.1};
    const PairTensor limit = point_tetrahedron_tensor(separated, tetrahedron);
    double previous_error = std::numeric_limits<double>::infinity();
    for (const double factor : {0.5, 0.25, 0.125}) {
        const RectangularPrism shrunk{factor * prism.hx, factor * prism.hy,
                                      factor * prism.hz};
        const double error = maximum_component_difference(
            rectangular_prism_tetrahedron_tensor(separated, shrunk,
                                                 tetrahedron), limit);
        REQUIRE(error < previous_error);
        previous_error = error;
    }
}

TEST_CASE("polyhedron pair tensors stay accurate across the far-separation switch")
{
    // The analytical surface integrals lose relative accuracy when the
    // separation greatly exceeds the face size; beyond
    // `polyhedron_far_separation_factor` summed circumradii the exact source
    // field is averaged over the target instead.  Both branches must agree
    // with the independent 8^3 quadrature on either side of the switch.
    const RectangularPrism prism{0.6, 0.4, 0.5};
    const Tetrahedron tetrahedron = reference_tetrahedron();
    const detail::PolyhedronBody prism_body =
        detail::prepare_polyhedron_body(prism);
    const detail::PolyhedronBody tetrahedron_body =
        detail::prepare_polyhedron_body(tetrahedron);
    const double radii = prism_body.surface.circumradius +
        tetrahedron_body.surface.circumradius;
    REQUIRE(prism_body.surface.circumradius ==
            Catch::Approx(0.5 * std::sqrt(0.36 + 0.16 + 0.25)));
    const Vec3 direction{0.6, -0.4, 0.5};
    const Vec3 unit = direction * (1.0 / std::sqrt(dot(direction, direction)));
    for (const double factor : {0.9, 1.1, 3.0, 30.0}) {
        const Vec3 displacement =
            unit * (factor * detail::polyhedron_far_separation_factor * radii);
        const PairTensor prism_tet = rectangular_prism_tetrahedron_tensor(
            displacement, prism, tetrahedron);
        const PairTensor prism_tet_quadrature = average_over_tetrahedron(
            displacement, tetrahedron, [&](const Vec3& point) {
                return rectangular_prism_point_tensor(point, prism);
            });
        require_close(prism_tet, prism_tet_quadrature, 1.0e-8);

        const PairTensor tet_prism = tetrahedron_rectangular_prism_tensor(
            displacement, tetrahedron, prism);
        const PairTensor tet_prism_quadrature = average_over_prism(
            displacement, prism, [&](const Vec3& point) {
                return tetrahedron_point_tensor(point, tetrahedron);
            });
        require_close(tet_prism, tet_prism_quadrature, 1.0e-8);

        const PairTensor tet_tet = tetrahedron_tetrahedron_tensor(
            displacement, tetrahedron, tetrahedron);
        const PairTensor tet_tet_quadrature = average_over_tetrahedron(
            displacement, tetrahedron, [&](const Vec3& point) {
                return tetrahedron_point_tensor(point, tetrahedron);
            });
        require_close(tet_tet, tet_tet_quadrature, 1.0e-8);
    }

    // Far beyond the switch the analytical surface sum alone has no valid
    // digits, while the body-level tensor stays close to the point limit.
    const Vec3 remote = unit * 150.0;
    const PairTensor remote_tensor = tetrahedron_tetrahedron_tensor(
        remote, tetrahedron, tetrahedron);
    const PairTensor remote_point = point_tensor(remote);
    const double remote_scale = std::abs(remote_point.xx) +
        std::abs(remote_point.yy) + std::abs(remote_point.zz);
    REQUIRE(maximum_component_difference(remote_tensor, remote_point) <
            1.0e-4 * remote_scale);
}

TEST_CASE("tetrahedron point tensor is continuous across an edge-line extension")
{
    // A point collinear with a tetrahedron edge but outside the segment is a
    // regular exterior point; the atanh edge primitives diverge individually
    // there and their naive difference lost every digit.  The value on the
    // line must match the values a little off the line from both sides.
    Tetrahedron tetrahedron;
    const std::array<Vec3, 4> base{{{-0.25, -0.25, -0.25},
                                    {0.75, -0.25, -0.25},
                                    {-0.25, 0.75, -0.25},
                                    {-0.25, -0.25, 0.75}}};
    for (std::size_t vertex = 0; vertex < 4; ++vertex) {
        const Vec3 v = base[vertex] * 0.04;
        tetrahedron.vertices[vertex] = {v.z, v.x, v.y};
    }
    // The edge from (-0.01,-0.01,-0.01) to (0.03,-0.01,-0.01) extends along x.
    const Vec3 on_line{-0.02, -0.01, -0.01};
    const PairTensor exact = tetrahedron_point_tensor(on_line, tetrahedron);
    require_finite(exact);
    for (const double offset : {1.0e-12, 1.0e-9, 1.0e-6}) {
        for (const Vec3 shift : {Vec3{0.0, offset, 0.0}, Vec3{0.0, -offset, 0.0},
                                 Vec3{0.0, 0.0, offset}, Vec3{0.0, offset, -offset}}) {
            const PairTensor nearby =
                tetrahedron_point_tensor(on_line + shift, tetrahedron);
            // The field varies smoothly at 1e-6 (relative change ~5e-5 per
            // 1e-6 of displacement at this distance), so compare loosely
            // there and tightly closer in.
            const double tolerance = offset >= 1.0e-6 ? 2.0e-4 : 1.0e-7;
            REQUIRE(maximum_component_difference(nearby, exact) <
                    tolerance * 1.0e4);
        }
    }
    // Reciprocity of the corrected kernel with the finite pair formulation.
    const PairTensor reciprocal = point_tetrahedron_tensor(on_line * -1.0,
                                                           tetrahedron);
    require_tight_close(reciprocal, exact);

    // Independent check farther out on the same edge line, where the 8^3
    // Gauss average of the point-dipole tensor over the tetrahedron is
    // converged: by reciprocity it equals the tetrahedron's field at the
    // point.
    const Vec3 far_on_line{-0.11, -0.01, -0.01};
    const PairTensor far_exact = tetrahedron_point_tensor(far_on_line,
                                                          tetrahedron);
    const PairTensor quadrature = average_over_tetrahedron(
        far_on_line * -1.0, tetrahedron, [&](const Vec3& point) {
            return point_tensor(point);
        });
    require_close(far_exact, quadrature, 2.0e-6);
    const PairTensor far_nearby = tetrahedron_point_tensor(
        far_on_line + Vec3{0.0, 1.0e-9, -1.0e-9}, tetrahedron);
    REQUIRE(maximum_component_difference(far_nearby, far_exact) <
            1.0e-7 * std::abs(far_exact.xx));
}

TEST_CASE("exact tetrahedron pair is continuous as a shared edge stops coinciding",
          "[tetrahedron][!shouldfail]")
{
    // FIXME(cdfmm): two tetrahedra of a conforming mesh that share an edge
    // are evaluated correctly while the edge coincides exactly (checked
    // against the source's point field averaged over the target, which does
    // not use this kernel). Moving one shared vertex off the edge by 3e-14 to
    // 1e-5 changes the tensor by up to 73 % -- and from 1e-6 to 1e-5 by
    // factors of hundreds -- where the physical field moves by roughly the
    // skew itself; from 1e-4 on it agrees again. Two branches of the
    // triangle-triangle recursion are responsible: below the Gram-Schmidt
    // rank tolerance (sqrt(256 eps) relative) a direction is dropped while its
    // residual height, snapped only below 256 eps absolute, is kept; above it
    // the full-rank closed forms cancel catastrophically. The FMM's coordinate
    // normalisation creates such ulp-level mismatches on a conforming mesh
    // whose coordinates it cannot scale exactly, which is how Article1's
    // irregular Kuhn mesh found it. Expected to fail until the recursion
    // handles near-degenerate pairs; Catch2 then reports the tag as stale.
    //
    // Bodies 363 (source) and 359 (target) of Article1's
    // tetra_mesh_irregular_8, as offsets from their representatives.
    const Tetrahedron source{{{{-0.25, -0.75, -0.5},
                               {-0.25, 0.25, -0.5},
                               {-0.25, 0.25, 0.5},
                               {0.75, 0.25, 0.5}}}};
    const Tetrahedron target{{{{-0.25, -0.5, -0.75},
                               {-0.25, -0.5, 0.25},
                               {-0.25, 0.5, 0.25},
                               {0.75, 0.5, 0.25}}}};
    const Vec3 displacement{0.0, -0.25, -0.75};
    const PairTensor shared =
        tetrahedron_tetrahedron_tensor(displacement, source, target);
    const double scale = std::sqrt(
        shared.xx * shared.xx + shared.yy * shared.yy + shared.zz * shared.zz +
        2.0 * (shared.xy * shared.xy + shared.xz * shared.xz +
               shared.yz * shared.yz));
    const double length = std::sqrt(0.6 * 0.6 + 0.3 * 0.3 + 0.74 * 0.74);
    const Vec3 direction = Vec3{0.6, -0.3, 0.74} * (1.0 / length);
    for (const double skew : {1.0e-13, 1.0e-10, 1.0e-7, 1.0e-6, 1.0e-5}) {
        INFO("skew " << skew);
        Tetrahedron moved = source;
        moved.vertices[0] = moved.vertices[0] + direction * skew;
        const PairTensor value =
            tetrahedron_tetrahedron_tensor(displacement, moved, target);
        // A skew of delta moves the tensor by O(delta log delta).
        CHECK(maximum_component_difference(value, shared) <= 1.0e-3 * scale);
    }
}
