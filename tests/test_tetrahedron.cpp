// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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
