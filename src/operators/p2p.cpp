// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/operators/p2p.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace cdfmm {
namespace {

StaticP2POperator build_static_p2p_operator_impl(
    const std::span<const Vec3> target_positions,
    const std::span<const Vec3> source_positions,
    const std::span<const StaticP2PInteraction> interactions,
    const SourceGeometry source_geometry,
    const std::span<const RectangularPrism> source_prisms,
    const std::span<const Tetrahedron> source_tetrahedra,
    const TargetGeometry target_geometry,
    const std::span<const RectangularPrism> target_prisms,
    const std::span<const Tetrahedron> target_tetrahedra,
    const SourceModel source_model,
    const TargetModel target_model)
{
    const SourceGeometry effective_source_geometry =
        source_model == SourceModel::ExactGeometry
            ? source_geometry : SourceGeometry::PointDipole;
    const TargetGeometry effective_target_geometry =
        target_model == TargetModel::ExactGeometry
            ? target_geometry : TargetGeometry::Point;

    const bool source_is_prism =
        effective_source_geometry == SourceGeometry::RectangularPrism;
    const bool source_is_tetrahedron =
        effective_source_geometry == SourceGeometry::Tetrahedron;
    const bool target_is_prism =
        effective_target_geometry == TargetGeometry::RectangularPrism;
    const bool target_is_tetrahedron =
        effective_target_geometry == TargetGeometry::Tetrahedron;

    if (source_is_prism && target_is_tetrahedron) {
        throw std::invalid_argument(
            "exact rectangular-prism to tetrahedron P2P is unsupported");
    }
    if (source_is_tetrahedron && target_is_prism) {
        throw std::invalid_argument(
            "exact tetrahedron to rectangular-prism P2P is unsupported");
    }

    const auto validate_count = [](const std::size_t geometry_count,
                                   const std::size_t object_count,
                                   const char* message) {
        if (geometry_count != 1 && geometry_count != object_count) {
            throw std::invalid_argument(message);
        }
    };
    if (source_is_prism) {
        validate_count(source_prisms.size(), source_positions.size(),
                       "static prism P2P sizes must be common or per source");
        for (const RectangularPrism& prism : source_prisms) {
            if (!(prism.hx > 0.0 && prism.hy > 0.0 && prism.hz > 0.0)) {
                throw std::invalid_argument(
                    "static source prism dimensions must be positive");
            }
        }
    }
    if (source_is_tetrahedron) {
        validate_count(
            source_tetrahedra.size(), source_positions.size(),
            "static tetrahedron P2P geometries must be common or per source");
        for (const Tetrahedron& tetrahedron : source_tetrahedra) {
            static_cast<void>(tetrahedron_volume(tetrahedron));
        }
    }
    if (target_is_prism) {
        validate_count(target_prisms.size(), target_positions.size(),
                       "static prism P2P sizes must be common or per target");
        for (const RectangularPrism& prism : target_prisms) {
            if (!(prism.hx > 0.0 && prism.hy > 0.0 && prism.hz > 0.0)) {
                throw std::invalid_argument(
                    "static target prism dimensions must be positive");
            }
        }
    }
    if (target_is_tetrahedron) {
        validate_count(
            target_tetrahedra.size(), target_positions.size(),
            "static tetrahedron P2P geometries must be common or per target");
        for (const Tetrahedron& tetrahedron : target_tetrahedra) {
            static_cast<void>(tetrahedron_volume(tetrahedron));
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
    for (std::size_t index = 0; index < sorted.size(); ++index) {
        const StaticP2PInteraction& interaction = sorted[index];
        const int target = interaction.target;
        const int source = interaction.source;
        const Vec3 shifted_source =
            source_positions[static_cast<std::size_t>(source)] +
            interaction.source_shift;
        const Vec3 displacement =
            target_positions[static_cast<std::size_t>(target)] -
            shifted_source;
        const double radius_squared = dot(displacement, displacement);

        if (effective_source_geometry == SourceGeometry::PointDipole &&
            effective_target_geometry == TargetGeometry::Point &&
            radius_squared == 0.0) {
            const double undefined =
                std::numeric_limits<double>::quiet_NaN();
            result.blocks[index] = {
                target, source, undefined, undefined, undefined,
                undefined, undefined, undefined, undefined, undefined,
                undefined,
                (interaction.skip_for_identity &&
                 effective_source_geometry == SourceGeometry::PointDipole)
                    ? 1 : 0};
            continue;
        }

        const RectangularPrism source_prism = source_is_prism
            ? source_prisms[source_prisms.size() == 1 ? 0 : source]
            : RectangularPrism{};
        const RectangularPrism target_prism = target_is_prism
            ? target_prisms[target_prisms.size() == 1 ? 0 : target]
            : RectangularPrism{};
        const Tetrahedron* source_tetrahedron = source_is_tetrahedron
            ? &source_tetrahedra[
                  source_tetrahedra.size() == 1 ? 0 : source]
            : nullptr;
        const Tetrahedron* target_tetrahedron = target_is_tetrahedron
            ? &target_tetrahedra[
                  target_tetrahedra.size() == 1 ? 0 : target]
            : nullptr;

        PairTensor tensor;
        if (source_is_tetrahedron && target_is_tetrahedron) {
            tensor = tetrahedron_tetrahedron_tensor(
                displacement, *source_tetrahedron, *target_tetrahedron);
        } else if (source_is_tetrahedron) {
            tensor = tetrahedron_point_tensor(
                displacement, *source_tetrahedron);
        } else if (target_is_tetrahedron) {
            tensor = point_tetrahedron_tensor(
                displacement, *target_tetrahedron);
        } else {
            tensor = build_pair_tensor(
                target_positions[static_cast<std::size_t>(target)],
                shifted_source, effective_source_geometry,
                effective_target_geometry, source_prism, target_prism);
        }

        const double potential_scale = radius_squared == 0.0
            ? 0.0
            : 1.0 / (4.0 * std::numbers::pi * radius_squared *
                     std::sqrt(radius_squared));
        result.blocks[index] = {
            target, source, potential_scale * displacement.x,
            potential_scale * displacement.y,
            potential_scale * displacement.z, tensor.xx, tensor.xy,
            tensor.xz, tensor.yy, tensor.yz, tensor.zz,
            (interaction.skip_for_identity &&
             effective_source_geometry == SourceGeometry::PointDipole)
                ? 1 : 0};
    }
    return result;
}

} // namespace

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
        source_geometry, source_sizes, {}, target_geometry, target_sizes, {},
        SourceModel::ExactGeometry, TargetModel::ExactGeometry);
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
    return build_static_p2p_operator(
        target_positions, source_positions, interactions, source_geometry,
        source_sizes, {}, target_geometry, target_sizes, {},
        SourceModel::ExactGeometry, TargetModel::ExactGeometry);
}

StaticP2POperator build_static_p2p_operator(
    const std::span<const Vec3> target_positions,
    const std::span<const Vec3> source_positions,
    const std::span<const std::array<int, 2>> interactions,
    const SourceGeometry source_geometry,
    const std::span<const RectangularPrism> source_prisms,
    const std::span<const Tetrahedron> source_tetrahedra,
    const TargetGeometry target_geometry,
    const std::span<const RectangularPrism> target_prisms,
    const std::span<const Tetrahedron> target_tetrahedra,
    const SourceModel source_model,
    const TargetModel target_model)
{
    std::vector<StaticP2PInteraction> image_interactions;
    image_interactions.reserve(interactions.size());
    for (const std::array<int, 2> pair : interactions) {
        image_interactions.push_back({pair[0], pair[1], {}, true});
    }
    return build_static_p2p_operator(
        target_positions, source_positions, image_interactions,
        source_geometry, source_prisms, source_tetrahedra, target_geometry,
        target_prisms, target_tetrahedra, source_model, target_model);
}

StaticP2POperator build_static_p2p_operator(
    const std::span<const Vec3> target_positions,
    const std::span<const Vec3> source_positions,
    const std::span<const StaticP2PInteraction> interactions,
    const SourceGeometry source_geometry,
    const std::span<const RectangularPrism> source_prisms,
    const std::span<const Tetrahedron> source_tetrahedra,
    const TargetGeometry target_geometry,
    const std::span<const RectangularPrism> target_prisms,
    const std::span<const Tetrahedron> target_tetrahedra,
    const SourceModel source_model,
    const TargetModel target_model)
{
    return build_static_p2p_operator_impl(
        target_positions, source_positions, interactions, source_geometry,
        source_prisms, source_tetrahedra, target_geometry, target_prisms,
        target_tetrahedra, source_model, target_model);
}

} // namespace cdfmm

namespace cdfmm::operators::p2p {

PairTensor build_pair(
    const Vec3& target, const Vec3& source,
    const SourceGeometry source_geometry,
    const TargetGeometry target_geometry,
    const CuboidSize& source_size, const CuboidSize& target_size,
    const bool omit_singular_point_pair)
{
    return build_pair_tensor(
        target, source, source_geometry, target_geometry, source_size,
        target_size, omit_singular_point_pair);
}

PotentialField evaluate_pair(
    const Vec3& target, const Vec3& source, const Vec3& moment,
    const OutputFlags output)
{
    return p2p_dipole_pair(target, source, moment, output);
}

CanonicalOperator build(
    const std::span<const Vec3> target_positions,
    const std::span<const Vec3> source_positions,
    const std::span<const Interaction> interactions,
    const SourceGeometry source_geometry,
    const std::span<const CuboidSize> source_sizes,
    const TargetGeometry target_geometry,
    const std::span<const CuboidSize> target_sizes)
{
    return build_static_p2p_operator(
        target_positions, source_positions, interactions, source_geometry,
        source_sizes, target_geometry, target_sizes);
}

CanonicalOperator build(
    const std::span<const Vec3> target_positions,
    const std::span<const Vec3> source_positions,
    const std::span<const std::array<int, 2>> interactions,
    const SourceGeometry source_geometry,
    const std::span<const CuboidSize> source_sizes,
    const TargetGeometry target_geometry,
    const std::span<const CuboidSize> target_sizes)
{
    return build_static_p2p_operator(
        target_positions, source_positions, interactions, source_geometry,
        source_sizes, target_geometry, target_sizes);
}

CanonicalOperator build(
    const std::span<const Vec3> target_positions,
    const std::span<const Vec3> source_positions,
    const std::span<const std::array<int, 2>> interactions,
    const SourceGeometry source_geometry,
    const std::span<const RectangularPrism> source_prisms,
    const std::span<const Tetrahedron> source_tetrahedra,
    const TargetGeometry target_geometry,
    const std::span<const RectangularPrism> target_prisms,
    const std::span<const Tetrahedron> target_tetrahedra,
    const SourceModel source_model,
    const TargetModel target_model)
{
    return build_static_p2p_operator(
        target_positions, source_positions, interactions, source_geometry,
        source_prisms, source_tetrahedra, target_geometry, target_prisms,
        target_tetrahedra, source_model, target_model);
}

CanonicalOperator build(
    const std::span<const Vec3> target_positions,
    const std::span<const Vec3> source_positions,
    const std::span<const Interaction> interactions,
    const SourceGeometry source_geometry,
    const std::span<const RectangularPrism> source_prisms,
    const std::span<const Tetrahedron> source_tetrahedra,
    const TargetGeometry target_geometry,
    const std::span<const RectangularPrism> target_prisms,
    const std::span<const Tetrahedron> target_tetrahedra,
    const SourceModel source_model,
    const TargetModel target_model)
{
    return build_static_p2p_operator(
        target_positions, source_positions, interactions, source_geometry,
        source_prisms, source_tetrahedra, target_geometry, target_prisms,
        target_tetrahedra, source_model, target_model);
}

} // namespace cdfmm::operators::p2p
