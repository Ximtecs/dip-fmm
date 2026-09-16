// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/operators/p2p.hpp"

#include "operators/p2p_point_kernel.hpp"

#include "../geometry/primitives/tetrahedron_detail.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <numeric>
#include <numbers>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace cdfmm {
namespace {

[[nodiscard]] constexpr double canonical_zero(const double value) noexcept
{
    return value == 0.0 ? 0.0 : value;
}

[[nodiscard]] constexpr bool exactly_equal(
    const Vec3& left, const Vec3& right) noexcept
{
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

[[nodiscard]] constexpr bool exactly_equal(
    const Tetrahedron& left, const Tetrahedron& right) noexcept
{
    for (std::size_t vertex = 0; vertex < left.vertices.size(); ++vertex) {
        if (!exactly_equal(left.vertices[vertex], right.vertices[vertex])) {
            return false;
        }
    }
    return true;
}

struct RowInteractionKey {
    int source{0};
    double shift_x{0.0};
    double shift_y{0.0};
    double shift_z{0.0};
};

[[nodiscard]] auto row_interaction_sort_key(
    const StaticP2PInteraction& interaction) noexcept
{
    return std::tuple{
        interaction.source, canonical_zero(interaction.source_shift.x),
        canonical_zero(interaction.source_shift.y),
        canonical_zero(interaction.source_shift.z)};
}

[[nodiscard]] auto row_interaction_sort_key(
    const RowInteractionKey& key) noexcept
{
    return std::tuple{
        key.source, canonical_zero(key.shift_x),
        canonical_zero(key.shift_y), canonical_zero(key.shift_z)};
}

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

    const auto validate_count = [](const std::size_t geometry_count,
                                   const std::size_t object_count,
                                   const char* message) {
        if (geometry_count != 1 && geometry_count != object_count) {
            throw std::invalid_argument(message);
        }
    };
    std::vector<detail::PreparedTetrahedron> prepared_sources;
    std::vector<detail::PreparedTetrahedron> prepared_targets;
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
        if (target_is_tetrahedron) {
            prepared_sources.reserve(source_tetrahedra.size());
        }
        for (const Tetrahedron& tetrahedron : source_tetrahedra) {
            if (target_is_tetrahedron) {
                prepared_sources.push_back(
                    detail::prepare_tetrahedron(tetrahedron));
            } else {
                static_cast<void>(tetrahedron_volume(tetrahedron));
            }
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
        if (source_is_tetrahedron) {
            prepared_targets.reserve(target_tetrahedra.size());
        }
        for (const Tetrahedron& tetrahedron : target_tetrahedra) {
            if (source_is_tetrahedron) {
                prepared_targets.push_back(
                    detail::prepare_tetrahedron(tetrahedron));
            } else {
                static_cast<void>(tetrahedron_volume(tetrahedron));
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

    if ((source_is_prism && target_is_tetrahedron) ||
        (source_is_tetrahedron && target_is_prism)) {
        // Mixed prism/tetrahedron pairs use the exact polyhedron surface
        // formulation shared with the tetrahedron pair.  Surfaces are prepared
        // once per distinct record (common or per object) and the face-pair
        // integrals run in parallel over the sorted interactions.
        std::vector<detail::PolyhedronBody> source_bodies;
        std::vector<detail::PolyhedronBody> target_bodies;
        if (source_is_prism) {
            source_bodies.reserve(source_prisms.size());
            for (const RectangularPrism& prism : source_prisms) {
                source_bodies.push_back(detail::prepare_polyhedron_body(prism));
            }
        } else {
            source_bodies.reserve(source_tetrahedra.size());
            for (const Tetrahedron& tetrahedron : source_tetrahedra) {
                source_bodies.push_back(
                    detail::prepare_polyhedron_body(tetrahedron));
            }
        }
        if (target_is_prism) {
            target_bodies.reserve(target_prisms.size());
            for (const RectangularPrism& prism : target_prisms) {
                target_bodies.push_back(detail::prepare_polyhedron_body(prism));
            }
        } else {
            target_bodies.reserve(target_tetrahedra.size());
            for (const Tetrahedron& tetrahedron : target_tetrahedra) {
                target_bodies.push_back(
                    detail::prepare_polyhedron_body(tetrahedron));
            }
        }
        const auto& source_body_at =
            [&](const int source) -> const detail::PolyhedronBody& {
            return source_bodies[
                source_bodies.size() == 1 ? 0 : static_cast<std::size_t>(source)];
        };
        const auto& target_body_at =
            [&](const int target) -> const detail::PolyhedronBody& {
            return target_bodies[
                target_bodies.size() == 1 ? 0 : static_cast<std::size_t>(target)];
        };

        std::exception_ptr first_exception;
        std::atomic<bool> failed{false};
        const std::ptrdiff_t interaction_count =
            static_cast<std::ptrdiff_t>(sorted.size());
#pragma omp parallel for schedule(dynamic, 16) if (interaction_count >= 64)
        for (std::ptrdiff_t raw_index = 0; raw_index < interaction_count;
             ++raw_index) {
            const std::size_t index = static_cast<std::size_t>(raw_index);
            if (failed.load(std::memory_order_relaxed)) {
                continue;
            }
            try {
                const StaticP2PInteraction& interaction = sorted[index];
                const int target = interaction.target;
                const int source = interaction.source;
                const Vec3 displacement =
                    target_positions[static_cast<std::size_t>(target)] -
                    (source_positions[static_cast<std::size_t>(source)] +
                     interaction.source_shift);
                const double radius_squared = dot(displacement, displacement);
                const double potential_scale = radius_squared == 0.0
                    ? 0.0
                    : 1.0 / (4.0 * std::numbers::pi * radius_squared *
                             std::sqrt(radius_squared));
                const PairTensor tensor = detail::polyhedron_pair_tensor(
                    displacement, source_body_at(source),
                    target_body_at(target));
                // Finite sources never carry the point identity marker: their
                // coincident self field is physical.
                result.blocks[index] = {
                    target, source, potential_scale * displacement.x,
                    potential_scale * displacement.y,
                    potential_scale * displacement.z, tensor.xx, tensor.xy,
                    tensor.xz, tensor.yy, tensor.yz, tensor.zz, 0};
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
#pragma omp critical(cdfmm_polyhedron_setup_exception)
                {
                    if (!first_exception) {
                        first_exception = std::current_exception();
                    }
                }
            }
        }
        if (first_exception) {
            std::rethrow_exception(first_exception);
        }
        return result;
    }

    if (source_is_tetrahedron && target_is_tetrahedron) {
        const auto& prepared_source_at =
            [&](const int source) -> const detail::PreparedTetrahedron& {
            return prepared_sources[
                prepared_sources.size() == 1 ? 0 :
                                                static_cast<std::size_t>(source)];
        };
        const auto& prepared_target_at =
            [&](const int target) -> const detail::PreparedTetrahedron& {
            return prepared_targets[
                prepared_targets.size() == 1 ? 0 :
                                                static_cast<std::size_t>(target)];
        };
        const auto& source_tetrahedron_at =
            [&](const int source) -> const Tetrahedron& {
            return source_tetrahedra[
                source_tetrahedra.size() == 1 ? 0 :
                                                static_cast<std::size_t>(source)];
        };
        const auto& target_tetrahedron_at =
            [&](const int target) -> const Tetrahedron& {
            return target_tetrahedra[
                target_tetrahedra.size() == 1 ? 0 :
                                                static_cast<std::size_t>(target)];
        };

        bool reciprocal_tetrahedron_layout =
            source_positions.size() == target_positions.size();
        for (std::size_t index = 0;
             reciprocal_tetrahedron_layout && index < source_positions.size();
             ++index) {
            reciprocal_tetrahedron_layout =
                exactly_equal(source_positions[index], target_positions[index]) &&
                exactly_equal(
                    source_tetrahedron_at(static_cast<int>(index)),
                    target_tetrahedron_at(static_cast<int>(index)));
        }

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
            const double potential_scale = radius_squared == 0.0
                ? 0.0
                : 1.0 / (4.0 * std::numbers::pi * radius_squared *
                         std::sqrt(radius_squared));
            result.blocks[index] = {
                target, source, potential_scale * displacement.x,
                potential_scale * displacement.y,
                potential_scale * displacement.z, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0};
        }

        std::vector<std::size_t> tensor_owner(sorted.size());
        std::iota(tensor_owner.begin(), tensor_owner.end(), std::size_t{0});
        if (reciprocal_tetrahedron_layout) {
            // For an identical source/target tetrahedral system,
            // K_{t<-s}(r) = K_{s<-t}(-r)^T. PairTensor stores the symmetric
            // Cartesian Hessian tensor, so reciprocal directed interactions
            // share the same Tensor6.
            for (std::size_t index = 0; index < sorted.size(); ++index) {
                const StaticP2PInteraction& interaction = sorted[index];
                const RowInteractionKey reverse_key{
                    interaction.target,
                    -canonical_zero(interaction.source_shift.x),
                    -canonical_zero(interaction.source_shift.y),
                    -canonical_zero(interaction.source_shift.z)};
                const auto row_begin = sorted.begin() +
                    result.row_offsets[
                        static_cast<std::size_t>(interaction.source)];
                const auto row_end = sorted.begin() +
                    result.row_offsets[
                        static_cast<std::size_t>(interaction.source) + 1];
                const auto reverse = std::lower_bound(
                    row_begin, row_end, reverse_key,
                    [](const StaticP2PInteraction& candidate,
                       const RowInteractionKey& key) {
                        return row_interaction_sort_key(candidate) <
                            row_interaction_sort_key(key);
                    });
                if (reverse != row_end &&
                    row_interaction_sort_key(*reverse) ==
                        row_interaction_sort_key(reverse_key)) {
                    const std::size_t reverse_index =
                        static_cast<std::size_t>(reverse - sorted.begin());
                    const std::size_t owner = std::min(index, reverse_index);
                    tensor_owner[index] = owner;
                    tensor_owner[reverse_index] = owner;
                }
            }
        }

        std::exception_ptr first_exception;
        std::atomic<bool> failed{false};
        const std::ptrdiff_t interaction_count =
            static_cast<std::ptrdiff_t>(sorted.size());
#pragma omp parallel for schedule(static) if (interaction_count >= 256)
        for (std::ptrdiff_t raw_index = 0; raw_index < interaction_count;
             ++raw_index) {
            const std::size_t index = static_cast<std::size_t>(raw_index);
            if (tensor_owner[index] != index ||
                failed.load(std::memory_order_relaxed)) {
                continue;
            }
            try {
                const StaticP2PInteraction& interaction = sorted[index];
                const int target = interaction.target;
                const int source = interaction.source;
                const Vec3 displacement =
                    target_positions[static_cast<std::size_t>(target)] -
                    (source_positions[static_cast<std::size_t>(source)] +
                     interaction.source_shift);
                const bool coincident_same_geometry =
                    reciprocal_tetrahedron_layout && target == source &&
                    interaction.source_shift.x == 0.0 &&
                    interaction.source_shift.y == 0.0 &&
                    interaction.source_shift.z == 0.0;
                const PairTensor tensor =
                    detail::tetrahedron_tetrahedron_tensor_prepared(
                        displacement, prepared_source_at(source),
                        prepared_target_at(target), coincident_same_geometry);
                result.blocks[index].xx = tensor.xx;
                result.blocks[index].xy = tensor.xy;
                result.blocks[index].xz = tensor.xz;
                result.blocks[index].yy = tensor.yy;
                result.blocks[index].yz = tensor.yz;
                result.blocks[index].zz = tensor.zz;
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
#pragma omp critical(cdfmm_tetrahedron_setup_exception)
                {
                    if (!first_exception) {
                        first_exception = std::current_exception();
                    }
                }
            }
        }
        if (first_exception) {
            std::rethrow_exception(first_exception);
        }

        for (std::size_t index = 0; index < sorted.size(); ++index) {
            const std::size_t owner = tensor_owner[index];
            if (owner == index) {
                continue;
            }
            result.blocks[index].xx = result.blocks[owner].xx;
            result.blocks[index].xy = result.blocks[owner].xy;
            result.blocks[index].xz = result.blocks[owner].xz;
            result.blocks[index].yy = result.blocks[owner].yy;
            result.blocks[index].yz = result.blocks[owner].yz;
            result.blocks[index].zz = result.blocks[owner].zz;
        }
        return result;
    }

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
            tensor = operators::p2p::build_pair(
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
    if (source_geometry == SourceGeometry::Tetrahedron ||
        target_geometry == TargetGeometry::Tetrahedron) {
        // NOTE(cdfmm): The message names the flat compatibility spelling
        // deliberately; it is part of the preserved observable behaviour.
        throw std::invalid_argument(
            "build_pair_tensor does not accept tetrahedral geometry; "
            "use the tetrahedron P2P operator with tetrahedron records");
    }
    const Vec3 r = target - source;
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

PotentialField evaluate_pair(
    const Vec3& target, const Vec3& source, const Vec3& moment,
    const OutputFlags output)
{
    PotentialField result;
    const Vec3 r = target - source;

    if (has_flag(output, OutputFlags::Potential)) {
        result.phi = point_dipole_potential(r.x, r.y, r.z, moment.x, moment.y,
                                            moment.z);
    }

    if (has_flag(output, OutputFlags::Field)) {
        // H_ij = 1/(4*pi) * [3*r*(m.r)/|r|^5 - m/|r|^3]
        accumulate_point_dipole_field(r.x, r.y, r.z, moment.x, moment.y,
                                      moment.z, result.H.x, result.H.y,
                                      result.H.z);
    }

    return result;
}

PotentialField evaluate_sum(
    const Vec3& target,
    const std::span<const Vec3> sources,
    const std::span<const Vec3> moments,
    const OutputFlags output,
    const int self_index)
{
    PotentialField result;

    for (size_t i = 0; i < sources.size(); ++i) {
        // WARNING(cdfmm): Self-interactions must be excluded when targets are
        // source points to avoid singular |r|=0 evaluation.
        if (static_cast<int>(i) == self_index) {
            continue;
        }

        const PotentialField pair =
            evaluate_pair(target, sources[i], moments[i], output);
        result.phi += pair.phi;
        result.H += pair.H;
    }

    return result;
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

namespace cdfmm {

// Flat compatibility entry point retained for source and Python compatibility.
// Mathematical ownership lives in `cdfmm::operators::p2p::build_pair`.

PairTensor build_pair_tensor(const Vec3& target_position,
                             const Vec3& source_position,
                             const SourceGeometry source_geometry,
                             const TargetGeometry target_geometry,
                             const CuboidSize& source_size,
                             const CuboidSize& target_size,
                             const bool omit_singular_point_pair)
{
    return operators::p2p::build_pair(
        target_position, source_position, source_geometry, target_geometry,
        source_size, target_size, omit_singular_point_pair);
}

} // namespace cdfmm
