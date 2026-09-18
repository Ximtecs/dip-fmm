// SPDX-License-Identifier: Apache-2.0

#include "cdfmm/operators/p2p.hpp"

#include "operators/exact_operator_reuse.hpp"
#include "operators/p2p_point_kernel.hpp"

#include "../geometry/primitives/tetrahedron_detail.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <numbers>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
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

// The exact-operator equivalence, its classification and the parallel
// failure report are shared with the dense all-to-all builder; see
// `src/operators/exact_operator_reuse.hpp` for what makes two pairs the
// same operator and why the comparison is bitwise.
using detail::exact_reuse::classify_exact_operators;
using detail::exact_reuse::ExactOperatorClasses;
using detail::exact_reuse::ExactOperatorKey;
using FirstPairFailure = detail::exact_reuse::FirstFailure;

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

    // The displacement every pair loop below works from.  A periodic image
    // shift is folded in here, so nothing downstream sees it separately and
    // two images that reach the same displacement are the same interaction.
    const auto displacement_at = [&](const std::size_t index) -> Vec3 {
        const StaticP2PInteraction& interaction = sorted[index];
        const Vec3 shifted_source =
            source_positions[static_cast<std::size_t>(interaction.source)] +
            interaction.source_shift;
        return target_positions[static_cast<std::size_t>(interaction.target)] -
            shifted_source;
    };
    const auto potential_scale_at = [](const Vec3& displacement) -> double {
        const double radius_squared = dot(displacement, displacement);
        return radius_squared == 0.0
            ? 0.0
            : 1.0 / (4.0 * std::numbers::pi * radius_squared *
                     std::sqrt(radius_squared));
    };

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

        // The face-pair integral is a pure function of the displacement and the
        // two body records, so pairs whose inputs agree bit for bit share one
        // tensor.  It is by far the most expensive tensor the builder has, so
        // classifying first pays for itself many times over on repeated
        // geometry and costs one hash lookup per pair otherwise.
        const auto exact_operator_key =
            [&](const std::size_t index) -> ExactOperatorKey {
            const StaticP2PInteraction& interaction = sorted[index];
            ExactOperatorKey key;
            key.push(displacement_at(index));
            if (source_is_prism) {
                key.push(source_prisms[source_prisms.size() == 1
                                           ? 0
                                           : static_cast<std::size_t>(
                                                 interaction.source)]);
            } else {
                key.push(source_tetrahedra[source_tetrahedra.size() == 1
                                               ? 0
                                               : static_cast<std::size_t>(
                                                     interaction.source)]);
            }
            if (target_is_prism) {
                key.push(target_prisms[target_prisms.size() == 1
                                           ? 0
                                           : static_cast<std::size_t>(
                                                 interaction.target)]);
            } else {
                key.push(target_tetrahedra[target_tetrahedra.size() == 1
                                               ? 0
                                               : static_cast<std::size_t>(
                                                     interaction.target)]);
            }
            return key;
        };

        const ExactOperatorClasses classes =
            classify_exact_operators(sorted.size(), exact_operator_key);

        FirstPairFailure failure;
        const std::ptrdiff_t interaction_count =
            static_cast<std::ptrdiff_t>(sorted.size());

        // Finite sources never carry the point identity marker: their
        // coincident self field is physical.
        const auto store_block = [&](const std::size_t index,
                                     const PairTensor& tensor) {
            const StaticP2PInteraction& interaction = sorted[index];
            const Vec3 displacement = displacement_at(index);
            const double potential_scale = potential_scale_at(displacement);
            result.blocks[index] = {
                interaction.target, interaction.source,
                potential_scale * displacement.x,
                potential_scale * displacement.y,
                potential_scale * displacement.z, tensor.xx, tensor.xy,
                tensor.xz, tensor.yy, tensor.yz, tensor.zz, 0};
        };
        const auto build_tensor = [&](const std::size_t index) -> PairTensor {
            const StaticP2PInteraction& interaction = sorted[index];
            return detail::polyhedron_pair_tensor(
                displacement_at(index), source_body_at(interaction.source),
                target_body_at(interaction.target));
        };

        if (classes.classified) {
            const std::ptrdiff_t class_count =
                static_cast<std::ptrdiff_t>(classes.representative.size());
            std::vector<PairTensor> tensors(classes.representative.size());
#pragma omp parallel for schedule(dynamic, 1) if (class_count >= 8)
            for (std::ptrdiff_t raw_class = 0; raw_class < class_count;
                 ++raw_class) {
                const std::size_t entry = static_cast<std::size_t>(raw_class);
                const std::size_t index =
                    static_cast<std::size_t>(classes.representative[entry]);
                if (failure.superseded(index)) {
                    continue;
                }
                try {
                    tensors[entry] = build_tensor(index);
                } catch (...) {
                    failure.record(index);
                }
            }
            failure.rethrow_any();

#pragma omp parallel for schedule(static) if (interaction_count >= 256)
            for (std::ptrdiff_t raw_index = 0; raw_index < interaction_count;
                 ++raw_index) {
                const std::size_t index = static_cast<std::size_t>(raw_index);
                store_block(index, tensors[classes.class_of_pair[index]]);
            }
            return result;
        }

#pragma omp parallel for schedule(dynamic, 16) if (interaction_count >= 64)
        for (std::ptrdiff_t raw_index = 0; raw_index < interaction_count;
             ++raw_index) {
            const std::size_t index = static_cast<std::size_t>(raw_index);
            if (failure.superseded(index)) {
                continue;
            }
            try {
                store_block(index, build_tensor(index));
            } catch (...) {
                failure.record(index);
            }
        }
        failure.rethrow_any();
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

        const std::ptrdiff_t block_count =
            static_cast<std::ptrdiff_t>(sorted.size());
#pragma omp parallel for schedule(static) if (block_count >= 256)
        for (std::ptrdiff_t raw_index = 0; raw_index < block_count;
             ++raw_index) {
            const std::size_t index = static_cast<std::size_t>(raw_index);
            const StaticP2PInteraction& interaction = sorted[index];
            const Vec3 displacement = displacement_at(index);
            const double potential_scale = potential_scale_at(displacement);
            result.blocks[index] = {
                interaction.target, interaction.source,
                potential_scale * displacement.x,
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

        const auto coincident_same_geometry_at =
            [&](const std::size_t index) -> bool {
            const StaticP2PInteraction& interaction = sorted[index];
            return reciprocal_tetrahedron_layout &&
                interaction.target == interaction.source &&
                interaction.source_shift.x == 0.0 &&
                interaction.source_shift.y == 0.0 &&
                interaction.source_shift.z == 0.0;
        };

        // Only the reciprocity owners are built; the rest copy their partner's
        // tensor.  Classification therefore runs over the owners alone, which
        // keeps the reciprocal pair sharing one set of bits exactly as before.
        //
        // The coincident flag belongs in the key because it selects the
        // symmetrised coincident algorithm rather than the general one.
        std::vector<std::uint32_t> owners;
        owners.reserve(sorted.size());
        for (std::size_t index = 0; index < sorted.size(); ++index) {
            if (tensor_owner[index] == index) {
                owners.push_back(static_cast<std::uint32_t>(index));
            }
        }

        const auto exact_operator_key =
            [&](const std::size_t owner) -> ExactOperatorKey {
            const std::size_t index = static_cast<std::size_t>(owners[owner]);
            const StaticP2PInteraction& interaction = sorted[index];
            ExactOperatorKey key;
            key.push(displacement_at(index));
            key.push(source_tetrahedron_at(interaction.source));
            key.push(target_tetrahedron_at(interaction.target));
            key.push(coincident_same_geometry_at(index) ? 1.0 : 0.0);
            return key;
        };

        const ExactOperatorClasses classes =
            classify_exact_operators(owners.size(), exact_operator_key);

        FirstPairFailure failure;
        const std::ptrdiff_t owner_count =
            static_cast<std::ptrdiff_t>(owners.size());

        const auto store_tensor = [&](const std::size_t index,
                                      const PairTensor& tensor) {
            result.blocks[index].xx = tensor.xx;
            result.blocks[index].xy = tensor.xy;
            result.blocks[index].xz = tensor.xz;
            result.blocks[index].yy = tensor.yy;
            result.blocks[index].yz = tensor.yz;
            result.blocks[index].zz = tensor.zz;
        };
        const auto build_tensor = [&](const std::size_t index) -> PairTensor {
            const StaticP2PInteraction& interaction = sorted[index];
            return detail::tetrahedron_tetrahedron_tensor_prepared(
                displacement_at(index),
                prepared_source_at(interaction.source),
                prepared_target_at(interaction.target),
                coincident_same_geometry_at(index));
        };

        if (classes.classified) {
            const std::ptrdiff_t class_count =
                static_cast<std::ptrdiff_t>(classes.representative.size());
            std::vector<PairTensor> tensors(classes.representative.size());
#pragma omp parallel for schedule(dynamic, 1) if (class_count >= 8)
            for (std::ptrdiff_t raw_class = 0; raw_class < class_count;
                 ++raw_class) {
                const std::size_t entry = static_cast<std::size_t>(raw_class);
                const std::size_t index = static_cast<std::size_t>(
                    owners[classes.representative[entry]]);
                if (failure.superseded(index)) {
                    continue;
                }
                try {
                    tensors[entry] = build_tensor(index);
                } catch (...) {
                    failure.record(index);
                }
            }
            failure.rethrow_any();

#pragma omp parallel for schedule(static) if (owner_count >= 256)
            for (std::ptrdiff_t raw_owner = 0; raw_owner < owner_count;
                 ++raw_owner) {
                const std::size_t owner = static_cast<std::size_t>(raw_owner);
                store_tensor(static_cast<std::size_t>(owners[owner]),
                             tensors[classes.class_of_pair[owner]]);
            }
        } else {
#pragma omp parallel for schedule(dynamic, 1) if (owner_count >= 8)
            for (std::ptrdiff_t raw_owner = 0; raw_owner < owner_count;
                 ++raw_owner) {
                const std::size_t index = static_cast<std::size_t>(
                    owners[static_cast<std::size_t>(raw_owner)]);
                if (failure.superseded(index)) {
                    continue;
                }
                try {
                    store_tensor(index, build_tensor(index));
                } catch (...) {
                    failure.record(index);
                }
            }
            failure.rethrow_any();
        }

        const std::ptrdiff_t copy_count =
            static_cast<std::ptrdiff_t>(sorted.size());
#pragma omp parallel for schedule(static) if (copy_count >= 256)
        for (std::ptrdiff_t raw_index = 0; raw_index < copy_count;
             ++raw_index) {
            const std::size_t index = static_cast<std::size_t>(raw_index);
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

    // The generic loop serves the point/prism combinations and both
    // point/tetrahedron directions.  Its iterations are independent: the block
    // vector is already sized and ordered above and each iteration writes one
    // distinct entry, so the loop runs in parallel with the same first-failure
    // handling as the polyhedron loops.
    //
    // When either body is finite the loop also classifies first and builds
    // each distinct exact operator once.  The classification costs one hash
    // lookup per pair against an exact tensor that costs orders of magnitude
    // more, whereas a point pair is cheaper to evaluate than to look up and a
    // point cloud has no duplicates to find, so point pairs build directly.
    const bool reuse_exact_operators = source_is_prism || target_is_prism ||
        source_is_tetrahedron || target_is_tetrahedron;

    const auto source_prism_at = [&](const int source) -> RectangularPrism {
        if (!source_is_prism) {
            return RectangularPrism{};
        }
        return source_prisms[source_prisms.size() == 1
                                 ? 0 : static_cast<std::size_t>(source)];
    };
    const auto target_prism_at = [&](const int target) -> RectangularPrism {
        if (!target_is_prism) {
            return RectangularPrism{};
        }
        return target_prisms[target_prisms.size() == 1
                                 ? 0 : static_cast<std::size_t>(target)];
    };
    const auto source_tetrahedron_at =
        [&](const int source) -> const Tetrahedron* {
        if (!source_is_tetrahedron) {
            return nullptr;
        }
        return &source_tetrahedra[source_tetrahedra.size() == 1
                                      ? 0
                                      : static_cast<std::size_t>(source)];
    };
    const auto target_tetrahedron_at =
        [&](const int target) -> const Tetrahedron* {
        if (!target_is_tetrahedron) {
            return nullptr;
        }
        return &target_tetrahedra[target_tetrahedra.size() == 1
                                      ? 0
                                      : static_cast<std::size_t>(target)];
    };
    // A point source exactly on a point target has no field; the block records
    // that explicitly and only the identity flag says whether it was excluded
    // or genuinely singular.  Finite geometry never reaches this: its
    // coincident field is the physical self limit.
    const auto singular_point_pair = [&](const Vec3& displacement) -> bool {
        return effective_source_geometry == SourceGeometry::PointDipole &&
            effective_target_geometry == TargetGeometry::Point &&
            dot(displacement, displacement) == 0.0;
    };

    const auto exact_operator_key =
        [&](const std::size_t index) -> ExactOperatorKey {
        const StaticP2PInteraction& interaction = sorted[index];
        ExactOperatorKey key;
        key.push(displacement_at(index));
        if (source_is_prism) {
            key.push(source_prism_at(interaction.source));
        }
        if (target_is_prism) {
            key.push(target_prism_at(interaction.target));
        }
        if (source_is_tetrahedron) {
            key.push(*source_tetrahedron_at(interaction.source));
        }
        if (target_is_tetrahedron) {
            key.push(*target_tetrahedron_at(interaction.target));
        }
        return key;
    };

    // `build_pair` uses only the difference of its two position arguments, so
    // evaluating the displacement against the origin reproduces what the
    // pair's own positions give, bit for bit.
    const auto build_pair_tensor =
        [&](const std::size_t index) -> PairTensor {
        const StaticP2PInteraction& interaction = sorted[index];
        const Vec3 displacement = displacement_at(index);
        if (source_is_tetrahedron && target_is_tetrahedron) {
            return tetrahedron_tetrahedron_tensor(
                displacement, *source_tetrahedron_at(interaction.source),
                *target_tetrahedron_at(interaction.target));
        }
        if (source_is_tetrahedron) {
            return tetrahedron_point_tensor(
                displacement, *source_tetrahedron_at(interaction.source));
        }
        if (target_is_tetrahedron) {
            return point_tetrahedron_tensor(
                displacement, *target_tetrahedron_at(interaction.target));
        }
        return operators::p2p::build_pair(
            displacement, Vec3{0.0, 0.0, 0.0}, effective_source_geometry,
            effective_target_geometry, source_prism_at(interaction.source),
            target_prism_at(interaction.target));
    };

    const auto store_block = [&](const std::size_t index,
                                 const PairTensor& tensor) {
        const StaticP2PInteraction& interaction = sorted[index];
        const Vec3 displacement = displacement_at(index);
        const double potential_scale = potential_scale_at(displacement);
        result.blocks[index] = {
            interaction.target, interaction.source,
            potential_scale * displacement.x,
            potential_scale * displacement.y,
            potential_scale * displacement.z, tensor.xx, tensor.xy, tensor.xz,
            tensor.yy, tensor.yz, tensor.zz,
            (interaction.skip_for_identity &&
             effective_source_geometry == SourceGeometry::PointDipole)
                ? 1 : 0};
    };

    const auto store_singular_block = [&](const std::size_t index) {
        const StaticP2PInteraction& interaction = sorted[index];
        const double undefined = std::numeric_limits<double>::quiet_NaN();
        result.blocks[index] = {
            interaction.target, interaction.source, undefined, undefined,
            undefined, undefined, undefined, undefined, undefined, undefined,
            undefined,
            (interaction.skip_for_identity &&
             effective_source_geometry == SourceGeometry::PointDipole)
                ? 1 : 0};
    };

    const ExactOperatorClasses classes = reuse_exact_operators
        ? classify_exact_operators(sorted.size(), exact_operator_key)
        : ExactOperatorClasses{};

    FirstPairFailure failure;

    if (classes.classified) {
        // Build one tensor per distinct set of inputs, then scatter.
        const std::ptrdiff_t class_count =
            static_cast<std::ptrdiff_t>(classes.representative.size());
        std::vector<PairTensor> tensors(classes.representative.size());
#pragma omp parallel for schedule(dynamic, 1) if (class_count >= 8)
        for (std::ptrdiff_t raw_class = 0; raw_class < class_count;
             ++raw_class) {
            const std::size_t index = static_cast<std::size_t>(
                classes.representative[static_cast<std::size_t>(raw_class)]);
            if (failure.superseded(index)) {
                continue;
            }
            try {
                tensors[static_cast<std::size_t>(raw_class)] =
                    build_pair_tensor(index);
            } catch (...) {
                failure.record(index);
            }
        }
        failure.rethrow_any();

        const std::ptrdiff_t pair_count =
            static_cast<std::ptrdiff_t>(sorted.size());
#pragma omp parallel for schedule(static) if (pair_count >= 256)
        for (std::ptrdiff_t raw_index = 0; raw_index < pair_count;
             ++raw_index) {
            const std::size_t index = static_cast<std::size_t>(raw_index);
            store_block(index, tensors[classes.class_of_pair[index]]);
        }
        return result;
    }

    const std::ptrdiff_t pair_count =
        static_cast<std::ptrdiff_t>(sorted.size());
#pragma omp parallel for schedule(static) if (pair_count >= 256)
    for (std::ptrdiff_t raw_index = 0; raw_index < pair_count; ++raw_index) {
        const std::size_t index = static_cast<std::size_t>(raw_index);
        if (failure.superseded(index)) {
            continue;
        }
        try {
            const Vec3 displacement = displacement_at(index);
            if (singular_point_pair(displacement)) {
                store_singular_block(index);
                continue;
            }
            store_block(index, build_pair_tensor(index));
        } catch (...) {
            failure.record(index);
        }
    }
    failure.rethrow_any();
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
