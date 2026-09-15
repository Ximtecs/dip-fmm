// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <span>

#include "cdfmm/core/output_flags.hpp"
#include "cdfmm/geometry/models.hpp"
#include "cdfmm/geometry/primitives/rectangular_prism.hpp"
#include "cdfmm/geometry/primitives/tetrahedron.hpp"
#include "cdfmm/math/pair_tensor.hpp"
#include "cdfmm/math/potential_field.hpp"
#include "cdfmm/math/vec3.hpp"
#include "cdfmm/plan/p2p/canonical.hpp"

namespace cdfmm {

/** @brief One exact near-field pair with a periodically shifted source. */
struct StaticP2PInteraction {
    int target{0};
    int source{0};
    Vec3 source_shift{};
    bool skip_for_identity{true};
};

} // namespace cdfmm

namespace cdfmm::operators::p2p {

using PairTensor = cdfmm::PairTensor;
using Interaction = cdfmm::StaticP2PInteraction;
using CanonicalOperator = cdfmm::StaticP2POperator;

/**
 * @brief Builds one geometry-specific pair tensor.
 *
 * This is the authoritative point/rectangular-prism pair-tensor construction.
 * The tensor maps a total source dipole moment to target field.  Point-point
 * singularity remains explicit; callers selecting source-point identity
 * exclusion must do so through an interaction's `skip_for_identity` marker.
 * Tetrahedral geometry is rejected; use the tetrahedron P2P operator instead.
 */
[[nodiscard]] PairTensor build_pair(const Vec3& target,
                                    const Vec3& source,
                                    SourceGeometry source_geometry =
                                        SourceGeometry::PointDipole,
                                    TargetGeometry target_geometry =
                                        TargetGeometry::Point,
                                    const CuboidSize& source_size = {},
                                    const CuboidSize& target_size = {},
                                    bool omit_singular_point_pair = false);

/** @brief Evaluates one direct point-dipole pair with the standard convention. */
[[nodiscard]] PotentialField evaluate_pair(const Vec3& target,
                                           const Vec3& source,
                                           const Vec3& moment,
                                           OutputFlags output =
                                               OutputFlags::Field);

/** @brief Sums direct point-dipole contributions at one target. */
[[nodiscard]] PotentialField evaluate_sum(
    const Vec3& target,
    std::span<const Vec3> sources,
    std::span<const Vec3> moments,
    OutputFlags output = OutputFlags::Field,
    int self_index = -1);

/**
 * @brief Builds canonical target-row P2P data from explicit pair records.
 *
 * Pair records are sorted deterministically by construction.  They retain
 * source shifts and identity markers, while all compact/dictionary/BSR forms
 * remain derived execution representations.
 */
[[nodiscard]] CanonicalOperator build(
    std::span<const Vec3> target_positions,
    std::span<const Vec3> source_positions,
    std::span<const Interaction> interactions,
    SourceGeometry source_geometry = SourceGeometry::PointDipole,
    std::span<const CuboidSize> source_sizes = {},
    TargetGeometry target_geometry = TargetGeometry::Point,
    std::span<const CuboidSize> target_sizes = {});

/** @brief Builds canonical data from non-periodic target/source index pairs. */
[[nodiscard]] CanonicalOperator build(
    std::span<const Vec3> target_positions,
    std::span<const Vec3> source_positions,
    std::span<const std::array<int, 2>> interactions,
    SourceGeometry source_geometry = SourceGeometry::PointDipole,
    std::span<const CuboidSize> source_sizes = {},
    TargetGeometry target_geometry = TargetGeometry::Point,
    std::span<const CuboidSize> target_sizes = {});

/** @brief Builds canonical data with independent exact source/target models. */
[[nodiscard]] CanonicalOperator build(
    std::span<const Vec3> target_positions,
    std::span<const Vec3> source_positions,
    std::span<const std::array<int, 2>> interactions,
    SourceGeometry source_geometry,
    std::span<const RectangularPrism> source_prisms,
    std::span<const Tetrahedron> source_tetrahedra,
    TargetGeometry target_geometry,
    std::span<const RectangularPrism> target_prisms,
    std::span<const Tetrahedron> target_tetrahedra,
    SourceModel source_model = SourceModel::ExactGeometry,
    TargetModel target_model = TargetModel::ExactGeometry);

/** @brief Builds image-aware canonical data with independent exact models. */
[[nodiscard]] CanonicalOperator build(
    std::span<const Vec3> target_positions,
    std::span<const Vec3> source_positions,
    std::span<const Interaction> interactions,
    SourceGeometry source_geometry,
    std::span<const RectangularPrism> source_prisms,
    std::span<const Tetrahedron> source_tetrahedra,
    TargetGeometry target_geometry,
    std::span<const RectangularPrism> target_prisms,
    std::span<const Tetrahedron> target_tetrahedra,
    SourceModel source_model = SourceModel::ExactGeometry,
    TargetModel target_model = TargetModel::ExactGeometry);

} // namespace cdfmm::operators::p2p

namespace cdfmm {

// Flat compatibility entry points.  Mathematical ownership lives in
// `cdfmm::operators::p2p`; these names are retained for source compatibility.

/**
 * @brief Compatibility spelling of `cdfmm::operators::p2p::build_pair`.
 *
 * Runtime inputs are total moments m=V*M. Cuboid source normalisation is
 * consequently included in the returned tensor. Point-point coincidence is
 * singular unless @p omit_singular_point_pair is true.
 */
[[nodiscard]] PairTensor build_pair_tensor(
    const Vec3& target_position,
    const Vec3& source_position,
    SourceGeometry source_geometry = SourceGeometry::PointDipole,
    TargetGeometry target_geometry = TargetGeometry::Point,
    const CuboidSize& source_size = {},
    const CuboidSize& target_size = {},
    bool omit_singular_point_pair = false
);

[[nodiscard]] StaticP2POperator build_static_p2p_operator(
    std::span<const Vec3> target_positions,
    std::span<const Vec3> source_positions,
    std::span<const std::array<int, 2>> interactions,
    SourceGeometry source_geometry = SourceGeometry::PointDipole,
    std::span<const CuboidSize> source_sizes = {},
    TargetGeometry target_geometry = TargetGeometry::Point,
    std::span<const CuboidSize> target_sizes = {});
[[nodiscard]] StaticP2POperator build_static_p2p_operator(
    std::span<const Vec3> target_positions,
    std::span<const Vec3> source_positions,
    std::span<const StaticP2PInteraction> interactions,
    SourceGeometry source_geometry = SourceGeometry::PointDipole,
    std::span<const CuboidSize> source_sizes = {},
    TargetGeometry target_geometry = TargetGeometry::Point,
    std::span<const CuboidSize> target_sizes = {});
[[nodiscard]] StaticP2POperator build_static_p2p_operator(
    std::span<const Vec3> target_positions,
    std::span<const Vec3> source_positions,
    std::span<const std::array<int, 2>> interactions,
    SourceGeometry source_geometry,
    std::span<const RectangularPrism> source_prisms,
    std::span<const Tetrahedron> source_tetrahedra,
    TargetGeometry target_geometry,
    std::span<const RectangularPrism> target_prisms,
    std::span<const Tetrahedron> target_tetrahedra,
    SourceModel source_model = SourceModel::ExactGeometry,
    TargetModel target_model = TargetModel::ExactGeometry);
[[nodiscard]] StaticP2POperator build_static_p2p_operator(
    std::span<const Vec3> target_positions,
    std::span<const Vec3> source_positions,
    std::span<const StaticP2PInteraction> interactions,
    SourceGeometry source_geometry,
    std::span<const RectangularPrism> source_prisms,
    std::span<const Tetrahedron> source_tetrahedra,
    TargetGeometry target_geometry,
    std::span<const RectangularPrism> target_prisms,
    std::span<const Tetrahedron> target_tetrahedra,
    SourceModel source_model = SourceModel::ExactGeometry,
    TargetModel target_model = TargetModel::ExactGeometry);

} // namespace cdfmm
