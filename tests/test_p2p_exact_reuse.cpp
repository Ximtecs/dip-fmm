// SPDX-License-Identifier: Apache-2.0
//
// Exact operator reuse in canonical P2P construction.
//
// The canonical builder groups pairs whose exact operator inputs -- the
// displacement and the participating body records -- agree bit for bit, builds
// one tensor per group, and scatters it.  These tests pin the contract that
// makes that safe:
//
//   * a reused tensor is bitwise identical to the one a pair-by-pair build
//     produces, for every geometry combination;
//   * inputs that differ, however slightly, never share a tensor;
//   * periodic images that reach the same displacement legitimately do;
//   * the finite self tensor and the point identity exclusion survive.
//
// Every comparison here is bitwise, because the reuse itself is bitwise: a
// tolerance would hide exactly the aliasing these tests exist to catch.

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include "cdfmm/operators/p2p.hpp"
#include "cdfmm/rectangular_prism.hpp"
#include "cdfmm/tetrahedron.hpp"

using namespace cdfmm;

namespace {

/// @brief Bitwise equality, so that a reused tensor must be the same value and
///        not merely a close one.
[[nodiscard]] bool same_bits(const double left, const double right)
{
    return std::bit_cast<std::uint64_t>(left) ==
        std::bit_cast<std::uint64_t>(right);
}

[[nodiscard]] bool same_tensor(const StaticDipoleBlock& left,
                               const StaticDipoleBlock& right)
{
    return same_bits(left.xx, right.xx) && same_bits(left.xy, right.xy) &&
        same_bits(left.xz, right.xz) && same_bits(left.yy, right.yy) &&
        same_bits(left.yz, right.yz) && same_bits(left.zz, right.zz);
}

[[nodiscard]] bool any_nonzero(const StaticDipoleBlock& block)
{
    return block.xx != 0.0 || block.xy != 0.0 || block.xz != 0.0 ||
        block.yy != 0.0 || block.yz != 0.0 || block.zz != 0.0;
}

/// @brief A small lattice of body centres, which is the layout that makes
///        displacements repeat exactly.
[[nodiscard]] std::vector<Vec3> lattice(const int side, const double spacing)
{
    std::vector<Vec3> positions;
    for (int z = 0; z < side; ++z) {
        for (int y = 0; y < side; ++y) {
            for (int x = 0; x < side; ++x) {
                positions.push_back({static_cast<double>(x) * spacing,
                                     static_cast<double>(y) * spacing,
                                     static_cast<double>(z) * spacing});
            }
        }
    }
    return positions;
}

/// @brief Every ordered pair, which exercises both repeated and unique inputs.
[[nodiscard]] std::vector<StaticP2PInteraction> all_pairs(
    const std::size_t count)
{
    std::vector<StaticP2PInteraction> interactions;
    for (std::size_t target = 0; target < count; ++target) {
        for (std::size_t source = 0; source < count; ++source) {
            interactions.push_back({static_cast<int>(target),
                                    static_cast<int>(source), Vec3{}, true});
        }
    }
    return interactions;
}

[[nodiscard]] Tetrahedron unit_tetrahedron(const double scale)
{
    return Tetrahedron{{{Vec3{-0.25 * scale, -0.25 * scale, -0.25 * scale},
                         Vec3{0.75 * scale, -0.25 * scale, -0.25 * scale},
                         Vec3{-0.25 * scale, 0.75 * scale, -0.25 * scale},
                         Vec3{-0.25 * scale, -0.25 * scale, 0.75 * scale}}}};
}

} // namespace

TEST_CASE("Repeated identical geometry reuses the same exact tensor")
{
    // A lattice of identical prisms reaches the same displacement from many
    // different pairs.  Each such pair must carry the identical tensor bits.
    const std::vector<Vec3> positions = lattice(3, 1.0);
    const std::vector<RectangularPrism> prisms{{0.4, 0.4, 0.4}};
    const std::vector<StaticP2PInteraction> interactions =
        all_pairs(positions.size());

    const StaticP2POperator operator_ = build_static_p2p_operator(
        positions, positions, interactions, SourceGeometry::RectangularPrism,
        prisms, {}, TargetGeometry::RectangularPrism, prisms, {});

    REQUIRE(operator_.blocks.size() == interactions.size());

    // Group the built blocks by their exact displacement and require that a
    // repeated displacement carries repeated bits.
    bool saw_a_repeat = false;
    for (std::size_t left = 0; left < operator_.blocks.size(); ++left) {
        const Vec3 left_displacement =
            positions[static_cast<std::size_t>(operator_.blocks[left].target)] -
            positions[static_cast<std::size_t>(operator_.blocks[left].source)];
        for (std::size_t right = left + 1; right < operator_.blocks.size();
             ++right) {
            const Vec3 right_displacement =
                positions[static_cast<std::size_t>(
                    operator_.blocks[right].target)] -
                positions[static_cast<std::size_t>(
                    operator_.blocks[right].source)];
            if (!(same_bits(left_displacement.x, right_displacement.x) &&
                  same_bits(left_displacement.y, right_displacement.y) &&
                  same_bits(left_displacement.z, right_displacement.z))) {
                continue;
            }
            saw_a_repeat = true;
            REQUIRE(same_tensor(operator_.blocks[left],
                                operator_.blocks[right]));
        }
    }
    REQUIRE(saw_a_repeat);
}

TEST_CASE("Reused exact tensors match a pair-by-pair build bit for bit")
{
    // The reference builds one interaction at a time, so no reuse is possible
    // in it.  The batched build must still agree bitwise on every geometry
    // combination the generic and polyhedron loops serve.
    // A small lattice keeps the pair-by-pair reference affordable while still
    // repeating several displacements.
    const std::vector<Vec3> positions = lattice(2, 1.0);
    const std::vector<RectangularPrism> prisms{{0.4, 0.4, 0.4}};
    const std::vector<Tetrahedron> tetrahedra{unit_tetrahedron(0.5)};
    const std::vector<StaticP2PInteraction> interactions =
        all_pairs(positions.size());

    struct Combination {
        SourceGeometry source;
        TargetGeometry target;
        const char* name;
    };
    const std::array<Combination, 8> combinations{{
        {SourceGeometry::PointDipole, TargetGeometry::RectangularPrism,
         "point->prism"},
        {SourceGeometry::RectangularPrism, TargetGeometry::Point,
         "prism->point"},
        {SourceGeometry::RectangularPrism, TargetGeometry::RectangularPrism,
         "prism->prism"},
        {SourceGeometry::PointDipole, TargetGeometry::Tetrahedron,
         "point->tetrahedron"},
        {SourceGeometry::Tetrahedron, TargetGeometry::Point,
         "tetrahedron->point"},
        {SourceGeometry::Tetrahedron, TargetGeometry::Tetrahedron,
         "tetrahedron->tetrahedron"},
        {SourceGeometry::RectangularPrism, TargetGeometry::Tetrahedron,
         "prism->tetrahedron"},
        {SourceGeometry::Tetrahedron, TargetGeometry::RectangularPrism,
         "tetrahedron->prism"},
    }};

    for (const Combination& combination : combinations) {
        INFO(combination.name);
        const StaticP2POperator batched = build_static_p2p_operator(
            positions, positions, interactions, combination.source, prisms,
            tetrahedra, combination.target, prisms, tetrahedra);

        for (std::size_t index = 0; index < interactions.size(); ++index) {
            const int target = batched.blocks[index].target;
            const int source = batched.blocks[index].source;

            // The tetrahedron pair shares one tensor between reciprocal
            // interactions, so the reference has to offer the same partner for
            // the comparison to be bitwise.  Offering it for every combination
            // keeps this one reference shape.
            std::vector<StaticP2PInteraction> reference_pairs{
                {target, source, Vec3{}, true}};
            if (target != source) {
                reference_pairs.push_back({source, target, Vec3{}, true});
            }
            const StaticP2POperator reference = build_static_p2p_operator(
                positions, positions, reference_pairs, combination.source,
                prisms, tetrahedra, combination.target, prisms, tetrahedra);

            const StaticDipoleBlock* wanted = nullptr;
            for (const StaticDipoleBlock& block : reference.blocks) {
                if (block.target == target && block.source == source) {
                    wanted = &block;
                }
            }
            REQUIRE(wanted != nullptr);
            INFO("pair " << index << " target " << target << " source "
                         << source);
            REQUIRE(same_tensor(batched.blocks[index], *wanted));
        }
    }
}

TEST_CASE("Almost identical prism sizes must not share a tensor")
{
    // Two sources one ULP apart in one half-width sit at the same displacement
    // from the target.  A tolerance-based key would merge them; an exact key
    // must not.
    const double base = 0.4;
    const double nudged = std::nextafter(base, 1.0);
    REQUIRE(base != nudged);

    const std::vector<Vec3> targets{{0.0, 0.0, 0.0}};
    const std::vector<Vec3> sources{{1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    const std::vector<RectangularPrism> source_prisms{{base, base, base},
                                                      {nudged, base, base}};
    const std::vector<RectangularPrism> target_prisms{{0.3, 0.3, 0.3}};
    const std::vector<StaticP2PInteraction> interactions{{0, 0, Vec3{}, false},
                                                         {0, 1, Vec3{}, false}};

    const StaticP2POperator operator_ = build_static_p2p_operator(
        targets, sources, interactions, SourceGeometry::RectangularPrism,
        source_prisms, {}, TargetGeometry::RectangularPrism, target_prisms, {});

    REQUIRE(operator_.blocks.size() == 2);
    REQUIRE(any_nonzero(operator_.blocks[0]));
    REQUIRE_FALSE(same_tensor(operator_.blocks[0], operator_.blocks[1]));
}

TEST_CASE("Almost identical displacements must not share a tensor")
{
    const double base = 1.0;
    const double nudged = std::nextafter(base, 2.0);
    REQUIRE(base != nudged);

    const std::vector<Vec3> targets{{0.0, 0.0, 0.0}};
    const std::vector<Vec3> sources{{base, 0.0, 0.0}, {nudged, 0.0, 0.0}};
    const std::vector<RectangularPrism> prisms{{0.4, 0.4, 0.4}};
    const std::vector<StaticP2PInteraction> interactions{{0, 0, Vec3{}, false},
                                                         {0, 1, Vec3{}, false}};

    const StaticP2POperator operator_ = build_static_p2p_operator(
        targets, sources, interactions, SourceGeometry::RectangularPrism,
        prisms, {}, TargetGeometry::RectangularPrism, prisms, {});

    REQUIRE(operator_.blocks.size() == 2);
    REQUIRE_FALSE(same_tensor(operator_.blocks[0], operator_.blocks[1]));
}

TEST_CASE("Different tetrahedron vertices must not share a tensor")
{
    // Two tetrahedra at the same displacement from the target.  The second is
    // stretched along one axis, so the records differ and the operators must
    // differ with them, down to a single perturbed vertex coordinate.
    const Tetrahedron upright = unit_tetrahedron(0.5);
    Tetrahedron stretched = upright;
    stretched.vertices[1].x *= 1.5;

    Tetrahedron nudged = upright;
    nudged.vertices[3].z = std::nextafter(nudged.vertices[3].z, 1.0);
    REQUIRE(nudged.vertices[3].z != upright.vertices[3].z);

    const std::vector<Vec3> targets{{0.0, 0.0, 0.0}};
    const std::vector<Vec3> sources{
        {1.0, 0.25, 0.5}, {1.0, 0.25, 0.5}, {1.0, 0.25, 0.5}};
    const std::vector<Tetrahedron> source_tetrahedra{upright, stretched,
                                                     nudged};
    const std::vector<StaticP2PInteraction> interactions{{0, 0, Vec3{}, false},
                                                         {0, 1, Vec3{}, false},
                                                         {0, 2, Vec3{}, false}};

    const StaticP2POperator operator_ = build_static_p2p_operator(
        targets, sources, interactions, SourceGeometry::Tetrahedron,
        std::span<const RectangularPrism>{}, source_tetrahedra,
        TargetGeometry::Point, std::span<const RectangularPrism>{},
        std::span<const Tetrahedron>{});

    REQUIRE(operator_.blocks.size() == 3);
    REQUIRE(any_nonzero(operator_.blocks[0]));
    REQUIRE_FALSE(same_tensor(operator_.blocks[0], operator_.blocks[1]));
    REQUIRE_FALSE(same_tensor(operator_.blocks[0], operator_.blocks[2]));
}

TEST_CASE("Periodic images that reach one displacement share one tensor")
{
    // The image shift is folded into the displacement before any tensor call,
    // so two different images that land on the same displacement are the same
    // interaction and must carry identical bits; a different image must not.
    const std::vector<Vec3> targets{{0.0, 0.0, 0.0}};
    const std::vector<Vec3> sources{{1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};
    const std::vector<RectangularPrism> prisms{{0.4, 0.4, 0.4}};
    const std::vector<StaticP2PInteraction> interactions{
        // source 0 shifted by +1 and source 1 unshifted both sit at x = 2.
        {0, 0, Vec3{1.0, 0.0, 0.0}, false},
        {0, 1, Vec3{0.0, 0.0, 0.0}, false},
        // A different image, which must stay distinct.
        {0, 1, Vec3{1.0, 0.0, 0.0}, false},
    };

    const StaticP2POperator operator_ = build_static_p2p_operator(
        targets, sources, interactions, SourceGeometry::RectangularPrism,
        prisms, {}, TargetGeometry::RectangularPrism, prisms, {});

    REQUIRE(operator_.blocks.size() == 3);

    // Locate the blocks by the source they name; the builder sorts them.
    const StaticDipoleBlock* shifted_zero = nullptr;
    const StaticDipoleBlock* plain_one = nullptr;
    const StaticDipoleBlock* shifted_one = nullptr;
    for (const StaticDipoleBlock& block : operator_.blocks) {
        if (block.source == 0) {
            shifted_zero = &block;
        } else if (plain_one == nullptr) {
            plain_one = &block;
        } else {
            shifted_one = &block;
        }
    }
    REQUIRE(shifted_zero != nullptr);
    REQUIRE(plain_one != nullptr);
    REQUIRE(shifted_one != nullptr);

    REQUIRE(same_tensor(*shifted_zero, *plain_one));
    REQUIRE_FALSE(same_tensor(*plain_one, *shifted_one));
}

TEST_CASE("The finite self tensor survives exact reuse")
{
    // A finite body on top of itself has a physical self field, and several
    // identical bodies share it exactly.  It must not be confused with the
    // point identity exclusion.
    const std::vector<Vec3> positions = lattice(2, 1.0);
    const std::vector<RectangularPrism> prisms{{0.4, 0.4, 0.4}};
    std::vector<StaticP2PInteraction> interactions;
    for (std::size_t body = 0; body < positions.size(); ++body) {
        interactions.push_back(
            {static_cast<int>(body), static_cast<int>(body), Vec3{}, true});
    }

    const StaticP2POperator operator_ = build_static_p2p_operator(
        positions, positions, interactions, SourceGeometry::RectangularPrism,
        prisms, {}, TargetGeometry::RectangularPrism, prisms, {});

    REQUIRE(operator_.blocks.size() == positions.size());
    for (const StaticDipoleBlock& block : operator_.blocks) {
        // The self field is physical, so it is neither zero nor excluded.
        REQUIRE(any_nonzero(block));
        REQUIRE(block.skip_for_identity == 0);
        REQUIRE(same_tensor(block, operator_.blocks[0]));
    }
}

TEST_CASE("Point self exclusion is preserved alongside reuse")
{
    // A coincident point pair carries an undefined tensor, and the identity
    // marker carries the interaction's own request through unchanged.  A point
    // plan must keep both whether or not anything was reused.
    const std::vector<Vec3> positions{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    const std::vector<StaticP2PInteraction> interactions{
        {0, 0, Vec3{}, true},
        {0, 1, Vec3{}, false},
        {1, 1, Vec3{}, false},
    };

    const StaticP2POperator operator_ = build_static_p2p_operator(
        positions, positions, interactions, SourceGeometry::PointDipole,
        std::span<const RectangularPrism>{}, std::span<const Tetrahedron>{},
        TargetGeometry::Point, std::span<const RectangularPrism>{},
        std::span<const Tetrahedron>{});

    REQUIRE(operator_.blocks.size() == 3);
    for (const StaticDipoleBlock& block : operator_.blocks) {
        if (block.target != block.source) {
            // A separated pair has a finite field and was not asked to be
            // excluded.
            REQUIRE(block.skip_for_identity == 0);
            REQUIRE(any_nonzero(block));
            continue;
        }
        // Coincident point pairs are recorded as undefined, and only the
        // marker says whether the pair was excluded by identity.  The marker
        // reproduces what the interaction asked for.
        REQUIRE(std::isnan(block.xx));
        REQUIRE(block.skip_for_identity == (block.target == 0 ? 1 : 0));
    }
}

TEST_CASE("Per-body prism sizes build their own operators")
{
    // Distinct sizes must reach distinct tensors even at one displacement,
    // and a repeated size at a repeated displacement must reach the same one.
    const std::vector<Vec3> targets{{0.0, 0.0, 0.0}};
    const std::vector<Vec3> sources{
        {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    const std::vector<RectangularPrism> source_prisms{
        {0.4, 0.4, 0.4}, {0.2, 0.3, 0.4}, {0.4, 0.4, 0.4}};
    const std::vector<RectangularPrism> target_prisms{{0.3, 0.3, 0.3}};
    const std::vector<StaticP2PInteraction> interactions{
        {0, 0, Vec3{}, false}, {0, 1, Vec3{}, false}, {0, 2, Vec3{}, false}};

    const StaticP2POperator operator_ = build_static_p2p_operator(
        targets, sources, interactions, SourceGeometry::RectangularPrism,
        source_prisms, {}, TargetGeometry::RectangularPrism, target_prisms, {});

    REQUIRE(operator_.blocks.size() == 3);
    const StaticDipoleBlock& first = operator_.blocks[0];
    const StaticDipoleBlock& second = operator_.blocks[1];
    const StaticDipoleBlock& third = operator_.blocks[2];
    REQUIRE(any_nonzero(first));
    REQUIRE_FALSE(same_tensor(first, second));
    REQUIRE(same_tensor(first, third));
}
