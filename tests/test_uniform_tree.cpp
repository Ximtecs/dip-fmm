// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <set>

#include "cdfmm/uniform_tree.hpp"

using namespace cdfmm;

TEST_CASE("Uniform tree node count", "[uniform_tree]")
{
    const std::vector<Vec3> points{{0.0, 0.0, 0.0}};
    REQUIRE(UniformTree(points, UniformTreeOptions{.max_level = 0}).nodes().size() == 1);
    REQUIRE(UniformTree(points, UniformTreeOptions{.max_level = 1}).nodes().size() == 9);
    REQUIRE(UniformTree(points, UniformTreeOptions{.max_level = 2}).nodes().size() == 73);
    REQUIRE(UniformTree(points, UniformTreeOptions{.max_level = 3}).nodes().size() == 585);
}

TEST_CASE("Morton helpers and node indexing", "[uniform_tree]")
{
    const std::array<std::array<int, 3>, 6> coords{{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,1,1},{5,3,7}}};
    for (const auto& c : coords) {
        const auto morton = morton_encode(c[0], c[1], c[2]);
        REQUIRE(morton_decode(morton) == c);
    }
    REQUIRE(node_index(3, 5, 3, 7) == level_offset(3) + static_cast<int>(morton_encode(5, 3, 7)));
}

TEST_CASE("Uniform tree topology and lists", "[uniform_tree]")
{
    std::vector<Vec3> sources{{-0.9,-0.9,-0.9},{0.9,0.9,0.9},{0.1,0.1,0.1},{-0.2,0.4,-0.3}};
    std::vector<Vec3> targets{{0.2,-0.2,0.2},{-0.8,0.8,-0.8}};
    UniformTree tree(sources, targets, UniformTreeOptions{.max_level = 3});
    const auto nodes = tree.nodes();

    REQUIRE(nodes[0].parent == -1);
    REQUIRE(nodes[0].level == 0);
    REQUIRE(nodes[0].list1.size() == 1);
    REQUIRE(nodes[0].list2.empty());

    for (const TreeNode& node : nodes) {
        if (node.level < tree.max_level()) {
            for (const int child : node.children) {
                REQUIRE(child >= 0);
                REQUIRE(nodes[child].parent == node.index);
                REQUIRE(nodes[child].level == node.level + 1);
            }
        } else {
            REQUIRE(node.is_leaf());
        }

        REQUIRE(std::is_sorted(node.list1.begin(), node.list1.end()));
        REQUIRE(std::is_sorted(node.list2.begin(), node.list2.end()));
        REQUIRE(std::set<int>(node.list1.begin(), node.list1.end()).size() == node.list1.size());
        REQUIRE(std::set<int>(node.list2.begin(), node.list2.end()).size() == node.list2.size());
    }

    const int interior = node_index(2, 1, 1, 1);
    REQUIRE(nodes[interior].list1.size() == 27);
    REQUIRE(!nodes[interior].list2.empty());

    REQUIRE(tree.source_permutation().size() == sources.size());
    REQUIRE(tree.source_inverse_permutation().size() == sources.size());
    REQUIRE(tree.target_permutation().size() == targets.size());
    REQUIRE(tree.target_inverse_permutation().size() == targets.size());

    for (std::size_t i = 0; i < sources.size(); ++i) {
        REQUIRE(tree.source_permutation()[tree.source_inverse_permutation()[i]] == static_cast<int>(i));
    }
    for (std::size_t i = 0; i < targets.size(); ++i) {
        REQUIRE(tree.target_permutation()[tree.target_inverse_permutation()[i]] == static_cast<int>(i));
    }

    REQUIRE(nodes[0].source_begin == 0);
    REQUIRE(nodes[0].source_end == sources.size());
    REQUIRE(nodes[0].target_begin == 0);
    REQUIRE(nodes[0].target_end == targets.size());

    // Empty-child sentinels must not extend an occupied internal node's range.
    // Count sorted points whose leaf ancestry contains each node independently
    // of the stored half-open ranges.
    for (const TreeNode& node : nodes) {
        std::size_t expected_source_count = 0;
        for (std::size_t sorted_index = 0;
             sorted_index < sources.size();
             ++sorted_index) {
            int ancestor = tree.leaf_index_for_source(sorted_index);
            while (ancestor >= 0 && nodes[ancestor].level > node.level) {
                ancestor = nodes[ancestor].parent;
            }
            if (ancestor == node.index) {
                ++expected_source_count;
            }
        }
        REQUIRE(node.source_count() == expected_source_count);

        std::size_t expected_target_count = 0;
        for (std::size_t sorted_index = 0;
             sorted_index < targets.size();
             ++sorted_index) {
            int ancestor = tree.leaf_index_for_target(sorted_index);
            while (ancestor >= 0 && nodes[ancestor].level > node.level) {
                ancestor = nodes[ancestor].parent;
            }
            if (ancestor == node.index) {
                ++expected_target_count;
            }
        }
        REQUIRE(node.target_count() == expected_target_count);
    }
}

TEST_CASE("Uniform tree resolves a shared cubic root", "[uniform_tree][root_box]")
{
    const std::vector<Vec3> sources{{-2.0, 0.0, 1.0}};
    const std::vector<Vec3> targets{{4.0, 1.0, -3.0}};
    UniformTree tree(sources, targets, UniformTreeOptions{.max_level = 0});

    CHECK(tree.root_centre().x == 1.0);
    CHECK(tree.root_centre().y == 0.5);
    CHECK(tree.root_centre().z == -1.0);
    CHECK(tree.root_half_width() == 3.0);
    CHECK(tree.nodes().front().source_count() == sources.size());
    CHECK(tree.nodes().front().target_count() == targets.size());
}

TEST_CASE("Uniform tree honours explicit root geometry", "[uniform_tree][root_box]")
{
    const std::vector<Vec3> sources{{-2.0, -1.0, 0.0}};
    const std::vector<Vec3> targets{{3.0, -5.0, 2.0}};

    UniformTreeOptions centre_only;
    centre_only.max_level = 1;
    centre_only.root_centre = Vec3{1.0, -2.0, 0.5};
    const UniformTree centred_tree(
        sources,
        targets,
        centre_only);
    CHECK(centred_tree.root_centre().x == 1.0);
    CHECK(centred_tree.root_centre().y == -2.0);
    CHECK(centred_tree.root_centre().z == 0.5);
    CHECK(centred_tree.root_half_width() == 3.0);

    UniformTreeOptions width_only;
    width_only.max_level = 1;
    width_only.root_half_width = 4.0;
    const UniformTree width_tree(
        sources,
        targets,
        width_only);
    CHECK(width_tree.root_centre().x == 0.5);
    CHECK(width_tree.root_centre().y == -3.0);
    CHECK(width_tree.root_centre().z == 1.0);
    CHECK(width_tree.root_half_width() == 4.0);

    UniformTreeOptions both;
    both.max_level = 1;
    both.root_centre = Vec3{1.0, -2.0, 0.5};
    both.root_half_width = 4.0;
    const UniformTree tree(
        sources,
        targets,
        both);

    CHECK(tree.root_centre().x == 1.0);
    CHECK(tree.root_centre().y == -2.0);
    CHECK(tree.root_centre().z == 0.5);
    CHECK(tree.root_half_width() == 4.0);
}

TEST_CASE("Uniform tree resolves empty and degenerate roots", "[uniform_tree][root_box]")
{
    const std::vector<Vec3> empty;
    const UniformTree empty_tree(empty, UniformTreeOptions{.max_level = 0});
    CHECK(empty_tree.root_centre().x == 0.0);
    CHECK(empty_tree.root_centre().y == 0.0);
    CHECK(empty_tree.root_centre().z == 0.0);
    CHECK(empty_tree.root_half_width() == 1.0);

    const UniformTree coincident_tree(
        std::vector<Vec3>{{2.0, -1.0, 0.5}, {2.0, -1.0, 0.5}},
        UniformTreeOptions{.max_level = 0});
    CHECK(coincident_tree.root_centre().x == 2.0);
    CHECK(coincident_tree.root_centre().y == -1.0);
    CHECK(coincident_tree.root_centre().z == 0.5);
    CHECK(coincident_tree.root_half_width() == 0.0);
}

TEST_CASE("Uniform tree validates requested root bounds", "[uniform_tree][root_box]")
{
    UniformTreeOptions invalid_width;
    invalid_width.root_half_width = 0.0;
    REQUIRE_THROWS_WITH(
        UniformTree(std::vector<Vec3>{{0.0, 0.0, 0.0}}, invalid_width),
        "UniformTreeOptions.root_half_width must be positive");

    UniformTreeOptions requested_box;
    requested_box.root_centre = Vec3{};
    requested_box.root_half_width = 1.0;
    REQUIRE_NOTHROW(UniformTree(
        std::vector<Vec3>{{1.0 + 5.0e-13, 0.0, 0.0}}, requested_box));
    REQUIRE_THROWS_WITH(
        UniformTree(std::vector<Vec3>{{1.0 + 2.0e-12, 0.0, 0.0}}, requested_box),
        "Point lies outside the requested root box");
}

TEST_CASE("Uniform tree build timings are opt-out and change nothing else",
          "[uniform_tree][timing]")
{
    // Scattered points so that every level has occupied and empty boxes and
    // the sorts actually permute.
    std::vector<Vec3> sources;
    std::vector<Vec3> targets;
    for (int index = 0; index < 150; ++index) {
        sources.push_back({
            -0.9 + 1.8 * static_cast<double>((index * 17) % 61) / 60.0,
            -0.9 + 1.8 * static_cast<double>((index * 29) % 67) / 66.0,
            -0.9 + 1.8 * static_cast<double>((index * 43) % 71) / 70.0});
    }
    for (int index = 0; index < 90; ++index) {
        targets.push_back({
            -0.9 + 1.8 * static_cast<double>((index * 31) % 59) / 58.0,
            -0.9 + 1.8 * static_cast<double>((index * 37) % 53) / 52.0,
            -0.9 + 1.8 * static_cast<double>((index * 41) % 47) / 46.0});
    }
    UniformTreeOptions timed;
    timed.max_level = 3;
    REQUIRE(timed.collect_build_timings);
    UniformTreeOptions silent = timed;
    silent.collect_build_timings = false;

    const UniformTree with_timings(sources, targets, timed);
    const UniformTree without_timings(sources, targets, silent);

    // The historical default fills every phase once; the opt-out fills none.
    const TreeBuildTimings& collected = with_timings.build_timings();
    REQUIRE(collected.total.calls == 1);
    REQUIRE(collected.root_bounds.calls == 1);
    REQUIRE(collected.node_construction.calls == 1);
    REQUIRE(collected.topology.calls == 1);
    REQUIRE(collected.source_morton.calls == 1);
    REQUIRE(collected.source_sorting.calls == 1);
    REQUIRE(collected.target_morton.calls == 1);
    REQUIRE(collected.target_sorting.calls == 1);
    REQUIRE(collected.ranges.calls == 1);
    REQUIRE(collected.interaction_lists.calls == 1);
    REQUIRE(collected.total.total_seconds > 0.0);
    const TreeBuildTimings& uncollected = without_timings.build_timings();
    REQUIRE(uncollected.total.calls == 0);
    REQUIRE(uncollected.total.total_seconds == 0.0);
    REQUIRE(uncollected.root_bounds.calls == 0);
    REQUIRE(uncollected.node_construction.calls == 0);
    REQUIRE(uncollected.topology.calls == 0);
    REQUIRE(uncollected.source_morton.calls == 0);
    REQUIRE(uncollected.source_sorting.calls == 0);
    REQUIRE(uncollected.target_morton.calls == 0);
    REQUIRE(uncollected.target_sorting.calls == 0);
    REQUIRE(uncollected.ranges.calls == 0);
    REQUIRE(uncollected.interaction_lists.calls == 0);

    // Everything the tree is for is identical: geometry, every node record
    // including both interaction lists, both permutations and the leaf maps.
    REQUIRE(with_timings.max_level() == without_timings.max_level());
    REQUIRE(with_timings.root_centre().x == without_timings.root_centre().x);
    REQUIRE(with_timings.root_centre().y == without_timings.root_centre().y);
    REQUIRE(with_timings.root_centre().z == without_timings.root_centre().z);
    REQUIRE(with_timings.root_half_width() == without_timings.root_half_width());
    const auto nodes_a = with_timings.nodes();
    const auto nodes_b = without_timings.nodes();
    REQUIRE(nodes_a.size() == nodes_b.size());
    for (std::size_t index = 0; index < nodes_a.size(); ++index) {
        const TreeNode& a = nodes_a[index];
        const TreeNode& b = nodes_b[index];
        REQUIRE(a.index == b.index);
        REQUIRE(a.level == b.level);
        REQUIRE(a.parent == b.parent);
        REQUIRE(a.children == b.children);
        REQUIRE(a.morton_index == b.morton_index);
        REQUIRE(a.source_begin == b.source_begin);
        REQUIRE(a.source_end == b.source_end);
        REQUIRE(a.target_begin == b.target_begin);
        REQUIRE(a.target_end == b.target_end);
        REQUIRE(a.list1 == b.list1);
        REQUIRE(a.list2 == b.list2);
    }
    const auto same_ints = [](std::span<const int> a, std::span<const int> b) {
        return std::vector<int>(a.begin(), a.end()) ==
            std::vector<int>(b.begin(), b.end());
    };
    REQUIRE(same_ints(with_timings.source_permutation(),
                      without_timings.source_permutation()));
    REQUIRE(same_ints(with_timings.source_inverse_permutation(),
                      without_timings.source_inverse_permutation()));
    REQUIRE(same_ints(with_timings.target_permutation(),
                      without_timings.target_permutation()));
    REQUIRE(same_ints(with_timings.target_inverse_permutation(),
                      without_timings.target_inverse_permutation()));
    REQUIRE(same_ints(with_timings.leaf_indices(),
                      without_timings.leaf_indices()));
    REQUIRE(same_ints(with_timings.occupied_source_leaves(),
                      without_timings.occupied_source_leaves()));
    REQUIRE(same_ints(with_timings.occupied_target_leaves(),
                      without_timings.occupied_target_leaves()));
    const auto sorted_a = with_timings.sorted_source_positions();
    const auto sorted_b = without_timings.sorted_source_positions();
    REQUIRE(sorted_a.size() == sorted_b.size());
    for (std::size_t index = 0; index < sorted_a.size(); ++index) {
        REQUIRE(sorted_a[index].x == sorted_b[index].x);
        REQUIRE(sorted_a[index].y == sorted_b[index].y);
        REQUIRE(sorted_a[index].z == sorted_b[index].z);
    }
}
