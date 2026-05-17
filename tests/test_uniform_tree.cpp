// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

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
}
