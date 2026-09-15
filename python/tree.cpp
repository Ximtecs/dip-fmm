// SPDX-License-Identifier: Apache-2.0

#include "internal.hpp"

namespace cdfmm::python_detail {

void bind_tree(py::module_& module)
{
  py::class_<TreeNode>(module, "TreeNode")
      .def_readonly("index", &TreeNode::index)
      .def_readonly("level", &TreeNode::level)
      .def_readonly("parent", &TreeNode::parent)
      .def_readonly("children", &TreeNode::children)
      .def_readonly("ix", &TreeNode::ix)
      .def_readonly("iy", &TreeNode::iy)
      .def_readonly("iz", &TreeNode::iz)
      .def_readonly("morton_index", &TreeNode::morton_index)
      .def_readonly("centre", &TreeNode::centre)
      .def_readonly("half_width", &TreeNode::half_width)
      .def_readonly("source_begin", &TreeNode::source_begin)
      .def_readonly("source_end", &TreeNode::source_end)
      .def_readonly("target_begin", &TreeNode::target_begin)
      .def_readonly("target_end", &TreeNode::target_end)
      .def_readonly("list1", &TreeNode::list1)
      .def_readonly("list2", &TreeNode::list2)
      .def_property_readonly("source_count", &TreeNode::source_count)
      .def_property_readonly("target_count", &TreeNode::target_count)
      .def_property_readonly("is_leaf", &TreeNode::is_leaf);

  py::class_<UniformTreeOptions>(module, "UniformTreeOptions")
      .def(py::init<>())
      .def_readwrite("max_level", &UniformTreeOptions::max_level)
      .def_readwrite("include_empty_nodes",
                     &UniformTreeOptions::include_empty_nodes)
      .def_readwrite("cubic_root_box", &UniformTreeOptions::cubic_root_box)
      .def_readwrite("root_centre", &UniformTreeOptions::root_centre)
      .def_readwrite("root_half_width", &UniformTreeOptions::root_half_width);

  py::enum_<PeriodicConvention>(module, "PeriodicConvention")
      .value("ZeroK0", PeriodicConvention::ZeroK0);
  py::class_<PeriodicCellOptions>(module, "PeriodicCellOptions")
      .def(py::init<>())
      .def_readwrite("enabled", &PeriodicCellOptions::enabled)
      .def_readwrite("axes", &PeriodicCellOptions::axes)
      .def_readwrite("centre", &PeriodicCellOptions::centre)
      .def_readwrite("lengths", &PeriodicCellOptions::lengths)
      .def_readwrite("convention", &PeriodicCellOptions::convention)
      .def_readwrite("setup_tolerance", &PeriodicCellOptions::setup_tolerance);

  py::class_<UniformTree>(module, "UniformTree")
      .def(py::init([](py::object source_positions,
                       const UniformTreeOptions& options) {
             return UniformTree(parse_tree_points(source_positions), options);
           }),
           py::arg("source_positions"), py::arg("options"))
      .def(py::init([](py::object source_positions, py::object target_positions,
                       const UniformTreeOptions& options) {
             return UniformTree(parse_tree_points(source_positions),
                                parse_tree_points(target_positions), options);
           }),
           py::arg("source_positions"), py::arg("target_positions"),
           py::arg("options"))
      .def_property_readonly("max_level", &UniformTree::max_level)
      .def_property_readonly("n_levels", &UniformTree::n_levels)
      .def_property_readonly("leaf_level", &UniformTree::leaf_level)
      .def_property_readonly("root_centre", &UniformTree::root_centre)
      .def_property_readonly("root_half_width", &UniformTree::root_half_width)
      .def_property_readonly("nodes", [](const UniformTree& tree) {
        return std::vector<TreeNode>(tree.nodes().begin(), tree.nodes().end());
      })
      .def_property_readonly("source_permutation", [](const UniformTree& tree) {
        return std::vector<int>(tree.source_permutation().begin(),
                                tree.source_permutation().end());
      })
      .def_property_readonly("source_inverse_permutation",
                             [](const UniformTree& tree) {
        return std::vector<int>(tree.source_inverse_permutation().begin(),
                                tree.source_inverse_permutation().end());
      })
      .def_property_readonly("target_permutation", [](const UniformTree& tree) {
        return std::vector<int>(tree.target_permutation().begin(),
                                tree.target_permutation().end());
      })
      .def_property_readonly("target_inverse_permutation",
                             [](const UniformTree& tree) {
        return std::vector<int>(tree.target_inverse_permutation().begin(),
                                tree.target_inverse_permutation().end());
      })
      .def("leaf_indices", [](const UniformTree& tree) {
        const auto indices = tree.leaf_indices();
        return std::vector<int>(indices.begin(), indices.end());
      })
      .def("leaf_index_for_source", &UniformTree::leaf_index_for_source)
      .def("leaf_index_for_target", &UniformTree::leaf_index_for_target)
      .def("sorted_source_positions", [](const UniformTree& tree) {
        // Return owned storage rather than a view tied to the tree.
        return points_to_array(tree.sorted_source_positions());
      })
      .def("sorted_target_positions", [](const UniformTree& tree) {
        return points_to_array(tree.sorted_target_positions());
      });

  py::class_<StaticLeafRange>(module, "StaticLeafRange")
      .def_readonly("node", &StaticLeafRange::node)
      .def_readonly("begin", &StaticLeafRange::begin)
      .def_readonly("count", &StaticLeafRange::count);
  py::class_<StaticFmmTopology::Node>(module, "StaticNode")
      .def_readonly("index", &StaticFmmTopology::Node::index)
      .def_readonly("level", &StaticFmmTopology::Node::level)
      .def_readonly("parent", &StaticFmmTopology::Node::parent)
      .def_readonly("children", &StaticFmmTopology::Node::children)
      .def_readonly("centre", &StaticFmmTopology::Node::centre)
      .def_readonly("half_width", &StaticFmmTopology::Node::half_width)
      .def_readonly("source_begin", &StaticFmmTopology::Node::source_begin)
      .def_readonly("source_end", &StaticFmmTopology::Node::source_end)
      .def_readonly("target_begin", &StaticFmmTopology::Node::target_begin)
      .def_readonly("target_end", &StaticFmmTopology::Node::target_end)
      .def_property_readonly("source_count",
                             &StaticFmmTopology::Node::source_count)
      .def_property_readonly("target_count",
                             &StaticFmmTopology::Node::target_count);
  py::class_<StaticM2LInteraction>(module, "StaticM2LInteraction")
      .def_readonly("source_node", &StaticM2LInteraction::source_node)
      .def_readonly("target_node", &StaticM2LInteraction::target_node)
      .def_readonly("source_level", &StaticM2LInteraction::source_level)
      .def_readonly("target_level", &StaticM2LInteraction::target_level);
  py::class_<StaticP2PLeafRecord>(module, "StaticP2PLeafRecord")
      .def_readonly("source_leaf", &StaticP2PLeafRecord::source_leaf)
      .def_readonly("target_leaf", &StaticP2PLeafRecord::target_leaf);
  py::class_<StaticFmmTopology, std::shared_ptr<StaticFmmTopology>>(
      module, "StaticFmmTopology")
      .def_readonly("nodes", &StaticFmmTopology::nodes)
      .def_property_readonly("sorted_source_positions",
                             [](const StaticFmmTopology& topology) {
        return points_to_array(topology.sorted_source_positions);
      })
      .def_property_readonly("sorted_target_positions",
                             [](const StaticFmmTopology& topology) {
        return points_to_array(topology.sorted_target_positions);
      })
      .def_readonly("m2l_target_row_offsets",
                    &StaticFmmTopology::m2l_target_row_offsets)
      .def_readonly("p2p_target_leaf_offsets",
                    &StaticFmmTopology::p2p_target_leaf_offsets)
      .def_readonly("source_leaves", &StaticFmmTopology::source_leaves)
      .def_readonly("target_leaves", &StaticFmmTopology::target_leaves)
      .def_readonly("source_permutation", &StaticFmmTopology::source_permutation)
      .def_readonly("target_permutation", &StaticFmmTopology::target_permutation)
      .def_readonly("m2l_interactions", &StaticFmmTopology::m2l_interactions)
      .def_readonly("p2p_leaf_records", &StaticFmmTopology::p2p_leaf_records)
      .def_readonly("maximum_level", &StaticFmmTopology::maximum_level)
      .def_readonly("root", &StaticFmmTopology::root)
      .def_readonly("coordinate_origin", &StaticFmmTopology::coordinate_origin)
      .def_readonly("coordinate_scale", &StaticFmmTopology::coordinate_scale)
      .def_property_readonly("memory_bytes", &StaticFmmTopology::memory_bytes)
      .def("validate", &StaticFmmTopology::validate);

  py::class_<AdaptiveTreeOptions>(module, "AdaptiveTreeOptions")
      .def(py::init<>())
      .def_readwrite("max_particles_per_leaf",
                     &AdaptiveTreeOptions::max_particles_per_leaf)
      .def_readwrite("max_depth", &AdaptiveTreeOptions::max_depth)
      .def_readwrite("root_centre", &AdaptiveTreeOptions::root_centre)
      .def_readwrite("root_half_width", &AdaptiveTreeOptions::root_half_width);
  py::class_<AdaptiveTree>(module, "AdaptiveTree")
      .def(py::init([](py::object sources,
                       const AdaptiveTreeOptions& options) {
        return AdaptiveTree(parse_tree_points(sources), options);
      }), py::arg("sources"), py::arg("options") = AdaptiveTreeOptions{})
      .def(py::init([](py::object sources, py::object targets,
                       const AdaptiveTreeOptions& options) {
        return AdaptiveTree(parse_tree_points(sources),
                            parse_tree_points(targets), options);
      }), py::arg("sources"), py::arg("targets"),
           py::arg("options") = AdaptiveTreeOptions{})
      .def_property_readonly("topology", [](const AdaptiveTree& tree) {
        return std::const_pointer_cast<StaticFmmTopology>(tree.shared_topology());
      })
      .def_property_readonly("leaf_stop_reasons", [](const AdaptiveTree& tree) {
        py::dict reasons;
        for (const auto& node : tree.topology().nodes) {
          if (std::any_of(node.children.begin(), node.children.end(),
                          [](int id) { return id >= 0; })) {
            continue;
          }
          reasons[py::int_(node.index)] =
              std::max(node.source_count(), node.target_count()) >
                      tree.options().max_particles_per_leaf
                  ? "depth_limit"
                  : "capacity";
        }
        return reasons;
      })
      .def_property_readonly("tree_seconds", &AdaptiveTree::tree_seconds)
      .def_property_readonly("interaction_seconds",
                             &AdaptiveTree::interaction_seconds)
      .def("build_fmm", [](const AdaptiveTree& tree,
                             const UniformFmmOptions& options) {
        return std::make_unique<UniformFmm>(tree.shared_topology(), options);
      }, py::arg("options") = UniformFmmOptions{});

  module.def("uniform_topology", [](const UniformTree& tree) {
    auto topology =
        std::make_shared<StaticFmmTopology>(build_uniform_fmm_topology(tree));
    topology->coordinate_origin = tree.root_centre();
    topology->coordinate_scale = 2.0 * tree.root_half_width();
    for (auto& node : topology->nodes) {
      node.centre = (node.centre - topology->coordinate_origin) *
                    (1.0 / topology->coordinate_scale);
      node.half_width /= topology->coordinate_scale;
    }
    for (auto& point : topology->sorted_source_positions) {
      point = (point - topology->coordinate_origin) *
              (1.0 / topology->coordinate_scale);
    }
    for (auto& point : topology->sorted_target_positions) {
      point = (point - topology->coordinate_origin) *
              (1.0 / topology->coordinate_scale);
    }
    for (auto& interaction : topology->m2l_interactions) {
      interaction.displacement =
          interaction.displacement * (1.0 / topology->coordinate_scale);
    }
    return topology;
  });
  module.def("build_static_fmm",
             [](std::shared_ptr<StaticFmmTopology> topology,
                const UniformFmmOptions& options) {
               return std::make_unique<UniformFmm>(std::move(topology), options);
             },
             py::arg("topology"), py::arg("options") = UniformFmmOptions{});

  module.def("morton_encode", &morton_encode);
  module.def("morton_decode", &morton_decode);
}

} // namespace cdfmm::python_detail
