// SPDX-License-Identifier: Apache-2.0

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "cdfmm/operators.hpp"
#include "cdfmm/uniform_tree.hpp"

namespace py = pybind11;
using namespace cdfmm;

static Vec3 to_vec3(py::handle h)
{
    const auto a = py::cast<py::array_t<double>>(h);
    auto r = a.unchecked<1>();
    return {r(0), r(1), r(2)};
}

static std::vector<Vec3> parse_vec3_list(const py::handle& input)
{
    std::vector<Vec3> values;
    for (const auto& item : py::reinterpret_borrow<py::iterable>(input)) {
        if (py::isinstance<py::sequence>(item)) {
            const auto seq = py::reinterpret_borrow<py::sequence>(item);
            if (seq.size() != 3) {
                throw std::invalid_argument("Each point must have exactly three components");
            }
            values.push_back({
                py::cast<double>(seq[0]),
                py::cast<double>(seq[1]),
                py::cast<double>(seq[2])
            });
        } else if (py::isinstance<Vec3>(item)) {
            values.push_back(py::cast<Vec3>(item));
        } else {
            throw std::invalid_argument("Points must be Vec3 objects or length-3 sequences");
        }
    }
    return values;
}

static OutputFlags parse(std::string s)
{
    if (s == "field") {
        return OutputFlags::Field;
    }
    if (s == "potential") {
        return OutputFlags::Potential;
    }
    return OutputFlags::Both;
}

PYBIND11_MODULE(cdfmm, m)
{
    py::class_<Vec3>(m, "Vec3")
        .def(py::init<double, double, double>())
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z);

    py::class_<TreeNode>(m, "TreeNode")
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
        .def_property_readonly("target_count", &TreeNode::target_count);

    py::class_<UniformTreeOptions>(m, "UniformTreeOptions")
        .def(py::init<>())
        .def_readwrite("max_level", &UniformTreeOptions::max_level)
        .def_readwrite("include_empty_nodes", &UniformTreeOptions::include_empty_nodes)
        .def_readwrite("cubic_root_box", &UniformTreeOptions::cubic_root_box)
        .def_readwrite("root_centre", &UniformTreeOptions::root_centre)
        .def_readwrite("root_half_width", &UniformTreeOptions::root_half_width);

    py::class_<UniformTree>(m, "UniformTree")
        .def(
            py::init([](py::object source_positions, const UniformTreeOptions& options) {
                return UniformTree(parse_vec3_list(source_positions), options);
            }),
            py::arg("source_positions"),
            py::arg("options")
        )
        .def(
            py::init([](py::object source_positions, py::object target_positions, const UniformTreeOptions& options) {
                return UniformTree(
                    parse_vec3_list(source_positions),
                    parse_vec3_list(target_positions),
                    options
                );
            }),
            py::arg("source_positions"),
            py::arg("target_positions"),
            py::arg("options")
        )
        .def_property_readonly("max_level", &UniformTree::max_level)
        .def_property_readonly("nodes", [](const UniformTree& t) { return std::vector<TreeNode>(t.nodes().begin(), t.nodes().end()); })
        .def_property_readonly("source_permutation", [](const UniformTree& t) { return std::vector<int>(t.source_permutation().begin(), t.source_permutation().end()); })
        .def_property_readonly("target_permutation", [](const UniformTree& t) { return std::vector<int>(t.target_permutation().begin(), t.target_permutation().end()); })
        .def("leaf_indices", &UniformTree::leaf_indices)
        .def("sorted_source_positions", [](const UniformTree& t) {
            const auto src = t.sorted_source_positions();
            py::array_t<double> arr({static_cast<py::ssize_t>(src.size()), static_cast<py::ssize_t>(3)});
            auto a = arr.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(src.size()); ++i) {
                a(i, 0) = src[i].x;
                a(i, 1) = src[i].y;
                a(i, 2) = src[i].z;
            }
            return arr;
        })
        .def("sorted_target_positions", [](const UniformTree& t) {
            const auto tgt = t.sorted_target_positions();
            py::array_t<double> arr({static_cast<py::ssize_t>(tgt.size()), static_cast<py::ssize_t>(3)});
            auto a = arr.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(tgt.size()); ++i) {
                a(i, 0) = tgt[i].x;
                a(i, 1) = tgt[i].y;
                a(i, 2) = tgt[i].z;
            }
            return arr;
        });

    m.def("morton_encode", &morton_encode);

    m.def(
        "p2p_dipole_pair",
        [](py::object target, py::object source, py::object moment,
           std::string output) {
            const auto r = p2p_dipole_pair(to_vec3(target), to_vec3(source),
                                           to_vec3(moment), parse(output));
            py::dict d;
            d["phi"] = r.phi;
            py::array_t<double> H(3);
            auto h = H.mutable_unchecked<1>();
            h(0) = r.H.x;
            h(1) = r.H.y;
            h(2) = r.H.z;
            d["H"] = H;
            return d;
        },
        py::arg("target"), py::arg("source"), py::arg("moment"),
        py::arg("output") = "field");

    m.def(
        "p2p_dipole_sum",
        [](py::object target, py::array_t<double> sources,
           py::array_t<double> moments, std::string output, int self_index) {
            auto s = sources.unchecked<2>();
            auto mm = moments.unchecked<2>();
            std::vector<Vec3> xs;
            std::vector<Vec3> ms;

            for (ssize_t i = 0; i < s.shape(0); ++i) {
                xs.push_back({s(i, 0), s(i, 1), s(i, 2)});
                ms.push_back({mm(i, 0), mm(i, 1), mm(i, 2)});
            }

            const auto r =
                p2p_dipole_sum(to_vec3(target), xs, ms, parse(output), self_index);
            py::dict d;
            d["phi"] = r.phi;
            py::array_t<double> H(3);
            auto h = H.mutable_unchecked<1>();
            h(0) = r.H.x;
            h(1) = r.H.y;
            h(2) = r.H.z;
            d["H"] = H;
            return d;
        },
        py::arg("target"), py::arg("sources"), py::arg("moments"),
        py::arg("output") = "field", py::arg("self_index") = -1);
}
