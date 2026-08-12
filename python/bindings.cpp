// SPDX-License-Identifier: Apache-2.0

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "cdfmm/operators.hpp"
#include "cdfmm/cuda_direct.hpp"
#include "cdfmm/uniform_fmm.hpp"
#include "cdfmm/uniform_tree.hpp"

namespace py = pybind11;
using namespace cdfmm;

namespace {

using DoubleArray = py::array_t<
    double,
    py::array::c_style | py::array::forcecast
>;

//------------------------------------------------------------------------------
// Python conversion helpers
//------------------------------------------------------------------------------

Vec3 parse_vec3(const py::handle& input, const std::string& argument_name)
{
    const DoubleArray array = py::cast<DoubleArray>(input);
    const py::buffer_info buffer = array.request();

    if (buffer.ndim != 1 || buffer.shape[0] != 3) {
        throw std::invalid_argument(
            argument_name + " must have shape (3,)"
        );
    }

    const auto values = array.unchecked<1>();
    return {values(0), values(1), values(2)};
}

std::vector<Vec3> parse_vec3_array(
    const py::handle& input,
    const std::string& argument_name
)
{
    const DoubleArray array = py::cast<DoubleArray>(input);
    const py::buffer_info buffer = array.request();

    if (buffer.ndim != 2 || buffer.shape[1] != 3) {
        throw std::invalid_argument(
            argument_name + " must have shape (n, 3)"
        );
    }

    const auto values = array.unchecked<2>();
    std::vector<Vec3> result;
    result.reserve(static_cast<std::size_t>(buffer.shape[0]));

    for (py::ssize_t i = 0; i < buffer.shape[0]; ++i) {
        result.push_back({values(i, 0), values(i, 1), values(i, 2)});
    }

    return result;
}

std::vector<Vec3> parse_tree_points(const py::handle& input)
{
    // Tree construction retains support for bound Vec3 instances while the
    // numerical operator interface uses shape-checked NumPy conversion.
    std::vector<Vec3> values;
    for (const auto& item : py::reinterpret_borrow<py::iterable>(input)) {
        if (py::isinstance<py::sequence>(item)) {
            const auto sequence = py::reinterpret_borrow<py::sequence>(item);
            if (sequence.size() != 3) {
                throw std::invalid_argument(
                    "Each point must have exactly three components"
                );
            }
            values.push_back({
                py::cast<double>(sequence[0]),
                py::cast<double>(sequence[1]),
                py::cast<double>(sequence[2])
            });
        } else if (py::isinstance<Vec3>(item)) {
            values.push_back(py::cast<Vec3>(item));
        } else {
            throw std::invalid_argument(
                "Points must be Vec3 objects or length-three sequences"
            );
        }
    }

    return values;
}

MultiIndexSet make_basis(int order)
{
    if (order < 0) {
        throw std::invalid_argument("order must be non-negative");
    }

    return MultiIndexSet(order);
}

CoeffVector parse_coefficients(
    const py::handle& input,
    const MultiIndexSet& basis,
    const std::string& argument_name
)
{
    const DoubleArray array = py::cast<DoubleArray>(input);
    const py::buffer_info buffer = array.request();

    if (buffer.ndim != 1 || buffer.shape[0] != basis.size()) {
        throw std::invalid_argument(
            argument_name + " must have shape (coefficient_count,) for order "
            + std::to_string(basis.order())
        );
    }

    const auto values = array.unchecked<1>();
    CoeffVector result(static_cast<std::size_t>(basis.size()));
    for (int i = 0; i < basis.size(); ++i) {
        result[static_cast<std::size_t>(i)] = values(i);
    }

    return result;
}

OutputFlags parse_output(const std::string& output)
{
    if (output == "field") {
        return OutputFlags::Field;
    }
    if (output == "potential") {
        return OutputFlags::Potential;
    }
    if (output == "both") {
        return OutputFlags::Both;
    }

    throw std::invalid_argument(
        "output must be 'field', 'potential', or 'both'"
    );
}

py::array_t<double> coefficients_to_array(const CoeffVector& coefficients)
{
    py::array_t<double> array(coefficients.size());
    auto values = array.mutable_unchecked<1>();

    for (py::ssize_t i = 0;
         i < static_cast<py::ssize_t>(coefficients.size());
         ++i) {
        values(i) = coefficients[static_cast<std::size_t>(i)];
    }

    return array;
}

py::array_t<double> matrix_to_array(
    const std::vector<double>& matrix,
    const int row_count,
    const int column_count)
{
    py::array_t<double> array({row_count, column_count});
    auto values = array.mutable_unchecked<2>();
    for (int column = 0; column < column_count; ++column) {
        for (int row = 0; row < row_count; ++row) {
            values(row, column) = matrix[
                static_cast<std::size_t>(row) +
                static_cast<std::size_t>(row_count) * column
            ];
        }
    }
    return array;
}

py::dict potential_field_to_dict(const PotentialField& result)
{
    py::dict output;
    output["phi"] = result.phi;

    py::array_t<double> field(3);
    auto values = field.mutable_unchecked<1>();
    values(0) = result.H.x;
    values(1) = result.H.y;
    values(2) = result.H.z;
    output["H"] = field;

    return output;
}

py::dict potential_fields_to_dict(std::span<const PotentialField> results)
{
    py::dict output;
    py::array_t<double> potential(results.size());
    py::array_t<double> field({
        static_cast<py::ssize_t>(results.size()),
        static_cast<py::ssize_t>(3)
    });
    auto potential_values = potential.mutable_unchecked<1>();
    auto field_values = field.mutable_unchecked<2>();

    for (py::ssize_t i = 0;
         i < static_cast<py::ssize_t>(results.size());
         ++i) {
        const PotentialField& result = results[static_cast<std::size_t>(i)];
        potential_values(i) = result.phi;
        field_values(i, 0) = result.H.x;
        field_values(i, 1) = result.H.y;
        field_values(i, 2) = result.H.z;
    }
    output["phi"] = potential;
    output["H"] = field;
    return output;
}

py::array_t<double> points_to_array(std::span<const Vec3> points)
{
    py::array_t<double> array({
        static_cast<py::ssize_t>(points.size()),
        static_cast<py::ssize_t>(3)
    });
    auto values = array.mutable_unchecked<2>();

    for (py::ssize_t i = 0;
         i < static_cast<py::ssize_t>(points.size());
         ++i) {
        values(i, 0) = points[static_cast<std::size_t>(i)].x;
        values(i, 1) = points[static_cast<std::size_t>(i)].y;
        values(i, 2) = points[static_cast<std::size_t>(i)].z;
    }

    return array;
}

} // namespace

//------------------------------------------------------------------------------
// Python module interface
//------------------------------------------------------------------------------

PYBIND11_MODULE(cdfmm, module)
{
    module.doc() =
        "Cartesian dipole FMM reference operators and uniform-tree geometry";

    py::class_<Vec3>(module, "Vec3")
        .def(py::init<double, double, double>())
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z);

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
        .def_readwrite(
            "include_empty_nodes",
            &UniformTreeOptions::include_empty_nodes
        )
        .def_readwrite("cubic_root_box", &UniformTreeOptions::cubic_root_box)
        .def_readwrite("root_centre", &UniformTreeOptions::root_centre)
        .def_readwrite("root_half_width", &UniformTreeOptions::root_half_width);

    py::class_<UniformTree>(module, "UniformTree")
        .def(
            py::init([](
                py::object source_positions,
                const UniformTreeOptions& options
            ) {
                return UniformTree(
                    parse_tree_points(source_positions),
                    options
                );
            }),
            py::arg("source_positions"),
            py::arg("options")
        )
        .def(
            py::init([](
                py::object source_positions,
                py::object target_positions,
                const UniformTreeOptions& options
            ) {
                return UniformTree(
                    parse_tree_points(source_positions),
                    parse_tree_points(target_positions),
                    options
                );
            }),
            py::arg("source_positions"),
            py::arg("target_positions"),
            py::arg("options")
        )
        .def_property_readonly("max_level", &UniformTree::max_level)
        .def_property_readonly("n_levels", &UniformTree::n_levels)
        .def_property_readonly("leaf_level", &UniformTree::leaf_level)
        .def_property_readonly("root_centre", &UniformTree::root_centre)
        .def_property_readonly("root_half_width", &UniformTree::root_half_width)
        .def_property_readonly("nodes", [](const UniformTree& tree) {
            return std::vector<TreeNode>(
                tree.nodes().begin(),
                tree.nodes().end()
            );
        })
        .def_property_readonly(
            "source_permutation",
            [](const UniformTree& tree) {
                return std::vector<int>(
                    tree.source_permutation().begin(),
                    tree.source_permutation().end()
                );
            }
        )
        .def_property_readonly(
            "source_inverse_permutation",
            [](const UniformTree& tree) {
                return std::vector<int>(
                    tree.source_inverse_permutation().begin(),
                    tree.source_inverse_permutation().end()
                );
            }
        )
        .def_property_readonly(
            "target_permutation",
            [](const UniformTree& tree) {
                return std::vector<int>(
                    tree.target_permutation().begin(),
                    tree.target_permutation().end()
                );
            }
        )
        .def_property_readonly(
            "target_inverse_permutation",
            [](const UniformTree& tree) {
                return std::vector<int>(
                    tree.target_inverse_permutation().begin(),
                    tree.target_inverse_permutation().end()
                );
            }
        )
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

    py::enum_<M2LBackend>(module, "M2LBackend")
        .value("Static", M2LBackend::Static)
        .value("Reference", M2LBackend::Reference);

    py::enum_<StaticMatrixBackend>(module, "StaticMatrixBackend")
        .value("PORTABLE", StaticMatrixBackend::Portable)
        .value("ONE_MKL", StaticMatrixBackend::OneMkl);

    py::enum_<ExecutionBackend>(module, "ExecutionBackend")
        .value("AUTO", ExecutionBackend::Auto)
        .value("CPU_REFERENCE", ExecutionBackend::CpuReference)
        .value("CPU_STATIC", ExecutionBackend::CpuStatic)
        .value("CUDA_M2L_P2P", ExecutionBackend::CudaM2LP2P)
        .value("CUDA_PARTIAL", ExecutionBackend::CudaPartial)
        .value("CUDA_FULL", ExecutionBackend::CudaFull)
        .value("CUDA_M2L", ExecutionBackend::CudaM2L)
        .value("CUDA_M2L_STATIC_P2P", ExecutionBackend::CudaM2LStaticP2P);

    module.def("cuda_compiled", &cuda_compiled);
    module.def("one_mkl_available", &one_mkl_available);
    module.def("cuda_available", &cuda_available);
    module.def("cuda_direct_available", &cuda_direct_available);
    module.def("cuda_m2l_p2p_available", &cuda_m2l_p2p_available);
    module.def("cuda_m2l_available", &cuda_m2l_available);
    module.def("cuda_full_available", &cuda_full_available);
    module.def("cuda_device_description", &cuda_device_description);

    py::class_<UniformFmmOptions>(module, "UniformFmmOptions")
        .def(py::init<>())
        .def_readwrite(
            "expansion_order",
            &UniformFmmOptions::expansion_order
        )
        .def_readwrite("tree", &UniformFmmOptions::tree)
        .def_readwrite("m2l_backend", &UniformFmmOptions::m2l_backend)
        .def_readwrite(
            "static_matrix_backend",
            &UniformFmmOptions::static_matrix_backend
        )
        .def_readwrite("backend", &UniformFmmOptions::backend);

    py::class_<UniformFmm>(module, "UniformFmm")
        .def(
            py::init([](
                py::object source_positions,
                const UniformFmmOptions& options
            ) {
                return UniformFmm(
                    parse_vec3_array(source_positions, "source_positions"),
                    options
                );
            }),
            py::arg("source_positions"),
            py::arg("options") = UniformFmmOptions{}
        )
        .def(
            py::init([](
                py::object source_positions,
                py::object target_positions,
                const UniformFmmOptions& options
            ) {
                return UniformFmm(
                    parse_vec3_array(source_positions, "source_positions"),
                    parse_vec3_array(target_positions, "target_positions"),
                    options
                );
            }),
            py::arg("source_positions"),
            py::arg("target_positions"),
            py::arg("options") = UniformFmmOptions{}
        )
        .def(
            "upward_pass",
            [](UniformFmm& fmm, py::object dipole_moments) {
                fmm.upward_pass(
                    parse_vec3_array(dipole_moments, "dipole_moments")
                );
            },
            py::arg("dipole_moments"),
            "Replace all node multipoles using moments in original source order."
        )
        .def("downward_pass", &UniformFmm::downward_pass)
        .def(
            "evaluate",
            [](UniformFmm& fmm,
               py::object dipole_moments,
               const std::string& output,
               py::object target_source_indices) {
                const std::vector<Vec3> moments = parse_vec3_array(
                    dipole_moments,
                    "dipole_moments"
                );
                std::vector<int> identities;
                if (!target_source_indices.is_none()) {
                    identities = py::cast<std::vector<int>>(
                        target_source_indices
                    );
                }
                return potential_fields_to_dict(
                    fmm.evaluate(moments, parse_output(output), identities)
                );
            },
            py::arg("dipole_moments"),
            py::arg("output") = "field",
            py::arg("target_source_indices") = py::none(),
            "Run the complete FMM and return values in target order."
        )
        .def_property_readonly(
            "tree",
            &UniformFmm::tree,
            py::return_value_policy::reference_internal
        )
        .def_property_readonly("expansion_order", [](const UniformFmm& fmm) {
            return fmm.basis().order();
        })
        .def_property_readonly("m2l_backend", &UniformFmm::m2l_backend)
        .def_property_readonly("backend", &UniformFmm::backend)
        .def_property_readonly("cuda_plan_statistics", [](const UniformFmm& fmm) {
            const CudaPlanStatistics& statistics = fmm.cuda_plan_statistics();
            py::dict result;
            result["setup_h2d_bytes"] = statistics.setup_h2d_bytes;
            result["evaluation_h2d_bytes"] = statistics.evaluation_h2d_bytes;
            result["evaluation_d2h_bytes"] = statistics.evaluation_d2h_bytes;
            result["evaluation_h2d_calls"] = statistics.evaluation_h2d_calls;
            result["evaluation_d2h_calls"] = statistics.evaluation_d2h_calls;
            result["persistent_device_bytes"] = statistics.persistent_device_bytes;
            result["plan_generation_count"] = statistics.plan_generation_count;
            result["static_upload_count"] = statistics.static_upload_count;
            result["static_m2l_upload_count"] = statistics.static_m2l_upload_count;
            return result;
        })
        .def_property_readonly("static_plan_statistics", [](const UniformFmm& fmm) {
            const StaticPlanStatistics& statistics = fmm.static_plan_statistics();
            py::dict result;
            result["transfer_classes"] = statistics.transfer_classes;
            result["interactions"] = statistics.interactions;
            result["operator_bytes"] = statistics.operator_bytes;
            result["interaction_bytes"] = statistics.interaction_bytes;
            result["scratch_bytes"] = statistics.scratch_bytes;
            result["total_bytes"] = statistics.total_bytes();
            result["setup_seconds"] = statistics.total.total_seconds;
            return result;
        })
        .def_property_readonly("root_multipole", [](const UniformFmm& fmm) {
            const auto coefficients = fmm.root_multipole();
            return coefficients_to_array(
                CoeffVector(coefficients.begin(), coefficients.end())
            );
        })
        .def("multipole", [](const UniformFmm& fmm, const int node_index) {
            const auto coefficients = fmm.multipole(node_index);
            return coefficients_to_array(
                CoeffVector(coefficients.begin(), coefficients.end())
            );
        }, py::arg("node_index"))
        .def("local", [](const UniformFmm& fmm, const int node_index) {
            const auto coefficients = fmm.local(node_index);
            return coefficients_to_array(
                CoeffVector(coefficients.begin(), coefficients.end())
            );
        }, py::arg("node_index"));

    module.def("morton_encode", &morton_encode);
    module.def("morton_decode", &morton_decode);

    module.def(
        "multi_indices",
        [](int order) {
            const MultiIndexSet basis = make_basis(order);
            py::array_t<int> indices({
                static_cast<py::ssize_t>(basis.size()),
                static_cast<py::ssize_t>(3)
            });
            auto values = indices.mutable_unchecked<2>();

            for (int i = 0; i < basis.size(); ++i) {
                values(i, 0) = basis[i].ax;
                values(i, 1) = basis[i].ay;
                values(i, 2) = basis[i].az;
            }

            return indices;
        },
        py::arg("order"),
        "Return Cartesian multi-indices through order p in coefficient order."
    );

    module.def(
        "p2p_dipole_pair",
        [](py::object target,
           py::object source,
           py::object moment,
           const std::string& output) {
            const PotentialField result = p2p_dipole_pair(
                parse_vec3(target, "target"),
                parse_vec3(source, "source"),
                parse_vec3(moment, "moment"),
                parse_output(output)
            );
            return potential_field_to_dict(result);
        },
        py::arg("target"),
        py::arg("source"),
        py::arg("moment"),
        py::arg("output") = "field",
        "Evaluate one point-dipole contribution at one target."
    );

    module.def(
        "p2p_dipole_sum",
        [](py::object target,
           py::object sources,
           py::object moments,
           const std::string& output,
           int self_index) {
            const std::vector<Vec3> source_positions =
                parse_vec3_array(sources, "sources");
            const std::vector<Vec3> dipole_moments =
                parse_vec3_array(moments, "moments");

            if (source_positions.size() != dipole_moments.size()) {
                throw std::invalid_argument(
                    "sources and moments must contain the same number of rows"
                );
            }

            const PotentialField result = p2p_dipole_sum(
                parse_vec3(target, "target"),
                source_positions,
                dipole_moments,
                parse_output(output),
                self_index
            );
            return potential_field_to_dict(result);
        },
        py::arg("target"),
        py::arg("sources"),
        py::arg("moments"),
        py::arg("output") = "field",
        py::arg("self_index") = -1,
        "Sum direct point-dipole contributions at one target."
    );

    module.def(
        "p2m_dipole",
        [](py::object centre,
           py::object source_positions,
           py::object dipole_moments,
           int order) {
            const MultiIndexSet basis = make_basis(order);
            const std::vector<Vec3> positions = parse_vec3_array(
                source_positions,
                "source_positions"
            );
            const std::vector<Vec3> moments = parse_vec3_array(
                dipole_moments,
                "dipole_moments"
            );

            if (positions.size() != moments.size()) {
                throw std::invalid_argument(
                    "source_positions and dipole_moments must contain the "
                    "same number of rows"
                );
            }

            const CoeffVector coefficients = p2m_dipole(
                basis,
                parse_vec3(centre, "centre"),
                positions,
                moments
            );
            return coefficients_to_array(coefficients);
        },
        py::arg("centre"),
        py::arg("source_positions"),
        py::arg("dipole_moments"),
        py::arg("order"),
        R"doc(Build dipole multipole coefficients about an expansion centre.

Coefficients follow ``multi_indices(order)``: total degree first, then
lexicographic ``(alpha_x, alpha_y)`` within each degree.)doc"
    );

    module.def(
        "m2m",
        [](py::object child_coefficients,
           py::object child_centre,
           py::object parent_centre,
           int order) {
            const MultiIndexSet basis = make_basis(order);
            const CoeffVector child = parse_coefficients(
                child_coefficients,
                basis,
                "child_coefficients"
            );
            const Vec3 child_position = parse_vec3(
                child_centre,
                "child_centre"
            );
            const Vec3 parent_position = parse_vec3(
                parent_centre,
                "parent_centre"
            );
            CoeffVector parent(static_cast<std::size_t>(basis.size()), 0.0);

            m2m_add(
                basis,
                parent_position - child_position,
                child,
                parent
            );
            return coefficients_to_array(parent);
        },
        py::arg("child_coefficients"),
        py::arg("child_centre"),
        py::arg("parent_centre"),
        py::arg("order"),
        "Translate one child multipole expansion to a parent centre."
    );

    module.def(
        "m2p",
        [](py::object multipole_coefficients,
           py::object source_centre,
           py::object target_position,
           int order,
           const std::string& output) {
            const MultiIndexSet basis = make_basis(order);
            const CoeffVector coefficients = parse_coefficients(
                multipole_coefficients,
                basis,
                "multipole_coefficients"
            );
            const PotentialField result = m2p_eval(
                basis,
                coefficients,
                parse_vec3(source_centre, "source_centre"),
                parse_vec3(target_position, "target_position"),
                parse_output(output)
            );
            return potential_field_to_dict(result);
        },
        py::arg("multipole_coefficients"),
        py::arg("source_centre"),
        py::arg("target_position"),
        py::arg("order"),
        py::arg("output") = "field",
        "Evaluate a multipole expansion at one target position."
    );

    module.def(
        "m2l",
        [](py::object multipole_coefficients,
           py::object source_centre,
           py::object target_centre,
           int order) {
            const MultiIndexSet basis = make_basis(order);
            const CoeffVector multipole = parse_coefficients(
                multipole_coefficients,
                basis,
                "multipole_coefficients"
            );
            const Vec3 source_position = parse_vec3(
                source_centre,
                "source_centre"
            );
            const Vec3 target_position = parse_vec3(
                target_centre,
                "target_centre"
            );
            CoeffVector local(static_cast<std::size_t>(basis.size()), 0.0);

            m2l_add(
                basis,
                target_position - source_position,
                multipole,
                local
            );
            return coefficients_to_array(local);
        },
        py::arg("multipole_coefficients"),
        py::arg("source_centre"),
        py::arg("target_centre"),
        py::arg("order"),
        "Convert a multipole expansion to a target-centred local expansion."
    );

    module.def(
        "static_m2l_matrix",
        [](py::object source_centre,
           py::object target_centre,
           const int order) {
            const MultiIndexSet basis = make_basis(order);
            const Vec3 source_position = parse_vec3(
                source_centre,
                "source_centre"
            );
            const Vec3 target_position = parse_vec3(
                target_centre,
                "target_centre"
            );
            return matrix_to_array(
                build_static_m2l_matrix(
                    basis,
                    target_position - source_position
                ),
                basis.size(),
                basis.size()
            );
        },
        py::arg("source_centre"),
        py::arg("target_centre"),
        py::arg("order"),
        "Return the canonical dense static M2L matrix in output-by-input order."
    );

    module.def(
        "l2l",
        [](py::object parent_coefficients,
           py::object parent_centre,
           py::object child_centre,
           int order) {
            const MultiIndexSet basis = make_basis(order);
            const CoeffVector parent = parse_coefficients(
                parent_coefficients,
                basis,
                "parent_coefficients"
            );
            const Vec3 parent_position = parse_vec3(
                parent_centre,
                "parent_centre"
            );
            const Vec3 child_position = parse_vec3(
                child_centre,
                "child_centre"
            );
            CoeffVector child(static_cast<std::size_t>(basis.size()), 0.0);

            l2l_add(
                basis,
                child_position - parent_position,
                parent,
                child
            );
            return coefficients_to_array(child);
        },
        py::arg("parent_coefficients"),
        py::arg("parent_centre"),
        py::arg("child_centre"),
        py::arg("order"),
        "Shift a parent local expansion to a child target centre."
    );

    module.def(
        "l2p",
        [](py::object local_coefficients,
           py::object local_centre,
           py::object target_position,
           int order,
           const std::string& output) {
            const MultiIndexSet basis = make_basis(order);
            const CoeffVector coefficients = parse_coefficients(
                local_coefficients,
                basis,
                "local_coefficients"
            );
            const PotentialField result = l2p_eval(
                basis,
                parse_vec3(local_centre, "local_centre"),
                parse_vec3(target_position, "target_position"),
                coefficients,
                parse_output(output)
            );
            return potential_field_to_dict(result);
        },
        py::arg("local_coefficients"),
        py::arg("local_centre"),
        py::arg("target_position"),
        py::arg("order"),
        py::arg("output") = "field",
        "Evaluate a local expansion at one target position."
    );
}
