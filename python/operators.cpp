// SPDX-License-Identifier: Apache-2.0

#include "internal.hpp"

namespace cdfmm::python_detail {

void bind_operators(py::module_& module)
{
  py::enum_<M2LBackend>(module, "M2LBackend")
      .value("Static", M2LBackend::Static)
      .value("Reference", M2LBackend::Reference);
  py::enum_<ExpansionBasis>(module, "ExpansionBasis")
      .value("Cartesian", ExpansionBasis::Cartesian)
      .value("Spherical", ExpansionBasis::Spherical)
      .value("CARTESIAN", ExpansionBasis::Cartesian)
      .value("SPHERICAL", ExpansionBasis::Spherical);
  py::enum_<SphericalM2LBackend>(module, "SphericalM2LBackend")
      .value("StaticDense", SphericalM2LBackend::StaticDense)
      .value("STATIC_DENSE", SphericalM2LBackend::StaticDense);
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
  py::enum_<SpatialLayout>(module, "SpatialLayout")
      .value("GENERAL", SpatialLayout::General)
      .value("REGULAR_GRID", SpatialLayout::RegularGrid);
  py::enum_<P2PExecutionPacking>(module, "P2PExecutionPacking")
      .value("REFERENCE", P2PExecutionPacking::Reference)
      .value("CANONICAL_AOS", P2PExecutionPacking::CanonicalAos)
      .value("PARTICLE_ROW_SOA", P2PExecutionPacking::ParticleRowSoa)
      .value("TENSOR_DICTIONARY", P2PExecutionPacking::TensorDictionary)
      .value("CUDA_BSR3", P2PExecutionPacking::CudaBsr3)
      .value("LEAF_BLOCK", P2PExecutionPacking::LeafBlock);
  py::enum_<StaticOperatorExecutor>(module, "StaticOperatorExecutor")
      .value("REFERENCE", StaticOperatorExecutor::Reference)
      .value("PORTABLE", StaticOperatorExecutor::Portable)
      .value("ONE_MKL", StaticOperatorExecutor::OneMkl)
      .value("CUDA", StaticOperatorExecutor::Cuda);
  py::class_<StaticExecutionPlan>(module, "StaticExecutionPlan")
      .def_readonly("p2m", &StaticExecutionPlan::p2m)
      .def_readonly("m2m", &StaticExecutionPlan::m2m)
      .def_readonly("m2l", &StaticExecutionPlan::m2l)
      .def_readonly("l2l", &StaticExecutionPlan::l2l)
      .def_readonly("l2p", &StaticExecutionPlan::l2p)
      .def_readonly("p2p", &StaticExecutionPlan::p2p);

  module.def(
      "multi_indices", [](const int order) {
        const MultiIndexSet basis = make_basis(order);
        py::array_t<int> indices({static_cast<py::ssize_t>(basis.size()),
                                  static_cast<py::ssize_t>(3)});
        auto values = indices.mutable_unchecked<2>();
        for (int i = 0; i < basis.size(); ++i) {
          values(i, 0) = basis[i].ax;
          values(i, 1) = basis[i].ay;
          values(i, 2) = basis[i].az;
        }
        return indices;
      },
      py::arg("order"),
      "Return Cartesian multi-indices through order p in coefficient order.");
  module.def(
      "spherical_modes", [](const int order) {
        const SphericalHarmonicBasis basis(order);
        py::array_t<int> modes({static_cast<py::ssize_t>(basis.size()),
                                static_cast<py::ssize_t>(2)});
        auto values = modes.mutable_unchecked<2>();
        for (int index = 0; index < basis.size(); ++index) {
          values(index, 0) = basis[index].l;
          values(index, 1) = basis[index].m;
        }
        return modes;
      },
      py::arg("order"),
      "Return real spherical modes (l,m) in coefficient order.");

  module.def(
      "p2p_dipole_pair",
      [](py::object target, py::object source, py::object moment,
         const std::string& output) {
        const PotentialField result = p2p_dipole_pair(
            parse_vec3(target, "target"), parse_vec3(source, "source"),
            parse_vec3(moment, "moment"), parse_output(output));
        return potential_field_to_dict(result);
      },
      py::arg("target"), py::arg("source"), py::arg("moment"),
      py::arg("output") = "field",
      "Evaluate one point-dipole contribution at one target.");
  module.def(
      "p2p_dipole_sum",
      [](py::object target, py::object sources, py::object moments,
         const std::string& output, const int self_index) {
        const std::vector<Vec3> source_positions =
            parse_vec3_array(sources, "sources");
        const std::vector<Vec3> dipole_moments =
            parse_vec3_array(moments, "moments");
        if (source_positions.size() != dipole_moments.size()) {
          throw std::invalid_argument(
              "sources and moments must contain the same number of rows");
        }
        const PotentialField result = p2p_dipole_sum(
            parse_vec3(target, "target"), source_positions, dipole_moments,
            parse_output(output), self_index);
        return potential_field_to_dict(result);
      },
      py::arg("target"), py::arg("sources"), py::arg("moments"),
      py::arg("output") = "field", py::arg("self_index") = -1,
      "Sum direct point-dipole contributions at one target.");

  module.def(
      "p2m_dipole",
      [](py::object centre, py::object source_positions,
         py::object dipole_moments, const int order) {
        const MultiIndexSet basis = make_basis(order);
        const std::vector<Vec3> positions =
            parse_vec3_array(source_positions, "source_positions");
        const std::vector<Vec3> moments =
            parse_vec3_array(dipole_moments, "dipole_moments");
        if (positions.size() != moments.size()) {
          throw std::invalid_argument(
              "source_positions and dipole_moments must contain the "
              "same number of rows");
        }
        const CoeffVector coefficients =
            p2m_dipole(basis, parse_vec3(centre, "centre"), positions, moments);
        return coefficients_to_array(coefficients);
      },
      py::arg("centre"), py::arg("source_positions"),
      py::arg("dipole_moments"), py::arg("order"),
      R"doc(Build dipole multipole coefficients about an expansion centre.

Coefficients follow ``multi_indices(order)``: total degree first, then
lexicographic ``(alpha_x, alpha_y)`` within each degree.)doc");
  module.def(
      "m2m",
      [](py::object child_coefficients, py::object child_centre,
         py::object parent_centre, const int order) {
        const MultiIndexSet basis = make_basis(order);
        const CoeffVector child = parse_coefficients(
            child_coefficients, basis, "child_coefficients");
        const Vec3 child_position = parse_vec3(child_centre, "child_centre");
        const Vec3 parent_position = parse_vec3(parent_centre, "parent_centre");
        CoeffVector parent(static_cast<std::size_t>(basis.size()), 0.0);
        m2m_add(basis, parent_position - child_position, child, parent);
        return coefficients_to_array(parent);
      },
      py::arg("child_coefficients"), py::arg("child_centre"),
      py::arg("parent_centre"), py::arg("order"),
      "Translate one child multipole expansion to a parent centre.");
  module.def(
      "m2p",
      [](py::object multipole_coefficients, py::object source_centre,
         py::object target_position, const int order,
         const std::string& output) {
        const MultiIndexSet basis = make_basis(order);
        const CoeffVector coefficients = parse_coefficients(
            multipole_coefficients, basis, "multipole_coefficients");
        const PotentialField result = m2p_eval(
            basis, coefficients, parse_vec3(source_centre, "source_centre"),
            parse_vec3(target_position, "target_position"),
            parse_output(output));
        return potential_field_to_dict(result);
      },
      py::arg("multipole_coefficients"), py::arg("source_centre"),
      py::arg("target_position"), py::arg("order"),
      py::arg("output") = "field",
      "Evaluate a multipole expansion at one target position.");
  module.def(
      "m2l",
      [](py::object multipole_coefficients, py::object source_centre,
         py::object target_centre, const int order) {
        const MultiIndexSet basis = make_basis(order);
        const CoeffVector multipole = parse_coefficients(
            multipole_coefficients, basis, "multipole_coefficients");
        const Vec3 source_position = parse_vec3(source_centre, "source_centre");
        const Vec3 target_position = parse_vec3(target_centre, "target_centre");
        CoeffVector local(static_cast<std::size_t>(basis.size()), 0.0);
        m2l_add(basis, target_position - source_position, multipole, local);
        return coefficients_to_array(local);
      },
      py::arg("multipole_coefficients"), py::arg("source_centre"),
      py::arg("target_centre"), py::arg("order"),
      "Convert a multipole expansion to a target-centred local expansion.");
  module.def(
      "static_m2l_matrix",
      [](py::object source_centre, py::object target_centre, const int order) {
        const MultiIndexSet basis = make_basis(order);
        const Vec3 source_position = parse_vec3(source_centre, "source_centre");
        const Vec3 target_position = parse_vec3(target_centre, "target_centre");
        return matrix_to_array(
            build_static_m2l_matrix(basis, target_position - source_position),
            basis.size(), basis.size());
      },
      py::arg("source_centre"), py::arg("target_centre"), py::arg("order"),
      "Return the canonical dense static M2L matrix in output-by-input order.");
  module.def(
      "l2l",
      [](py::object parent_coefficients, py::object parent_centre,
         py::object child_centre, const int order) {
        const MultiIndexSet basis = make_basis(order);
        const CoeffVector parent = parse_coefficients(
            parent_coefficients, basis, "parent_coefficients");
        const Vec3 parent_position = parse_vec3(parent_centre, "parent_centre");
        const Vec3 child_position = parse_vec3(child_centre, "child_centre");
        CoeffVector child(static_cast<std::size_t>(basis.size()), 0.0);
        l2l_add(basis, child_position - parent_position, parent, child);
        return coefficients_to_array(child);
      },
      py::arg("parent_coefficients"), py::arg("parent_centre"),
      py::arg("child_centre"), py::arg("order"),
      "Shift a parent local expansion to a child target centre.");
  module.def(
      "l2p",
      [](py::object local_coefficients, py::object local_centre,
         py::object target_position, const int order,
         const std::string& output) {
        const MultiIndexSet basis = make_basis(order);
        const CoeffVector coefficients =
            parse_coefficients(local_coefficients, basis, "local_coefficients");
        const PotentialField result = l2p_eval(
            basis, parse_vec3(local_centre, "local_centre"),
            parse_vec3(target_position, "target_position"), coefficients,
            parse_output(output));
        return potential_field_to_dict(result);
      },
      py::arg("local_coefficients"), py::arg("local_centre"),
      py::arg("target_position"), py::arg("order"),
      py::arg("output") = "field", "Evaluate a local expansion at one target position.");
}

} // namespace cdfmm::python_detail
