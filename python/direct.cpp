// SPDX-License-Identifier: Apache-2.0

#include "internal.hpp"

namespace cdfmm::python_detail {

void bind_direct(py::module_& module)
{
  module.def("cuda_compiled", &cuda_compiled);
  module.def("one_mkl_available", &one_mkl_available);
  module.def("cuda_available", &cuda_available);
  module.def("cuda_direct_available", &cuda_direct_available);
  module.def("cuda_m2l_p2p_available", &cuda_m2l_p2p_available);
  module.def("cuda_m2l_available", &cuda_m2l_available);
  module.def("cuda_full_available", &cuda_full_available);
  module.def("cuda_device_description", &cuda_device_description);

  py::class_<CudaDirectPlan>(
      module, "CudaDirectPlan",
      "Persistent O(N^2) CUDA direct plan for fixed source/target geometry.")
      .def(py::init([](py::object source_positions,
                       py::object target_positions,
                       py::object target_source_indices) {
        const std::vector<Vec3> sources =
            parse_vec3_array(source_positions, "source_positions");
        const std::vector<Vec3> targets =
            parse_vec3_array(target_positions, "target_positions");
        const std::vector<int> identities = target_source_indices.is_none()
                                                ? std::vector<int>{}
                                                : py::cast<std::vector<int>>(
                                                      target_source_indices);
        std::unique_ptr<CudaDirectPlan> plan;
        {
          py::gil_scoped_release release;
          plan = std::make_unique<CudaDirectPlan>(sources, targets, identities);
        }
        return plan;
      }),
           py::arg("source_positions"), py::arg("target_positions"),
           py::arg("target_source_indices") = py::none(),
           "Create a persistent CUDA direct plan for fixed geometry.")
      .def("evaluate", [](CudaDirectPlan& plan, py::object dipole_moments,
                           const std::string& output) {
        const std::vector<Vec3> moments =
            parse_vec3_array(dipole_moments, "dipole_moments");
        const OutputFlags output_flags = parse_output(output);
        std::vector<PotentialField> results(plan.target_count());
        {
          py::gil_scoped_release release;
          plan.evaluate(moments, results, output_flags);
        }
        return potential_fields_to_dict(results);
      },
           py::arg("dipole_moments"), py::arg("output") = "field",
           "Evaluate one moment state using the persistent CUDA geometry.")
      .def_property_readonly("source_count", &CudaDirectPlan::source_count)
      .def_property_readonly("target_count", &CudaDirectPlan::target_count);

  module.def(
      "direct_p2p_reference",
      [](py::object target_positions, py::object source_positions,
         py::object dipole_moments, const std::string& output,
         py::object target_source_indices) {
        const std::vector<Vec3> targets =
            parse_vec3_array(target_positions, "target_positions");
        const std::vector<Vec3> sources =
            parse_vec3_array(source_positions, "source_positions");
        const std::vector<Vec3> moments =
            parse_vec3_array(dipole_moments, "dipole_moments");
        const OutputFlags output_flags = parse_output(output);
        const std::vector<int> identities = target_source_indices.is_none()
                                                ? std::vector<int>{}
                                                : py::cast<std::vector<int>>(
                                                      target_source_indices);
        std::vector<PotentialField> results;
        {
          py::gil_scoped_release release;
          results = direct_p2p_reference(targets, sources, moments,
                                         output_flags, identities);
        }
        return potential_fields_to_dict(results);
      },
      py::arg("target_positions"), py::arg("source_positions"),
      py::arg("dipole_moments"), py::arg("output") = "field",
      py::arg("target_source_indices") = py::none(),
      "Evaluate the batched O(N^2) direct dipole reference on the CPU.");
  module.def(
      "cuda_direct_p2p_reference",
      [](py::object target_positions, py::object source_positions,
         py::object dipole_moments, const std::string& output,
         py::object target_source_indices) {
        const std::vector<Vec3> targets =
            parse_vec3_array(target_positions, "target_positions");
        const std::vector<Vec3> sources =
            parse_vec3_array(source_positions, "source_positions");
        const std::vector<Vec3> moments =
            parse_vec3_array(dipole_moments, "dipole_moments");
        const OutputFlags output_flags = parse_output(output);
        const std::vector<int> identities = target_source_indices.is_none()
                                                ? std::vector<int>{}
                                                : py::cast<std::vector<int>>(
                                                      target_source_indices);
        std::vector<PotentialField> results;
        {
          py::gil_scoped_release release;
          results = cuda_direct_p2p_reference(targets, sources, moments,
                                              output_flags, identities);
        }
        return potential_fields_to_dict(results);
      },
      py::arg("target_positions"), py::arg("source_positions"),
      py::arg("dipole_moments"), py::arg("output") = "field",
      py::arg("target_source_indices") = py::none(),
      "Evaluate the batched O(N^2) direct dipole reference on a CUDA device.");

  py::class_<DenseDirectPlan>(module, "DenseDirectPlan")
      .def(py::init([](py::object sources, py::object targets,
                       SourceGeometry source_geometry,
                       TargetGeometry target_geometry,
                       const std::vector<RectangularPrism>& source_sizes,
                       const std::vector<RectangularPrism>& target_sizes,
                       const std::vector<int>& identities,
                       const std::string& static_precision,
                       const std::vector<Tetrahedron>& source_tetrahedra,
                       const std::vector<Tetrahedron>& target_tetrahedra,
                       const SourceModel source_model,
                       const TargetModel target_model) {
        return std::make_unique<DenseDirectPlan>(
            parse_vec3_array(sources, "source_positions"),
            parse_vec3_array(targets, "target_positions"), source_geometry,
            target_geometry, source_sizes, target_sizes, identities,
            parse_static_precision(static_precision), source_tetrahedra,
            target_tetrahedra, source_model, target_model);
      }),
           py::arg("source_positions"), py::arg("target_positions"),
           py::arg("source_geometry") = SourceGeometry::PointDipole,
           py::arg("target_geometry") = TargetGeometry::Point,
           py::arg("source_sizes") = std::vector<RectangularPrism>{},
           py::arg("target_sizes") = std::vector<RectangularPrism>{},
           py::arg("target_source_indices") = std::vector<int>{},
           py::arg("static_precision") = "float32",
           py::arg("source_tetrahedra") = std::vector<Tetrahedron>{},
           py::arg("target_tetrahedra") = std::vector<Tetrahedron>{},
           py::arg("source_model") = SourceModel::ExactGeometry,
           py::arg("target_model") = TargetModel::ExactGeometry)
      .def("evaluate", [](const DenseDirectPlan& plan, py::object moments,
                           const DenseDirectBackend backend) {
        const std::vector<Vec3> parsed_moments =
            parse_vec3_array(moments, "total_moments");
        std::vector<Vec3> result;
        {
          py::gil_scoped_release release;
          result = plan.evaluate(parsed_moments, backend);
        }
        return points_to_array(result);
      },
           py::arg("total_moments"),
           py::arg("backend") = DenseDirectBackend::Automatic)
      .def_property_readonly("source_count", &DenseDirectPlan::source_count)
      .def_property_readonly("target_count", &DenseDirectPlan::target_count)
      .def_property_readonly("tensor_memory_bytes",
                             &DenseDirectPlan::tensor_memory_bytes)
      .def_property_readonly("static_precision",
                             &DenseDirectPlan::static_precision)
      .def_property_readonly("tensor_component_count",
                             &DenseDirectPlan::tensor_component_count);

  module.def("dense_direct_mkl_available", &dense_direct_mkl_available);
  module.def("cuda_dense_direct_available", &cuda_dense_direct_available);
  py::class_<CudaDenseDirectPlan>(module, "CudaDenseDirectPlan")
      .def(py::init([](py::object sources, py::object targets,
                       SourceGeometry source_geometry,
                       TargetGeometry target_geometry,
                       const std::vector<RectangularPrism>& source_sizes,
                       const std::vector<RectangularPrism>& target_sizes,
                       const std::vector<int>& identities,
                       const std::string& static_precision,
                       const std::vector<Tetrahedron>& source_tetrahedra,
                       const std::vector<Tetrahedron>& target_tetrahedra,
                       const SourceModel source_model,
                       const TargetModel target_model) {
        const std::vector<Vec3> parsed_sources =
            parse_vec3_array(sources, "source_positions");
        const std::vector<Vec3> parsed_targets =
            parse_vec3_array(targets, "target_positions");
        std::unique_ptr<CudaDenseDirectPlan> plan;
        {
          py::gil_scoped_release release;
          plan = std::make_unique<CudaDenseDirectPlan>(
              parsed_sources, parsed_targets, source_geometry, target_geometry,
              source_sizes, target_sizes, identities,
              parse_static_precision(static_precision), source_tetrahedra,
              target_tetrahedra, source_model, target_model);
        }
        return plan;
      }),
           py::arg("source_positions"), py::arg("target_positions"),
           py::arg("source_geometry") = SourceGeometry::PointDipole,
           py::arg("target_geometry") = TargetGeometry::Point,
           py::arg("source_sizes") = std::vector<RectangularPrism>{},
           py::arg("target_sizes") = std::vector<RectangularPrism>{},
           py::arg("target_source_indices") = std::vector<int>{},
           py::arg("static_precision") = "float64",
           py::arg("source_tetrahedra") = std::vector<Tetrahedron>{},
           py::arg("target_tetrahedra") = std::vector<Tetrahedron>{},
           py::arg("source_model") = SourceModel::ExactGeometry,
           py::arg("target_model") = TargetModel::ExactGeometry)
      .def("evaluate", [](CudaDenseDirectPlan& plan, py::object moments) {
        const std::vector<Vec3> parsed_moments =
            parse_vec3_array(moments, "total_moments");
        std::vector<Vec3> result;
        {
          py::gil_scoped_release release;
          result = plan.evaluate(parsed_moments);
        }
        return points_to_array(result);
      }, py::arg("total_moments"))
      .def_property_readonly("source_count", &CudaDenseDirectPlan::source_count)
      .def_property_readonly("target_count", &CudaDenseDirectPlan::target_count)
      .def_property_readonly("tensor_memory_bytes",
                             &CudaDenseDirectPlan::tensor_memory_bytes)
      .def_property_readonly("persistent_device_bytes",
                             &CudaDenseDirectPlan::persistent_device_bytes)
      .def_property_readonly("static_precision",
                             &CudaDenseDirectPlan::static_precision);
}

} // namespace cdfmm::python_detail
