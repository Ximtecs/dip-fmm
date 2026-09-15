// SPDX-License-Identifier: Apache-2.0

#include "internal.hpp"

namespace cdfmm::python_detail {

void bind_geometry(py::module_& module)
{
  // Geometry records are intentionally named after their physical meaning;
  // the former CuboidSize spelling is not part of the Python API.
  py::class_<RectangularPrism>(module, "RectangularPrism")
      .def(py::init<double, double, double>(), py::arg("hx"),
           py::arg("hy"), py::arg("hz"))
      .def_readwrite("hx", &RectangularPrism::hx)
      .def_readwrite("hy", &RectangularPrism::hy)
      .def_readwrite("hz", &RectangularPrism::hz)
      .def_property_readonly("volume", &RectangularPrism::volume);
  py::class_<Tetrahedron>(module, "Tetrahedron")
      .def(py::init<>())
      .def(py::init([](py::object vertices) {
        return parse_tetrahedron(vertices, "vertices");
      }), py::arg("vertices"))
      .def_readwrite("vertices", &Tetrahedron::vertices)
      .def_property_readonly("signed_volume", &Tetrahedron::signed_volume)
      .def_property_readonly("volume", &Tetrahedron::volume)
      .def_property_readonly("centroid_offset", &Tetrahedron::centroid_offset);

  py::enum_<SourceGeometry>(module, "SourceGeometry")
      .value("POINT_DIPOLE", SourceGeometry::PointDipole)
      .value("RECTANGULAR_PRISM", SourceGeometry::RectangularPrism)
      .value("TETRAHEDRON", SourceGeometry::Tetrahedron);
  py::enum_<TargetGeometry>(module, "TargetGeometry")
      .value("POINT", TargetGeometry::Point)
      .value("RECTANGULAR_PRISM", TargetGeometry::RectangularPrism)
      .value("TETRAHEDRON", TargetGeometry::Tetrahedron);
  py::enum_<SourceModel>(module, "SourceModel")
      .value("POINT_DIPOLE", SourceModel::PointDipole)
      .value("EXACT_GEOMETRY", SourceModel::ExactGeometry);
  py::enum_<TargetModel>(module, "TargetModel")
      .value("POINT", TargetModel::Point)
      .value("EXACT_GEOMETRY", TargetModel::ExactGeometry);
  py::enum_<DenseDirectBackend>(module, "DenseDirectBackend")
      .value("AUTOMATIC", DenseDirectBackend::Automatic)
      .value("PORTABLE", DenseDirectBackend::Portable)
      .value("ONE_MKL", DenseDirectBackend::OneMkl);
  py::enum_<StaticPrecision>(module, "StaticPrecision")
      .value("FLOAT32", StaticPrecision::Float32)
      .value("FLOAT64", StaticPrecision::Float64);
}

} // namespace cdfmm::python_detail
