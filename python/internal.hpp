// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// Binding translation units include the supported public interfaces only. The
// canonical operators umbrella still forwards a few legacy operator entry
// points whose declarations have not migrated; no direct compatibility-header
// include is needed here.
#include "cdfmm/backend/cuda/dense_direct.hpp"
#include "cdfmm/backend/cuda/direct.hpp"
#include "cdfmm/core/output_flags.hpp"
#include "cdfmm/core/precision.hpp"
#include "cdfmm/geometry/models.hpp"
#include "cdfmm/geometry/primitives/rectangular_prism.hpp"
#include "cdfmm/geometry/primitives/tetrahedron.hpp"
#include "cdfmm/math/multi_index.hpp"
#include "cdfmm/math/coefficients.hpp"
#include "cdfmm/math/potential_field.hpp"
#include "cdfmm/math/spherical_harmonics.hpp"
#include "cdfmm/math/vec3.hpp"
#include "cdfmm/operators/operators.hpp"
#include "cdfmm/parameter_selection.hpp"
#include "cdfmm/periodic.hpp"
#include "cdfmm/plan/direct/dense.hpp"
#include "cdfmm/plan/static_plan.hpp"
#include "cdfmm/timings.hpp"
#include "cdfmm/tree/adaptive_tree.hpp"
#include "cdfmm/tree/morton.hpp"
#include "cdfmm/tree/node.hpp"
#include "cdfmm/tree/static_topology.hpp"
#include "cdfmm/tree/uniform_topology.hpp"
#include "cdfmm/tree/uniform_tree.hpp"
#include "cdfmm/uniform_fmm.hpp"
#include "cdfmm/validation.hpp"

namespace py = pybind11;

namespace cdfmm::python_detail {

using DoubleArray =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

StaticPrecision parse_static_precision(const std::string& value);
Vec3 parse_vec3(const py::handle& input, const std::string& argument_name);
std::vector<Vec3> parse_vec3_array(const py::handle& input,
                                   const std::string& argument_name);
std::vector<Vec3> parse_tree_points(const py::handle& input);
Tetrahedron parse_tetrahedron(const py::handle& input,
                              const std::string& argument_name);
MultiIndexSet make_basis(int order);
CoeffVector parse_coefficients(const py::handle& input,
                               const MultiIndexSet& basis,
                               const std::string& argument_name);
OutputFlags parse_output(const std::string& output);

py::array_t<double> coefficients_to_array(const CoeffVector& coefficients);
py::array_t<float> coefficients_to_array(std::span<const float> values);
py::array_t<double> points_to_array(std::span<const Vec3> points);
py::dict potential_field_to_dict(const PotentialField& result);
py::dict potential_fields_to_dict(std::span<const PotentialField> results);
py::dict potential_fields_to_dict(std::span<const FloatPotentialField> results);
py::array_t<double> matrix_to_array(const std::vector<double>& matrix,
                                    int row_count, int column_count);
py::dict evaluation_timings_to_dict(const EvaluationTimings& timings);
py::dict performance_candidate_to_dict(const PerformanceCandidate& candidate);
py::dict accuracy_candidate_to_dict(const AccuracyCandidate& candidate);

void bind_core(py::module_& module);
void bind_geometry(py::module_& module);
void bind_tree(py::module_& module);
void bind_operators(py::module_& module);
void bind_direct(py::module_& module);
void bind_fmm_options(py::module_& module);
void bind_fmm(py::module_& module);

} // namespace cdfmm::python_detail
