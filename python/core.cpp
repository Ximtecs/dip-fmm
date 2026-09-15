// SPDX-License-Identifier: Apache-2.0

#include "internal.hpp"

namespace cdfmm::python_detail {

StaticPrecision parse_static_precision(const std::string& value)
{
    if (value == "float32") {
        return StaticPrecision::Float32;
    }
    if (value == "float64") {
        return StaticPrecision::Float64;
    }
    throw std::invalid_argument(
        "static_precision must be 'float32' or 'float64'");
}

py::dict evaluation_timings_to_dict(const EvaluationTimings& timings)
{
  py::dict result;
  const auto add = [&result](const char* name, const PhaseTiming& phase) {
    result[name] = phase.total_seconds;
  };
  add("moment_permutation", timings.moment_permutation);
  add("multipole_reset", timings.multipole_reset);
  add("p2m", timings.p2m);
  add("m2m", timings.m2m);
  add("local_reset", timings.local_reset);
  add("m2l", timings.m2l);
  add("m2l_scale", timings.m2l_scale);
  add("m2l_gather", timings.m2l_gather);
  add("m2l_multiply", timings.m2l_multiply);
  add("m2l_scatter", timings.m2l_scatter);
  add("l2l", timings.l2l);
  add("l2p", timings.l2p);
  add("p2p", timings.p2p);
  add("result_unpermutation", timings.result_unpermutation);
  // Preserve the independent CUDA lanes as diagnostics. These durations may
  // overlap and must not be summed as a sequential critical path.
  add("cuda_h2d", timings.cuda_h2d);
  add("cuda_kernel", timings.cuda_kernel);
  add("cuda_d2h", timings.cuda_d2h);
  add("cuda_m2l_h2d", timings.cuda_m2l_h2d);
  add("cuda_m2l_d2h", timings.cuda_m2l_d2h);
  add("cuda_p2p_h2d", timings.cuda_p2p_h2d);
  add("cuda_p2p_kernel", timings.cuda_p2p_kernel);
  add("cuda_p2p_d2h", timings.cuda_p2p_d2h);
  add("cuda_p2p_wait", timings.cuda_p2p_wait);
  add("total", timings.total);
  result["evaluations"] = timings.evaluations;
  return result;
}

Vec3 parse_vec3(const py::handle& input, const std::string& argument_name)
{
  const DoubleArray array = py::cast<DoubleArray>(input);
  const py::buffer_info buffer = array.request();

  if (buffer.ndim != 1 || buffer.shape[0] != 3) {
    throw std::invalid_argument(argument_name + " must have shape (3,)");
  }

  const auto values = array.unchecked<1>();
  return {values(0), values(1), values(2)};
}

std::vector<Vec3> parse_vec3_array(const py::handle& input,
                                   const std::string& argument_name)
{
  const DoubleArray array = py::cast<DoubleArray>(input);
  const py::buffer_info buffer = array.request();

  if (buffer.ndim != 2 || buffer.shape[1] != 3) {
    throw std::invalid_argument(argument_name + " must have shape (n, 3)");
  }

  const auto values = array.unchecked<2>();
  std::vector<Vec3> result;
  result.reserve(static_cast<std::size_t>(buffer.shape[0]));

  for (py::ssize_t i = 0; i < buffer.shape[0]; ++i) {
    result.push_back({values(i, 0), values(i, 1), values(i, 2)});
  }

  return result;
}

Tetrahedron parse_tetrahedron(const py::handle& input,
                              const std::string& argument_name)
{
  const std::vector<Vec3> vertices = parse_vec3_array(input, argument_name);
  if (vertices.size() != 4) {
    throw std::invalid_argument(argument_name + " must have shape (4, 3)");
  }
  Tetrahedron result;
  std::copy(vertices.begin(), vertices.end(), result.vertices.begin());
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
            "Each point must have exactly three components");
      }
      values.push_back({py::cast<double>(sequence[0]),
                        py::cast<double>(sequence[1]),
                        py::cast<double>(sequence[2])});
    } else if (py::isinstance<Vec3>(item)) {
      values.push_back(py::cast<Vec3>(item));
    } else {
      throw std::invalid_argument(
          "Points must be Vec3 objects or length-three sequences");
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

CoeffVector parse_coefficients(const py::handle& input,
                               const MultiIndexSet& basis,
                               const std::string& argument_name)
{
  const DoubleArray array = py::cast<DoubleArray>(input);
  const py::buffer_info buffer = array.request();

  if (buffer.ndim != 1 || buffer.shape[0] != basis.size()) {
    throw std::invalid_argument(
        argument_name + " must have shape (coefficient_count,) for order " +
        std::to_string(basis.order()));
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
  throw std::invalid_argument("output must be 'field', 'potential', or 'both'");
}

py::array_t<double> coefficients_to_array(const CoeffVector& coefficients)
{
  py::array_t<double> array(coefficients.size());
  auto values = array.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(coefficients.size()); ++i) {
    values(i) = coefficients[static_cast<std::size_t>(i)];
  }
  return array;
}

py::array_t<float> coefficients_to_array(std::span<const float> values)
{
  py::array_t<float> array(values.size());
  auto output = array.mutable_unchecked<1>();
  for (py::ssize_t index = 0;
       index < static_cast<py::ssize_t>(values.size()); ++index) {
    output(index) = values[static_cast<std::size_t>(index)];
  }
  return array;
}

py::array_t<double> points_to_array(std::span<const Vec3> points)
{
  py::array_t<double> array({static_cast<py::ssize_t>(points.size()),
                             static_cast<py::ssize_t>(3)});
  auto values = array.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(points.size()); ++i) {
    values(i, 0) = points[static_cast<std::size_t>(i)].x;
    values(i, 1) = points[static_cast<std::size_t>(i)].y;
    values(i, 2) = points[static_cast<std::size_t>(i)].z;
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
  py::array_t<double> field({static_cast<py::ssize_t>(results.size()),
                             static_cast<py::ssize_t>(3)});
  auto potential_values = potential.mutable_unchecked<1>();
  auto field_values = field.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(results.size()); ++i) {
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

py::dict potential_fields_to_dict(std::span<const FloatPotentialField> results)
{
  py::dict output;
  py::array_t<float> potential(results.size());
  py::array_t<float> field({static_cast<py::ssize_t>(results.size()),
                            static_cast<py::ssize_t>(3)});
  auto potential_values = potential.mutable_unchecked<1>();
  auto field_values = field.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(results.size()); ++i) {
    const FloatPotentialField& result = results[static_cast<std::size_t>(i)];
    potential_values(i) = result.phi;
    field_values(i, 0) = result.H.x;
    field_values(i, 1) = result.H.y;
    field_values(i, 2) = result.H.z;
  }
  output["phi"] = potential;
  output["H"] = field;
  return output;
}

py::array_t<double> matrix_to_array(const std::vector<double>& matrix,
                                    const int row_count,
                                    const int column_count)
{
  py::array_t<double> array({row_count, column_count});
  auto values = array.mutable_unchecked<2>();
  for (int column = 0; column < column_count; ++column) {
    for (int row = 0; row < row_count; ++row) {
      values(row, column) = matrix[static_cast<std::size_t>(row) +
                                  static_cast<std::size_t>(row_count) * column];
    }
  }
  return array;
}

py::dict performance_candidate_to_dict(const PerformanceCandidate& candidate)
{
  py::dict result;
  result["depth"] = candidate.depth;
  result["status"] = candidate.succeeded ? "ok" : "failed";
  result["reason"] = candidate.reason;
  result["near_seconds"] = candidate.near_seconds;
  result["far_seconds"] = candidate.far_seconds;
  result["balance_ratio"] = candidate.balance_ratio;
  result["evaluation_seconds"] = candidate.evaluation_seconds;
  result["estimated_concurrent_time"] = candidate.estimated_concurrent_seconds;
  return result;
}

py::dict accuracy_candidate_to_dict(const AccuracyCandidate& candidate)
{
  py::dict result;
  result["order"] = candidate.order;
  result["depth"] = candidate.depth;
  result["status"] = candidate.succeeded ? "ok" : "failed";
  result["reason"] = candidate.reason;
  result["satisfies_accuracy"] = candidate.satisfies_accuracy;
  result["evaluation_seconds"] = candidate.evaluation_seconds;
  result["mean_relative_error"] = candidate.mean_relative_error;
  result["rms_relative_error"] = candidate.rms_relative_error;
  result["maximum_relative_error"] = candidate.maximum_relative_error;
  result["mean_absolute_error"] = candidate.mean_absolute_error;
  result["maximum_absolute_error"] = candidate.maximum_absolute_error;
  return result;
}

void bind_core(py::module_& module)
{
  py::class_<Vec3>(module, "Vec3")
      .def(py::init<double, double, double>())
      .def_readwrite("x", &Vec3::x)
      .def_readwrite("y", &Vec3::y)
      .def_readwrite("z", &Vec3::z);
}

} // namespace cdfmm::python_detail
