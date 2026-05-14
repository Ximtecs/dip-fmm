// SPDX-License-Identifier: Apache-2.0

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "cdfmm/operators.hpp"

namespace py = pybind11;
using namespace cdfmm;

static Vec3 to_vec3(py::handle h) {
    const auto a = py::cast<py::array_t<double>>(h);
    auto r = a.unchecked<1>();
    return {r(0), r(1), r(2)};
}

static OutputFlags parse(std::string s) {
    if (s == "field") {
        return OutputFlags::Field;
    }
    if (s == "potential") {
        return OutputFlags::Potential;
    }
    return OutputFlags::Both;
}

PYBIND11_MODULE(cdfmm, m) {
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
