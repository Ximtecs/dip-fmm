// SPDX-License-Identifier: Apache-2.0

#include "internal.hpp"

PYBIND11_MODULE(cdfmm, module)
{
  module.doc() =
      "Fixed-geometry Cartesian and spherical dipole FMM operators";

  // Keep the dependency order explicit. UniformFmmOptions is registered first
  // because tree.cpp registers build_static_fmm with that default argument.
  // The UniformFmm class itself remains after tree registration, matching the
  // monolith's effective return-type registration and docstring.
  cdfmm::python_detail::bind_core(module);
  cdfmm::python_detail::bind_geometry(module);
  cdfmm::python_detail::bind_operators(module);
  cdfmm::python_detail::bind_direct(module);
  cdfmm::python_detail::bind_fmm_options(module);
  cdfmm::python_detail::bind_tree(module);
  cdfmm::python_detail::bind_fmm(module);
}
