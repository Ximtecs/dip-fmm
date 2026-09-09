# C++ test guidance

Inherit `../AGENTS.md`. Tests are the behavioural contract during the refactor;
do not prune or consolidate them in the architecture steps.

## Current structure

All sources currently build into one Catch2 executable:

```text
tests/
|-- CMakeLists.txt
|-- test_{multi_index,taylor_jet,laplace_derivatives,
|   spherical_harmonics}.cpp                   mathematics
|-- test_{p2p,p2m_m2l_l2p,multipole_accuracy,
|   operator_consistency,static_m2l}.cpp       operators
|-- test_{uniform_tree,static_topology}.cpp    tree/topology
|-- test_{cuboid,rectangular_prism_magtense,
|   tetrahedron,tetrahedron_static}.cpp        geometry
|-- test_{uniform_fmm,periodic,precision,
|   normalisation,output_flags,cache}.cpp      integration
|-- test_{cuda_backend,cuda_m2l}.cpp           CUDA/backend
|-- test_parameter_selection.cpp               policy/advice
|-- test_c_api.cpp
`-- test_fortran_api.f90                       optional smoke test
```

The future `unit/`, `backend/`, and `integration/` split should follow tested
responsibility, not file size alone.

## Rules

- Put a regression at the lowest layer that can demonstrate the contract, then
  add integration/cross-backend coverage only when the interaction matters.
- Prefer deterministic analytic cases before convergence and randomised cases.
- Cover supported precision, output, geometry/model, tree, periodic, cache,
  and backend variants relevant to a change.
- Self-exclusion tests use explicit source identity, never coordinate equality.
- CUDA tests may skip when CUDA is not compiled or no runtime device exists;
  do not turn a real failure into a skip. CTest recognises return code 4 as a
  skip for the discovered Catch2 cases.

## Validation

```bash
cmake --fresh --preset dev
cmake --build --preset dev -j
ctest --preset dev

# Focus one discovered case or group.
ctest --test-dir build --output-on-failure -R '<test-name-regex>'

# CUDA when the toolkit and a runtime device are available.
cmake --fresh --preset cuda
cmake --build --preset cuda -j
ctest --preset cuda
```

Python regressions live in `../python_tests/` and run with
`PYTHONPATH=build python -m pytest python_tests -v` after a `dev` build.
