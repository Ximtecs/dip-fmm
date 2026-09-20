# C++ test guidance

Inherit `../AGENTS.md`. Every retained test protects one distinct invariant
or failure mode; the Phase-4 audit (`agent_docs/project_progress.md`) records
why each file exists and where its overlaps are deliberate.

## Current structure

All sources build into one Catch2 executable:

```text
tests/
|-- CMakeLists.txt
|-- test_{multi_index,taylor_jet,laplace_derivatives,
|   spherical_harmonics}.cpp                   mathematics
|-- test_{p2p,p2m_m2l_l2p,multipole_accuracy,
|   operator_consistency,static_m2l}.cpp       operators
|-- test_{p2p_exact_reuse,dense_direct_exact_reuse}.cpp
|                                              exact-operator reuse (near
|                                              field and dense builders)
|-- test_p2p_geometry_matrix.cpp               nine geometry pairs x backends x
|                                              precisions x packings, free-space
|                                              and periodic, against dense direct
|-- test_{uniform_tree,static_topology}.cpp    tree/topology
|-- test_{cuboid,rectangular_prism_magtense,
|   tetrahedron,tetrahedron_static}.cpp        geometry (the prism and
|                                              tetrahedron trust anchors)
|-- test_{uniform_fmm,periodic,precision,
|   normalisation,output_flags,cache,
|   procedural_point_expansion}.cpp            integration
|-- test_{cuda_backend,cuda_m2l}.cpp           CUDA/backend
|-- test_cuda_p2p_stub.cpp                     non-CUDA backend boundary
|-- test_foundational_headers.cpp              canonical/compatibility headers
|-- test_cuda_{direct,dense_direct,p2p,m2l}_headers.cpp
|                                              canonical CUDA header coverage
|-- test_cuda_legacy_{direct,cuboid,p2p}_header.cpp
|                                              legacy CUDA façade coverage
|-- test_parameter_selection.cpp               policy/advice
|-- test_c_api.cpp
`-- test_fortran_api.f90                       optional smoke test
```

## Tiers

- **Portable CI tier**: everything that runs in a CUDA-less, oneMKL-less
  build. GitHub Actions runs this tier; keep it fast. Reproduction and
  exactness checks use the smallest scene that populates every stage, never
  a convergence-sized one.
- **Optional backend tier**: CUDA and oneMKL cases. They return early
  (`SUCCEED()`) or skip when the build or device is absent and execute for
  real on the `cuda`/`notebooks` presets; do not turn a real failure into a
  skip. CTest recognises return code 4 as a skip for the discovered cases.
- **Regression-tool tier**: benchmark drivers and analysers are tested under
  `../python_tests/`, not here.

## Deliberate per-layer coverage

The same physical contract is pinned once per implementation that could
regress independently; this is not duplication and must not be merged:

- point self exclusion comes from the identity map, finite self fields are
  physical: `test_p2p_exact_reuse.cpp` (canonical near-field builder),
  `test_dense_direct_exact_reuse.cpp` (dense builder), `test_cuboid.cpp` and
  `test_tetrahedron_static.cpp` (geometry-specific tensors),
  `test_uniform_fmm.cpp` (the `UniformFmm` façade) and
  `test_cuda_backend.cpp` (every CUDA packing);
- one header per translation unit in the `*_headers.cpp` and
  `*_legacy_*_header.cpp` files: self-containment of a public header is only
  tested when it is the sole `cdfmm` include of its TU, so these stay
  separate files, while `test_foundational_headers.cpp` tests coexistence.

## Rules

- Put a regression at the lowest layer that can demonstrate the contract, then
  add integration/cross-backend coverage only when the interaction matters.
- Prefer deterministic analytic cases before convergence and randomised cases.
- Cover supported precision, output, geometry/model, tree, periodic, cache,
  and backend variants relevant to a change.
- Self-exclusion tests use explicit source identity, never coordinate equality.
- Two same-layer cases with near-identical inputs asserting one contract are
  merged or parameterised (SECTIONs or a table); two layers asserting one
  physical result are kept.

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
