# dip-fmm project overview

`dip-fmm` is a C++20 library in namespace `cdfmm` for repeated magnetic-dipole
field evaluation on fixed source and target geometry. It implements a Laplace
fast multipole method (FMM), with direct references, analytical finite-geometry
operators, reusable static plans, and optional CPU, oneMKL, hybrid CUDA, and
full CUDA execution. The primary result is the magnetic field
`H = -grad(phi)`; scalar potential is available on supported paths.

## Stable capability boundary

- Real spherical harmonics are the default expansion basis; Cartesian Taylor
  expansions remain a complete independent formulation.
- `UniformTree` is a complete Morton-sorted octree. `AdaptiveTree` supplies a
  compact geometry-only topology that can feed the shared static FMM plan.
- Static plans retain P2M, M2M, M2L, L2L, L2P, and exact list-1 P2P data for
  repeated moment states. Direct CPU and CUDA plans remain independent exact
  references, not FMM fallbacks.
- Point dipoles, axis-aligned rectangular prisms, and tetrahedra are supported
  by the documented source/target model combinations. Fully periodic evaluation
  currently means an explicit cubic cell with the zero-`k=0` convention.
- True FP32 and FP64 execution include static operators, coefficient state,
  scratch, and CUDA buffers. The C ABI is unconditional; Python/pybind11 and
  the ISO C binding Fortran wrapper are interface layers, with the latter
  optional at build time.

The authoritative support boundary is distributed across `docs/overview.md`,
`docs/geometry-models.md`, `docs/backends.md`, `docs/periodic-boundaries.md`,
and the tests. In particular, partial periodicity, rectangular periodic cells,
and runtime MagTense integration are not current API promises.

## Build and test entry points

The portable CI baseline is a Release CMake build with C++ tests and Python
bindings, followed by CTest, package installation, and Python regressions:

```console
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCDFMM_BUILD_TESTS=ON -DCDFMM_BUILD_PYTHON=ON \
  -DCDFMM_ENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
python -m pip install .
python -m pytest python_tests -v
```

For local work, the checked-in `dev` preset is the no-install portable CPU
path; use `PYTHONPATH=build` so Python imports the just-built extension:

```console
cmake --fresh --preset dev
cmake --build --preset dev -j
ctest --preset dev
PYTHONPATH=build python -m pytest python_tests -v
```

Optional configurations use the same configure/build pattern: `cuda` for
manual CUDA validation, `notebooks` for CUDA plus oneMKL notebook work,
`magtense` for the comparison environment, `release` for installed CUDA plus
oneMKL, and `benchmark`, `benchmark-mkl`, `benchmark-all`, or `profile-all`
for performance/profiling builds. CUDA and oneMKL require the matching active
environment and hardware; they are not implied by a successful CPU build.

## Where to start

Read `project_core_tech.md` for mathematical and numerical contracts,
`project_structure.md` for ownership and navigation, and
`project_progress.md` for the current refactor state. The user-facing build
and test workflow is in `docs/installation.md` and `docs/validation.md`.
