# dip-fmm agent instructions

These instructions are the repository-wide architectural constitution. A
nested `AGENTS.md` inherits this file and adds local guidance; the nearest file
to a changed path wins if instructions conflict. Read every applicable file
before inspecting, editing, or validating that area.

## Project and current scope

`dip-fmm` is a C++20 library in namespace `cdfmm` for direct and fast-multipole
evaluation of magnetic dipole interactions. It is optimised for stationary
geometry evaluated repeatedly with changing moments or magnetisation.

Verified current capabilities include:

- direct all-to-all and complete FMM evaluation;
- uniform trees and a geometry-only adaptive tree feeding shared static plans;
- real spherical-harmonic and Cartesian expansions where supported;
- point, rectangular-prism, and tetrahedron sources and targets, including
  finite target averaging and analytical tetrahedron-to-tetrahedron near field;
- free-space and fully periodic cubic zero-`k=0` evaluation;
- FP32 and FP64 storage/execution;
- portable CPU, optional oneMKL, CUDA-partial, and CUDA-full execution;
- canonical and derived static P2P representations, persistent caches, and
  plan/timing/memory diagnostics;
- Python bindings, an unconditional C ABI, and an optional Fortran wrapper;
  and
- C++/Python regression tests, examples, notebooks, and benchmarks.

The precise support matrix and limitations remain authoritative in
`docs/overview.md`, `docs/geometry-models.md`, `docs/backends.md`, and the
tests. Do not infer support from names alone.

## Current refactor stage

The `v0.1.0` annotated tag and `release/v0.1` branch preserve the pre-refactor
implementation. Architectural work occurs on `refactor/architecture-v0.2`.
Never move, recreate, or rewrite the tag or preserved branch.

Step 2 has organised the foundational `core`, `math`, `geometry`, and `tree`
layers. Preserve the flat forwarding headers until an explicit compatibility
cleanup. The next production step may address operators and plans only when it
is explicitly authorised; do not pre-emptively split backends, orchestration,
cache, or bindings. Geometry packing, grain generation/discretisation, and
prism/tetrahedron refinement remain future work.

## Repository memory

The project-specific memory in `agent_docs/` supplements this architectural
constitution. Read the relevant `agent_docs/` files before work, and maintain
the current-state, diary, and latest-session entries when a change materially
alters architecture, validation evidence, or repository status. Keep global
orchestration instructions inherited rather than copying them into these
project files. Preserve the nested repository boundary: this checkout is a
separate Git repository from its MagTense parent.

## Navigation

Current structure:

```text
dip-fmm/
|-- include/cdfmm/    structured core/math/geometry/tree plus compatibility headers
|-- src/              structured math/geometry/tree plus deferred flat higher layers
|-- tests/            C++ tests and optional Fortran smoke test
|-- python_tests/     Python, runner, and notebook regression tests
|-- benchmarks/       C++ benchmarks and Python benchmark runners
|-- examples/         C++, Fortran, Python, and notebook examples
|-- python/           pybind11 bindings
|-- fortran/          ISO_C_BINDING wrapper
|-- tools/            cache precomputation executable
|-- docs/             user, mathematical, validation, and developer docs
|-- CMakeLists.txt
`-- CMakePresets.json
```

Read the nested guidance before work in:

- `include/cdfmm/AGENTS.md` for public interface rules;
- `src/AGENTS.md` for implementation boundaries;
- `tests/AGENTS.md` for C++ validation;
- `benchmarks/AGENTS.md` for performance experiments;
- `examples/AGENTS.md` for user-facing examples and notebooks;
- `docs/AGENTS.md` for documentation ownership;
- `python/AGENTS.md` for pybind11 adaptation;
- `python_tests/AGENTS.md` for Python and notebook regressions; and
- `fortran/AGENTS.md` for the Fortran interface.

The target taxonomy is documented in `docs/architecture.md`. In summary:

```text
include/cdfmm/{core,math,geometry,tree,operators,plan,backend,fmm}
src/{math,geometry,tree,operators,plan,backend,fmm,cache,bindings}
src/backend/{cpu,mkl,cuda}
src/backend/cuda/{common,direct,p2p,far_field,fmm}
tests/{unit,backend,integration}
benchmarks/{direct,p2p,far_field,fmm}
```

This is a target, not the current tree. Do not create empty directories for
symmetry. Add more local `AGENTS.md` files only when corresponding substantive
subsystems exist. Current support areas such as `python_tests/`, `python/`,
`fortran/`, and `tools/` remain in place until a later bindings/build step
explicitly assigns a different home.

## Architecture contract

A function performs one operation at one abstraction level. A file implements
one cohesive concept. A folder represents one subsystem or meaningful
specialisation. Extract substantial named steps from mixed functions, but do
not split trivial logic or clear mathematical code merely to reduce line
counts. Avoid generic `utils`, `misc`, and `helpers` dumping grounds.

Approximate review thresholds are:

```text
function: 40-100 lines is normal; >150 strongly merits responsibility review
file:     300-600 lines is normal; >1000 usually signals mixed responsibilities
```

These are diagnostics, not limits. Fused kernels, template-heavy code,
cohesive mathematics, and generated/static data may justifiably be larger.

The default dependency direction is:

```text
public API -> fmm -> plan -> operators/tree/geometry -> math -> core
                         |
                         `-> backend executes prepared plans
bindings -> public API / fmm
cache -> already-defined solver data
```

Subsystem responsibilities and allowed dependencies:

- `core`: precision, basic shared types, flags, and small status/error types;
  no solver concepts.
- `math`: vectors, multi-indices, derivatives, and expansion mathematics;
  depends on `core`.
- `geometry`: physical/integration elements and models; depends on `math` and
  `core`.
- `tree`: spatial hierarchy and interaction topology; depends on `math` and
  `core`, never execution backends.
- `operators`: mathematical P2M/M2M/M2L/L2L/L2P/P2P definitions and
  construction; depends on geometry/math/core, not orchestration or vendor
  execution libraries.
- `plan`: immutable execution descriptions built from operators, tree, and
  geometry.
- `backend`: persistent resources, upload, scheduling, and execution of plans
  for CPU, oneMKL, or CUDA. It may use low-level math/core but may not redefine
  the mathematics.
- `fmm`: public solver lifecycle and high-level near/far orchestration.
- `cache`: keying, metadata, validation, serialisation, and deserialisation of
  solver data; it defines no solver mathematics.
- `bindings`: thin Python, C, and Fortran adaptation; no independent solver
  logic.

Prohibited by default: geometry or tree depending on CUDA/backend; math
depending on FMM; operators depending on orchestration, cuBLAS, or cuSPARSE;
plans depending on Python. Document and review any technically necessary
exception.

## Static geometry and plans

Preserve the defining lifecycle:

```text
geometry
  -> tree/topology
  -> canonical mathematical operators
  -> immutable static plans
  -> backend-specific persistent resources
  -> repeated evaluations with changing moments
```

Geometry-dependent preparation, packing, allocation, validation, and upload
should happen once wherever practical. Mutable coefficient arrays, result
buffers, and scratch are state, not plans.

Canonical P2P interaction data is the backend-independent mathematical truth.
SoA, compact, leaf-packed, tensor-dictionary, signed/reduced-dictionary, and
BSR forms are derived execution representations. Backends consume a chosen
packing; they do not create a competing definition of the operator.

Physical grains are distinct from point/prism/tetrahedron integration elements.
Future grain generation belongs in `geometry/generation`; prism and
tetrahedron refinement belongs in `geometry/refinement`. Execution-oriented
packing belongs in plan construction or the geometry-to-plan boundary, not in
physical geometry primitives.

## Public and internal code

Use `include/cdfmm/` primarily for deliberately public, supported interfaces.
Keep implementation-only headers and facilities below `src/`. Do not expose a
header merely because multiple translation units use it. CUDA `.cuh` files,
device helpers, kernels, launch helpers, and RAII implementation should
normally remain internal below the future `src/backend/cuda/` hierarchy.

Preserve source and binary compatibility during the behaviour-preserving
refactor unless a separate API change is explicitly approved.

## CUDA and performance constraints

Source modularity must not impose runtime modularity. A load/scale/multiply/
reduce/store sequence may remain one kernel with inlined helpers. Do not add
kernel launches, global-memory intermediates, host/device transfers,
synchronisation, evaluation-loop allocations, or runtime polymorphism merely
to make source units smaller.

Preserve intentional kernel fusion, persistent plan/buffer reuse, stream and
event ordering, cuBLAS/cuSPARSE handle/stream association, near/far overlap,
precision and accumulation semantics, launch geometry, and host/device
transfer boundaries. Benchmark performance-sensitive restructuring.

## Mathematical conventions

Use C++20 and namespace `cdfmm`. The Laplace Green's function is
`G(r) = 1/(4*pi*|r|)`. For `r_ij = x_i - x_j`, the point-dipole field is:

```text
phi(x_i) = sum_j m_j . r_ij / (4*pi*|r_ij|^3)
H_ij = 1/(4*pi) * [3*r_ij*(m_j.r_ij)/|r_ij|^5 - m_j/|r_ij|^3]
H = -grad(phi)
```

`H` is the primary result; scalar potential is optional on supported paths.
Source-point evaluation excludes self-interaction using explicit identity,
not coordinate equality. Keep full notation consistent with `docs/math.md`.

## Code and documentation style

- Use readable multi-line C++; never compress functions, loops, conditionals,
  lambdas, tests, examples, or benchmarks into dense one-liners.
- Use Doxygen comments for public interfaces and concise `//` comments for
  mathematical intent, assumptions, indexing, ownership, and non-obvious
  implementation details. Do not narrate syntax.
- Use British English (`centre`, `initialise`, `normalise`, `behaviour`).
- Use the established FMM names P2M, M2M, M2L, L2L, L2P, and P2P and nearby
  mathematical names such as `alpha`, `M`, `L`, `R`, `dx`, `r`, `H`, `phi`.
- Mark future work as `// TODO(cdfmm): ...`, known incorrect behaviour as
  `// FIXME(cdfmm): ...`, constraints as `// NOTE(cdfmm): ...`, and
  correctness hazards as `// WARNING(cdfmm): ...`.
- New files use `// SPDX-License-Identifier: Apache-2.0` where applicable.

## Build and validation

Use commands from the checked-in presets and CI, not historical notes. The
portable Release baseline used by CI is:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCDFMM_BUILD_TESTS=ON \
  -DCDFMM_BUILD_PYTHON=ON \
  -DCDFMM_ENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
python -m pip install .
python -m pytest python_tests -v
```

For local portable CPU development, the equivalent checked-in preset is:

```bash
cmake --fresh --preset dev
cmake --build --preset dev -j
ctest --preset dev
PYTHONPATH=build python -m pytest python_tests -v
```

The Python tests must import the just-built/installed module, not an unrelated
older installation. `PYTHONPATH=build` is the no-install path verified for the
`dev` preset. The CI workflow instead installs with `python -m pip install .`;
that packaging path requires its declared build dependencies to be available
from the environment or package index. Record how the module was selected when
validating.

Optional configured workflows are:

```bash
# CUDA without oneMKL
cmake --fresh --preset cuda
cmake --build --preset cuda -j
ctest --preset cuda

# CUDA plus oneMKL, used by notebooks
cmake --fresh --preset notebooks
cmake --build --preset notebooks -j
ctest --preset notebooks

# Full release install: CUDA plus oneMKL
cmake --fresh --preset release
cmake --build --preset release -j
ctest --preset release
```

The install-oriented CUDA/oneMKL/release presets require an appropriate active
Conda environment, compiler/toolkit, GPU/runtime, and libraries. Do not assume
they exist. Report unavailable configurations precisely. The CI workflow is
portable CPU only. Build the optional Fortran interface by adding
`-DCDFMM_BUILD_FORTRAN_INTERFACE=ON` to a suitable configure command.

Useful targeted validation:

```bash
ctest --test-dir build --output-on-failure -R '<test-name-regex>'
python -m pytest python_tests/path_or_test.py -v
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Every numerical change needs analytic checks where possible, convergence or
cross-backend comparisons as relevant, precision/output-mode coverage, and a
regression test at the lowest appropriate layer. Tests may skip only a truly
unavailable optional dependency/device and must say why.

## Git and refactor workflow

- Begin by checking branch, status, refs, applicable instructions, and existing
  user changes. Preserve unrelated work in a dirty tree.
- Keep commits focused and reviewable. Do not mix architecture documentation,
  file moves, behaviour changes, and performance changes.
- Before and after a behaviour-preserving step, run the same relevant tests.
- Compare against `v0.1.0` and inspect both the stat and full diff. Never alter
  `v0.1.0` or `release/v0.1`.
- Keep each production-code step within its explicitly authorised layers and
  stop after its acceptance criteria pass. Preserve compatibility shims until
  their removal is separately approved.

The authoritative rationale, dependency table, target tree, and deferred-work
inventory are in `docs/architecture.md`.
