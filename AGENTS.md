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
`docs/index.md`, `docs/geometry.md`, `docs/backends.md`, and the
tests. Do not infer support from names alone.

## Current refactor stage

**Phase 1 of the `v0.2` architecture refactor is COMPLETE.** Ownership,
dependency direction, public/ABI/cache compatibility, the supported build
matrix, installation, and documentation were validated together; the evidence,
the validation matrix, its limitations, and the Phase-2 handoff are recorded
under "Phase 1 closure" in `docs/architecture.md`. **Phase 2 (post-Phase-1
cleanup) is also COMPLETE**: internal-implementation duplication, the
tree/topology boundary, the cache/`UniformFmm` boundary, and the
public/internal API, header ownership, and packaging boundary have each been
audited and closed as their own explicit task; see "Phase 2 handoff" in
`docs/architecture.md`. Phase 3 (performance optimization) is
in progress: 3A GPU evaluation, 3B CPU / oneMKL evaluation, and the Phase-3
P2P execution unification / geometry-backend coverage / CudaPartial crossover
follow-up are COMPLETE (see `agent_docs/performance_optimization.md` and
`docs/architecture.md`: geometry builds tensors, executors apply tensors; all
nine point/prism/tetrahedron pairs run on every production backend through
every stored-tensor packing, `UniformFmmOptions::p2p_packing` forces a
packing, and `P2PExecutionPacking::PointGeometry` is the one deliberate
geometry-specific executor). The Phase-3B.5 study of procedural versus
precomputed point operators is also COMPLETE: precomputation is an execution
choice, not a mathematical requirement (`docs/architecture.md`, "Precomputed
and procedural representations"); the position-based `PointGeometry` P2P now
runs on the CUDA backends too and is the FP32 default for point pairs, and
the point-source P2M / point-target L2P have a procedural representation
(`UniformFmmOptions::point_expansion_execution`, spherical basis, orders 1-15;
`Auto` selects it only to order 10, the measured range) that is the default on the CPU hierarchy and for FP32 `CudaFull`; finite
tiles keep their exact precomputed operators. **Phase 3B.5b is also
COMPLETE**: the finite half of that sentence is now measured rather than
assumed. Every exact finite P2P, P2M and L2P operator was benchmarked in both
representations (`benchmark_operator_representation`), reconstruction lost by
150x to 1,300,000x per field update and precomputation amortised within one to
ten updates, so finite operators stay precomputed on every backend and no
production policy changed; the one accepted production change is the
precision-generic prism point-tensor kernel that production and the device
benchmark now share. **Phase 3C (static-plan construction and plan
preparation) is also COMPLETE.** Cold construction was dominated by building
the same operator over and over: an exact pair tensor is a pure function of
the displacement and the participating body records, a finite P2M or L2P
operator of the basis, the body's shape record and its offset from its leaf
centre, and normalisation puts body centres on a canonical grid, so
equivalent interactions agree *bitwise* rather than approximately. All three
canonical pair loops and both finite endpoint loops now classify by those
exact bit patterns, build each distinct operator once, and do so in parallel
(`src/operators/p2p.cpp`, `src/fmm/plan_preparation.cpp`,
`docs/architecture.md`). Cold construction of a
4096-body regular prism lattice falls from 66.8 s to 1.5 s, the tetrahedron
lattice from 15.9 s to 1.5 s, and the exact finite tetrahedron endpoint
operators by two orders of magnitude, with byte-identical persisted plans
throughout. No cache format, cache key, C ABI, Python API or Fortran
interface changed; the canonical near-field build for point plans was
deliberately left in place, because skipping it would change cache contents
under a key that distinguishes neither packing nor backend. (It has since been
removed for plans whose policy resolves to `PointGeometry` before preparation:
such a plan builds and persists no pair tensors and keys its geometry file
with a separate `_p2p_positions_` segment, so every stored-tensor key is
unchanged; see `src/cache/AGENTS.md`. The near field is now built in chunks
of target leaves, bitwise equal to one chunk, and a dictionary plan persists
its dictionary under `_p2p_dictionary_`, so a warm plan builds no pair
tensor; see `agent_docs/performance_optimization.md`, "Construction memory
and persisted dictionaries".) **Phase 3C.5
(dense/all-to-all construction) is also COMPLETE.** The same invariant carries
to the exact dense baseline, which Phase 3C had not touched: a dense pair
tensor is a pure function of the displacement, the two body records and
whether the pair is an omitted point self interaction, so `DenseDirectPlan`
now prepares each distinct finite record once, groups pairs by those exact
bits, builds one tensor per group in parallel and scatters. The equivalence
itself moved to `src/operators/exact_operator_reuse.hpp`, shared with the
canonical near-field builder. Cold dense construction falls 38-85x on regular
geometry and 5-13x on locally refined geometry, bit-identically, with repeated
evaluation unchanged at 0.94-1.02x on portable CPU, oneMKL and CUDA; CUDA
inherits the gain because it builds a host plan and uploads it. Two limits are
structural and are recorded rather than worked around: point-to-point plans
never classify, because a point pair costs about 4 ns against about 260 ns for
the cheapest finite pair, and irregular geometry has no exact reuse at all,
because every pair of an all-to-all plan holds a unique combination of
records. Exact reuse also finds less on a lattice whose spacing is not exactly
representable, since equal index differences then round to different
displacement bits; dense plans have no canonical-grid normalisation, so that
rests on the caller's geometry. No public API, C ABI, Python API, Fortran
interface or cache format changed. **Phase 3D (final cross-backend
integration and production policy) is also COMPLETE.** All fourteen automatic
production policies were re-derived from the source, then measured against
their credible forced alternatives on the phase each rule governs, on one
combined CPU/oneMKL/CUDA tree so every comparison is same-session. Every one
resolves as the Phase-3 measurements intend and **no automatic performance
policy changed**: point pairs recompute from positions on the CPU and on CUDA
FP32 and keep stored tensors on CUDA FP64, point expansions are procedural
everywhere except `CudaFull` FP64, finite operators stay precomputed
everywhere, `RegularGrid` selects the signed dictionary subject to the built
plan's token width, portable CPU M2L stays the default with oneMKL
explicit-only, and `ExecutionBackend::Auto` stays `CpuStatic`. What the phase
changed is three defects: the unguarded `std::uint32_t` narrowing of dense pair
indices in `ExactOperatorClasses` (now abandons classification instead of
wrapping), an explicit FP32 `CudaBsr3` request under a lowered
`cuda_p2p_bsr_max_bytes` that reported success and then failed the evaluation,
and a stale example notebook. Phase 3C's largest warm-cache cost no longer
reproduces; the FP32 precision conversion replaced it and needs a cache-format
change, so it is recorded for post-v0.2. The final production decision ledger,
every rejected change with the measurement that rejected it, and the deferred
Phase-4 inventory are in `agent_docs/performance_optimization.md`, "Final
cross-backend integration and production policy (Phase 3D)"; the retained
numbers are in `benchmarks/baselines/phase3d/` and are an engineering
regression baseline, not an Article1 benchmark. **Phase 4 repository pruning
and documentation cleanup is also COMPLETE.** Nothing executable was removed:
every backend, P2P packing, dictionary executor, point-expansion
representation, expansion basis, precision path, explicit override and
benchmark driver stands as Phase 3D left it. What was pruned is duplication —
two dead member groups, an unused overload, 20 of 29 notebooks, and 14 of
28 documentation pages — and what was added is explanation: public headers
went from 430 undocumented members to none, and `src/` comment density rose by
44 %. Serial CTest fell from 550 s to 44 s with the same 244 cases. The
refactor history that used to live in `docs/` is preserved verbatim in
`agent_docs/architecture_history.md`, and the phase's own audit ledger is
`agent_docs/phase4_pruning_ledger.md`. No public C++ API, C ABI, Python API,
Fortran interface, cache format or cache key changed.

One planned merge was **deliberately not taken**: the finite P2M/L2P endpoint
classifier in `src/fmm/plan_preparation.cpp` duplicates the algorithm of
`src/operators/exact_operator_reuse.hpp`, but sharing it makes GCC 13's LTO
plugin mark ordinary `std::vector`/`std::array` COMDATs as prevailing in two
archive members at once and abort the link (`lto1: fatal error: multiple
prevailing defs`). GCC 15 links it without complaint; the project's CI is
GCC 13, so the duplication stays until the supported toolchain moves. Do not
re-merge it without checking that configuration first. **The Article1
publication campaign is NEXT**; it runs against this frozen implementation
with its own measurement protocol. Do not begin benchmark redesign as a side
effect of another change.

The `v0.1.0` annotated tag and `release/v0.1` branch preserve the pre-refactor
implementation. Architectural work occurs on `refactor/architecture-v0.2`.
Never move, recreate, or rewrite the tag or preserved branch. No `v0.2` release
ref exists; creating one is a separate, explicit release step.

The foundational `core`, `math`, `geometry`, and `tree` layers, followed by the
operator/static-plan step and backend boundaries, are now organised. The
compatibility/transitional-source review is complete, so the remaining
compatibility surface is now intentional rather than merely inherited. Every
flat header under `include/cdfmm/` was an installed public path at `v0.1.0`;
they are therefore retained deliberately as public compatibility façades and
must not be removed without an approved API change. Canonical structured
headers and canonical implementation no longer include those façades: the
dependency runs façade -> canonical, never the reverse. The unused internal
shims `src/cuda_fmm_plan.hpp`, `src/cuda_m2l_plan.hpp`, `src/cuda_p2p_plan.hpp`
and the dead `src/static_operators.cpp` are removed; internal `src/` headers
carry no downstream compatibility obligation. Dynamic mathematical operator
implementations live under `src/operators/*`; `src/operators.cpp` is
compatibility-only thin delegation and is retained for the supported flat C++
and Python operator names. `src/cuboid.cpp` is removed: its prism-averaged
monomial is now owned by `src/geometry/primitives/rectangular_prism.cpp` and
its pair-tensor dispatch by `src/operators/p2p.cpp`, with
`cuboid_averaged_monomial` and `build_pair_tensor` retained as thin flat
spellings. CUDA direct execution, complete
P2P, M2L, far-field, and full-FMM backends now have explicit homes under
`src/backend/cuda/`: far-field execution is under
`src/backend/cuda/far_field/`, while complete FMM declarations and orchestration
are under `src/backend/cuda/fmm/`. `CudaFullPlan` still owns changing
evaluation state, stream/event/timing and near/far orchestration,
combination/reordering, and D2H transfer. CPU direct, P2P, M2L, and far-field
execution now also have explicit homes under
`src/backend/cpu/{direct,p2p,m2l,far_field}/`; the legacy static-plan header
remains a compatibility umbrella. High-level `UniformFmm` implementation is
decomposed under `src/fmm/` by construction, plan preparation, backend setup,
evaluation, far-field sequencing, diagnostics, and lifecycle/accessor
ownership. Cache identity and persistence are decomposed under `src/cache/`
by file container/environment policy, payload records, key generation, the
universal/periodic payload, and the geometry-plan payload; its persistent
format and its keys are compatibility contracts, so read `src/cache/AGENTS.md`
before changing them. Every cache entry point is a free function taking an
explicit identity/payload record built from specific solver/plan/tree/topology
types; no file under `src/cache/` depends on `cdfmm/uniform_fmm.hpp` or defines
a `UniformFmm` member function, and `src/fmm/execution_setup.cpp`/
`plan_preparation.cpp` remain the sole callers and the owners of the
cache-vs-build decision. The bindings boundary is now structured: the C ABI
implementation is `src/bindings/c_api.cpp`, while the Python adapter is split
into `python/internal.hpp`, `module.cpp`, `core.cpp`, `geometry.cpp`,
`tree.cpp`, `operators.cpp`, `direct.cpp`, and `fmm.cpp`; the public C header
and ABI version remain unchanged. Further backend, orchestration, cache, or
binding changes are authorised only when explicitly scoped. Geometry packing,
grain generation/discretisation, and prism/tetrahedron refinement remain
future work.

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
|-- src/              structured math/geometry/tree/backend plus decomposed
|                    higher-level FMM orchestration
|-- tests/            C++ tests and optional Fortran smoke test
|-- python_tests/     Python, runner, and tutorial-execution regression tests
|-- benchmarks/       C++ benchmarks, Python runners, retained baselines, and
|                    the external FMM3D comparison material
|-- examples/         C++/Fortran examples, six notebook tutorials, and the
|                    MagTense validation notebooks
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
src/backend/cpu/{direct,p2p,m2l,far_field}
src/backend/cuda/{common,direct,p2p,m2l,far_field,fmm}
tests/{unit,backend,integration}
benchmarks/{direct,p2p,far_field,fmm}
```

This is a target, not the current tree. Do not create empty directories for
symmetry. Add more local `AGENTS.md` files only when corresponding substantive
subsystems exist. Current support areas such as `python_tests/`, `python/`,
`fortran/`, and `tools/` remain in place; the implemented C and Python
binding adaptation units are listed above and below.

The closure audit found no violation of the prohibited-dependency list. Besides
`src/operators.cpp` and `python/internal.hpp`, the only reverse-direction edge
still deliberate is the `DenseDirectPlan::evaluate()` compatibility dispatch in
`src/plan/direct/dense.cpp`; it is enumerated in `docs/architecture.md`. The
canonical headers still include the substantive flat `cdfmm/timings.hpp` and
`cdfmm/periodic.hpp`, which is intentional (see "Public/internal API, header
ownership, and packaging cleanup" below), not a reverse edge. A later
cache/plan-preparation boundary cleanup removed the
`src/cache/internal.hpp -> cdfmm/uniform_fmm.hpp` edge entirely: cache code
now depends only on the specific solver/plan/tree/topology types it persists,
never on `UniformFmm`. A subsequent public-header/packaging cleanup removed the
`src/backend/cuda/fmm/internal.hpp -> cdfmm/uniform_fmm.hpp` edge the same way:
`cuda_compiled`, `cuda_available`, `cuda_direct_available`,
`cuda_m2l_p2p_available`, `cuda_m2l_available`, `cuda_full_available`,
`cuda_device_description`, and `one_mkl_available` are now declared by the
canonical `cdfmm/backend/cuda/availability.hpp` and
`cdfmm/backend/mkl/availability.hpp` headers, which the CUDA backend includes
directly instead of the complete solver header; `cdfmm/uniform_fmm.hpp`
includes both and continues to expose every name transitively for source
compatibility. Do not add new reverse-direction edges.

**Public/internal API, header ownership, and packaging cleanup (complete).**
The remaining Phase-2 API/packaging item is done. `ExecutionBackend` moved
from `uniform_fmm.hpp` to `cdfmm/backend/execution.hpp` so
`cdfmm/parameter_selection.hpp` depends only on `Vec3` and `ExecutionBackend`
(via `cdfmm/math/vec3.hpp` and `cdfmm/backend/execution.hpp`) instead of the
complete `uniform_fmm.hpp`; `src/parameter_selection.cpp` now includes
`cdfmm/uniform_fmm.hpp` explicitly, since it is the translation unit that
actually constructs `UniformFmm`. `include/cdfmm/tensor_dictionary.hpp`'s
Tensor6 canonicalisation/token primitives moved to the canonical
`include/cdfmm/plan/p2p/tensor_dictionary.hpp`, alongside the other P2P
packing headers that consume them; the flat path is now a forwarding façade.
A minimal downstream CMake package (`find_package(cdfmm CONFIG)`,
`cdfmm::cdfmm_c`) was added: `cdfmm_c` is the only installed target, its
public interface has no CUDA/oneMKL/OpenMP link dependency (those are private
to the internal static `cdfmm_core`), so the generated package config calls no
`find_dependency`. `timings.hpp`, `periodic.hpp`, and `validation.hpp` were
each audited and deliberately kept flat: none has one clear subsystem owner
(`timings.hpp` aggregates fmm/plan/CUDA diagnostics used together;
`periodic.hpp` is consumed symmetrically by the `tree` and `operators`
siblings; `validation.hpp` is explicitly non-production test/diagnostic
utility). `P2MPlan`/`FloatP2MPlan` and `ExpansionBasis` were reviewed and kept
at their prior cache-cleanup homes. See "Public/internal API, header
ownership, and packaging cleanup" in `docs/architecture.md` for the full
audit, rationale, and validation evidence. No algorithm, cache format, cache
key, C ABI, Python API, or Fortran interface changed.

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
  logic. The C ABI implementation lives under `src/bindings/`; Python is
  split by exposed subsystem under `python/`, with `module.cpp` as its sole
  module entry point. Binding code uses supported canonical structured headers
  where available and does not include internal backend headers.

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
normally remain internal below the `src/backend/cuda/` hierarchy.

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
not coordinate equality. Keep full notation consistent with `docs/math/conventions.md`.

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
  stop after its acceptance criteria pass. Preserve public compatibility
  façades under `include/cdfmm/` unless an API change is separately approved;
  an internal `src/` forwarding header may be removed once it has no users.

The authoritative rationale, dependency table, target tree, and deferred-work
inventory are in `docs/architecture.md`.
