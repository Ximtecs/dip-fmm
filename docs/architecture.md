# Architecture

This page describes how the library is organised and why: the layers and the
direction of dependencies between them, the lifecycle of a static plan, the
invariants that keep the mathematics defined once, and the compatibility
surface. It describes the current tree. The history of how it got here (the
`v0.2` refactor phases, the per-cleanup audits and the performance campaign)
is project memory under `agent_docs/`, not user documentation.

## Layers

```text
                         PUBLIC API   (include/cdfmm, python/, c_api.h, fortran/)
                             |
                            FMM       solver lifecycle, near/far orchestration
                             |
                            PLAN      immutable execution descriptions
                             |
              +--------------+--------------+
              |              |              |
          OPERATORS         TREE         GEOMETRY
              |              |              |
              +--------------+--------------+
                             |
                            MATH
                             |
                            CORE
```

Execution backends consume prepared plans:

```text
                   canonical static plans
                             |
                  +----------+----------+
                  |          |          |
                 CPU       oneMKL      CUDA
```

| Layer | Owns | May depend on |
|---|---|---|
| `core` | precision, output flags, small shared types | standard library |
| `math` | `Vec3`, multi-indices, Laplace derivatives, Cartesian and spherical expansion mathematics | `core` |
| `geometry` | point, rectangular-prism and tetrahedron integration elements and their exact primitives | `math`, `core` |
| `tree` | bounds, nodes, Morton machinery, uniform and adaptive construction, the static interaction topology | `math`, `core` |
| `operators` | mathematical P2M, M2M, M2L, L2L, L2P, M2P and P2P construction and dynamic application | `geometry`, `math`, `core` |
| `plan` | immutable execution-oriented structures assembled from operators, topology and geometry, including the P2P packings and the dense direct plan | `operators`, `tree`, `geometry`, `math`, `core` |
| `backend` | CPU, oneMKL and CUDA resource ownership, upload, scheduling and execution of plans | `plan`, low-level `math`/`core` |
| `fmm` | `UniformFmm` lifecycle, construction, plan preparation, backend setup, evaluation, diagnostics | `geometry`, `tree`, `plan`, `backend` |
| `cache` | keys, container, serialisation and deserialisation of already-defined solver data | the data it persists; no mathematics |
| `bindings` | C, Fortran and Python adaptation of the public interface | public API |

Prohibited by default: `geometry` or `tree` depending on a backend or CUDA,
`math` depending on `fmm`, `operators` depending on orchestration, cuBLAS or
cuSPARSE, `plan` depending on Python. A backend may not redefine the
mathematics it executes. Design rules: a function performs one operation at
one abstraction level, a file implements one cohesive concept, a directory is
a subsystem; size is a warning sign, not a rule, and fused kernels or clear
mathematics may be large.

## Source map

```text
include/cdfmm/
  cdfmm.hpp                          umbrella
  core/ math/ geometry/ tree/ operators/ plan/ backend/
                                     canonical structured headers
  uniform_fmm.hpp                    solver API and options
  timings.hpp periodic.hpp validation.hpp parameter_selection.hpp
                                     substantive flat headers with no single
                                     subsystem owner
  <other flat *.hpp>                 compatibility facades of the v0.1 paths
  c_api.h                            the C ABI (version 1)
src/
  math/ geometry/primitives/ tree/{common,uniform,adaptive}/
  operators/                         authoritative operator construction and
                                     dynamic application; the point-dipole
                                     pair formula, the procedural point
                                     expansion kernels and the exact-operator
                                     classifier live here
  plan/{direct,p2p}/ plan/precision.cpp
                                     dense plan, P2P packings, FP32 conversion
  backend/cpu/{direct,p2p,m2l,far_field}/   portable execution
  backend/mkl/                       oneMKL M2L and dense direct
  backend/cuda/{common,direct,p2p,m2l,far_field,fmm}/  device execution
  backend/cuda/execution_policy.*    deterministic representation choices
  backend/cuda/stub/                 non-CUDA builds
  fmm/{construction,plan_preparation,execution_setup,evaluation,far_field,
       diagnostics,uniform_fmm}.cpp
  cache/{io,format,keys,universal,geometry}.cpp
  bindings/c_api.cpp
python/{module,core,geometry,tree,operators,direct,fmm}.cpp
fortran/cdfmm_fortran.f90
```

## The static-plan lifecycle

`dip-fmm` is designed for geometry that is prepared once while dipole moments
change repeatedly:

```text
geometry (positions, finite-body records, identity map)
    -> canonical root normalisation
    -> UniformTree, or AdaptiveTree -> StaticFmmTopology
    -> canonical mathematical operators (P2M, M2M, M2L, L2L, L2P, exact P2P)
    -> immutable static plans (FP64, then quantised for an FP32 plan)
    -> backend-specific persistent resources (packings, device uploads)
    -> repeated evaluations with changing moments
```

Construction fixes the geometry, the tree, the order and basis, the precision,
the backend, and any fixed identity map. An evaluation permutes the moments
into Morton order, runs P2M, M2M, M2L, L2L and L2P on the far field and the
exact P2P on `list1`, adds the two, and returns results in user order:

```text
REPEATED EVALUATION

moments -> Morton order -> P2M -> M2M -> M2L -> L2L -> L2P -> far field
    |
    +---------------------------> P2P ---------------------> near field

far field + near field -> target unsorting -> field (and potential)
```

The far and near partitions do not overlap. Mutable coefficients, result
buffers and scratch are state, not plans, and one evaluator is not
re-entrant; separate evaluators may run concurrently.

### Ownership

`UniformTree` owns the physical tree and its inspection API.
`StaticFmmTopology` is the compact, tree-owned interaction topology both tree
builders produce: compact node ids, leaf ranges, translation edges, M2L rows
with transfer classes and levels, and P2P leaf-pair records with image shifts.
`UniformFmm` owns the topology, the canonical operators, the derived packings,
the backend resources and the per-evaluation state. The construction unit
normalises geometry and builds the topology; plan preparation builds the
immutable operators, quantises FP32 state and makes the cache calls; execution
setup resolves the backend and the representation policy; evaluation
coordinates the near/far lifecycle and timing; far-field sequencing, exact
summary formatting and accessors are separate units.

Schedules are grouped by level barriers, not by node index: P2M writes source
leaf ranges; M2M follows child-to-parent edges from deep to shallow; M2L
consumes target-row CSR; L2L follows parent-to-child edges from shallow to
deep; L2P reads target leaf ranges. Sparse coefficient maps store
`(output, input, value)` triples; dense M2L matrices are column-major and
shared by transfer class with target-row offsets plus parallel source-node,
matrix-id and level arrays, so portable CPU and CUDA executors consume
identical mathematical data.

### Canonical operators and derived packings

Canonical P2P interaction data is the backend-independent mathematical truth:
`StaticP2POperator` holds the six symmetric tensor components per target-row
entry with a per-pair identity marker. SoA rows, dense leaf blocks, the signed
tensor dictionary and BSR(3) are **derived execution representations**
selected for a backend; they are validated against the canonical data and
never become competing definitions of the operator.

```text
point / prism / tetrahedron geometry
    -> geometry-specific pair-tensor construction    (operators/p2p.cpp,
                                                     geometry/primitives/*)
    -> canonical tensors + topology + identity marker (StaticP2POperator,
                                                     StaticP2PLeafRecord)
    -> geometry-independent packings                 (plan/p2p/*)
    -> CPU / CUDA executors                          (backend/*/p2p)
```

Below the canonical operator no builder or executor reads a geometry record.
The `skip_for_identity` marker, not the geometry, decides whether an identity
match omits a pair (point dipole) or applies a physical self tensor (finite
body); every packing carries it. Periodic image records are ordinary stored
tensors. The one deliberate exception is `P2PExecutionPacking::PointGeometry`,
a fused evaluation for point sources and point targets that recomputes the
pair formula from the resident positions; it is named as such and rejected
for finite geometry. `tests/test_p2p_geometry_matrix.cpp` enforces this
matrix over every pair, backend, precision and packing.

### Precomputed and procedural representations

Where an operator is defined does not dictate whether an executor holds it as
data:

```text
physical operator
    +--> precomputed representation   tensor / coefficient rows / compressed
    |                                 packing, built once at construction
    +--> procedural representation    the mathematically identical point
                                      operator reconstructed from positions
                                      during every evaluation
```

Finite bodies are always precomputed: reconstructing an analytical prism or
tetrahedron tensor on every evaluation was measured at 150 to 1,300,000 times
the cost of streaming the stored one, and construction is one pass of the
same arithmetic. Point P2P, point P2M and point L2P have procedural forms (the
position-based P2P executors and the solid-harmonic recurrence) that are
selected per backend and precision by the measured policy in
[Execution backends](backends.md), with explicit overrides. The mathematics
is defined once: the procedural kernels call the same formula or recurrence the
stored construction is validated against.

### Exact operator reuse

An exact near-field pair tensor is a pure function of the displacement and the
participating body records, with any periodic image shift folded into the
displacement; a finite P2M or L2P operator is a pure function of the basis,
the body's shape record and its offset from its leaf centre. Root
normalisation puts body centres on a canonical $10^{-9}$ grid, so equivalent
interactions agree **bitwise**. Construction therefore classifies by those
exact bit patterns, builds each distinct operator once, in parallel, and
scatters it (`src/operators/exact_operator_reuse.hpp`, shared by the
near-field, the dense all-to-all and the finite endpoint builders). Keys are
never compared with a tolerance, because the exact corner formulas have no
continuity that would justify one; classes are numbered in first-seen order,
so a plan does not depend on thread count or hash iteration order;
classification is abandoned when an initial sample shows too few duplicates to
repay it, which is purely a performance decision. A regular lattice reaches a
few hundred distinct operators however many bodies it holds.

## Execution policy

Within a chosen backend, every representation choice (P2P packing and
dictionary executor, M2L pairs per thread, translation lane groups, stream
priorities, point-expansion execution) is resolved once, deterministically,
from facts of the constructed plan and the options
(`src/backend/cuda/execution_policy.*` and `src/fmm/execution_setup.cpp`;
despite its home the policy also settles the CPU dictionary choice). Nothing
is measured at run time and no choice changes the result. The resolved choices
appear in the initialisation summary and in the plan's accessors
([Execution backends](backends.md)).

## Cache boundary

`src/cache/` persists already-defined solver data and defines no mathematics.
`io.cpp` owns the root policy and the validated atomic file container,
`format.cpp` the payload records, `keys.cpp` the identity, `universal.cpp` and
`geometry.cpp` the two payloads. Every entry point is a free function taking
an explicit identity or payload record built from specific solver, plan, tree
and topology types; nothing under `src/cache/` includes `uniform_fmm.hpp`, and
`plan_preparation.cpp`/`execution_setup.cpp` are the sole callers and the
owners of the cache-versus-build decision. The persistent format and the keys
are compatibility contracts (`src/cache/AGENTS.md`); their semantics are in
[Caching and periodicity](caching-and-periodicity.md).

## CUDA constraints

Source modularity does not impose runtime modularity. A load/scale/multiply/
reduce/store sequence may stay one kernel with inlined helpers; no kernel
launches, global-memory intermediates, host/device transfers, synchronisation,
evaluation-loop allocations or runtime polymorphism are added to make source
units smaller. Persistent plan and buffer reuse, stream/event ordering and
near/far overlap, cuBLAS/cuSPARSE handle and stream association, precision and
accumulation semantics, launch geometry and transfer boundaries are preserved
and benchmarked when touched. `.cuh` files, device helpers and RAII stay below
`src/backend/cuda/`.

## Public and internal interfaces, compatibility

`include/cdfmm/` is for deliberately supported downstream interfaces;
implementation-only headers live under `src/`. Every flat header under
`include/cdfmm/` was an installed public path at `v0.1.0` and is retained as a
compatibility façade that forwards to its canonical structured header and
reproduces the transitive surface downstream code relied on; removing one is
an API change requiring separate approval. Compatibility flows one way:
canonical headers and canonical implementation include canonical headers, the
façades include those. The deliberate exceptions are `src/operators.cpp`
(which implements the flat `cdfmm/operators.hpp` API by thin delegation),
`python/internal.hpp` (which exports those flat operator names), the
`DenseDirectPlan::evaluate()` dispatch in `src/plan/direct/dense.cpp` (a
pre-v0.2 public method, confined to one unit), and the canonical headers'
includes of the substantive flat `cdfmm/timings.hpp` and `cdfmm/periodic.hpp`,
which have no single subsystem owner. Internal `src/` headers carry no
downstream obligation.

The C ABI (`include/cdfmm/c_api.h`, `CDFMM_ABI_VERSION 1`) is built
unconditionally as `libcdfmm_c`, the only installed target; its CMake package
(`find_package(cdfmm CONFIG)`, `cdfmm::cdfmm_c`) exports no CUDA, oneMKL or
OpenMP dependency because those are private to the never-installed
`cdfmm_core`. The Fortran wrapper is `ISO_C_BINDING -> C ABI -> C++`. The
Python module is a thin adaptation split by exposed subsystem with
`python/module.cpp` as its sole entry point.

## Validation map

| Layer | Where it is tested |
|---|---|
| mathematics | `tests/test_{multi_index,taylor_jet,laplace_derivatives,spherical_harmonics}.cpp` |
| operators | `tests/test_{p2p,p2m_m2l_l2p,multipole_accuracy,operator_consistency,static_m2l}.cpp` |
| exact operator reuse | `tests/test_{p2p_exact_reuse,dense_direct_exact_reuse}.cpp` (bitwise) |
| finite geometry trust anchors | `tests/test_{rectangular_prism_magtense,tetrahedron,tetrahedron_static,cuboid}.cpp` |
| the P2P execution invariant | `tests/test_p2p_geometry_matrix.cpp` (nine pairs × backends × precisions × packings × periodicity, against the FP64 dense plan) |
| complete FMM, precision, periodic, cache, procedural expansions | `tests/test_{uniform_fmm,precision,periodic,cache,procedural_point_expansion,normalisation,output_flags}.cpp` |
| CUDA backends | `tests/test_{cuda_backend,cuda_m2l}.cpp` (skip or return early without a device) |
| compatibility headers, C ABI, Fortran | `tests/test_foundational_headers.cpp`, `test_cuda_*_header*.cpp`, `test_c_api.cpp`, `test_fortran_api.f90` |
| Python, tutorials, benchmark tooling | `python_tests/` |

Every numerical change needs an analytic check where possible, convergence or
cross-backend agreement where relevant, precision and output-mode coverage,
and a regression test at the lowest layer that demonstrates the contract.
CUDA tests are manual: hosted CI is portable CPU only.
