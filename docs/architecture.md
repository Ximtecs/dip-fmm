# Developer architecture

This document is the architectural contract for the `v0.2` refactor. It
describes the ownership and dependency boundaries implemented in the
repository. The foundational `core`, `math`, `geometry`, and `tree` layers,
operators, static plans, FMM orchestration, portable CPU and oneMKL execution,
and CUDA execution all have responsibility-specific homes. The high-level
`UniformFmm` implementation is also decomposed into construction, plan
preparation, backend setup, evaluation, far-field sequencing, diagnostics, and
lifecycle/accessor units. Cache persistence and the bindings boundary are now
structured responsibility-specific layers. Remaining Phase 1 work is limited
to compatibility/transitional-source review followed by whole-refactor
validation.

## Design rules

- A function performs one operation at one abstraction level.
- A file implements one cohesive concept.
- A directory represents a subsystem or a substantial specialisation.
- Size is a warning sign, not a correctness rule. Functions above roughly 150
  lines and files above roughly 1,000 lines deserve a responsibility review,
  but tightly fused kernels, clear mathematical implementations, templates,
  and generated or static data may justifiably remain larger.
- Do not create generic `utils`, `misc`, or `helpers` dumping grounds. Name a
  shared facility for the concept it owns.

High-level FMM code should read like the algorithm. Geometry construction,
tree construction, operator construction, plan preparation, and plan execution
must be separate operations rather than hidden phases of one large routine.

## Layers and ownership

The default dependency direction is downward through this hierarchy:

```text
                         PUBLIC API
                             |
                             v
                            FMM
                             |
                             v
                            PLAN
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
          OPERATORS          TREE        GEOMETRY
              |              |              |
              +--------------+--------------+
                             |
                             v
                            MATH
                             |
                             v
                            CORE
```

Execution backends consume prepared plans:

```text
                   canonical/static plans
                             |
                  +----------+----------+
                  |          |          |
                  v          v          v
                 CPU       oneMKL      CUDA
```

The subsystem responsibilities are:

| Layer | Owns | May normally depend on |
|---|---|---|
| `core` | Precision, basic common types, output flags, small shared status/error types | Standard library only |
| `math` | `Vec3`, multi-indices, Laplace derivatives, Cartesian and spherical expansion mathematics | `core` |
| `geometry` | Point, rectangular-prism, and tetrahedron integration geometry and models | `math`, `core` |
| `tree` | Bounds, nodes, Morton machinery, uniform/adaptive construction, interaction topology | `math`, `core` |
| `operators` | Mathematical P2M, M2M, M2L, L2L, L2P, and P2P definitions/construction | `geometry`, `math`, `core` |
| `plan` | Immutable execution-oriented structures assembled from geometry, topology, and operators | `operators`, `tree`, `geometry`, `math`, `core` |
| `backend` | CPU, oneMKL, and CUDA resource ownership, upload, scheduling, and plan execution | `plan`, with low-level `math`/`core` where necessary |
| `fmm` | Public solver lifecycle and high-level near/far orchestration | `geometry`, `tree`, `plan`, `backend` |
| `cache` | Keys, metadata, validation, serialisation, and deserialisation of already-defined solver data | The data types it persists; it defines no mathematics |
| `bindings` | C, Fortran, and Python adaptation of the public library/FMM interface | Public API/FMM |

Dependencies such as `geometry -> backend`, `geometry -> CUDA`, `tree ->
CUDA`, `math -> FMM`, `operators -> FMM orchestration`, `operators ->
cuSPARSE`, and `plan -> Python` are prohibited by default. A deviation needs a
documented technical reason and review; convenience alone is insufficient.

## Geometry, operators, plans, and execution

Physical grains and FMM integration elements are distinct concepts:

```text
physical grain
      |
      | discretise (future functionality)
      v
point / rectangular-prism / tetrahedron integration elements
      |
      v
dip-fmm
```

The `geometry` layer describes those elements. The `tree` layer describes
spatial hierarchy and interaction topology. The `operators` layer defines the
mathematical maps on them. The `plan` layer packages immutable work for an
executor. Backends execute the plan and may own persistent backend resources;
they must not independently redefine the mathematics.

Integration elements use a representative-position convention. A point stores
its representative `Vec3` position and has no additional primitive record. A
finite rectangular prism stores its representative position and side lengths;
a tetrahedron stores its representative position and representative-relative
vertex offsets. No artificial `Point` marker or wrapper is required.

For P2P in particular, preserve this separation:

```text
canonical mathematical target-row interaction data
                         |
                         v
              derived execution packing
```

The canonical representation is the backend-independent mathematical truth.
Compact/SoA, leaf-packed, tensor-dictionary, signed/reduced-dictionary, and
BSR forms are derived representations selected for execution. They must be
validated against the canonical data and must not become competing operator
definitions.

## Static-geometry invariant

`dip-fmm` is designed for geometry that is prepared once while dipole moments
or magnetisation states change repeatedly:

```text
geometry
   |
   v
tree and interaction topology
   |
   v
canonical mathematical operators
   |
   v
immutable static plans
   |
   v
backend-specific persistent resources
   |
   v
repeated evaluations with changing moments
```

Keep geometry-dependent construction, packing, validation, allocation, and
device upload out of repeated evaluation wherever practical. Evaluation paths
reuse plans and persistent resources. Mutable coefficients, result buffers,
and scratch are state, not plans.

At the FMM level, execution should eventually be readable as:

```text
prepare changing input
enqueue/execute near field
enqueue/execute far field
combine contributions
return output in user order
```

Near/far overlap and the explicit level/interaction ordering needed for
correctness remain implementation concerns below that orchestration.

## Public and internal interfaces

`include/cdfmm/` is primarily for interfaces deliberately supported for
downstream users. `src/` owns implementation-only facilities. A type does not
become public merely because several translation units share it: internal
headers may live beside their implementation until an internal include tree is
worth introducing.

CUDA RAII types, device helpers, launch helpers, `.cuh` files, kernels, and
backend plan-upload details normally remain below `src/backend/cuda/`. Public
backend selection and genuinely supported diagnostic interfaces may remain in
`include/cdfmm/backend/`; CUDA implementation mechanics may not.

Compatibility is preserved throughout the behaviour-preserving refactor.
Moving a declaration is not permission to redesign its public API.

## Performance-aware CUDA decomposition

Source modularity must not force runtime modularity. A logical sequence such
as load, scale, multiply, reduce, and store may remain one kernel with inlined
device helpers. Do not introduce extra kernel launches, global-memory
intermediates, host/device transfers, synchronisation, or evaluation-loop
allocations merely to make files or functions shorter.

Future CUDA moves must preserve:

- persistent plan and buffer reuse;
- current numerical precision and accumulation behaviour;
- stream/event ordering and near/far overlap;
- cuBLAS/cuSPARSE handle lifetime and stream association;
- host/device transfer boundaries;
- kernel launch geometry and intentional fusion; and
- direct, partial-FMM, and full-device-FMM execution semantics.

Measure performance-sensitive changes against the pre-refactor baseline.

## Current and target repository structure

The current production layout is:

```text
dip-fmm/
|-- include/cdfmm/    canonical core/math/geometry/tree/operators/plan/backend
|                    plus compatibility shims
|-- src/              structured operators/plan/fmm/backend code, including
|                    explicit CPU/CUDA execution and bindings homes
|-- tests/            C++ and optional Fortran tests
|-- python_tests/     Python and notebook regression tests
|-- benchmarks/       C++ drivers and Python runners
|-- examples/         C++, Fortran, scripts, and notebooks
|-- python/           split pybind11 binding units
|-- fortran/          ISO_C_BINDING module
|-- tools/            cache precomputation tool
`-- docs/             user, mathematical, validation, and developer documents
```

The target taxonomy is approximate. Create a directory only when moving a
cohesive implementation into it; do not manufacture empty symmetry:

```text
dip-fmm/
|-- include/cdfmm/
|   |-- cdfmm.hpp
|   |-- core/
|   |-- math/
|   |-- geometry/
|   |-- tree/
|   |-- operators/       # implemented responsibility-specific interfaces
|   |-- plan/            # implemented immutable static representations
|   |-- backend/         # CPU static-plan apply and canonical CUDA APIs
|   `-- fmm/
|-- src/
|   |-- math/
|   |-- geometry/
|   |   |-- primitives/
|   |   |-- generation/       # future, not part of this refactor step
|   |   `-- refinement/       # future, not part of this refactor step
|   |-- tree/
|   |   |-- uniform/
|   |   `-- adaptive/
|   |-- operators/       # authoritative construction and dynamic application
|   |                    # p2m.cpp, m2m.cpp, m2l.cpp, l2l.cpp, l2p.cpp, m2p.cpp, p2p.cpp
|   |-- plan/            # precision.cpp and p2p packing builders
|   |-- backend/
|   |   |-- cpu/
|   |   |-- mkl/
|   |   `-- cuda/
|   |       |-- common/
|   |       |-- direct/
|   |       |-- p2p/          # complete extracted P2P backend
|   |       |-- m2l/          # reusable extracted M2L backend
|   |       |-- far_field/    # extracted CUDA far-field execution
|   |       `-- fmm/          # complete CUDA FMM orchestration
|   |-- fmm/
|   |-- cache/
|   `-- bindings/       C ABI implementation
|-- tests/
|   |-- unit/
|   |-- backend/
|   `-- integration/
|-- benchmarks/
|   |-- direct/
|   |-- p2p/
|   |-- far_field/
|   `-- fmm/
|-- examples/
`-- docs/
```

This target sketch focuses on production subsystems. Existing top-level
support areas such as `python_tests/`, `python/`, `fortran/`, and `tools/`
remain valid; the bindings boundary is implemented by the C ABI unit under
`src/bindings/` and the split Python units under `python/`.

The implemented operator and plan homes are:

```text
include/cdfmm/
|-- operators/
|   |-- p2m.hpp       P2M construction and dynamic application
|   |-- m2m.hpp       M2M translation
|   |-- m2l.hpp       M2L matrices and dynamic translation
|   |-- l2l.hpp       L2L translation
|   |-- l2p.hpp       L2P rows and evaluation
|   |-- m2p.hpp       direct multipole evaluation reference
|   |-- p2p.hpp       pair semantics and canonical P2P construction
|   `-- operators.hpp operator umbrella
|-- plan/
|   |-- static_coefficient.hpp  sparse P2M/translation maps
|   |-- m2l.hpp                 immutable M2L schedules and matrices
|   |-- l2p.hpp                 immutable L2P rows
|   |-- direct/dense.hpp        persistent dense-direct plan interface
|   `-- p2p/
|       |-- canonical.hpp       authoritative target-row P2P data
|       |-- compact.hpp         derived particle-row SoA packing
|       |-- leaf.hpp            derived dense leaf packing
|       |-- dictionary.hpp      derived Tensor6 dictionary packing
|       |-- signed_dictionary.hpp derived signed/reduced packing
|       `-- bsr.hpp              backend-neutral BSR(3) packing
|-- backend/cpu/{p2p,m2l,far_field}.hpp CPU execution interfaces
`-- backend/cpu/static_plan_apply.hpp  compatibility umbrella

src/
|-- operators/{p2m,m2m,m2l,l2l,l2p,m2p,p2p}.cpp  authoritative construction
|                                                  and dynamic application
|-- plan/{precision,direct,p2p}               plan preparation and representations
|-- cuboid.cpp                                compatibility geometry/pair math
|-- fmm/construction.cpp                      geometry normalisation and construction
|-- fmm/plan_preparation.cpp                  immutable plans, FP32 quantisation, cache calls
|-- fmm/execution_setup.cpp                   backend wiring and P2P policy
|-- fmm/evaluation.cpp                        complete near/far evaluation and timing
|-- fmm/far_field.cpp                         hierarchy sequencing
|-- fmm/diagnostics.cpp                       exact summary formatting
|-- fmm/uniform_fmm.cpp                       lifecycle, accessors, and inspection
|-- fmm/internal.hpp                          opaque backend-owner declarations
|-- backend/cpu/direct/dense.cpp                portable dense-direct execution
|-- backend/cpu/p2p/{executor,dictionary,near_field}.cpp
|                                               portable P2P and list-1 execution
|-- backend/cpu/m2l/executor.cpp                portable prepared M2L execution
|-- backend/cpu/far_field/{executor,entries,translation}.{cpp,hpp}
|                                               portable far-field mechanics
|-- backend/mkl/m2l.cpp                        grouped oneMKL M2L execution
|-- backend/mkl/direct/dense.cpp               oneMKL dense-direct execution
|-- backend/cuda/p2p/{internal.hpp,plan.cu}    complete CUDA P2P backend
|-- backend/cuda/m2l/{internal.hpp,plan.cu}    reusable CUDA M2L backend
|-- backend/cuda/far_field/{internal.hpp,executor.cu,entries.cuh,translation.cuh}
                                                CUDA far-field execution
|-- backend/cuda/fmm/{internal.hpp,plan.cu}     complete CUDA FMM orchestration
|-- cache/internal.hpp                         shared cache interface
|-- cache/io.cpp                               root policy and file container
|-- cache/format.cpp                           payload records
|-- cache/keys.cpp                             cache identity
|-- cache/universal.cpp                        translation-bank and periodic root
`-- cache/geometry.cpp                         geometry-plan payload
```

The high-level `UniformFmm` implementation is split by responsibility. The
construction unit normalises input geometry, builds the tree/topology, and
delegates construction of canonical operators. Plan preparation creates
immutable static plans, performs FP32 quantisation, and makes high-level cache
calls. Execution setup resolves the requested backend, wires its resources,
and owns the unchanged P2P policy and derived-packing selection. Evaluation
owns the complete near-plus-far lifecycle and whole-evaluation timing; the
hybrid CUDA path retains asynchronous P2P begin, CPU far-field work, finish,
and cancellation-guard behaviour. Far-field sequencing remains in
`far_field.cpp`. Diagnostics owns the exact initialisation-summary formatting.
The small `uniform_fmm.cpp` unit retains lifecycle, accessors, and inspection;
`internal.hpp` declares opaque MKL/CUDA owner wrappers without exposing backend
implementation types from the installed header.

The cache subsystem under `src/cache/` is split the same way. `io.cpp` owns the
cache root and `CDFMM_CACHE_DIR`/`CDFMM_DISABLE_CACHE` policy together with the
validated file container: header fields, payload checksum, memory-mapped
reads, and the unique-temporary-file plus `fsync` plus `rename` write that lets
independent processes race safely for the same deterministic file. `format.cpp`
owns the field-wise records for the solver types a payload holds, including the
direct FP32 decode paths. `keys.cpp` owns identity: the digest, the canonical
`1e-9` coordinate representation, compact uniform-grid and permutation-layout
recognition, and the key strings that name a file. `universal.cpp` and
`geometry.cpp` own the two payloads. `internal.hpp` is implementation-only and
is not installed; there is no public cache API.

The persistent format and the cache keys are compatibility contracts rather
than implementation details, because existing installations hold files written
by earlier builds. `plan_preparation.cpp` remains the coordinator: it asks the
cache for already-defined data, constructs what is missing through the operator
and plan layers, and asks the cache to write the result. Cache code performs no
operator mathematics, tree construction, plan policy, backend selection, or
P2P packing policy.

### Bindings boundary

The bindings refactor is complete. The unconditional C ABI implementation is
`src/bindings/c_api.cpp`, moved byte-identically from its former flat home;
`include/cdfmm/c_api.h` is unchanged and `CDFMM_ABI_VERSION` remains `1`.
The Fortran layer remains `ISO_C_BINDING -> C ABI -> supported C++`, so it does
not reach into solver internals. The former monolithic `python/bindings.cpp`
has been replaced by `python/internal.hpp`, `module.cpp` (the sole pybind11
entry point), and the subsystem units `core.cpp`, `geometry.cpp`, `tree.cpp`,
`operators.cpp`, `direct.cpp`, and `fmm.cpp`. CMake registers these units.

Binding units use supported canonical structured headers where available, with
only justified compatibility dependencies. They contain adaptation and
validation glue, not solver logic, and do not include internal backend headers.

The compatibility headers `cdfmm/operators.hpp` and
`cdfmm/static_operators.hpp` remain supported forwarding umbrellas; they do
not define a second operator or plan representation. `StaticFmmTopology`
stores topology-native occupied leaf interaction metadata; conversion to
derived leaf-packing records occurs at the P2P plan boundary. It remains the
principal seam for a later FMM/topology integration step.

The responsibility-specific files under `src/operators/` are the authoritative
homes for both mathematical operator construction and dynamic application.
The flat `src/operators.cpp` translation unit is retained only for thin
compatibility wrappers around those implementations; it does not own a second
set of operator formulas.

Dense direct follows the same plan/backend boundary. `plan/direct/dense.cpp`
validates geometry and prepares the six immutable target-major matrices;
`backend/cpu/direct/dense.cpp` owns the portable row-major nine-GEMV path;
and `backend/mkl/direct/dense.cpp` owns the guarded SGEMV/DGEMV calls and
availability query. The private dense-direct workspace keeps reusable staging
arrays beside execution while the public `DenseDirectPlan` remains a
source-compatible façade with value semantics.

CUDA direct execution follows the same boundary. The canonical public headers
are `include/cdfmm/backend/cuda/direct.hpp` and
`include/cdfmm/backend/cuda/dense_direct.hpp`; the legacy
`include/cdfmm/cuda_direct.hpp` and `include/cdfmm/cuda_cuboid.hpp` paths are
compatibility façades. Shared internal CUDA error and runtime handling lives
under `src/backend/cuda/common/`. Point O(N^2) direct execution and dense
cuBLAS direct execution live under `src/backend/cuda/direct/`, while
`src/backend/cuda/stub/direct.cpp` supplies the non-CUDA direct stubs. The
extracted P2P execution follows the same boundary: the canonical public
`include/cdfmm/backend/cuda/p2p.hpp` owns `CudaP2PPlan`, while
`include/cdfmm/cuda_p2p.hpp` remains a forwarding compatibility façade.
`src/backend/cuda/p2p/{internal.hpp,plan.cu}` owns all canonical, compact,
leaf, signed tensor-dictionary, and cuSPARSE BSR(3) executors in FP64 and
FP32, together with plan lifecycle, asynchronous evaluation, persistent
device/host state, and shared full-plan primitives. The non-CUDA boundary is
`src/backend/cuda/stub/p2p.cpp`. The canonical public
`include/cdfmm/backend/cuda/m2l.hpp` owns `CudaM2LPlan`, while the legacy
`src/cuda_m2l_plan.hpp` is a forwarding compatibility shim. The reusable
`src/backend/cuda/m2l/{internal.hpp,plan.cu}` executor owns M2L device
metadata, kernels, bounded scaled-multipole scratch, and both FP64/FP32
lifecycles; standalone `CudaM2LPlan` and `CudaFullPlan` consume that same
executor. The non-CUDA boundary is `src/backend/cuda/stub/m2l.cpp`.
The far-field backend is internal only: `src/backend/cuda/far_field/` owns the
immutable FP32/FP64 P2M and L2P entries, coefficient degrees, M2M/L2L matrices,
interactions and metadata, uploads, lifecycle, statistics, and kernels.
`entries.cuh` owns shared P2M/L2P entry application mechanics, while
`translation.cuh` owns shared M2M/L2L translation mechanics. It adds no public
API, stub, or new library. Complete CUDA FMM declarations and orchestration are
in `src/backend/cuda/fmm/{internal.hpp,plan.cu}`; changing moments/coefficient/
field buffers, permutations, P2P and separate M2L executor wiring,
streams/events/timing, near/far overlap, combination/reordering, and D2H
transfer remain there. Direct, P2P, M2L, and far-field remain authoritative
execution backends.
These moves preserve behaviour and performance, including resource lifetime,
precision, launch, and transfer semantics.

The CPU FMM layer follows the same orchestration/execution boundary.
`fmm/far_field.cpp` retains CPU-side P2M, M2M, M2L backend dispatch, L2L, and L2P
sequencing. `backend/mkl/m2l.cpp` derives stable transfer-class groups once
from the canonical `StaticM2LPlan`, owns reusable FP32/FP64 gather and
translation buffers, and performs the guarded SGEMM/DGEMM gather/multiply/
serial-scatter path. `UniformFmm` references that state through an opaque
internal owner; no vendor-oriented group or scratch layout appears in the
installed header. Portable CPU list-1 and prepared-plan execution live under
`backend/cpu/p2p`, prepared M2L under `backend/cpu/m2l`, and P2M/L2P entries
plus M2M/L2L translations under `backend/cpu/far_field`. The old
`static_plan_apply.hpp` remains a source-compatible umbrella.

`generation/` will eventually own physical grain generation. `refinement/`
will eventually own prism and tetrahedron refinement. Geometry packing belongs
with plan construction or a geometry-to-plan adapter according to whether it
is canonical geometry data or an execution representation. The P2P execution
packings listed above are implemented; grain generation, discretisation, and
geometric refinement remain outside this step.

## Deferred refactor inventory

This inventory records remaining pressure points; it does not authorise a
future repair by itself.

The initial audit found these concrete boundary violations in the remaining
transitional layout:

- `tree/static_topology.hpp` is the canonical topology-only interface;
  `tree/uniform_topology.hpp` keeps the UniformTree adapter as a separate
  transitional seam between tree facts and static plans. The flat
  `static_topology.hpp` remains a compatibility forwarding header.
- `adaptive_tree.hpp` returns `StaticFmmTopology` directly, coupling adaptive
  construction to the current static-plan representation.
- `cuboid.hpp` remains a compatibility umbrella for finite geometry/pair math
  and the dense-direct public API. `DenseDirectPlan` is now declared by
  `plan/direct/dense.hpp` and implemented by `src/plan/direct/dense.cpp`.
  Portable, oneMKL, and CUDA direct execution now have dedicated internal
  backend homes; the façades retain only their supported API surfaces.
- compatibility `static_operators.hpp/.cpp` remain as forwarding umbrellas;
  their former mixed implementation has been separated into operators, plans,
  and the portable CPU apply boundary described below.
- `fmm/far_field.cpp` contains CPU expansion sequencing and no longer contains
  oneMKL vendor mechanics; complete CUDA FMM orchestration is in
  `backend/cuda/fmm/plan.cu`.
  `periodic.cpp` includes `uniform_tree.hpp`, and `parameter_selection.hpp`
  includes the complete `uniform_fmm.hpp` API.
- the cache subsystem under `cache/` separates the file container and root
  policy, payload records, identity, and the universal/periodic and
  geometry-plan payloads. Its entry points remain `UniformFmm` members, so
  cache code still has intimate knowledge of the topology and operator
  representations it persists; deeper encapsulation awaits an API/ABI step.
  `cache/format.cpp` also builds the derived compact P2P representation while
  decoding the canonical blocks, so the canonical-to-compact rule is stated
  both there and in `plan/p2p/compact.cpp`.
- the remaining public CUDA/FMM headers expose transitional plan concepts and
  depend on broad operator or geometry headers; direct, P2P, and M2L CUDA APIs
  now have canonical `backend/cuda/` headers with legacy forwarding façades.
- the root CMake target registers the CPU subsystems together with explicit
  CUDA common/direct/P2P/M2L/far-field/FMM units; CUDA libraries are currently
  propagated from `cdfmm_core` to consumers.

The largest remaining responsibility-review candidates are
`backend/cuda/p2p/plan.cu`, `geometry/primitives/tetrahedron.cpp`, and
`backend/cuda/fmm/plan.cu`. These measurements locate remaining audit work;
they do not require mechanical splitting.

### Foundational layout: core, math, geometry, and tree

- Foundational precision, output, vector, index, derivative, Cartesian, and
  spherical interfaces have canonical subsystem headers with flat shims.
- Geometry models and exact rectangular-prism/tetrahedron primitives have
  canonical homes; point geometry remains its representative `Vec3` plus model
  selection, without an artificial marker type.
- Common node, Morton, indexing, and tree-statistics concepts are separated
  from uniform and adaptive implementations.
- Internal cubic root-box resolution and validation is shared by UniformTree
  and AdaptiveTree in `src/tree/common/root_box.{hpp,cpp}`. Shared resolution
  retains zero for coincident roots; AdaptiveTree applies its existing fallback
  half-width `1` at its call site. The helper is not part of the public API.
- `StaticFmmTopology`, dense-direct pair dispatch, and adaptive plan adaptation
  remain transitional seams for the operator/plan/backend phases.

### Implemented: operators, static plans, and CPU/oneMKL FMM boundaries

- Mathematical P2M, M2M, M2L, L2L, L2P, and P2P construction now lives under
  `include/cdfmm/operators/` and `src/operators/`. Geometry-specific P2M/L2P
  and P2P dispatch remains below the operator layer and uses the established
  geometry primitives.
- Canonical static data is represented under `include/cdfmm/plan/`, with
  FP64-to-FP32 conversion in `src/plan/precision.cpp`. Static M2L data owns
  immutable matrices, scaling, and interaction schedules without backend
  resources.
- `plan/p2p/canonical.hpp` is authoritative for target-row P2P tensors and
  identity markers. Compact/SoA, leaf, tensor dictionary, signed/reduced
  dictionary, and BSR forms are deterministic derived packings under
  `plan/p2p/`; BSR construction is backend-neutral and does not depend on
  CUDA or cuSPARSE.
- Portable CPU application is isolated under
  `backend/cpu/{p2p,m2l,far_field}`. P2P consumes canonical or derived plans,
  M2L consumes prepared schedules, and far-field entry/translation helpers
  consume prepared static operators; none redefine pair or translation
  mathematics.
- Dense-direct plan preparation remains under `plan/direct`; portable execution
  is in `backend/cpu/direct`, oneMKL execution is in `backend/mkl/direct`, and
  reusable staging is private backend execution state.
- High-level FMM lifecycle and far-field pass orchestration live under
  `src/fmm`; portable list-1 and static-plan execution live under
  `src/backend/cpu`; and grouped oneMKL M2L packing, reusable scratch, vendor
  calls, thread control, and availability live under `src/backend/mkl`.
- CUDA direct/common execution now lives under `src/backend/cuda/{common,direct}`
  and the complete P2P, M2L, far-field, and full-FMM backends under
  `src/backend/cuda/{p2p,m2l,far_field,fmm}/`,
  with canonical public headers under `include/cdfmm/backend/cuda/` and
  legacy forwarding façades retained. The reusable M2L executor is consumed
  by both standalone `CudaM2LPlan` and `CudaFullPlan`; `backend/cuda/fmm/plan.cu`
  coordinates and enqueues the P2M/M2M/L2L/L2P stages and full-FMM orchestration
  plus internal backend wiring, while static execution lives in
  `backend/cuda/far_field`.

The remaining transitional seam is `StaticFmmTopology`: it adapts tree
interaction topology to topology-native records consumed by FMM orchestration
and the P2P plan boundary. Tree ownership remains spatial
hierarchy/topology, while derived schedule assembly stays behind the plan
boundary without making the tree depend on a particular P2P packing.

### Stable backend boundaries

- Keep the accepted direct/common/P2P/M2L/far-field/full-FMM extraction stable.
- Preserve the separation of immutable far-field execution state from mutable
  moments, coefficient/field buffers, streams/events, timing, overlap,
  combination/reordering, and D2H resources.
- Preserve and benchmark all transfer, launch, overlap, and reuse semantics;
  source boundaries must not add runtime work.

### Remaining Phase 1 work

- Review remaining compatibility/transitional source seams and remove or
  narrow them only when an explicit compatibility plan exists.
- Resolve the duplicated canonical-to-compact P2P packing rule, which the
  cache decode in `cache/format.cpp` and the plan builder in
  `plan/p2p/compact.cpp` currently state independently. The cache decode fuses
  packing into its single pass deliberately, so this is a scoped plan/cache
  boundary decision rather than a mechanical de-duplication.
- Complete whole-refactor validation across the supported CPU, oneMKL, CUDA,
  bindings, and integration configurations; unavailable hardware or optional
  toolchains must remain explicitly reported rather than inferred as passing.

## Refactor validation contract

Each production-code step starts and ends with the same relevant build and
test configurations. Numerical tests must compare unchanged formulas,
precision modes, output flags, geometries, tree modes, periodic behaviour,
cache behaviour, and supported CPU/oneMKL/CUDA paths. Performance-sensitive
backend changes additionally compare plan reuse, transfers, allocations,
launches, synchronisation, and representative benchmark results.

The high-level `UniformFmm`, cache, and bindings boundaries are complete. These
refactor steps change file ownership and include structure without changing
runtime algorithms or public names. Compatibility/transitional source review
and whole-refactor validation are the remaining Phase 1 work. API redesign,
packing, generation, discretisation, and refinement remain deferred.
