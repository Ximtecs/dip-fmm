# Developer architecture

This document is the architectural contract for the `v0.2` refactor. It
describes the ownership and dependency boundaries implemented in the
repository. The foundational `core`, `math`, `geometry`, and `tree` layers,
operators, static plans, FMM orchestration, portable CPU and oneMKL execution,
and CUDA execution all have responsibility-specific homes. The high-level
`UniformFmm` implementation is also decomposed into construction, plan
preparation, backend setup, evaluation, far-field sequencing, diagnostics, and
lifecycle/accessor units. Cache persistence and the bindings boundary are now
structured responsibility-specific layers. The compatibility/transitional-source
review and the whole-refactor validation are both complete, so **Phase 1 is
COMPLETE**. The validation matrix and its limitations are recorded under
"Phase 1 closure" below.

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

`cdfmm/static_operators.hpp` is a genuine forwarding umbrella. `cdfmm/operators.hpp`
is different in kind: it declares the flat dynamic-operator names in their own
right and is a compatibility API adapter, backed by the thin delegations in
`src/operators.cpp`. Neither defines a second operator or plan representation. `StaticFmmTopology`
stores topology-native occupied leaf interaction metadata; conversion to
derived leaf-packing records occurs at the P2P plan boundary. It remains the
principal seam for a later FMM/topology integration step.

The responsibility-specific files under `src/operators/` are the authoritative
homes for both mathematical operator construction and dynamic application.
The flat `src/operators.cpp` translation unit is retained only for thin
compatibility wrappers around those implementations; it does not own a second
set of operator formulas. `src/operators/p2p.cpp` also owns the authoritative
point/rectangular-prism pair-tensor construction as
`cdfmm::operators::p2p::build_pair`, with the flat `cdfmm::build_pair_tensor`
defined beside it as a thin delegation. The prism-averaged monomial is owned by
`src/geometry/primitives/rectangular_prism.cpp` as
`rectangular_prism_averaged_monomial`, with `cuboid_averaged_monomial` as its
flat spelling. The former `src/cuboid.cpp` home no longer exists, and
`include/cdfmm/cuboid.hpp` is now a pure compatibility façade.

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
`include/cdfmm/backend/cuda/m2l.hpp` owns `CudaM2LPlan`; the former internal
`src/cuda_m2l_plan.hpp` shim is removed and its consumers include the canonical
header directly. The reusable
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

## Compatibility surface

The compatibility/transitional-source review is complete. The resulting surface
is intentional, and the distinction below is the rule for future work.

Canonical structured implementation is the single authority for a concept. An
intentional public compatibility façade is a supported pre-v0.2 include path or
callable name that forwards to that authority and owns no mathematics.

- Every flat header under `include/cdfmm/` was an installed public path at the
  `v0.1.0` tag, so all are retained. Removing one is an API change requiring
  separate approval. A façade also reproduces the pre-v0.2 transitive surface
  of its path.
- Compatibility flows one way. Canonical headers under
  `include/cdfmm/{core,math,geometry,tree,operators,plan,backend}` and canonical
  implementation under `src/` and `python/` include canonical headers; the flat
  façades include those. The deliberate exceptions are `src/operators.cpp`,
  which implements `cdfmm/operators.hpp`, and `python/internal.hpp`, which must
  see the flat operator declarations the Python module exports.
- `cdfmm/timings.hpp`, `cdfmm/periodic.hpp`, `cdfmm/uniform_fmm.hpp`,
  `cdfmm/validation.hpp`, `cdfmm/parameter_selection.hpp`, and
  `cdfmm/tensor_dictionary.hpp` are substantive public headers, not façades.
  They are flat only because no canonical subsystem home exists yet; their
  migration is deferred, not overdue cleanup.
- Internal `src/` headers carry no downstream obligation. The unused
  `src/cuda_fmm_plan.hpp`, `src/cuda_m2l_plan.hpp`, `src/cuda_p2p_plan.hpp` and
  the dead `src/static_operators.cpp` were removed on that basis.
- `tests/test_foundational_headers.cpp` and the `test_cuda_legacy_*_header.cpp`
  units are deliberate compatibility coverage. They must keep their flat
  includes, and they assert that legacy and canonical spellings agree.

## Deferred refactor inventory

This inventory records remaining pressure points; it does not authorise a
future repair by itself.

The initial audit found these concrete boundary violations in the remaining
transitional layout:

- `tree/static_topology.hpp` is the canonical topology-only interface;
  `tree/uniform_topology.hpp` declares its UniformTree producer
  (`build_uniform_fmm_topology`) and `adaptive_tree.hpp` returns the same
  `StaticFmmTopology` from `AdaptiveTree`'s own construction. A later
  tree/topology-boundary audit (see "Tree/topology boundary cleanup" below)
  found this is ordinary intra-`tree`-layer usage, not a tree-to-plan seam:
  `StaticFmmTopology` holds only spatial-tree and interaction-topology facts,
  no operator coefficients, execution packing, or backend state. The flat
  `static_topology.hpp` remains a compatibility forwarding header.
- `cuboid.hpp` is a compatibility façade that declares nothing of its own.
  `DenseDirectPlan` is declared by `plan/direct/dense.hpp`, the prism record,
  `CuboidSize`, and the averaged monomial by
  `geometry/primitives/rectangular_prism.hpp`, and the pair tensor by
  `operators/p2p.hpp`. Portable, oneMKL, and CUDA direct execution have
  dedicated internal backend homes; the façades retain only their supported
  API surfaces.
- compatibility `static_operators.hpp/.cpp` remain as forwarding umbrellas;
  their former mixed implementation has been separated into operators, plans,
  and the portable CPU apply boundary described below.
- `fmm/far_field.cpp` contains CPU expansion sequencing and no longer contains
  oneMKL vendor mechanics; complete CUDA FMM orchestration is in
  `backend/cuda/fmm/plan.cu`.
  `periodic.cpp` now includes the narrow `cdfmm/tree/indexing.hpp`, but
  `parameter_selection.hpp` still includes the complete `uniform_fmm.hpp` API.
- the cache subsystem under `cache/` separates the file container and root
  policy, payload records, identity, and the universal/periodic and
  geometry-plan payloads. Its entry points remain `UniformFmm` members, so
  cache code still has intimate knowledge of the topology and operator
  representations it persists; deeper encapsulation awaits an API/ABI step.
  `cache/format.cpp` still builds the derived compact P2P representation
  while decoding the canonical blocks in the same pass, but it now does so by
  calling `assign_static_p2p_compact_row` (`src/plan/p2p/compact_row.hpp`, an
  internal implementation header, not an installed public path), the same
  row-packing primitive `plan/p2p/compact.cpp` uses; the canonical-to-compact
  rule itself is stated once. Only the fused-pass *choice* remains cache-local,
  as a performance decision, not a restatement of the mapping.
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
- Dense-direct pair dispatch remains a transitional seam for the
  operator/plan/backend phases (the isolated `plan/direct/dense.cpp`
  compatibility edge recorded under "Ownership and dependency audit" below).
  `StaticFmmTopology` and adaptive tree construction were audited in the
  tree/topology boundary cleanup and found to be legitimate tree-owned
  topology, not a plan-layer seam; see "Tree/topology boundary cleanup".

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

`StaticFmmTopology` is tree-owned interaction topology, not a plan artefact:
the tree/topology boundary cleanup (see "Tree/topology boundary cleanup")
confirmed every field is a spatial-tree fact, an interaction-topology fact,
or schedule indexing over those facts, with no operator coefficients,
execution packing, or backend state. Both `build_uniform_fmm_topology` (from
a built `UniformTree`) and `AdaptiveTree`'s own construction populate it
directly; `fmm` and `plan` consume the result through
`shared_ptr<const StaticFmmTopology>` and never rebuild it. Tree ownership
remains spatial hierarchy/topology, while derived schedule assembly stays
behind the plan boundary without making the tree depend on a particular P2P
packing.

### Stable backend boundaries

- Keep the accepted direct/common/P2P/M2L/far-field/full-FMM extraction stable.
- Preserve the separation of immutable far-field execution state from mutable
  moments, coefficient/field buffers, streams/events, timing, overlap,
  combination/reordering, and D2H resources.
- Preserve and benchmark all transfer, launch, overlap, and reuse semantics;
  source boundaries must not add runtime work.

### Phase 1 work: complete

The compatibility/transitional-source review is complete, and whole-refactor
validation across the supported CPU, oneMKL, CUDA, bindings, cache, install,
and documentation configurations has been performed. See "Phase 1 closure".

One item raised here during the refactor was deferred rather than closed: the
duplicated canonical-to-compact P2P packing rule, independently stated by the
cache decode in `cache/format.cpp` and the plan builder in
`plan/p2p/compact.cpp`. A follow-up internal-duplication cleanup, after Phase
1 closure, resolved the duplicated *rule* without touching the plan/cache
boundary: both call sites now share one row-packing primitive,
`assign_static_p2p_compact_row`. It was first added to the installed
`include/cdfmm/plan/p2p/compact.hpp`, then relocated to the internal
`src/plan/p2p/compact_row.hpp` once it became clear the helper had no
downstream purpose beyond letting `plan/p2p/compact.cpp` and
`cache/format.cpp` share packing mechanics; `compact.hpp` keeps the public
`StaticP2PCompactPlan`/`FloatStaticP2PCompactPlan` types and builders. The
cache decode still fuses canonical decode and compact construction into one
pass deliberately for performance; that fused-pass structure, and the
broader question of whether cache entry points should stop being `UniformFmm`
members that know the topology/operator representations they persist, remain
open and are recorded as a separate later cache/plan-boundary item, not
mechanical de-duplication.

That same cleanup also removed a small duplicated `n!` primitive: a
private `monomial_factorial` in `src/geometry/primitives/rectangular_prism.cpp`
restated the loop already implemented by `MultiIndexSet::factorial`
(`include/cdfmm/math/multi_index.hpp`). The prism code now calls the `math`
implementation directly, matching the `geometry -> math` dependency direction;
no new file or shared `utils`/`helpers` header was introduced.

Before that, the P2P row-packing helper itself moved once more: a narrow
follow-up correction relocated `assign_static_p2p_compact_row` from the
installed `include/cdfmm/plan/p2p/compact.hpp` to the internal
`src/plan/p2p/compact_row.hpp`, because the helper existed solely so
`plan/p2p/compact.cpp` and `cache/format.cpp` could share packing mechanics,
not as a supported downstream API. `compact.hpp` keeps the public
`StaticP2PCompactPlan`/`FloatStaticP2PCompactPlan` types and builders; the
canonical-to-compact mapping and the fused cache-decode pass are unchanged.

### Tree/topology boundary cleanup

A separate follow-up task, after the internal-duplication cleanup, audited
the `StaticFmmTopology`/tree-to-plan boundary that Phase 1 left recorded as a
transitional seam (see the former wording in "Deferred refactor inventory"
and "Phase 2 handoff"). The audit read `StaticFmmTopology` field by field —
`Node`, permutations, sorted positions, occupied leaves, `m2m_edges`/
`l2l_edges`, `m2l_interactions`, `p2p_leaf_records`, row/level offsets, and
the coordinate-normalisation fields — against every producer
(`build_uniform_fmm_topology`, `AdaptiveTree`'s own construction) and every
consumer (`fmm/construction.cpp`, `plan_preparation.cpp`,
`execution_setup.cpp`, `far_field.cpp`, `cache/geometry.cpp`,
`backend/cpu/p2p/near_field.cpp`, `python/tree.cpp`, and the C++/Python test
suites). The finding: every field is a spatial-tree fact, an
interaction-topology fact (near/far admissibility, M2M/L2L translation
structure, P2P leaf pairing), or CSR-style schedule indexing over those
facts. None is an operator coefficient, a derived P2P execution packing, or
backend state — `plan/p2p/*` and `src/fmm/plan_preparation.cpp` build the
canonical operator data and derived packings strictly downstream, from
`StaticP2PLeafRecord`/`StaticM2LInteraction` fields, never the other way
round. `fmm` consumes the result through
`shared_ptr<const StaticFmmTopology>` by reference and never rebuilds it;
the cache subsystem never persists `StaticFmmTopology` directly and disables
caching entirely whenever a topology is supplied externally
(`cache/keys.cpp`'s `supplied_topology_` check), so `AdaptiveTree`'s topology
never touches the persistent cache format at all.

The `AdaptiveTree` constructor was also checked for whether it interleaves
spatial-octree construction with far/near interaction-topology assembly in a
way that would justify splitting them into two internal representations.
It does not: the constructor already runs the octant partition and M2M/L2L
edge derivation as one pass, then the dual-tree near/far `visit` as a
strictly later pass over the completed tree, with `tree_seconds_` and
`interaction_seconds_` already timing exactly that boundary. Introducing a
separate adaptive-only spatial-tree type to mirror `UniformTree` would
duplicate nearly every field `StaticFmmTopology` already has (adaptive
nodes carry no Morton indices or other data that would make a distinct
representation non-redundant, unlike `UniformTree`'s own `TreeNode`, which
has genuinely different content: level-wise dense Morton arrays and
`list1`/`list2` candidate lists that are never stored in `StaticFmmTopology`
at all). Manufacturing that duplicate type purely to make the layering
diagram prettier was rejected; `UniformTree::build` itself remains one
monolithic, internally phase-timed function for the same reason, so a forced
split in `AdaptiveTree` alone would also be inconsistent with its own
sibling's established construction style.

The conclusion is that `StaticFmmTopology` was already correctly tree-owned
topology, and the previously recorded "transitional tree-to-plan coupling"
in `adaptive_tree.hpp`'s `@warning` and in `docs/architecture.md` was a
documentation defect, not a live architectural problem: the seam appears to
have been named while `StaticFmmTopology`'s ownership was still unsettled,
and the language was never revisited once the type's contents were fully
audited. No production code changed; the misleading comment and the doc
passages that repeated it were corrected, and the two resolved items were
removed from the "Deferred refactor inventory" and "Phase 2 handoff" lists
below. Everything else about this boundary — field layout, ordering,
`validate()`'s invariants, the `supplied_topology_` M2L transfer-class branch
in `plan_preparation.cpp` that treats uniform- and adaptive-built topologies
differently, and all public `AdaptiveTree`/`StaticFmmTopology`/
`build_uniform_fmm_topology` signatures — is unchanged.

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
and whole-refactor validation are both complete, so Phase 1 is closed. API
redesign, packing, generation, discretisation, and refinement remain deferred.

## Phase 1 closure

Phase 1 of the `v0.2` architecture refactor is **COMPLETE**. The preserved
pre-refactor baseline remains the `v0.1.0` annotated tag (commit `2d3d4ea`) and
the `release/v0.1` branch; neither was moved. Closure rests on the evidence
below rather than on assertion.

### Ownership and dependency audit

A repository-wide include-edge audit of `include/cdfmm/`, `src/`, and `python/`
found no violation of the prohibited-by-default list: no `geometry -> backend`
or `geometry -> CUDA`, no `tree -> CUDA`, no `math -> FMM`, no
`operators -> orchestration`, no `operators -> cuBLAS/cuSPARSE`, no
`plan -> Python`, no `CPU backend -> CUDA internals`, and no binding unit
including an internal `src/` header. No CUDA backend redefines canonical
mathematics or topology: each consumes prepared `Static*Plan` data.

Compatibility flows one way. Every flat header under `include/cdfmm/` was an
installed public path at `v0.1.0` and is retained; canonical subsystem headers
and canonical implementation include canonical headers, and the façades include
those. The audited exceptions are each deliberate and are recorded here in full:

- `src/operators.cpp` implements `cdfmm/operators.hpp`, the flat operator
  compatibility API, and so includes it.
- `python/internal.hpp` must see the flat operator declarations the Python
  module exports.
- `cdfmm/timings.hpp` and `cdfmm/periodic.hpp` are still included by some
  canonical headers. They are substantive public headers awaiting a subsystem
  home, not façades; their migration is deferred to Phase 2.
- `src/plan/direct/dense.cpp` includes `backend/cpu/direct/dense.hpp` and
  `backend/mkl/direct/dense.hpp`. `DenseDirectPlan::evaluate()` is a pre-v0.2
  public method, so the plan object itself must dispatch to an executor. This
  is an isolated compatibility seam confined to one translation unit; the
  public `plan/direct/dense.hpp` header depends on no backend, and no other
  file under `src/plan/` or `src/operators/` includes a backend header.
- `src/backend/cuda/fmm/internal.hpp` includes `cdfmm/uniform_fmm.hpp` because
  the CUDA backend *implements* the public availability queries declared there
  (`cuda_m2l_p2p_available`, `cuda_m2l_available`). A translation unit must see
  the declaration of the function it defines, so this is a supported public-API
  relationship, not an implementation-layer inversion. Relocating those
  declarations to a backend header would be an API change and is deferred.
- `src/cache/internal.hpp` includes `cdfmm/uniform_fmm.hpp`: the sanctioned
  `cache -> already-defined solver data` edge. Cache code performs no operator
  mathematics, tree construction, plan policy, or backend selection.

`tree/adaptive_tree.hpp` returning `StaticFmmTopology` is not a
dependency-direction exception: both types live in the `tree` layer, and the
tree/topology boundary cleanup confirmed `StaticFmmTopology` carries no
plan/backend data, so this is ordinary intra-layer usage. It was listed here
as a deferred exception through Phase 1; see "Tree/topology boundary
cleanup".

Every `.cpp`/`.cu` file under `src/` is referenced by `CMakeLists.txt`; no
stale or transitional source remains compiled, and no build file references a
removed path.

### Comparison against `v0.1.0`

No header present at `v0.1.0` is absent at HEAD: all 29 pre-refactor public
paths are retained as compatibility façades. `include/cdfmm/c_api.h` is
byte-identical to `v0.1.0` (same blob hash), `CDFMM_ABI_VERSION` remains `1`,
and the built `libcdfmm_c.so` exports exactly the expected 14 `cdfmm_*`
functions. `fortran/cdfmm_fortran.f90` is likewise byte-identical. The Python
surface is unchanged: 20 classes, 14 enums, 29 module functions, 151
methods/properties, 130 `py::arg` entries with identical defaults, and 123
diagnostic dictionary keys all compare zero-diff against the pre-split
`python/bindings.cpp`.

The installed-header rule widened from `c_api.h` only to the whole
`include/cdfmm/` tree with `AGENTS.md` excluded. That is a deliberate widening,
not a narrowing, and no installed header points into `src/`, `python/`, or a
build-tree path.

Behaviour is preserved exactly where it can be compared directly: a probe
building five representative plans (spherical FP32, spherical FP64, prism FP32,
Cartesian FP64, and periodic FP64) compiles unchanged against both `v0.1.0` and
HEAD public headers and produces **bit-identical** field results.

`9003b666` (tetrahedron P2P construction) is separate, intentional
performance/numerical work carried on this branch and is not an architecture
regression.

### Validation matrix

All builds used the `cdfmm` Conda environment: GCC 15.3, CMake 4.4.3,
CUDA 13.3.73, oneMKL 2026.1, Python 3.11, on an NVIDIA RTX 5090 (SM120,
driver 595.84). Python runs verified `cdfmm.__file__` resolved to the
just-built module in every case.

| Configuration | Build | CTest | Python |
|---|---|---|---|
| Portable CPU (`dev` preset, fresh) | clean, no warnings | 198/198 passed, 4 skipped | 137 passed, 7 skipped |
| oneMKL, no CUDA | clean | 198/198 passed, 3 skipped | 139 passed, 5 skipped |
| CUDA, no oneMKL (`cuda` preset, fresh, SM120) | clean | 198/198 passed, 1 skipped | 141 passed, 3 skipped |
| CUDA + oneMKL (`notebooks` preset, fresh) | clean | **198/198 passed, 0 skipped** | 143 passed, 1 skipped |
| Fortran interface (`ifx` 2025.2.1) | clean | 199/199 passed, 4 skipped | n/a |

Every skip is a genuinely absent optional dependency or device: CUDA
unavailable, oneMKL unavailable, or the optional external MagTense package. The
CUDA + oneMKL configuration exercises the complete matrix with no skips.

CUDA numerical execution is real, not vacuous. In a CUDA build the
device-backed cases run for measurable time (for example "spherical CUDA
partial and full agree with CPU static" 0.36 s, "cache and no-cache CUDA-full
paths agree" 0.35 s, "shared CUDA M2L executor agrees with the canonical CPU
plan" 2.05 s), whereas in a portable build the same cases report 0.00 s because
they return early. CPU-reference, CPU-static, CUDA M2L/P2P hybrid, and CUDA
full-FMM agreement is therefore covered by tests that genuinely executed on the
GPU.

### Cache backward compatibility

A cache corpus of ten files was generated by the `v0.1.0` baseline build
(universal spherical FP32/FP64, universal Cartesian FP64, a periodic root, and
five geometry plans). Running HEAD against that untouched corpus produced, for
every one of the five cases: the **same key strings**, `universal.hit: true`,
`geometry.hit: true`, `periodic.hit: true` where applicable, and
`cache.bytes_written: 0`. All ten files remained byte-identical afterwards, and
the field results were bit-identical to the baseline run. The schema version,
operator version, magic, hash inputs, and format layout were not changed.

### Installation and downstream consumption

A clean install to an isolated prefix produced 84 files: 79 headers, the
`cdfmm_c` shared library, the `cdfmm-precompute` tool, the Python extension,
and the installed Fortran source. No `AGENTS.md` was installed and no installed
header references `src/` or `python/`. Every one of the 79 installed C++
headers compiles standalone against the prefix alone, with no include-ordering
requirement. A translation unit mixing all legacy flat paths with all canonical
subsystem paths compiles in both include orders. A downstream C consumer built
only against the installed prefix creates a plan, evaluates it, and reproduces
the analytic two-dipole field to a relative error of `1.6e-16`.

Note that dip-fmm exports no CMake package configuration, at `v0.1.0` or at
HEAD; downstream C++ consumers link `cdfmm_c` directly. This is a pre-existing
characteristic, not a refactor regression.

### Documentation

`sphinx-build -W --keep-going -b html docs docs/_build/html` now succeeds with
**zero** warnings. The two long-standing `docs/api.rst` C++ declaration
warnings were a genuine Phase-1 defect and were fixed rather than tolerated:
`docs/Doxyfile` predefined `CDFMM_HOST_DEVICE`, a macro used nowhere, while the
canonical `plan/p2p/canonical.hpp` header introduced by the refactor guards two
declarations with `CDFMM_PLAN_HOST_DEVICE`. With `EXPAND_ONLY_PREDEF = YES`,
Doxygen left that token unexpanded and Breathe could not parse the two
declarations. `PREDEFINED` now names the macro actually in use.

### Performance sanity

A representative 20k-source/20k-target, depth-3, order-4 FP32 evaluation gives
a median of 11.47 ms on the portable CPU static-matrix backend at HEAD against
11.56 ms at `v0.1.0` — no regression. CUDA medians are 0.96 ms (partial) and
0.88 ms (full), so near/far overlap and persistent device state are intact and
no accidental backend fallback occurs.

### Defects found and fixed during closure

- `tests/test_fortran_api.f90` requested a periodic cubic cell of side 2
  centred at the origin while its two prisms sit at `z = 0` and `z = 1` with
  side 0.2, so the `z = 1` prism lay outside the periodic root and plan
  creation correctly failed. The cell centre is now `(0, 0, 0.5)`, matching the
  period-2 chain the case intends. This is a **pre-existing** defect: the file
  is byte-identical to `v0.1.0`, and a `v0.1.0` build of the same test fails
  identically. It went undetected because no Fortran compiler was available in
  any previous session.
- `docs/Doxyfile` named a macro that does not exist, as described above.
- `include/cdfmm/AGENTS.md`, `src/AGENTS.md`, and three passages of this
  document described already-implemented work as pending. Documentation that
  contradicts the implementation is a Phase-1 defect, and these were corrected.

## Phase 2 handoff

Recorded, not started. Phase 2 requires its own explicit task.

- **Implementation deduplication.** The canonical-to-compact P2P packing rule
  stated independently by `cache/format.cpp` and `plan/p2p/compact.cpp`; the
  factorial helper duplicated against `MultiIndexSet::factorial`.
- **Remaining flat-file and public-header homes.** Relocating the coherent flat
  units `src/periodic.cpp`, `src/parameter_selection.cpp`, and
  `src/validation.cpp`; giving `timings.hpp`, `periodic.hpp`, `uniform_fmm.hpp`,
  `validation.hpp`, `parameter_selection.hpp`, and `tensor_dictionary.hpp`
  subsystem homes behind their retained public façades. Do these where they
  clarify ownership, not for symmetry.
- **Remaining public-header seam.** `parameter_selection.hpp` including the
  complete `uniform_fmm.hpp`. (The `StaticFmmTopology`/tree-to-plan seam
  recorded here through Phase 1 — `tree/uniform_topology.hpp` and
  `adaptive_tree.hpp` returning `StaticFmmTopology` — was audited and
  resolved by the tree/topology boundary cleanup: see "Tree/topology
  boundary cleanup".)
- **Plan/cache duplication and encapsulation.** Cache entry points remain
  `UniformFmm` members, so cache code retains intimate knowledge of the
  representations it persists. Deeper encapsulation needs an API/ABI step.
- **API and internal simplification.** Relocating the CUDA availability queries
  out of `uniform_fmm.hpp`; introducing a CMake package configuration; stopping
  CUDA libraries propagating from `cdfmm_core` to consumers. Each is an API or
  packaging change requiring separate approval.
- **Responsibility-review candidates.** `backend/cuda/p2p/plan.cu`,
  `geometry/primitives/tetrahedron.cpp`, and `backend/cuda/fmm/plan.cu` are the
  largest units. Size locates audit work; it does not mandate splitting.
- **Obsolete test/doc/example audit and repository pruning**, and the future
  `tests/{unit,backend,integration}` and `benchmarks/{direct,p2p,far_field,fmm}`
  taxonomies.
- **Coverage gaps.** CI remains portable CPU only; oneMKL, CUDA, and Fortran
  are validated manually. Most CUDA-gated C++ cases return through `SUCCEED()`
  rather than a true `SKIP()`, so a portable-CPU run reports them as passed;
  only four cases report a real CTest skip. Making that distinction visible
  would make portable-run results easier to read literally.

A future `v0.2` release tag or branch is a separate, explicit release step. No
release ref was created by this closure.
