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
"Phase 1 closure" below. **Phase 2 (post-Phase-1 cleanup) is also COMPLETE**:
internal-implementation duplication, the tree/topology boundary, the
cache/`UniformFmm` boundary, and the public/internal API, header ownership,
and packaging boundary have all been audited and closed; see "Phase 2
handoff" below. The performance-optimization phase is in progress: 3A GPU
evaluation and 3B CPU / oneMKL evaluation are complete
(`agent_docs/performance_optimization.md`), 3C construction / plan
preparation is next, then the 3D cross-backend review; repository pruning
follows.

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

### P2P execution invariant: geometry builds tensors, executors apply tensors

Since the Phase-3 P2P unification the boundary above is enforced rather than
merely intended:

```text
point / prism / tetrahedron geometry
    -> geometry-specific pair-tensor construction    (operators/p2p.cpp,
                                                     geometry/primitives/*)
    -> canonical tensors + topology + identity marker (StaticP2POperator,
                                                     StaticP2PLeafRecord)
    -> geometry-independent packings                 (plan/p2p/*)
    -> CPU / CUDA executors                          (backend/*/p2p)
```

- All nine source/target pairs build canonical tensors; prism/tetrahedron
  pairs share the tetrahedron pair's polyhedron surface formulation.
- Below the canonical operator no builder or executor reads a geometry
  record. The packings see tensor components, indices, dense leaf ranges,
  periodic image ordinals, and the per-pair `skip_for_identity` marker; that
  marker, not the geometry, decides whether an identity match omits a pair
  (point dipole) or applies a physical self tensor (finite body). The dense
  leaf block carries the marker so every derived packing obeys it.
- Periodic image records are ordinary stored tensors to every packing.
- `P2PExecutionPacking::PointGeometry` is the single deliberate exception, a
  fused point-geometry evaluation for point sources and point targets on the
  CPU and (since Phase 3B.5) on the CUDA backends; it is named as such and
  rejected for finite geometry.
- `UniformFmmOptions::p2p_packing` forces any packing a backend can execute,
  with precedence over the automatic policy; impossible combinations fail
  at construction with their representational reason.

The capability matrix and its reasons live in `static-p2p.md`; the enforcing
test is `tests/test_p2p_geometry_matrix.cpp` (every pair, every packing of
every available backend, both precisions, free-space and periodic, against
the FP64 `DenseDirectPlan` reference).

### Precomputed and procedural operator representations

The invariant above says where an operator is defined; it does not say that
an executor must hold it as stored data. Phase 3B.5 made the refinement
explicit:

```text
physical operator
    +--> precomputed representation   tensor / coefficient rows / compressed
    |                                 packing, built once at construction
    +--> procedural representation    the mathematically identical cheap
                                      point operator reconstructed from the
                                      positions during every evaluation
```

Precomputation is an execution choice, not a mathematical requirement. For
finite tiles (rectangular prisms, tetrahedra, future grains) it is the
production strategy on measured evidence, not by assumption: Phase 3B.5b
benchmarked every exact finite P2P, P2M and L2P operator in both
representations and reconstruction lost by 150x to 1,300,000x per field
update, amortising its construction within one to ten updates on the CPU
(`docs/static-p2p.md`, `agent_docs/performance_optimization.md`). For point
sources and point targets the operator is a closed formula or a short
recurrence of resident positions, and the executor may reconstruct it when
that was measured faster or when it removes substantial persistent memory at
equal speed. The procedural representations in the tree are the
position-based point P2P (`PointGeometry`, CPU and CUDA), the procedural
point P2M and the procedural point L2P (spherical basis; `src/math/
solid_harmonic_recurrence.hpp`, `src/operators/point_expansion_kernel.hpp`,
`src/backend/cpu/far_field/procedural.{hpp,cpp}`,
`src/backend/cuda/far_field/procedural.cuh`). The mathematics stays defined
once: the procedural kernels call the same formula or recurrence the stored
construction is validated against, and `tests/test_spherical_harmonics.cpp`
checks the recurrence against the polynomial basis at every compiled order.
Selection follows the measured policies in `docs/backends.md`, with explicit
overrides (`p2p_packing`, `point_expansion_execution`) taking precedence.

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
  `periodic.cpp` now includes the narrow `cdfmm/tree/indexing.hpp`, and the
  public-header/packaging cleanup (see below) narrowed
  `parameter_selection.hpp` the same way: it now includes only
  `cdfmm/backend/execution.hpp` and `cdfmm/math/vec3.hpp` instead of the
  complete `uniform_fmm.hpp`.
- the cache subsystem under `cache/` separates the file container and root
  policy, payload records, identity, and the universal/periodic and
  geometry-plan payloads. A later cache/plan-preparation boundary cleanup (see
  "Cache/plan-preparation boundary cleanup" below) removed the remaining
  `UniformFmm` entry points: every cache function is now a free function
  taking an explicit identity/payload record built from specific solver/plan
  types, and `src/fmm/execution_setup.cpp`/`plan_preparation.cpp` are the sole
  callers. `cache/format.cpp` still builds the derived compact P2P representation
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
  CUDA common/direct/P2P/M2L/far-field/FMM units. CUDA/oneMKL/OpenMP link
  requirements are marked `PUBLIC` on the internal static `cdfmm_core` (so its
  own in-tree consumers — tests, benchmarks, the Python module, the Fortran
  example, `cdfmm-precompute` — build correctly), but this does not leak
  downstream: `cdfmm_c`, the only installed target, links `cdfmm_core`
  `PRIVATE` and adds no `PUBLIC`/`INTERFACE` link libraries of its own, so
  none of `cdfmm_core`'s dependencies appear in `cdfmm_c`'s exported usage
  requirements. See "Public/internal API, header ownership, and packaging
  cleanup" below for the added package config and its validation.

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

### Cache/plan-preparation boundary cleanup

A later task audited the `UniformFmm`/cache-persistence boundary that Phase 1
and the internal-duplication cleanup had both left recorded as open: cache
entry points (`initialise_cache_keys`, `load_universal_cache`,
`write_universal_cache`, `load_geometry_cache`, `write_geometry_cache`) were
private `UniformFmm` member functions defined in `src/cache/*.cpp`, so
`src/cache/internal.hpp` included the complete `cdfmm/uniform_fmm.hpp` and
cache translation units had direct access to essentially all private solver
state. A dependency map (two read-only workers, one over
`src/cache/*`/`uniform_fmm.hpp`/`src/fmm/*`, one over the public/ABI/Python/test
compatibility surface) classified every field each of the five functions read
or wrote as cache identity, persisted payload, statistics, or unrelated
plan/lifecycle state, and confirmed none of the five were referenced outside
`src/cache/*.cpp` and `src/fmm/*.cpp`, so no public/ABI/Python compatibility
constraint bore on the redesign.

The chosen design is narrow free functions plus explicit records, not a
service object or a friend god-class. Every cache entry point is now a free
function in `cdfmm::detail::cache` taking one of three explicit records
declared in `internal.hpp`: `CacheIdentityInputs`/`CacheIdentity` (identity;
`keys.cpp`), `UniversalCacheIdentity`/`UniversalCachePayload` (the
depth-independent bank and periodic root; `universal.cpp`), and
`GeometryCacheIdentity`/`GeometryCachePayload` (the geometry-dependent plan;
`geometry.cpp`). Payload records hold references to already-existing
`UniformFmm` members (`m2m_operators_`, `m2l_plan_`, `p2m_plans_`, and so on)
rather than a second copy of solver state; identity records hold small
values (strings, enums, an `int`) plus references to the `UniformTree`/
`StaticFmmTopology`/geometry data the identity is computed from.
`src/fmm/execution_setup.cpp` (identity) and `src/fmm/plan_preparation.cpp`
(universal/geometry load and write) assemble these records from `UniformFmm`'s
private state, call the free functions, and copy results back; they remain
the sole callers and plan preparation remains the sole owner of the
cache-vs-build decision. No `UniformFmm` cache method survives: all five
private declarations were removed from `include/cdfmm/uniform_fmm.hpp`, and
the three trivial `universal_cache_key()`/`geometry_cache_key()`/
`periodic_cache_key()` public accessors (previously also defined in
`keys.cpp`) moved to `src/fmm/uniform_fmm.cpp`, which already owns
`UniformFmm` lifecycle/accessors.

Two narrow types moved out of `uniform_fmm.hpp` to make the removal possible
without an ABI-affecting layout change: `P2MPlan`/`FloatP2MPlan` (previously
private nested structs of `UniformFmm`, needed by the geometry payload) moved
to `include/cdfmm/plan/static_coefficient.hpp` as free-standing types
alongside `StaticCoefficientOperator`; `ExpansionBasis` (previously declared
directly in `uniform_fmm.hpp`, needed by `CacheDescriptor`) moved to
`include/cdfmm/core/precision.hpp` alongside `StaticPrecision`. Both were
already unreachable except through `UniformFmm`'s public API by value, so
neither move changes any public name's meaning; `uniform_fmm.hpp` still
transitively provides both through its existing includes.
`sizeof(UniformFmm)` is unchanged (verified: 7272 bytes before and after), and
no public C++, C ABI, Python, or Fortran surface changed.

`StaticPlanStatistics` continued to be mutated directly by cache free
functions (hits, byte counts, phase timings), matching prior behaviour: it is
already a narrow plan-layer type declared in `include/cdfmm/timings.hpp` with
no dependency on `UniformFmm`, so this is not a coupling concern, and
introducing an alternative returned-result type for every cache call would
have added complexity without narrowing anything. The persisted format, cache
keys, and cache-key hash inputs/order are byte-for-byte unchanged; so is the
direct FP32 decode path and the asymmetric FP32-only failure cleanup that
`cache/AGENTS.md` already documented. Cache misses/corruption remain silent
and non-fatal.

Validation: portable `dev` (fresh) CTest 198 total, 193 passed, 4 expected
skips, and the same pre-existing "static triangular translations match M2M
and L2L references" failure confirmed identical on the unmodified `03be467`
baseline (unrelated floating-point comparison, not touched by this task);
Python 137 passed/7 skipped. Focused `ctest -R 'cache' -E 'tetrahedron
pairs|tetrahedron dispatch'`: 11 total, 9 passed, 2 expected oneMKL/CUDA
skips. A standalone cross-version probe (public-API-only source, unchanged
between builds) wrote an 8-file FP64/FP32 free/periodic cache corpus with an
isolated `v0.1.0` build, then read it back at HEAD: identical universal/
periodic/geometry cache keys, cache hits on every scenario (including
periodic), zero bytes written, cache files byte-identical before/after
(`sha256sum`), and field results bit-identical. A before/after timing probe
(identical geometry/options, both built without LTO to permit direct
linking) showed no measurable warm-construction slowdown (~0.725 s vs
~0.726 s mean over 5 repeats on a ~4,500-point case, within run-to-run
noise). CUDA + oneMKL `notebooks` (fresh, RTX 5090 SM120, CUDA 13.3.73):
**198/198 CTest with zero skips** (including the triangular-translation case,
which is sensitive to build configuration and passes here, matching this
project's established pattern of zero-skip CUDA+oneMKL runs), and the full
focused cache selector 11/11, including the two CUDA/oneMKL-gated cases that
the portable build skips; Python 143 passed/1 skipped against the
`build-notebooks` module. `git diff --check` clean.

### Public/internal API, header ownership, and packaging cleanup

A later task audited the remaining public/internal C++ interface and
packaging seams that Phase 1 closure and the internal-duplication,
tree/topology, and cache/plan-preparation cleanups had all left open: which
flat `include/cdfmm/` headers still lacked a canonical subsystem home,
whether they should get one, the `parameter_selection.hpp -> uniform_fmm.hpp`
dependency, the ownership of the CUDA/oneMKL capability queries, the exposure
of `P2MPlan`/`FloatP2MPlan` and `ExpansionBasis` from the cache-boundary
cleanup, and whether dip-fmm should offer a downstream CMake package. Three
small read-only audits (header ownership, the CMake/link-interface graph, and
the `parameter_selection.hpp`/`uniform_fmm.hpp`/CUDA-availability dependency
chain) preceded any change; each substantive flat header was read in full,
not assumed, before a relocate-or-keep decision.

**`parameter_selection.hpp` narrowed.** The header's own signatures need only
`Vec3` and `ExecutionBackend`; the complete `UniformFmm`/`UniformFmmOptions`
API is needed only by `src/parameter_selection.cpp`, which constructs
`UniformFmm` objects to time candidate configurations. `ExecutionBackend`
moved from `uniform_fmm.hpp` to a new narrow `include/cdfmm/backend/execution.hpp`
(a backend-selection enum, naturally owned by `backend/` alongside the CPU/
oneMKL/CUDA execution interfaces it selects between); `parameter_selection.hpp`
now includes only `cdfmm/backend/execution.hpp` and `cdfmm/math/vec3.hpp`.
`src/parameter_selection.cpp` gained an explicit `#include "cdfmm/uniform_fmm.hpp"`,
since it is the translation unit that actually needs the complete API — it
previously received that header only incidentally, through its own header's
transitive include. `uniform_fmm.hpp` includes `backend/execution.hpp` and
keeps re-exporting `ExecutionBackend` under its own name, so `#include
<cdfmm/uniform_fmm.hpp>` followed by `ExecutionBackend` usage is unchanged.
The other backend/plan enums that were also declared in `uniform_fmm.hpp`
(`SphericalM2LBackend`, `StaticMatrixBackend`, `StaticOperatorExecutor`,
`P2PExecutionPacking`, `StaticExecutionPlan`) were deliberately left in place:
nothing audited in this task needs them independently of the complete solver
API, and moving a type nothing depends on narrowly would only add an include
indirection without narrowing any real dependency edge. All four consumers of
`parameter_selection.hpp` (`cdfmm.hpp`, `python/internal.hpp`,
`src/parameter_selection.cpp`, `tests/test_parameter_selection.cpp`) were
checked individually: the first two already include `cdfmm/uniform_fmm.hpp`
directly, the third gained the explicit include above, and the fourth uses
only `deterministic_target_sample`/`branch_balance_ratio`, neither of which
touches `Vec3`, `ExecutionBackend`, or `UniformFmm`.

**CUDA/oneMKL availability queries relocated.** `cuda_compiled`,
`cuda_available`, `cuda_direct_available`, `cuda_m2l_p2p_available`,
`cuda_m2l_available`, `cuda_full_available`, and `cuda_device_description`
moved from `uniform_fmm.hpp` to a new canonical
`include/cdfmm/backend/cuda/availability.hpp`, matching the existing pattern
of `cdfmm/backend/cuda/dense_direct.hpp` declaring
`cuda_dense_direct_available()` beside the API it reports on; `one_mkl_available`
moved the same way to a new `include/cdfmm/backend/mkl/availability.hpp`, the
first public header under `backend/mkl/`. `uniform_fmm.hpp` includes both and
keeps re-exporting every name transitively, so existing `#include
<cdfmm/uniform_fmm.hpp>` source compiles unchanged. This closes the one
dependency-audit exception Phase 1 closure recorded and repeated through the
"Deferred refactor inventory": `src/backend/cuda/fmm/internal.hpp` included
`cdfmm/uniform_fmm.hpp` solely so `backend/cuda/fmm/plan.cu` and
`backend/cuda/stub/fmm.cpp` could see the declarations of the three
availability functions they define (`cuda_m2l_available`,
`cuda_m2l_p2p_available`, `cuda_full_available`) — confirmed by inspection:
every other type that header/translation-unit pair uses
(`StaticOperatorEntry`, `StaticM2LPlan`, `FloatStaticM2LPlan`,
`StaticP2POperator`, `StaticP2PBsrPlan`, `StaticP2PSignedTensorDictionaryPlan`
and their FP32 counterparts, `CudaPlanStatistics`, `CudaEvaluationTimings`,
`Vec3`, `FloatVec3`) already arrives through `cdfmm/plan/static_plan.hpp`,
`cdfmm/timings.hpp`, and `cdfmm/math/potential_field.hpp`, which that header
already included directly. `internal.hpp` now includes
`cdfmm/backend/cuda/availability.hpp` instead of the complete solver header,
so the CUDA backend no longer depends on the top-level public API merely to
declare functions it implements; it depends only on the narrow backend-facing
header that owns those declarations. `cuda_direct_available` (defined in
`backend/cuda/direct/direct.cu`/`backend/cuda/stub/direct.cpp`) and
`one_mkl_available` (defined in `backend/mkl/m2l.cpp`) were already being
defined without their declaration in scope — legal in C++ for a free function,
if untidy — and were left as found: fixing that pre-existing decl-visibility
gap was not needed to resolve the one documented reverse-dependency edge and
was out of this task's evidenced scope.

**`tensor_dictionary.hpp` relocated.** `CanonicalTensor6`, `canonicalise_tensor6`,
`Tensor6BitKey`, `tensor6_bit_key`, `pack_tensor6_token`, `tensor6_token_id`,
and `tensor6_token_sign_mask` moved from the flat `cdfmm/tensor_dictionary.hpp`
to `include/cdfmm/plan/p2p/tensor_dictionary.hpp`, alongside the other P2P
packing headers (`dictionary.hpp`, `signed_dictionary.hpp`, `bsr.hpp`, ...)
that are their only consumers. The header has no dependency on any other
`cdfmm` header, so the move is a pure relocation with no cascading include
changes. `src/plan/precision.cpp`, `src/plan/p2p/dictionary_detail.hpp`,
`src/plan/p2p/signed_dictionary.cpp`, and `src/backend/cpu/p2p/dictionary.cpp`
(its four internal consumers) now include the canonical path directly;
`include/cdfmm/static_operators.hpp` keeps including the flat
`cdfmm/tensor_dictionary.hpp` path, because it is itself a compatibility
umbrella whose job is to reproduce the pre-v0.2 transitive surface, so a
façade including a façade is correct there, not an exception to fix.

**`timings.hpp`, `periodic.hpp`, and `validation.hpp` audited and kept flat.**
Each was read in full and found to have no single clearer canonical owner
than "flat public header", rather than being deferred only because no one had
looked yet:

- `timings.hpp` declares `EvaluationTimings` (`fmm`-owned: complete
  near/far/M2L-phase wall-clock breakdown), `StaticPlanStatistics`
  (`plan`-owned: one-time construction cost/storage), and
  `CudaPlanStatistics`/`CudaEvaluationTimings` (CUDA-backend-owned: device
  traffic/persistent-allocation diagnostics and device-stream phase timings).
  These are read together through one diagnostics surface
  (`UniformFmm::last_timings()`/`static_plan_statistics()`/
  `cuda_plan_statistics()`, and `fmm/diagnostics.cpp`'s summary), so splitting
  the file across `fmm/`, `plan/`, and `backend/cuda/` would scatter one
  cohesive observability concept across three directories for no consumer's
  benefit. This mirrors why `uniform_fmm.hpp` itself stays flat: a genuinely
  cross-cutting public surface does not need an owner among its lower layers.
- `periodic.hpp` declares periodic-cell configuration
  (`PeriodicCellOptions`/`PeriodicConvention`) and periodic-boundary
  mathematics (box wrapping, `build_periodic_list1`/`list2`,
  `periodic_laplace_derivatives_raw`). Its consumers are
  `include/cdfmm/operators/m2l.hpp` (periodic M2L construction) and
  `include/cdfmm/tree/uniform_topology.hpp` (periodic box-list topology) —
  the `operators` and `tree` layers, which the architecture's own layering
  diagram places as *siblings* feeding `plan`, never depending on each other.
  Giving `periodic.hpp` to either would create exactly the sibling-to-sibling
  edge the diagram is designed to avoid; it is legitimately shared
  lower-level content, symmetrically depended on, like `math`/`core` but not
  itself pure math (it also owns solver-facing configuration types embedded
  directly in `UniformFmmOptions`). Left flat, with the pre-existing
  `canonical headers -> cdfmm/periodic.hpp` edge documented as intentional
  rather than transitional.
- `validation.hpp` declares generic vector-error metrics
  (`norm`/`absolute_error`/`relative_error`/`compute_error_metrics`, pure
  `Vec3` math with no FMM concept) and `direct_p2p_reference`, an O(N^2) P2P
  reference wrapper explicitly documented as "not part of production FMM
  traversal logic" and intended for tests/examples. This is cross-cutting
  validation/diagnostic utility by design, not a subsystem's production
  interface; the header's own doc comment already said so. Left flat.

**`P2MPlan`/`FloatP2MPlan` and `ExpansionBasis` reviewed, kept in place.**
Both moves were made by the preceding cache-boundary cleanup so cache payload
records could name them without depending on `UniformFmm`; this task's brief
asked whether that placement should now be reconsidered rather than assumed
permanent. `P2MPlan`/`FloatP2MPlan` in `include/cdfmm/plan/static_coefficient.hpp`
are a minimal, self-contained specialisation ("leaf index plus
`StaticCoefficientOperator`") of the coefficient-map types already declared
immediately above them in the same file; they introduce no god-object, no
PIMPL, and no ABI concern, and are exactly the kind of "legitimate canonical
static-plan record" the brief anticipated as one acceptable outcome. Moving
them again, or hiding them behind a `detail` namespace, would add machinery to
solve a problem that does not exist: nothing about their current visibility
is unsafe or confusing, and `sizeof(UniformFmm)` is unaffected either way.
`ExpansionBasis` in `include/cdfmm/core/precision.hpp`, alongside
`StaticPrecision`, is a coherent pairing: both are small, `core`-level scalar
configuration enums with no dependency of their own, and `ExpansionBasis` is,
per the brief's own framing, already a genuine user-facing configuration
concept (`UniformFmmOptions::expansion_basis`) rather than an incidental
implementation detail — `core/precision.hpp` is a defensible, non-arbitrary
canonical home for it, not merely wherever the previous task happened to put
it. Both are kept unchanged.

**Public enum/value-type survey.** `SphericalM2LBackend`, `StaticMatrixBackend`,
`StaticOperatorExecutor`, `P2PExecutionPacking`, and `StaticExecutionPlan`
remain declared in `uniform_fmm.hpp`. Each was classified (backend
capability/plan representation, all read or set only through the complete
`UniformFmm` API and its options/diagnostics) and none has an external
consumer that would benefit from a narrower dependency the way
`parameter_selection.hpp` did for `ExecutionBackend`; moving them would not
reduce any dependency width in the current tree, only relocate declarations
for their own sake, which the brief explicitly warned against.

**Downstream CMake package added.** `cdfmm_c` — the shared library exposing
the stable, unconditional C ABI (`include/cdfmm/c_api.h`) — is the only
installed target; its private static implementation library, `cdfmm_core`,
resolves OpenMP, oneMKL, and CUDA (`CUDA::cudart`/`CUDA::cublas`/`CUDA::cusparse`)
as `PUBLIC` link dependencies of its own build, but `cdfmm_c` links
`cdfmm_core` `PRIVATE` and declares no `PUBLIC`/`INTERFACE` link libraries of
its own (only `PUBLIC` include directories), so none of `cdfmm_core`'s
dependencies appear in `cdfmm_c`'s exported usage requirements — confirmed by
inspecting the generated `cdfmm_c-targets.cmake`, whose imported target has
`INTERFACE_INCLUDE_DIRECTORIES` and nothing else. `cdfmm_core` itself is
never installed, so there is no static-linking downstream-consumer case to
support: the exported package describes exactly one, shared, library. The
project gained an explicit `VERSION 0.1.0` (matching the released
`pyproject.toml` Python package version; not a v0.2 marker). A new
`add_library(cdfmm::cdfmm_c ALIAS cdfmm_c)`, `install(EXPORT cdfmm_c-targets
...)`, a generated `cmake/cdfmmConfig.cmake.in` (via
`configure_package_config_file`), and a generated version file (via
`write_basic_package_version_file(... COMPATIBILITY SameMajorVersion)`) are
installed to `${CMAKE_INSTALL_LIBDIR}/cmake/cdfmm`, cleaned up by the existing
`cdfmm-clean-install` target. The config calls no `find_dependency(...)`: it
does not need to, since the exported target's interface genuinely has no
CUDA/oneMKL/OpenMP dependency to discover — the alternative of adding
conditional `find_dependency` calls "to be safe" was rejected as describing a
dependency that does not exist. A downstream consumer's runtime need for
`libcudart`/`libmkl*`/`libomp` (when `cdfmm_c` was built with those backends)
remains an ordinary transitive shared-library runtime requirement resolved by
the dynamic loader from `libcdfmm_c.so`'s own recorded `DT_NEEDED` entries,
exactly as it was before this task; nothing about that changed.

Validation: an isolated-prefix portable install (`CDFMM_ENABLE_CUDA=OFF`,
`CDFMM_ENABLE_MKL` unset) produced `lib/cmake/cdfmm/{cdfmmConfig.cmake,
cdfmmConfigVersion.cmake,cdfmm_c-targets.cmake,cdfmm_c-targets-release.cmake}`
alongside the existing headers/library; grepping the installed prefix for the
source and build tree paths found none. A standalone out-of-tree consumer
(`cmake_minimum_required(VERSION 3.20)`, `find_package(cdfmm CONFIG REQUIRED)`,
`target_link_libraries(consumer PRIVATE cdfmm::cdfmm_c)`) configured, built,
and linked against only that isolated prefix (`CMAKE_PREFIX_PATH` pointed at
it, nothing else), then ran a two-point-dipole evaluation through the C ABI
and reproduced the analytic field to a relative error of `1.7e-16` — using
only the installed prefix, matching the pre-existing "Installation and
downstream consumption" C-consumer evidence above but now driven entirely
through `find_package` rather than a manually supplied include path and
library name. (The dynamic loader initially resolved a stale, previously
installed `libcdfmm_c.so` from the active Conda environment's own `lib/`
directory ahead of the freshly built one, because the IntelLLVM host compiler
used in this environment embeds that directory in `DT_RPATH`; this is a
pre-existing environment artifact of that toolchain, not a consequence of the
new package config, confirmed by forcing resolution to the freshly built
library with `LD_PRELOAD` and observing its own build-configuration
diagnostics — `build.cuda_compiled: false`, matching that isolated build's
options — replace the stale library's.)

No solver algorithm, cache format, cache key, C ABI, Python API, or Fortran
interface changed. Portable `dev` (fresh build, incremental after the header
moves): CTest 197/198 passed, the one failure being the same pre-existing
"static triangular translations match M2M and L2L references" case (confirmed
present before this task's changes, per the baseline check below), 4 expected
skips; Python 137 passed/7 skipped against the just-built `build/` module. CUDA
+ oneMKL `notebooks` (fresh, RTX 5090 SM120, CUDA 13.3.73, oneMKL 2026.1;
built at `-j 2` after the default unbounded `-j` hit unrelated `nvcc`
front-end internal-compiler-errors on four `.cu` files under parallel load —
confirmed environmental, not a defect, by recompiling one of the four in
isolation with identical flags before retrying at lower parallelism):
**198/198 CTest with zero skips**, including the triangular-translation case,
matching this project's established zero-skip CUDA+oneMKL pattern; Python 143
passed/1 skipped against `build-notebooks`. One pinned Python test,
`python_tests/test_profiling_setup.py::test_notebook_preset_supports_gnu_mkl_without_intel_header_leakage`,
asserted the exact pre-`VERSION` `project(...)` text as an anchor for its real
assertion (`LANGUAGES C CXX` staying enabled); updated to the new literal. C
ABI: unchanged 14 `cdfmm_*` symbols. Python: unchanged 63 top-level names (34
classes, 29 functions). Standalone-compile and mixed/reverse-include-order
probes covered every header this task moved or added.

### Baseline check before this task

Before any production change, "static triangular translations match M2M and
L2L references" was run once against the unmodified `9bab9ee6` tree in the
portable `dev` configuration and failed with the same `REQUIRE` and printed
values as it did after this task's changes, confirming the failure predates
this task and is an unrelated floating-point comparison, not a regression it
introduced.

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
- At Phase 1 closure, `src/backend/cuda/fmm/internal.hpp` included
  `cdfmm/uniform_fmm.hpp` because the CUDA backend *implements* the public
  availability queries declared there (`cuda_m2l_p2p_available`,
  `cuda_m2l_available`). That was a supported public-API relationship, not an
  implementation-layer inversion, but it is now historical: the later
  public-header/packaging cleanup relocated those declarations (and the rest
  of the CUDA/oneMKL capability queries) to the canonical
  `cdfmm/backend/cuda/availability.hpp`/`cdfmm/backend/mkl/availability.hpp`,
  which `backend/cuda/fmm/internal.hpp` now includes instead of the complete
  solver header. See "Public/internal API, header ownership, and packaging
  cleanup" below.
- At Phase 1 closure, `src/cache/internal.hpp` included `cdfmm/uniform_fmm.hpp`
  as a sanctioned `cache -> already-defined solver data` edge. This is now
  historical, not current: the later "Cache/plan-preparation boundary
  cleanup" (below) removed the include entirely, so `src/cache/internal.hpp`
  no longer depends on `cdfmm/uniform_fmm.hpp` at all. This paragraph is kept
  as a record of what Phase 1 closure found at the time; do not read it as a
  statement about the current tree.

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

The Phase-2 cleanup list is now fully resolved. Every item below was either
closed by a narrow follow-up task after Phase 1 closure or, for public
headers with no single clear subsystem owner, deliberately and explicitly
kept flat rather than left merely unmigrated. Phase 2 as a whole is
**COMPLETE**.

Resolved, kept here only as a record: the canonical-to-compact P2P packing
rule and the duplicated factorial helper (internal-duplication cleanup); the
`StaticFmmTopology`/tree-to-plan seam (tree/topology boundary cleanup, see
above); cache entry points as `UniformFmm` members (cache/plan-preparation
boundary cleanup, see above — every cache function is now a free function
taking an explicit identity/payload record, and no `UniformFmm` cache method
remains); and the public/internal API, header ownership, and packaging
cleanup (see above):

- **Public-header homes.** `tensor_dictionary.hpp` moved to
  `plan/p2p/tensor_dictionary.hpp`. `ExecutionBackend` moved to
  `backend/execution.hpp`. The CUDA/oneMKL availability queries moved to
  `backend/cuda/availability.hpp`/`backend/mkl/availability.hpp`.
  `timings.hpp`, `periodic.hpp`, `validation.hpp`, and `uniform_fmm.hpp`
  itself were each audited and kept flat, with the specific reason recorded
  above for each — not because no one had looked, but because none has one
  single clearer subsystem owner. `src/periodic.cpp`, `src/parameter_selection.cpp`,
  and `src/validation.cpp` accordingly stay flat too: relocating an
  implementation file to a subsystem its own public header does not belong to
  would not clarify anything.
- **`parameter_selection.hpp` → `uniform_fmm.hpp` seam.** Resolved:
  `parameter_selection.hpp` now depends only on `backend/execution.hpp` and
  `math/vec3.hpp`.
- **API and internal simplification.** CUDA availability queries relocated;
  a minimal CMake package configuration (`find_package(cdfmm CONFIG)`,
  `cdfmm::cdfmm_c`) added; CUDA/oneMKL/OpenMP dependencies confirmed private
  to the never-installed `cdfmm_core` and absent from `cdfmm_c`'s exported
  interface, so there was no propagation to stop — the exported target
  already carried none of them.
- **Responsibility-review candidates.** `backend/cuda/p2p/plan.cu`,
  `geometry/primitives/tetrahedron.cpp`, and `backend/cuda/fmm/plan.cu` remain
  the largest units and were not touched by this cleanup. Size locates audit
  work; it does not mandate splitting, and this is not a Phase-2 item.
- **Coverage gaps.** CI remains portable CPU only; oneMKL, CUDA, and Fortran
  are validated manually. Most CUDA-gated C++ cases return through `SUCCEED()`
  rather than a true `SKIP()`, so a portable-CPU run reports them as passed;
  only four cases report a real CTest skip. Making that distinction visible
  would make portable-run results easier to read literally. Not a Phase-2
  item; left as a standing observation.

**Obsolete test/doc/example audit and repository pruning**, and the future
`tests/{unit,backend,integration}` and `benchmarks/{direct,p2p,far_field,fmm}`
taxonomies, remain explicitly out of scope and unstarted. They are **not**
the next planned work: the next phase is a dedicated performance-optimization
phase covering both construction/plan-preparation and evaluation/repeated
execution across the CPU, oneMKL, and CUDA paths (profiling, allocation,
memory traffic, cache construction, P2P/M2L/hierarchy passes, kernel
structure, occupancy, launch overhead, transfers, asynchronous overlap,
reuse, and representative end-to-end workloads), because benchmark/test
infrastructure should remain available while that work is done. Repository
pruning follows the optimization phase, not this cleanup. A future `v0.2`
release tag or branch is a separate, explicit step that only follows both;
none was created by this closure.
