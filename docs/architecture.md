# Developer architecture

This document is the architectural contract for the `v0.2` refactor. It
describes the ownership and dependency boundaries now implemented in the
repository. Step 2 made the foundational `core`, `math`, `geometry`, and
`tree` layout concrete; the operators, static plans, and portable CPU apply
boundary are now also structured. FMM orchestration, cache, and most backend
execution remain transitional layers.

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

The current production layout after the operator/plan step is:

```text
dip-fmm/
|-- include/cdfmm/    canonical core/math/geometry/tree/operators/plan/backend
|                    plus compatibility shims
|-- src/              structured operators/plan/backend/cpu plus transitional
|                    FMM, topology, and support layers
|-- tests/            C++ and optional Fortran tests
|-- python_tests/     Python and notebook regression tests
|-- benchmarks/       C++ drivers and Python runners
|-- examples/         C++, Fortran, scripts, and notebooks
|-- python/           pybind11 bindings
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
|   |-- backend/         # CPU static-plan apply plus transitional CUDA APIs
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
|   |       |-- p2p/
|   |       |-- far_field/
|   |       `-- fmm/
|   |-- fmm/
|   |-- cache/
|   `-- bindings/
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
remain valid until the later bindings/build step deliberately reorganises
them.

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
|   `-- p2p/
|       |-- canonical.hpp       authoritative target-row P2P data
|       |-- compact.hpp         derived particle-row SoA packing
|       |-- leaf.hpp            derived dense leaf packing
|       |-- dictionary.hpp      derived Tensor6 dictionary packing
|       |-- signed_dictionary.hpp derived signed/reduced packing
|       `-- bsr.hpp              backend-neutral BSR(3) packing
`-- backend/cpu/static_plan_apply.hpp  CPU application boundary

src/
|-- operators/{p2m,m2m,m2l,l2l,l2p,m2p,p2p}.cpp  authoritative construction
|                                                  and dynamic application
|-- plan/{precision,p2p}                     conversion and packing builders
`-- backend/cpu/static_plan_apply.cpp        CPU plan application
```

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
- `cuboid.hpp` combines finite geometry operations with `DenseDirectPlan` and
  matrix-backend policy; geometry consequently owns plan/backend concepts.
- compatibility `static_operators.hpp/.cpp` remain as forwarding umbrellas;
  their former mixed implementation has been separated into operators, plans,
  and the portable CPU apply boundary described below.
- `uniform_fmm.hpp/.cpp` combine public policy, geometry normalisation,
  topology and operator construction, cache use, backend selection, packing,
  mutable state, CUDA upload, evaluation, and inspection.
- `far_field.cpp` contains CPU stages while directly including oneMKL and CUDA
  execution internals. `periodic.cpp` includes `uniform_tree.hpp`, and
  `parameter_selection.hpp` includes the complete `uniform_fmm.hpp` API.
- `cache.cpp` implements low-level persistence, identities and validation as
  methods of `UniformFmm`, giving cache mechanics intimate ownership knowledge
  of all topology/operator representations.
- public CUDA headers expose plan concepts and depend on broad operator or
  geometry headers; `timings.hpp` also includes CUDA-specific plan statistics.
- `python/bindings.cpp` adapts nearly every layer in one translation unit, and
  the C API implementation depends directly on the high-level FMM type.
- the root CMake target registers all CPU subsystems together and adds one
  5,000-line CUDA translation unit; CUDA libraries are currently propagated
  from `cdfmm_core` to consumers.

The largest responsibility-review candidates are `cuda_fmm.cu` (about 5,200
lines), `uniform_fmm.cpp` (about 3,600), `cache.cpp` (about 1,800), and
`python/bindings.cpp` (about 1,500). These measurements locate remaining audit
work; they do not require mechanical splitting.

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
- `StaticFmmTopology`, dense-direct policy, pair dispatch, and adaptive plan
  adaptation remain transitional seams for the operator/plan/backend phases.

### Implemented: operators and static plans

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
- Portable CPU application is isolated at
  `backend/cpu/static_plan_apply.cpp`. It consumes canonical or derived plans;
  it does not redefine pair or translation mathematics.

The remaining transitional seam is `StaticFmmTopology`: it adapts tree
interaction topology to topology-native records consumed by FMM orchestration
and the P2P plan boundary. Tree ownership remains spatial
hierarchy/topology, while derived schedule assembly stays behind the plan
boundary without making the tree depend on a particular P2P packing.

### Later: backend and CUDA

- Decompose `cuda_fmm.cu` by common infrastructure, direct evaluation, P2P,
  far-field stages, and full-FMM execution without splitting fused runtime
  work merely along source-file boundaries.
- Isolate persistent CUDA plans, buffers, streams/events, and cuBLAS/cuSPARSE
  resources from public mathematical interfaces.
- Move oneMKL execution mechanics out of `uniform_fmm.cpp`, `far_field.cpp`,
  and geometry-related implementation files into the oneMKL backend.
- Preserve and benchmark all transfer, launch, overlap, and reuse semantics.

### Later: FMM, cache, bindings, and build

- Reduce `uniform_fmm.cpp` to high-level lifecycle and orchestration after its
  lower layers have stable homes.
- Decompose `cache.cpp` into keys, metadata/validation, and serialisation while
  keeping cache formats and invalidation behaviour stable.
- Keep Python, C, and Fortran layers as adapters; split the large pybind11
  translation unit by exposed subsystem without duplicating solver logic.
- Rework CMake source grouping only as files actually move, then align tests
  and benchmarks with the resulting subsystem tree.

## Refactor validation contract

Each production-code step starts and ends with the same relevant build and
test configurations. Numerical tests must compare unchanged formulas,
precision modes, output flags, geometries, tree modes, periodic behaviour,
cache behaviour, and supported CPU/oneMKL/CUDA paths. Performance-sensitive
backend changes additionally compare plan reuse, transfers, allocations,
launches, synchronisation, and representative benchmark results.

Step 2 changes file ownership and include structure without changing runtime
algorithms or public names. API redesign, CUDA decomposition, packing,
generation, discretisation, and refinement remain deferred.
