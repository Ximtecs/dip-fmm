# Developer architecture

This document is the architectural contract for the `v0.2` refactor. It
describes the intended ownership and dependency boundaries; the source tree is
still in the pre-refactor, largely flat layout. The contract does not by itself
authorise moving or changing production code.

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

The current production layout is deliberately flat during Step 1:

```text
dip-fmm/
|-- include/cdfmm/    public headers, currently flat
|-- src/              CPU, CUDA, plans, cache, and orchestration, currently flat
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
|   |-- operators/
|   |-- plan/
|   |-- backend/
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
|   |-- operators/
|   |   |-- p2m/
|   |   |-- m2m/
|   |   |-- m2l/
|   |   |-- l2l/
|   |   |-- l2p/
|   |   `-- p2p/
|   |-- plan/
|   |   `-- p2p/
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

`generation/` will eventually own physical grain generation. `refinement/`
will eventually own prism and tetrahedron refinement. Geometry packing belongs
with plan construction or a geometry-to-plan adapter according to whether it
is canonical geometry data or an execution representation. None of packing,
generation, discretisation, or refinement is implemented in Step 1.

## Deferred refactor inventory

This inventory records pressure points; it does not authorise their repair in
the architecture-contract step.

The Step 1 audit found these concrete boundary violations in the current flat
layout:

- `static_topology.hpp` includes both `static_operators.hpp` and
  `uniform_tree.hpp`, and the topology stores P2P leaf-plan records. Tree data
  and execution-plan interaction data are therefore not physically separated.
- `adaptive_tree.hpp` returns `StaticFmmTopology` directly, coupling adaptive
  construction to the current static-plan representation.
- `cuboid.hpp` combines finite geometry operations with `DenseDirectPlan` and
  matrix-backend policy; geometry consequently owns plan/backend concepts.
- `static_operators.hpp/.cpp` combine operator mathematics, canonical plans,
  geometry-specific construction, every derived P2P packing, CPU execution,
  SIMD/device-callable helpers, and FP32 conversion.
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
lines), `uniform_fmm.cpp` (about 3,600), `static_operators.cpp` (about 3,000),
`cache.cpp` (about 1,800), `python/bindings.cpp` (about 1,500), and
`static_operators.hpp` (about 900). These measurements locate audit work; they
do not require mechanical splitting.

### Next: core, math, geometry, and tree

- Extract foundational types and mathematical machinery from the flat public
  and source layouts while preserving include compatibility.
- Separate point, rectangular-prism, cuboid compatibility, and tetrahedron
  geometry from operator and orchestration concerns.
- Split uniform and adaptive tree construction from static-plan adaptation;
  remove timing/solver ownership from tree interfaces where that boundary is
  currently blurred.
- Review `tetrahedron.cpp` and the public geometry headers for mixed geometry,
  quadrature, exact-interaction, and expansion responsibilities.

### Later: operators and plans

- Decompose `static_operators.hpp` and `static_operators.cpp` by mathematical
  operator and by canonical-plan construction responsibility.
- Separate canonical P2P tensors from each derived SoA, compact, dictionary,
  reduced/signed, leaf-packed, and BSR execution representation.
- Clarify ownership between static topology, geometry adapters, translation
  banks, precision-specific storage, and immutable execution plans.

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

Step 1 changes documentation and agent guidance only, so its runtime
implementation is identical by construction. Production source moves, API
redesign, algorithm changes, CUDA decomposition, packing, generation,
discretisation, and refinement are all deferred.
