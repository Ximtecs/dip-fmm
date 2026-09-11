# Implementation guidance

Inherit `../AGENTS.md`. Foundational math, geometry, tree, operator, and plan
implementations now have subsystem directories. High-level FMM orchestration
and CPU/oneMKL execution also have explicit homes; cache, bindings, and CUDA
remain transitional.

## Current structure

```text
src/
|-- math/                                  Taylor, Laplace, spherical mathematics
|-- geometry/primitives/                   exact prism and tetrahedron analysis
|-- tree/{common,uniform,adaptive}/        spatial hierarchy and topology
|-- tree/common/static_topology.cpp        canonical topology validation/storage
|-- tree/uniform/static_topology_adapter.cpp UniformTree topology adapter
|-- operators.cpp                           flat compatibility wrappers only
|-- cuboid.cpp                              transitional geometry/direct code
|-- operators/                              authoritative operator construction
|                                          and dynamic application
|-- periodic.cpp                            periodic support code
|-- fmm/{uniform_fmm,far_field}.cpp         lifecycle and pass orchestration
|-- fmm/uniform_fmm_internal.hpp            opaque backend owner adapters
|-- backend/cpu/near_field.cpp              CPU list-1 execution
|-- backend/cpu/static_plan_apply.cpp       portable static-plan execution
|-- backend/mkl/{m2l,direct/dense}.cpp      oneMKL execution
|-- cache.cpp                              cache identity and persistence
|-- cuda_fmm.cu                            all CUDA implementation
|-- cuda_fmm_stub.cpp                      non-CUDA API stubs
|-- c_api.cpp                              C adapter
|-- parameter_selection.cpp, validation.cpp
`-- *.hpp                                  implementation-only support
```

Expected later destinations:

```text
src/
|-- math/
|-- geometry/{primitives,generation,refinement}/
|-- tree/{uniform,adaptive}/
|-- operators/{p2m,m2m,m2l,l2l,l2p,m2p,p2p}/
|-- plan/p2p/
|-- backend/{cpu,mkl,cuda/{common,direct,p2p,far_field,fmm}}/
|-- fmm/
|-- cache/
`-- bindings/
```

Do not create target directories before substantive code belongs in them.

## Boundaries and pressure points

- `cuda_fmm.cu`, `fmm/uniform_fmm.cpp`, and `cache.cpp` are known
  decomposition candidates. Do not split them without a later explicitly
  scoped step.
- Separate mathematical operator construction, canonical plans, derived
  execution packings, and backend execution in that order during later work.
- Tree code owns spatial hierarchy/topology, not CUDA execution or
  backend-oriented packing.
- Geometry owns integration elements and physical models, not dense-direct
  plans, oneMKL selection, or FMM orchestration.
- Cache code persists defined solver data; it must not define that data's
  mathematics.
- `UniformFmm` should eventually express lifecycle and orchestration while
  delegating construction and execution mechanics downward.
- `fmm/far_field.cpp` owns P2M/M2M/M2L-dispatch/L2L/L2P sequencing. Grouped
  gather/GEMM/scatter execution, persistent M2L scratch, and vendor includes
  belong under `backend/mkl`.

CUDA restructuring must preserve the event graph, near/far stream overlap,
persistent allocations and uploads, cuBLAS/cuSPARSE resource lifetime, launch
geometry, precision semantics, and intentional kernel fusion. File boundaries
do not require additional runtime kernels.

## Validation

Run the same applicable configuration before and after a production move.
Use the root portable build and tests at minimum; use `cuda`, `notebooks`, and
backend-focused tests/benchmarks for affected optional paths. A source-only
move still requires numerical regression tests because linkage, macros,
templates, and backend selection can alter behaviour.
