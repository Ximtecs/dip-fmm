# Implementation guidance

Inherit `../AGENTS.md`. Foundational math, geometry, tree, operator, and plan
implementations now have subsystem directories. High-level FMM orchestration
and CPU/oneMKL execution also have explicit homes. CUDA direct execution,
complete CUDA P2P execution, shared CUDA runtime/error infrastructure, and
far-field execution now have explicit homes, including the extracted M2L
backend; complete CUDA FMM orchestration now has an explicit backend home.

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
|-- fmm/construction.cpp                    geometry normalisation/construction
|-- fmm/plan_preparation.cpp                immutable plans, FP32 quantisation,
|                                          high-level cache calls
|-- fmm/execution_setup.cpp                 backend resolution/wiring and P2P policy
|-- fmm/evaluation.cpp                      complete near/far lifecycle and timing
|-- fmm/far_field.cpp                       hierarchy sequencing
|-- fmm/diagnostics.cpp                     exact summary formatting
|-- fmm/uniform_fmm.cpp                     lifecycle, accessors, inspection
|-- fmm/internal.hpp                        opaque backend owner declarations
|-- backend/cpu/direct/dense.cpp            portable dense-direct execution
|-- backend/cpu/p2p/{executor,dictionary,near_field}.cpp
|                                          portable P2P and list-1 execution
|-- backend/cpu/m2l/executor.cpp            portable prepared M2L execution
|-- backend/cpu/far_field/{executor,entries,translation}.{cpp,hpp}
|                                          portable far-field mechanics
|-- backend/mkl/{m2l,direct/dense}.cpp      oneMKL execution
|-- backend/cuda/common/error.hpp           shared CUDA error helpers
|-- backend/cuda/common/runtime.{hpp,cu}    CUDA runtime helpers
|-- backend/cuda/direct/{direct,dense}.cu   CUDA point/dense direct execution
|-- backend/cuda/stub/direct.cpp            non-CUDA direct stubs
|-- backend/cuda/p2p/{internal.hpp,plan.cu} CUDA P2P plans, kernels, and state
|-- backend/cuda/stub/p2p.cpp               non-CUDA P2P stubs
|-- backend/cuda/m2l/{internal.hpp,plan.cu} CUDA M2L plan, kernels, and state
|-- backend/cuda/stub/m2l.cpp               non-CUDA M2L stubs
|-- backend/cuda/far_field/{internal.hpp,executor.cu,entries.cuh,translation.cuh}
|                                          CUDA P2M/L2P entries and M2M/L2L
|                                          execution state and shared kernels
|-- backend/cuda/fmm/{internal.hpp,plan.cu} complete CUDA FMM orchestration
|-- backend/cuda/stub/fmm.cpp                 non-CUDA full-FMM stubs
|-- cache.cpp                              cache identity and persistence
|-- cuda_fmm_plan.hpp                      forwarding compatibility shim
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
|-- backend/{cpu,mkl,cuda/{common,direct,p2p,m2l,far_field,fmm}}/
|-- fmm/
|-- cache/
`-- bindings/
```

Do not create target directories before substantive code belongs in them.

## Boundaries and pressure points

- `cache.cpp` remains a known decomposition candidate. High-level
  `UniformFmm` responsibilities now have explicit homes under `fmm/`; CUDA
  direct, complete P2P, M2L, far-field, and full-FMM execution also have
  explicit backend homes. Do not simplify backend resource ownership without
  a later explicitly scoped step.
- Separate mathematical operator construction, canonical plans, derived
  execution packings, and backend execution in that order during later work.
- Tree code owns spatial hierarchy/topology, not CUDA execution or
  backend-oriented packing.
- Geometry owns integration elements and physical models, not dense-direct
  plans, oneMKL selection, or FMM orchestration.
- Cache code persists defined solver data; it must not define that data's
  mathematics.
- `UniformFmm` source ownership is split by lifecycle, construction, plan
  preparation, backend setup, evaluation, far-field sequencing, and
  diagnostics while delegating mathematical and backend mechanics downward.
- `fmm/far_field.cpp` owns CPU-side P2M/M2M/M2L-dispatch/L2L/L2P sequencing. Grouped
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
