# Implementation guidance

Inherit `../AGENTS.md`. This directory currently contains all implementation
layers in a flat layout. The target hierarchy in `../docs/architecture.md` is
future structure, not a description of directories that already exist.

## Current structure

```text
src/
|-- {taylor_jet,laplace_derivatives,
|   spherical_harmonics,operators}.cpp     mathematics/reference operators
|-- {cuboid,rectangular_prism,
|   tetrahedron}.cpp                       finite geometry/exact interactions
|-- {uniform_tree,adaptive_tree,
|   static_topology,periodic}.cpp          trees and topology
|-- static_operators.cpp                   operators, plans, packings, CPU work
|-- {near_field,far_field}.cpp              execution stages
|-- uniform_fmm.cpp                        construction and orchestration
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
|-- operators/{p2m,m2m,m2l,l2l,l2p,p2p}/
|-- plan/p2p/
|-- backend/{cpu,mkl,cuda/{common,direct,p2p,far_field,fmm}}/
|-- fmm/
|-- cache/
`-- bindings/
```

Do not create target directories before substantive code belongs in them.

## Boundaries and pressure points

- `cuda_fmm.cu`, `uniform_fmm.cpp`, `static_operators.cpp`, and `cache.cpp`
  are known decomposition candidates. Do not split them during Step 1.
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
