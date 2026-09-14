# Project structure and ownership

The current checkout has substantive homes for foundational layers,
operators/static plans, high-level FMM orchestration, portable CPU execution,
oneMKL execution, and CUDA common/direct/P2P/M2L/far-field execution. Full-FMM,
cache, and binding layers remain partly flat; the target taxonomy in
`docs/architecture.md` is not permission to create empty directories or perform
an unauthorised refactor.

## Current map

```text
include/cdfmm/
  cdfmm.hpp, uniform_fmm.hpp, static_*.hpp                    public and
                                                               compatibility APIs
  core/                                                        precision/output/timing
  math/                                                        vectors, indices,
                                                               derivatives, bases
  geometry/                                                    models/primitives
  tree/                                                        uniform/adaptive tree
  tree/static_topology.hpp                                    canonical topology data
  tree/uniform_topology.hpp                                   UniformTree adapter API
  operators/                                                   P2M/M2M/M2L/L2L/L2P/M2P/P2P interfaces
  plan/                                                        immutable static data, direct plans,
                                                               and P2P packings
  backend/cpu/                                                 portable static and dense-direct interfaces
  backend/cuda/{direct,dense_direct,p2p,m2l}.hpp               canonical CUDA direct/P2P/M2L interfaces
src/
  math/                                                        mathematical kernels
  geometry/primitives/                                         prism/tetrahedron
  tree/{common,uniform,adaptive}/                             hierarchy/topology
  operators.cpp                                                 thin flat compatibility wrappers only
  cuboid.cpp                                                    compatibility geometry/pair math
  operators/                                                    authoritative mathematical construction
                                                               and dynamic application
  plan/direct/dense.cpp                                         dense-direct preparation/dispatch
  plan/                                                         precision conversion and P2P packing builders
  backend/cpu/direct/dense.cpp                                  portable dense-direct application
  backend/cpu/{static_plan_apply,near_field}.cpp                 portable static-plan/list-1 application
  backend/mkl/direct/dense.cpp                                  oneMKL dense-direct application
  backend/mkl/m2l.{hpp,cpp}                                     opaque grouped M2L execution state
  backend/cuda/common/error.hpp                                 shared CUDA error helpers
  backend/cuda/common/runtime.{hpp,cu}                          CUDA runtime helpers
  backend/cuda/direct/{direct,dense}.cu                         CUDA direct execution
  backend/cuda/stub/direct.cpp                                  non-CUDA direct stubs
  backend/cuda/p2p/{internal.hpp,plan.cu}                       CUDA P2P backend
  backend/cuda/stub/p2p.cpp                                     non-CUDA P2P stubs
  backend/cuda/m2l/{internal.hpp,plan.cu}                       CUDA M2L backend
  backend/cuda/stub/m2l.cpp                                     non-CUDA M2L stubs
  backend/cuda/far_field/{internal.hpp,executor.cu}              CUDA far-field backend
  fmm/{uniform_fmm,far_field}.cpp                               lifecycle and pass orchestration
  fmm/uniform_fmm_internal.hpp                                  opaque backend owner adapters
  tree/common/static_topology.cpp                              canonical topology validation
  tree/uniform/static_topology_adapter.cpp                     UniformTree topology adapter
  periodic.cpp, cache.cpp, validation.cpp                       support boundaries
  cuda_fmm.cu, cuda_*_plan.hpp, cuda_fmm_stub.cpp                changing full-FMM
                                                                state, orchestration,
                                                                and stubs
  c_api.cpp                                                     C ABI adapter
python/bindings.cpp                                             pybind11 module
fortran/cdfmm_fortran.f90                                       ISO_C_BINDING wrapper
tools/cdfmm_precompute.cpp                                     universal-cache tool
tests/                                                          Catch2 C++ suite and
                                                                optional Fortran smoke test
python_tests/                                                   Python, benchmark, and
                                                                notebook contract tests
examples/                                                       C++/Fortran examples
examples/notebooks/                                             interactive comparisons
benchmarks/                                                     drivers and runners
docs/                                                           user/math/architecture/
                                                                validation/performance docs
```

`include/cdfmm/` is for deliberate downstream interfaces. Implementation-only
headers, CUDA RAII, kernels, and launch helpers stay under `src/`. `geometry`
owns physical integration elements; `tree` owns hierarchy and interaction
topology; `operators` owns mathematical maps; `plan` concepts own immutable
execution descriptions; `backend` concepts own resources and execution; and
`bindings` adapt the public API without reimplementing solver logic.

## Execution data flow

```text
source/target geometry
  -> UniformTree or shared AdaptiveTree topology
  -> canonical static topology and operators
  -> CPU/oneMKL/CUDA derived packing
  -> portable CPU static-plan application or CUDA P2P/direct backend
  -> changing moments -> P2M -> M2M -> M2L -> L2L -> L2P
                         \-> exact list-1 P2P
  -> near + far field -> original target order
```

`UniformTree` is complete and stores level-major flat nodes with Morton-sorted
source/target permutations and half-open occupied-leaf ranges. Free-space
`list1` is the clipped same-level 3x3x3 neighbourhood; `list2` is the
well-separated children-of-parent-list1 remainder. Periodic records add image
identity without replicating the central tree.

The operator/plan step makes the following homes concrete:

```text
include/cdfmm/operators/       mathematical operator interfaces, including M2P
src/operators/                  operator construction and dynamic application
include/cdfmm/plan/             canonical static data and derived representations
src/plan/                       FP32 conversion, direct preparation, and deterministic P2P builders
include/cdfmm/backend/cpu/      portable application interface
src/backend/cpu/                portable static and dense-direct application
src/fmm/                        UniformFmm lifecycle and far-field sequencing
src/backend/mkl/                oneMKL dense-direct and grouped M2L application
include/cdfmm/backend/cuda/     canonical CUDA direct/P2P/M2L public interfaces
src/backend/cuda/common/        shared internal CUDA error/runtime helpers
src/backend/cuda/direct/        CUDA point and dense direct execution
src/backend/cuda/stub/          non-CUDA direct stubs
src/backend/cuda/p2p/           complete CUDA P2P plans, kernels, and state
src/backend/cuda/m2l/           reusable CUDA M2L plans, kernels, and state
src/backend/cuda/stub/m2l.cpp   non-CUDA M2L stubs
```

Canonical P2P target rows remain authoritative. Compact/SoA, leaf, tensor
dictionary, signed/reduced dictionary, and BSR data are deterministic derived
packings. `StaticFmmTopology` owns topology-native P2P leaf interaction
records: occupied source/target ranges, leaf IDs, source shifts, periodic
image identities, and self-identity flags. It no longer includes or embeds
`StaticP2PLeafPair`. The FMM plan-building boundary converts these records
into the derived P2P leaf-packing representation. The canonical topology
boundary is separate from the UniformTree adapter, which remains a
transitional tree-to-topology construction seam.

The grouped oneMKL M2L executor is built once from the canonical plan. Its
stable transfer-class metadata and reusable gathered/translated buffers remain
private behind an opaque owner, while FMM orchestration adds the backend's
gather/multiply/scatter timings to the existing public timing record.

The CUDA P2P backend is built once from a canonical or derived static plan.
`CudaP2PPlan` is declared by `include/cdfmm/backend/cuda/p2p.hpp`, with the
legacy flat header forwarding to it. `src/backend/cuda/p2p/` owns the FP64 and
FP32 canonical, compact, leaf, signed tensor-dictionary, and BSR(3) executors,
their asynchronous lifecycle, persistent resources, and shared full-plan
primitives. The CUDA M2L backend is declared by
`include/cdfmm/backend/cuda/m2l.hpp`, with `src/cuda_m2l_plan.hpp` retained as
a forwarding compatibility shim. `src/backend/cuda/m2l/{internal.hpp,plan.cu}`
owns the reusable FP64/FP32 device representation, kernels, bounded scratch
policy, lifecycle, and statistics used by both standalone `CudaM2LPlan` and
`CudaFullPlan`. The far-field executor at
`src/backend/cuda/far_field/{internal.hpp,executor.cu}` owns immutable FP32/
FP64 P2M/L2P entries, coefficient degrees, M2M/L2L matrices/interactions/
metadata, uploads, lifecycle/statistics, and kernels. It has no public API,
stub, or new library. `cuda_fmm.cu` retains changing moments/coefficient/field
buffers, permutations, P2P and separate M2L executor wiring, streams/events/
timing, near/far overlap, combination/reordering, and D2H transfer.

## Validation and documentation areas

Tests are currently one Catch2 executable assembled by `tests/CMakeLists.txt`.
They cover foundational math, operators, trees/topology, geometries, static and
periodic FMM, precision/output, caches, CUDA, parameter selection, the C ABI,
and the optional Fortran smoke test. `python_tests/` checks bindings,
end-to-end FMM behaviour, precision, geometry, caches, benchmark runners, and
notebook contracts. `benchmarks/` measures setup separately from repeated
evaluation and records accuracy/traffic/timing fields. `docs/` is authoritative
for current support, mathematics, and measured performance; `agent_docs/`
records repository workflow memory and current state.
