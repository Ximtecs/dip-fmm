# Project structure and ownership

The current checkout has substantive homes for foundational layers,
operators/static plans, high-level FMM orchestration, portable CPU execution,
and oneMKL dense-direct and M2L execution. CUDA, cache, and binding layers
remain partly flat; the target taxonomy in `docs/architecture.md` is not
permission to create empty directories or perform an unauthorised refactor.

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
  fmm/{uniform_fmm,far_field}.cpp                               lifecycle and pass orchestration
  fmm/uniform_fmm_internal.hpp                                  opaque backend owner adapters
  tree/common/static_topology.cpp                              canonical topology validation
  tree/uniform/static_topology_adapter.cpp                     UniformTree topology adapter
  periodic.cpp, cache.cpp, validation.cpp                       support boundaries
  cuda_fmm.cu, cuda_*_plan.hpp, cuda_fmm_stub.cpp                CUDA implementation/stub
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
  -> portable CPU static-plan application or existing optional backend path
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
