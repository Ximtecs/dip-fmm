# Project structure and ownership

**Phase 1 is COMPLETE**: this map describes the final Phase-1 architecture, and
it was verified against the tree during closure. Do not relocate anything here
without an explicit Phase-2 task.

The current checkout has substantive homes for foundational layers,
operators/static plans, high-level FMM orchestration, portable CPU execution,
oneMKL execution, the complete CUDA common/direct/P2P/M2L/far-field/FMM
backend hierarchy, cache identity/persistence, and the split Python/C binding
boundary. The target taxonomy in `docs/architecture.md` is not permission to
create empty directories or perform an unauthorised refactor.

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
  plan/p2p/tensor_dictionary.hpp                              Tensor6 canonicalisation/token packing
  backend/execution.hpp                                       ExecutionBackend (backend-selection enum)
  backend/cuda/availability.hpp,
  backend/mkl/availability.hpp                                CUDA/oneMKL capability queries
  backend/cpu/{p2p,m2l,far_field}.hpp                         portable static-plan interfaces
  backend/cuda/{direct,dense_direct,p2p,m2l}.hpp               canonical CUDA direct/P2P/M2L interfaces
src/
  math/                                                        mathematical kernels
  geometry/primitives/                                         prism/tetrahedron
  tree/{common,uniform,adaptive}/                             hierarchy/topology
  operators.cpp                                                 thin flat compatibility wrappers only
  operators/                                                    authoritative mathematical construction
                                                               and dynamic application
  plan/direct/dense.cpp                                         dense-direct preparation/dispatch
  plan/                                                         precision conversion and P2P packing builders
  backend/cpu/direct/dense.cpp                                  portable dense-direct application
  backend/cpu/p2p/{executor,dictionary,near_field}.{cpp,hpp}    portable stored-tensor P2P/list-1 application
  backend/cpu/p2p/geometry.{hpp,cpp}                            position-based point P2P executor
  backend/cpu/m2l/executor.cpp                                  portable per-target M2L application
  backend/cpu/m2l/schedule.{hpp,cpp}                            transfer-class-sorted M2L block schedule
  backend/cpu/far_field/{internal,executor}.{hpp,cpp}           far-field execution boundary
  backend/cpu/far_field/entries.hpp                             public entry-map reference kernels
  backend/cpu/far_field/packing.{hpp,cpp}                       dense level-scaled P2M/M2M/L2L/L2P packing
  operators/p2p_point_kernel.hpp                                the single point-dipole pair formula
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
  backend/cuda/far_field/{internal.hpp,executor.cu,entries.cuh,translation.cuh}
                                                                CUDA far-field backend
  backend/cuda/fmm/{internal.hpp,plan.cu}                       complete CUDA FMM
  fmm/construction.cpp                                          geometry normalisation/construction
  fmm/plan_preparation.cpp                                      immutable plans, FP32 quantisation,
                                                               high-level cache calls
  fmm/execution_setup.cpp                                       backend resolution/wiring and P2P policy
  fmm/evaluation.cpp                                            complete near/far lifecycle and timing
  fmm/far_field.cpp                                             hierarchy sequencing
  fmm/diagnostics.cpp                                           exact summary formatting
  fmm/uniform_fmm.cpp                                           lifecycle, accessors, inspection
  fmm/internal.hpp                                              opaque backend-owner declarations
  tree/common/static_topology.cpp                              canonical topology validation
  tree/uniform/static_topology_adapter.cpp                     UniformTree topology adapter
  periodic.cpp, validation.cpp                                  support boundaries
  cache/internal.hpp                                            shared cache interface
  cache/io.cpp                                                  root policy and file container
  cache/format.cpp                                              payload records
  cache/keys.cpp                                                cache identity and key strings
  cache/universal.cpp                                           translation-bank/periodic payloads
  cache/geometry.cpp                                            geometry-plan payload
  bindings/c_api.cpp                                           C ABI adapter
python/internal.hpp                                             shared binding declarations
python/module.cpp                                               sole pybind11 module entry point
python/{core,geometry,tree,operators,direct,fmm}.cpp             subsystem adapters
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
`bindings` adapt the public API without reimplementing solver logic. The C ABI
implementation is under `src/bindings/`; Python adaptation is split under
`python/`, with `module.cpp` as the sole entry point. The Fortran wrapper
continues to use `ISO_C_BINDING -> C ABI -> supported C++`.

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
include/cdfmm/backend/cpu/      canonical portable P2P/M2L/far-field interfaces
                                 plus static-plan compatibility umbrella
src/backend/cpu/direct/          portable dense-direct execution
src/backend/cpu/p2p/             canonical/packed P2P and near-field execution
src/backend/cpu/m2l/             portable prepared M2L execution
src/backend/cpu/far_field/       portable P2M/L2P entries and M2M/L2L translation
src/fmm/                        UniformFmm lifecycle and far-field sequencing
src/cache/                      cache identity, container, and payload persistence
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
type is separate from its two producers, the UniformTree adapter
(`build_uniform_fmm_topology`) and `AdaptiveTree`'s own construction; a
tree/topology-boundary audit confirmed both populate the same tree-owned
topology directly and neither is a tree-to-plan seam (`docs/architecture.md`,
"Tree/topology boundary cleanup").

The grouped oneMKL M2L executor is built once from the canonical plan. Its
stable transfer-class metadata and reusable gathered/translated buffers remain
private behind an opaque owner, while FMM orchestration adds the backend's
gather/multiply/scatter timings to the existing public timing record.

The high-level `UniformFmm` source is now responsibility-driven: construction
normalises geometry and builds the fixed tree/topology; plan preparation builds
immutable plans, quantises FP32 state, and performs high-level cache calls;
execution setup resolves backends and preserves P2P selection policy;
evaluation coordinates complete near/far evaluation and timing; far-field
retains hierarchy sequencing; diagnostics retains exact summary formatting;
and `uniform_fmm.cpp` retains lifecycle, accessors, and inspection. The
internal owner declarations are in `src/fmm/internal.hpp`.

The CUDA P2P backend is built once from a canonical or derived static plan.
`CudaP2PPlan` is declared by `include/cdfmm/backend/cuda/p2p.hpp`, with the
legacy flat header forwarding to it. `src/backend/cuda/p2p/` owns the FP64 and
FP32 canonical, compact, leaf, signed tensor-dictionary, and BSR(3) executors,
their asynchronous lifecycle, persistent resources, and shared full-plan
primitives. The CUDA M2L backend is declared by
`include/cdfmm/backend/cuda/m2l.hpp`, with the former internal `src/cuda_m2l_plan.hpp` shim removed and its consumers
pointed at the canonical header, formerly described as
a forwarding compatibility shim (now removed).
`src/backend/cuda/m2l/{internal.hpp,plan.cu}`
owns the reusable FP64/FP32 device representation, kernels, bounded scratch
policy, lifecycle, and statistics used by both standalone `CudaM2LPlan` and
`CudaFullPlan`. The far-field executor at
`src/backend/cuda/far_field/{internal.hpp,executor.cu,entries.cuh,translation.cuh}` owns immutable FP32/
FP64 P2M/L2P entries, coefficient degrees, M2M/L2L matrices/interactions/
metadata, uploads, lifecycle/statistics, and kernels. `entries.cuh` groups the
shared P2M/L2P entry application mechanics and `translation.cuh` groups shared
M2M/L2L translation mechanics. It has no public API, stub, or new library.
Complete CUDA FMM declarations and orchestration now live in
`src/backend/cuda/fmm/{internal.hpp,plan.cu}`, which retains changing moments/
coefficient/field buffers, permutations, P2P and separate M2L executor wiring,
streams/events/timing, near/far overlap, combination/reordering, and D2H
transfer. The former `src/cuda_fmm_plan.hpp` forwarding shim is removed.

The cache subsystem is responsibility-driven. `cache/io.cpp` owns the cache
root and `CDFMM_CACHE_DIR`/`CDFMM_DISABLE_CACHE` policy plus the validated file
container: header fields, payload checksum, memory-mapped reads, and the
unique-temporary-file plus `fsync` plus `rename` write that lets independent
processes race safely. `cache/format.cpp` owns the field-wise records for the
persisted solver types, including the direct FP32 decode paths.
`cache/keys.cpp` owns identity: the digest, the canonical 1e-9 coordinate, the
compact uniform-grid and permutation-layout recognition, and the key strings.
`cache/universal.cpp` and `cache/geometry.cpp` own the two payloads.
`cache/internal.hpp` is the implementation-only shared interface in the named
namespace `cdfmm::detail::cache`; nothing is installed and there is no public
cache API. The persistent format and the cache keys are compatibility
contracts, so `src/cache/AGENTS.md` governs changes there.
`fmm/plan_preparation.cpp` remains the coordinator that asks for cached data,
builds what is missing, and asks for a write. No file under `src/cache/`
defines a `UniformFmm` member function or includes `cdfmm/uniform_fmm.hpp`;
every cache entry point is a free function taking an explicit identity/payload
record (`CacheIdentityInputs`/`CacheIdentity`,
`UniversalCacheIdentity`/`UniversalCachePayload`,
`GeometryCacheIdentity`/`GeometryCachePayload`) built from specific
solver/plan/tree/topology types, assembled by
`src/fmm/execution_setup.cpp`/`plan_preparation.cpp`, which remain the sole
callers.

The CPU backend now follows the responsibility taxonomy
`backend/cpu/{direct,p2p,m2l,far_field}/`. P2P keeps all canonical and derived
representations plus near-field dispatch, M2L has its own prepared-plan
executor, and far-field owns shared P2M/L2P entry and M2M/L2L translation
mechanics. The old static-plan header remains an umbrella for source
compatibility. High-level `UniformFmm` ownership is split across the
responsibility-specific `src/fmm` units listed above; no implementation type is
exposed by the installed header.

## Validation and documentation areas

Tests are currently one Catch2 executable assembled by `tests/CMakeLists.txt`.
They cover foundational math, operators, trees/topology, geometries, static and
periodic FMM, precision/output, caches, CUDA, parameter selection, the C ABI,
and the optional Fortran smoke test. `tests/test_p2p_geometry_matrix.cpp` is
the table-driven near-field matrix (nine geometry pairs x backends x
precisions x P2P packings, free-space and periodic, against `DenseDirectPlan`)
that enforces the "geometry builds tensors, executors apply tensors"
invariant; `benchmarks/run_p2p_packing_matrix.py` drives
`benchmark_uniform_fmm` over the matching performance matrix (finite bodies,
forced packings, periodic cells, CudaPartial/CudaFull crossover).
`benchmarks/benchmark_operator_representation.cpp`, its CUDA procedural prism
kernel, `run_operator_representation.py` and
`analyse_operator_representation.py` compare precomputed against procedural
execution of one operator family, separating construction from repeated
application (Phase 3B.5b). `python_tests/` checks bindings,
end-to-end FMM behaviour, precision, geometry, caches, benchmark runners, and
notebook contracts. `benchmarks/` measures setup separately from repeated
evaluation and records accuracy/traffic/timing fields. `docs/` is authoritative
for current support, mathematics, and measured performance; `agent_docs/`
records repository workflow memory and current state.


## Compatibility surface — 2026-09-15

The flat root of `src/` holds no forwarding shims. `cuboid.cpp`,
`cuda_fmm_plan.hpp`, `cuda_m2l_plan.hpp`, `cuda_p2p_plan.hpp`, and the dead
`static_operators.cpp` are removed. The remaining flat `.cpp` files each own
one coherent responsibility: `operators.cpp` (thin compatibility delegation),
`periodic.cpp`, `parameter_selection.cpp`, and `validation.cpp`, plus the
internal `profile.hpp`.

Ownership of the former `cuboid.cpp` mathematics:

```text
rectangular_prism_averaged_monomial  src/geometry/primitives/rectangular_prism.cpp
cuboid_averaged_monomial             same file, flat spelling delegating to it
operators::p2p::build_pair           src/operators/p2p.cpp  (authoritative)
build_pair_tensor                    same file, flat spelling delegating to it
CuboidSize                           include/cdfmm/geometry/primitives/rectangular_prism.hpp
DenseDirectPlan                      include/cdfmm/plan/direct/dense.hpp
```

All 29 flat headers under `include/cdfmm/` were installed public paths at
`v0.1.0` and are retained as intentional compatibility façades. Canonical
structured headers and canonical implementation include canonical headers only;
the façades include those, never the reverse. The two deliberate exceptions are
`src/operators.cpp`, which implements `cdfmm/operators.hpp`, and
`python/internal.hpp`, which must see the flat operator declarations the Python
module exports.

## Audited dependency exceptions

Closure enumerated every reverse-direction edge that remains. Besides
`src/operators.cpp` and `python/internal.hpp` above, these are deliberate and
must not be "cleaned up" without an approved API change:

```text
src/plan/direct/dense.cpp -> backend/{cpu,mkl}/direct/dense.hpp
    DenseDirectPlan::evaluate() is pre-v0.2 public API, so the plan object
    itself dispatches to an executor. Confined to this one unit; the public
    plan/direct/dense.hpp header depends on no backend.

canonical headers -> cdfmm/timings.hpp, cdfmm/periodic.hpp
    Substantive public headers, audited and kept flat deliberately (no single
    subsystem owns either — see docs/architecture.md, "Public/internal API,
    header ownership, and packaging cleanup"), not façades and not deferred.
```

`src/backend/cuda/fmm/internal.hpp -> cdfmm/uniform_fmm.hpp` was resolved by
the public-header/packaging cleanup and no longer exists: the CUDA/oneMKL
availability queries the CUDA backend implements now have their own canonical
`cdfmm/backend/cuda/availability.hpp`/`cdfmm/backend/mkl/availability.hpp`
homes, which `internal.hpp` includes instead of the complete solver header.

`include/cdfmm/tree/adaptive_tree.hpp` returning `StaticFmmTopology` is not
a dependency-direction exception: both live in the `tree` layer, and a
tree/topology-boundary audit confirmed `StaticFmmTopology` carries no
plan/backend data (`docs/architecture.md`, "Tree/topology boundary
cleanup"). It was listed above as a deferred exception through Phase 1.

No prohibited edge exists: no geometry/tree depending on a backend or CUDA, no
math depending on FMM, no operators depending on orchestration or vendor
libraries, no plan depending on Python, no CPU backend reaching into CUDA
internals, and no binding unit including an internal `src/` header. Every
`.cpp`/`.cu` under `src/` is referenced by `CMakeLists.txt`.
