# Implementation guidance

Inherit `../AGENTS.md`. Foundational math, geometry, tree, operator, and plan
implementations now have subsystem directories. High-level FMM orchestration
and CPU/oneMKL execution also have explicit homes. CUDA direct execution,
complete CUDA P2P execution, shared CUDA runtime/error infrastructure, and
far-field execution now have explicit homes, including the extracted M2L
backend; complete CUDA FMM orchestration now has an explicit backend home.
Cache identity, the file container, payload records, and the universal/periodic
and geometry payloads now have explicit homes under `cache/`.

## Current structure

```text
src/
|-- math/                                  Taylor, Laplace, spherical mathematics
|-- math/solid_harmonic_recurrence.hpp     allocation-free host/device recurrence
|                                          for the real regular solid harmonics
|                                          and their gradients (procedural point
|                                          P2M/L2P on both targets)
|-- geometry/primitives/                   exact prism and tetrahedron analysis
|-- tree/{common,uniform,adaptive}/        spatial hierarchy and topology
|-- tree/common/static_topology.cpp        canonical topology validation/storage
|-- tree/uniform/static_topology_adapter.cpp UniformTree topology adapter
|-- operators.cpp                           flat compatibility wrappers only
|-- operators/                              authoritative operator construction
|                                          and dynamic application
|-- operators/p2p_point_kernel.hpp          the single point-dipole pair formula
|                                          shared by evaluate_pair and the CPU
|                                          and CUDA position-based executors
|-- geometry/primitives/rectangular_prism_point_kernel.hpp
|                                          precision-generic MagTense prism
|                                          point tensor; production evaluates
|                                          it in long double, the
|                                          representation benchmark in double,
|                                          float and on the device
|-- operators/point_expansion_kernel.hpp    procedural point P2M/L2P kernels and
|                                          their per-mode factor tables, shared
|                                          by the CPU and CUDA executors
|-- plan/precision.cpp                       FP64-to-FP32 static conversion
|-- plan/direct/dense.cpp                   dense-direct plan preparation
|-- plan/p2p/                               canonical and derived P2P packings
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
|                                          portable stored-tensor P2P and list-1 execution
|-- backend/cpu/p2p/geometry.{hpp,cpp}     position-based point P2P executor
|-- backend/cpu/m2l/schedule.{hpp,cpp}     transfer-class-sorted M2L block schedule
|-- backend/cpu/m2l/executor.cpp            portable prepared M2L execution
|-- backend/cpu/far_field/{executor,entries}.{cpp,hpp}
|                                          public entry-map reference kernels
|-- backend/cpu/far_field/packing.{hpp,cpp} derived CPU execution packing of
|                                          P2M/M2M/L2L/L2P (dense rows,
|                                          level-scaled column banks) built
|                                          once at construction
|-- backend/cpu/far_field/procedural.{hpp,cpp}, lanes.hpp
|                                          procedural point P2M/L2P executor
|                                          (SIMD packs of points through the
|                                          shared recurrence; no rows)
|-- backend/mkl/{m2l,direct/dense}.cpp      oneMKL execution
|-- backend/cuda/execution_policy.{hpp,cpp} deterministic CUDA strategy choices
|                                          (P2P packing/executor, M2L and
|                                          translation tuning) from plan facts
|-- backend/cuda/common/error.hpp           shared CUDA error helpers
|-- backend/cuda/common/runtime.{hpp,cu}    CUDA runtime helpers
|-- backend/cuda/direct/{direct,dense}.cu   CUDA point/dense direct execution
|-- backend/cuda/stub/direct.cpp            non-CUDA direct stubs
|-- backend/cuda/p2p/{internal.hpp,plan.cu} CUDA P2P plans, kernels (stored
|                                          packings and the position-based
|                                          point kernel), and state
|-- backend/cuda/stub/p2p.cpp               non-CUDA P2P stubs
|-- backend/cuda/m2l/{internal.hpp,plan.cu} CUDA M2L plan, kernels, and state
|-- backend/cuda/stub/m2l.cpp               non-CUDA M2L stubs
|-- backend/cuda/far_field/{internal.hpp,executor.cu,entries.cuh,translation.cuh,
|                           procedural.cuh}
|                                          CUDA P2M/L2P entries, procedural
|                                          point P2M/L2P kernels, and M2M/L2L
|                                          execution state and shared kernels
|-- backend/cuda/fmm/{internal.hpp,plan.cu} complete CUDA FMM orchestration
|-- backend/cuda/stub/fmm.cpp                 non-CUDA full-FMM stubs
|-- cache/internal.hpp                      shared cache constants, container
|                                          descriptors, and stream primitives
|-- cache/io.cpp                            cache root/environment policy and the
|                                          validated atomic file container
|-- cache/format.cpp                        payload records for persisted
|                                          solver types
|-- cache/keys.cpp                          cache identity and key strings
|-- cache/universal.cpp                     translation-bank and periodic-root
|                                          payloads
|-- cache/geometry.cpp                      geometry-plan payload
|-- bindings/c_api.cpp                     C adapter
|-- parameter_selection.cpp, validation.cpp
`-- profile.hpp                            internal NVTX range support
```

The flat root now contains no forwarding shims. `cuda_fmm_plan.hpp`,
`cuda_m2l_plan.hpp`, `cuda_p2p_plan.hpp`, `static_operators.cpp`, and
`cuboid.cpp` are removed; include the canonical structured header directly.
A file under `src/` is internal, so it carries no downstream source-compatibility
obligation: an internal forwarding header may be deleted once it has no users.
That is the opposite of `include/cdfmm/`, where the flat paths are supported
public façades and are retained.

The remaining flat `.cpp` files each own one coherent responsibility and are
kept deliberately: `operators.cpp` is thin compatibility delegation to
`operators/*` and backs the flat C++/Python operator names; `periodic.cpp`
owns periodic support; `parameter_selection.cpp` owns the advisory search; and
`validation.cpp` owns error metrics and the direct reference. Moving them is
Phase-2 work, not cleanup to fold into an unrelated step.

Expected later destinations. `plan/` and its `p2p/` packings, `backend/`, `fmm/`,
`cache/`, and `bindings/` are already implemented above; the remaining entries
are per-operator directories and the future geometry generation/refinement
homes:

```text
src/
|-- math/
|-- geometry/{primitives,generation,refinement}/
|-- tree/{uniform,adaptive}/
|-- operators/{p2m,m2m,m2l,l2l,l2p,m2p,p2p}/
|-- backend/{cpu,mkl,cuda/{common,direct,p2p,m2l,far_field,fmm}}/
|-- fmm/
|-- cache/
`-- bindings/
```

Do not create target directories before substantive code belongs in them.

## Boundaries and pressure points

- High-level `UniformFmm` responsibilities have explicit homes under `fmm/`;
  CUDA direct, complete P2P, M2L, far-field, and full-FMM execution have
  explicit backend homes; cache identity and persistence have explicit homes
  under `cache/`. Do not simplify backend resource ownership without a later
  explicitly scoped step.
- Separate mathematical operator construction, canonical plans, derived
  execution packings, and backend execution in that order during later work.
- Tree code owns spatial hierarchy/topology, not CUDA execution or
  backend-oriented packing.
- Geometry owns integration elements and physical models, not dense-direct
  plans, oneMKL selection, or FMM orchestration.
- Canonical implementation includes canonical structured headers. Do not add a
  new `#include "cdfmm/<flat>.hpp"` for a header that has a canonical
  subsystem home. The two deliberate exceptions are `src/operators.cpp`, which
  implements `cdfmm/operators.hpp`, and `python/internal.hpp`, which must see
  the flat operator declarations it exports to Python.
- Cache code persists defined solver data; it must not define that data's
  mathematics. Its persistent format and its keys are compatibility contracts;
  read `cache/AGENTS.md` before changing anything under `cache/`.
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
