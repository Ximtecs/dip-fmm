# Project progress

## Portable CPU backend decomposition — 2026-09-15

The accepted portable CPU execution split is complete. Prepared P2P
application, including canonical, compact, leaf, tensor-dictionary,
signed-dictionary/whole-tile, BSR, FP32/FP64, and SIMD paths, now lives under
`src/backend/cpu/p2p/` alongside near-field dispatch and reference-neighbour
evaluation. Prepared M2L application is under `src/backend/cpu/m2l/`. Shared
portable P2M/L2P entry mechanics and level-scaled M2M/L2L translation mechanics
are under `src/backend/cpu/far_field/{entries.hpp,translation.hpp}`, with
`executor.cpp` providing the narrow far-field execution seam. Dense direct
remains under `src/backend/cpu/direct/`.

`include/cdfmm/backend/cpu/{p2p,m2l,far_field}.hpp` are the authoritative
portable execution declarations. The former
`include/cdfmm/backend/cpu/static_plan_apply.hpp` remains a source-compatible
umbrella, while `src/backend/cpu/static_plan_apply.cpp` is removed. Reference
mathematics and high-level pass sequencing remain in `src/fmm/far_field.cpp`;
the final higher-level `UniformFmm` cleanup is the next architecture task.

Trusted validation for this closure:

- the initial CUDA hierarchy was structurally present; root
  `src/cuda_fmm.cu` and `src/cuda_fmm_stub.cpp` were absent, and CUDA 13.2 was
  available;
- fresh/full portable CPU CTest completed 193/193 with four expected optional
  skips; repaired focused CPU suites passed 13/13 and 34/34, and standalone
  canonical/legacy header probes passed;
- the oneMKL configure/core build and full CTest completed 193/193 with three
  expected CUDA skips, using environment-specific `-latomic` for linking;
- fresh CUDA SM75 configuration plus `cdfmm_core`/`cdfmm_tests` build passed
  198/198 steps, including rebuilt repaired CPU objects, and focused
  CUDA-configured coverage passed 13/13. No GPU numerical runtime result is
  claimed;
- vectorisation-report configure/build passed and generated a nonempty
  `cdfmm-cpu-backend.vec`; and
- `git diff --check` passed.

The accepted task is committed as `refactor(cpu): decompose portable backend`.
Unrelated untracked `Article1/` and
`examples/simple_notebooks/tetrahedron_target_average_fair_sampling.ipynb`
remain preserved outside the task.

## CUDA full-FMM backend ownership — 2026-09-14

The complete CUDA FMM implementation now has its structured backend home under
`src/backend/cuda/fmm/{internal.hpp,plan.cu}`. The former root implementation
is removed; `src/cuda_fmm_plan.hpp` remains a forwarding shim and the
non-CUDA full-FMM boundary is `src/backend/cuda/stub/fmm.cpp`. Far-field
shared mechanics are grouped in `entries.cuh` (P2M/L2P) and `translation.cuh`
(M2M/L2L), with `CudaTranslationInteraction` owned by the low-level
far-field internal header. CPU backend decomposition remains the next Phase 1
architecture task; this change does not move CPU files.

Validation handoff for this completed closure: the initial far-field sanity
and ownership audit was clean, with the focused CUDA-configured probe passing
5/5; a fresh CPU `dev` configure/build completed 55 targets and full CTest
completed 193/193 with expected optional skips #16, #51, #59, and #63. A
fresh CUDA 13.2 SM75 configure produced a successful 205-step build covering
`cdfmm_core`, `cdfmm_tests`, the Python extension, `cdfmm-precompute`,
`benchmark_uniform_fmm`, `benchmark_p2p`, and
`benchmark_cache_initialisation`. Full CUDA-configured CTest completed
193/193; runtime CUDA cases took their unavailable-device skips because
`nvidia-smi` could not communicate with the driver. The built Python module
imported successfully, and diff/ownership audits were clean. No GPU numerical
runtime result is claimed. Unrelated `Article1/` and notebook changes remain
preserved and outside this task.

## CUDA far-field execution extraction closure — 2026-09-14

The accepted CUDA far-field ownership split is complete. The internal
`src/backend/cuda/far_field/{internal.hpp,executor.cu}` package owns immutable
FP32/FP64 P2M and L2P entries, coefficient degrees, M2M/L2L matrices,
interactions and metadata, uploads/lifecycle/statistics, and the associated
kernels. It adds no public API, stub, or new library. `CudaFullPlan` retains
changing moments/coefficient/field buffers, permutations, P2P and separate M2L
executor wiring, streams/events/timing, near/far overlap,
combination/reordering, and D2H transfer. Direct, P2P, and M2L remain their
authoritative backends. This is not the end of the `CudaFullPlan`
decomposition: orchestration and resource simplification are the next scoped
task.

Validation truth for this closure:

- the initial M2L sanity ownership audit was clean and its focused 14-case
  probe passed;
- a fresh CPU development configure/build completed 55 steps, followed by
  full CTest 193/193 with expected skips #16, #51, #59, and #63;
- a fresh CUDA configure used `module cuda/13.2` (`nvcc 13.2.78`) for SM75;
  the preset presented a 206-step graph and compiled the requested
  `cdfmm_core`, tests, Python extension, `cdfmm-precompute`, and three benchmark
  targets, but failed at its final install step because the Conda
  site-packages directory was read-only in the sandbox; explicitly requested
  target builds then succeeded;
- focused CUDA CTest completed 30/30 with expected unavailable-device skips
  #16, #59, and #63, and full CUDA-configured CTest completed 193/193 with
  expected skips #16, #51, #59, and #63;
- `git diff --check` and the independent code/ownership review passed; and
- `nvidia-smi` could not communicate with the driver, so no numerical GPU
  runtime execution is claimed.

The untracked `Article1/` directory and unrelated notebook changes remain
untouched and outside this task.

## CUDA M2L backend extraction closure — 2026-09-14

The accepted CUDA M2L ownership split is complete. The canonical public
`CudaM2LPlan` declaration is `include/cdfmm/backend/cuda/m2l.hpp`, while
`src/cuda_m2l_plan.hpp` remains a forwarding compatibility shim. The reusable
FP64/FP32 execution representation, kernels, bounded scaled-multipole
scratch policy, persistent resources, statistics, and lifecycle now live in
`src/backend/cuda/m2l/{internal.hpp,plan.cu}`; the non-CUDA boundary is
`src/backend/cuda/stub/m2l.cpp`. Both standalone `CudaM2LPlan` and
`CudaFullPlan` consume the same internal executor. `src/cuda_fmm.cu` retains
P2M, M2M, L2L, L2P, and full-FMM/far-field orchestration plus backend wiring.

Trusted validation for this closure:

- fresh CPU configure/build completed 194/194;
- fresh CPU full CTest completed 193/193 with expected skips #16, #51, #59,
  and #63;
- fresh CUDA 13.2 SM75 build completed 204/204, covering core, tests, Python,
  tools, and benchmarks;
- focused CUDA M2L/header/stub coverage completed 23/23, with unavailable
  runtime-device cases skipped as expected;
- full CUDA CTest completed 193/193 with the same four expected skips;
- the install probe, complete diff review, ownership audits, and
  `git diff --check` passed; and
- `nvidia-smi` could not reach a driver/device, so no numerical GPU runtime
  result is claimed.

The unrelated untracked `Article1/` directory and
`examples/simple_notebooks/tetrahedron_target_average_comsol_compare_sampling_and_gauss.ipynb`
remain untouched and outside this task.

## CUDA P2P backend extraction closure — 2026-09-14

The complete CUDA P2P backend extraction was accepted as the preceding step
and is included in the current CUDA backend series. The canonical public declaration is
`include/cdfmm/backend/cuda/p2p.hpp`; `include/cdfmm/cuda_p2p.hpp` remains a
legacy forwarding façade. `src/backend/cuda/p2p/{internal.hpp,plan.cu}` owns
the P2P plan lifecycle, asynchronous evaluation, persistent device/host state,
and all list-1 execution variants: canonical AoS, compact/source-only SoA,
leaf-block, signed tensor dictionary (source-warp, target-owned, and
power-of-two microtiles), and cuSPARSE BSR(3), in both FP64 and FP32. It also
owns the shared full-plan primitives. Non-CUDA builds use
`src/backend/cuda/stub/p2p.cpp`. At that checkpoint, `src/cuda_fmm.cu`
retained M2L and far-field/full-FMM orchestration and consumed the P2P
internal primitives; the accepted M2L extraction is recorded above.

Trusted validation for this closure:

- fresh CPU full CTest completed 192/192 with four expected skips;
- fresh CUDA 13.2 SM75 configuration completed a 206-step build covering the
  core, tests, Python, examples, tools, and benchmarks;
- the CUDA-configured full CTest completed 192/192;
- runtime numerical GPU validation was unavailable because `nvidia-smi` could
  not communicate with the driver; and
- ownership, `nm`, and `git diff --check` audits were clean.

The untracked `Article1/` directory and unrelated
`examples/simple_notebooks/tetrahedron_target_average_comsol_compare.ipynb`
remain untouched and outside this task. Its focused validation was recorded
before the combined CUDA M2L closure.

## CUDA direct backend extraction closure — 2026-09-14

The accepted CUDA direct extraction gives point and dense direct execution
explicit backend homes without changing intended behaviour or performance:

- `include/cdfmm/backend/cuda/direct.hpp` and
  `include/cdfmm/backend/cuda/dense_direct.hpp` are the canonical public
  headers;
- `include/cdfmm/cuda_direct.hpp` and `include/cdfmm/cuda_cuboid.hpp` remain
  compatibility façades;
- `src/backend/cuda/common/` owns shared internal CUDA error/runtime helpers;
- `src/backend/cuda/direct/` owns point O(N^2) and dense cuBLAS direct
  execution; and
- `src/backend/cuda/stub/direct.cpp` owns non-CUDA direct stubs.

At that earlier checkpoint, `src/cuda_fmm.cu` retained P2P, M2L, and full-FMM
execution. No decomposition of those paths was performed at that checkpoint,
and no behaviour/performance change was intended.

Trusted validation for this closure:

- fresh development configure/build: 59 targets;
- focused direct/header/stub coverage: 8/8, including the exact disabled-stub
  behaviour case;
- full CPU CTest: 189/189, with expected skips #16, #51, #59, and #63;
- fresh CUDA 13.2 configure with explicit architecture 75 and cached Catch2
  source; and
- full CUDA build: 198 targets, including tests, Python, and benchmarks.

The independent full CUDA CTest run reported 189/189, but CUDA numerical/runtime
cases only exercised their unavailable-device guards because `/dev/nvidia` was
absent and `nvidia-smi` could not reach the driver. Ownership and `nm` audits,
and `git diff --check`, were clean. `Article1/` remains untouched.

## FMM, CPU, and oneMKL execution boundaries

High-level FMM implementation now lives under `src/fmm`: construction and
lifecycle remain in `uniform_fmm.cpp`, while `far_field.cpp` retains the
P2M/M2M/M2L-dispatch/L2L/L2P sequence and public timing integration. CPU
list-1 execution lives in `src/backend/cpu/near_field.cpp`; portable canonical
static-plan application remains in `src/backend/cpu/static_plan_apply.cpp`.

The oneMKL M2L executor is isolated in `src/backend/mkl/m2l.{hpp,cpp}`. It
derives stable transfer-class groups once from `StaticM2LPlan`, owns reusable
FP32/FP64 gathered and translated buffers behind an opaque `UniformFmm` owner,
and provides narrow apply, storage-statistics, and phase-timing interfaces.
All guarded vendor includes, MKL integer types, SGEMM/DGEMM calls, thread-local
MKL control, grouping, gather, multiplication, and serial scatter now live in
the backend. The installed header no longer exposes `M2LGroup` layouts.

Validation evidence for this boundary:

- fresh portable `dev` configure/build passed 52/52 build steps;
- focused near/far/static-plan/FP32/FP64 tests completed 21 cases: 20 passed
  and the oneMKL cache case was expectedly skipped;
- full portable CTest completed 184 cases: 180 passed and four unavailable
  oneMKL/CUDA cases skipped (#16, #51, #59, and #63);
- the existing CPU+oneMKL configuration reconfigured and rebuilt, and focused
  static-M2L/spherical/cache/precision tests completed 50 cases: 48 passed and
  two CUDA cases skipped (#16 and #59);
- full oneMKL CTest completed 184 cases: 181 passed and three CUDA cases
  skipped (#16, #59, and #63); and
- focused coverage verifies source/target level separation, stable mixed-level
  grouping, FP32 and FP64 execution, persistent storage statistics, timing
  calls, repeated application, and `UniformFmm` move behaviour.

Ownership searches, the old/new implementation comparison, symbol inspection,
and `git diff --check` confirmed the vendor mechanics are confined to
`src/backend/mkl`, the root-level FMM files remain absent, and Package 1 was a
move rather than a duplicate. CUDA runtime, Python, Fortran, and documentation
builds were not exercised; CUDA sources were not changed. `Article1/` and
unrelated parent-worktree changes remain untouched.

## Dense-direct plan/backend separation

The dense direct plan has an explicit plan-layer home:
`include/cdfmm/plan/direct/dense.hpp` is the canonical public declaration and
`src/plan/direct/dense.cpp` owns geometry validation, pair dispatch, tensor
construction, precision storage, backend selection, and tensor-memory
accounting. `include/cdfmm/cuboid.hpp` remains a compatibility umbrella and
`src/cuboid.cpp` retains cuboid monomial and pair tensor mathematics.

Portable execution and oneMKL execution now have dedicated internal homes at
`src/backend/cpu/direct/dense.cpp` and `src/backend/mkl/direct/dense.cpp`.
Reusable staging arrays live in a private shared dense-direct workspace owned
by the façade's pimpl. The public plan retains its constructor, evaluate API,
matrix accessors, backend enum, and deep-copy/value move semantics.

Validation evidence for this step:

- fresh portable `dev` configure/build passed 66/66;
- focused dense/direct/geometry/precision CTest passed 28/28;
- full portable CTest passed 183/183 with expected CUDA/optional skips #16,
  #51, #59, and #63;
- existing oneMKL configuration reconfigured and rebuilt successfully,
  focused direct/cuboid/oneMKL/precision tests passed 31/31, and full oneMKL
  CTest passed 183/183 with
  expected skips #16, #59, and #63; and
- `git diff --check` and ownership searches found no plan-side vendor or
  portable GEMV mechanics. CUDA runtime paths remain deferred by scope.

Validation for this source/include ownership move:

- fresh `dev` configure/build passed (64/64 build steps);
- focused geometry/direct/precision CTest passed 41/41;
- full portable CTest passed 182/182, with expected skips #16, #51, #59,
  and #63;
- a separate CPU+oneMKL configure/build passed, and its dense/cuboid focused
  CTest passed 25/25;
- canonical-only and legacy-header compile probes passed, including a safe
  temporary-prefix install check for `plan/direct/dense.hpp`; and
- `git diff --check` and authoritative declaration/source searches were clean.

CUDA runtime paths were not exercised in this CPU-only environment. The
oneMKL path was exercised from the existing environment; no dependency was
installed.

Status recorded 2026-09-10 after completing the shared tree root-box
extraction in the nested checkout. The implementation and documentation are
recorded in this focused change/commit.

## Shared tree root-box resolution

- `src/tree/common/root_box.{hpp,cpp}` is the internal shared boundary for
  resolving and validating common cubic roots used by both UniformTree and
  AdaptiveTree.
- AdaptiveTree no longer depends on or constructs UniformTree.
- Shared resolution retains an inferred coincident half-width of `0`; AdaptiveTree
  applies its existing fallback half-width of `1` at its call site. Their
  validation differences remain unchanged.
- The helper is not installed and introduces no public API.

Trusted validation evidence for this change:

- fresh development configure with
  `/home/mihaa/.conda/envs/cdfmm/bin/cmake` succeeded and the build passed
  48/48;
- focused UniformTree CTest passed 6/6;
- focused adaptive/integration C++ tests passed 5/5;
- `python_tests/test_adaptive_tree.py` passed 10 tests with `PYTHONPATH=build`;
- full CTest passed 182/182, with four expected CUDA/oneMKL optional skips;
- `git diff --check` passed, and dependency/source and symbol searches were
  clean; and
- the internal helper is absent from the install tree.

CUDA runtime validation and the full Python suite were not run and are not
claimed. `Article1/` and unrelated parent-worktree changes remain untouched.

## Static topology and UniformTree adapter closure

The accepted current change separates static topology from the uniform-tree
adapter:

- `include/cdfmm/tree/static_topology.hpp` is the canonical topology-only
  interface and `src/tree/common/static_topology.cpp` owns validation/storage
  behaviour;
- `include/cdfmm/tree/uniform_topology.hpp` declares the UniformTree adapter and
  `src/tree/uniform/static_topology_adapter.cpp` builds canonical topology from
  a UniformTree; and
- `include/cdfmm/static_topology.hpp` remains a compatibility forwarding
  header, while the removed `src/static_topology.cpp` is no longer a source
  home.

The function bodies were verified byte-identical apart from split context, so
the refactor preserves behaviour while clarifying ownership and dependencies.

Trusted validation evidence for this source/include-only change:

- development fresh configure/build passed;
- independent focused CTest passed 23/23, with one expected CUDA skip;
- full CTest reported 178 total, 174 passed, four expected skips (#16, #51,
  #59, and #63), and zero failures;
- standalone canonical-only and legacy-header compile probes passed; and
- `git diff --check` passed.

Python tests were not rerun because this change only moves source and include
ownership. No CUDA runtime result is claimed beyond the expected focused-test
skip.

## Repository and refactor state

- Branch: `refactor/architecture-v0.2`.
- The refactor branch has advanced from `a8565dc` through focused plan,
  operator, integration, and boundary-test commits (`4395f4a`, `d1dcfd7`,
  `14a94d6`, and `21cb5b1`).
- The immutable `v0.1.0` tag and `release/v0.1` branch remain the preserved
  pre-refactor baseline. The current branch follows the v0.2 architecture
  contract and has completed the foundational Step-2 organisation of core,
  math, geometry, and tree layers.
- The nested checkout has an untracked `Article1/` checkout/artifact. It is
  outside this documentation task and must remain untouched. The parent
  MagTense worktree also contains unrelated user changes; do not fold them
  into the nested refactor.

- The accepted dynamic P2M/M2M/M2L/L2L/L2P/P2P mathematics now lives in
  responsibility-specific operator sources. M2P has a narrow reference
  header/source, P2P's direct sum is namespaced, and `src/operators.cpp`
  contains only thin compatibility wrappers. Direct namespaced-vs-flat
  compatibility coverage is included in the foundational header test.
- The operator/plan history remains committed before this session; the current
  topology split and these memory updates are recorded in a focused commit.
  Preserve `Article1/` and all unrelated parent-worktree changes.

## Implemented foundation and operator/plan step

The root guidance, README, docs, headers, source, and tests describe a
production static `UniformFmm` plan with complete uniform and geometry-only
adaptive topology paths; Cartesian and real-spherical operators; exact direct
CPU/CUDA references; FP32/FP64 state; portable CPU, oneMKL M2L, hybrid CUDA,
and CUDA-full backends; persistent universal/geometry caches; point, prism,
and tetrahedron models; and a C ABI plus Python and optional Fortran bindings.
These are documented capabilities, not a claim that every optional build was
run in this session. The current refactor additionally gives mathematical
P2M/M2M/M2L/L2L/L2P/P2P construction explicit homes under
`include/cdfmm/operators/` and `src/operators/`; canonical static data,
precision conversion, and deterministic P2P packings explicit homes under
`include/cdfmm/plan/` and `src/plan/`; and portable static-plan application an
explicit `backend/cpu/` boundary. The legacy flat operator headers remain
compatibility umbrellas. `StaticFmmTopology` is now the canonical topology
boundary; the separate UniformTree adapter remains the transitional tree-
to-topology construction seam for a later FMM/topology step.

## Current strategic work and gaps

The roadmap identifies consolidation work rather than a new foundational
refactor: extend validation to representative micromagnetic workloads, improve
evidence-based parameter selection, continue Cartesian/spherical accuracy,
setup/runtime/memory measurements, and keep cache/backend documentation aligned
with implementation. The next integration milestone is an experimental
MagTense demagnetisation backend after spherical cuboid validation and field-
level comparison; runtime MagTense integration is not currently implemented.

Known capability boundaries include no partial periodicity or rectangular
periodic cells, CUDA tests not running in hosted GitHub Actions, and optional
oneMKL/CUDA/Fortran paths requiring their respective environments/toolchains.
The current branch's checked-in CI is portable CPU: configure a Release build
with tests and Python, run CTest, install the package, then run Python tests.

## Validation and remaining gaps

The accepted topology-split evidence is the configure/build, focused CTest,
full CTest, compile probes, body comparison, and diff-check results listed
above. Python tests were not rerun. A read-only artifact audit found no stray
temporary, reject, backup, secret, or task-artifact files outside `Article1/`;
that directory was not scanned or altered.

## Earlier session handoff — 2026-09-10

The shared orchestration setup was verified as six roles at concurrency six,
with routes/templates/bootstrap files and an original-config backup. Static
TOML/shell/diff checks passed; a fresh CLI recognized every role, the intended
Light/Medium/Heavy semantics, and absolute route paths. These were tooling
checks only: no project build or test ran because production code was unchanged.
