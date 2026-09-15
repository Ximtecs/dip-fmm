# Project diary

## 2026-09-15 — bindings boundary and validation closure

At implementation base/newest starting HEAD `9003b666`, the bindings boundary
was structured without changing the public contract. The C ABI implementation
moved byte-identically to `src/bindings/c_api.cpp`; `include/cdfmm/c_api.h`
remains unchanged and ABI version 1 remains. Fortran continues through
`ISO_C_BINDING -> C ABI -> supported C++`. The former `python/bindings.cpp` was
replaced by `python/internal.hpp`, `module.cpp` as the sole entry point, and
the subsystem units `core.cpp`, `geometry.cpp`, `tree.cpp`, `operators.cpp`,
`direct.cpp`, and `fmm.cpp`. CMake registers these units. Binding code uses
supported canonical structured headers where available, retains only justified
compatibility dependencies, and contains no solver logic or internal backend
headers.

The isolated `9e111ee8` manifest comparison found 63 top-level Python names,
with object kinds, docs/defaults, and member surface identical after
normalising object addresses. The C ABI dynamic symbol set remained exactly
14 `cdfmm_*` symbols, and the C source move was byte-identical.

Trusted final evidence: portable affected/full build 73/73, CTest 197/197
with four expected optional skips, and Python 137 passed/7 skipped; oneMKL
notebooks core/C/Python/tests build, CTest 197/197 with three CUDA skips, and
Python 139 passed/5 skipped; RTX 5090 visible with driver 595.84 under full
access; fresh CUDA 13.3.73 build with explicit SM120, affected targets
107/107, CTest 197/197 with only the oneMKL-only skip, and Python 141 passed/
3 skipped. The initial stale SM75 artifacts failed PTX on the 5090 and are not
a final result. No gfortran, ifx, or ifort was available, so Fortran smoke and
example checks were not run. An independent temporary-prefix install passed,
and diff/ownership audits passed. `sphinx-build -W --keep-going -b html docs
docs/_build/html` rendered the documentation and exited nonzero only for the
same two pre-existing `docs/api.rst` C++ declaration warnings.

The cache change is already committed as `9e111ee`; it is not pending. The
remaining Phase 1 sequence is compatibility/transitional-source review,
followed by whole-Phase-1 validation. Neither is claimed complete here.

## 2026-09-15 — cache subsystem ownership decision

Accepted the cache decomposition by responsibility rather than by the smallest
possible file count. Two separate "format" concerns were deliberately kept
apart: the validated file container (magic, header fields, checksum, atomic
temporary-file write) stays with the persistence mechanics in `cache/io.cpp`,
because it cannot be separated from the read/write path, while `cache/format.cpp`
owns only the field-wise records for the solver types inside a payload. The
stream primitives (`Writer`, `Reader`, `CachePayload`, the unaligned load/store
templates) live in `cache/internal.hpp` because they are templates crossing
every cache translation unit, not for convenience.

The anonymous namespace that previously covered the whole file could not be
carried into a shared header: a type defined in an anonymous namespace in a
header is a distinct type per translation unit. The shared entities therefore
moved into the named internal namespace `cdfmm::detail::cache`, matching the
existing `detail::cpu`/`detail::mkl` convention, while entities with a single
consumer stayed in per-unit anonymous namespaces.

Three known layering warts were deliberately left in place because this step
was behaviour-preserving: `read_p2p_blocks` still populates the derived compact
plan in the same pass as decoding, the universal payload still encodes the M2L
bank layout arithmetic, and `load_geometry_cache` still validates plan
invariants and resets only FP32 state on failure. Each is recorded rather than
repaired. `UniformFmm` keeps its existing private cache members and object
layout; deeper encapsulation belongs to a later API/ABI step.

The decisive evidence was a before/after comparison against real cache files,
not the unit tests: the existing suite only compares keys between two live
instances, so a uniform key change would have passed it unnoticed. Final
validation recorded a portable fresh configure requiring an explicit `cdfmm`
environment PATH because literal `cmake` was unavailable, a 67/67 build, full
CTest 193/193 with skips #16/#51/#59/#63, and focused cache coverage 9/11 with
expected skips #51/#59. The oneMKL notebooks configuration used CUDA 13.2 and
oneMKL 2026.1.0 and passed its core/tests/Python build, full CTest 190 plus
three CUDA skips, focused 31 plus one CUDA skip, and Python 139 plus five
skips. The CUDA 13.2 SM75 core/tests/Python build passed 93/93; full CTest
passed 189 plus four skips, direct CUDA passed 16 plus three skips, and Python
import/smoke checks passed. Pytest was unavailable for matching Python 3.13,
and `nvidia-smi` could not reach the driver, so no GPU runtime is claimed.

Prior old-cache compatibility evidence is preserved and now backed by the
recovered 11-file corpus: both directions produced hits with zero writes and
identical manifests, the repository's 522-file corpus remained unchanged, and
an independent current probe hit old universal/periodic files without
modifying their hashes. No implementation defect was found. The cache task is
already committed as `9e111ee` with subject `refactor(cache): structure
persistence subsystem`; at that checkpoint, the bindings boundary,
compatibility/transitional review, and whole-Phase-1 validation remained.

## 2026-09-15 — high-level ownership audit

The independent final audit identified and verified the L2P ownership repair:
far-field sequencing now owns L2P evaluation, while evaluation retains the
whole near/far lifecycle, timing, and hybrid CUDA begin/CPU-far/finish
coordination. The re-review passed; no public API, cache/backend, binding, or
test changes were introduced.

## 2026-09-15 — CPU execution boundary decision

Accepted the portable CPU split by responsibility rather than by mirroring
CUDA file counts: P2P owns every prepared list-1 representation and near-field
dispatch, M2L owns prepared-plan application, and far-field groups only the
shared P2M/L2P entry and M2M/L2L translation mechanics. The legacy
`static_plan_apply.hpp` is intentionally retained as a compatibility umbrella;
reference mathematics stays in `src/fmm/far_field.cpp`, and no generic helper
bucket or public API redesign was introduced. The next architecture step is
the final higher-level `UniformFmm` cleanup.

## 2026-09-14 — CUDA far-field execution extraction closure

Accepted the CUDA far-field ownership split. The internal
`src/backend/cuda/far_field/{internal.hpp,executor.cu}` package owns immutable
FP32/FP64 P2M and L2P entries, coefficient degrees, M2M/L2L matrices,
interactions and metadata, uploads/lifecycle/statistics, and kernels. It adds
no public API, stub, or new library. `CudaFullPlan` retains changing
moments/coefficient/field buffers, permutations, P2P and separate M2L executor
wiring, streams/events/timing, near/far overlap, combination/reordering, and
D2H transfer. Direct, P2P, and M2L remain authoritative in their existing
backends. The next scoped task is orchestration/resource simplification; this
does not complete the `CudaFullPlan` decomposition.

Trusted evidence: the initial M2L sanity ownership audit was clean and its
focused 14-case probe passed; a fresh CPU development configure/build completed
55 steps and full CTest completed 193/193 with expected skips #16/#51/#59/#63;
fresh CUDA configuration used `module cuda/13.2` (`nvcc 13.2.78`) for SM75; the
preset presented a 206-step graph and compiled the requested `cdfmm_core`,
tests, Python extension, `cdfmm-precompute`, and three benchmark targets, but
failed at its final install step on sandbox read-only Conda site-packages;
explicitly requested target builds then succeeded; focused CUDA CTest completed
30/30 with
expected unavailable-device skips #16/#59/#63; full CUDA-configured CTest
completed 193/193 with expected skips #16/#51/#59/#63; and `git diff --check`
plus independent code/ownership review passed. `nvidia-smi` could not
communicate with a driver/device, so no numerical GPU runtime execution is
claimed. Unrelated `Article1/` and notebook changes were preserved.

## 2026-09-14 — CUDA M2L backend extraction closure

Accepted the reusable CUDA M2L ownership split. The canonical public
`CudaM2LPlan` declaration is now `include/cdfmm/backend/cuda/m2l.hpp`, with
`src/cuda_m2l_plan.hpp` retained as a forwarding compatibility shim.
`src/backend/cuda/m2l/{internal.hpp,plan.cu}` owns the shared FP64/FP32 device
representation, kernels, bounded scaled-multipole scratch policy, persistent
resources, lifecycle, and statistics used by standalone `CudaM2LPlan` and
`CudaFullPlan`; `src/backend/cuda/stub/m2l.cpp` owns non-CUDA stubs.
`src/cuda_fmm.cu` retains P2M/M2M/L2L/L2P and full-FMM/far-field orchestration
plus internal backend wiring.

Trusted validation: fresh CPU configure/build 194/194; fresh CPU full CTest
193/193 with expected skips #16/#51/#59/#63; fresh CUDA 13.2 SM75 build
204/204 covering core, tests, Python, tools, and benchmarks; focused CUDA
M2L/header/stub coverage 23/23 with unavailable runtime-device cases skipped
as expected; full CUDA CTest 193/193 with the same four skips; and install,
ownership, complete-diff, and `git diff --check` audits passed. `nvidia-smi`
could not reach a driver/device, so no numerical GPU runtime result is
claimed. The untracked `Article1/` directory and unrelated
`examples/simple_notebooks/tetrahedron_target_average_comsol_compare_sampling_and_gauss.ipynb`
were preserved untouched.

## 2026-09-14 — complete CUDA P2P backend extraction closure

Accepted the complete CUDA P2P ownership split as the preceding step. The canonical public
`CudaP2PPlan` declaration is now in
`include/cdfmm/backend/cuda/p2p.hpp`; the flat `include/cdfmm/cuda_p2p.hpp`
path remains a forwarding compatibility façade. `src/backend/cuda/p2p/`
owns canonical AoS, compact/source-only SoA, leaf-block, signed tensor
dictionary source-warp/target-owned/power-of-two microtile, and cuSPARSE
BSR(3) execution in FP64 and FP32, including lifecycle, asynchronous state,
persistent resources, and shared full-plan primitives. The non-CUDA stub is
`src/backend/cuda/stub/p2p.cpp`; the current `src/cuda_fmm.cu` retains
far-field/full-FMM orchestration and consumes the internal P2P and M2L
primitives.

Trusted validation: fresh CPU full CTest completed 192/192 with four expected
skips; fresh CUDA 13.2 SM75 completed a 206-step build covering core, tests,
Python, examples, tools, and benchmarks; CUDA-configured full CTest completed
192/192; runtime numerical GPU validation was unavailable because `nvidia-smi`
could not communicate with the driver; and ownership, `nm`, and
`git diff --check` audits were clean. The untracked `Article1/` directory and
unrelated `examples/simple_notebooks/tetrahedron_target_average_comsol_compare.ipynb`
were preserved untouched. Its focused validation was recorded before the
combined CUDA M2L closure.

## 2026-09-14 — CUDA direct backend extraction closure

Accepted the CUDA direct/common ownership split. Canonical public interfaces
are `include/cdfmm/backend/cuda/direct.hpp` and
`include/cdfmm/backend/cuda/dense_direct.hpp`; flat
`cuda_direct.hpp`/`cuda_cuboid.hpp` remain compatibility façades. Shared CUDA
error/runtime facilities live in `src/backend/cuda/common/`, point O(N^2) and
dense cuBLAS execution in `src/backend/cuda/direct/`, and non-CUDA direct
stubs in `src/backend/cuda/stub/direct.cpp`. At that earlier checkpoint,
`src/cuda_fmm.cu` retained P2P, M2L, and full-FMM execution. No behaviour or
performance change was intended.

Trusted validation: fresh dev configure/build completed 59 targets; focused
direct/header/stub coverage passed 8/8, including the exact disabled-stub
behaviour case; full CPU CTest passed 189/189 with expected skips
#16/#51/#59/#63. A fresh CUDA 13.2 configure with explicit architecture 75 and
cached Catch2 source succeeded, followed by a 198-target CUDA build including
tests, Python, and benchmarks. Independent full CUDA CTest reported 189/189,
but numerical/runtime
cases only reached unavailable-device guards because `/dev/nvidia` was absent
and `nvidia-smi` could not reach the driver. Ownership/`nm` audits and
`git diff --check` were clean. `Article1/` was untouched.

## 2026-09-11 — FMM, CPU, and oneMKL execution boundaries

Established explicit execution boundaries without changing the public FMM
API or pass sequencing. `src/fmm` owns lifecycle and the
P2M/M2M/M2L-dispatch/L2L/L2P chain, `src/backend/cpu` owns list-1 and portable
static-plan execution, and `src/backend/mkl` owns grouped oneMKL M2L execution.
An opaque internal owner now retains stable transfer grouping and reusable
FP32/FP64 gather/translation buffers; the canonical plan continues to own the
matrices, scaling, and mathematical interaction schedule.

Fresh portable and reconfigured oneMKL builds passed. Full portable CTest
completed 184 cases with 180 passes and four expected optional skips; full
oneMKL CTest completed 184 cases with 181 passes and three expected CUDA skips.
Focused tests cover cache sharing, spherical and Cartesian M2L, both
precisions, source-level scaling, sorting, buffer reuse statistics, timings,
and move behaviour. Ownership and diff audits passed. CUDA runtime and other
optional interface/documentation paths were not exercised, and no CUDA source
was changed. `Article1/` and parent-repository work remain untouched.

## 2026-09-11 — dense-direct plan/backend separation

Completed dense-direct execution ownership separation. The plan layer retains
geometry validation, deterministic pair-tensor construction, six target-major
matrices, precision storage, dispatch, and memory accounting. Portable row-major
nine-GEMV execution now lives in `src/backend/cpu/direct/dense.cpp`; guarded
oneMKL SGEMV/DGEMV execution and availability live in
`src/backend/mkl/direct/dense.cpp`. A private pimpl workspace retains reusable
FP32/FP64 staging arrays while explicit special members preserve deep-copy and
move value semantics. CUDA/FMM backend decomposition remains deferred.

Fresh portable and oneMKL builds passed after the move. Focused direct tests,
full portable CTest, and full oneMKL CTest passed with only the expected
CUDA/optional skips; repeated evaluation and copy/move regression coverage is
included. `Article1/` and unrelated parent-worktree changes remain untouched.

## 2026-09-11 — dense-direct plan ownership

Moved the authoritative `DenseDirectPlan` interface to
`include/cdfmm/plan/direct/dense.hpp` and its implementation to
`src/plan/direct/dense.cpp`. The legacy `cuboid.hpp` include remains supported
as a compatibility umbrella, while `src/cuboid.cpp` now retains only cuboid
monomial and pair-tensor mathematics. Portable and oneMKL GEMV execution stays
with the plan by design; dedicated backend extraction is the next dense-direct
step.

Fresh dev configure/build passed, focused geometry/direct/precision tests passed
41/41, and full portable CTest passed 182/182 with four expected optional
skips. A separate CPU+oneMKL build passed and its dense/cuboid focused tests
passed 25/25. Canonical and legacy header probes, installed-header presence,
diff checks, and symbol/source ownership searches passed. CUDA runtime was not
available and was not claimed; no oneMKL dependency was installed.

## 2026-09-10 — shared tree root-box resolution

Completed the internal root-box extraction for both tree modes. Shared
resolution and validation now live in `src/tree/common/root_box.{hpp,cpp}`;
AdaptiveTree no longer depends on or constructs UniformTree. Shared resolution
retains an inferred coincident half-width of `0`; AdaptiveTree applies its
existing fallback `1` at its call site, and their validation differences are
preserved. The helper is not installed and does not change the public API.

Fresh development configure/build passed 48/48; focused UniformTree tests
passed 6/6; focused adaptive/integration C++ tests passed 5/5; the adaptive
Python test passed 10/10 with `PYTHONPATH=build`; and full CTest passed
182/182 with four expected CUDA/oneMKL optional skips. Diff, dependency/source,
and symbol checks were clean. CUDA runtime validation and the full Python suite
were not run. The implementation is recorded in this focused change/commit.
`Article1/` and unrelated parent changes remain untouched.

## 2026-09-10 — static topology and UniformTree adapter closure

Accepted the source/include-only split between canonical static topology and
the UniformTree adapter. `StaticFmmTopology` declarations now live in
`include/cdfmm/tree/static_topology.hpp`, validation/storage in
`src/tree/common/static_topology.cpp`, and UniformTree construction in
`include/cdfmm/tree/uniform_topology.hpp` plus
`src/tree/uniform/static_topology_adapter.cpp`. The flat
`include/cdfmm/static_topology.hpp` remains a compatibility forwarding header;
the old `src/static_topology.cpp` source home is removed.

Function bodies were verified byte-identical apart from split context. Fresh
development configure/build, independent focused CTest (23/23 with one
expected CUDA skip), full CTest (178 total, 174 passed, four expected skips
#16/#51/#59/#63, zero failures), canonical-only and legacy-header compile
probes, and `git diff --check` all passed. Python tests were not rerun for this
source/include-only change. The accepted changes are recorded in a focused
topology-refactor commit; `Article1/` and unrelated parent-worktree changes are
preserved.

## 2026-09-10 — dynamic operator ownership closure

Completed the accepted mathematical operator ownership refactor. P2M, M2M,
M2L, L2L, L2P, and P2P construction/application now have responsibility-
specific homes under `src/operators/`; M2P is a narrow direct multipole
reference operator; P2P direct summation is namespaced; and the flat
`src/operators.cpp` translation unit is compatibility wrappers only. The
direct namespaced-vs-flat test covers all dynamic operators and output/self
semantics.

Clean dev configure/build and independent audits are green: CTest 178/178 with
expected skips #16/#51/#59/#63, Python 137 passed and 7 skipped via
`PYTHONPATH=build`, targeted operators 64/64, and Python operator bindings
10/10. Symbol/`rg` checks found only wrapper-to-namespaced linkage and no
forbidden operator dependencies. CUDA was not available because `nvidia-smi`
could not access an NVIDIA driver, so CUDA runtime equivalence remains
unverified. The final focused closure commit contains only task files;
the comparison notebook and `Article1/` remain untouched.

## 2026-09-10 — operators and static plans refactor

The committed architecture-v0.2 series now has explicit operator homes for
P2M, M2M, M2L, L2L, L2P, and P2P construction; explicit plan homes for
canonical static data, FP32 conversion, translation/L2P records, and derived
P2P packings; and a portable CPU static-plan application boundary. The legacy
flat headers remain compatibility umbrellas. The tree-to-plan adaptation in
`StaticFmmTopology` is intentionally still transitional, while CUDA/backend
decomposition and FMM orchestration are deferred.

The earlier dev build and CTest run were green: 177/177 tests passed with
expected unavailable-feature skips. Python regressions passed 137 tests with 7
skips; public/header, install-tree, and vectorisation checks also passed.

The implementation and boundary tests were recorded as four focused commits:
`4395f4a`, `d1dcfd7`, `14a94d6`, and `21cb5b1`. The modified comparison
notebook and untracked `Article1/` research checkout/artifacts were pre-existing
and preserved. No stray temporary or reject files were found; ignored
build/cache artifacts remain local.

## 2026-09-10 — nested repository memory bootstrap

Established project-specific memory for the separate `dip-fmm` Git checkout.
The durable decisions are to preserve the C++20 static-plan architecture, keep
canonical mathematical operators distinct from backend-derived packings, treat
the math/precision/self-identity/periodic conventions as normative, and keep
the v0.1.0 baseline and `release/v0.1` immutable. The current v0.2 branch has
completed the foundational core/math/geometry/tree organisation; higher-layer
refactoring remains explicitly staged.

Open questions are intentionally evidence-bound: optional CUDA/oneMKL/Fortran
availability depends on the active environment, hosted CI does not validate
CUDA, and representative MagTense field/simulation validation remains on the
roadmap. The pre-existing notebook modification and untracked `Article1/` were
observed and preserved. This bootstrap did not run validation or alter
production code.

## 2026-09-10 — orchestration verification

Verified the six-role, concurrency-six global orchestration installation,
including routes/templates/bootstrap files and the original-config backup.
Static Tester TOML/shell/diff checks passed, and the fresh CLI resolved all
roles, Light/Medium/Heavy semantics, and absolute route paths. `codex exec`
0.153.4 JSON delegation smoke attempts remain limited by empty-receiver wait
calls with no observable spawn event, while the app session ran Explorer,
Researcher, two Luna executors, and Tester. No project validation or commit was
performed; both worktrees contain unrelated user changes.
