# Latest session work

## 2026-09-14 — CUDA full-FMM backend ownership

Moved complete CUDA FMM declarations and orchestration to
`src/backend/cuda/fmm/{internal.hpp,plan.cu}` and moved non-CUDA full-FMM stubs
to `src/backend/cuda/stub/fmm.cpp`. The root `cuda_fmm.cu` and
`cuda_fmm_stub.cpp` implementation homes are removed; `cuda_fmm_plan.hpp` is
now a forwarding shim. Far-field shared P2M/L2P mechanics live in
`entries.cuh`, M2M/L2L mechanics in `translation.cuh`, and the far-field
internal header owns `CudaTranslationInteraction` without depending on the
full-FMM header. CPU decomposition remains the next backend task.

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

## 2026-09-14 — CUDA far-field execution extraction closure

The accepted CUDA far-field execution extraction is complete. Internal
`src/backend/cuda/far_field/{internal.hpp,executor.cu}` owns immutable FP32/
FP64 P2M/L2P entries, coefficient degrees, M2M/L2L matrices/interactions/
metadata, uploads/lifecycle/statistics, and kernels; there is no public API,
stub, or new library. `CudaFullPlan` retains changing moments/coefficient/
field buffers, permutations, P2P and separate M2L executor wiring,
streams/events/timing, near/far overlap, combination/reordering, and D2H.
Direct, P2P, and M2L remain authoritative in their existing backends. The
next task is orchestration/resource simplification; the `CudaFullPlan`
decomposition is not complete.

Validation handoff: initial M2L sanity ownership audit clean and focused
14-case probe passed; fresh CPU development configure/build 55 steps and full
CTest 193/193 with expected skips #16/#51/#59/#63; fresh CUDA configuration
with `module cuda/13.2` (`nvcc 13.2.78`) for SM75; the preset presented a
206-step graph and compiled the requested `cdfmm_core`, tests, Python
extension, `cdfmm-precompute`, and three benchmark targets, but failed at its
final install step on sandbox read-only Conda site-packages; explicitly
requested target builds then succeeded; focused CUDA CTest 30/30 with expected
unavailable-device skips #16/#59/#63; full
CUDA-configured CTest 193/193 with expected skips #16/#51/#59/#63;
`git diff --check` and independent code/ownership review passed. `nvidia-smi`
could not communicate with the driver, so no numerical GPU runtime execution
is claimed. Unrelated `Article1/` and notebook changes remain untouched.

## 2026-09-14 — CUDA M2L backend extraction closure

The accepted CUDA M2L ownership split is complete. The canonical public
`CudaM2LPlan` declaration is `include/cdfmm/backend/cuda/m2l.hpp`, with
`src/cuda_m2l_plan.hpp` retained as a forwarding compatibility shim.
`src/backend/cuda/m2l/{internal.hpp,plan.cu}` owns the reusable FP64/FP32
device representation, kernels, bounded scaled-multipole scratch policy,
persistent resources, lifecycle, and statistics. Standalone `CudaM2LPlan` and
`CudaFullPlan` consume the same internal executor; `src/cuda_fmm.cu` retains
P2M/M2M/L2L/L2P and full-FMM/far-field orchestration. The non-CUDA boundary is
`src/backend/cuda/stub/m2l.cpp`.

Validation truth: fresh CPU configure/build 194/194; fresh CPU full CTest
193/193 with expected skips #16/#51/#59/#63; fresh CUDA 13.2 SM75 build
204/204 covering core, tests, Python, tools, and benchmarks; focused CUDA
M2L/header/stub coverage 23/23 with unavailable runtime-device cases skipped
as expected; full CUDA CTest 193/193 with the same four skips; and install,
ownership, complete-diff, and `git diff --check` audits passed. `nvidia-smi`
could not reach a driver/device, so no numerical GPU runtime result is
claimed.

The untracked `Article1/` directory and unrelated
`examples/simple_notebooks/tetrahedron_target_average_comsol_compare_sampling_and_gauss.ipynb`
remain untouched and outside this task.

## 2026-09-14 — complete CUDA P2P backend extraction closure

The complete CUDA P2P extraction was accepted as the preceding step and is
included in the current CUDA backend series. `include/cdfmm/backend/cuda/p2p.hpp` is the canonical public header
for `CudaP2PPlan`; `include/cdfmm/cuda_p2p.hpp` remains a legacy forwarding
façade. `src/backend/cuda/p2p/{internal.hpp,plan.cu}` owns all list-1 CUDA P2P
variants and lifecycle/state: canonical AoS, compact/source-only SoA,
leaf-block, signed tensor dictionary (source-warp, target-owned, and
power-of-two microtiles), and cuSPARSE BSR(3), in FP64 and FP32. The package
also owns shared full-plan primitives and the non-CUDA path is
`src/backend/cuda/stub/p2p.cpp`. At that checkpoint, `src/cuda_fmm.cu`
retained M2L and far-field/full-FMM orchestration and consumed P2P internal
primitives; the accepted M2L extraction is recorded above.

Validation truth for this closure:

- fresh CPU full CTest: 192/192, with four expected skips;
- fresh CUDA 13.2 SM75 full build: 206 steps covering core, tests, Python,
  examples, tools, and benchmarks;
- CUDA-configured full CTest: 192/192;
- runtime numerical GPU validation unavailable because `nvidia-smi` could not
  communicate with the driver; and
- ownership, `nm`, and `git diff --check` audits clean.

The untracked `Article1/` directory and unrelated
`examples/simple_notebooks/tetrahedron_target_average_comsol_compare.ipynb`
were preserved untouched and are outside this task. Its focused validation
was recorded before the combined CUDA M2L closure.

## 2026-09-14 — CUDA direct backend extraction closure

The accepted CUDA direct extraction is complete. The
canonical public headers are
`include/cdfmm/backend/cuda/{direct,dense_direct}.hpp`; the old
`include/cdfmm/cuda_direct.hpp` and `include/cdfmm/cuda_cuboid.hpp` paths are
compatibility façades. Shared internal error/runtime handling is in
`src/backend/cuda/common/`, point O(N^2) and dense cuBLAS direct execution are
in `src/backend/cuda/direct/`, and non-CUDA direct stubs are in
`src/backend/cuda/stub/direct.cpp`. At that earlier checkpoint,
`src/cuda_fmm.cu` still owned P2P, M2L, and full-FMM execution. No
behaviour/performance change was intended and no P2P/M2L/full decomposition
had yet been attempted.

Validation handoff: fresh dev configure/build 59 targets; focused
direct/header/stub coverage 8/8, including the exact disabled-stub behaviour
case; full CPU CTest 189/189 with expected skips #16/#51/#59/#63; fresh CUDA
13.2 configure with explicit arch 75 and cached Catch2 source; and a full
198-target CUDA build covering tests, Python, and benchmarks. Independent full
CUDA CTest reported 189/189, but CUDA numerical/runtime
runtime cases only took unavailable-device guards because `/dev/nvidia` was
absent and `nvidia-smi` could not reach the driver. Ownership/`nm` searches
and `git diff --check` were clean. `Article1/` remains untouched.

At that earlier checkpoint, no implementation follow-up was implied beyond the
then-deferred P2P/M2L/full-FMM CUDA decomposition.

## 2026-09-11 — FMM, CPU, and oneMKL execution boundaries

The two-package FMM boundary refactor is complete. Commit `f8218fc` moved
`UniformFmm` and far-field orchestration under `src/fmm` and CPU list-1
execution under `src/backend/cpu`. The follow-up oneMKL package places grouped
M2L preparation, reusable FP32/FP64 scratch, SGEMM/DGEMM, thread-local MKL
control, serial scatter, storage accounting, phase timing, and availability in
`src/backend/mkl/m2l.{hpp,cpp}`. `src/fmm/far_field.cpp` retains the expansion
pass order and timing aggregation, while an opaque internal owner removes the
group/scratch layout from `include/cdfmm/uniform_fmm.hpp`.

Fresh portable construction passed 52/52 build steps. Focused portable checks
completed 21 cases (20 passed, one expected oneMKL skip), and full portable
CTest completed 184 cases (180 passed, expected skips #16/#51/#59/#63). The
CPU+oneMKL build reconfigured and rebuilt successfully; focused M2L, spherical,
cache, and precision checks completed 50 cases (48 passed, CUDA skips #16/#59),
and full oneMKL CTest completed 184 cases (181 passed, CUDA skips
#16/#59/#63). Source-level scaling, mixed-level stable grouping, persistent
scratch accounting, phase timing, repeated execution, and move semantics have
focused regression coverage. Ownership, symbol, source-duplication, and diff
audits passed.

CUDA runtime, Python, Fortran, and documentation builds were not run. No CUDA
source was changed. The untracked `Article1/` directory and unrelated parent
MagTense work remain untouched.

## 2026-09-11 — dense-direct plan/backend separation

The dense-direct plan/backend separation is complete: the canonical declaration
is `include/cdfmm/plan/direct/dense.hpp`, preparation is in
`src/plan/direct/dense.cpp`, portable execution is in
`src/backend/cpu/direct/dense.cpp`, and oneMKL execution plus availability is
in `src/backend/mkl/direct/dense.cpp`. A private shared workspace keeps FP32
and FP64 staging arrays reusable without exposing backend state in the public
header. Explicit deep-copy and move members preserve prior value semantics and
ensure copies do not share mutable workspace. The legacy
`include/cdfmm/cuboid.hpp` path remains a compatibility umbrella; cuboid math
and pair tensors remain in `src/cuboid.cpp`.

Trusted validation for this completed step:

- fresh dev configure/build: 66/66;
- focused geometry/direct/precision CTest: 28/28 passed;
- full portable CTest: 183/183 passed with expected skips #16/#51/#59/#63;
- CPU+oneMKL reconfigure/build and focused direct/cuboid/oneMKL/precision
  CTest: 31/31 passed;
- full oneMKL CTest: 183/183 passed with expected skips #16/#59/#63;
- canonical-only and legacy-header compile probes remain covered by the public
  header suite; and
- `git diff --check` and backend ownership searches passed.

CUDA runtime validation was not part of this CPU/oneMKL subphase. `Article1/`
and unrelated parent-worktree changes remain untouched.

## 2026-09-10 — shared tree root-box resolution

The root-box extraction is complete and recorded in this focused change/commit.
Internal
`src/tree/common/root_box.{hpp,cpp}` now resolves and validates common cubic
roots for both UniformTree and AdaptiveTree. AdaptiveTree no longer depends on
or constructs UniformTree. Shared resolution retains its inferred coincident
half-width `0`; AdaptiveTree applies its existing fallback `1` at its call
site, and the mode-specific validation differences remain intact. The helper
is not installed, and no public API changed.

Trusted validation:

- fresh development configure with
  `/home/mihaa/.conda/envs/cdfmm/bin/cmake` succeeded;
- build: 48/48;
- focused UniformTree tests: 6/6;
- focused adaptive/integration C++ tests: 5/5;
- `python_tests/test_adaptive_tree.py`: 10 passed with `PYTHONPATH=build`;
- full CTest: 182/182, with four expected CUDA/oneMKL optional skips; and
- `git diff --check`, dependency/source search, and symbol search: clean.

CUDA runtime validation and the full Python suite were not run. `Article1/`
and unrelated parent-worktree changes remain untouched. Changed implementation
files are `src/tree/common/root_box.{hpp,cpp}`,
`src/tree/uniform/uniform_tree.cpp`, and
`src/tree/adaptive/adaptive_tree.cpp`; the documentation handoff is recorded
in this file, `project_progress.md`, `project_diary.md`, and
`docs/architecture.md`.
