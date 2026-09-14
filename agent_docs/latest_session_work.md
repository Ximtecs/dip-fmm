# Latest session work

## 2026-09-14 — complete CUDA P2P backend extraction closure

The complete CUDA P2P extraction is complete in the current uncommitted
checkout. `include/cdfmm/backend/cuda/p2p.hpp` is the canonical public header
for `CudaP2PPlan`; `include/cdfmm/cuda_p2p.hpp` remains a legacy forwarding
façade. `src/backend/cuda/p2p/{internal.hpp,plan.cu}` owns all list-1 CUDA P2P
variants and lifecycle/state: canonical AoS, compact/source-only SoA,
leaf-block, signed tensor dictionary (source-warp, target-owned, and
power-of-two microtiles), and cuSPARSE BSR(3), in FP64 and FP32. The package
also owns shared full-plan primitives and the non-CUDA path is
`src/backend/cuda/stub/p2p.cpp`. `src/cuda_fmm.cu` now retains M2L and
far-field/full-FMM orchestration and consumes P2P internal primitives.

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
were preserved untouched and are outside this task. No commit was made; the
main agent must inspect the complete nested diff before committing.

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
