# Latest session work

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
