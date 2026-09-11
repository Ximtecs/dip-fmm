# Latest session work

## 2026-09-11 — dense-direct plan ownership

The dense-direct plan ownership move is complete: the canonical declaration is
`include/cdfmm/plan/direct/dense.hpp`, the implementation is
`src/plan/direct/dense.cpp`, and CMake builds the new source. The legacy
`include/cdfmm/cuboid.hpp` path remains a compatibility umbrella; cuboid math
and pair tensors remain in `src/cuboid.cpp`. Portable and oneMKL GEMV mechanics
remain intentionally co-located with the plan pending the next focused backend
extraction step.

Trusted validation:

- fresh dev configure/build: 64/64;
- focused geometry/direct/precision CTest: 41/41 passed;
- full portable CTest: 182/182 passed with expected skips #16/#51/#59/#63;
- CPU+oneMKL build and dense/cuboid focused CTest: 25/25 passed;
- canonical-only and legacy-header compile probes passed; and
- installed canonical header presence, `git diff --check`, and ownership
  searches passed.

CUDA runtime validation was not available. oneMKL was already present and was
exercised without installing anything. `Article1/` and unrelated parent-worktree
changes remain untouched.

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
