# Project diary

## 2026-09-10 — operators and static plans refactor

The committed architecture-v0.2 series now has explicit operator homes for
P2M, M2M, M2L, L2L, L2P, and P2P construction; explicit plan homes for
canonical static data, FP32 conversion, translation/L2P records, and derived
P2P packings; and a portable CPU static-plan application boundary. The legacy
flat headers remain compatibility umbrellas. The tree-to-plan adaptation in
`StaticFmmTopology` is intentionally still transitional, while CUDA/backend
decomposition and FMM orchestration are deferred.

The dev build and full CTest run are green: 177/177 tests passed with expected
unavailable-feature skips. Python regressions passed 137 tests with 7 skips;
public/header, install-tree, and vectorisation checks also passed. The
independent tester was interrupted by the user before its final report, so a
fresh audit with an explicit CUDA-availability check and a representative
comparison with `v0.1.0` / `release/v0.1` remains the next step.

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
