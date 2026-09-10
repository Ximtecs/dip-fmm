# Project diary

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
