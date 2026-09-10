# Latest session work

## 2026-09-10 — dynamic operator ownership closure

The accepted refactor is closed. Responsibility-specific sources under
`src/operators/` now own dynamic P2M/M2M/M2L/L2L/L2P/P2P mathematics, with a
narrow M2P reference source/header and namespaced P2P summation. Flat
`src/operators.cpp` is a compatibility-wrapper translation unit only. CMake,
public operator umbrella/header guidance, architecture/validation docs, and a
direct namespaced-vs-flat compatibility test were updated accordingly.

Verification: clean dev configure/build; CTest 178/178 with expected skips
#16/#51/#59/#63; Python 137 passed and 7 skipped with `PYTHONPATH=build`;
independent CTest 178/178; targeted operator tests 64/64; Python operator
bindings 10/10; and symbol/`rg` audit clean. CUDA was unavailable (`nvidia-smi`
could not access an NVIDIA driver), so CUDA runtime validation remains open.

The closure commit is the only new task commit. Preserve the pre-existing
modified `examples/simple_notebooks/simple_geometry_magtense_compare.ipynb`
and untracked `Article1/`; do not stage them. Next work, if authorised, is
optional CUDA/oneMKL validation or later backend/FMM orchestration decomposition.

## 2026-09-10 — operators and static plans refactor handoff

The committed `refactor/architecture-v0.2` series now separates the
mathematical P2M/M2M/M2L/L2L/L2P/P2P operators, canonical static plan data,
derived P2P execution packings, FP32 conversion, and portable CPU static-plan
application. Compatibility flat headers remain forwarding interfaces. The
remaining intentional seam is `StaticFmmTopology`, which still adapts tree
interaction data to FMM plan records; CUDA/backend decomposition and FMM
orchestration remain future work.

Earlier validation completed through the dev path: full CTest 177/177 passed
with expected unavailable-feature skips; Python regressions were 137 passed and
7 skipped; public/header, install-tree, and vectorisation-report checks passed.
The CTest evidence is retained in `build/Testing/Temporary/LastTest.log`.
The implementation and boundary tests were committed as `4395f4a`, `d1dcfd7`,
`14a94d6`, and `21cb5b1`; preserve the modified
`examples/simple_notebooks/simple_geometry_magtense_compare.ipynb` and the
untracked `Article1/` checkout/artifacts that predated this work. Ignored build,
cache, and pytest-cache directories remain local artifacts; no stray temporary
or reject files were found.

## 2026-09-10 — repository-memory bootstrap

Inspected the nested root guidance, source/include ownership, CMake presets and
top-level build, CI workflow, tests, Python/Fortran/C bindings, and the
mathematical, backend, precision, cache, validation, benchmark, and roadmap
documents. Added this `agent_docs/` set to capture the current persistent FMM
architecture and evidence-backed workflow memory.

Added one minimal paragraph to `AGENTS.md` requiring agents to read and
maintain `agent_docs/` while retaining the existing hierarchy and global
orchestration inheritance.

Observed state before the documentation change: branch
`refactor/architecture-v0.2` at `8f24e4d`, equal to its origin tracking ref;
modified `examples/simple_notebooks/simple_geometry_magtense_compare.ipynb`;
and untracked `Article1/`. No pre-existing file was reset, cleaned, committed,
or otherwise changed. No build, test, benchmark, Sphinx build, or backend
validation was run or claimed.

## 2026-09-10 — orchestration handoff

Global orchestration was installed under `~/.codex` with six roles and
concurrency six, including routes, templates, bootstrap files, and a backup of
the original configuration. The root MagTense and this nested `dip-fmm`
project-memory set were bootstrapped. Static orchestration Tester checks passed
for TOML, shell, and diff files. A fresh CLI recognized all roles with the
intended Light/Medium/Heavy semantics and successfully read absolute route
paths.

`codex exec` 0.153.4 JSON smoke attempts to auto- or explicitly delegate
emitted wait calls with empty receiver IDs and showed no observable spawn event.
The current app session's collaboration tools did successfully run Explorer,
Researcher, two Luna executors, and Tester. No project builds or tests were run
because production code was unchanged. No commit is planned while the root and
nested worktrees contain unrelated user changes.
