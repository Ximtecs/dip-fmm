# Latest session work

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
