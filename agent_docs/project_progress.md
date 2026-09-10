# Project progress

Status recorded 2026-09-10 from the nested checkout and checked-in documents.

## Repository and refactor state

- Branch: `refactor/architecture-v0.2`.
- HEAD: `8f24e4d` (`fix(architecture): close foundational review findings`),
  matching `origin/refactor/architecture-v0.2`.
- The immutable `v0.1.0` tag and `release/v0.1` branch remain the preserved
  pre-refactor baseline. The current branch follows the v0.2 architecture
  contract and has completed the foundational Step-2 organisation of core,
  math, geometry, and tree layers.
- Pre-existing nested-worktree changes are a modified
  `examples/simple_notebooks/simple_geometry_magtense_compare.ipynb` and an
  untracked `Article1/` checkout/artifact. They are outside this documentation
  task and must remain untouched.

## Implemented foundation evidenced by the repository

The root guidance, README, docs, headers, source, and tests describe a
production static `UniformFmm` plan with complete uniform and geometry-only
adaptive topology paths; Cartesian and real-spherical operators; exact direct
CPU/CUDA references; FP32/FP64 state; portable CPU, oneMKL M2L, hybrid CUDA,
and CUDA-full backends; persistent universal/geometry caches; point, prism,
and tetrahedron models; and a C ABI plus Python and optional Fortran bindings.
These are documented capabilities, not a claim that every optional build was
run in this session.

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

No build, test, benchmark, documentation build, commit, cleanup, or optional
backend probe was performed for this documentation bootstrap; outcomes remain
unreported rather than inferred.

## Session handoff — 2026-09-10

The shared orchestration setup was verified as six roles at concurrency six,
with routes/templates/bootstrap files and an original-config backup. Static
TOML/shell/diff checks passed; a fresh CLI recognized every role, the intended
Light/Medium/Heavy semantics, and absolute route paths. These were tooling
checks only: no project build or test ran because production code was unchanged.
