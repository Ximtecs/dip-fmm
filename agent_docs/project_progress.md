# Project progress

Status recorded 2026-09-10 after closing the accepted dynamic-operator
ownership refactor from the nested checkout, current worktree, and available
validation artifacts.

## Repository and refactor state

- Branch: `refactor/architecture-v0.2`.
- The refactor branch has advanced from `a8565dc` through focused plan,
  operator, integration, and boundary-test commits (`4395f4a`, `d1dcfd7`,
  `14a94d6`, and `21cb5b1`).
- The immutable `v0.1.0` tag and `release/v0.1` branch remain the preserved
  pre-refactor baseline. The current branch follows the v0.2 architecture
  contract and has completed the foundational Step-2 organisation of core,
  math, geometry, and tree layers.
- Pre-existing nested-worktree changes are a modified
  `examples/simple_notebooks/simple_geometry_magtense_compare.ipynb` and an
  untracked `Article1/` checkout/artifact. They are outside this documentation
  task and must remain untouched.

- The accepted dynamic P2M/M2M/M2L/L2L/L2P/P2P mathematics now lives in
  responsibility-specific operator sources. M2P has a narrow reference
  header/source, P2P's direct sum is namespaced, and `src/operators.cpp`
  contains only thin compatibility wrappers. Direct namespaced-vs-flat
  compatibility coverage is included in the foundational header test.
- The final operator migration, CMake registration, boundary test, focused
  architecture/validation/guidance corrections, and this session's memory
  updates are committed in the closure commit reported with this handoff.
  Continue preserving the unrelated notebook and `Article1/` work.

## Implemented foundation and operator/plan step

The root guidance, README, docs, headers, source, and tests describe a
production static `UniformFmm` plan with complete uniform and geometry-only
adaptive topology paths; Cartesian and real-spherical operators; exact direct
CPU/CUDA references; FP32/FP64 state; portable CPU, oneMKL M2L, hybrid CUDA,
and CUDA-full backends; persistent universal/geometry caches; point, prism,
and tetrahedron models; and a C ABI plus Python and optional Fortran bindings.
These are documented capabilities, not a claim that every optional build was
run in this session. The current refactor additionally gives mathematical
P2M/M2M/M2L/L2L/L2P/P2P construction explicit homes under
`include/cdfmm/operators/` and `src/operators/`; canonical static data,
precision conversion, and deterministic P2P packings explicit homes under
`include/cdfmm/plan/` and `src/plan/`; and portable static-plan application an
explicit `backend/cpu/` boundary. The legacy flat operator headers remain
compatibility umbrellas. `StaticFmmTopology` still contains transitional
tree-to-plan adaptation and is the main seam for a later FMM/topology step.

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

## Validation and remaining gaps

The clean dev configure/build passed. CTest completed all 178/178 entries with
only the expected unavailable-feature skips (#16, #51, #59, and #63). The full
Python suite completed with 137 passed and 7 skipped using `PYTHONPATH=build`.
Independent verification reproduced CTest 178/178, targeted operator tests
64/64, and Python operator-binding tests 10/10. A symbol/`rg` audit confirmed
flat-wrapper to namespaced linkage and no forbidden operator dependencies.
CUDA was unavailable because the NVIDIA driver probe (`nvidia-smi`) was
unavailable; CUDA execution is therefore not claimed. No stray temporary,
reject, backup, secret, or untracked build/cache files were added. The modified
comparison notebook and untracked `Article1/` (including generated research
artifacts) remain preserved and outside the closure commit.

## Earlier session handoff — 2026-09-10

The shared orchestration setup was verified as six roles at concurrency six,
with routes/templates/bootstrap files and an original-config backup. Static
TOML/shell/diff checks passed; a fresh CLI recognized every role, the intended
Light/Medium/Heavy semantics, and absolute route paths. These were tooling
checks only: no project build or test ran because production code was unchanged.
