# Project progress

## Dense-direct plan ownership

The dense direct plan now has an explicit plan-layer home:
`include/cdfmm/plan/direct/dense.hpp` is the canonical public declaration and
`src/plan/direct/dense.cpp` owns construction, evaluation, backend selection,
and tensor-memory accounting. `include/cdfmm/cuboid.hpp` remains a
compatibility umbrella and `src/cuboid.cpp` retains cuboid monomial and pair
tensor mathematics. Portable and oneMKL GEMV mechanics intentionally remain
co-located with the plan; extraction to dedicated direct backends is the next
focused dense-direct step.

Validation for this source/include ownership move:

- fresh `dev` configure/build passed (64/64 build steps);
- focused geometry/direct/precision CTest passed 41/41;
- full portable CTest passed 182/182, with expected skips #16, #51, #59,
  and #63;
- a separate CPU+oneMKL configure/build passed, and its dense/cuboid focused
  CTest passed 25/25;
- canonical-only and legacy-header compile probes passed, including a safe
  temporary-prefix install check for `plan/direct/dense.hpp`; and
- `git diff --check` and authoritative declaration/source searches were clean.

CUDA runtime paths were not exercised in this CPU-only environment. The
oneMKL path was exercised from the existing environment; no dependency was
installed.

Status recorded 2026-09-10 after completing the shared tree root-box
extraction in the nested checkout. The implementation and documentation are
recorded in this focused change/commit.

## Shared tree root-box resolution

- `src/tree/common/root_box.{hpp,cpp}` is the internal shared boundary for
  resolving and validating common cubic roots used by both UniformTree and
  AdaptiveTree.
- AdaptiveTree no longer depends on or constructs UniformTree.
- Shared resolution retains an inferred coincident half-width of `0`; AdaptiveTree
  applies its existing fallback half-width of `1` at its call site. Their
  validation differences remain unchanged.
- The helper is not installed and introduces no public API.

Trusted validation evidence for this change:

- fresh development configure with
  `/home/mihaa/.conda/envs/cdfmm/bin/cmake` succeeded and the build passed
  48/48;
- focused UniformTree CTest passed 6/6;
- focused adaptive/integration C++ tests passed 5/5;
- `python_tests/test_adaptive_tree.py` passed 10 tests with `PYTHONPATH=build`;
- full CTest passed 182/182, with four expected CUDA/oneMKL optional skips;
- `git diff --check` passed, and dependency/source and symbol searches were
  clean; and
- the internal helper is absent from the install tree.

CUDA runtime validation and the full Python suite were not run and are not
claimed. `Article1/` and unrelated parent-worktree changes remain untouched.

## Static topology and UniformTree adapter closure

The accepted current change separates static topology from the uniform-tree
adapter:

- `include/cdfmm/tree/static_topology.hpp` is the canonical topology-only
  interface and `src/tree/common/static_topology.cpp` owns validation/storage
  behaviour;
- `include/cdfmm/tree/uniform_topology.hpp` declares the UniformTree adapter and
  `src/tree/uniform/static_topology_adapter.cpp` builds canonical topology from
  a UniformTree; and
- `include/cdfmm/static_topology.hpp` remains a compatibility forwarding
  header, while the removed `src/static_topology.cpp` is no longer a source
  home.

The function bodies were verified byte-identical apart from split context, so
the refactor preserves behaviour while clarifying ownership and dependencies.

Trusted validation evidence for this source/include-only change:

- development fresh configure/build passed;
- independent focused CTest passed 23/23, with one expected CUDA skip;
- full CTest reported 178 total, 174 passed, four expected skips (#16, #51,
  #59, and #63), and zero failures;
- standalone canonical-only and legacy-header compile probes passed; and
- `git diff --check` passed.

Python tests were not rerun because this change only moves source and include
ownership. No CUDA runtime result is claimed beyond the expected focused-test
skip.

## Repository and refactor state

- Branch: `refactor/architecture-v0.2`.
- The refactor branch has advanced from `a8565dc` through focused plan,
  operator, integration, and boundary-test commits (`4395f4a`, `d1dcfd7`,
  `14a94d6`, and `21cb5b1`).
- The immutable `v0.1.0` tag and `release/v0.1` branch remain the preserved
  pre-refactor baseline. The current branch follows the v0.2 architecture
  contract and has completed the foundational Step-2 organisation of core,
  math, geometry, and tree layers.
- The nested checkout has an untracked `Article1/` checkout/artifact. It is
  outside this documentation task and must remain untouched. The parent
  MagTense worktree also contains unrelated user changes; do not fold them
  into the nested refactor.

- The accepted dynamic P2M/M2M/M2L/L2L/L2P/P2P mathematics now lives in
  responsibility-specific operator sources. M2P has a narrow reference
  header/source, P2P's direct sum is namespaced, and `src/operators.cpp`
  contains only thin compatibility wrappers. Direct namespaced-vs-flat
  compatibility coverage is included in the foundational header test.
- The operator/plan history remains committed before this session; the current
  topology split and these memory updates are recorded in a focused commit.
  Preserve `Article1/` and all unrelated parent-worktree changes.

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
compatibility umbrellas. `StaticFmmTopology` is now the canonical topology
boundary; the separate UniformTree adapter remains the transitional tree-
to-topology construction seam for a later FMM/topology step.

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

The accepted topology-split evidence is the configure/build, focused CTest,
full CTest, compile probes, body comparison, and diff-check results listed
above. Python tests were not rerun. A read-only artifact audit found no stray
temporary, reject, backup, secret, or task-artifact files outside `Article1/`;
that directory was not scanned or altered.

## Earlier session handoff — 2026-09-10

The shared orchestration setup was verified as six roles at concurrency six,
with routes/templates/bootstrap files and an original-config backup. Static
TOML/shell/diff checks passed; a fresh CLI recognized every role, the intended
Light/Medium/Heavy semantics, and absolute route paths. These were tooling
checks only: no project build or test ran because production code was unchanged.
