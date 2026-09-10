# Project diary

## 2026-09-10 — shared tree root-box resolution

Completed the internal root-box extraction for both tree modes. Shared
resolution and validation now live in `src/tree/common/root_box.{hpp,cpp}`;
AdaptiveTree no longer depends on or constructs UniformTree. Shared resolution
retains an inferred coincident half-width of `0`; AdaptiveTree applies its
existing fallback `1` at its call site, and their validation differences are
preserved. The helper is not installed and does not change the public API.

Fresh development configure/build passed 48/48; focused UniformTree tests
passed 6/6; focused adaptive/integration C++ tests passed 5/5; the adaptive
Python test passed 10/10 with `PYTHONPATH=build`; and full CTest passed
182/182 with four expected CUDA/oneMKL optional skips. Diff, dependency/source,
and symbol checks were clean. CUDA runtime validation and the full Python suite
were not run. The implementation is recorded in this focused change/commit.
`Article1/` and unrelated parent changes remain untouched.

## 2026-09-10 — static topology and UniformTree adapter closure

Accepted the source/include-only split between canonical static topology and
the UniformTree adapter. `StaticFmmTopology` declarations now live in
`include/cdfmm/tree/static_topology.hpp`, validation/storage in
`src/tree/common/static_topology.cpp`, and UniformTree construction in
`include/cdfmm/tree/uniform_topology.hpp` plus
`src/tree/uniform/static_topology_adapter.cpp`. The flat
`include/cdfmm/static_topology.hpp` remains a compatibility forwarding header;
the old `src/static_topology.cpp` source home is removed.

Function bodies were verified byte-identical apart from split context. Fresh
development configure/build, independent focused CTest (23/23 with one
expected CUDA skip), full CTest (178 total, 174 passed, four expected skips
#16/#51/#59/#63, zero failures), canonical-only and legacy-header compile
probes, and `git diff --check` all passed. Python tests were not rerun for this
source/include-only change. The accepted changes are recorded in a focused
topology-refactor commit; `Article1/` and unrelated parent-worktree changes are
preserved.

## 2026-09-10 — dynamic operator ownership closure

Completed the accepted mathematical operator ownership refactor. P2M, M2M,
M2L, L2L, L2P, and P2P construction/application now have responsibility-
specific homes under `src/operators/`; M2P is a narrow direct multipole
reference operator; P2P direct summation is namespaced; and the flat
`src/operators.cpp` translation unit is compatibility wrappers only. The
direct namespaced-vs-flat test covers all dynamic operators and output/self
semantics.

Clean dev configure/build and independent audits are green: CTest 178/178 with
expected skips #16/#51/#59/#63, Python 137 passed and 7 skipped via
`PYTHONPATH=build`, targeted operators 64/64, and Python operator bindings
10/10. Symbol/`rg` checks found only wrapper-to-namespaced linkage and no
forbidden operator dependencies. CUDA was not available because `nvidia-smi`
could not access an NVIDIA driver, so CUDA runtime equivalence remains
unverified. The final focused closure commit contains only task files;
the comparison notebook and `Article1/` remain untouched.

## 2026-09-10 — operators and static plans refactor

The committed architecture-v0.2 series now has explicit operator homes for
P2M, M2M, M2L, L2L, L2P, and P2P construction; explicit plan homes for
canonical static data, FP32 conversion, translation/L2P records, and derived
P2P packings; and a portable CPU static-plan application boundary. The legacy
flat headers remain compatibility umbrellas. The tree-to-plan adaptation in
`StaticFmmTopology` is intentionally still transitional, while CUDA/backend
decomposition and FMM orchestration are deferred.

The earlier dev build and CTest run were green: 177/177 tests passed with
expected unavailable-feature skips. Python regressions passed 137 tests with 7
skips; public/header, install-tree, and vectorisation checks also passed.

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
