# Project progress

## Tree/topology/FMM-plan boundary cleanup: COMPLETE — 2026-09-16

Two parts. First, a narrow carry-over correction: `assign_static_p2p_compact_row`
(added to the installed `include/cdfmm/plan/p2p/compact.hpp` by the prior
internal-duplication cleanup) was relocated to the internal
`src/plan/p2p/compact_row.hpp`, since it existed only so
`plan/p2p/compact.cpp` and `cache/format.cpp` could share packing mechanics,
not as a supported downstream API. `compact.hpp` keeps the public
`StaticP2PCompactPlan`/`FloatStaticP2PCompactPlan` types and builders; no
cache format, key, or numerical behaviour changed. Committed separately as
`refactor(p2p): keep compact row packing internal`.

Second, the main task: an audit of the `StaticFmmTopology`/tree-to-plan
boundary that Phase 1 had recorded as a transitional seam (the "still open"
item at the top of the previous entry below). Two read-only workers
classified every `StaticFmmTopology` field/function and traced every
producer and consumer. Finding: `StaticFmmTopology` was already legitimate
tree-owned topology — every field is a spatial-tree fact, an
interaction-topology fact, or CSR-style schedule indexing over those facts;
no operator coefficients, execution packing, or backend state anywhere in
`tree/`. `fmm`/`plan` consume it by `shared_ptr<const StaticFmmTopology>`
reference and never rebuild it; the cache subsystem never persists it
directly and disables caching entirely whenever a topology is supplied
externally. `AdaptiveTree`'s constructor already separates spatial-octree
construction from near/far interaction-topology assembly into two
sequential, non-interleaved passes (with `tree_seconds_`/`interaction_seconds_`
already timing exactly that boundary); introducing a separate adaptive-only
spatial-tree type would duplicate nearly every `StaticFmmTopology` field, so
none was added. **No production code changed.** The "transitional tree-to-plan
coupling" language in `adaptive_tree.hpp`'s `@warning` and in several
`docs/architecture.md` passages was a documentation defect, not a live
architectural problem, and was corrected; see "Tree/topology boundary
cleanup" in `docs/architecture.md` for the full audit record. Full CPU CTest
198 total: 193 passed, 4 expected skips, and one pre-existing unrelated
failure (below); Python 137 passed/7 skipped; the P2P correction's own
focused/cross-version validation all pass. See
`agent_docs/latest_session_work.md`.

One pre-existing, unrelated failure was found and left alone (out of scope):
"static triangular translations match M2M and L2L references" fails
identically on this HEAD and on the unmodified `76d6c9ab` baseline (an
M2M/L2L floating-point comparison issue, not a P2P/cache/tree regression).

Still open, each its own future task: deeper cache/`UniformFmm` encapsulation
(cache entry points are still `UniformFmm` members); the flat-header/packaging
relocation candidates (`periodic.cpp`, `parameter_selection.cpp`,
`validation.cpp`, and the headers awaiting a canonical subsystem home,
including `parameter_selection.hpp` including the complete `uniform_fmm.hpp`);
and repository pruning.

## Internal-duplication cleanup after Phase 1: COMPLETE — 2026-09-15

The canonical-to-compact P2P packing rule (independently stated by
`src/plan/p2p/compact.cpp` and the fused warm-cache decode in
`src/cache/format.cpp`, and previously recorded as deferred rather than
resolved) now has one authoritative statement,
`assign_static_p2p_compact_row` in `include/cdfmm/plan/p2p/compact.hpp`. The
cache's fused single-pass decode is unchanged in structure — it now calls the
shared function per row instead of restating the field mapping. A duplicated
`n!` loop (`rectangular_prism.cpp` vs `MultiIndexSet::factorial`) was also
removed. See `agent_docs/latest_session_work.md` for full validation evidence
(cross-version `v0.1.0` cache-compatibility probe, before/after warm-cache
timing, and the full CPU/CUDA/oneMKL/CUDA+oneMKL test matrix, all clean). No
cache format, cache key, or public API/ABI change.

Still open, each its own future task: the `StaticFmmTopology`/tree-to-plan
boundary seam; deeper cache/`UniformFmm` encapsulation (cache entry points
are still `UniformFmm` members); the flat-header/packaging relocation
candidates (`periodic.cpp`, `parameter_selection.cpp`, `validation.cpp`, and
the headers awaiting a canonical subsystem home); and repository pruning.

## Phase 1 architecture refactor: COMPLETE — 2026-09-15

Phase 1 of the `v0.2` architecture refactor is closed. Starting HEAD `66b9436`.
The authoritative record — ownership/dependency audit, `v0.1.0` comparison,
validation matrix, cache and install evidence, defects fixed, and the Phase-2
handoff — is in the "Phase 1 closure" and "Phase 2 handoff" sections of
`docs/architecture.md`. Summary of status:

| Area | Status |
|---|---|
| Preserved baseline (`v0.1.0`, `release/v0.1`) | untouched, verified |
| Ownership / dependency direction | no prohibited edge; exceptions enumerated |
| Public headers (29 v0.1.0 paths) | all retained; none removed |
| C ABI | `c_api.h` byte-identical; version 1; 14 exported symbols |
| Python surface | zero-diff vs pre-split bindings |
| Cache format and keys | backward compatible against a `v0.1.0` corpus |
| Portable CPU | 198/198 CTest, 137 pytest |
| oneMKL (no CUDA) | 198/198 CTest, 139 pytest |
| CUDA (no oneMKL, SM120) | 198/198 CTest, 141 pytest |
| CUDA + oneMKL | 198/198 CTest, **zero skips**, 143 pytest |
| Fortran (`ifx` 2025.2.1) | 199/199 CTest; smoke test and example run |
| Isolated install + downstream consumer | pass, 1.6e-16 analytic agreement |
| Documentation build | `sphinx-build -W` clean, zero warnings |
| Performance | CPU 11.47 ms vs 11.56 ms at `v0.1.0`; no regression |

Behaviour preservation was demonstrated directly: a probe compiling unchanged
against both `v0.1.0` and HEAD public headers returns bit-identical fields for
spherical FP32, spherical FP64, prism FP32, Cartesian FP64, and periodic FP64.

Four narrow Phase-1 defects were fixed during closure: a **pre-existing**
periodic-cell-centre error in `tests/test_fortran_api.f90` (identical failure at
`v0.1.0`, undetected because no Fortran compiler was previously available); a
stale macro name in `docs/Doxyfile` that caused the two long-standing Sphinx
warnings; and stale structure/status claims in `include/cdfmm/AGENTS.md`,
`src/AGENTS.md`, and `docs/architecture.md`.

Standing limitations, recorded rather than hidden: CI remains portable CPU
only, so oneMKL, CUDA, and Fortran are validated manually; the optional external
MagTense package is absent, so one Python comparison skips; and most CUDA-gated
C++ cases return via `SUCCEED()` rather than a true `SKIP()`, so a portable-CPU
run reports them as passed rather than skipped.

Phase 2 is **not started** and needs its own explicit task.

## Compatibility and transitional-source review — 2026-09-15

Completed the late-Phase-1 compatibility/transitional-source review from
starting HEAD `5cb5576` (`refactor(bindings): structure language adapters`).
Every remaining transitional file was audited and classified; the remaining
compatibility surface is now intentional rather than inherited.

Decisive evidence: `git ls-tree v0.1.0 include/cdfmm/` lists exactly the same
29 flat headers present today, and `CMakeLists.txt` installs
`include/cdfmm/` by whole-directory glob. Every flat public header is therefore
a shipped, installed pre-v0.2 include path, so none qualified for removal.

Classification outcome:

- **A, public compatibility façades retained (23).** The 19 pure forwarding
  headers plus `cuboid.hpp`, `geometry.hpp`, `static_operators.hpp`, and
  `tetrahedron.hpp`. Each now carries an explicit comment stating why it
  exists. `cuboid.hpp` and `static_operators.hpp` were rewritten to reproduce
  their pre-v0.2 transitive surface exactly, which the build caught when an
  initial narrowing dropped `DenseDirectBackend` from
  `cdfmm/static_operators.hpp`.
- **B, internal shims removed (4).** `src/cuda_fmm_plan.hpp` and
  `src/cuda_p2p_plan.hpp` had zero includers; `src/cuda_m2l_plan.hpp` had two,
  both retargeted to `cdfmm/backend/cuda/m2l.hpp`; `src/static_operators.cpp`
  was dead code, absent from every source list since `14a94d6`.
- **C, compatibility implementations retained (2).** `src/operators.cpp` is
  eight one-expression delegations with no inline mathematics, and backs the
  supported flat C++ and Python operator names. `include/cdfmm/operators.hpp`
  is an API adapter, not a forwarding umbrella; `docs/architecture.md` was
  corrected, as it had described it inaccurately.
- **D, substantive implementation moved (1).** `src/cuboid.cpp`.

`src/cuboid.cpp` decision: it was not a compatibility adapter. It owned
`cuboid_averaged_monomial`, consumed by `src/operators/p2m.cpp` and
`src/operators/l2p.cpp`, and `build_pair_tensor`, consumed by
`src/operators/p2p.cpp` and `src/plan/direct/dense.cpp`. The canonical
`cdfmm::operators::p2p::build_pair` was a pass-through to it, so the dependency
ran canonical -> legacy. The prism-averaged monomial moved to
`src/geometry/primitives/rectangular_prism.cpp`, restoring symmetry with the
existing `tetrahedron_averaged_monomial`; the pair-tensor dispatch became the
body of `operators::p2p::build_pair` in `src/operators/p2p.cpp`. Both flat
spellings remain as thin delegations. The file is removed and dropped from
CMake. The monomial path keeps its own side-length check rather than the
stricter volume validation used by the exact pair tensors, and keeps the
historical "cuboid" exception wording, so behaviour is unchanged.

`CuboidSize` moved from `include/cdfmm/plan/direct/dense.hpp` to
`include/cdfmm/geometry/primitives/rectangular_prism.hpp`, which is where the
geometry alias belongs and which let the canonical operator headers stop
including a plan header to obtain it.

Canonical production code no longer depends on compatibility façades. Flat
includes were narrowed in `src/operators/{m2l,m2p}.cpp`,
`src/backend/cpu/far_field/internal.hpp`, `src/backend/cuda/fmm/internal.hpp`,
`src/plan/direct/dense.cpp`, and the canonical public headers
`include/cdfmm/operators/*.hpp`, `include/cdfmm/backend/cuda/{direct,dense_direct}.hpp`,
and `include/cdfmm/backend/cpu/static_plan_apply.hpp`. Internal call sites in
`src/fmm/far_field.cpp`, `src/backend/cpu/p2p/near_field.cpp`,
`src/parameter_selection.cpp`, and `src/validation.cpp` now call
`cdfmm::operators::*` directly. `python/internal.hpp` keeps
`cdfmm/operators.hpp` with an explicit note: the Python module exports those
flat names, so it is a justified compatibility dependency.

Flat files retained deliberately: `src/periodic.cpp`,
`src/parameter_selection.cpp`, and `src/validation.cpp` each own one coherent
responsibility. Both audit workers proposed splitting or relocating them; that
was declined as Phase-2 work outside this scope. Likewise the substantive flat
public headers (`timings.hpp`, `periodic.hpp`, `uniform_fmm.hpp`,
`validation.hpp`, `parameter_selection.hpp`, `tensor_dictionary.hpp`) are
supported API awaiting a subsystem home, not debt.

Coverage added: `tests/test_foundational_headers.cpp` now compiles
`cdfmm/cuboid.hpp` and `cdfmm/tensor_dictionary.hpp`, which the audit found
were absent from the compatibility include block, and a new test case asserts
that the legacy and canonical spellings of the moved mathematics agree
exactly.

Validation for this task:

- portable `dev`: fresh configure, full build, CTest 198/198 with the four
  expected optional skips (#16, #55, #63, #67); Python 137 passed and 7
  skipped, matching the previous task exactly;
- CUDA `cuda` preset: fresh configure and build, CTest 198/198 with only the
  oneMKL-only skip, on an RTX 5090 with driver 595.84 and CUDA 13.3.73, so a
  full runtime regression rather than the compile-only regression the scope
  required;
- oneMKL + CUDA `notebooks` preset: fresh configure, build, CTest 198/198 with
  zero skips;
- C ABI unchanged: the same 14 `cdfmm_*` dynamic symbols, with
  `include/cdfmm/c_api.h` and `src/bindings/` untouched;
- installed-header probe: headers installed to a temporary prefix, then two
  separate translation units compiled against legacy-only and canonical-only
  include paths. Both compile, and `nm` shows the legacy and canonical
  spellings resolving to symbols with identical mangled signatures over the
  same types. `nm` on `libcdfmm_core.a` confirms `build_pair_tensor` is now
  defined in `p2p.cpp.o` and `cuboid_averaged_monomial` in
  `rectangular_prism.cpp.o`;
- note for future sessions: LTO links fail with "Disk quota exceeded" unless
  `TMPDIR` is moved off the shared `/tmp` tmpfs.

No FMM, geometry, P2P, tetrahedron, tree, plan, cache, CPU, oneMKL, CUDA,
Python, C ABI, Fortran, or performance behaviour changed. The intervening
`9003b666` tetrahedron P2P performance work is untouched.

Remaining Phase 1: whole-Phase-1 validation and closure. Phase 1 is **not**
claimed closed.

## Bindings boundary and current Phase 1 handoff — 2026-09-15

The bindings boundary is now structured from implementation base/newest
starting HEAD `9003b666` (with the cache commit `9e111ee` already preserved in
history). The C ABI implementation moved byte-identically to
`src/bindings/c_api.cpp`; `include/cdfmm/c_api.h` is unchanged and ABI version
1 remains in force. Fortran remains the
`ISO_C_BINDING -> C ABI -> supported C++` path. The former `python/bindings.cpp`
was removed and replaced by `python/internal.hpp`, `module.cpp` (sole entry
point), and `core.cpp`, `geometry.cpp`, `tree.cpp`, `operators.cpp`,
`direct.cpp`, and `fmm.cpp`. CMake registers those sources. Binding code uses
supported canonical structured headers where available, with only justified
compatibility dependencies; it contains no solver logic or internal backend
headers.

The isolated `9e111ee8` manifest comparison found 63 top-level Python names,
with object kinds, docs/defaults, and member surface identical after
normalising object addresses. The C ABI dynamic symbol set was exactly the
same 14 `cdfmm_*` symbols before and after, and the moved C source is
byte-identical.

Trusted validation for this boundary and the current checkout:

- portable affected/full build: 73/73; CTest 197/197 with four expected
  optional skips; Python 137 passed and 7 skipped;
- oneMKL notebooks configuration built core/C/Python/tests; CTest 197/197
  with three CUDA skips; Python 139 passed and 5 skipped;
- under full access, an RTX 5090 was visible with driver 595.84. A fresh CUDA
  build used the active environment's CUDA compiler/toolkit 13.3.73 with
  explicit SM120; affected targets built 107/107, CTest 197/197 with only
  the oneMKL-only skip, and Python 141 passed and 3 skipped (two oneMKL-only,
  one external MagTense);
- the initial stale SM75 artifacts failed PTX on the RTX 5090 and are not a
  final result;
- no gfortran, ifx, or ifort was available, so Fortran smoke/example checks
  were not run; and
- an independent temporary-prefix install passed. Git diff checks and
  ownership audits passed; `sphinx-build -W --keep-going -b html docs
  docs/_build/html` rendered the documentation and exited nonzero only for the
  same two pre-existing `docs/api.rst` C++ declaration warnings.

The remaining Phase 1 sequence is compatibility/transitional-source review,
then whole-Phase-1 validation. Neither is claimed complete here. Unrelated
`Article1/` and notebook changes remain preserved.

## Cache persistence subsystem — 2026-09-15

`src/cache.cpp` is decomposed into `src/cache/` and removed. The units are
`internal.hpp` (shared format constants, `CacheKind`/`CacheDescriptor`,
`CachePayload`, `Writer`/`Reader`, io and format declarations), `io.cpp` (cache
root and `CDFMM_CACHE_DIR`/`CDFMM_DISABLE_CACHE` policy, the validated file
container, checksum, memory-mapped reads, and the unique-temporary-file plus
`fsync` plus `rename` write), `format.cpp` (field-wise records for the
persisted solver types, including the direct FP32 decode paths), `keys.cpp`
(cache identity and key strings), `universal.cpp` (translation-bank and
periodic-root payloads), and `geometry.cpp` (geometry-plan payload).
`cdfmm_core` builds all five sources on the same target, which preserves
`CDFMM_DEFAULT_CACHE_DIR`, OpenMP, and the floating-point contract.
`src/cache/AGENTS.md` records the local invariants.

Nothing about the persisted artefacts changed. The schema and operator
versions remain 4 and 2, the header field order and widths, the checksum, the
`CacheKind` values, the `/v1` root suffix, the `universal/`, `periodic/` and
`plans/` categories, the filename padding and the `_v02`/`_v04` suffixes, and
every cache-key hash input and its order are unchanged. `plan_preparation.cpp`
remains the coordinator that asks for cached data, builds what is missing, and
asks for a write; the cache subsystem defines no solver mathematics and chooses
no backend or packing.

Deliberately preserved rather than repaired, because this step was
behaviour-preserving: the fused decode-and-compact-pack in `read_p2p_blocks`,
the universal M2L bank layout arithmetic, the plan-invariant validation on
load, and the FP32-only state cleanup in the geometry-cache failure path.
`UniformFmm` keeps its existing private cache members and object layout.

Trusted validation:

- portable: the fresh configure required an explicit `cdfmm` environment PATH
  because literal `cmake` was unavailable; the build completed 67/67, full
  CTest completed 193/193 with expected skips #16, #51, #59, and #63, and the
  focused cache selector completed 9/11 with expected skips #51 and #59;
- oneMKL notebooks configuration: CUDA 13.2 and oneMKL 2026.1.0 built core,
  tests, and Python; full CTest passed 190 with three CUDA skips (193 total),
  focused coverage passed 31 with one CUDA skip (32 total), and Python passed
  139 with five skips;
- CUDA 13.2 SM75: the core/tests/Python build completed 93/93, full CTest
  passed 189 with four expected skips (193 total), direct-CUDA focused coverage
  passed 16 with three expected skips (19 total), and Python import/smoke
  checks passed. Pytest was unavailable for the matching Python 3.13
  environment; `nvidia-smi` could not reach the driver, so no GPU runtime is
  claimed.
- prior old-cache compatibility evidence is preserved and now backed by the
  recovered 11-file corpus: both directions produced cache hits with zero
  writes and identical manifests; the repository's 522-file cache corpus was
  unchanged; and an independent current probe hit old universal/periodic files
  without modifying their hashes.

No implementation defect was found.

This compatibility evidence matters because the existing suite compares cache
keys only between two live instances; a uniform key change would pass it.

The cache task is already committed as `9e111ee` with subject
`refactor(cache): structure persistence subsystem`. Unrelated untracked
`Article1/` and
`examples/simple_notebooks/tetrahedron_target_average_fair_sampling.ipynb`
remain preserved outside the task.

## High-level UniformFmm cleanup — 2026-09-15

The final Phase 1 high-level FMM source decomposition is complete. The
`src/fmm` units now have stable ownership: `construction.cpp` handles geometry
normalisation and construction; `plan_preparation.cpp` handles immutable plan
creation, FP32 quantisation, and high-level cache calls;
`execution_setup.cpp` handles backend resolution/wiring and P2P policy;
`evaluation.cpp` handles the complete near/far lifecycle and timing, preserving
hybrid CUDA begin/CPU-far/finish and cancellation; `far_field.cpp` retains
hierarchy sequencing; `diagnostics.cpp` retains exact summary formatting;
`uniform_fmm.cpp` retains lifecycle, accessors, and inspection; and
`internal.hpp` contains opaque backend-owner declarations. Geometry uses a
representative position plus representative-relative finite-primitive data;
points have no additional primitive record.

Trusted final validation:

- baseline hierarchy and legacy-home sanity checks are clean;
- fresh portable configure/build: 63/63;
- focused portable coverage: 39 total, 36 passed, three expected skips;
- full CPU CTest: 193/193 with expected skips #16, #51, #59, and #63;
- CUDA+oneMKL rebuild: 27/27, followed by full oneMKL CTest 193/193 with
  expected CUDA skips #16, #59, and #63;
- CUDA 13.2 SM75 core/tests/Python build: 27/27;
- focused and full CUDA-configured CTest ultimately completed 193/193 with
  optional runtime skips after the unavailable-device state; and
- diff and ownership audits are clean.

GPU numerical runtime remains unavailable because `nvidia-smi` could not
communicate with the driver and the driver rejected CUDA 13.2 PTX; no
numerical device pass is claimed. At that earlier checkpoint, the cache and
bindings boundaries plus compatibility/transitional review were next; current
remaining work is recorded at the top of this file.

## Portable CPU backend decomposition — 2026-09-15

The accepted portable CPU execution split is complete. Prepared P2P
application, including canonical, compact, leaf, tensor-dictionary,
signed-dictionary/whole-tile, BSR, FP32/FP64, and SIMD paths, now lives under
`src/backend/cpu/p2p/` alongside near-field dispatch and reference-neighbour
evaluation. Prepared M2L application is under `src/backend/cpu/m2l/`. Shared
portable P2M/L2P entry mechanics and level-scaled M2M/L2L translation mechanics
are under `src/backend/cpu/far_field/{entries.hpp,translation.hpp}`, with
`executor.cpp` providing the narrow far-field execution seam. Dense direct
remains under `src/backend/cpu/direct/`.

`include/cdfmm/backend/cpu/{p2p,m2l,far_field}.hpp` are the authoritative
portable execution declarations. The former
`include/cdfmm/backend/cpu/static_plan_apply.hpp` remains a source-compatible
umbrella, while `src/backend/cpu/static_plan_apply.cpp` is removed. Reference
mathematics and high-level pass sequencing remain in `src/fmm/far_field.cpp`.
At that checkpoint the final higher-level `UniformFmm` cleanup was the next
architecture task.

Trusted validation for this closure:

- the initial CUDA hierarchy was structurally present; root
  `src/cuda_fmm.cu` and `src/cuda_fmm_stub.cpp` were absent, and CUDA 13.2 was
  available;
- fresh/full portable CPU CTest completed 193/193 with four expected optional
  skips; repaired focused CPU suites passed 13/13 and 34/34, and standalone
  canonical/legacy header probes passed;
- the oneMKL configure/core build and full CTest completed 193/193 with three
  expected CUDA skips, using environment-specific `-latomic` for linking;
- fresh CUDA SM75 configuration plus `cdfmm_core`/`cdfmm_tests` build passed
  198/198 steps, including rebuilt repaired CPU objects, and focused
  CUDA-configured coverage passed 13/13. No GPU numerical runtime result is
  claimed;
- vectorisation-report configure/build passed and generated a nonempty
  `cdfmm-cpu-backend.vec`; and
- `git diff --check` passed.

The accepted task is committed as `refactor(cpu): decompose portable backend`.
Unrelated untracked `Article1/` and
`examples/simple_notebooks/tetrahedron_target_average_fair_sampling.ipynb`
remain preserved outside the task.

## CUDA full-FMM backend ownership — 2026-09-14

The complete CUDA FMM implementation now has its structured backend home under
`src/backend/cuda/fmm/{internal.hpp,plan.cu}`. The former root implementation
is removed; `src/cuda_fmm_plan.hpp` remains a forwarding shim and the
non-CUDA full-FMM boundary is `src/backend/cuda/stub/fmm.cpp`. Far-field
shared mechanics are grouped in `entries.cuh` (P2M/L2P) and `translation.cuh`
(M2M/L2L), with `CudaTranslationInteraction` owned by the low-level
far-field internal header. CPU backend decomposition remains the next Phase 1
architecture task; this change does not move CPU files.

Validation handoff for this completed closure: the initial far-field sanity
and ownership audit was clean, with the focused CUDA-configured probe passing
5/5; a fresh CPU `dev` configure/build completed 55 targets and full CTest
completed 193/193 with expected optional skips #16, #51, #59, and #63. A
fresh CUDA 13.2 SM75 configure produced a successful 205-step build covering
`cdfmm_core`, `cdfmm_tests`, the Python extension, `cdfmm-precompute`,
`benchmark_uniform_fmm`, `benchmark_p2p`, and
`benchmark_cache_initialisation`. Full CUDA-configured CTest completed
193/193; runtime CUDA cases took their unavailable-device skips because
`nvidia-smi` could not communicate with the driver. The built Python module
imported successfully, and diff/ownership audits were clean. No GPU numerical
runtime result is claimed. Unrelated `Article1/` and notebook changes remain
preserved and outside this task.

## CUDA far-field execution extraction closure — 2026-09-14

The accepted CUDA far-field ownership split is complete. The internal
`src/backend/cuda/far_field/{internal.hpp,executor.cu}` package owns immutable
FP32/FP64 P2M and L2P entries, coefficient degrees, M2M/L2L matrices,
interactions and metadata, uploads/lifecycle/statistics, and the associated
kernels. It adds no public API, stub, or new library. `CudaFullPlan` retains
changing moments/coefficient/field buffers, permutations, P2P and separate M2L
executor wiring, streams/events/timing, near/far overlap,
combination/reordering, and D2H transfer. Direct, P2P, and M2L remain their
authoritative backends. This is not the end of the `CudaFullPlan`
decomposition: orchestration and resource simplification are the next scoped
task.

Validation truth for this closure:

- the initial M2L sanity ownership audit was clean and its focused 14-case
  probe passed;
- a fresh CPU development configure/build completed 55 steps, followed by
  full CTest 193/193 with expected skips #16, #51, #59, and #63;
- a fresh CUDA configure used `module cuda/13.2` (`nvcc 13.2.78`) for SM75;
  the preset presented a 206-step graph and compiled the requested
  `cdfmm_core`, tests, Python extension, `cdfmm-precompute`, and three benchmark
  targets, but failed at its final install step because the Conda
  site-packages directory was read-only in the sandbox; explicitly requested
  target builds then succeeded;
- focused CUDA CTest completed 30/30 with expected unavailable-device skips
  #16, #59, and #63, and full CUDA-configured CTest completed 193/193 with
  expected skips #16, #51, #59, and #63;
- `git diff --check` and the independent code/ownership review passed; and
- `nvidia-smi` could not communicate with the driver, so no numerical GPU
  runtime execution is claimed.

The untracked `Article1/` directory and unrelated notebook changes remain
untouched and outside this task.

## CUDA M2L backend extraction closure — 2026-09-14

The accepted CUDA M2L ownership split is complete. The canonical public
`CudaM2LPlan` declaration is `include/cdfmm/backend/cuda/m2l.hpp`, while
`src/cuda_m2l_plan.hpp` remains a forwarding compatibility shim. The reusable
FP64/FP32 execution representation, kernels, bounded scaled-multipole
scratch policy, persistent resources, statistics, and lifecycle now live in
`src/backend/cuda/m2l/{internal.hpp,plan.cu}`; the non-CUDA boundary is
`src/backend/cuda/stub/m2l.cpp`. Both standalone `CudaM2LPlan` and
`CudaFullPlan` consume the same internal executor. `src/cuda_fmm.cu` retains
P2M, M2M, L2L, L2P, and full-FMM/far-field orchestration plus backend wiring.

Trusted validation for this closure:

- fresh CPU configure/build completed 194/194;
- fresh CPU full CTest completed 193/193 with expected skips #16, #51, #59,
  and #63;
- fresh CUDA 13.2 SM75 build completed 204/204, covering core, tests, Python,
  tools, and benchmarks;
- focused CUDA M2L/header/stub coverage completed 23/23, with unavailable
  runtime-device cases skipped as expected;
- full CUDA CTest completed 193/193 with the same four expected skips;
- the install probe, complete diff review, ownership audits, and
  `git diff --check` passed; and
- `nvidia-smi` could not reach a driver/device, so no numerical GPU runtime
  result is claimed.

The unrelated untracked `Article1/` directory and
`examples/simple_notebooks/tetrahedron_target_average_comsol_compare_sampling_and_gauss.ipynb`
remain untouched and outside this task.

## CUDA P2P backend extraction closure — 2026-09-14

The complete CUDA P2P backend extraction was accepted as the preceding step
and is included in the current CUDA backend series. The canonical public declaration is
`include/cdfmm/backend/cuda/p2p.hpp`; `include/cdfmm/cuda_p2p.hpp` remains a
legacy forwarding façade. `src/backend/cuda/p2p/{internal.hpp,plan.cu}` owns
the P2P plan lifecycle, asynchronous evaluation, persistent device/host state,
and all list-1 execution variants: canonical AoS, compact/source-only SoA,
leaf-block, signed tensor dictionary (source-warp, target-owned, and
power-of-two microtiles), and cuSPARSE BSR(3), in both FP64 and FP32. It also
owns the shared full-plan primitives. Non-CUDA builds use
`src/backend/cuda/stub/p2p.cpp`. At that checkpoint, `src/cuda_fmm.cu`
retained M2L and far-field/full-FMM orchestration and consumed the P2P
internal primitives; the accepted M2L extraction is recorded above.

Trusted validation for this closure:

- fresh CPU full CTest completed 192/192 with four expected skips;
- fresh CUDA 13.2 SM75 configuration completed a 206-step build covering the
  core, tests, Python, examples, tools, and benchmarks;
- the CUDA-configured full CTest completed 192/192;
- runtime numerical GPU validation was unavailable because `nvidia-smi` could
  not communicate with the driver; and
- ownership, `nm`, and `git diff --check` audits were clean.

The untracked `Article1/` directory and unrelated
`examples/simple_notebooks/tetrahedron_target_average_comsol_compare.ipynb`
remain untouched and outside this task. Its focused validation was recorded
before the combined CUDA M2L closure.

## CUDA direct backend extraction closure — 2026-09-14

The accepted CUDA direct extraction gives point and dense direct execution
explicit backend homes without changing intended behaviour or performance:

- `include/cdfmm/backend/cuda/direct.hpp` and
  `include/cdfmm/backend/cuda/dense_direct.hpp` are the canonical public
  headers;
- `include/cdfmm/cuda_direct.hpp` and `include/cdfmm/cuda_cuboid.hpp` remain
  compatibility façades;
- `src/backend/cuda/common/` owns shared internal CUDA error/runtime helpers;
- `src/backend/cuda/direct/` owns point O(N^2) and dense cuBLAS direct
  execution; and
- `src/backend/cuda/stub/direct.cpp` owns non-CUDA direct stubs.

At that earlier checkpoint, `src/cuda_fmm.cu` retained P2P, M2L, and full-FMM
execution. No decomposition of those paths was performed at that checkpoint,
and no behaviour/performance change was intended.

Trusted validation for this closure:

- fresh development configure/build: 59 targets;
- focused direct/header/stub coverage: 8/8, including the exact disabled-stub
  behaviour case;
- full CPU CTest: 189/189, with expected skips #16, #51, #59, and #63;
- fresh CUDA 13.2 configure with explicit architecture 75 and cached Catch2
  source; and
- full CUDA build: 198 targets, including tests, Python, and benchmarks.

The independent full CUDA CTest run reported 189/189, but CUDA numerical/runtime
cases only exercised their unavailable-device guards because `/dev/nvidia` was
absent and `nvidia-smi` could not reach the driver. Ownership and `nm` audits,
and `git diff --check`, were clean. `Article1/` remains untouched.

## FMM, CPU, and oneMKL execution boundaries

High-level FMM implementation now lives under `src/fmm`: construction and
lifecycle remain in `uniform_fmm.cpp`, while `far_field.cpp` retains the
P2M/M2M/M2L-dispatch/L2L/L2P sequence and public timing integration. Portable
CPU execution now follows `src/backend/cpu/{direct,p2p,m2l,far_field}/`:
`p2p/` owns canonical and packed P2P plus near-field execution, `m2l/` owns
prepared M2L, and `far_field/` owns prepared P2M/L2P entries and shared
M2M/L2L translation. `static_plan_apply.hpp` remains a compatibility umbrella;
the former mixed `static_plan_apply.cpp` implementation home is removed.

The oneMKL M2L executor is isolated in `src/backend/mkl/m2l.{hpp,cpp}`. It
derives stable transfer-class groups once from `StaticM2LPlan`, owns reusable
FP32/FP64 gathered and translated buffers behind an opaque `UniformFmm` owner,
and provides narrow apply, storage-statistics, and phase-timing interfaces.
All guarded vendor includes, MKL integer types, SGEMM/DGEMM calls, thread-local
MKL control, grouping, gather, multiplication, and serial scatter now live in
the backend. The installed header no longer exposes `M2LGroup` layouts.

Validation evidence for this boundary:

- fresh portable `dev` configure/build passed 52/52 build steps;
- focused near/far/static-plan/FP32/FP64 tests completed 21 cases: 20 passed
  and the oneMKL cache case was expectedly skipped;
- full portable CTest completed 184 cases: 180 passed and four unavailable
  oneMKL/CUDA cases skipped (#16, #51, #59, and #63);
- the existing CPU+oneMKL configuration reconfigured and rebuilt, and focused
  static-M2L/spherical/cache/precision tests completed 50 cases: 48 passed and
  two CUDA cases skipped (#16 and #59);
- full oneMKL CTest completed 184 cases: 181 passed and three CUDA cases
  skipped (#16, #59, and #63); and
- focused coverage verifies source/target level separation, stable mixed-level
  grouping, FP32 and FP64 execution, persistent storage statistics, timing
  calls, repeated application, and `UniformFmm` move behaviour.

Ownership searches, the old/new implementation comparison, symbol inspection,
and `git diff --check` confirmed the vendor mechanics are confined to
`src/backend/mkl`, the root-level FMM files remain absent, and Package 1 was a
move rather than a duplicate. CUDA runtime, Python, Fortran, and documentation
builds were not exercised; CUDA sources were not changed. `Article1/` and
unrelated parent-worktree changes remain untouched.

## Dense-direct plan/backend separation

The dense direct plan has an explicit plan-layer home:
`include/cdfmm/plan/direct/dense.hpp` is the canonical public declaration and
`src/plan/direct/dense.cpp` owns geometry validation, pair dispatch, tensor
construction, precision storage, backend selection, and tensor-memory
accounting. `include/cdfmm/cuboid.hpp` remains a compatibility umbrella and
`src/cuboid.cpp` retains cuboid monomial and pair tensor mathematics.

Portable execution and oneMKL execution now have dedicated internal homes at
`src/backend/cpu/direct/dense.cpp` and `src/backend/mkl/direct/dense.cpp`.
Reusable staging arrays live in a private shared dense-direct workspace owned
by the façade's pimpl. The public plan retains its constructor, evaluate API,
matrix accessors, backend enum, and deep-copy/value move semantics.

Validation evidence for this step:

- fresh portable `dev` configure/build passed 66/66;
- focused dense/direct/geometry/precision CTest passed 28/28;
- full portable CTest passed 183/183 with expected CUDA/optional skips #16,
  #51, #59, and #63;
- existing oneMKL configuration reconfigured and rebuilt successfully,
  focused direct/cuboid/oneMKL/precision tests passed 31/31, and full oneMKL
  CTest passed 183/183 with
  expected skips #16, #59, and #63; and
- `git diff --check` and ownership searches found no plan-side vendor or
  portable GEMV mechanics. CUDA runtime paths remain deferred by scope.

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
