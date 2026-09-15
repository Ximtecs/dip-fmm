# Latest session work

## 2026-09-15 — internal-duplication cleanup after Phase-1 closure

Starting HEAD `47bbe5e2` (`refactor: close phase 1 architecture`). Task:
remove genuinely duplicated implementation rules identified during Phase-1
closure, without reopening the architecture, changing public API/ABI, or
touching the persistent cache format/keys. Two items were in scope.

**Canonical-to-compact P2P packing.** `src/plan/p2p/compact.cpp`'s ordinary
plan builder and the fused warm-cache decode in `src/cache/format.cpp`'s
`read_p2p_blocks` each independently restated the same field-by-field mapping
from a canonical `StaticDipoleBlock`/`FloatStaticDipoleBlock` into
`StaticP2PCompactPlan`/`FloatStaticP2PCompactPlan` (source index, identity
marker, three potential components, six tensor components). `docs/architecture.md`
and `src/cache/AGENTS.md` had recorded this as deferred rather than resolved,
warning specifically against unfusing `read_p2p_blocks`'s single parallel pass
into decode-then-build. The fix respects that: a new inline free function,
`assign_static_p2p_compact_row` (two precision overloads), was added to
`include/cdfmm/plan/p2p/compact.hpp` as the one authoritative statement of the
mapping. `compact.cpp`'s builder now resizes its output arrays and calls it
per row (replacing `reserve`+`push_back`); `format.cpp`'s fused decode calls
the exact same function per row, inside the same `#pragma omp parallel for`
loop, at the same point where it previously hand-copied the fields — no new
pass, no new allocation, no change to the persistent format or cache keys.
`format.cpp`'s own field-copy code for decoding the canonical block *from
persisted bytes* (the on-disk byte layout) is untouched and correctly remains
cache-owned; only the canonical-to-compact semantic mapping moved.

**`n!` duplication.** A private `monomial_factorial` in
`src/geometry/primitives/rectangular_prism.cpp` was byte-for-byte identical
to the existing public `MultiIndexSet::factorial` (`include/cdfmm/math/multi_index.hpp`),
which `rectangular_prism.hpp` already transitively includes. Removed the
private copy; the two call sites in `rectangular_prism_averaged_monomial` now
call `MultiIndexSet::factorial` directly, matching the established
`geometry -> math` dependency direction. Two adjacent-but-different factorial
helpers were found and deliberately left alone: `spherical_harmonics.cpp`'s
`long double factorial` (different return type/precision, for spherical-
harmonic normalisation) and a duplicated `odd_double_factorial` between
`spherical_harmonics.cpp` and `operators/spherical_cartesian_conversion.hpp`
(flagged for a possible future follow-up, out of scope here).

Validation: full portable-CPU CTest 198/198 (4 expected CUDA/oneMKL skips);
full CUDA-only (RTX 5090, SM120, CUDA 13.3.73) CTest 198/198 (1 expected
oneMKL skip); full oneMKL-only CTest 198/198 (4 expected CUDA skips); full
CUDA+oneMKL CTest 198/198 with **zero skips**. Python: 137 passed/7 skipped
against the portable `build/` module, 143 passed/1 skipped against the
CUDA+oneMKL `build-notebooks/` module (`cdfmm.__file__` checked in both
cases). A standalone cross-version probe (public-API-only source, compiled
unchanged against `v0.1.0` in an isolated worktree and against this HEAD)
wrote an 8-file FP64/FP32 point/periodic cache corpus with `v0.1.0`, then read
it back at HEAD: identical universal/geometry cache keys, cache hits on every
scenario, zero bytes written, cache files byte-identical before/after
(`sha256sum`), and field results bit-identical (17 significant digits FP64, 9
FP32). A before/after `git stash` A/B comparison of the exact touched
warm-cache path (216,000 P2P interactions, 5 repeated warm constructions each)
showed no measurable timing difference (~11.78 ms vs ~11.79 ms mean,
within run-to-run noise) — no extra pass or allocation was introduced.
`git diff --check` clean.

Files changed: `include/cdfmm/plan/p2p/compact.hpp`, `src/plan/p2p/compact.cpp`,
`src/cache/format.cpp`, `src/geometry/primitives/rectangular_prism.cpp`.
Cache format, cache keys, and all public API/ABI surfaces are unchanged.
`docs/architecture.md` and `src/cache/AGENTS.md` updated to reflect the
resolved duplication (the fused-pass *performance choice* in `read_p2p_blocks`
remains, deliberately). `Article1/` and an unrelated untracked notebook in
`examples/simple_notebooks/` were left untouched.

Not started, and explicitly not in scope for this task: the
`StaticFmmTopology`/tree-topology-to-plan boundary seam, deeper cache/`UniformFmm`
encapsulation (cache entry points are still `UniformFmm` members that know the
topology/operator representations they persist), the flat-header/packaging
relocation candidates (`periodic.cpp`, `parameter_selection.cpp`,
`validation.cpp`, and friends), and repository pruning. Each remains its own
future task, as recorded in `docs/architecture.md`'s deferred-work notes.

## 2026-09-15 — whole-Phase-1 validation and formal closure

Starting HEAD `66b9436` (`refactor: close transitional architecture`). Task:
audit and validate the whole Phase-1 architecture, then close it. **Outcome:
Phase 1 is COMPLETE and closed.** The full evidence, the validation matrix, the
enumerated dependency exceptions, and the Phase-2 handoff now live in the
"Phase 1 closure" and "Phase 2 handoff" sections of `docs/architecture.md`;
this entry records how the session ran and what it changed.

Baseline preserved and untouched: `v0.1.0` -> tag object `26eb955`, commit
`2d3d4ea`; `release/v0.1` -> `2d3d4ea`; both matching `origin`.

### Audit

Three read-only workers audited architecture/dependencies, API/compatibility/
install, and documentation/test-matrix consistency. Their findings were
re-verified by the lead before any of them were acted on; two were wrong and
were rejected:

- a worker reported the `cdfmm/uniform_fmm.hpp` include in
  `src/backend/cuda/fmm/internal.hpp` as dead. Removing it broke the build:
  `plan.cu` *defines* `cuda_m2l_p2p_available()`, which that header declares.
  It is a supported public-API relationship, not an inversion. Reverted.
- a worker reported `SourceGeometry::UniformCuboid` and a `std::vector<Vec3>`
  return from `evaluate`. Both were wrong; the enumerators are
  `RectangularPrism` and `evaluate` returns `std::vector<PotentialField>`, at
  `v0.1.0` and at HEAD alike.

Worker findings that did hold up were confirmed against the tree first: the
`operators/`/`plan/` structure claims, the three stale `docs/architecture.md`
passages, and the `docs/Doxyfile` macro-name diagnosis.

### Validation performed

Environment: `cdfmm` Conda env, GCC 15.3, CMake 4.4.3, CUDA 13.3.73, oneMKL
2026.1, Python 3.11, RTX 5090 SM120 driver 595.84. Note `cmake` and `nvcc` are
not on the base PATH; that env must be active. `python -c "import cdfmm"`
resolves to an installed module in that env's site-packages, so every Python
run below asserted `cdfmm.__file__` pointed at the just-built extension.

- portable CPU `dev` (fresh): clean build, 198/198 CTest, 137 pytest passed;
- oneMKL without CUDA: 198/198 CTest, 139 pytest passed;
- CUDA without oneMKL `cuda` (fresh, SM120): 198/198 CTest, 141 pytest passed;
- CUDA + oneMKL `notebooks` (fresh): **198/198 CTest with zero skips**, 143
  pytest passed;
- Fortran via `ifx` 2025.2.1: 199/199 CTest including `cdfmm_fortran_smoke`,
  and `magtense_style_demag` builds and runs.
- isolated install: 84 files, all 79 installed headers compile standalone,
  mixed legacy/canonical TU compiles in both include orders, downstream C
  consumer reproduces the analytic field to 1.6e-16;
- cache: a 10-file corpus written by a `v0.1.0` build is read back by HEAD with
  identical keys, hits everywhere, `bytes_written: 0`, corpus byte-identical
  afterwards, results bit-identical;
- docs: `sphinx-build -W` now succeeds with zero warnings (was two);
- performance: CPU 11.47 ms at HEAD vs 11.56 ms at `v0.1.0`; CUDA 0.96 ms
  partial, 0.88 ms full.

A cross-version probe compiled unchanged against both `v0.1.0` and HEAD public
headers and produced bit-identical results for five configurations. That is the
strongest single piece of behaviour-preservation evidence the session produced.

### Changes made

Four genuine Phase-1 defects, all narrow:

1. `tests/test_fortran_api.f90` — periodic cell centre set to `(0, 0, 0.5)`.
   **Pre-existing, not a refactor regression**: the file is byte-identical to
   `v0.1.0` and a `v0.1.0` build fails the same way. Undetected until now
   because no Fortran compiler was available in earlier sessions; `ifx` exists
   in the `magtense-env` Conda env.
2. `docs/Doxyfile` — `PREDEFINED` named `CDFMM_HOST_DEVICE`, used nowhere,
   while `plan/p2p/canonical.hpp` uses `CDFMM_PLAN_HOST_DEVICE`. Fixed; the two
   long-standing Sphinx warnings are gone.
3. `include/cdfmm/AGENTS.md` and `src/AGENTS.md` — `operators/`, `plan/`, and
   `src/plan/` were implemented but documented as future work.
4. `docs/architecture.md` — three passages still called the finished
   compatibility review "remaining Phase 1 work", contradicting the same
   document's own "Compatibility surface" section.

Also: `tests/AGENTS.md` and `python_tests/AGENTS.md` structure listings
completed, and Phase-1 closure recorded across `AGENTS.md`,
`docs/architecture.md`, and `agent_docs/`.

### Explicitly not done

Phase-2 work was recorded, not executed. The handoff list is in
`docs/architecture.md`. No release tag or branch was created.

## 2026-09-15 — compatibility/transitional-source review and current handoff

Starting HEAD `5cb5576` (`refactor(bindings): structure language adapters`).
The compatibility/transitional-source review is complete. Classification:

- retained public compatibility façades: all 29 flat headers under
  `include/cdfmm/`, each an installed public path at `v0.1.0` (verified with
  `git ls-tree v0.1.0 include/cdfmm/` against a whole-directory install glob),
  now each carrying an explicit reason-to-exist comment;
- removed internal shims: `src/cuda_fmm_plan.hpp`, `src/cuda_p2p_plan.hpp`
  (zero includers), `src/cuda_m2l_plan.hpp` (two includers retargeted to
  `cdfmm/backend/cuda/m2l.hpp`), and the dead `src/static_operators.cpp`;
- retained compatibility implementations: `src/operators.cpp` (eight
  one-expression delegations, backing the supported flat C++/Python operator
  names) and `include/cdfmm/operators.hpp` (an API adapter, not a forwarding
  umbrella — `docs/architecture.md` corrected);
- moved substantive implementation: `src/cuboid.cpp` is removed.
  `rectangular_prism_averaged_monomial` now lives in
  `src/geometry/primitives/rectangular_prism.cpp` and the authoritative pair
  tensor is `cdfmm::operators::p2p::build_pair` in `src/operators/p2p.cpp`.
  `cuboid_averaged_monomial` and `build_pair_tensor` remain as thin flat
  spellings. `CuboidSize` moved from `plan/direct/dense.hpp` to
  `geometry/primitives/rectangular_prism.hpp`.

Canonical code no longer includes compatibility façades. The only deliberate
exceptions are `src/operators.cpp`, which implements `cdfmm/operators.hpp`, and
`python/internal.hpp`, which must see the flat operator declarations the Python
module exports; both are commented. `cdfmm/timings.hpp` and `cdfmm/periodic.hpp`
are still included by canonical headers, but they are substantive public
headers with no subsystem home yet, not façades.

Validation evidence: portable `dev` fresh configure/build, CTest 198/198 with
expected skips #16, #55, #63, #67, Python 137 passed and 7 skipped (identical
to the bindings task). CUDA preset fresh configure/build, CTest 198/198 with
only the oneMKL-only skip, on RTX 5090, driver 595.84, CUDA 13.3.73 — a full
runtime regression, beyond the compile-only regression the scope required.
oneMKL+CUDA `notebooks` preset fresh configure/build, CTest 198/198 with zero
skips. C ABI unchanged at the same 14 `cdfmm_*` symbols with `c_api.h` and
`src/bindings/` untouched. An installed-prefix probe compiled separate
legacy-only and canonical-only consumers, and `nm` confirmed both spellings
resolve to identically typed symbols, with `build_pair_tensor` now defined in
`p2p.cpp.o` and `cuboid_averaged_monomial` in `rectangular_prism.cpp.o`.
Fortran was not exercised: no gfortran, ifx, or ifort available.

Environment note: LTO links fail with "Disk quota exceeded" unless `TMPDIR` is
moved off the shared `/tmp` tmpfs.

`tests/test_foundational_headers.cpp` gained `cdfmm/cuboid.hpp` and
`cdfmm/tensor_dictionary.hpp` in its compatibility include block, plus a case
asserting the legacy and canonical spellings of the moved mathematics agree.

Declined as out of scope and left for Phase 2: relocating `src/periodic.cpp`,
`src/parameter_selection.cpp`, and `src/validation.cpp`; giving the substantive
flat public headers subsystem homes; and the duplicate factorial helper shared
with `MultiIndexSet::factorial`.

Remaining Phase 1: whole-Phase-1 validation and closure. Phase 1 is not
claimed closed. Unrelated untracked `Article1/` and the
`tetrahedron_target_average_fair_sampling.ipynb` notebook remain preserved.

## 2026-09-15 — bindings boundary and current handoff

The bindings boundary is complete from implementation base/newest starting HEAD
`9003b666`; committed cache parent `9e111ee` is already preserved. The C ABI
implementation moved byte-identically to `src/bindings/c_api.cpp`, while
`include/cdfmm/c_api.h` and ABI version 1 remain unchanged. Fortran remains
`ISO_C_BINDING -> C ABI -> supported C++`. The former `python/bindings.cpp` was
replaced by `python/internal.hpp`, `module.cpp` (sole entry point), and
`core.cpp`, `geometry.cpp`, `tree.cpp`, `operators.cpp`, `direct.cpp`, and
`fmm.cpp`; CMake registers these sources. Binding code uses supported canonical
structured headers where available, only justified compatibility dependencies,
and no solver logic or internal backend headers.

The isolated `9e111ee8` manifest comparison found 63 top-level Python names,
with object kinds, docs/defaults, and member surface identical after
normalising object addresses. The C ABI dynamic symbol set was exactly the
same 14 `cdfmm_*` symbols before and after, and the moved C source was
byte-identical.

Validation evidence: portable affected/full build 73/73; CTest 197/197 with
four expected optional skips; Python 137 passed/7 skipped. The oneMKL
notebooks configuration built core/C/Python/tests; CTest 197/197 with three
CUDA skips; Python 139 passed/5 skipped. Under full access, the RTX 5090 was
visible with driver 595.84; a fresh CUDA build used CUDA 13.3.73 and explicit
SM120, affected targets built 107/107, CTest 197/197 with only the oneMKL-only
skip, and Python 141 passed/3 skipped (two oneMKL-only, one external MagTense).
Initial stale SM75 artifacts failed PTX on the 5090 and are not final results.
No gfortran, ifx, or ifort was available, so Fortran smoke/example checks were
not run. An independent temporary-prefix install and the diff/ownership audits
passed. `sphinx-build -W --keep-going -b html docs docs/_build/html` rendered
the documentation and exited nonzero only for the same two pre-existing
`docs/api.rst` C++ declaration warnings.

Remaining Phase 1: compatibility/transitional-source review, then whole-Phase-1
validation. Neither is claimed complete. Unrelated `Article1/` and notebook
changes remain preserved.

## 2026-09-15 — cache persistence subsystem

The flat `src/cache.cpp` is decomposed into `src/cache/`. Ownership is now
explicit: `internal.hpp` holds the shared constants, `CacheKind`/
`CacheDescriptor`, `CachePayload`, the `Writer`/`Reader` stream primitives, and
the io/format declarations; `io.cpp` owns cache root and environment policy
plus the validated file container, checksum, and atomic temporary-file write;
`format.cpp` owns the field-wise records for the persisted solver types;
`keys.cpp` owns identity, including SHA-256, the canonical 1e-9 coordinate,
compact grid/permutation recognition, and the key strings; `universal.cpp`
owns the translation-bank and periodic-root payloads; and `geometry.cpp` owns
the geometry-plan payload. `src/cache.cpp` is removed and `cdfmm_core` builds
the five new units. `src/cache/AGENTS.md` records the local invariants.

The moved code is byte-identical to the original apart from namespace
scaffolding and three deliberate `inline` additions (the shared format
constants, `checked_bytes`, `p2p_record_bytes`). The shared entities moved from
the file-wide anonymous namespace into the named `cdfmm::detail::cache`, which
was required: an anonymous-namespace type in a shared header is a distinct type
per translation unit. Single-consumer helpers stayed in per-unit anonymous
namespaces. No public API, `UniformFmm` layout, persistent format, or cache key
changed; `universal_cache_key()`, `geometry_cache_key()`, and
`periodic_cache_key()` are untouched.

Final validation handoff:

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

Prior old-cache compatibility evidence is preserved and is now backed by the
recovered 11-file corpus: both directions produced cache hits with zero writes
and identical manifests, and the repository's 522-file cache corpus remained
unchanged. An independent current probe also hit old universal/periodic files
without modifying their hashes. No implementation defect was found.

Documentation builds with warnings as errors still report only the two
pre-existing `docs/api.rst` declaration warnings.

The cache task is already committed as `9e111ee` with subject
`refactor(cache): structure persistence subsystem`.

At that earlier cache checkpoint, the bindings boundary, compatibility/
transitional source review, and whole-Phase-1 validation were next. The
bindings boundary is now complete; current remaining work is recorded at the
top of this file. Unrelated untracked `Article1/`
and `examples/simple_notebooks/tetrahedron_target_average_fair_sampling.ipynb`
remain preserved outside this task.

## 2026-09-15 — high-level UniformFmm cleanup closure

The final Phase 1 high-level FMM source decomposition is complete. The
`src/fmm` ownership is now explicit: `construction.cpp` owns geometry
normalisation and construction; `plan_preparation.cpp` owns immutable plan
creation, FP32 quantisation, and high-level cache calls; `execution_setup.cpp`
owns backend resolution/wiring and unchanged P2P policy; `evaluation.cpp`
owns the complete near/far lifecycle and timing, including hybrid CUDA
begin/CPU-far/finish and cancellation; `far_field.cpp` retains hierarchy
sequencing; `diagnostics.cpp` retains exact summary formatting;
`uniform_fmm.cpp` retains lifecycle, accessors, and inspection; and
`internal.hpp` declares opaque owner wrappers. The representative-position
geometry convention is documented: points have no extra primitive record,
while finite primitives store representative-relative data.

Final validation handoff: baseline hierarchy and legacy-home sanity checks
were clean; fresh portable configure/build completed 63/63; focused portable
coverage completed 39 cases (36 passed, three expected skips); full CPU CTest
completed 193/193 with expected skips #16, #51, #59, and #63; CUDA+oneMKL
rebuild completed 27/27 and full oneMKL CTest completed 193/193 with expected
CUDA skips #16, #59, and #63; CUDA 13.2 SM75 core/tests/Python build completed
27/27; and focused/full CUDA-configured CTest ultimately completed 193/193
with optional runtime skips after the unavailable-device state. GPU numerical
runtime was unavailable because `nvidia-smi` could not communicate with the
driver and the driver rejected CUDA 13.2 PTX, so no numerical device pass is
claimed. Diff and ownership audits were clean.

At that earlier high-level-FMM checkpoint, cache and bindings work had not yet
been applied; current remaining work is recorded at the top of this file.

## 2026-09-15 — Portable CPU backend decomposition closure

Completed and committed the accepted responsibility-driven CPU backend split.
The portable taxonomy is now `src/backend/cpu/{direct,p2p,m2l,far_field}/`:
P2P and near-field execution are under `p2p/`, prepared M2L under `m2l/`, and
prepared P2M/L2P entries plus shared level-scaled M2M/L2L translation under
`far_field/`. The far-field `executor.cpp` is a narrow mechanics boundary;
`src/fmm/far_field.cpp` retains reference mathematics, traversal, sequencing,
backend selection, and timing ownership. Direct execution was preserved.

`include/cdfmm/backend/cpu/{p2p,m2l,far_field}.hpp` now own the canonical
portable declarations. `static_plan_apply.hpp` remains a compatibility
umbrella and the mixed `static_plan_apply.cpp` implementation home is removed.
All existing P2P representations, FP32/FP64 paths, OpenMP thresholds and
scheduling, SIMD guards and widths, ordering, identity handling, and
reference-vs-static semantics are preserved. At that checkpoint the final
higher-level `UniformFmm` cleanup was the next architecture task; cache,
bindings, and repository pruning remained out of scope.

Validation handoff (trusted evidence): initial CUDA hierarchy sanity was clean:
the complete hierarchy was present, root `src/cuda_fmm.cu` and
`src/cuda_fmm_stub.cpp` were absent, and CUDA 13.2 was available. Fresh/full
portable CPU CTest passed 193/193 with four expected optional skips; repaired
focused CPU suites passed 13/13 and 34/34; standalone canonical/legacy header
probes passed. The oneMKL configure/core build and full CTest passed 193/193
with three expected CUDA skips using environment-specific `-latomic`.

Fresh CUDA SM75 configuration and `cdfmm_core`/`cdfmm_tests` build passed
198/198 steps, repaired CPU objects rebuilt, and focused CUDA-configured
coverage passed 13/13. No GPU numerical runtime claim is made. The
vectorisation-report configure/build passed and produced a nonempty
`cdfmm-cpu-backend.vec` (an earlier repair-worker attempt was network-blocked;
the independent tester later succeeded). `git diff --check` passed.

Commit: `refactor(cpu): decompose portable backend` (exact SHA reported at
handoff). Unrelated untracked `Article1/` and
`examples/simple_notebooks/tetrahedron_target_average_fair_sampling.ipynb`
remain preserved.

## 2026-09-14 — CUDA full-FMM backend ownership

Moved complete CUDA FMM declarations and orchestration to
`src/backend/cuda/fmm/{internal.hpp,plan.cu}` and moved non-CUDA full-FMM stubs
to `src/backend/cuda/stub/fmm.cpp`. The root `cuda_fmm.cu` and
`cuda_fmm_stub.cpp` implementation homes are removed; `cuda_fmm_plan.hpp` is
now a forwarding shim. Far-field shared P2M/L2P mechanics live in
`entries.cuh`, M2M/L2L mechanics in `translation.cuh`, and the far-field
internal header owns `CudaTranslationInteraction` without depending on the
full-FMM header. CPU decomposition remains the next backend task.

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

## 2026-09-14 — CUDA far-field execution extraction closure

The accepted CUDA far-field execution extraction is complete. Internal
`src/backend/cuda/far_field/{internal.hpp,executor.cu}` owns immutable FP32/
FP64 P2M/L2P entries, coefficient degrees, M2M/L2L matrices/interactions/
metadata, uploads/lifecycle/statistics, and kernels; there is no public API,
stub, or new library. `CudaFullPlan` retains changing moments/coefficient/
field buffers, permutations, P2P and separate M2L executor wiring,
streams/events/timing, near/far overlap, combination/reordering, and D2H.
Direct, P2P, and M2L remain authoritative in their existing backends. The
next task is orchestration/resource simplification; the `CudaFullPlan`
decomposition is not complete.

Validation handoff: initial M2L sanity ownership audit clean and focused
14-case probe passed; fresh CPU development configure/build 55 steps and full
CTest 193/193 with expected skips #16/#51/#59/#63; fresh CUDA configuration
with `module cuda/13.2` (`nvcc 13.2.78`) for SM75; the preset presented a
206-step graph and compiled the requested `cdfmm_core`, tests, Python
extension, `cdfmm-precompute`, and three benchmark targets, but failed at its
final install step on sandbox read-only Conda site-packages; explicitly
requested target builds then succeeded; focused CUDA CTest 30/30 with expected
unavailable-device skips #16/#59/#63; full
CUDA-configured CTest 193/193 with expected skips #16/#51/#59/#63;
`git diff --check` and independent code/ownership review passed. `nvidia-smi`
could not communicate with the driver, so no numerical GPU runtime execution
is claimed. Unrelated `Article1/` and notebook changes remain untouched.

## 2026-09-14 — CUDA M2L backend extraction closure

The accepted CUDA M2L ownership split is complete. The canonical public
`CudaM2LPlan` declaration is `include/cdfmm/backend/cuda/m2l.hpp`, with
`src/cuda_m2l_plan.hpp` retained as a forwarding compatibility shim.
`src/backend/cuda/m2l/{internal.hpp,plan.cu}` owns the reusable FP64/FP32
device representation, kernels, bounded scaled-multipole scratch policy,
persistent resources, lifecycle, and statistics. Standalone `CudaM2LPlan` and
`CudaFullPlan` consume the same internal executor; `src/cuda_fmm.cu` retains
P2M/M2M/L2L/L2P and full-FMM/far-field orchestration. The non-CUDA boundary is
`src/backend/cuda/stub/m2l.cpp`.

Validation truth: fresh CPU configure/build 194/194; fresh CPU full CTest
193/193 with expected skips #16/#51/#59/#63; fresh CUDA 13.2 SM75 build
204/204 covering core, tests, Python, tools, and benchmarks; focused CUDA
M2L/header/stub coverage 23/23 with unavailable runtime-device cases skipped
as expected; full CUDA CTest 193/193 with the same four skips; and install,
ownership, complete-diff, and `git diff --check` audits passed. `nvidia-smi`
could not reach a driver/device, so no numerical GPU runtime result is
claimed.

The untracked `Article1/` directory and unrelated
`examples/simple_notebooks/tetrahedron_target_average_comsol_compare_sampling_and_gauss.ipynb`
remain untouched and outside this task.

## 2026-09-14 — complete CUDA P2P backend extraction closure

The complete CUDA P2P extraction was accepted as the preceding step and is
included in the current CUDA backend series. `include/cdfmm/backend/cuda/p2p.hpp` is the canonical public header
for `CudaP2PPlan`; `include/cdfmm/cuda_p2p.hpp` remains a legacy forwarding
façade. `src/backend/cuda/p2p/{internal.hpp,plan.cu}` owns all list-1 CUDA P2P
variants and lifecycle/state: canonical AoS, compact/source-only SoA,
leaf-block, signed tensor dictionary (source-warp, target-owned, and
power-of-two microtiles), and cuSPARSE BSR(3), in FP64 and FP32. The package
also owns shared full-plan primitives and the non-CUDA path is
`src/backend/cuda/stub/p2p.cpp`. At that checkpoint, `src/cuda_fmm.cu`
retained M2L and far-field/full-FMM orchestration and consumed P2P internal
primitives; the accepted M2L extraction is recorded above.

Validation truth for this closure:

- fresh CPU full CTest: 192/192, with four expected skips;
- fresh CUDA 13.2 SM75 full build: 206 steps covering core, tests, Python,
  examples, tools, and benchmarks;
- CUDA-configured full CTest: 192/192;
- runtime numerical GPU validation unavailable because `nvidia-smi` could not
  communicate with the driver; and
- ownership, `nm`, and `git diff --check` audits clean.

The untracked `Article1/` directory and unrelated
`examples/simple_notebooks/tetrahedron_target_average_comsol_compare.ipynb`
were preserved untouched and are outside this task. Its focused validation
was recorded before the combined CUDA M2L closure.

## 2026-09-14 — CUDA direct backend extraction closure

The accepted CUDA direct extraction is complete. The
canonical public headers are
`include/cdfmm/backend/cuda/{direct,dense_direct}.hpp`; the old
`include/cdfmm/cuda_direct.hpp` and `include/cdfmm/cuda_cuboid.hpp` paths are
compatibility façades. Shared internal error/runtime handling is in
`src/backend/cuda/common/`, point O(N^2) and dense cuBLAS direct execution are
in `src/backend/cuda/direct/`, and non-CUDA direct stubs are in
`src/backend/cuda/stub/direct.cpp`. At that earlier checkpoint,
`src/cuda_fmm.cu` still owned P2P, M2L, and full-FMM execution. No
behaviour/performance change was intended and no P2P/M2L/full decomposition
had yet been attempted.

Validation handoff: fresh dev configure/build 59 targets; focused
direct/header/stub coverage 8/8, including the exact disabled-stub behaviour
case; full CPU CTest 189/189 with expected skips #16/#51/#59/#63; fresh CUDA
13.2 configure with explicit arch 75 and cached Catch2 source; and a full
198-target CUDA build covering tests, Python, and benchmarks. Independent full
CUDA CTest reported 189/189, but CUDA numerical/runtime
runtime cases only took unavailable-device guards because `/dev/nvidia` was
absent and `nvidia-smi` could not reach the driver. Ownership/`nm` searches
and `git diff --check` were clean. `Article1/` remains untouched.

At that earlier checkpoint, no implementation follow-up was implied beyond the
then-deferred P2P/M2L/full-FMM CUDA decomposition.

## 2026-09-11 — FMM, CPU, and oneMKL execution boundaries

The two-package FMM boundary refactor is complete. Commit `f8218fc` moved
`UniformFmm` and far-field orchestration under `src/fmm` and CPU list-1
execution under `src/backend/cpu`. The follow-up oneMKL package places grouped
M2L preparation, reusable FP32/FP64 scratch, SGEMM/DGEMM, thread-local MKL
control, serial scatter, storage accounting, phase timing, and availability in
`src/backend/mkl/m2l.{hpp,cpp}`. `src/fmm/far_field.cpp` retains the expansion
pass order and timing aggregation, while an opaque internal owner removes the
group/scratch layout from `include/cdfmm/uniform_fmm.hpp`.

Fresh portable construction passed 52/52 build steps. Focused portable checks
completed 21 cases (20 passed, one expected oneMKL skip), and full portable
CTest completed 184 cases (180 passed, expected skips #16/#51/#59/#63). The
CPU+oneMKL build reconfigured and rebuilt successfully; focused M2L, spherical,
cache, and precision checks completed 50 cases (48 passed, CUDA skips #16/#59),
and full oneMKL CTest completed 184 cases (181 passed, CUDA skips
#16/#59/#63). Source-level scaling, mixed-level stable grouping, persistent
scratch accounting, phase timing, repeated execution, and move semantics have
focused regression coverage. Ownership, symbol, source-duplication, and diff
audits passed.

CUDA runtime, Python, Fortran, and documentation builds were not run. No CUDA
source was changed. The untracked `Article1/` directory and unrelated parent
MagTense work remain untouched.

## 2026-09-11 — dense-direct plan/backend separation

The dense-direct plan/backend separation is complete: the canonical declaration
is `include/cdfmm/plan/direct/dense.hpp`, preparation is in
`src/plan/direct/dense.cpp`, portable execution is in
`src/backend/cpu/direct/dense.cpp`, and oneMKL execution plus availability is
in `src/backend/mkl/direct/dense.cpp`. A private shared workspace keeps FP32
and FP64 staging arrays reusable without exposing backend state in the public
header. Explicit deep-copy and move members preserve prior value semantics and
ensure copies do not share mutable workspace. The legacy
`include/cdfmm/cuboid.hpp` path remains a compatibility umbrella; cuboid math
and pair tensors remain in `src/cuboid.cpp`.

Trusted validation for this completed step:

- fresh dev configure/build: 66/66;
- focused geometry/direct/precision CTest: 28/28 passed;
- full portable CTest: 183/183 passed with expected skips #16/#51/#59/#63;
- CPU+oneMKL reconfigure/build and focused direct/cuboid/oneMKL/precision
  CTest: 31/31 passed;
- full oneMKL CTest: 183/183 passed with expected skips #16/#59/#63;
- canonical-only and legacy-header compile probes remain covered by the public
  header suite; and
- `git diff --check` and backend ownership searches passed.

CUDA runtime validation was not part of this CPU/oneMKL subphase. `Article1/`
and unrelated parent-worktree changes remain untouched.

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
