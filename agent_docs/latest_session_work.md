# Latest session work

## 2026-09-23 — Article1 preparation: four defects found while benchmarking, position-based point plans

Starting HEAD `aa9d75f` (the frozen Phase-5 SHA). Branch
`article1-benchmark-fixes`. The Article1 campaign exposed four defects and
one construction limit; each was fixed at the lowest layer with a regression
test that fails on `aa9d75f`. No public API, C ABI, Python API, Fortran
interface or cache format changed; the one cache-key change is additive (below).

**1. Explicit CPU `p2p_packing` lost to the regular-grid hint.** On
`CpuStatic`, `CanonicalAos`, `ParticleRowSoa` and `PointGeometry` requests
returned from `apply_p2p_packing_request` without being recorded, so the
`RegularGrid` layout rule still selected, derived and then preferred the
signed dictionary (resolved `TensorDictionary` for a `CanonicalAos` request).
The requests now set `explicit_packing`, which the resolver honours verbatim,
as the public header already promised. `docs/backends.md`'s policy table also
listed CPU point pairs above the layout rule; the order now matches the code.

**2. `benchmark_uniform_fmm` measured warm CUDA builds as cold.** The CUDA
runtime warm-up construction, and every workload-comparison construction,
used the default cache, so an empty `CDFMM_CACHE_DIR` was filled before the
timed construction read it. Both are now cache-free. Cold `CudaFull` setup of
a 32^3 lattice is 3.97 s against 3.70 s on `CpuStatic`; warm 0.47 s.

**3. Finite-body accuracy used a point-dipole reference.** The driver's
`--direct`/`--accuracy-targets` compared exact finite near fields with a
point-dipole sum. Finite bodies now use an FP64 `DenseDirectPlan` reference
built in blocks (exact pair tensors, physical self terms). Order 4 on a
32^3 prism lattice: 2.4e-3 RMS.

**4. The dense exact-reuse gate could never see reuse above 32768 sources.**
`DenseDirectPlan` classifies pairs target-major and decided from the first
65536 pairs; one target row has only distinct displacements, so with
`N_s >= 32768` the sample held at most two rows and classification was always
abandoned, even on a perfect lattice. The sample now spans at least sixteen
target rows (`max(65536, 16 N_s)`, identical below 4096 sources, so Phase-3C.5
numbers are unaffected). 512 targets x 32768 prisms: 271 s -> 3.5 s,
bit-identical by construction.

**5. `PointGeometry` plans no longer build stored pair tensors.** Recorded as
Phase-3C lead 7 in `performance_optimization.md`, now resolved there: a plan
whose policy guarantees `PointGeometry` before preparation skips the pair
list, canonical operator, compact rows and speculative FP32 BSR, counts
`p2p_interactions` from the leaf records, and keys its geometry file with
`_p2p_positions_` plus a conditionally hashed `"POSG"` marker. Stored-tensor
keys were verified byte-identical against the `aa9d75f` module. 64^3 points
at 64 per leaf: 66.6 GB / 63.8 s -> 1.46 GB / 2.4 s; 512 per leaf now fits in
about 1.3 GB (CPU) and 2.0 GB (CudaFull FP32 host).

**Validation (this machine, E-cores 16-31, concurrent single_grain job).**
CUDA + oneMKL + OpenMP Release build (`build-all`, g++ 15.3, nvcc 13.3,
RTX 5090): CTest 260/260; `python_tests` 186 passed, 1 skipped, module
imported from `build-all` via `PYTHONPATH`. Each new test was re-run against
the reverted source and fails there. The portable CI configuration (GCC 13)
and the Fortran interface were not built locally.

**Measured, not changed.** With packings honoured, the dictionary reduced the
P2P operator 20-29x (host) and total device memory 5-11x, but was only
1.2-1.3x faster than the SoA rows on the CPU; an explicit dictionary request
without an executor flag runs the source-warp kernel, 1.9x slower than leaf
blocks for FP64 at 8 per leaf on CUDA (the documented caveat). A CPU point
lattice with a fixed identity map still selects the dictionary automatically,
and here it was 1.4x slower than `PointGeometry`, which contradicts the
Phase-3D ledger; it needs a P-core re-measurement before any policy change.
Host memory still holds the canonical and row operators beside the dictionary,
because the free-space potential path reads them.

## 2026-09-21 — Phase 5 closure follow-up: residual construction clocks, benchmark semantics, C accessor

Starting HEAD `d691830` (tip of `phase5-timing-freeze`, the agent-record
commit above the first freeze at `62835b5`; CI green on `5cdfaa8`). Narrow
follow-up; nothing in mathematics, policy, representation, cache format or
keys changed. Commits: `60a361f` timing gates (tree and dense construction clocks by level, tests), `985ef6b` benchmark semantics (external versus internal columns, `internal_timing_source`, Python test, benchmarks guide), `d989270` documentation (C accessor compatibility status, tree clocks), `48c2142` a comment correction in the tree build.

**Audit first, then measurement, then the change.** Every `steady_clock`
read left in the library was inventoried: `UniformTree::build` (13 per tree,
26 per `UniformFmm` plan, unconditional), `AdaptiveTree` (3, standalone
only), `DenseDirectPlan` (8-10 into a thread-local internal record) and
`CudaDenseDirectPlan` (8 more), plus one functional temp-file stamp in the
cache writer. A hard-off copy of `d691830` with those clocks deleted was
timed against the branch with an external clock around complete
constructors (7 interleaved rounds, medians of medians). The clocks cost
about 17 ns per read: resolvable only on a 6 us standalone tree (+3.8 %),
0.1 % on a 415 us plan, inside a 1-7 % round spread on every other plan and
dense build. Small, but the Article1 property is that `Off` means no
diagnostic clock in the benchmarked path, and the gates are small, so they
were added where a plan owns the clock or a benchmark constructs the object.

**What changed.** `UniformTreeOptions::collect_build_timings` (default
`true`, Python-exposed): standalone trees keep filling `build_timings()`;
`UniformFmm` sets it to `timing_level == Detailed` on both trees it builds,
so `Off`/`Coarse` plans read no tree clock and `Detailed` still fills
`tree_construction` (`calls == 2`). `DenseDirectPlan` and
`CudaDenseDirectPlan` take a trailing `TimingLevel timing_level = Off`
(`Coarse` = total, `Detailed` = every phase; counts and bytes always);
the construction benchmark, the only reader of the records, passes
`Detailed`. `AdaptiveTree` (3 reads, under 0.1 us, not plan-owned) stays as
it is and is documented. `benchmark_uniform_fmm` no longer copies its own
clock into `EvaluationTimings` for the direct references (the CPU reference
has no collector; the CUDA direct plan contributes its lanes only when its
record says `Detailed`) and gains a trailing `internal_timing_source` column
(`uniform_fmm` / `cuda_direct_plan` / `none`); headline columns are external
at every level, internal columns follow the level. The C accessor's
behaviour change is now documented as ABI-compatible (`CDFMM_ABI_VERSION`
1), source-compatible and intentionally different (opt-in timing); the C test
covers `Off` -> unsupported, `Coarse`/`Detailed` -> measured, back to `Off`
-> unsupported.

**Re-measurement.** Same protocol with the final build added at `Off` and
`Detailed`: the 6 us tree returns to hard-off to 0.01 us (5.75 vs 5.76 us;
`Detailed` +0.18 us); every plan row within spread; single-threaded dense
point plans (`OMP_NUM_THREADS=1`, 10 rounds x 4000 builds) put `final` at or
below hard-off on 15, 98 and 1181 us builds. Record:
`benchmarks/baselines/phase5-timing/construction_clocks_{before,final,serial}.csv`
and the README there; narrative in `agent_docs/performance_optimization.md`,
Phase 5, section M.

**Tests.** `test_uniform_tree.cpp` (opt-out changes no node, list,
permutation, leaf map or sorted position); `test_timing_levels.cpp` (tree
clocks off at `Off`/`Coarse`, every tree phase once at `Detailed`;
`DenseDirectPlan` bit-identical matrices and results across levels with the
record gated per path; `CudaDenseDirectPlan` record gated, level forwarded,
results agree); `test_c_api.cpp` extended; `python_tests/
test_construction_timing.py` (Python option, `tree_construction_seconds`
zero at `OFF`, and the driver's CSV semantics when a built
`benchmark_uniform_fmm` is found, skipping otherwise).

**Validation.** Four warning-as-error configurations rebuilt, zero warnings each: portable CPU CI reproduction (conda-forge g++ 13.4, Unix Makefiles, LTO) 255/255 CTest, 178 passed / 9 skipped pytest; CPU + oneMKL without CUDA 255/255, 167 passed / 7 skipped (notebooks excluded); CUDA without oneMKL (`cuda` preset) 255/255; CUDA + oneMKL (`notebooks` preset plus benchmarks, g++ 15.3 / nvcc 13.3) 255/255, 186 passed / 1 skipped including the six executed tutorials and the benchmark-CSV tests against the rebuilt driver. The three new CTest cases are the tree opt-out and the two dense timing cases. `compute-sanitizer` memcheck, racecheck, initcheck and synccheck over the `[dense]` and `[timing]` groups (12 cases, 9 643 assertions), since the CUDA dense constructor changed: 0 errors, 0 hazards each; no other CUDA source, event or synchronisation code changed, so the Phase-5 sanitizer campaign was not repeated. `sphinx-build -W` clean, `git diff --check` clean. Unvalidated, as before: Fortran (no compiler here; `fortran/` untouched), MSVC/Windows.

**Implementation is FROZEN FOR ARTICLE1 BENCHMARKING** at
`48c2142` (supersedes `62835b5`). CI: green on `d989270` (run 35578346697, both jobs: portable CPU build and tests, first-party warning surface); the run on the final HEAD that adds this record is listed below. Next:
fast-forward `phase5-timing-freeze` into `refactor/architecture-v0.2` after
review, then the Article1 benchmark campaign with `timing_level = Off` and
external wall-clock timing of repeated `evaluate_into` calls.

## 2026-09-21 — Phase 5: opt-in timing levels and the implementation freeze

Starting HEAD `8f54e6f` (tip of `phase4-pruning`); work on
`phase5-timing-freeze`, based exactly on it. Seven commits: `62e4ce8` CUDA event separation, `776d26a` UniformFmm timing levels, `fa870c6` Python/C ABI, `244b341` benchmark drivers and the overhead record, `7688d6a` tests, `3fb36d2` examples, `62835b5` documentation.

Question answered: what does the always-on instrumentation cost, and can the
production path be made free of it without touching results, policy, cache or
the functional CUDA synchronisation?

**It was measured before anything was redesigned.** A temporary "hard-off"
copy of the starting HEAD had every diagnostic `cudaEventRecord`, every
`cudaEventElapsedTime` and every host evaluation clock removed, with the three
functional events (the two cross-stream waits and the completion point) kept
and created without timestamps. Against the untouched HEAD, under one external
clock and five interleaved repetitions, the instrumentation cost a `CudaFull`
evaluation 12-20 us whatever its size: 15 % of an 86-108 us evaluation,
3.3 % at 50k points, 20 % of the fastest standalone P2P plan, and nothing
measurable on the CPU (the ~14 host clock reads are below 0.3 % of a
millisecond). The RTX 5090 run-to-run spread was about 1 us, so the numbers
were not in doubt.

**What changed.** `TimingLevel { Off, Coarse, Detailed }` with `Off` the
default; `UniformFmmOptions::timing_level`, `UniformFmm::timing_level()`
and `set_timing_level()`; every timing record now names the level it was
collected at. `Off` reads no clock and records no diagnostic event; `Coarse`
is host wall times only (`total`, the new `far_field`, `p2p`,
`cuda_p2p_wait`, `total_setup`, static-plan `total`); `Detailed` is the old
depth. The CUDA plans got a *separate* diagnostic event graph — the three
functional events have timing twins — so the functional graph is identical at
every level and nothing is recreated when the level changes. Host clocks are
one `detail::PhaseStopwatch` per level (`src/phase_stopwatch.hpp`), the
MagTense idea that the verbosity gate precedes the clock. Python exposes
`cdfmm.TimingLevel` and the setter; the C ABI gains the additive
`cdfmm_plan_set_timing_level`, and its wall-time accessor now fails while a
plan is `Off` instead of returning a zero that looks like a measurement.
Fortran is untouched (it exposes no timing). NVTX is untouched and independent.

**Acceptance.** `Off` sits within 0.1-0.6 % of hard-off on every FMM workload
(inside the spread; the fastest CUDA cases agree to 0.1 us); `Coarse` costs at
most 0.3 %; `Detailed` reproduces the old cost, now quantified: +6 % to +18 %
on the sub-millisecond CUDA cases, +4 % at 50k FP32, nothing on the CPU. One
Phase-J row (standalone leaf-block P2P, Off 1.12 vs Coarse 0.97 on the same
binary and device path) was noise in a minimum-of-50 at 40 us; re-measured
alone with ten repetitions, Off and hard-off agree to 0.3 us there too. The
record is `benchmarks/baselines/phase5-timing/` (INTERNAL TIMING OVERHEAD
STUDY, NOT ARTICLE1); `baselines/phase3d/` is unchanged and its runner now
requests `--timing detailed` so later runs compare like with like.

**Benchmark drivers** take `--timing off|coarse|detailed` (default `off`)
and record the level in every row; the headline `evaluation_median` was and
is the driver's own external clock. `run_benchmarks.py`, the Phase-3D
runner, `benchmark_cache_initialisation`, the operator-representation driver,
the parameter-selection search, the adaptive showcase and three tutorials ask
for `Detailed` explicitly because they *display* phases.

**Validation.** Four warning-as-error configurations, zero warnings each.
CI reproduction with conda-forge g++ 13.4 (Unix Makefiles, LTO): 252/252
CTest, 173 passed / 9 skipped pytest. CPU + oneMKL without CUDA: 252/252,
162 passed / 7 skipped. CUDA without oneMKL: 252/252. CUDA + oneMKL
(`notebooks` preset + benchmarks): 252/252, 181 passed / 1 skipped with the
six tutorials executed. `compute-sanitizer` memcheck, racecheck, initcheck and synccheck over the 35 CUDA- and timing-tagged C++ test cases (41 312 assertions): 0 errors, 0 hazards each.
`sphinx-build -W` clean, `git diff --check` clean. New
`tests/test_timing_levels.cpp` (and the Python twin) pins: Off collects
nothing, Coarse only the coarse fields, Detailed everything, aggregates and
reset, run-time level changes leave results/policy unchanged, cache keys and
*persisted bytes* identical across levels, CUDA lanes gated and results
unchanged across levels. Cache format and keys unchanged; C ABI additive
(`CDFMM_ABI_VERSION` 1). CI: green on `5cdfaa8` (run 35571431509), both jobs and every step, including the C++ and Python test steps. The run on `62835b5` shows as cancelled because the workflow sets `cancel-in-progress` per ref and the documentation push superseded it; `5cdfaa8` contains that commit's tree plus the agent docs.

**Unvalidated, as before:** Fortran (no compiler here; `fortran/` is
byte-identical to the start), MSVC/Windows.

**Implementation was frozen for Article1 benchmarking** at `62835b5`; that
freeze is superseded by the closure follow-up above, which gated the residual
construction clocks and records the final production SHA.

## 2026-09-20 — Phase 4 repository pruning and documentation cleanup

Starting HEAD `ad48459` on `refactor/architecture-v0.2`; work on
`phase4-pruning`. Eight commits: `3244036` source, `7745f82` tests,
`32afc23` + `2daa833` notebooks and their regression, `242f708` + `c7bcc6c`
documentation, `fc2e806` + `da08ed3` comments.

Question answered: can a reader who did not live through Phases 1-3 open this
repository and understand it — without losing anything Article1 will need?

The governing rule was inverted from the usual one. Anything *executable* was
kept by default and only duplication was removed, because a benchmark that no
longer has an alternative to measure against is worthless. So the backends,
the six P2P packings and three dictionary executors, both point-expansion
representations, both bases, both precisions, every explicit override and
every driver under `benchmarks/` are exactly as Phase 3D left them; the
ledger's section 0 lists them as untouchable and it was honoured.

**What the audit actually found.** Three read-only `repo-auditor` passes
produced the C++ test ledger, the Python/notebook mapping and a reference
audit of the five symbols Phase 3D had deferred. Four of the five were dead
by reference count and went; the fifth — `cuda_policy::resolve_cuda_execution_
policy` deciding the CPU dictionary too — was a naming complaint, not a
defect, so it got a comment instead of a rename across a public-adjacent
surface. The one merge that looked worth doing was the endpoint classifier:
it and `classify_exact_operators` are the same first-seen-numbering algorithm
with the same 32-bit abandon guard, differing only in key container and
sampling gate, so the shared one was made generic over the key type. It
passed everything locally — four configurations, 244 CTest cases, a 25-file
cache corpus hashed before and after to prove the persisted plans did not
move — and then broke CI at link time under GCC 13 (see below). It is
reverted; the duplication is now a recorded, explained decision instead of an
unexamined one.

**The 496-second test.** The procedural-expansion suite dominated CTest, and
the reason was instructive: it proved a *bitwise* property (procedural
kernels reproduce the precomputed rows) using 2000-point plan comparisons,
so it paid for an order-10 universal operator bank four times and bought no
coverage that an 11-point leaf does not. Replacing it with a kernel-level
comparison at every compiled order 1-10, plus a plan-level loop over {1, 3,
6} that absorbed the potential test, took the file from 496 s to 7.5 s.
Separately, moving `resolve_point_expansion_execution()` ahead of
`build_static_plan()` made the validation test reject an impossible request
before building an order-11 bank: 178 s to 0.34 s. Serial CTest: 550 s to
44 s, same 244 cases.

**Notebooks were the largest single reduction.** 29 of them, in three
directories, of which the README listed 22, three had no Markdown at all, one
called a removed API and one never imported `cdfmm`. Six tutorials replace
them. Writing the tutorials found three teaching errors worth recording:
lattice spacings that divide the root box put every particle on an octree
corner and made the honest error numbers look terrible (fixed with offset
roots); a uniformly magnetised periodic cell does **not** give `H = 0` for
point dipoles but the Lorentz field `+M/3` (cubes filling the cell give 0,
and both are now verified numerically in the notebook); and the FP64 accuracy
floor at depth 1 is `6e-9`, set by the 1e-9 canonical coordinate grid, not
the `1e-15` that had been claimed. `python_tests/test_tutorial_notebooks.py`
now *executes* every retained tutorial with `nbclient`, which is strictly more
than the old contract tests did, and the notebooks that no longer exist no
longer have tests.

**Documentation.** The 85 KB `architecture.md` was roughly 70 % closure
narrative, and `benchmarks.md` about 35 % historical result tables. Both were
rewritten to describe the current library, with the history moved verbatim
into `agent_docs/architecture_history.md` — nothing was summarised away,
because a closure record is evidence. The mathematics was split into
`docs/math/` rather than compressed: conventions, spherical expansions and
finite geometry now hold every normative statement that was spread across
six pages, including the root normalisation, the FP32 root-width scaling and
the periodic Ewald construction.

**Comments.** The public headers carried 430 undocumented members, which made
the generated API reference close to useless; they now carry none. Under
`src/`, the additions explain the things that cost the most to re-derive: the
CUDA full-plan event graph and why its phase timings must not be summed, the
identity marker that every P2P packing carries instead of inferring identity
from geometry, why exact operator reuse compares bits and never tolerances,
how the cache container publishes atomically so concurrent writers race
safely, and the two prism/tetrahedron formulations with their singular-limit
safeguards.

**Validation.** Four fresh configurations, each with
`-DCDFMM_WARNINGS_AS_ERRORS=ON`: portable CPU, portable + oneMKL, CUDA, and
CUDA + oneMKL. Every one built with zero warnings and passed CTest 244/244;
pytest passed 170/172/175/177 as the optional backends became available, with
only the MagTense-environment skip left in the last. `sphinx-build -W` is
clean and `git diff --check` is clean.

**The cache question, answered properly.** The corpus keys were identical
before and after, but 8 of 25 files differed in content, all of them carrying
FP64 operator values. Rather than assume, three controls were run: the same
binary twice (byte-identical, so it is deterministic per binary), the start
commit built from a clean archive (same eight differ), and — decisively —
**the same HEAD source built in a second tree** (the same eight differ
again). So the difference is build-to-build variation of the LTO Release
build, not a Phase-4 change. Its size was then measured rather than assumed:
1 ULP on `H`, 2 ULP on `phi`, root multipole bitwise identical. Worth
recording because it means a shared cache file is portable between builds but
not bit-reproducible across them.

**Performance.** A 41-case subset of the Phase-3D regression matrix
(`A-random/S`, `B-lattice-low/occ8`, `D-prism/regular`, covering point and
finite geometry on all four backends, both precisions and both layout hints)
was rerun from a fresh `benchmark-all` tree under the baseline's pinning and
protocol. Every resolved representation column and every retained-byte column
matches `benchmarks/baselines/phase3d/` exactly — 0 policy mismatches, 0 byte
mismatches in 41 cases — and repeated-evaluation medians are 0.98-1.03 of
baseline. The `p2m + l2p` sum swings more widely (0.63-1.62) because it is
33-300 us against 0.2-13 ms totals; the totals it sits inside agree within
1.5 %.

**The CI failure, and why it took controls rather than a guess.** The first
push failed both CI jobs at the build step. The workflow's annotations showed
only `collect2: error: ld returned 1 exit status`, because its grep matches
`: error:` and GCC writes `lto1: fatal error:`; the logs endpoint needs a
token this environment does not have. Reproducing CI's configuration locally
with the conda GCC 13.4 toolchain (Unix Makefiles, LTO, `-Werror`, unlimited
`-j`) reproduced it immediately and produced the real message, `multiple
prevailing defs for 'allocate'`, and a peak RSS of 394 MB that ruled out the
obvious memory explanation. Building the start commit the same way passed, so
the regression was the branch's. The attractive theory — the archive has six
duplicated member basenames, because `ar` stores basenames and the tree has
several `dense.cpp`/`p2p.cpp`/`geometry.cpp` — was tested by rebuilding the
archive with unique names through `gcc-ar` and **refuted**: it fails
identically. Linking with `-save-temps` kept the plugin's resolution file,
which named fifteen `std::vector`/`std::array` COMDATs marked
`PREVAILING_DEF` in two members at once, and mapping those archive offsets
gave `src/plan/direct/dense.cpp.o` and `src/fmm/plan_preparation.cpp.o` — the
two translation units the merge had just made share a header. Reverting only
that merge (five files, now differing from the start commit in comments only)
builds clean under GCC 13, and the whole local matrix was rerun on the
reverted tree.

**Unvalidated, as before:** Fortran (no compiler in this environment;
`fortran/` is byte-identical to `ad48459`), MSVC/Windows, and CUDA
sanitizers, which were not rerun because no CUDA code changed beyond comments
and three reformatted statements.

## 2026-09-19 — Pre-pruning closure after Phase 3D

Starting HEAD `51b2434` on `phase3d-final-integration`; work on
`phase3d-pre-pruning-closure`, based exactly on it. **Not Phase 4**: nothing
was pruned, no obsolete file was deleted and no benchmark was redesigned.

Question answered: is the refactored implementation clean and mechanically
trustworthy enough that Phase-4 pruning can start from it?

Answer: it is now, but two things had to be repaired that the Phase-3D review
did not cover. The project set **no compiler warning flags at all**, so ~115
`.cpp` and 8 `.cu` files had never been read by `-Wall -Wextra`; and GitHub
Actions had been failing on **every** branch since 2026-09-07, including the
integration branch and the Phase-3D head.

**No production policy changed.** All fourteen automatic policies stand as
Phase 3D measured them. The only behavioural line added to `src/` is a `break`
after a call that already throws.

**The four review findings.** The analyser derived its comparison group by
position from the display name, which put a regular prism lattice and an
irregular prism cloud in one group and divided them by whichever automatic row
came first; cases now record `comparison_group` and `comparison_variant`, and
the retained baseline regroups 21 into 23 without being regenerated, because
the analyser reconstructs legacy groups from the driver's own generators. A
failed case returned `None` and the driver exited zero with a short CSV; it
now raises, names the case, checks the row count against the defined matrix
and fails. Resume reused any non-empty file, silently mixing commits and
binaries; it is now fresh by default and guarded by a scratch manifest of the
revision, the binary's SHA-256 and the sampling settings. And both Phase-3D
records claimed the integration *branch* still pointed at a pre-3C SHA, when
it was this checkout's local ref that lagged — the remote had already been
advanced, and the reflog shows a fast-forward pull.

**Warnings.** `cdfmm_enable_warnings()` applies the project warning set per
first-party target at the same twelve call sites as `cdfmm_enable_ipo()`, so
Catch2 and pybind11 are excluded by construction. Thirty-three diagnostics,
all first-party, all fixed at the source; no suppression of any kind was
added and no warning level was lowered. The substantive ones were an unroll
pragma keyed on `__CUDACC__` (so the host half of every `.cu` parsed a pragma
g++ does not know), an unreachable switch fall-through after a throwing
`reject()`, a dead helper and two dead edge vectors in the tetrahedron
primitive, and five by-value structured bindings that only the CI compiler
flagged.

**CI.** The pre-existing failure was `pytest` failing at *collection*: two
notebook tests import `nbformat` and the workflow's hand-written dependency
list had drifted away from what the tests import. `nbformat` now sits in the
`test` extra and CI installs `.[test]`; in a CI-faithful interpreter that
turns the collection error into 163 passed and 8 skipped. Since run logs need
repository rights, the workflow was first taught to re-publish its own
diagnostics as annotations, and every later fix was read from the runner's
output rather than guessed.

Both CI jobs are green on the closure head `bbe3926` -- the first green run
on this repository since 2026-09-07 -- with wider coverage than before, since
the benchmarks are now compiled and the two notebook tests actually run.

Clean `-Werror` builds pass in portable CPU, CPU + oneMKL, CUDA, CUDA +
oneMKL and `Debug` under g++ 15.3.0, and in portable CPU under conda g++
13.4.0. The **Fortran interface could not be built — no Fortran compiler is
installed here** — and MSVC was not exercised; `fortran/` is byte-identical to
`51b2434` and the warning helper cannot reach a Fortran target, so the
interface is unchanged by construction rather than by test.

On the final tree: `ctest` 244/244, `pytest python_tests` 171 passed and 1
skipped, `git diff --check` clean. A representative CUDA sanitizer matrix
(memcheck over six cases, plus racecheck, initcheck and synccheck) reports no
errors and no hazards; this closure changes no device code, since the device
pass still expands `#pragma unroll` exactly as before.

`git diff 51b2434..HEAD -- include/ src/bindings/ src/cache/ python/ fortran/`
is empty: no public API, C ABI, Python API, Fortran interface, cache format or
cache key changed. The retained Phase-3D baseline is unchanged at 158 rows.

Full detail, including the build matrix and the Phase-4 inventory, is under
"Pre-pruning closure after Phase 3D" in `performance_optimization.md`.

## 2026-09-18 — Final cross-backend integration and production policy (Phase 3D)

Starting HEAD `49b5fe3` on `phase3c5-dense-construction`; work on
`phase3d-final-integration`, based exactly on it. A note recorded here at the
time said `refactor/architecture-v0.2` still pointed at `a2af367`. That was a
stale *local* observation rather than the state of the branch: the remote
integration branch had already been advanced through Phase 3C/3C.5 to
`49b5fe3`, and only this checkout's own ref lagged behind. The reflog records
the local ref catching up by fast-forward pull, not by a push. `a2af367` is an
ancestor of `49b5fe3` either way, so nothing had diverged and no history was
rewritten.

Question answered: does the combined implementation, after 3A, 3B, 3B.5,
3B.5b, 3C and 3C.5, still make the production decisions its measurements
chose — and is anything left that a user would notice?

Answer: yes, and almost nothing. All fourteen automatic policies resolve
exactly as the measured record intends, on every backend and both precisions,
and every one of them is confirmed against its credible forced alternative on
the phase it governs. **No automatic performance policy changed**, because no
measurement asked for one. What the phase did find were three defects and a
set of stale statements.

The defects. `ExactOperatorClasses` narrowed `std::size_t` pair indices into
`std::uint32_t` with no guard; a dense all-to-all plan passes that range at
roughly 65536 bodies per side. Widening would have doubled `class_of_pair`,
which holds one entry per pair, so the compact words stay and classification
is abandoned instead — the file's established idiom, and a decision that costs
only reuse, never correctness. The guard precedes the allocation, so its test
costs no memory. The duplicated `classify_endpoint_operators` took the same
guard. Second, an explicit FP32 `CudaBsr3` request under a lowered
`cuda_p2p_bsr_max_bytes` left construction reporting `CudaBsr3` and then failed
the evaluation with "CUDA FP32 P2P dimensions are inconsistent": the budget
gated a speculative prebuild that the executor consumed unconditionally.
`docs/static-p2p.md` already promised that `p2p_packing` outranks the budget,
so this aligned the code to its documented contract; FP32 is the default
precision, so only the budget had to be lowered to reach it. The pre-existing
test asserted the packing *name*, which a default-constructed plan satisfies,
so the new case evaluates the field and fails without the fix. Third,
`ReducedSymmetryP2P.ipynb` set two options that no longer exist and raised
`AttributeError`; no test executes that notebook.

The measurements. 158 evaluation cases, a cold construction matrix and a
cold/warm startup decomposition, all on one `benchmark-all` tree so every
comparison is same-session. The sharpest lesson is methodological: a rule that
governs one stage must be judged on that stage. FP64 `CudaFull` point
expansions differ by 2 % end to end — which would have read as "procedural is
fine now" — and by 1.39x on P2M+L2P, with P2M alone 1.86x, reproducing Phase
3B.5's figure almost exactly. Judged on the total, the FP64 rule would have
looked wrong; judged on its own phase, it is right.

Startup changed more than policy did. Phase 3C had named warm derived packing
"the clearest next lead" at 0.469 s of a 0.978 s warm setup at 32,768 bodies.
Re-measured at that exact size, the warm setup is 0.419 s and derived packing
is 0.00 ms, for a point plan and a finite lattice alike; the timer is still
charged on warm hits and the stages account for the total, so the zero is
real. What replaced it is the FP32 precision conversion at 167 ms, 40 % of the
warm setup, whose removal means persisting the FP32 plan — a cache-format
change this phase is explicitly not authorised to make, and not worth one for
a one-time 167 ms against a 1 ms evaluation. The universal operator bank is
confirmed as the dominant truly-cold cost (0.083 / 1.34 / 11.82 s at p = 4 /
6 / 8, identical across geometries and backends) and is removed entirely by
its cache.

One audit finding was wrong and is worth recording as such. A subagent flagged
`CudaPartial`'s unconditional top-priority M2L stream as an inconsistency
against `CudaFull`'s conditional rule. It is deliberate and measured — the host
blocks on exactly that stream — so the code now says so at both call sites
rather than inviting the same conclusion again.

Rejected on measurement, each with its number: the CPU lattice dictionary as a
default (8 % faster, 20-31x more host memory), oneMKL as the default M2L (it
wins one row of four), removing `CudaPartial` (slower everywhere but far
smaller at high occupancy), FP64 CUDA `PointGeometry` (1.30x slower, retained
as an explicit memory trade), FP64 procedural expansions (above), and
calibrating the explicit dictionary executor — that last one because "neither
flag set" is the only way to select the source-warp kernel, so calibrating it
would remove that ability; it is documented instead, with the 2.59x it costs
at 8 targets per leaf and the one flag that recovers it.

Deferred to Phase 4, all confirmed by reading the code: the `cuda_policy`
module's name understating that it also decides CPU policy, the two
structurally identical classifiers, the dead `use_cuboid_p2m_` /
`use_cuboid_l2p_` members, the write-only `periodic` and `bsr_*` policy inputs,
the uncalled three-argument `far_field_stream_priority` overload that carries
the rule's documentation, and the notebook's remaining pre-3A idioms.

No public API, C ABI, Python API, Fortran interface, cache format or cache key
changed. `benchmarks/baselines/phase3d/` retains the numbers, labelled an
engineering regression baseline and not an Article1 benchmark; the publication
campaign remains deferred until after Phase-4 pruning.

## 2026-09-18 — Dense/all-to-all construction optimization (Phase 3C.5)

Starting HEAD `cefc975` on the worktree branch `phase3c-construction`, a
fast-forward descendant of `refactor/architecture-v0.2` carrying the nine
Phase-3C commits; work was done on `phase3c5-dense-construction`. Question
answered with measurements: how quickly can the exact dense all-to-all plan be
built, and how much of Phase 3C's construction strategy transfers to it?

Answer: most of it, with one sharper limit and one caveat Phase 3C did not
have. The audit found that `CudaDenseDirectPlan` builds a complete host
`DenseDirectPlan` and uploads it, and that portable CPU and oneMKL share that
same constructor, so there is one construction path for all three backends;
that FP32 quantises at the point of store and never materialises an FP64
matrix, so there is no conversion pass to optimise; and that the pair loop was
already parallel, so parallelism was not the missing piece. What was missing
was reuse: the constructor evaluated one exact tensor per pair although a pair
tensor is a pure function of the displacement, the two body records and
whether the pair is an omitted point self interaction. It now classifies by
those bits, builds each distinct operator once in parallel, and scatters.

Cold construction, eight P-cores, bit-identical output throughout: prism
lattice 512² 3.07 s -> 0.046 s (67x), asymmetric 1024x512 6.07 s -> 0.072 s
(85x), tetrahedron-prism lattice 256² 7.42 s -> 0.128 s (58x), tetrahedron
lattice 256² 3.30 s -> 0.087 s (38x), prism-to-point 1024² 0.34 s -> 0.024 s
(14x). CUDA inherits the same figures because it delegates to the host plan.
Repeated evaluation is unchanged at 0.94-1.02x across portable CPU, oneMKL and
CUDA in both precisions.

Two limits are honest rather than incidental. Point-to-point never classifies:
a point pair costs about 4 ns against 260 ns for the cheapest finite pair, so
the lookup would cost more than the arithmetic, and 77x redundancy is left
unexploited deliberately. Irregular geometry has no reuse *at all*, because in
an all-to-all plan every pair holds a unique combination of records — a
sharper limit than the FMM near field, where a neighbourhood list still left
19.7x on irregular prisms.

Two findings came from widening the workloads mid-phase rather than from the
original plan. Anisotropic spacing costs reuse through floating-point rounding
alone: `spacing * a - spacing * b` rounds differently with magnitude once the
spacing is not exactly representable, so fifteen index differences become
thirty-three distinct displacement bit patterns and a point plan's distinct
count rises 4.5x. Reuse stays exact, it simply finds less, and `DenseDirectPlan`
has no canonical-grid normalisation to hide it. And a locally refined grid —
one octant subdivided, which is what a real discretisation looks like — sits
at about 5.5x reuse, in the band between the two extremes. The classification
gate was first written as a fixed eightfold reuse requirement justified by the
two extremes alone, and it discarded exactly that case: an exact
prism-to-tetrahedron build stayed at 5.11 s when 0.91 s was available. The gate
is now a byte budget on transient tensor storage, half the matrix bytes the
plan retains anyway, which keeps those cases and still bounds the memory.

Also accepted: the Phase-3C exact-equivalence machinery moved to a shared
internal header so the near-field and dense builders cannot drift apart on
what "the same operator" means, and a tetrahedron's point-field geometry is now
derived once per record rather than at each of the 216 quadrature nodes. The
latter was predicted to dominate tetrahedron-source far pairs and is worth
about 9%; it is bit-identical and free so it stays, but the measurement, not
the prediction, decided it.

Correctness is bitwise against the pinned starting-SHA build over 150
configurations, including asymmetric counts, partial identity maps and
throwing configurations compared on their message.
`tests/test_dense_direct_exact_reuse.cpp` pins the same contract at the lowest
layer. No public API, C ABI, Python API, Fortran interface or cache format
changed; the construction diagnostics the benchmark reads are internal to
`src/`.

Full detail, including rejected options and the CUDA setup decomposition, is in
`agent_docs/performance_optimization.md` under "Dense/all-to-all construction
optimization (Phase 3C.5)".


## 2026-09-18 — Static-plan construction and plan preparation (Phase 3C)

Starting HEAD `a2af367` on `refactor/architecture-v0.2`, which
`origin/refactor/architecture-v0.2` already pointed at; the local branch was
fast-forwarded to it at the start. Work was done on the worktree branch
`phase3c-construction`. Question answered with measurements: where does cold
plan construction spend its time, and what of that can be parallelised,
reused or avoided without slowing repeated evaluation?

Answer: almost all of it was building the same operator over and over. A pair
tensor is a pure function of the displacement and the participating body
records, and normalisation puts body centres on a canonical grid, so pairs
that describe the same interaction agree *bitwise*, not approximately. A
4096-body regular prism lattice reaches 681,472 near-field pairs from 343
distinct displacements, and the same is true of the finite P2M and L2P
operators, which depend only on a body's shape record and its offset from its
leaf centre. Construction now classifies by those exact bit patterns, builds
each distinct operator once, and does so in parallel; the generic pair loop
and both endpoint loops were serial before, with no data dependency forcing
it.

Cold construction, 4096 bodies, order 6, FP32, eight P-cores, cache disabled:
prism lattice 66.81 s -> 1.52 s (44x), prism irregular 66.11 s -> 1.95 s
(34x), tetrahedron lattice 15.87 s -> 1.54 s (10x). The tetrahedron irregular
case has no exact duplicates to find and gains only the rebalanced schedule
(19.7 s -> 16.1 s), which is the honest limit of the mechanism. Under exact
finite far-field models the endpoint operators fall further: the tetrahedron
order-6 case goes from 17.6 s to 1.58 s, with P2M 436x and L2P 486x.

Correctness is pinned by byte-comparing the persisted geometry plan, which
serialises the canonical P2P blocks, the P2M plans and the L2P evaluators:
identical bytes prove identical operators, which is stronger than a tolerance
comparison. 22 configurations for the endpoint change and 17 for the
near-field change are byte identical, covering all nine geometry pairs, both
layouts, irregular bodies, exact and point far-field models, FP32 and FP64,
orders 4, 6 and 8, the Cartesian basis, and periodic evaluation. A new
`tests/test_p2p_exact_reuse.cpp` pins the reuse contract itself, including
inputs one ULP apart that must not alias, per-body sizes, periodic images that
legitimately do share, the finite self tensor and the point identity
exclusion.

Two measurement faults were found and corrected, and both changed
conclusions. `benchmark_uniform_fmm` hard-coded point far-field models for
finite bodies, so the exact finite P2M/L2P operators -- and the tetrahedron
barycentric expansion Phase 3B.5b had flagged -- were never exercised by any
benchmark although exact geometry is the `UniformFmmOptions` default;
`--far-field-model exact` fixes that and is what made the endpoint cost
visible at all. Separately, one killed benchmark process survived and competed
for the cores during an early sweep, inflating those rows about twofold; they
were discarded and re-measured on an idle machine.

One defect of my own was caught before validation rather than by it: the
tetrahedron pair's exact key needs 28 words against a 27-word capacity, a
one-past-the-end write that would have corrupted the key length and merged
unrelated interactions.

Not done, and why. The prism `long double` arithmetic keeps its precision:
after reuse the exact tensor evaluation is a small fraction of a near-field
stage that is itself under a tenth of construction, so the 2.7x Phase 3B.5b
measured for `double` no longer buys enough to justify changing the numerical
contract of the production exact tensor, and no numerical audit was needed
because the change was not made. Skipping the canonical near-field build for
point plans is real (1.16 s of a 3.15 s 32,768-point build) and no in-process
consumer blocks it, but the geometry cache serialises the operator
unconditionally and its key contains neither the P2P packing nor the backend,
so one geometry has one entry shared by every packing; an entry written by a
skipped build would silently give a later `CanonicalAos` or CUDA plan an empty
near field. That is a persistent-cache change, which this phase is not
authorised to make, so the subtask stopped and is documented.

Benchmarks left in tree for Article1: `benchmarks/run_construction_matrix.py`,
a resumable cold-construction matrix that records the per-phase timings, the
plan's byte accounting, each row's peak resident set and an evaluation median
as the regression guard. `StaticPlanStatistics` gained
`p2p_interaction_setup`, `p2p_canonical_operator`, `p2p_derived_packing` and
`precision_conversion`, and the warm-cache path now records the same timings,
so a cache hit's remaining cost is attributable. No cache format, cache key,
C ABI, Python API or Fortran interface changed.

## 2026-09-17 — Exact finite-geometry procedural vs precomputed operators (Phase 3B.5b)

Starting HEAD `551c790` on `worktree-p2p-unification`, a clean fast-forward of
`refactor/architecture-v0.2` (`38f1b98`), which stayed checked out in the main
working copy, so the work continued on the worktree branch and
`refactor/architecture-v0.2` was fast-forwarded at the end. Question answered
with measurements: is precomputation actually faster for the exact operators
of uniformly magnetised prisms and tetrahedra? Yes, by 150x to 1,300,000x per
field update, amortising within one to ten updates on the CPU.

Added `benchmarks/benchmark_operator_representation.cpp` with its CUDA
procedural prism kernel, `run_operator_representation.py` (resumable matrix
driver) and `analyse_operator_representation.py` (ratios, retained bytes and
the break-even). The benchmark measures construction, precomputed application
through the production packings, and procedural reconstruction through the
production builders, in a hot single-threaded mode and a streaming
eight-thread mode over a complete list-1 neighbourhood, with changing moments,
checksums and a per-row error against the FP64 canonical operator. 1003 CPU
and 178 CUDA rows.

One production change: `src/geometry/primitives/rectangular_prism_point_kernel.hpp`
makes the MagTense prism point tensor precision-generic so production
(`long double`) and the device benchmark (`float`/`double`) share one
definition instead of two copies. Verified bit-identical over 20,680 tensors
covering every branch, and locked by a new case in
`tests/test_rectangular_prism_magtense.cpp`. No policy, option, executor or
cache change; `PointExpansionExecution` keeps its point-only semantics.

A defect found and fixed during the study: the first CUDA procedural kernel
applied a point source's coincident self pair instead of excluding it, which
the benchmark's own correctness column caught as an order-unity error; the
affected rows were discarded and re-measured. compute-sanitizer memcheck,
racecheck, initcheck and synccheck are clean on the final kernel in both pair
directions.

Documentation: a new "Exact finite-geometry procedural vs precomputed
operators (Phase 3B.5b)" section in `agent_docs/performance_optimization.md`,
the measured statement in `docs/static-p2p.md` and `docs/architecture.md`
replacing the previous assumption, and the benchmark contract in
`docs/benchmarks.md`. Recorded for Phase 3C and not acted on: the point/prism
pair loops and the P2M/L2P plan construction are serial while the polyhedron
loops are parallel; the prism tensor's `long double` arithmetic costs 2.7x its
`double` equivalent for 4.5e-15 of agreement; and
`tetrahedron_averaged_monomial` heap-allocates per (mode, term, axis).

## 2026-09-17 — Procedural vs precomputed point operators (Phase 3B.5)

Starting HEAD `38f1b98` on `worktree-p2p-unification` (the user
fast-forwarded `refactor/architecture-v0.2` to the same SHA at the start).
Question answered with measurements: when is recomputing a cheap point
operator every evaluation faster than streaming its precomputed form.
Result: the near-field invariant now has an explicit refinement
(precomputed versus procedural representation, `docs/static-p2p.md`,
`docs/architecture.md`); finite tiles keep precomputed operators. Added:
`P2PExecutionPacking::PointGeometry` on both CUDA backends (one warp per
list-1 record, leaf-relative positions, no pair tensors; FP32 default for
point pairs: 1.3-7.5x faster kernels than leaf blocks on random points,
equal or faster evaluations than the lattice dictionary, 2-33x less device
memory; FP64 keeps stored tensors); an allocation-free host/device
solid-harmonic recurrence (`src/math/solid_harmonic_recurrence.hpp`, tested
against the polynomial basis); procedural point P2M/L2P on the CPU (SIMD
packs) and CudaFull (lane groups per leaf) behind
`UniformFmmOptions::point_expansion_execution` (Auto/Precomputed/
Procedural, Python mirrored), default on the CPU hierarchy in both
precisions and on FP32 CudaFull (3-7x faster P2M/L2P from p 6, rows gone),
FP64 CudaFull keeps rows; the stream-priority cost model now follows the
resolved packing. End to end (`Auto`, FP32): cuda-full S 178 -> 116 us,
M 689 -> 450, L 1885 -> 657, 128 per leaf 4501 -> 654 with 17-350x less
device memory; cuda-partial M 1686 -> 1081; the prior 1-2 % hybrid crossover
is gone (cuda-full wins everywhere). Correctness: procedural results match
the stored operators to rounding (FP32 P2P 3-5e-6 of scale after the
leaf-relative fix, P2M/L2P 5e-8-1.5e-7; FP64 <= 1e-15); new tests
`test_procedural_point_expansion.cpp`, recurrence test, geometry-matrix and
policy tests extended; compute-sanitizer memcheck/racecheck clean on every
new kernel. Full record: `agent_docs/performance_optimization.md`, "Procedural
vs precomputed point operators (Phase 3B.5)". Not done (recorded for 3C):
construction still builds the canonical operators that procedural plans
never read; prism pair construction is serial (64 s for 4096 prisms).

## 2026-09-16 — P2P execution unification, geometry/backend coverage, CudaPartial crossover

Starting HEAD `af50b69`, worktree branch `worktree-p2p-unification`. Commits,
in order: exact prism/tetrahedron pair tensors via the polyhedron surface
formulation with a far-separation Gauss safeguard (`2ab0edc`); identity
metadata carried in the leaf packing and obeyed by every CPU/CUDA tensor
executor (`e15ff80`); explicit `UniformFmmOptions::p2p_packing`
(`8e1d110`); tetrahedron point-kernel conditioning on edge lines
(`2d5ea1f`); the nine-pair geometry-matrix test (`3b1109e`); explicit
periodic `PointGeometry` (`1bd8562`); periodic image records in the leaf,
dictionary and BSR packings (`14701c8`); benchmark geometry/packing/periodic
options (`1c8ca3b`) and the matrix driver (`3dc5b8b`); then the measured
policy changes and the stream-priority optimisation (see the final commits
of the branch). Measured: periodic point geometry 2.7-5.1x faster near
field; finite lattices 2-7.5x faster with the dictionary on the CPU, 3x on
CUDA; leaf blocks beat BSR(3) on finite bodies; irregular bodies do not
compress (token-width guard); `cuda-partial` loses everywhere but 128-160
points per leaf (1-2 % ahead after the stream-priority change). Nsight
timelines: GPU P2P fully overlaps the CPU upward pass, M2L and P2P contend
when they overlap, PCIe is never on the critical path, host preparation and
combination are 15-20 % at M. Validation: fresh g++ trees for portable CPU,
CUDA, oneMKL and CUDA+oneMKL all pass the full CTest (222/222) and the
Python suites; compute-sanitizer memcheck (0 errors) and racecheck (0
hazards) on the changed kernels; the whitespace check; cache format/keys,
C ABI and Fortran interface untouched. Note: the conda env's default
`icpx`/`-ccbin=icpx` must be overridden with g++ (see the performance log).
Not done: construction optimisation (recorded), backend auto-selection
changes, cache format/key changes.

## 2026-09-16 — CPU / oneMKL evaluation optimization (Phase 3B)

`CPU_PERF_BASELINE = e4f1c79` (Task-A HEAD plus a link fix for the non-CUDA
stub, which had left every portable build unable to link since Phase 3A).
Scope: repeated CPU evaluation only. Accepted, each measured on identical
builds and workloads: per-target unit-stride portable M2L (`82c4b35`), dense
level-scaled P2M/M2M/L2L/L2P packing built once at construction
(`a5d8d05`), oneMKL per-level column ranges with a deterministic parallel
scatter and one thread-setting bracket per level (`ed67134`), a
transfer-class-sorted portable M2L block schedule (`5b97793`), and
position-based point-source P2P with the shared kernel in
`operators/p2p_point_kernel.hpp` (`cad666d`, new
`P2PExecutionPacking::PointGeometry`; `cdfmm_core` now compiles with
`-fno-math-errno -fno-trapping-math`). Portable evaluation at 8 threads is
3.5-10x faster (M 50k p6: 99.7 -> 12.7 ms FP32, 115 -> 18 ms FP64), oneMKL
1.8-4.1x, `cuda-partial` 1.4-5.7x; the portable executor now beats oneMKL
at every measured size. Full CTest (three builds) and Python suites pass; CUDA
results unchanged. Findings, tables, rejected experiments, the RPATH test
hazard and remaining bottlenecks: `agent_docs/performance_optimization.md`.
Phase 3B is complete; 3C construction optimization is next.

## 2026-09-16 — Regular-grid dictionary policy at high occupancy (3A closure)

Starting HEAD `a0a7003`. Calibrated the CUDA dictionary executor on regular
lattices at 48, 64, 80, 96, 128, 160 and 192 targets per leaf (depth 3) plus
48 per leaf at depth 4, FP32 and FP64, source-warp versus target-owned versus
power-of-two microtiles. The crossover from target-owned to source-warp lies
between 64 and 80 in both precisions, so the automatic `RegularGrid` policy
in `src/backend/cuda/execution_policy.cpp` now has three regimes: microtiles
below 48, target-owned from 48 to below 72, source-warp from 72. No kernel,
cache key/format, `General` behaviour or explicit-option semantics changed.
`benchmark_uniform_fmm --regular-grid` accepts `odd * 2^k` counts (the odd
factor stretches the shortest axis) so non-power-of-two occupancies are
exact. Calibration table, rationale and automatic-versus-best results are in
`agent_docs/performance_optimization.md`. Phase 3A GPU evaluation is closed.

## 2026-09-16 — CUDA execution policy and regular-grid hint (3A follow-up)

Starting HEAD `a187c76`. Added `src/backend/cuda/execution_policy.{hpp,cpp}`,
one deterministic module that resolves the CUDA strategy choices found in
Phase 3A from constructed-plan facts (precision, geometry, periodicity,
fixed identity, occupied leaves and mean leaf occupancy, pair and translation
counts, BSR estimate/budget, options): the P2P packing (leaf block / BSR /
canonical / signed dictionary), the dictionary executor, the grouped-M2L
pairs-per-thread rule and the M2M/L2L lane-group rule, which the executors
now query instead of owning. New public hint
`SpatialLayout {General, RegularGrid}` in `cdfmm/backend/execution.hpp`,
`UniformFmmOptions::spatial_layout` (default `General`, Python
`SpatialLayout.GENERAL/REGULAR_GRID`). `RegularGrid` selects the reduced-
symmetry dictionary automatically for non-periodic point sources with a fixed
identity map on CUDA backends, with power-of-two microtiles below 48 targets
per leaf and target-owned above (calibrated at 4-128 per leaf); explicit
options keep precedence and meaning. Not in the cache identity. Details,
calibration and benchmarks in `agent_docs/performance_optimization.md`.

## 2026-09-16 — GPU evaluation optimization (Phase 3A)

Starting HEAD `293144bf` (Phase 2 closure) is the fixed performance
baseline (`GPU_PERF_BASELINE`), built in a detached worktree with the
identical CUDA configuration. Scope: repeated `CudaFull`/`CudaPartial`
evaluation only; construction, CPU/oneMKL, APIs and pruning untouched.
Evidence, methodology, Nsight Systems findings, accepted/rejected
experiments, the dictionary-versus-leaf study and remaining bottlenecks are
in `agent_docs/performance_optimization.md`.

Accepted commits: grouped M2L with shared-memory matrix staging
(`721fe93`), host-turnaround removal around the device-resident evaluation
(`1d92a28`), atomic-free lane-group far-field kernels with per-level
launches (`8312083`), pinned staging in the standalone M2L plan
(`e508382`), warp-per-block leaf P2P as the point-source default with the
new `P2PExecutionPacking::LeafBlock` (`9ac2c58`), workload-sized tuning and
dictionary benchmark flags (`86f82d9`), and a fix plus regression test for
FP32 leaf plans built without a geometry cache (`d4061d1`).

RTX 5090, `cuda-full`, baseline -> final: FP32 2.2x-4.9x (50k/p6/d4
2397 -> 755 us; 200k/p6/d5 15.5 -> 3.2 ms), FP64 1.3x-2.5x;
`cuda-partial` 1.07x-1.9x (CPU hierarchy dominates it, deferred to 3B).
Nsight Compute counters are unavailable on this machine
(`ERR_NVGPUCTRPERM`); analysis used Nsight Systems, CUDA events,
arithmetic-intensity estimates and controlled experiments.

## 2026-09-16 — public/internal API, header ownership, and packaging cleanup

Starting HEAD `9bab9ee6` (`refactor(cache): decouple persistence from
UniformFmm`). Baseline check: "static triangular translations match M2M and
L2L references" reproduces identically (same `REQUIRE`, same printed values)
in a fresh portable `dev` build before any production change — confirmed
unrelated to this task, matching every prior session's finding.

Task: the last recorded Phase-2 cleanup item — public/internal C++ interface
and packaging. Scope: canonical homes for remaining substantive flat public
headers where warranted, the `parameter_selection.hpp -> uniform_fmm.hpp`
seam, CUDA/oneMKL availability-query ownership, a downstream CMake package,
and CUDA/oneMKL link-interface propagation. Audited first (header-by-header
reading, a full `CMakeLists.txt` read, and a dependency trace of
`parameter_selection.hpp`/`uniform_fmm.hpp`/the CUDA availability functions)
rather than mechanically implementing every historical suggestion.

**`parameter_selection.hpp` narrowed.** Its signatures need only `Vec3` and
`ExecutionBackend`; the complete `UniformFmm` API is needed only by
`src/parameter_selection.cpp`, which constructs `UniformFmm` objects.
`ExecutionBackend` moved to a new `include/cdfmm/backend/execution.hpp`
(backend-selection enum, naturally owned by `backend/`); the header now
includes only that plus `cdfmm/math/vec3.hpp`. `src/parameter_selection.cpp`
gained an explicit `#include "cdfmm/uniform_fmm.hpp"`, since it is the unit
that actually needs the complete API. All four consumers of
`parameter_selection.hpp` were checked individually; none broke (two already
included `uniform_fmm.hpp` directly, one gained the explicit include, and the
C++ test uses neither `Vec3` nor `UniformFmm`). The other backend/plan enums
still in `uniform_fmm.hpp` (`SphericalM2LBackend`, `StaticMatrixBackend`,
`StaticOperatorExecutor`, `P2PExecutionPacking`, `StaticExecutionPlan`) were
left in place: nothing audited needs them independently of the complete
solver API.

**CUDA/oneMKL availability queries relocated.** All seven CUDA queries
(`cuda_compiled`, `cuda_available`, `cuda_direct_available`,
`cuda_m2l_p2p_available`, `cuda_m2l_available`, `cuda_full_available`,
`cuda_device_description`) moved to a new
`include/cdfmm/backend/cuda/availability.hpp`, matching the existing
`backend/cuda/dense_direct.hpp`/`cuda_dense_direct_available()` pattern;
`one_mkl_available` moved the same way to a new
`include/cdfmm/backend/mkl/availability.hpp`. `uniform_fmm.hpp` includes both
and keeps re-exporting every name. This resolved the one dependency-audit
exception Phase 1 closure had recorded and repeated ever since:
`src/backend/cuda/fmm/internal.hpp` included the complete `uniform_fmm.hpp`
solely to see the declarations of the three availability functions
`plan.cu`/`stub/fmm.cpp` define — every other type that pair uses already
arrived through headers it already included directly
(`plan/static_plan.hpp`, `timings.hpp`, `math/potential_field.hpp`). It now
includes `cdfmm/backend/cuda/availability.hpp` instead, so the CUDA backend
no longer depends on the top-level public API to declare functions it
implements.

**`tensor_dictionary.hpp` relocated** to `include/cdfmm/plan/p2p/tensor_dictionary.hpp`,
alongside the P2P dictionary-packing headers that are its only consumers
(dependency-free header, pure relocation). Its four internal consumers
(`src/plan/precision.cpp`, `src/plan/p2p/dictionary_detail.hpp`,
`src/plan/p2p/signed_dictionary.cpp`, `src/backend/cpu/p2p/dictionary.cpp`)
now include the canonical path; `include/cdfmm/static_operators.hpp` keeps
the flat path, since it is itself a compatibility umbrella (façade including
façade is correct there).

**`timings.hpp`, `periodic.hpp`, `validation.hpp` audited and kept flat** —
each read in full and found to have no single clearer subsystem owner, not
merely undecided: `timings.hpp` aggregates `fmm`/`plan`/CUDA-backend
diagnostics read together through one observability surface; `periodic.hpp`
is depended on symmetrically by the sibling `tree` and `operators` layers, so
giving it to either creates the sibling-to-sibling edge the layering diagram
avoids; `validation.hpp` is explicitly documented as non-production
test/diagnostic utility. **`P2MPlan`/`FloatP2MPlan`/`ExpansionBasis`
reviewed, kept at their cache-cleanup homes** — both are coherent, minimal,
non-ABI-affecting placements; moving them again would add machinery without
fixing a real problem.

**CMake package added.** `cdfmm_c` (the only installed target; a stable C-ABI
shared library) gained `add_library(cdfmm::cdfmm_c ALIAS cdfmm_c)`,
`install(EXPORT cdfmm_c-targets ...)`, a generated `cmake/cdfmmConfig.cmake.in`
config, and a version file (`project(... VERSION 0.1.0 ...)`, matching
`pyproject.toml`; not a v0.2 marker), installed to
`lib/cmake/cdfmm`. No `find_dependency` calls are needed or present: `cdfmm_c`
links its private static `cdfmm_core` implementation library `PRIVATE` and
declares no `PUBLIC`/`INTERFACE` link libraries of its own, so none of
`cdfmm_core`'s CUDA/oneMKL/OpenMP dependencies appear in `cdfmm_c`'s exported
interface (confirmed by inspecting the generated `cdfmm_c-targets.cmake`).
`cdfmm_core` is never installed, so there is no static-linking downstream
case to support. Validated with an isolated-prefix portable install (no
source/build-tree paths leaked into installed headers or exported CMake
files) and a standalone out-of-tree consumer
(`find_package(cdfmm CONFIG REQUIRED)` + `target_link_libraries(consumer
PRIVATE cdfmm::cdfmm_c)`) that built and ran against only that prefix,
reproducing the analytic two-dipole field to `1.7e-16`. (The dynamic loader
initially resolved a stale, previously conda-env-installed `libcdfmm_c.so`
ahead of the fresh one, because IntelLLVM embeds `$CONDA_PREFIX/lib` in
`DT_RPATH`; forcing resolution with `LD_PRELOAD` confirmed this was a
pre-existing toolchain/environment artifact, not caused by the new package —
the freshly built library's own build-configuration diagnostics matched its
actual build options once forced.)

**Fixed one incidentally-broken pinned test.** Adding `VERSION 0.1.0` to
`project(...)` changed the exact `CMakeLists.txt` text
`python_tests/test_profiling_setup.py::test_notebook_preset_supports_gnu_mkl_without_intel_header_leakage`
pinned (it exists to catch a regression to CXX-only, not to pin the exact
`project()` spelling); updated the asserted literal to match.

**Corrected stale documentation.** `docs/architecture.md`'s "Phase 1 closure"
> "Ownership and dependency audit" still described
`src/cache/internal.hpp -> cdfmm/uniform_fmm.hpp` as a live sanctioned edge;
it was removed by the earlier cache/plan-preparation boundary cleanup and the
paragraph was never corrected afterward. Annotated both that bullet and the
CUDA-availability bullet (now also resolved by this task) as historical
Phase-1-closure findings, not current-tree statements, and fixed the
"Deferred refactor inventory"/"Phase 2 handoff" bullets that repeated the
stale `parameter_selection.hpp`/CUDA-availability/CMake-package items as
still open. Added a new "Public/internal API, header ownership, and
packaging cleanup" subsection recording the full audit, decisions, and
validation evidence. Updated `AGENTS.md`, `include/cdfmm/AGENTS.md`,
`include/cdfmm/plan/AGENTS.md`, `src/backend/mkl/AGENTS.md`, and
`agent_docs/project_structure.md` to match.

**Validation.** Portable `dev` (fresh header changes, incremental build):
197/198 CTest passed (the one failure is the same pre-existing unrelated
triangular-translation case), 4 expected skips; Python 137 passed/7 skipped.
CUDA + oneMKL `notebooks` (fresh, RTX 5090 SM120, CUDA 13.3.73, oneMKL
2026.1; built with `-j 2` after the default unbounded `-j` triggered
unrelated nvcc front-end internal-compiler-errors under parallel load on four
unrelated `.cu` files — confirmed environmental by recompiling one of the
four in isolation successfully with identical flags before retrying at lower
parallelism, not a defect in the header change): **198/198 CTest with zero
skips** (including the triangular-translation case, matching this project's
established zero-skip CUDA+oneMKL pattern), Python 143 passed/1 skipped
against `build-notebooks` (after the pinned-test fix above). C ABI: same 14
`cdfmm_*` symbols. Python surface: 63 top-level names (34 classes, 29
functions), matching the documented baseline. Standalone-compile and
mixed/reverse-include-order probes covered every header moved or added in
this task (`backend/execution.hpp`, `backend/cuda/availability.hpp`,
`backend/mkl/availability.hpp`, `plan/p2p/tensor_dictionary.hpp`, the
`tensor_dictionary.hpp` façade, and the narrowed `parameter_selection.hpp`).
`git diff --check` clean.

No solver algorithm, cache format, cache key, C ABI, Python API, or Fortran
interface changed. `Article1/` (now present again, alongside the previously
renamed `Article1_old/` — unexplained, apparently touched by something
outside this session again) and the unrelated untracked
`examples/simple_notebooks/tetrahedron_target_average_fair_sampling.ipynb`
remain preserved outside this task.

This closes the Phase-2 cleanup list recorded in `docs/architecture.md`'s
"Phase 2 handoff": internal-implementation duplication, the tree/topology
boundary, the cache/`UniformFmm` boundary, and now the public/internal
API/header/packaging boundary are all resolved. The next planned work is a
dedicated performance-optimization phase (construction and evaluation, CPU/
oneMKL/CUDA), followed by repository pruning; neither started here.

## 2026-09-16 — cache/plan-preparation boundary cleanup

Starting HEAD `03be4671` (`refactor(tree): clarify topology ownership`).
Baseline check: the known pre-existing failure, "static triangular
translations match M2M and L2L references", reproduces identically in a
fresh portable `dev` build (same `REQUIRE`, same printed values) — confirmed
unrelated to this task before any production change.

Task: narrow the cache subsystem's dependency on `UniformFmm`. Cache
persistence (`initialise_cache_keys`, `load_universal_cache`,
`write_universal_cache`, `load_geometry_cache`, `write_geometry_cache`) was
implemented as five private `UniformFmm` member functions defined in
`src/cache/*.cpp`, forcing `src/cache/internal.hpp` to include the complete
`cdfmm/uniform_fmm.hpp` and giving cache translation units access to
essentially all private solver state.

**Audit (two read-only workers, run in parallel).** One classified every
field each of the five functions reads or writes from `UniformFmm` as cache
identity, persisted payload, statistics, or unrelated plan/lifecycle state
merely read in passing; the other confirmed none of the five functions is
referenced anywhere outside `src/cache/*.cpp`/`src/fmm/*.cpp` (they are
private — the compiler already enforces this), catalogued the full public
C++/C-ABI/Python-visible surface (public accessors `universal_cache_key()`/
`geometry_cache_key()`/`periodic_cache_key()`, the `StaticPlanStatistics`
cache fields returned by `static_plan_statistics()`, `UniformFmmOptions::
enable_cache`), confirmed no `sizeof(UniformFmm)` check exists anywhere and
that removing private member-function declarations cannot affect object
layout, and confirmed no committed old-cache compatibility probe exists (each
session writes one fresh, ad hoc, against an isolated `v0.1.0` worktree).

**Design.** Narrow free functions plus explicit records, not a service
object or a friend god-class (both were explicitly out of scope per the
brief). Every cache entry point is now a free function in
`cdfmm::detail::cache` taking one of three explicit records declared in
`src/cache/internal.hpp`: `CacheIdentityInputs`/`CacheIdentity` (identity;
`keys.cpp`), `UniversalCacheIdentity`/`UniversalCachePayload` (the
depth-independent bank and periodic root; `universal.cpp`), and
`GeometryCacheIdentity`/`GeometryCachePayload` (the geometry-dependent plan;
`geometry.cpp`). Payload records hold references to already-existing
`UniformFmm` members, not a second copy of solver state.
`src/fmm/execution_setup.cpp` (identity) and `plan_preparation.cpp`
(universal/geometry load and write) assemble these records from `this`'s
private state, call the free functions, and copy results back; they remain
the sole callers, so plan preparation remains the sole owner of the
cache-vs-build decision — now structurally, not just by convention. All five
private `UniformFmm` cache-method declarations were removed from
`include/cdfmm/uniform_fmm.hpp`; the three trivial cache-key accessors
(previously also defined in `keys.cpp`) moved to `src/fmm/uniform_fmm.cpp`,
which already owns lifecycle/accessors.

Two narrow types moved out of `uniform_fmm.hpp` to make this possible without
an ABI-affecting layout change (a public installed header cannot include an
internal `src/` header, so these types had to move to an existing installed
canonical header instead): `P2MPlan`/`FloatP2MPlan` (private nested structs
needed by the geometry payload) moved to
`include/cdfmm/plan/static_coefficient.hpp`, alongside
`StaticCoefficientOperator`, the type category they already belonged to;
`ExpansionBasis` (declared directly in `uniform_fmm.hpp` with no header of
its own, needed by `CacheDescriptor`) moved to
`include/cdfmm/core/precision.hpp`, alongside `StaticPrecision`. Neither was
reachable except through `UniformFmm`'s public API by value, so neither move
changes a public name's meaning; `uniform_fmm.hpp` still transitively
provides both. `sizeof(UniformFmm)` is unchanged: 7272 bytes measured before
and after with a standalone probe. `StaticPlanStatistics` continues to be
mutated directly by cache free functions (unchanged from before): it is
already a narrow, `UniformFmm`-independent plan-layer type declared in
`cdfmm/timings.hpp`, so this is not a coupling concern, and adding a returned-
result type for every cache call would have added complexity without
narrowing anything.

Persistent format, cache keys, hash inputs/order, the direct FP32 decode
path, and the asymmetric FP32-only failure-cleanup behaviour are all
byte-for-byte/behaviourally unchanged; `src/cache/format.cpp` and `io.cpp`
were not touched (neither ever depended on `UniformFmm`).

**Validation.** Fresh portable `dev`: clean configure/build; full CTest 198
total, 193 passed, 4 expected skips, the same pre-existing "triangular
translations" failure (confirmed identical on the unmodified `03be467`
baseline); Python 137 passed/7 skipped against the just-built module.
Focused `ctest -R 'cache' -E 'tetrahedron pairs|tetrahedron dispatch'`: 11
total, 9 passed, 2 expected oneMKL/CUDA skips. `sizeof(UniformFmm)` compared
byte-for-byte via a standalone probe against the unmodified `03be467` tree:
identical (7272). A cross-version probe (public-API-only source, unchanged)
built against an isolated `v0.1.0` worktree wrote an 8-file FP64/FP32
free/periodic cache corpus cold, then HEAD read it back: identical universal/
geometry/periodic cache keys, every hit, zero bytes written, corpus
byte-identical by `sha256sum`, field results bit-identical. A before/after
timing probe (both built without LTO to permit direct static-library
linking, identical ~4,500-point geometry, 5 repeats) showed no measurable
warm-construction slowdown (~0.725 s baseline vs ~0.726 s HEAD, within
run-to-run noise); the same probe also surfaced a geometry-cache-miss
characteristic for that specific synthetic point spacing that reproduces
identically on baseline and HEAD, confirming it predates this task and is
unrelated to it (not investigated further, per scope). CUDA + oneMKL
`notebooks` (fresh, RTX 5090 SM120, CUDA 13.3.73; `TMPDIR` moved off the
shared `/tmp` tmpfs to avoid the known disk-quota LTO failure): **198/198
CTest with zero skips**, including the triangular-translation case, which is
sensitive to build configuration and passes here — matching this project's
established pattern of zero-skip CUDA+oneMKL runs, not a change caused by
this task; the full focused cache selector 11/11, including the two
CUDA/oneMKL-only cases the portable build skips; Python 143 passed/1 skipped
against the `build-notebooks` module. `git diff --check` clean.

`docs/architecture.md`'s stale "Deferred refactor inventory" and "Phase 2
handoff" cache bullets (still describing cache entry points as `UniformFmm`
members, and separately still listing the already-resolved canonical-to-
compact P2P/factorial deduplication as open Phase-2 work) were corrected, and
a new "Cache/plan-preparation boundary cleanup" subsection records the full
audit and design. `src/cache/AGENTS.md`, `AGENTS.md`, and
`agent_docs/project_structure.md` were updated to match; the now-obsolete
`src/cache/internal.hpp -> cdfmm/uniform_fmm.hpp` "sanctioned edge" bullet was
removed from both `AGENTS.md` and `project_structure.md`'s audited-exceptions
lists, since the edge no longer exists.

An unrelated, pre-existing `Article1/` directory (roughly 640 GB of research
data, untracked) was found renamed to `Article1_old/` partway through this
session; nothing in this task's commands referenced that path, and its
contents look intact, so this was very likely a concurrent, unrelated action
on this shared machine. Left untouched and unexplained; flagged for the user.
The unrelated untracked
`examples/simple_notebooks/tetrahedron_target_average_fair_sampling.ipynb`
remains preserved outside this task.

## 2026-09-16 — tree/topology/FMM-plan boundary cleanup

Starting HEAD `76d6c9ab` (`refactor(p2p): unify canonical-to-compact packing
semantics`). Two parts, committed separately.

**P2P visibility correction (carry-over from the previous task).** The
previous internal-duplication cleanup had added
`assign_static_p2p_compact_row` to the installed
`include/cdfmm/plan/p2p/compact.hpp`, but the helper existed only so
`src/plan/p2p/compact.cpp` and `src/cache/format.cpp` could share packing
mechanics — not as a supported downstream API. Moved both FP64/FP32
overloads to a new internal `src/plan/p2p/compact_row.hpp` (namespace
`cdfmm::plan_detail`, matching the existing `plan_detail` convention used by
`src/plan/p2p/dictionary_detail.hpp`); `compact.hpp` keeps only the public
`StaticP2PCompactPlan`/`FloatStaticP2PCompactPlan` types and
`build_static_p2p_compact_plan` builders. `compact.cpp` pulls the helper in
with a `using` declaration (ADL can't find it: the primitive lives in
`cdfmm::plan_detail`, its arguments in `cdfmm`); `format.cpp`'s one call site
(inside `cdfmm::detail::cache`) qualifies it explicitly. No cache format,
key, or numerical behaviour changed — this is a pure visibility/location
fix. Updated `src/cache/AGENTS.md` and `docs/architecture.md` to point at
the new internal path instead of the installed header.

Validation: fresh portable `dev` configure/build clean; full CTest 198
total, 193 passed, 4 expected skips, and one **pre-existing** failure —
"static triangular translations match M2M and L2L references" — confirmed
identical (same REQUIRE, same printed values) on a `git stash`-restored
`76d6c9ab` tree, so unrelated to this change (an M2M/L2L floating-point
comparison in an unrelated test, not touched by this task). Focused
`ctest -R 'cache|p2p|foundational'` (excluding tetrahedron cases): 12 total,
10 passed, 2 expected CUDA/oneMKL skips. Python 137 passed/7 skipped against
the just-built `build/` module
(`cdfmm.__file__` checked). `git diff --check` clean. Committed as `5af3782`
("refactor(p2p): keep compact row packing internal").

**Tree/topology/FMM-plan boundary audit (main task).** The task brief
described `StaticFmmTopology` and `AdaptiveTree` returning it as a
"transitional tree-to-plan coupling" per `adaptive_tree.hpp`'s `@warning`
and several `docs/architecture.md` passages, and asked for an audit before
assuming that framing was correct. Two read-only Explore workers ran in
parallel:

- **Ownership audit** (`StaticFmmTopology`, `UniformTree`, `AdaptiveTree`,
  `build_uniform_fmm_topology`): classified every field/method. Every one is
  a raw spatial-tree fact (nodes, permutations, sorted positions, leaves,
  coordinate origin/scale), an interaction-topology fact (M2L
  admissibility/pairing via `m2l_interactions`, near-field leaf pairing via
  `p2p_leaf_records` — confirmed backend-neutral: no packing choice, no
  operator coefficients, three independent downstream consumers each derive
  their *own* packing from the same records), or CSR-style schedule indexing
  over those facts (M2M/L2L level/parent offsets, M2L row offsets). `validate()`
  checks only structural invariants — no operator, packing, or backend
  concept anywhere in `tree/`. `build_uniform_fmm_topology` is pure topology
  extraction/reformatting from `UniformTree`, no plan construction. The one
  minor finding: `StaticFmmTopology::occupied_source_leaves()`/
  `occupied_target_leaves()` appear dead (every real call site reads
  `.source_leaves`/`.target_leaves` directly, or calls the differently-typed
  same-named method on `UniformTree` instead) — left alone as an unused but
  harmless public-API leftover; removing a public method is out of this
  task's scope (deferred to the later public-header/API cleanup task).
- **Consumer/API audit** (`StaticFmmTopology`, `AdaptiveTree::topology()`/
  `shared_topology()`, `build_uniform_fmm_topology` across `fmm/`, `cache/`,
  `plan/`, `python/`, tests, benchmarks): confirmed `StaticFmmTopology` was a
  genuine `v0.1.0` public type (not invented in v0.2); traced the one real
  uniform/adaptive divergence (`fmm/plan_preparation.cpp`'s
  `supplied_topology_` branch keys M2L transfer class off the integer
  `transfer_class` field for uniform-built topologies but the continuous
  `displacement` field for supplied/adaptive ones — a plan-layer concern,
  correctly downstream of topology, untouched here); confirmed
  `plan/p2p/*` never references `StaticFmmTopology` at all, only the
  narrower record types; confirmed the cache subsystem never persists
  `StaticFmmTopology` and disables caching entirely whenever a topology is
  supplied externally (`cache/keys.cpp`'s `supplied_topology_` check), so
  `AdaptiveTree` output never touches the persistent cache format; and
  catalogued exactly which fields Python (`python/tree.cpp`) and the C++
  tests (`test_static_topology.cpp`, `test_uniform_fmm.cpp`,
  `python_tests/test_adaptive_tree.py`) pin versus which are free internal
  detail.

Independent confirmation the lead found directly: `python_tests/test_adaptive_tree.py`
asserts `plan.topology is topology` — Python object identity through the
`shared_ptr<const StaticFmmTopology>` boundary — meaning the FMM/plan layer
treats topology as a borrowed reference, never rebuilding or taking
ownership away from the tree that produced it. That is the plan-does-not-
own-topology property the task asked to verify, demonstrated by an existing
test rather than by inspection alone.

`AdaptiveTree`'s ~200-line constructor was checked for whether spatial
subdivision and interaction-topology assembly are genuinely separable. They
already are: the octant-partition-plus-M2M/L2L-edge pass and the dual-tree
near/far `visit` pass are sequential and non-interleaved, and the
constructor already times them separately (`tree_seconds_` ends and
`interaction_seconds_` begins exactly at that boundary, `adaptive_tree.cpp:149`).
Introducing a separate adaptive-only spatial-tree type (mirroring
`UniformTree`) was considered and rejected: unlike `UniformTree`'s `TreeNode`
(which has genuinely distinct content — level-wise dense Morton arrays,
`list1`/`list2` candidate lists never stored in `StaticFmmTopology`), an
adaptive-only node type would duplicate essentially every `StaticFmmTopology`
field, which the task explicitly warned against manufacturing "to make the
layering diagram prettier." Comparing against `UniformTree::build` (also one
monolithic, internally phase-timed ~340-line function, never split into
multiple top-level functions) confirmed that splitting `AdaptiveTree` alone
into named functions would be inconsistent with the codebase's own
established convention for tree builders, not an improvement.

**Outcome: no production code changed.** The "transitional tree-to-plan
coupling" framing was a documentation defect predating this audit, not a
live architectural problem. Corrected: `include/cdfmm/tree/adaptive_tree.hpp`'s
misleading `@warning`; `docs/architecture.md`'s "Deferred refactor
inventory" (two bullets), "Foundational layout" bullet, the "remaining
transitional seam" paragraph, one dependency-audit-exceptions bullet, and
the "Phase 2 handoff" "Transitional seams" entry (replaced with a new
"Tree/topology boundary cleanup" subsection recording the full audit);
`src/tree/AGENTS.md`'s closing sentence; and two passages in
`agent_docs/project_structure.md`. `docs/architecture.md`'s
`sphinx-build -W --keep-going -b html docs docs/_build/html` was not
re-run in this session (no non-code-comment content changed there beyond
prose); C++/Python test suites were not expected to and did not need to
change, since no field, signature, ordering, or numerical behaviour moved.
No new tests were required or added; existing coverage
(`test_static_topology.cpp`, `test_uniform_fmm.cpp`,
`python_tests/test_adaptive_tree.py`) already pins everything this audit
touched, and it continues to pass unchanged (same portable CTest/Python
results reported above, since both parts of this task's validation ran
against the same build).

`Article1/` and the unrelated untracked
`examples/simple_notebooks/tetrahedron_target_average_fair_sampling.ipynb`
remain preserved outside this task. Not started, and explicitly out of
scope: deeper cache/`UniformFmm` encapsulation, the flat-header/packaging
relocation candidates (including `parameter_selection.hpp` including the
complete `uniform_fmm.hpp`), and repository pruning — each its own future
task, as recorded in `docs/architecture.md`'s Phase 2 handoff.

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
