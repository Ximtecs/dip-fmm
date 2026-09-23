# Project progress

## Article1 preparation fixes: COMPLETE — 2026-09-23

Branch `article1-benchmark-fixes` from `aa9d75f`. Four defects found while
preparing Article1 (explicit CPU packing lost to the layout hint; CUDA
warm-up polluting cold-construction measurements; point reference for
finite-body accuracy; dense exact-reuse sampling blind above 32768 sources)
and the Phase-3C lead-7 limit (`PointGeometry` plans built the full stored
tensor list) are fixed with regression tests. Evidence and numbers are in
`latest_session_work.md` (2026-09-23). The Article1 campaign must be rebuilt
from the resulting SHA.

## Phase 5 low-overhead timing and implementation freeze: COMPLETE — 2026-09-21

Starting HEAD `8f54e6f` (tip of `phase4-pruning`); work on
`phase5-timing-freeze`. Seven commits: `62e4ce8` CUDA event separation, `776d26a` UniformFmm timing levels, `fa870c6` Python/C ABI, `244b341` benchmark drivers and the overhead record, `7688d6a` tests, `3fb36d2` examples, `62835b5` documentation.

**What it answered.** The always-on instrumentation cost a `CudaFull`
evaluation 12-20 us regardless of size (15 % of an 86-108 us evaluation,
3.3 % at 50k points, 20 % of the fastest standalone P2P plan; nothing
measurable on the CPU), measured against a temporary hard-off copy of the
starting HEAD before anything was redesigned.

**What changed.** `TimingLevel {Off, Coarse, Detailed}`, default `Off`, on
`UniformFmmOptions`, with a run-time `set_timing_level`. `Off` reads no clock
and records no diagnostic CUDA event; the three functional events (two
cross-stream waits, the completion point) are created without timestamps and
the diagnostic graph has its own twins, so the functional graph is identical
at every level. `Coarse` is host wall times (total, far field, near field,
setup totals); `Detailed` is the old depth. Every record names its level.
Python enum and setter; additive C ABI setter, with the wall-time accessor
failing while `Off`; Fortran untouched; NVTX untouched and independent.
Benchmark drivers take `--timing` (default `off`) and record it.

**Acceptance.** `Off` within 0.1-0.6 % of hard-off on every FMM workload,
`Coarse` at most 0.3 %, `Detailed` reproduces the old cost (+6 to +18 % on
sub-millisecond CUDA cases). Numerical results, resolved policy, cache keys
and persisted cache bytes are identical across levels (tested). Record:
`benchmarks/baselines/phase5-timing/` (NOT ARTICLE1).

**Validation.** All four warning-as-error configurations, zero warnings:
GCC 13 CI reproduction 252/252 CTest and 173/9 pytest; CPU + oneMKL 252/252
and 162/7; CUDA without oneMKL 252/252; CUDA + oneMKL 252/252 and 181/1 with
the tutorials executed;
`compute-sanitizer` memcheck, racecheck, initcheck and synccheck over the 35 CUDA- and timing-tagged C++ test cases (41 312 assertions): 0 errors, 0 hazards each. `sphinx-build -W` and `git diff --check` clean. Unvalidated:
Fortran (no compiler), MSVC/Windows.

**Closure follow-up (same day).** Review found three residues: construction
clocks outside the `UniformFmm` gate (`UniformTree` 13 reads per tree, 26
per plan; `AdaptiveTree` 3; the dense-direct records 8-10, CUDA 8 more), a
benchmark driver that wrote its own clock into internal-looking phase columns
for the direct references, and an undocumented compatibility status for the C
accessor. Measured first against a hard-off copy: about 17 ns per read,
resolvable only on a 6 us standalone tree (+3.8 %), inside the noise on every
plan and dense build. Gated anyway where a plan owns the clock or a benchmark
constructs the object: `UniformTreeOptions::collect_build_timings` (default
true; set from the plan's level, `Detailed` only), a trailing
`TimingLevel timing_level = Off` on `DenseDirectPlan`/`CudaDenseDirectPlan`;
`AdaptiveTree` left alone and documented. `benchmark_uniform_fmm` keeps its
external clock apart from the solver's records and names the collector in a
new trailing `internal_timing_source` column. `docs/c-and-fortran.md` states
ABI-compatible / source-compatible / behaviour changed. Re-measured: `Off`
equals hard-off (6 us tree to 0.01 us; dense builds at or below hard-off
single-threaded). Tests at the tree, dense, CUDA dense, C ABI, Python and
benchmark-CSV layers. Record: section M of the Phase-5 chapter in
`performance_optimization.md`;
`benchmarks/baselines/phase5-timing/construction_clocks_*.csv`.

**Validation of the follow-up.** Four warning-as-error configurations rebuilt, zero warnings each: portable CPU CI reproduction (conda-forge g++ 13.4, Unix Makefiles, LTO) 255/255 CTest, 178 passed / 9 skipped pytest; CPU + oneMKL without CUDA 255/255, 167 passed / 7 skipped (notebooks excluded); CUDA without oneMKL (`cuda` preset) 255/255; CUDA + oneMKL (`notebooks` preset plus benchmarks, g++ 15.3 / nvcc 13.3) 255/255, 186 passed / 1 skipped including the six executed tutorials and the benchmark-CSV tests against the rebuilt driver. The three new CTest cases are the tree opt-out and the two dense timing cases. `compute-sanitizer` memcheck, racecheck, initcheck and synccheck over the `[dense]` and `[timing]` groups (12 cases, 9 643 assertions), since the CUDA dense constructor changed: 0 errors, 0 hazards each; no other CUDA source, event or synchronisation code changed, so the Phase-5 sanitizer campaign was not repeated. `sphinx-build -W` clean, `git diff --check` clean. Unvalidated, as before: Fortran (no compiler here; `fortran/` untouched), MSVC/Windows.

**Implementation FROZEN FOR ARTICLE1 BENCHMARKING at
`48c2142`** (supersedes `62835b5`; CI green on `d989270` (run 35578346697, both jobs: portable CPU build and tests, first-party warning surface); the run on the final HEAD that adds this record is listed below).
Next: fast-forward into `refactor/architecture-v0.2` after review, then the
Article1 benchmark campaign.

## Phase 4 repository pruning and documentation cleanup: COMPLETE — 2026-09-20

Starting HEAD `ad48459` on `refactor/architecture-v0.2`; work on
`phase4-pruning`, eight commits. The last phase before the Article1 campaign:
make the repository comprehensible without removing anything a benchmark or a
paper might later need.

**Nothing executable was removed.** Every backend (portable CPU, oneMKL,
`CudaPartial`, `CudaFull`, dense direct on CPU/oneMKL/CUDA), every P2P
packing and dictionary executor, both point-expansion representations, both
expansion bases, both precisions, every explicit override and every benchmark
driver stands exactly as Phase 3D left it. What was pruned is *duplication*:
two dead member groups, a duplicated classifier, 20 notebooks, 14
documentation pages, and a test that spent 496 s proving less than its 7.5 s
replacement proves.

**Source (4A).** Four of the five items Phase 3D deferred are resolved and
the fifth was attempted, measured and deliberately left alone.
`UniformFmm::use_cuboid_p2m_`/`use_cuboid_l2p_` and
`CudaExecutionPolicyInputs::{periodic, bsr_estimate_bytes, bsr_budget_bytes}`
were written and never read, so they are gone; the unused three-argument
`far_field_stream_priority` overload is gone with its prose moved onto the
five-argument rule and its test ported;
`cuda_policy::resolve_cuda_execution_policy` keeps its name although it also
settles the CPU dictionary; the ownership is now stated in a comment rather
than renamed across a public-adjacent surface.

The fifth was the finite P2M/L2P endpoint classifier in
`plan_preparation.cpp`, which is the same first-seen-numbering algorithm as
`operators/exact_operator_reuse.hpp` with a different key container. Merging
it built and passed everything locally under g++ 15.3 in four
configurations, and then **failed CI at link time**: with GCC 13 the LTO
plugin marks ordinary `std::vector`/`std::array` COMDATs as prevailing in two
archive members at once (`src/plan/direct/dense.cpp.o` and
`src/fmm/plan_preparation.cpp.o`) and `lto1` aborts with `multiple prevailing
defs for 'allocate'`. The merge is not what is wrong — the same fifteen
symbols are emitted by both translation units before and after — but it
perturbs the plugin's resolution enough to trip the bug. Since the supported
CI toolchain is GCC 13, the duplication stays and the merge is deferred with
this note; `AGENTS.md` records what to check before attempting it again. `resolve_point_expansion_
execution()` moved ahead of `build_static_plan()` so that an impossible
`Procedural` request is rejected before an order-11 operator bank is built,
which is also why the procedural validation test fell from 178 s to 0.34 s.

**Tests (4B).** Serial CTest falls from 550 s to 44 s with the same 244 cases
and no lost coverage. Almost all of it was one file: the procedural-expansion
exactness test compared plan output on two 2000-point scenes, which bought
nothing an 11-point leaf does not, and paid for an order-10 universal bank
four times over. It is now a kernel-level test of
`ProceduralPointExpansion::apply_p2m`/`apply_l2p` against the canonical rows
at *every* compiled order 1-10, plus a plan-level loop over orders {1, 3, 6}
that absorbed the separate potential test. The per-layer identity and
finite-self coverage, the nine-pair geometry matrix, the two exact-reuse
tests and the eight per-header self-containment tests were examined and kept:
they protect different builders, not the same one twice.

**Notebooks (4C).** 29 notebooks in three directories became six canonical
tutorials under `examples/tutorials/` (getting started, finite geometry,
backends and execution, cache and periodicity, trees and parameters, operator
chain), two MagTense comparisons under `examples/validation/`, and the FMM3D
comparison material under `benchmarks/external/fmm3d/` where it belongs with
the other Article1 raw material. Every tutorial runs on the portable build in
seconds, guards its CUDA/oneMKL cells, uses only the public API, and is
*executed* by `python_tests/test_tutorial_notebooks.py`; the old
notebook-contract tests, which parsed notebooks that no longer exist, are
gone. Three teaching bugs were fixed while merging: lattice points sitting on
octree box boundaries (which made the reported errors look bad), a claim that
a uniformly magnetised periodic cell gives `H = 0` for point dipoles (it
gives the Lorentz field `+M/3`; cubes filling the cell give 0), and an FP64
accuracy floor quoted as `1e-15` when the canonical 1e-9 grid puts it at
`6e-9`.

**Documentation (4D).** 28 user/developer pages became 14 plus `api.rst`, and
281 KB became 134 KB, while every unique mathematical statement was moved
rather than deleted: `math/conventions.md`, `math/spherical-expansions.md`
and `math/finite-geometry.md` now hold the normative formulas, including the
root normalisation, the FP32 root-width scaling and the periodic Ewald
construction. `architecture.md` is a concise description of the current tree;
the Phase-1/2 closure narratives, the static-P2P sweep history, the CUDA M2L
performance report and the v0.1.0 baseline moved verbatim to
`agent_docs/architecture_history.md`. `backends.md` absorbed the P2P
capability matrix and the precision table; `benchmarks.md` absorbed
profiling; the FMM3D/MagTense installation notes point at their new homes.

**Comments and Doxygen (4E).** Public headers went from 430 undocumented
members to zero, and the comment density under `src/` rose from 2111 to 3044
lines: file-level explanations of what each unit owns and why (the CUDA event
graph and its overlap, the identity semantics every P2P packing carries, the
canonical-grid bitwise reuse, the cache container's atomic publication, the
prism/tetrahedron formulations and their safeguards), plus the measured
reasons behind the execution policies. The only non-comment edits are three
compressed CUDA statements reformatted to the house style and one enum
definition split from its member declaration because Doxygen misparsed it.

**Validation.** Four configurations, each configured fresh with
`-DCDFMM_WARNINGS_AS_ERRORS=ON`:

| Configuration | Build warnings | CTest | pytest |
|---|---|---|---|
| portable CPU (`dev`) | 0 | 244/244 | 170 passed, 8 skipped |
| portable CPU + oneMKL | 0 | 244/244 | 172 passed, 6 skipped |
| CUDA | 0 | 244/244 | 175 passed, 3 skipped |
| CUDA + oneMKL | 0 | 244/244 | 177 passed, 1 skipped |

The last skip needs the separate `cdfmm-magtense` environment. The six
tutorials execute inside each of those pytest runs.
`sphinx-build -W --keep-going` is clean, and `git diff --check` is clean.

**The one regression this phase caused, and how it was found.** The first
CI run on the branch failed both jobs at the build step, with only
`collect2: error: ld returned 1 exit status` visible in the workflow's
annotations, because the grep that republishes diagnostics does not match
`lto1: fatal error:`. The failure was reproduced locally by building with the
conda GCC 13.4 toolchain using CI's exact configuration (Unix Makefiles, LTO,
`-Werror`, unlimited `-j`), which also ruled out memory exhaustion: peak RSS
was 394 MB. A control build of the start commit with the same toolchain
passed, so the regression was the branch's. The duplicate-archive-member
theory was tested and **refuted** (rebuilding the archive with unique member
names changes nothing), and the linker's own resolution file then named the
two conflicting members outright. Reverting only the classifier merge makes
the GCC 13 build clean again, which is the state this branch ships.

**Nothing in the public or persistent surface changed.** `git diff
ad48459..HEAD` is empty over `fortran/`, `python/` and
`include/cdfmm/c_api.h`; `src/cache/` and `src/bindings/` have zero
non-comment changed lines; `libcdfmm_c.so` exports the same fourteen
`cdfmm_*` symbols against `CDFMM_ABI_VERSION 1`; and `sizeof(UniformFmm)` is
7416 as before.

**One measured caveat, which predates Phase 4.** A 25-file cache corpus
produces identical file names — the keys — before and after, and 17 of the 25
files are byte-identical. The eight that carry FP64 operator values (the four
universal banks and the four FP64 geometry plans) differ. This is *not* a
Phase-4 change: building the **same** HEAD source in two different trees with
the same configure line reproduces exactly the same eight-file difference,
while running one binary twice is byte-identical, so it is build-to-build
variation of the LTO Release build, most likely from LTO partitioning
changing floating-point contraction. The user-visible size of it is 1 ULP on
`H`, 2 ULP on `phi`, with the root multipole bitwise identical, and the FP32
plans are byte-identical because quantisation absorbs it. Recorded here
because it means cache files are portable between builds but not
bit-reproducible across them.

**Not validated, and recorded as such:** the Fortran interface (no Fortran
compiler exists in this environment; `fortran/` is byte-identical to
`ad48459`), MSVC/Windows, and CUDA sanitizers — no CUDA code changed beyond
comments and three reformatted statements.

**Next: the Article1 publication campaign** against this frozen
implementation, with its own measurement protocol. The benchmark drivers,
runners, analysers and `benchmarks/baselines/phase3d/` were deliberately left
untouched.

## Pre-pruning closure after Phase 3D: COMPLETE — 2026-09-19

Starting HEAD `51b2434`. Not Phase 4: nothing was pruned. This pass closes the
four review findings against the Phase-3D drivers, makes every first-party
compilation path warning-clean, and repairs a GitHub Actions workflow that had
been failing on every branch since 2026-09-07 — including the integration
branch and the Phase-3D head.

**No production policy changed.** All fourteen automatic policies stand as
Phase 3D measured them; the only behavioural line added to `src/` is a `break`
after a call that already throws.

The Phase-3D driver now records `comparison_group` and `comparison_variant`
instead of letting the analyser guess a group from the display name, which had
merged a regular prism lattice with an irregular prism cloud; it fails the run
on a case that cannot run instead of writing a short CSV and exiting zero; and
it resumes only against a scratch manifest of the revision, the binary's
SHA-256 and the sampling settings. The retained baseline was **not**
regenerated: its 158 case names, their order and its numbers are unchanged,
and the analyser reconstructs the legacy grouping from the driver's own
generators. A stale claim about the integration branch's state was corrected.

The project previously set no compiler warning flags at all.
`cdfmm_enable_warnings()` now applies `-Wall -Wextra` per first-party target,
with `CDFMM_WARNINGS_AS_ERRORS` off by default and on in CI. Thirty-three
diagnostics were found and all were fixed at the source; no suppression was
added and no warning level was lowered. Clean `-Werror` builds pass in
portable CPU, CPU + oneMKL, CUDA, CUDA + oneMKL and `Debug` under g++ 15.3.0,
and in portable CPU under conda g++ 13.4.0.

**Not validated, and recorded as such:** the Fortran interface could not be
compiled — no Fortran compiler is installed in this environment — and MSVC and
Windows were not exercised. `fortran/` is byte-identical to `51b2434`.

Both GitHub Actions jobs are green on the closure head `bbe3926`, the first
green run since 2026-09-07, with wider coverage than the workflow it replaces.

Validation on the final tree: `ctest` 244/244, `pytest python_tests` 171
passed and 1 skipped, a clean `git diff --check`, and a representative CUDA
sanitizer matrix with no errors or hazards.

No public C++ API, C ABI, Python API, Fortran interface, cache format or cache
key changed; `git diff 51b2434..HEAD` over `include/`, `src/bindings/`,
`src/cache/`, `python/` and `fortran/` is empty.

**Next: Phase 4 repository pruning and documentation cleanup.**

## Final cross-backend integration and production policy (Phase 3D): COMPLETE — 2026-09-18

Starting HEAD `49b5fe3`. The last optimisation phase before repository
pruning: reconcile the measured history of Phase 3 with what the code
actually does, and establish a final engineering baseline.

All fourteen automatic production policies resolve exactly as the Phase-3
measurements intend, on every backend and both precisions, and each is
confirmed against its credible forced alternative on the phase that rule
governs. **No automatic performance policy changed.** Point pairs recompute
from positions on the CPU and on CUDA FP32 (8.4x faster on the CUDA P2P phase,
14x less device memory) and keep stored tensors on CUDA FP64 (recomputation is
1.30x slower there); point expansions are procedural everywhere except
`CudaFull` FP64 (procedural is 1.39x slower on P2M+L2P); finite operators are
precomputed everywhere; `RegularGrid` selects the signed dictionary, verified
against the built plan's token width; portable CPU M2L remains the default
with oneMKL explicit-only; `ExecutionBackend::Auto` remains `CpuStatic` and
never selects a CUDA backend.

Three defects were found and fixed. `ExactOperatorClasses` narrowed
`std::size_t` pair indices into `std::uint32_t` unguarded, which a dense plan
passes at roughly 65536 bodies per side; classification is now abandoned
rather than wrapped, keeping the compact per-pair class map. An explicit FP32
`CudaBsr3` request under a lowered `cuda_p2p_bsr_max_bytes` reported success at
construction and then failed the evaluation, because the budget gated a
prebuild the executor consumed unconditionally — the documented contract was
already that `p2p_packing` outranks the budget. And a stale example notebook
raised `AttributeError` on two options that no longer exist.

Startup is the part that moved. Phase 3C's "clearest next lead", warm derived
packing at 0.469 s of a 0.978 s warm setup at 32,768 bodies, no longer
reproduces: the warm setup is 0.419 s and its derived packing is 0.00 ms. The
largest remaining warm cost is the FP32 precision conversion at 167 ms, whose
removal needs a cache-format change and is recorded for post-v0.2. The
universal operator bank remains the dominant truly-cold cost (11.8 s at p = 8)
and is fully removed by its cache.

No public API, C ABI, Python API, Fortran interface, cache format or cache key
changed. Validation ran four fresh pinned trees (portable CPU, oneMKL, CUDA,
CUDA + oneMKL) with the full CTest and Python suites, plus Compute Sanitizer
over the changed FP32 BSR construction path. `benchmarks/baselines/phase3d/`
retains the numbers as an engineering regression baseline, explicitly not an
Article1 benchmark. Phase 4 repository pruning is NEXT; the publication
campaign follows it.

## Dense/all-to-all construction optimization (Phase 3C.5): COMPLETE — 2026-09-18

Starting HEAD `cefc975`. Phase 3C optimised the FMM hierarchy's construction
and left `DenseDirectPlan`, the exact dense all-to-all baseline, untouched.
The audit found one construction path serving all three backends —
`CudaDenseDirectPlan` builds a complete host plan and uploads it, and portable
CPU and oneMKL differ only in `evaluate()` — an FP32 path that quantises at
the point of store and so has no conversion pass to optimise, and a pair loop
that was already parallel. What was missing was reuse: one exact tensor was
evaluated per pair, although a pair tensor is a pure function of the
displacement, the two body records and whether the pair is an omitted point
self interaction. Construction now prepares each distinct finite record once,
classifies pairs by those exact bits, builds one tensor per class in parallel
and scatters into the entries each pair already owned.

Cold construction, eight P-cores, output bit-identical throughout: prism
lattice 512² 3.07 s -> 0.046 s (67x), asymmetric 1024x512 6.07 s -> 0.072 s
(85x), tetrahedron-prism lattice 256² 7.42 s -> 0.128 s (58x), tetrahedron
lattice 256² 3.30 s -> 0.087 s (38x), prism-to-point 1024² 0.34 s -> 0.024 s
(14x). A locally refined grid reaches 10-13x and a refined anisotropic one
5.4-5.7x. CUDA inherits these because it delegates to the host plan; its
context creation and device allocation are a fraction of a millisecond and the
upload is 1% or less of setup for every finite geometry. Repeated evaluation
is unchanged at 0.94-1.02x across portable CPU, oneMKL and CUDA in both
precisions. Where reuse cannot help the parallel build scales 7.9-8.0x on
eight cores.

Two limits are structural rather than incidental, and both are recorded with
their numbers. Point-to-point plans never classify: a point pair costs about
4 ns against 260 ns for the cheapest finite pair, so the key lookup would cost
more than the arithmetic it saves, and 77x redundancy is deliberately left
unexploited. Irregular geometry has no exact reuse at all, because in an
all-to-all plan every pair holds a unique combination of source and target
record — a sharper limit than the FMM near field, where a neighbourhood list
still left 19.7x on irregular prisms.

Widening the workloads mid-phase changed two conclusions. Anisotropic spacing
costs reuse through floating-point rounding alone: once the spacing is not
exactly representable, equal index differences no longer produce equal
displacement bits, and a point plan's distinct-input count rises 4.5x. Reuse
stays exact and results stay bit-identical; it simply finds less, and unlike
the FMM path `DenseDirectPlan` has no canonical-grid normalisation to hide it,
so bitwise agreement rests on how the caller generated its positions. And a
locally refined grid sits at about 5.5x reuse, between the two extremes the
original workloads reported. The classification gate had been written as a
fixed eightfold reuse requirement justified by those extremes alone, and it
discarded exactly that case: an exact prism-to-tetrahedron build stayed at
5.11 s when 0.91 s was available. The gate is now a byte budget on transient
tensor storage — half the matrix bytes the plan retains anyway — which keeps
the intermediate cases and still bounds the memory; the worst measured
transient ratio is 0.35 against a cap of 0.5.

Correctness is bitwise against the pinned starting-SHA build over 150
configurations: nine geometry pairs, seven workloads, both precisions,
asymmetric counts, and plans with and without an identity map, with throwing
configurations compared on their exception message.
`tests/test_dense_direct_exact_reuse.cpp` pins the contract at the lowest
layer, including one-ULP inputs that must not alias and the identity semantics
that must come from the map rather than from coordinates. No public API, C
ABI, Python API, Fortran interface or cache format changed; the construction
diagnostics the benchmark reads are internal to `src/`.

`benchmarks/benchmark_dense_direct_construction` and
`run_dense_construction_matrix.py` remain in tree for Article1: they report
setup and repeated evaluation separately, decomposed by phase, with host and
device bytes and ns per pair for both halves.

**Phase 3D, the final cross-backend review, is NEXT.**


## Static-plan construction and plan preparation (Phase 3C): COMPLETE — 2026-09-18

Starting HEAD `a2af367`. Cold plan construction was dominated by building the
same operator over and over. An exact near-field pair tensor is a pure
function of the displacement and the participating body records, with any
periodic image shift folded into the displacement; a finite P2M or L2P
operator is a pure function of the expansion basis, the body's shape record
and its offset from its leaf centre. Normalisation puts body centres on a
canonical grid, so equivalent interactions agree *bitwise* rather than
approximately: a 4096-body regular prism lattice reaches its 681,472
near-field pairs from 343 distinct displacements, and 343 survive even when
every body carries its own size record. All three canonical pair loops and
both finite endpoint loops now classify by those exact bit patterns, build
each distinct operator once, and do so in parallel; the generic pair loop and
both endpoint loops had been serial with no data dependency forcing it.

Cold construction, 4096 bodies, order 6, FP32, eight P-cores, cache disabled:
regular prism lattice 66.81 s -> 1.52 s (44x), irregular prisms 66.11 s ->
1.95 s (34x), regular tetrahedra 15.87 s -> 1.54 s (10x). Irregular
tetrahedra have no exact duplicates to find and gain only the rebalanced
schedule (19.7 s -> 16.1 s), which is the honest limit of the mechanism.
Under exact finite far-field models the endpoint operators fall by two orders
of magnitude: the order-6 tetrahedron case goes from 17.6 s to 1.58 s, with
P2M 436x and L2P 486x. Where reuse cannot help, the parallel build scales
7.8x on eight cores.

Correctness is pinned by byte-comparing the persisted geometry plan, which
serialises the canonical P2P blocks, the P2M plans and the L2P evaluators:
39 configurations are byte identical across all nine geometry pairs, both
layouts, irregular bodies, exact and point far-field models, FP32 and FP64,
orders 4, 6 and 8, the Cartesian basis, and periodic evaluation. A new
`tests/test_p2p_exact_reuse.cpp` pins the reuse contract itself, including
inputs one ULP apart that must not alias.

Two things were deliberately not done and are recorded with their evidence.
The production prism tensor keeps its `long double` arithmetic: after reuse
the exact evaluation is a small fraction of a near-field stage that is itself
under a tenth of construction, so the 2.7x Phase 3B.5b measured for `double`
no longer justifies changing a numerical contract. Skipping the canonical
near-field build for point plans is real and unblocked in process, but the
geometry cache serialises the operator unconditionally under a key that
distinguishes neither packing nor backend, so a skipped build would silently
hand a later `CanonicalAos` or CUDA plan an empty near field; that is a
persistent-cache change this phase is not authorised to make.

Left in tree for Article1: `benchmarks/run_construction_matrix.py`, a
resumable cold-construction matrix recording per-phase timings, plan bytes,
peak resident set and an evaluation median, and the benchmark's new
`--far-field-model exact`, without which the exact finite endpoint operators
were never exercised at all. `StaticPlanStatistics` gained four timers that
decompose the near-field stage and the precision conversion. No cache format,
cache key, C ABI, Python API or Fortran interface changed. Full record in
`agent_docs/performance_optimization.md`. **Phase 3D, the final cross-backend
review, is NEXT.**

## Exact finite-geometry procedural vs precomputed operators (Phase 3B.5b): COMPLETE — 2026-09-17

Starting HEAD `551c790`. The finite counterpart of Phase 3B.5, and the answer
is the expected one, now measured: precomputation wins for every exact finite
operator. All eight finite-containing P2P pairs, plus finite P2M and L2P for
prisms and tetrahedra at orders 4-10, were benchmarked in both
representations, hot and streaming, FP32 and FP64, on the CPU and (for the one
device-practical family) on CUDA. Reconstruction costs 150x to 1,300,000x a
stored apply per field update, and because construction is one pass of the
same arithmetic, precomputation amortises within one to ten complete updates
on the CPU. The gap narrows but does not close when the stored operator
leaves the cache: with 176 MB of tensors, four times the LLC, it is still
840x. On CUDA the prism point tensor closes to 2.5-5.7x in FP32 arithmetic
while retaining no operator, but it is nine times less accurate, 68-196x
slower in FP64, and needs 30,000-38,000 updates in one run to repay a
construction the persistent cache already pays once. Finite operators
therefore stay precomputed on every backend, no production policy or executor
was added, and `PointExpansionExecution` keeps its point-only semantics. The
one accepted production change is
`src/geometry/primitives/rectangular_prism_point_kernel.hpp`, the
precision-generic MagTense prism point tensor that production instantiates in
`long double` (bit-identical over 20,680 tensors) and that the benchmark
instantiates in `double`, `float` and on the device, so the mathematics is not
duplicated. Left behind for Article1: `benchmark_operator_representation`,
its CUDA kernel, `run_operator_representation.py` and
`analyse_operator_representation.py`, which reproduce every table from the
final refactor code. Full record in `agent_docs/performance_optimization.md`.
**Phase 3C construction / plan preparation is NEXT.**

## Procedural vs precomputed point operators (Phase 3B.5): COMPLETE — 2026-09-17

Starting HEAD `38f1b98`. Precomputation is now an explicit execution choice
rather than a mathematical requirement: point P2P, point P2M and point L2P
have procedural representations selected by measured policies with explicit
overrides, while finite prism/tetrahedron operators stay precomputed. CUDA
backends gained the position-based `PointGeometry` P2P (FP32 default for point
pairs; 1.3-7.5x faster than leaf blocks on random points, never slower than
the lattice dictionary in evaluation time, 2-33x less device memory; FP64
keeps stored tensors). Point P2M/L2P are recomputed from a shared
allocation-free solid-harmonic recurrence on the CPU hierarchy (both
precisions) and on FP32 `CudaFull` (`UniformFmmOptions::
point_expansion_execution`; 3-7x faster stages from p 6, 588 bytes of tables
instead of 12-24 C bytes per point). End to end (FP32, `Auto`): `cuda-full`
S/M/L/128-per-leaf 1.5x / 1.5x / 2.9x / 6.9x faster than the previous HEAD
with 17-350x less device memory, `cuda-partial` M 1.56x; `cuda-full` now beats
the hybrid in every measured regime. Cache format/keys, C ABI and Fortran
interface unchanged. Full record in `agent_docs/performance_optimization.md`.
**Phase 3C construction / plan preparation is NEXT.**

## P2P execution unification, full geometry/backend coverage, CudaPartial crossover: COMPLETE — 2026-09-16

Starting HEAD `af50b69`. The near-field invariant "geometry builds tensors,
executors apply tensors" is now enforced: all nine point/prism/tetrahedron
source-target pairs build canonical tensors (prism/tetrahedron via the
tetrahedron pair's polyhedron surface formulation, plus a far-separation
Gauss safeguard and a conditioning fix of the tetrahedron point kernel on
edge lines), the dense leaf packing carries the canonical identity marker so
every stored-tensor executor obeys metadata instead of geometry, periodic
image records pack into leaf blocks / merged BSR blocks / dictionary tokens,
and `UniformFmmOptions::p2p_packing` forces any packing (Python `AUTO`,
`requested_p2p_packing`). `tests/test_p2p_geometry_matrix.cpp` runs every
pair through every packing of CpuStatic, CudaPartial and CudaFull in both
precisions, free-space and periodic, against `DenseDirectPlan`.
Policy, on measurements: periodic point plans use `PointGeometry` (2.7-5.1x
faster near field), leaf blocks are the general CUDA default for every
geometry (BSR(3) explicit only), the `RegularGrid` hint selects the
dictionary for any geometry on every backend guarded by the built plan's
token width (2-7.5x faster than SoA rows on finite lattices, 3x faster than
leaf blocks on CUDA), and the hybrid's M2L stream / the full backend's far
field (when P2P-dominated) run at high stream priority (`cuda-partial`
-4..-11 %, `cuda-full` -2..-10 %). `cuda-partial` never beats `cuda-full`
except by 1-2 % at 128-160 points per leaf; recorded as Phase-3D input.
Construction findings recorded, not optimised (serial prism pair loop:
600 s for 32^3 prisms). Full record in
`agent_docs/performance_optimization.md`; capability table in
`docs/static-p2p.md`. **Phase 3C construction / plan preparation is NEXT.**

## CPU / oneMKL evaluation optimization (Phase 3B): COMPLETE — 2026-09-16

Against `CPU_PERF_BASELINE = e4f1c79`: portable CPU evaluation 3.5-10x faster
at 8 threads (and 6.5-7.8x single-threaded), oneMKL 1.8-4.1x, `cuda-partial`
1.4-5.7x, with unchanged accuracy. Changes: per-target portable M2L, dense
level-scaled far-field packing, oneMKL parallel scatter and level ranges,
class-sorted M2L block schedule, position-based point-source P2P
(`P2PExecutionPacking::PointGeometry`, no resident pair tensors on CPU point
plans). The portable executor now beats oneMKL M2L at every measured size
(recorded for 3D; no automatic selection changed). Full record in
`agent_docs/performance_optimization.md`. **Phase 3A and 3B are COMPLETE;
3C construction / plan preparation is NEXT; 3D and Phase 4 remain unstarted.**

## Regular-grid dictionary policy at high occupancy (3A closure): COMPLETE — 2026-09-16

The automatic `SpatialLayout::RegularGrid` dictionary executor is now a
three-regime rule calibrated on lattices from 48 to 192 targets per leaf:
power-of-two microtiles below 48, target-owned from 48 to below 72,
source-warp from 72 upwards (source-warp is 1.1-2.1x faster than target-owned
at 80-192 per leaf in both precisions). Kernels, cache identity, `General`
defaults and explicit options are unchanged. **Phase 3A GPU evaluation is
CLOSED.** Phase 3B (CPU / oneMKL evaluation) is next; 3C, 3D and 4 remain
unstarted.

## CUDA execution policy and regular-grid hint (3A follow-up): COMPLETE — 2026-09-16

`src/backend/cuda/execution_policy.{hpp,cpp}` centralises the CUDA strategy
choices (P2P packing/executor, M2L pairs per thread, translation lane groups)
as a deterministic function of plan facts; `UniformFmmOptions::spatial_layout`
(`SpatialLayout::General` default, `RegularGrid` hint) selects the
reduced-symmetry dictionary and an occupancy-matched executor automatically on
lattices, within 1.4 % of the best explicit executor and 1.4-3.1x faster than
the general default where P2P dominates. General defaults unchanged. Phases
3B/3C/3D and 4 remain unstarted.

## GPU evaluation optimization (Phase 3A): COMPLETE — 2026-09-16

Repeated CUDA evaluation was profiled and optimised against the fixed
baseline `293144bf` on an RTX 5090: `cuda-full` FP32 2.2x-4.9x and FP64
1.3x-2.5x faster end to end, `cuda-partial` 1.07x-1.9x. Changes: grouped
shared-memory M2L, pinned-direct host staging, atomic-free far-field
kernels with per-level launches, pinned hybrid M2L staging, and a
warp-per-block leaf P2P packing that replaces cuSPARSE BSR(3) as the
point-source default (new `P2PExecutionPacking::LeafBlock`); finite sources
keep BSR(3). Full record in `agent_docs/performance_optimization.md`.
Phases 3B (CPU/oneMKL), 3C (construction), 3D and 4 remain unstarted.

## Public/internal API, header ownership, and packaging cleanup: COMPLETE — 2026-09-16

The last recorded Phase-2 cleanup item. `parameter_selection.hpp` now depends
only on `cdfmm/backend/execution.hpp` (new: `ExecutionBackend`, moved out of
`uniform_fmm.hpp`) and `cdfmm/math/vec3.hpp`, instead of the complete
`uniform_fmm.hpp`; `src/parameter_selection.cpp` gained an explicit include of
the full header it actually needs. CUDA availability queries
(`cuda_compiled`, `cuda_available`, `cuda_direct_available`,
`cuda_m2l_p2p_available`, `cuda_m2l_available`, `cuda_full_available`,
`cuda_device_description`) and `one_mkl_available` moved to new
`include/cdfmm/backend/cuda/availability.hpp`/`backend/mkl/availability.hpp`
headers; `uniform_fmm.hpp` includes both and re-exports every name, and
`src/backend/cuda/fmm/internal.hpp` now includes the narrow header instead of
the complete solver header, resolving the one reverse-dependency edge Phase 1
closure had recorded. `tensor_dictionary.hpp`'s Tensor6 primitives moved to
`include/cdfmm/plan/p2p/tensor_dictionary.hpp`, next to their only consumers.
`timings.hpp`, `periodic.hpp`, and `validation.hpp` were each audited in full
and kept flat: none has one clear subsystem owner (see
`docs/architecture.md`). `P2MPlan`/`FloatP2MPlan`/`ExpansionBasis` were
reviewed and kept at their cache-cleanup homes. A minimal downstream CMake
package (`find_package(cdfmm CONFIG)`, `cdfmm::cdfmm_c`) was added with no
`find_dependency` calls, since `cdfmm_c`'s exported interface has no
CUDA/oneMKL/OpenMP dependency to discover (those are private to the
never-installed static `cdfmm_core`); validated with an isolated-prefix
install and a standalone out-of-tree consumer reproducing the analytic
two-dipole field to `1.7e-16`. `docs/architecture.md`'s stale "Phase 1
closure" statements that `src/cache/internal.hpp` and
`src/backend/cuda/fmm/internal.hpp` still included `cdfmm/uniform_fmm.hpp`
were corrected (the cache one had been stale since the earlier
cache-boundary cleanup and was never fixed; the CUDA one is newly resolved by
this task). See `agent_docs/latest_session_work.md` for the full audit,
per-decision rationale, and validation evidence (portable CTest 197/198 with
the same pre-existing unrelated failure; CUDA+oneMKL CTest 198/198 with zero
skips; Python 137/7 portable, 143/1 CUDA+oneMKL; C ABI 14 symbols and Python
63 top-level names unchanged).

No solver algorithm, cache format, cache key, C ABI, Python API, or Fortran
interface changed. This closes the Phase-2 cleanup list: internal-duplication,
tree/topology, cache/`UniformFmm`, and now public/internal API/packaging are
all resolved. The next planned work is a dedicated performance-optimization
phase (construction and evaluation, CPU/oneMKL/CUDA), followed by repository
pruning — neither started here.

`Article1/` reappeared alongside `Article1_old/` during this session
(unexplained, apparently touched again by something outside this session);
left untouched and flagged, as in every prior session. The unrelated
untracked notebook in `examples/simple_notebooks/` also remains preserved.

## Cache/plan-preparation boundary cleanup: COMPLETE — 2026-09-16

Cache persistence (`initialise_cache_keys`, `load_universal_cache`,
`write_universal_cache`, `load_geometry_cache`, `write_geometry_cache`) was
five private `UniformFmm` member functions defined in `src/cache/*.cpp`,
which forced `src/cache/internal.hpp` to include the complete
`cdfmm/uniform_fmm.hpp`. A dependency audit (two read-only workers)
classified every field the five functions touched and confirmed none of them
is referenced outside `src/cache/*.cpp`/`src/fmm/*.cpp`, so no public/ABI/
Python/Fortran constraint bore on the redesign. Every cache entry point is
now a free function in `cdfmm::detail::cache` taking an explicit identity/
payload record (`CacheIdentityInputs`/`CacheIdentity`,
`UniversalCacheIdentity`/`UniversalCachePayload`,
`GeometryCacheIdentity`/`GeometryCachePayload`) built from specific
solver/plan/tree/topology types; `src/fmm/execution_setup.cpp` and
`plan_preparation.cpp` assemble these records and remain the sole callers, so
plan preparation remains the sole owner of the cache-vs-build decision. No
`UniformFmm` cache method survives; all five private declarations were
removed. `P2MPlan`/`FloatP2MPlan` moved from private nested `UniformFmm`
types to `include/cdfmm/plan/static_coefficient.hpp`; `ExpansionBasis` moved
from `uniform_fmm.hpp` to `include/cdfmm/core/precision.hpp`. Neither move
changes `sizeof(UniformFmm)` (verified: 7272 bytes before and after) or any
public/ABI/Python surface. Persistent format, cache keys, and all documented
cache invariants (direct FP32 decode, asymmetric FP32-only failure cleanup,
non-fatal miss/corruption behaviour) are unchanged. See
`agent_docs/latest_session_work.md` for full validation evidence: full CPU
CTest 198 total (193 passed, 4 expected skips, the same pre-existing
unrelated triangular-translation failure), Python 137/7; a cross-version
`v0.1.0` cache-compatibility probe with identical keys/hits/zero
writes/byte-identical files/bit-identical results; a before/after warm-cache
timing probe showing no measurable slowdown; and a full CUDA+oneMKL
`notebooks` run at 198/198 with zero skips.

Still open, each its own future task: the flat-header/packaging relocation
candidates (`periodic.cpp`, `parameter_selection.cpp`, `validation.cpp`, and
the headers awaiting a canonical subsystem home, including
`parameter_selection.hpp` including the complete `uniform_fmm.hpp`); CUDA
availability declarations and a CMake package/export interface; and
repository pruning.

An unrelated, pre-existing `Article1/` directory (untracked, ~640 GB) was
found renamed to `Article1_old/` during this session, apparently by a
concurrent unrelated process on this shared machine — not touched by
anything in this task. Left as found and flagged for the user.

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
