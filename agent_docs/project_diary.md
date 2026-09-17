# Project diary

## 2026-09-17 — the expected answer, and the one place it was close

Phase 3B.5b was asked to confirm something everyone believed: that exact
prism and tetrahedron operators are worth precomputing. Confirming a belief is
the kind of task where it is easy to measure loosely and call it done, so the
benchmark was built to be able to disprove it: the procedural side calls the
same production builders, keeps the per-body invariants the builders already
hoist, and every row carries its own error against the canonical operator. The
answer came back at three to six orders of magnitude, which is not a close
call.

Two things were worth the trouble anyway. The first is that the interesting
number was never the ratio but the break-even, and it is tiny on the CPU
precisely because construction *is* one procedural pass: a serially built
operator costs about eight parallel updates, a parallel one about one. That
reframes the result from "precomputation is faster" to "precomputation is
free after the first update", which is a much stronger statement for the
paper.

The second is the GPU. Porting the prism point tensor cost one
precision-generic header, and on the device the gap collapsed from 3000x to
between 2.5x and 5.7x, with the entire tensor array gone. For a moment it
looked like an exception worth integrating, until the amortisation was written
out: 30,000 updates per run to repay a construction the persistent cache
already pays once per geometry. The honest write-up is the one that gives
those numbers and the threshold they were judged against, so a later reader
with a memory-bound problem can disagree.

The bug of the day was mine and the benchmark caught it: the first device
kernel applied the point source's singular self pair instead of excluding it.
It produced beautifully plausible timings and an order-unity error, which is
exactly why every row carries a correctness column.

## 2026-09-16 — P2P unification: the invariant found real bugs, and the benchmark disagreed with the geometry names

The brief asked for an architectural invariant — geometry builds tensors,
executors apply tensors — and for a correctness matrix to enforce it. The
matrix earned its keep before it ever passed: point->tetrahedron on a
per-body scene disagreed with the dense reference by 40 % at one target. The
FMM and the reference used the same kernel, but the plan's canonical-grid
normalisation moved one point by a single ulp off the line through a
tetrahedron edge, and the MagTense-adapted edge primitives lost every digit
on that line. Neither implementation was "right"; both were sitting on a
removable singularity. The fix (a cancellation-free atanh difference and the
Van Oosterom solid angle for the normal term) is bitwise identical
elsewhere, and the far-separation probe that came out of the same test
exposed a second, older hazard for sparse grains. Lesson kept: when two
paths differ by a suspicious amount, find a third independent value before
deciding which one to trust.

The other lesson was that the execution policy had been written in geometry
nouns ("finite sources take BSR") because nobody had measured the
alternatives for finite bodies — they could not be executed there at all
until the leaf packing carried the identity marker. Once every packing could
run every geometry, the benchmark said the same thing for prisms and
tetrahedra as for points: leaf blocks over BSR, dictionary on lattices, and
a dictionary that does not compress is slower on the GPU. So the policy now
asks the built plan (token width) instead of the geometry type, exactly as
the brief hoped. `cuda-partial` did not find its crossover except at the very
edge (128-160 points per leaf, after fixing a stream-priority starvation the
timelines made obvious); that negative result is recorded as carefully as
the positive ones so Phase 3D does not have to rediscover it.

## 2026-09-16 — public/internal API cleanup: the honest answer was often "leave it"

The brief for this task read like a long TODO list — relocate `timings.hpp`,
relocate `periodic.hpp`, relocate `validation.hpp`, relocate every enum out
of `uniform_fmm.hpp` — but it also explicitly warned against mechanically
implementing every historical suggestion once the code shows one is
unnecessary. Taking that warning seriously changed the shape of the task:
about half of the candidate relocations turned out to have a genuinely good
reason to stay put, and the interesting work was in telling those cases apart
from the two that had a real, evidenced seam.

`parameter_selection.hpp` was the clean case. Reading the header's own
signatures (not the whole file it happened to include) showed it needed
exactly two things, `Vec3` and `ExecutionBackend`; everything else came along
for the ride because `#include "cdfmm/uniform_fmm.hpp"` was the path of least
resistance when the file was written. Moving `ExecutionBackend` to a new
`backend/execution.hpp` and swapping the include fixed the real problem
without inventing a `types.hpp` dumping ground for the four sibling enums
that had no consumer needing them narrowly — they stayed exactly where they
were, because moving something nobody depends on independently doesn't
narrow any dependency, it just adds an indirection.

The CUDA availability queries were the same shape of fix but more
satisfying, because the "why" was already written down: Phase 1 closure had
recorded `src/backend/cuda/fmm/internal.hpp -> cdfmm/uniform_fmm.hpp` as a
sanctioned exception, reasoning that a definition must see its declaration.
True, but that only justifies depending on *wherever the declaration lives*
— it doesn't justify the declaration living in the top-level solver header
in the first place. Checking what else that file used from `uniform_fmm.hpp`
(nothing — every other type it touched already arrived through headers it
included directly) confirmed the whole edge existed for three one-line
function declarations. Moving those three lines to a header that already had
a sibling precedent (`backend/cuda/dense_direct.hpp` already declares
`cuda_dense_direct_available()` next to the plan it describes) deleted a
documented architectural exception outright rather than re-justifying it
again.

`timings.hpp`, `periodic.hpp`, and `validation.hpp` were the opposite
lesson. Each had been carried in the "deferred, not overdue" bucket since
Phase 1, which reads, if you don't check, like "nobody's gotten around to
these yet." Reading each one in full instead said something more specific:
`timings.hpp` is one header because three different layers' diagnostics are
*read together* by one observability surface, and splitting it would scatter
a single concept across three directories to satisfy a rule about flat
files, not to help anyone. `periodic.hpp` is flat because its two real
consumers, `tree` and `operators`, are siblings in the architecture's own
layering diagram — giving it to either one manufactures exactly the
sibling-to-sibling edge that diagram exists to prevent. `validation.hpp`
already says in its own doc comment that it isn't production code. None of
these needed a subsystem home invented for them; they needed someone to
finish reading them once and write down why "flat" is the actual answer,
not a placeholder for one.

The CMake package work had its own small surprise: I expected to need
`find_dependency(CUDAToolkit)`/`find_dependency(MKL)` behind some
conditional, and went looking for where to put it — but `cdfmm_c`, the only
installed target, links its CUDA/oneMKL-dependent static implementation
library `PRIVATE` and exports no link libraries of its own. There was
nothing to find-dependency, because the exported interface genuinely has no
such dependency; adding one "to be safe" would have been describing a
dependency that doesn't exist. Validating that against a real isolated
install and a real out-of-tree `find_package` consumer (rather than trusting
the CMake reasoning alone) was worth doing anyway — it surfaced an unrelated
environment artifact (the IntelLLVM toolchain embeds `$CONDA_PREFIX/lib` in
the consumer's `RPATH`, ahead of the fresh install, so the first run silently
exercised a stale library from a previous session). `LD_PRELOAD` forced
resolution to the actual freshly built library and its own diagnostics
confirmed it. A second, unrelated red herring: the first CUDA+oneMKL build
under unbounded `-j` hit `nvcc` front-end internal-compiler-errors on four
different `.cu` files, none of which I had touched in three of the four
cases. Recompiling one of them alone, with identical flags, succeeded — a
parallelism/resource artifact of this environment's `nvcc`+GCC-15.3
combination, not a defect, confirmed before spending any time suspecting the
header change.

## 2026-09-16 — cache/plan-preparation boundary: narrow records, not a service class

The brief for this task was explicit about what *not* to do: don't reach for
`CacheManager`/`CacheService`/`CachePimpl` just because the name sounds clean,
and don't build a friend-class bridge that hands a new type unrestricted
access to `UniformFmm` and calls it encapsulation. That framing turned out to
matter, because the obvious lazy fix — `friend class CacheManager` wrapping
the same five method bodies unchanged — would have satisfied none of the
actual goals.

Two read-only workers mapped the territory before any design decision: one
classified every field the five cache functions (`initialise_cache_keys`,
`load_universal_cache`, `write_universal_cache`, `load_geometry_cache`,
`write_geometry_cache`) read or wrote from `UniformFmm`, sorting each into
identity state, persisted payload, statistics, or unrelated plan/lifecycle
data they merely happened to see; the other confirmed none of the five were
referenced anywhere outside `src/cache/*.cpp` and `src/fmm/*.cpp` — no test,
binding, or public accessor calls them directly (they're private), so the
entire public/C-ABI/Python/Fortran surface was unconstrained by whatever
redesign followed.

The map made the shape of the fix obvious: each function already reads a
specific, enumerable set of fields and writes another specific set. That's
not "the whole `UniformFmm` object" — it just looked that way because the
functions happened to be members. So the fix is free functions in
`cdfmm::detail::cache` taking explicit records (`CacheIdentityInputs`/
`CacheIdentity`, `UniversalCacheIdentity`/`UniversalCachePayload`,
`GeometryCacheIdentity`/`GeometryCachePayload`) built from references to the
exact fields each function touches — no new copy of solver state, no
service object owning anything, no friendship required, because the free
functions never see `UniformFmm` at all, only the records passed to them.
`src/fmm/execution_setup.cpp` and `plan_preparation.cpp` build those records
and remain the only callers, which is exactly the "plan preparation owns
cache-vs-build" property the brief asked for, just made structurally true
instead of true by convention.

One real obstacle: `P2MPlan`/`FloatP2MPlan` were private nested types of
`UniformFmm`, and the geometry payload needs to reference them by value.
Nested-and-private meant no free function could name the type without seeing
the whole class. The honest fix wasn't a workaround — it was recognising
these were misplaced from the start: they're canonical static-plan records
(leaf index, offsets, a coefficient map), the same category as everything
else already living in `include/cdfmm/plan/static_coefficient.hpp`. Moved
them there as free-standing types. Same story for `ExpansionBasis`, which had
been declared directly inside `uniform_fmm.hpp` with no header of its own,
even though `CacheDescriptor` needed it — moved to
`include/cdfmm/core/precision.hpp` next to `StaticPrecision`, the enum it's
conceptually paired with. Neither move touches `sizeof(UniformFmm)` (checked:
7272 bytes before and after) or any public name's meaning; both types were
already reachable only through `UniformFmm`'s public API by value, never by
name.

The old-cache compatibility check was the most satisfying part to run: a
throwaway probe, compiled unchanged against an isolated `v0.1.0` worktree and
then against this HEAD, wrote an 8-file corpus cold and read it back warm —
identical keys, every hit, zero bytes written, files byte-identical by
`sha256sum`, fields bit-identical. That's the strongest evidence available
that moving *who calls what* didn't perturb *what gets written*, which is the
one thing this task was absolutely not allowed to touch.

## 2026-09-16 — tree/topology boundary cleanup: the seam wasn't there

The brief for this task assumed a coupling problem existed: `AdaptiveTree`
returns `StaticFmmTopology` directly, its header calls this "a transitional
tree-to-plan coupling," and `docs/architecture.md` had repeated that framing
in half a dozen places since Phase 1. The instructions were explicit that
this might be a false alarm — "do not assume `StaticFmmTopology` is wrongly
owned merely because its name contains `Fmm`" — and that turned out to be
exactly what happened.

Two read-only workers did the legwork: one classified every field of
`StaticFmmTopology` (nodes, permutations, M2M/L2L edges, M2L interactions,
P2P leaf records, coordinate normalisation) against the tree/topology/plan
taxonomy, the other traced every producer and consumer across `fmm/`,
`cache/`, `plan/`, `python/`, and the test suites. Neither found anything
resembling plan or backend data inside the type. The clinching piece of
evidence wasn't even something I asked for: the Python test suite already
asserts `plan.topology is plan.topology` — object identity through
`shared_ptr` — which only makes sense if the FMM layer treats topology as
something it borrows, not something it owns or rebuilds. That single
assertion says more about the actual boundary than any amount of comment
archaeology.

The one place I expected to find real coupling — `AdaptiveTree`'s
200-line constructor building nodes, edges, and interactions all in one
function — turned out to already be two clean passes with no shared state
beyond the topology object itself, and the code already times them
separately (`tree_seconds_` vs `interaction_seconds_`, split at exactly the
boundary a refactor would have introduced). I drafted a plan to extract that
into two named functions before checking `UniformTree::build` for
comparison, and found it's the same shape: one big function, internally
phased, timed per phase, never split into multiple top-level functions.
Splitting `AdaptiveTree` alone would have made it inconsistent with its own
sibling, for a boundary that was already explicit via timers. I dropped that
idea.

So the actual output of this task is almost entirely documentation:
`adaptive_tree.hpp`'s misleading `@warning`, and every place in
`docs/architecture.md`, `src/tree/AGENTS.md`, and `agent_docs/project_structure.md`
that called this a tree-to-plan seam or a dependency-audit exception. None of
it was. Zero production code changed. The lesson worth keeping: a
"transitional" label in a comment is a claim about the past, not a live
fact about the code, and it should have been re-verified the first time
someone actually read what `StaticFmmTopology` contained rather than
re-quoted at each subsequent Phase-1 status update.

The one substantive code change in this session was the small carry-over
correction from the previous task: `assign_static_p2p_compact_row` had
landed in the installed `include/cdfmm/plan/p2p/compact.hpp` when it was
really only needed to let two `.cpp` files share packing mechanics. Moved to
`src/plan/p2p/compact_row.hpp`, no behaviour change, committed separately
before the audit began.

## 2026-09-15 — internal-duplication cleanup: unify without unfusing

Phase 1 had left one item explicitly deferred rather than closed: the
canonical-to-compact P2P row mapping was stated twice, once in the ordinary
plan builder and once inside the cache's fused warm-decode loop, and both
`docs/architecture.md` and `src/cache/AGENTS.md` warned against the obvious
"fix" of unfusing that loop just to remove the duplication. The interesting
part of this task was resisting that obvious fix. The two restatements were
genuinely the same rule — same fields, same order, same semantics — so the
real question was where a shared primitive could live without forcing the
fused loop apart. The answer was a small inline free function taking a
plan-and-slot and a canonical block, callable from inside an indexed
loop body either way. `compact.cpp`'s builder switched from `reserve`+`push_back`
to `resize`+indexed-write so it could call the same function; the cache decode
kept its single `#pragma omp parallel for` untouched and just replaced eleven
lines of hand-copied fields with one call. Neither side changed what it
computes.

The other item, a duplicated `n!` loop between `rectangular_prism.cpp` and
`MultiIndexSet::factorial`, was unambiguous once an audit worker confirmed the
two were byte-for-byte identical in algorithm, domain, and (lack of) overflow
handling — a straightforward delete-and-call-through.

The most convincing evidence, again, was the cheapest to gather: a standalone
probe compiled unchanged against `v0.1.0` and against this HEAD, sharing one
`CDFMM_CACHE_DIR`, showing identical cache keys, hits, zero bytes written, and
bit-identical fields — and a stash-based before/after timing comparison on the
exact touched loop, showing no measurable difference. Both are cheaper and
more convincing than reasoning about the change from first principles.

From `66b9436`, the job was to decide whether Phase 1 was actually finished,
using evidence rather than the repository's own say-so, and then to close it.

The useful discipline this session was refusing to take audit reports at face
value. Three read-only workers produced a lot of correct detail, but two
confident findings were wrong, and both would have caused damage. One claimed
the `cdfmm/uniform_fmm.hpp` include in `src/backend/cuda/fmm/internal.hpp` was
dead, having grepped for class names only; removing it broke the build, because
`plan.cu` *defines* `cuda_m2l_p2p_available()`, which that header declares. A
backend implementing a public availability query is a supported API
relationship, not a layering inversion — and the fix was to revert, not to
invent an architectural justification. The lesson is cheap to state and easy to
forget: verify a claimed dead dependency by deleting it and rebuilding.

The most satisfying evidence was the cheapest. Rather than reasoning about
whether the cache refactor preserved its format, a `v0.1.0` build wrote a
ten-file corpus and HEAD read it back: same keys, every lookup a hit, zero bytes
written, corpus byte-identical afterwards. The same probe, compiled unchanged
against both versions' public headers, returned bit-identical fields across five
configurations. That single artefact answers "is this behaviour-preserving?"
better than any amount of diff reading.

The Fortran result was the genuine surprise. Every previous session recorded
"no Fortran compiler available" — true of `PATH`, but `ifx` sits in the
`magtense-env` Conda env. With it, the smoke test failed, and for a moment that
looked like a refactor regression. It was not: the test asks for a cubic
periodic cell of side 2 centred at the origin while its prisms sit at `z = 0`
and `z = 1`, so the second one pokes outside the periodic root. Building the
same test at `v0.1.0` reproduced the failure exactly. A pre-existing bug, hidden
for as long as the toolchain was missing. Classifying it correctly mattered more
than fixing it: the refactor preserved the behaviour faithfully, including the
broken part.

The documentation warnings told a similar story. Two Sphinx warnings had been
carried forward as "pre-existing" for several sessions. They were neither
mysterious nor pre-existing in the sense implied: `docs/Doxyfile` predefined
`CDFMM_HOST_DEVICE`, a macro used nowhere, while the canonical header the
refactor introduced guards two declarations with `CDFMM_PLAN_HOST_DEVICE`. A
one-word fix cleared both. It is worth noticing how long a warning can survive
once it has been labelled expected.

The last thing worth recording is what was left alone. The plan/backend dispatch
in `src/plan/direct/dense.cpp` is a real reverse-direction edge, and it is there
because `DenseDirectPlan::evaluate()` is public API from `v0.1.0`. Removing it
would be an API change, not a cleanup. It is now written down as a deliberate
exception rather than left to be rediscovered and "fixed" by someone later. The
same goes for the flat `timings.hpp`/`periodic.hpp` includes and the
`StaticFmmTopology` seam. Phase 1 closes with its compromises enumerated, which
is the honest form of done.

## 2026-09-15 — compatibility and transitional-source review

From starting HEAD `5cb5576`, audited every remaining legacy-looking or
transitional file and made the surviving compatibility surface intentional.

The decisive fact came from `git ls-tree v0.1.0 include/cdfmm/`: the flat
header set is unchanged since the preserved tag, and CMake installs the
directory by glob. Every flat public header is therefore a shipped public
include path. That turned the question from "which of these look old?" into
"which carry a real obligation?", and the answer was all of them, so none were
removed. Each now says so in a comment.

The interesting case was `src/cuboid.cpp`, and it was not the shim its
neighbours suggested. It owned the prism-averaged monomial used by
`src/operators/{p2m,l2p}.cpp` and the pair-tensor dispatch used by
`src/operators/p2p.cpp` and `src/plan/direct/dense.cpp`, while the canonical
`operators::p2p::build_pair` merely forwarded to it. The dependency ran
canonical -> legacy. The monomial moved to
`src/geometry/primitives/rectangular_prism.cpp`, next to the existing
`tetrahedron_averaged_monomial` it had been separated from; the dispatch became
the body of `operators::p2p::build_pair`. Both flat names survive as thin
delegations, so nothing downstream moved.

Two behaviour details were preserved deliberately rather than tidied. The
monomial path keeps its own side-length check instead of adopting the stricter
volume validation the exact pair tensors use, and the tetrahedral rejection
still names `build_pair_tensor` in its message even though it is now thrown
from the canonical function. Both are observable, so both stayed.

The internal `src/` shims were the opposite case. `cuda_fmm_plan.hpp` and
`cuda_p2p_plan.hpp` had no includers at all, `cuda_m2l_plan.hpp` had two that
could name the canonical header directly, and `static_operators.cpp` had been
absent from every source list since `14a94d6` while still claiming in its own
comment to exist "for downstream source lists". Internal headers carry no
downstream obligation, so all four were removed.

Narrowing the canonical headers surfaced the one genuine compatibility trap.
Removing `cdfmm/cuboid.hpp` from the canonical operator headers also removed
`DenseDirectBackend` from everything reaching it through
`cdfmm/static_operators.hpp`, which a test caught immediately, and dropped the
flat operator names from `python/operators.cpp`. The fix was to restore the
pre-v0.2 transitive surface at the façades themselves rather than to re-widen
the canonical headers, which is the rule now written into
`include/cdfmm/AGENTS.md`: compatibility flows one way.

Both audit workers recommended relocating `periodic.cpp`,
`parameter_selection.cpp`, and `validation.cpp` as well. That was declined.
Each owns one coherent responsibility, a flat source file is not wrong merely
for being flat, and moving them belongs to Phase 2.

Validation: portable, CUDA, and oneMKL+CUDA configurations all built fresh and
passed CTest 198/198 (four, one, and zero skips respectively); Python 137
passed and 7 skipped, matching the previous task; the C ABI kept exactly its 14
symbols; and tiny consumers compiled against an installed prefix through both
legacy-only and canonical-only include paths, with `nm` confirming the two
spellings resolve to identically typed symbols. Phase 1 is not closed; the
remaining task is whole-Phase-1 validation and closure.

## 2026-09-15 — bindings boundary and validation closure

At implementation base/newest starting HEAD `9003b666`, the bindings boundary
was structured without changing the public contract. The C ABI implementation
moved byte-identically to `src/bindings/c_api.cpp`; `include/cdfmm/c_api.h`
remains unchanged and ABI version 1 remains. Fortran continues through
`ISO_C_BINDING -> C ABI -> supported C++`. The former `python/bindings.cpp` was
replaced by `python/internal.hpp`, `module.cpp` as the sole entry point, and
the subsystem units `core.cpp`, `geometry.cpp`, `tree.cpp`, `operators.cpp`,
`direct.cpp`, and `fmm.cpp`. CMake registers these units. Binding code uses
supported canonical structured headers where available, retains only justified
compatibility dependencies, and contains no solver logic or internal backend
headers.

The isolated `9e111ee8` manifest comparison found 63 top-level Python names,
with object kinds, docs/defaults, and member surface identical after
normalising object addresses. The C ABI dynamic symbol set remained exactly
14 `cdfmm_*` symbols, and the C source move was byte-identical.

Trusted final evidence: portable affected/full build 73/73, CTest 197/197
with four expected optional skips, and Python 137 passed/7 skipped; oneMKL
notebooks core/C/Python/tests build, CTest 197/197 with three CUDA skips, and
Python 139 passed/5 skipped; RTX 5090 visible with driver 595.84 under full
access; fresh CUDA 13.3.73 build with explicit SM120, affected targets
107/107, CTest 197/197 with only the oneMKL-only skip, and Python 141 passed/
3 skipped. The initial stale SM75 artifacts failed PTX on the 5090 and are not
a final result. No gfortran, ifx, or ifort was available, so Fortran smoke and
example checks were not run. An independent temporary-prefix install passed,
and diff/ownership audits passed. `sphinx-build -W --keep-going -b html docs
docs/_build/html` rendered the documentation and exited nonzero only for the
same two pre-existing `docs/api.rst` C++ declaration warnings.

The cache change is already committed as `9e111ee`; it is not pending. The
remaining Phase 1 sequence is compatibility/transitional-source review,
followed by whole-Phase-1 validation. Neither is claimed complete here.

## 2026-09-15 — cache subsystem ownership decision

Accepted the cache decomposition by responsibility rather than by the smallest
possible file count. Two separate "format" concerns were deliberately kept
apart: the validated file container (magic, header fields, checksum, atomic
temporary-file write) stays with the persistence mechanics in `cache/io.cpp`,
because it cannot be separated from the read/write path, while `cache/format.cpp`
owns only the field-wise records for the solver types inside a payload. The
stream primitives (`Writer`, `Reader`, `CachePayload`, the unaligned load/store
templates) live in `cache/internal.hpp` because they are templates crossing
every cache translation unit, not for convenience.

The anonymous namespace that previously covered the whole file could not be
carried into a shared header: a type defined in an anonymous namespace in a
header is a distinct type per translation unit. The shared entities therefore
moved into the named internal namespace `cdfmm::detail::cache`, matching the
existing `detail::cpu`/`detail::mkl` convention, while entities with a single
consumer stayed in per-unit anonymous namespaces.

Three known layering warts were deliberately left in place because this step
was behaviour-preserving: `read_p2p_blocks` still populates the derived compact
plan in the same pass as decoding, the universal payload still encodes the M2L
bank layout arithmetic, and `load_geometry_cache` still validates plan
invariants and resets only FP32 state on failure. Each is recorded rather than
repaired. `UniformFmm` keeps its existing private cache members and object
layout; deeper encapsulation belongs to a later API/ABI step.

The decisive evidence was a before/after comparison against real cache files,
not the unit tests: the existing suite only compares keys between two live
instances, so a uniform key change would have passed it unnoticed. Final
validation recorded a portable fresh configure requiring an explicit `cdfmm`
environment PATH because literal `cmake` was unavailable, a 67/67 build, full
CTest 193/193 with skips #16/#51/#59/#63, and focused cache coverage 9/11 with
expected skips #51/#59. The oneMKL notebooks configuration used CUDA 13.2 and
oneMKL 2026.1.0 and passed its core/tests/Python build, full CTest 190 plus
three CUDA skips, focused 31 plus one CUDA skip, and Python 139 plus five
skips. The CUDA 13.2 SM75 core/tests/Python build passed 93/93; full CTest
passed 189 plus four skips, direct CUDA passed 16 plus three skips, and Python
import/smoke checks passed. Pytest was unavailable for matching Python 3.13,
and `nvidia-smi` could not reach the driver, so no GPU runtime is claimed.

Prior old-cache compatibility evidence is preserved and now backed by the
recovered 11-file corpus: both directions produced hits with zero writes and
identical manifests, the repository's 522-file corpus remained unchanged, and
an independent current probe hit old universal/periodic files without
modifying their hashes. No implementation defect was found. The cache task is
already committed as `9e111ee` with subject `refactor(cache): structure
persistence subsystem`; at that checkpoint, the bindings boundary,
compatibility/transitional review, and whole-Phase-1 validation remained.

## 2026-09-15 — high-level ownership audit

The independent final audit identified and verified the L2P ownership repair:
far-field sequencing now owns L2P evaluation, while evaluation retains the
whole near/far lifecycle, timing, and hybrid CUDA begin/CPU-far/finish
coordination. The re-review passed; no public API, cache/backend, binding, or
test changes were introduced.

## 2026-09-15 — CPU execution boundary decision

Accepted the portable CPU split by responsibility rather than by mirroring
CUDA file counts: P2P owns every prepared list-1 representation and near-field
dispatch, M2L owns prepared-plan application, and far-field groups only the
shared P2M/L2P entry and M2M/L2L translation mechanics. The legacy
`static_plan_apply.hpp` is intentionally retained as a compatibility umbrella;
reference mathematics stays in `src/fmm/far_field.cpp`, and no generic helper
bucket or public API redesign was introduced. The next architecture step is
the final higher-level `UniformFmm` cleanup.

## 2026-09-14 — CUDA far-field execution extraction closure

Accepted the CUDA far-field ownership split. The internal
`src/backend/cuda/far_field/{internal.hpp,executor.cu}` package owns immutable
FP32/FP64 P2M and L2P entries, coefficient degrees, M2M/L2L matrices,
interactions and metadata, uploads/lifecycle/statistics, and kernels. It adds
no public API, stub, or new library. `CudaFullPlan` retains changing
moments/coefficient/field buffers, permutations, P2P and separate M2L executor
wiring, streams/events/timing, near/far overlap, combination/reordering, and
D2H transfer. Direct, P2P, and M2L remain authoritative in their existing
backends. The next scoped task is orchestration/resource simplification; this
does not complete the `CudaFullPlan` decomposition.

Trusted evidence: the initial M2L sanity ownership audit was clean and its
focused 14-case probe passed; a fresh CPU development configure/build completed
55 steps and full CTest completed 193/193 with expected skips #16/#51/#59/#63;
fresh CUDA configuration used `module cuda/13.2` (`nvcc 13.2.78`) for SM75; the
preset presented a 206-step graph and compiled the requested `cdfmm_core`,
tests, Python extension, `cdfmm-precompute`, and three benchmark targets, but
failed at its final install step on sandbox read-only Conda site-packages;
explicitly requested target builds then succeeded; focused CUDA CTest completed
30/30 with
expected unavailable-device skips #16/#59/#63; full CUDA-configured CTest
completed 193/193 with expected skips #16/#51/#59/#63; and `git diff --check`
plus independent code/ownership review passed. `nvidia-smi` could not
communicate with a driver/device, so no numerical GPU runtime execution is
claimed. Unrelated `Article1/` and notebook changes were preserved.

## 2026-09-14 — CUDA M2L backend extraction closure

Accepted the reusable CUDA M2L ownership split. The canonical public
`CudaM2LPlan` declaration is now `include/cdfmm/backend/cuda/m2l.hpp`, with
`src/cuda_m2l_plan.hpp` retained as a forwarding compatibility shim.
`src/backend/cuda/m2l/{internal.hpp,plan.cu}` owns the shared FP64/FP32 device
representation, kernels, bounded scaled-multipole scratch policy, persistent
resources, lifecycle, and statistics used by standalone `CudaM2LPlan` and
`CudaFullPlan`; `src/backend/cuda/stub/m2l.cpp` owns non-CUDA stubs.
`src/cuda_fmm.cu` retains P2M/M2M/L2L/L2P and full-FMM/far-field orchestration
plus internal backend wiring.

Trusted validation: fresh CPU configure/build 194/194; fresh CPU full CTest
193/193 with expected skips #16/#51/#59/#63; fresh CUDA 13.2 SM75 build
204/204 covering core, tests, Python, tools, and benchmarks; focused CUDA
M2L/header/stub coverage 23/23 with unavailable runtime-device cases skipped
as expected; full CUDA CTest 193/193 with the same four skips; and install,
ownership, complete-diff, and `git diff --check` audits passed. `nvidia-smi`
could not reach a driver/device, so no numerical GPU runtime result is
claimed. The untracked `Article1/` directory and unrelated
`examples/simple_notebooks/tetrahedron_target_average_comsol_compare_sampling_and_gauss.ipynb`
were preserved untouched.

## 2026-09-14 — complete CUDA P2P backend extraction closure

Accepted the complete CUDA P2P ownership split as the preceding step. The canonical public
`CudaP2PPlan` declaration is now in
`include/cdfmm/backend/cuda/p2p.hpp`; the flat `include/cdfmm/cuda_p2p.hpp`
path remains a forwarding compatibility façade. `src/backend/cuda/p2p/`
owns canonical AoS, compact/source-only SoA, leaf-block, signed tensor
dictionary source-warp/target-owned/power-of-two microtile, and cuSPARSE
BSR(3) execution in FP64 and FP32, including lifecycle, asynchronous state,
persistent resources, and shared full-plan primitives. The non-CUDA stub is
`src/backend/cuda/stub/p2p.cpp`; the current `src/cuda_fmm.cu` retains
far-field/full-FMM orchestration and consumes the internal P2P and M2L
primitives.

Trusted validation: fresh CPU full CTest completed 192/192 with four expected
skips; fresh CUDA 13.2 SM75 completed a 206-step build covering core, tests,
Python, examples, tools, and benchmarks; CUDA-configured full CTest completed
192/192; runtime numerical GPU validation was unavailable because `nvidia-smi`
could not communicate with the driver; and ownership, `nm`, and
`git diff --check` audits were clean. The untracked `Article1/` directory and
unrelated `examples/simple_notebooks/tetrahedron_target_average_comsol_compare.ipynb`
were preserved untouched. Its focused validation was recorded before the
combined CUDA M2L closure.

## 2026-09-14 — CUDA direct backend extraction closure

Accepted the CUDA direct/common ownership split. Canonical public interfaces
are `include/cdfmm/backend/cuda/direct.hpp` and
`include/cdfmm/backend/cuda/dense_direct.hpp`; flat
`cuda_direct.hpp`/`cuda_cuboid.hpp` remain compatibility façades. Shared CUDA
error/runtime facilities live in `src/backend/cuda/common/`, point O(N^2) and
dense cuBLAS execution in `src/backend/cuda/direct/`, and non-CUDA direct
stubs in `src/backend/cuda/stub/direct.cpp`. At that earlier checkpoint,
`src/cuda_fmm.cu` retained P2P, M2L, and full-FMM execution. No behaviour or
performance change was intended.

Trusted validation: fresh dev configure/build completed 59 targets; focused
direct/header/stub coverage passed 8/8, including the exact disabled-stub
behaviour case; full CPU CTest passed 189/189 with expected skips
#16/#51/#59/#63. A fresh CUDA 13.2 configure with explicit architecture 75 and
cached Catch2 source succeeded, followed by a 198-target CUDA build including
tests, Python, and benchmarks. Independent full CUDA CTest reported 189/189,
but numerical/runtime
cases only reached unavailable-device guards because `/dev/nvidia` was absent
and `nvidia-smi` could not reach the driver. Ownership/`nm` audits and
`git diff --check` were clean. `Article1/` was untouched.

## 2026-09-11 — FMM, CPU, and oneMKL execution boundaries

Established explicit execution boundaries without changing the public FMM
API or pass sequencing. `src/fmm` owns lifecycle and the
P2M/M2M/M2L-dispatch/L2L/L2P chain, `src/backend/cpu` owns list-1 and portable
static-plan execution, and `src/backend/mkl` owns grouped oneMKL M2L execution.
An opaque internal owner now retains stable transfer grouping and reusable
FP32/FP64 gather/translation buffers; the canonical plan continues to own the
matrices, scaling, and mathematical interaction schedule.

Fresh portable and reconfigured oneMKL builds passed. Full portable CTest
completed 184 cases with 180 passes and four expected optional skips; full
oneMKL CTest completed 184 cases with 181 passes and three expected CUDA skips.
Focused tests cover cache sharing, spherical and Cartesian M2L, both
precisions, source-level scaling, sorting, buffer reuse statistics, timings,
and move behaviour. Ownership and diff audits passed. CUDA runtime and other
optional interface/documentation paths were not exercised, and no CUDA source
was changed. `Article1/` and parent-repository work remain untouched.

## 2026-09-11 — dense-direct plan/backend separation

Completed dense-direct execution ownership separation. The plan layer retains
geometry validation, deterministic pair-tensor construction, six target-major
matrices, precision storage, dispatch, and memory accounting. Portable row-major
nine-GEMV execution now lives in `src/backend/cpu/direct/dense.cpp`; guarded
oneMKL SGEMV/DGEMV execution and availability live in
`src/backend/mkl/direct/dense.cpp`. A private pimpl workspace retains reusable
FP32/FP64 staging arrays while explicit special members preserve deep-copy and
move value semantics. CUDA/FMM backend decomposition remains deferred.

Fresh portable and oneMKL builds passed after the move. Focused direct tests,
full portable CTest, and full oneMKL CTest passed with only the expected
CUDA/optional skips; repeated evaluation and copy/move regression coverage is
included. `Article1/` and unrelated parent-worktree changes remain untouched.

## 2026-09-11 — dense-direct plan ownership

Moved the authoritative `DenseDirectPlan` interface to
`include/cdfmm/plan/direct/dense.hpp` and its implementation to
`src/plan/direct/dense.cpp`. The legacy `cuboid.hpp` include remains supported
as a compatibility umbrella, while `src/cuboid.cpp` now retains only cuboid
monomial and pair-tensor mathematics. Portable and oneMKL GEMV execution stays
with the plan by design; dedicated backend extraction is the next dense-direct
step.

Fresh dev configure/build passed, focused geometry/direct/precision tests passed
41/41, and full portable CTest passed 182/182 with four expected optional
skips. A separate CPU+oneMKL build passed and its dense/cuboid focused tests
passed 25/25. Canonical and legacy header probes, installed-header presence,
diff checks, and symbol/source ownership searches passed. CUDA runtime was not
available and was not claimed; no oneMKL dependency was installed.

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
