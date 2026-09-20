# Architecture and performance history moved out of the public documentation

Phase 4 (repository pruning, 2026-09-20) reduced `docs/` to a description of
the current library. The material below was public documentation up to that
point and is preserved here verbatim as provenance: the Phase-1 and Phase-2
closure narratives of `docs/architecture.md`, the per-cleanup audit records,
the CUDA M2L performance report of 2026-08-17, the pre-refactor `v0.1.0`
baseline record, the sweep history of the static P2P execution study, and the
historical measurements section of `docs/benchmarks.md`. The current
statements that these records supported live in `docs/architecture.md`,
`docs/backends.md` and `docs/benchmarks.md`; the Phase-3 performance record is
`agent_docs/performance_optimization.md`.

---

# From `docs/architecture.md`: deferred inventory, cleanup records, Phase 1 closure, Phase 2 handoff

## Deferred refactor inventory

This inventory records remaining pressure points; it does not authorise a
future repair by itself.

The initial audit found these concrete boundary violations in the remaining
transitional layout:

- `tree/static_topology.hpp` is the canonical topology-only interface;
  `tree/uniform_topology.hpp` declares its UniformTree producer
  (`build_uniform_fmm_topology`) and `adaptive_tree.hpp` returns the same
  `StaticFmmTopology` from `AdaptiveTree`'s own construction. A later
  tree/topology-boundary audit (see "Tree/topology boundary cleanup" below)
  found this is ordinary intra-`tree`-layer usage, not a tree-to-plan seam:
  `StaticFmmTopology` holds only spatial-tree and interaction-topology facts,
  no operator coefficients, execution packing, or backend state. The flat
  `static_topology.hpp` remains a compatibility forwarding header.
- `cuboid.hpp` is a compatibility façade that declares nothing of its own.
  `DenseDirectPlan` is declared by `plan/direct/dense.hpp`, the prism record,
  `CuboidSize`, and the averaged monomial by
  `geometry/primitives/rectangular_prism.hpp`, and the pair tensor by
  `operators/p2p.hpp`. Portable, oneMKL, and CUDA direct execution have
  dedicated internal backend homes; the façades retain only their supported
  API surfaces.
- compatibility `static_operators.hpp/.cpp` remain as forwarding umbrellas;
  their former mixed implementation has been separated into operators, plans,
  and the portable CPU apply boundary described below.
- `fmm/far_field.cpp` contains CPU expansion sequencing and no longer contains
  oneMKL vendor mechanics; complete CUDA FMM orchestration is in
  `backend/cuda/fmm/plan.cu`.
  `periodic.cpp` now includes the narrow `cdfmm/tree/indexing.hpp`, and the
  public-header/packaging cleanup (see below) narrowed
  `parameter_selection.hpp` the same way: it now includes only
  `cdfmm/backend/execution.hpp` and `cdfmm/math/vec3.hpp` instead of the
  complete `uniform_fmm.hpp`.
- the cache subsystem under `cache/` separates the file container and root
  policy, payload records, identity, and the universal/periodic and
  geometry-plan payloads. A later cache/plan-preparation boundary cleanup (see
  "Cache/plan-preparation boundary cleanup" below) removed the remaining
  `UniformFmm` entry points: every cache function is now a free function
  taking an explicit identity/payload record built from specific solver/plan
  types, and `src/fmm/execution_setup.cpp`/`plan_preparation.cpp` are the sole
  callers. `cache/format.cpp` still builds the derived compact P2P representation
  while decoding the canonical blocks in the same pass, but it now does so by
  calling `assign_static_p2p_compact_row` (`src/plan/p2p/compact_row.hpp`, an
  internal implementation header, not an installed public path), the same
  row-packing primitive `plan/p2p/compact.cpp` uses; the canonical-to-compact
  rule itself is stated once. Only the fused-pass *choice* remains cache-local,
  as a performance decision, not a restatement of the mapping.
- the remaining public CUDA/FMM headers expose transitional plan concepts and
  depend on broad operator or geometry headers; direct, P2P, and M2L CUDA APIs
  now have canonical `backend/cuda/` headers with legacy forwarding façades.
- the root CMake target registers the CPU subsystems together with explicit
  CUDA common/direct/P2P/M2L/far-field/FMM units. CUDA/oneMKL/OpenMP link
  requirements are marked `PUBLIC` on the internal static `cdfmm_core` (so its
  own in-tree consumers — tests, benchmarks, the Python module, the Fortran
  example, `cdfmm-precompute` — build correctly), but this does not leak
  downstream: `cdfmm_c`, the only installed target, links `cdfmm_core`
  `PRIVATE` and adds no `PUBLIC`/`INTERFACE` link libraries of its own, so
  none of `cdfmm_core`'s dependencies appear in `cdfmm_c`'s exported usage
  requirements. See "Public/internal API, header ownership, and packaging
  cleanup" below for the added package config and its validation.

The largest remaining responsibility-review candidates are
`backend/cuda/p2p/plan.cu`, `geometry/primitives/tetrahedron.cpp`, and
`backend/cuda/fmm/plan.cu`. These measurements locate remaining audit work;
they do not require mechanical splitting.

### Foundational layout: core, math, geometry, and tree

- Foundational precision, output, vector, index, derivative, Cartesian, and
  spherical interfaces have canonical subsystem headers with flat shims.
- Geometry models and exact rectangular-prism/tetrahedron primitives have
  canonical homes; point geometry remains its representative `Vec3` plus model
  selection, without an artificial marker type.
- Common node, Morton, indexing, and tree-statistics concepts are separated
  from uniform and adaptive implementations.
- Internal cubic root-box resolution and validation is shared by UniformTree
  and AdaptiveTree in `src/tree/common/root_box.{hpp,cpp}`. Shared resolution
  retains zero for coincident roots; AdaptiveTree applies its existing fallback
  half-width `1` at its call site. The helper is not part of the public API.
- Dense-direct pair dispatch remains a transitional seam for the
  operator/plan/backend phases (the isolated `plan/direct/dense.cpp`
  compatibility edge recorded under "Ownership and dependency audit" below).
  `StaticFmmTopology` and adaptive tree construction were audited in the
  tree/topology boundary cleanup and found to be legitimate tree-owned
  topology, not a plan-layer seam; see "Tree/topology boundary cleanup".

### Implemented: operators, static plans, and CPU/oneMKL FMM boundaries

- Mathematical P2M, M2M, M2L, L2L, L2P, and P2P construction now lives under
  `include/cdfmm/operators/` and `src/operators/`. Geometry-specific P2M/L2P
  and P2P dispatch remains below the operator layer and uses the established
  geometry primitives.
- Canonical static data is represented under `include/cdfmm/plan/`, with
  FP64-to-FP32 conversion in `src/plan/precision.cpp`. Static M2L data owns
  immutable matrices, scaling, and interaction schedules without backend
  resources.
- `plan/p2p/canonical.hpp` is authoritative for target-row P2P tensors and
  identity markers. Compact/SoA, leaf, tensor dictionary, signed/reduced
  dictionary, and BSR forms are deterministic derived packings under
  `plan/p2p/`; BSR construction is backend-neutral and does not depend on
  CUDA or cuSPARSE.
- Portable CPU application is isolated under
  `backend/cpu/{p2p,m2l,far_field}`. P2P consumes canonical or derived plans,
  M2L consumes prepared schedules, and far-field entry/translation helpers
  consume prepared static operators; none redefine pair or translation
  mathematics.
- Dense-direct plan preparation remains under `plan/direct`; portable execution
  is in `backend/cpu/direct`, oneMKL execution is in `backend/mkl/direct`, and
  reusable staging is private backend execution state.
- High-level FMM lifecycle and far-field pass orchestration live under
  `src/fmm`; portable list-1 and static-plan execution live under
  `src/backend/cpu`; and grouped oneMKL M2L packing, reusable scratch, vendor
  calls, thread control, and availability live under `src/backend/mkl`.
- CUDA direct/common execution now lives under `src/backend/cuda/{common,direct}`
  and the complete P2P, M2L, far-field, and full-FMM backends under
  `src/backend/cuda/{p2p,m2l,far_field,fmm}/`,
  with canonical public headers under `include/cdfmm/backend/cuda/` and
  legacy forwarding façades retained. The reusable M2L executor is consumed
  by both standalone `CudaM2LPlan` and `CudaFullPlan`; `backend/cuda/fmm/plan.cu`
  coordinates and enqueues the P2M/M2M/L2L/L2P stages and full-FMM orchestration
  plus internal backend wiring, while static execution lives in
  `backend/cuda/far_field`.

`StaticFmmTopology` is tree-owned interaction topology, not a plan artefact:
the tree/topology boundary cleanup (see "Tree/topology boundary cleanup")
confirmed every field is a spatial-tree fact, an interaction-topology fact,
or schedule indexing over those facts, with no operator coefficients,
execution packing, or backend state. Both `build_uniform_fmm_topology` (from
a built `UniformTree`) and `AdaptiveTree`'s own construction populate it
directly; `fmm` and `plan` consume the result through
`shared_ptr<const StaticFmmTopology>` and never rebuild it. Tree ownership
remains spatial hierarchy/topology, while derived schedule assembly stays
behind the plan boundary without making the tree depend on a particular P2P
packing.

### Stable backend boundaries

- Keep the accepted direct/common/P2P/M2L/far-field/full-FMM extraction stable.
- Preserve the separation of immutable far-field execution state from mutable
  moments, coefficient/field buffers, streams/events, timing, overlap,
  combination/reordering, and D2H resources.
- Preserve and benchmark all transfer, launch, overlap, and reuse semantics;
  source boundaries must not add runtime work.

### Phase 1 work: complete

The compatibility/transitional-source review is complete, and whole-refactor
validation across the supported CPU, oneMKL, CUDA, bindings, cache, install,
and documentation configurations has been performed. See "Phase 1 closure".

One item raised here during the refactor was deferred rather than closed: the
duplicated canonical-to-compact P2P packing rule, independently stated by the
cache decode in `cache/format.cpp` and the plan builder in
`plan/p2p/compact.cpp`. A follow-up internal-duplication cleanup, after Phase
1 closure, resolved the duplicated *rule* without touching the plan/cache
boundary: both call sites now share one row-packing primitive,
`assign_static_p2p_compact_row`. It was first added to the installed
`include/cdfmm/plan/p2p/compact.hpp`, then relocated to the internal
`src/plan/p2p/compact_row.hpp` once it became clear the helper had no
downstream purpose beyond letting `plan/p2p/compact.cpp` and
`cache/format.cpp` share packing mechanics; `compact.hpp` keeps the public
`StaticP2PCompactPlan`/`FloatStaticP2PCompactPlan` types and builders. The
cache decode still fuses canonical decode and compact construction into one
pass deliberately for performance; that fused-pass structure, and the
broader question of whether cache entry points should stop being `UniformFmm`
members that know the topology/operator representations they persist, remain
open and are recorded as a separate later cache/plan-boundary item, not
mechanical de-duplication.

That same cleanup also removed a small duplicated `n!` primitive: a
private `monomial_factorial` in `src/geometry/primitives/rectangular_prism.cpp`
restated the loop already implemented by `MultiIndexSet::factorial`
(`include/cdfmm/math/multi_index.hpp`). The prism code now calls the `math`
implementation directly, matching the `geometry -> math` dependency direction;
no new file or shared `utils`/`helpers` header was introduced.

Before that, the P2P row-packing helper itself moved once more: a narrow
follow-up correction relocated `assign_static_p2p_compact_row` from the
installed `include/cdfmm/plan/p2p/compact.hpp` to the internal
`src/plan/p2p/compact_row.hpp`, because the helper existed solely so
`plan/p2p/compact.cpp` and `cache/format.cpp` could share packing mechanics,
not as a supported downstream API. `compact.hpp` keeps the public
`StaticP2PCompactPlan`/`FloatStaticP2PCompactPlan` types and builders; the
canonical-to-compact mapping and the fused cache-decode pass are unchanged.

### Tree/topology boundary cleanup

A separate follow-up task, after the internal-duplication cleanup, audited
the `StaticFmmTopology`/tree-to-plan boundary that Phase 1 left recorded as a
transitional seam (see the former wording in "Deferred refactor inventory"
and "Phase 2 handoff"). The audit read `StaticFmmTopology` field by field —
`Node`, permutations, sorted positions, occupied leaves, `m2m_edges`/
`l2l_edges`, `m2l_interactions`, `p2p_leaf_records`, row/level offsets, and
the coordinate-normalisation fields — against every producer
(`build_uniform_fmm_topology`, `AdaptiveTree`'s own construction) and every
consumer (`fmm/construction.cpp`, `plan_preparation.cpp`,
`execution_setup.cpp`, `far_field.cpp`, `cache/geometry.cpp`,
`backend/cpu/p2p/near_field.cpp`, `python/tree.cpp`, and the C++/Python test
suites). The finding: every field is a spatial-tree fact, an
interaction-topology fact (near/far admissibility, M2M/L2L translation
structure, P2P leaf pairing), or CSR-style schedule indexing over those
facts. None is an operator coefficient, a derived P2P execution packing, or
backend state — `plan/p2p/*` and `src/fmm/plan_preparation.cpp` build the
canonical operator data and derived packings strictly downstream, from
`StaticP2PLeafRecord`/`StaticM2LInteraction` fields, never the other way
round. `fmm` consumes the result through
`shared_ptr<const StaticFmmTopology>` by reference and never rebuilds it;
the cache subsystem never persists `StaticFmmTopology` directly and disables
caching entirely whenever a topology is supplied externally
(`cache/keys.cpp`'s `supplied_topology_` check), so `AdaptiveTree`'s topology
never touches the persistent cache format at all.

The `AdaptiveTree` constructor was also checked for whether it interleaves
spatial-octree construction with far/near interaction-topology assembly in a
way that would justify splitting them into two internal representations.
It does not: the constructor already runs the octant partition and M2M/L2L
edge derivation as one pass, then the dual-tree near/far `visit` as a
strictly later pass over the completed tree, with `tree_seconds_` and
`interaction_seconds_` already timing exactly that boundary. Introducing a
separate adaptive-only spatial-tree type to mirror `UniformTree` would
duplicate nearly every field `StaticFmmTopology` already has (adaptive
nodes carry no Morton indices or other data that would make a distinct
representation non-redundant, unlike `UniformTree`'s own `TreeNode`, which
has genuinely different content: level-wise dense Morton arrays and
`list1`/`list2` candidate lists that are never stored in `StaticFmmTopology`
at all). Manufacturing that duplicate type purely to make the layering
diagram prettier was rejected; `UniformTree::build` itself remains one
monolithic, internally phase-timed function for the same reason, so a forced
split in `AdaptiveTree` alone would also be inconsistent with its own
sibling's established construction style.

The conclusion is that `StaticFmmTopology` was already correctly tree-owned
topology, and the previously recorded "transitional tree-to-plan coupling"
in `adaptive_tree.hpp`'s `@warning` and in `docs/architecture.md` was a
documentation defect, not a live architectural problem: the seam appears to
have been named while `StaticFmmTopology`'s ownership was still unsettled,
and the language was never revisited once the type's contents were fully
audited. No production code changed; the misleading comment and the doc
passages that repeated it were corrected, and the two resolved items were
removed from the "Deferred refactor inventory" and "Phase 2 handoff" lists
below. Everything else about this boundary — field layout, ordering,
`validate()`'s invariants, the `supplied_topology_` M2L transfer-class branch
in `plan_preparation.cpp` that treats uniform- and adaptive-built topologies
differently, and all public `AdaptiveTree`/`StaticFmmTopology`/
`build_uniform_fmm_topology` signatures — is unchanged.

### Cache/plan-preparation boundary cleanup

A later task audited the `UniformFmm`/cache-persistence boundary that Phase 1
and the internal-duplication cleanup had both left recorded as open: cache
entry points (`initialise_cache_keys`, `load_universal_cache`,
`write_universal_cache`, `load_geometry_cache`, `write_geometry_cache`) were
private `UniformFmm` member functions defined in `src/cache/*.cpp`, so
`src/cache/internal.hpp` included the complete `cdfmm/uniform_fmm.hpp` and
cache translation units had direct access to essentially all private solver
state. A dependency map (two read-only workers, one over
`src/cache/*`/`uniform_fmm.hpp`/`src/fmm/*`, one over the public/ABI/Python/test
compatibility surface) classified every field each of the five functions read
or wrote as cache identity, persisted payload, statistics, or unrelated
plan/lifecycle state, and confirmed none of the five were referenced outside
`src/cache/*.cpp` and `src/fmm/*.cpp`, so no public/ABI/Python compatibility
constraint bore on the redesign.

The chosen design is narrow free functions plus explicit records, not a
service object or a friend god-class. Every cache entry point is now a free
function in `cdfmm::detail::cache` taking one of three explicit records
declared in `internal.hpp`: `CacheIdentityInputs`/`CacheIdentity` (identity;
`keys.cpp`), `UniversalCacheIdentity`/`UniversalCachePayload` (the
depth-independent bank and periodic root; `universal.cpp`), and
`GeometryCacheIdentity`/`GeometryCachePayload` (the geometry-dependent plan;
`geometry.cpp`). Payload records hold references to already-existing
`UniformFmm` members (`m2m_operators_`, `m2l_plan_`, `p2m_plans_`, and so on)
rather than a second copy of solver state; identity records hold small
values (strings, enums, an `int`) plus references to the `UniformTree`/
`StaticFmmTopology`/geometry data the identity is computed from.
`src/fmm/execution_setup.cpp` (identity) and `src/fmm/plan_preparation.cpp`
(universal/geometry load and write) assemble these records from `UniformFmm`'s
private state, call the free functions, and copy results back; they remain
the sole callers and plan preparation remains the sole owner of the
cache-vs-build decision. No `UniformFmm` cache method survives: all five
private declarations were removed from `include/cdfmm/uniform_fmm.hpp`, and
the three trivial `universal_cache_key()`/`geometry_cache_key()`/
`periodic_cache_key()` public accessors (previously also defined in
`keys.cpp`) moved to `src/fmm/uniform_fmm.cpp`, which already owns
`UniformFmm` lifecycle/accessors.

Two narrow types moved out of `uniform_fmm.hpp` to make the removal possible
without an ABI-affecting layout change: `P2MPlan`/`FloatP2MPlan` (previously
private nested structs of `UniformFmm`, needed by the geometry payload) moved
to `include/cdfmm/plan/static_coefficient.hpp` as free-standing types
alongside `StaticCoefficientOperator`; `ExpansionBasis` (previously declared
directly in `uniform_fmm.hpp`, needed by `CacheDescriptor`) moved to
`include/cdfmm/core/precision.hpp` alongside `StaticPrecision`. Both were
already unreachable except through `UniformFmm`'s public API by value, so
neither move changes any public name's meaning; `uniform_fmm.hpp` still
transitively provides both through its existing includes.
`sizeof(UniformFmm)` is unchanged (verified: 7272 bytes before and after), and
no public C++, C ABI, Python, or Fortran surface changed.

`StaticPlanStatistics` continued to be mutated directly by cache free
functions (hits, byte counts, phase timings), matching prior behaviour: it is
already a narrow plan-layer type declared in `include/cdfmm/timings.hpp` with
no dependency on `UniformFmm`, so this is not a coupling concern, and
introducing an alternative returned-result type for every cache call would
have added complexity without narrowing anything. The persisted format, cache
keys, and cache-key hash inputs/order are byte-for-byte unchanged; so is the
direct FP32 decode path and the asymmetric FP32-only failure cleanup that
`cache/AGENTS.md` already documented. Cache misses/corruption remain silent
and non-fatal.

Validation: portable `dev` (fresh) CTest 198 total, 193 passed, 4 expected
skips, and the same pre-existing "static triangular translations match M2M
and L2L references" failure confirmed identical on the unmodified `03be467`
baseline (unrelated floating-point comparison, not touched by this task);
Python 137 passed/7 skipped. Focused `ctest -R 'cache' -E 'tetrahedron
pairs|tetrahedron dispatch'`: 11 total, 9 passed, 2 expected oneMKL/CUDA
skips. A standalone cross-version probe (public-API-only source, unchanged
between builds) wrote an 8-file FP64/FP32 free/periodic cache corpus with an
isolated `v0.1.0` build, then read it back at HEAD: identical universal/
periodic/geometry cache keys, cache hits on every scenario (including
periodic), zero bytes written, cache files byte-identical before/after
(`sha256sum`), and field results bit-identical. A before/after timing probe
(identical geometry/options, both built without LTO to permit direct
linking) showed no measurable warm-construction slowdown (~0.725 s vs
~0.726 s mean over 5 repeats on a ~4,500-point case, within run-to-run
noise). CUDA + oneMKL `notebooks` (fresh, RTX 5090 SM120, CUDA 13.3.73):
**198/198 CTest with zero skips** (including the triangular-translation case,
which is sensitive to build configuration and passes here, matching this
project's established pattern of zero-skip CUDA+oneMKL runs), and the full
focused cache selector 11/11, including the two CUDA/oneMKL-gated cases that
the portable build skips; Python 143 passed/1 skipped against the
`build-notebooks` module. `git diff --check` clean.

### Public/internal API, header ownership, and packaging cleanup

A later task audited the remaining public/internal C++ interface and
packaging seams that Phase 1 closure and the internal-duplication,
tree/topology, and cache/plan-preparation cleanups had all left open: which
flat `include/cdfmm/` headers still lacked a canonical subsystem home,
whether they should get one, the `parameter_selection.hpp -> uniform_fmm.hpp`
dependency, the ownership of the CUDA/oneMKL capability queries, the exposure
of `P2MPlan`/`FloatP2MPlan` and `ExpansionBasis` from the cache-boundary
cleanup, and whether dip-fmm should offer a downstream CMake package. Three
small read-only audits (header ownership, the CMake/link-interface graph, and
the `parameter_selection.hpp`/`uniform_fmm.hpp`/CUDA-availability dependency
chain) preceded any change; each substantive flat header was read in full,
not assumed, before a relocate-or-keep decision.

**`parameter_selection.hpp` narrowed.** The header's own signatures need only
`Vec3` and `ExecutionBackend`; the complete `UniformFmm`/`UniformFmmOptions`
API is needed only by `src/parameter_selection.cpp`, which constructs
`UniformFmm` objects to time candidate configurations. `ExecutionBackend`
moved from `uniform_fmm.hpp` to a new narrow `include/cdfmm/backend/execution.hpp`
(a backend-selection enum, naturally owned by `backend/` alongside the CPU/
oneMKL/CUDA execution interfaces it selects between); `parameter_selection.hpp`
now includes only `cdfmm/backend/execution.hpp` and `cdfmm/math/vec3.hpp`.
`src/parameter_selection.cpp` gained an explicit `#include "cdfmm/uniform_fmm.hpp"`,
since it is the translation unit that actually needs the complete API — it
previously received that header only incidentally, through its own header's
transitive include. `uniform_fmm.hpp` includes `backend/execution.hpp` and
keeps re-exporting `ExecutionBackend` under its own name, so `#include
<cdfmm/uniform_fmm.hpp>` followed by `ExecutionBackend` usage is unchanged.
The other backend/plan enums that were also declared in `uniform_fmm.hpp`
(`SphericalM2LBackend`, `StaticMatrixBackend`, `StaticOperatorExecutor`,
`P2PExecutionPacking`, `StaticExecutionPlan`) were deliberately left in place:
nothing audited in this task needs them independently of the complete solver
API, and moving a type nothing depends on narrowly would only add an include
indirection without narrowing any real dependency edge. All four consumers of
`parameter_selection.hpp` (`cdfmm.hpp`, `python/internal.hpp`,
`src/parameter_selection.cpp`, `tests/test_parameter_selection.cpp`) were
checked individually: the first two already include `cdfmm/uniform_fmm.hpp`
directly, the third gained the explicit include above, and the fourth uses
only `deterministic_target_sample`/`branch_balance_ratio`, neither of which
touches `Vec3`, `ExecutionBackend`, or `UniformFmm`.

**CUDA/oneMKL availability queries relocated.** `cuda_compiled`,
`cuda_available`, `cuda_direct_available`, `cuda_m2l_p2p_available`,
`cuda_m2l_available`, `cuda_full_available`, and `cuda_device_description`
moved from `uniform_fmm.hpp` to a new canonical
`include/cdfmm/backend/cuda/availability.hpp`, matching the existing pattern
of `cdfmm/backend/cuda/dense_direct.hpp` declaring
`cuda_dense_direct_available()` beside the API it reports on; `one_mkl_available`
moved the same way to a new `include/cdfmm/backend/mkl/availability.hpp`, the
first public header under `backend/mkl/`. `uniform_fmm.hpp` includes both and
keeps re-exporting every name transitively, so existing `#include
<cdfmm/uniform_fmm.hpp>` source compiles unchanged. This closes the one
dependency-audit exception Phase 1 closure recorded and repeated through the
"Deferred refactor inventory": `src/backend/cuda/fmm/internal.hpp` included
`cdfmm/uniform_fmm.hpp` solely so `backend/cuda/fmm/plan.cu` and
`backend/cuda/stub/fmm.cpp` could see the declarations of the three
availability functions they define (`cuda_m2l_available`,
`cuda_m2l_p2p_available`, `cuda_full_available`) — confirmed by inspection:
every other type that header/translation-unit pair uses
(`StaticOperatorEntry`, `StaticM2LPlan`, `FloatStaticM2LPlan`,
`StaticP2POperator`, `StaticP2PBsrPlan`, `StaticP2PSignedTensorDictionaryPlan`
and their FP32 counterparts, `CudaPlanStatistics`, `CudaEvaluationTimings`,
`Vec3`, `FloatVec3`) already arrives through `cdfmm/plan/static_plan.hpp`,
`cdfmm/timings.hpp`, and `cdfmm/math/potential_field.hpp`, which that header
already included directly. `internal.hpp` now includes
`cdfmm/backend/cuda/availability.hpp` instead of the complete solver header,
so the CUDA backend no longer depends on the top-level public API merely to
declare functions it implements; it depends only on the narrow backend-facing
header that owns those declarations. `cuda_direct_available` (defined in
`backend/cuda/direct/direct.cu`/`backend/cuda/stub/direct.cpp`) and
`one_mkl_available` (defined in `backend/mkl/m2l.cpp`) were already being
defined without their declaration in scope — legal in C++ for a free function,
if untidy — and were left as found: fixing that pre-existing decl-visibility
gap was not needed to resolve the one documented reverse-dependency edge and
was out of this task's evidenced scope.

**`tensor_dictionary.hpp` relocated.** `CanonicalTensor6`, `canonicalise_tensor6`,
`Tensor6BitKey`, `tensor6_bit_key`, `pack_tensor6_token`, `tensor6_token_id`,
and `tensor6_token_sign_mask` moved from the flat `cdfmm/tensor_dictionary.hpp`
to `include/cdfmm/plan/p2p/tensor_dictionary.hpp`, alongside the other P2P
packing headers (`dictionary.hpp`, `signed_dictionary.hpp`, `bsr.hpp`, ...)
that are their only consumers. The header has no dependency on any other
`cdfmm` header, so the move is a pure relocation with no cascading include
changes. `src/plan/precision.cpp`, `src/plan/p2p/dictionary_detail.hpp`,
`src/plan/p2p/signed_dictionary.cpp`, and `src/backend/cpu/p2p/dictionary.cpp`
(its four internal consumers) now include the canonical path directly;
`include/cdfmm/static_operators.hpp` keeps including the flat
`cdfmm/tensor_dictionary.hpp` path, because it is itself a compatibility
umbrella whose job is to reproduce the pre-v0.2 transitive surface, so a
façade including a façade is correct there, not an exception to fix.

**`timings.hpp`, `periodic.hpp`, and `validation.hpp` audited and kept flat.**
Each was read in full and found to have no single clearer canonical owner
than "flat public header", rather than being deferred only because no one had
looked yet:

- `timings.hpp` declares `EvaluationTimings` (`fmm`-owned: complete
  near/far/M2L-phase wall-clock breakdown), `StaticPlanStatistics`
  (`plan`-owned: one-time construction cost/storage), and
  `CudaPlanStatistics`/`CudaEvaluationTimings` (CUDA-backend-owned: device
  traffic/persistent-allocation diagnostics and device-stream phase timings).
  These are read together through one diagnostics surface
  (`UniformFmm::last_timings()`/`static_plan_statistics()`/
  `cuda_plan_statistics()`, and `fmm/diagnostics.cpp`'s summary), so splitting
  the file across `fmm/`, `plan/`, and `backend/cuda/` would scatter one
  cohesive observability concept across three directories for no consumer's
  benefit. This mirrors why `uniform_fmm.hpp` itself stays flat: a genuinely
  cross-cutting public surface does not need an owner among its lower layers.
- `periodic.hpp` declares periodic-cell configuration
  (`PeriodicCellOptions`/`PeriodicConvention`) and periodic-boundary
  mathematics (box wrapping, `build_periodic_list1`/`list2`,
  `periodic_laplace_derivatives_raw`). Its consumers are
  `include/cdfmm/operators/m2l.hpp` (periodic M2L construction) and
  `include/cdfmm/tree/uniform_topology.hpp` (periodic box-list topology) —
  the `operators` and `tree` layers, which the architecture's own layering
  diagram places as *siblings* feeding `plan`, never depending on each other.
  Giving `periodic.hpp` to either would create exactly the sibling-to-sibling
  edge the diagram is designed to avoid; it is legitimately shared
  lower-level content, symmetrically depended on, like `math`/`core` but not
  itself pure math (it also owns solver-facing configuration types embedded
  directly in `UniformFmmOptions`). Left flat, with the pre-existing
  `canonical headers -> cdfmm/periodic.hpp` edge documented as intentional
  rather than transitional.
- `validation.hpp` declares generic vector-error metrics
  (`norm`/`absolute_error`/`relative_error`/`compute_error_metrics`, pure
  `Vec3` math with no FMM concept) and `direct_p2p_reference`, an O(N^2) P2P
  reference wrapper explicitly documented as "not part of production FMM
  traversal logic" and intended for tests/examples. This is cross-cutting
  validation/diagnostic utility by design, not a subsystem's production
  interface; the header's own doc comment already said so. Left flat.

**`P2MPlan`/`FloatP2MPlan` and `ExpansionBasis` reviewed, kept in place.**
Both moves were made by the preceding cache-boundary cleanup so cache payload
records could name them without depending on `UniformFmm`; this task's brief
asked whether that placement should now be reconsidered rather than assumed
permanent. `P2MPlan`/`FloatP2MPlan` in `include/cdfmm/plan/static_coefficient.hpp`
are a minimal, self-contained specialisation ("leaf index plus
`StaticCoefficientOperator`") of the coefficient-map types already declared
immediately above them in the same file; they introduce no god-object, no
PIMPL, and no ABI concern, and are exactly the kind of "legitimate canonical
static-plan record" the brief anticipated as one acceptable outcome. Moving
them again, or hiding them behind a `detail` namespace, would add machinery to
solve a problem that does not exist: nothing about their current visibility
is unsafe or confusing, and `sizeof(UniformFmm)` is unaffected either way.
`ExpansionBasis` in `include/cdfmm/core/precision.hpp`, alongside
`StaticPrecision`, is a coherent pairing: both are small, `core`-level scalar
configuration enums with no dependency of their own, and `ExpansionBasis` is,
per the brief's own framing, already a genuine user-facing configuration
concept (`UniformFmmOptions::expansion_basis`) rather than an incidental
implementation detail — `core/precision.hpp` is a defensible, non-arbitrary
canonical home for it, not merely wherever the previous task happened to put
it. Both are kept unchanged.

**Public enum/value-type survey.** `SphericalM2LBackend`, `StaticMatrixBackend`,
`StaticOperatorExecutor`, `P2PExecutionPacking`, and `StaticExecutionPlan`
remain declared in `uniform_fmm.hpp`. Each was classified (backend
capability/plan representation, all read or set only through the complete
`UniformFmm` API and its options/diagnostics) and none has an external
consumer that would benefit from a narrower dependency the way
`parameter_selection.hpp` did for `ExecutionBackend`; moving them would not
reduce any dependency width in the current tree, only relocate declarations
for their own sake, which the brief explicitly warned against.

**Downstream CMake package added.** `cdfmm_c` — the shared library exposing
the stable, unconditional C ABI (`include/cdfmm/c_api.h`) — is the only
installed target; its private static implementation library, `cdfmm_core`,
resolves OpenMP, oneMKL, and CUDA (`CUDA::cudart`/`CUDA::cublas`/`CUDA::cusparse`)
as `PUBLIC` link dependencies of its own build, but `cdfmm_c` links
`cdfmm_core` `PRIVATE` and declares no `PUBLIC`/`INTERFACE` link libraries of
its own (only `PUBLIC` include directories), so none of `cdfmm_core`'s
dependencies appear in `cdfmm_c`'s exported usage requirements — confirmed by
inspecting the generated `cdfmm_c-targets.cmake`, whose imported target has
`INTERFACE_INCLUDE_DIRECTORIES` and nothing else. `cdfmm_core` itself is
never installed, so there is no static-linking downstream-consumer case to
support: the exported package describes exactly one, shared, library. The
project gained an explicit `VERSION 0.1.0` (matching the released
`pyproject.toml` Python package version; not a v0.2 marker). A new
`add_library(cdfmm::cdfmm_c ALIAS cdfmm_c)`, `install(EXPORT cdfmm_c-targets
...)`, a generated `cmake/cdfmmConfig.cmake.in` (via
`configure_package_config_file`), and a generated version file (via
`write_basic_package_version_file(... COMPATIBILITY SameMajorVersion)`) are
installed to `${CMAKE_INSTALL_LIBDIR}/cmake/cdfmm`, cleaned up by the existing
`cdfmm-clean-install` target. The config calls no `find_dependency(...)`: it
does not need to, since the exported target's interface genuinely has no
CUDA/oneMKL/OpenMP dependency to discover — the alternative of adding
conditional `find_dependency` calls "to be safe" was rejected as describing a
dependency that does not exist. A downstream consumer's runtime need for
`libcudart`/`libmkl*`/`libomp` (when `cdfmm_c` was built with those backends)
remains an ordinary transitive shared-library runtime requirement resolved by
the dynamic loader from `libcdfmm_c.so`'s own recorded `DT_NEEDED` entries,
exactly as it was before this task; nothing about that changed.

Validation: an isolated-prefix portable install (`CDFMM_ENABLE_CUDA=OFF`,
`CDFMM_ENABLE_MKL` unset) produced `lib/cmake/cdfmm/{cdfmmConfig.cmake,
cdfmmConfigVersion.cmake,cdfmm_c-targets.cmake,cdfmm_c-targets-release.cmake}`
alongside the existing headers/library; grepping the installed prefix for the
source and build tree paths found none. A standalone out-of-tree consumer
(`cmake_minimum_required(VERSION 3.20)`, `find_package(cdfmm CONFIG REQUIRED)`,
`target_link_libraries(consumer PRIVATE cdfmm::cdfmm_c)`) configured, built,
and linked against only that isolated prefix (`CMAKE_PREFIX_PATH` pointed at
it, nothing else), then ran a two-point-dipole evaluation through the C ABI
and reproduced the analytic field to a relative error of `1.7e-16` — using
only the installed prefix, matching the pre-existing "Installation and
downstream consumption" C-consumer evidence above but now driven entirely
through `find_package` rather than a manually supplied include path and
library name. (The dynamic loader initially resolved a stale, previously
installed `libcdfmm_c.so` from the active Conda environment's own `lib/`
directory ahead of the freshly built one, because the IntelLLVM host compiler
used in this environment embeds that directory in `DT_RPATH`; this is a
pre-existing environment artifact of that toolchain, not a consequence of the
new package config, confirmed by forcing resolution to the freshly built
library with `LD_PRELOAD` and observing its own build-configuration
diagnostics — `build.cuda_compiled: false`, matching that isolated build's
options — replace the stale library's.)

No solver algorithm, cache format, cache key, C ABI, Python API, or Fortran
interface changed. Portable `dev` (fresh build, incremental after the header
moves): CTest 197/198 passed, the one failure being the same pre-existing
"static triangular translations match M2M and L2L references" case (confirmed
present before this task's changes, per the baseline check below), 4 expected
skips; Python 137 passed/7 skipped against the just-built `build/` module. CUDA
+ oneMKL `notebooks` (fresh, RTX 5090 SM120, CUDA 13.3.73, oneMKL 2026.1;
built at `-j 2` after the default unbounded `-j` hit unrelated `nvcc`
front-end internal-compiler-errors on four `.cu` files under parallel load —
confirmed environmental, not a defect, by recompiling one of the four in
isolation with identical flags before retrying at lower parallelism):
**198/198 CTest with zero skips**, including the triangular-translation case,
matching this project's established zero-skip CUDA+oneMKL pattern; Python 143
passed/1 skipped against `build-notebooks`. One pinned Python test,
`python_tests/test_profiling_setup.py::test_notebook_preset_supports_gnu_mkl_without_intel_header_leakage`,
asserted the exact pre-`VERSION` `project(...)` text as an anchor for its real
assertion (`LANGUAGES C CXX` staying enabled); updated to the new literal. C
ABI: unchanged 14 `cdfmm_*` symbols. Python: unchanged 63 top-level names (34
classes, 29 functions). Standalone-compile and mixed/reverse-include-order
probes covered every header this task moved or added.

### Baseline check before this task

Before any production change, "static triangular translations match M2M and
L2L references" was run once against the unmodified `9bab9ee6` tree in the
portable `dev` configuration and failed with the same `REQUIRE` and printed
values as it did after this task's changes, confirming the failure predates
this task and is an unrelated floating-point comparison, not a regression it
introduced.

## Refactor validation contract

Each production-code step starts and ends with the same relevant build and
test configurations. Numerical tests must compare unchanged formulas,
precision modes, output flags, geometries, tree modes, periodic behaviour,
cache behaviour, and supported CPU/oneMKL/CUDA paths. Performance-sensitive
backend changes additionally compare plan reuse, transfers, allocations,
launches, synchronisation, and representative benchmark results.

The high-level `UniformFmm`, cache, and bindings boundaries are complete. These
refactor steps change file ownership and include structure without changing
runtime algorithms or public names. Compatibility/transitional source review
and whole-refactor validation are both complete, so Phase 1 is closed. API
redesign, packing, generation, discretisation, and refinement remain deferred.

## Phase 1 closure

Phase 1 of the `v0.2` architecture refactor is **COMPLETE**. The preserved
pre-refactor baseline remains the `v0.1.0` annotated tag (commit `2d3d4ea`) and
the `release/v0.1` branch; neither was moved. Closure rests on the evidence
below rather than on assertion.

### Ownership and dependency audit

A repository-wide include-edge audit of `include/cdfmm/`, `src/`, and `python/`
found no violation of the prohibited-by-default list: no `geometry -> backend`
or `geometry -> CUDA`, no `tree -> CUDA`, no `math -> FMM`, no
`operators -> orchestration`, no `operators -> cuBLAS/cuSPARSE`, no
`plan -> Python`, no `CPU backend -> CUDA internals`, and no binding unit
including an internal `src/` header. No CUDA backend redefines canonical
mathematics or topology: each consumes prepared `Static*Plan` data.

Compatibility flows one way. Every flat header under `include/cdfmm/` was an
installed public path at `v0.1.0` and is retained; canonical subsystem headers
and canonical implementation include canonical headers, and the façades include
those. The audited exceptions are each deliberate and are recorded here in full:

- `src/operators.cpp` implements `cdfmm/operators.hpp`, the flat operator
  compatibility API, and so includes it.
- `python/internal.hpp` must see the flat operator declarations the Python
  module exports.
- `cdfmm/timings.hpp` and `cdfmm/periodic.hpp` are still included by some
  canonical headers. They are substantive public headers awaiting a subsystem
  home, not façades; their migration is deferred to Phase 2.
- `src/plan/direct/dense.cpp` includes `backend/cpu/direct/dense.hpp` and
  `backend/mkl/direct/dense.hpp`. `DenseDirectPlan::evaluate()` is a pre-v0.2
  public method, so the plan object itself must dispatch to an executor. This
  is an isolated compatibility seam confined to one translation unit; the
  public `plan/direct/dense.hpp` header depends on no backend, and no other
  file under `src/plan/` or `src/operators/` includes a backend header.
- At Phase 1 closure, `src/backend/cuda/fmm/internal.hpp` included
  `cdfmm/uniform_fmm.hpp` because the CUDA backend *implements* the public
  availability queries declared there (`cuda_m2l_p2p_available`,
  `cuda_m2l_available`). That was a supported public-API relationship, not an
  implementation-layer inversion, but it is now historical: the later
  public-header/packaging cleanup relocated those declarations (and the rest
  of the CUDA/oneMKL capability queries) to the canonical
  `cdfmm/backend/cuda/availability.hpp`/`cdfmm/backend/mkl/availability.hpp`,
  which `backend/cuda/fmm/internal.hpp` now includes instead of the complete
  solver header. See "Public/internal API, header ownership, and packaging
  cleanup" below.
- At Phase 1 closure, `src/cache/internal.hpp` included `cdfmm/uniform_fmm.hpp`
  as a sanctioned `cache -> already-defined solver data` edge. This is now
  historical, not current: the later "Cache/plan-preparation boundary
  cleanup" (below) removed the include entirely, so `src/cache/internal.hpp`
  no longer depends on `cdfmm/uniform_fmm.hpp` at all. This paragraph is kept
  as a record of what Phase 1 closure found at the time; do not read it as a
  statement about the current tree.

`tree/adaptive_tree.hpp` returning `StaticFmmTopology` is not a
dependency-direction exception: both types live in the `tree` layer, and the
tree/topology boundary cleanup confirmed `StaticFmmTopology` carries no
plan/backend data, so this is ordinary intra-layer usage. It was listed here
as a deferred exception through Phase 1; see "Tree/topology boundary
cleanup".

Every `.cpp`/`.cu` file under `src/` is referenced by `CMakeLists.txt`; no
stale or transitional source remains compiled, and no build file references a
removed path.

### Comparison against `v0.1.0`

No header present at `v0.1.0` is absent at HEAD: all 29 pre-refactor public
paths are retained as compatibility façades. `include/cdfmm/c_api.h` is
byte-identical to `v0.1.0` (same blob hash), `CDFMM_ABI_VERSION` remains `1`,
and the built `libcdfmm_c.so` exports exactly the expected 14 `cdfmm_*`
functions. `fortran/cdfmm_fortran.f90` is likewise byte-identical. The Python
surface is unchanged: 20 classes, 14 enums, 29 module functions, 151
methods/properties, 130 `py::arg` entries with identical defaults, and 123
diagnostic dictionary keys all compare zero-diff against the pre-split
`python/bindings.cpp`.

The installed-header rule widened from `c_api.h` only to the whole
`include/cdfmm/` tree with `AGENTS.md` excluded. That is a deliberate widening,
not a narrowing, and no installed header points into `src/`, `python/`, or a
build-tree path.

Behaviour is preserved exactly where it can be compared directly: a probe
building five representative plans (spherical FP32, spherical FP64, prism FP32,
Cartesian FP64, and periodic FP64) compiles unchanged against both `v0.1.0` and
HEAD public headers and produces **bit-identical** field results.

`9003b666` (tetrahedron P2P construction) is separate, intentional
performance/numerical work carried on this branch and is not an architecture
regression.

### Validation matrix

All builds used the `cdfmm` Conda environment: GCC 15.3, CMake 4.4.3,
CUDA 13.3.73, oneMKL 2026.1, Python 3.11, on an NVIDIA RTX 5090 (SM120,
driver 595.84). Python runs verified `cdfmm.__file__` resolved to the
just-built module in every case.

| Configuration | Build | CTest | Python |
|---|---|---|---|
| Portable CPU (`dev` preset, fresh) | clean, no warnings | 198/198 passed, 4 skipped | 137 passed, 7 skipped |
| oneMKL, no CUDA | clean | 198/198 passed, 3 skipped | 139 passed, 5 skipped |
| CUDA, no oneMKL (`cuda` preset, fresh, SM120) | clean | 198/198 passed, 1 skipped | 141 passed, 3 skipped |
| CUDA + oneMKL (`notebooks` preset, fresh) | clean | **198/198 passed, 0 skipped** | 143 passed, 1 skipped |
| Fortran interface (`ifx` 2025.2.1) | clean | 199/199 passed, 4 skipped | n/a |

Every skip is a genuinely absent optional dependency or device: CUDA
unavailable, oneMKL unavailable, or the optional external MagTense package. The
CUDA + oneMKL configuration exercises the complete matrix with no skips.

CUDA numerical execution is real, not vacuous. In a CUDA build the
device-backed cases run for measurable time (for example "spherical CUDA
partial and full agree with CPU static" 0.36 s, "cache and no-cache CUDA-full
paths agree" 0.35 s, "shared CUDA M2L executor agrees with the canonical CPU
plan" 2.05 s), whereas in a portable build the same cases report 0.00 s because
they return early. CPU-reference, CPU-static, CUDA M2L/P2P hybrid, and CUDA
full-FMM agreement is therefore covered by tests that genuinely executed on the
GPU.

### Cache backward compatibility

A cache corpus of ten files was generated by the `v0.1.0` baseline build
(universal spherical FP32/FP64, universal Cartesian FP64, a periodic root, and
five geometry plans). Running HEAD against that untouched corpus produced, for
every one of the five cases: the **same key strings**, `universal.hit: true`,
`geometry.hit: true`, `periodic.hit: true` where applicable, and
`cache.bytes_written: 0`. All ten files remained byte-identical afterwards, and
the field results were bit-identical to the baseline run. The schema version,
operator version, magic, hash inputs, and format layout were not changed.

### Installation and downstream consumption

A clean install to an isolated prefix produced 84 files: 79 headers, the
`cdfmm_c` shared library, the `cdfmm-precompute` tool, the Python extension,
and the installed Fortran source. No `AGENTS.md` was installed and no installed
header references `src/` or `python/`. Every one of the 79 installed C++
headers compiles standalone against the prefix alone, with no include-ordering
requirement. A translation unit mixing all legacy flat paths with all canonical
subsystem paths compiles in both include orders. A downstream C consumer built
only against the installed prefix creates a plan, evaluates it, and reproduces
the analytic two-dipole field to a relative error of `1.6e-16`.

Note that dip-fmm exports no CMake package configuration, at `v0.1.0` or at
HEAD; downstream C++ consumers link `cdfmm_c` directly. This is a pre-existing
characteristic, not a refactor regression.

### Documentation

`sphinx-build -W --keep-going -b html docs docs/_build/html` now succeeds with
**zero** warnings. The two long-standing `docs/api.rst` C++ declaration
warnings were a genuine Phase-1 defect and were fixed rather than tolerated:
`docs/Doxyfile` predefined `CDFMM_HOST_DEVICE`, a macro used nowhere, while the
canonical `plan/p2p/canonical.hpp` header introduced by the refactor guards two
declarations with `CDFMM_PLAN_HOST_DEVICE`. With `EXPAND_ONLY_PREDEF = YES`,
Doxygen left that token unexpanded and Breathe could not parse the two
declarations. `PREDEFINED` now names the macro actually in use.

### Performance sanity

A representative 20k-source/20k-target, depth-3, order-4 FP32 evaluation gives
a median of 11.47 ms on the portable CPU static-matrix backend at HEAD against
11.56 ms at `v0.1.0` — no regression. CUDA medians are 0.96 ms (partial) and
0.88 ms (full), so near/far overlap and persistent device state are intact and
no accidental backend fallback occurs.

### Defects found and fixed during closure

- `tests/test_fortran_api.f90` requested a periodic cubic cell of side 2
  centred at the origin while its two prisms sit at `z = 0` and `z = 1` with
  side 0.2, so the `z = 1` prism lay outside the periodic root and plan
  creation correctly failed. The cell centre is now `(0, 0, 0.5)`, matching the
  period-2 chain the case intends. This is a **pre-existing** defect: the file
  is byte-identical to `v0.1.0`, and a `v0.1.0` build of the same test fails
  identically. It went undetected because no Fortran compiler was available in
  any previous session.
- `docs/Doxyfile` named a macro that does not exist, as described above.
- `include/cdfmm/AGENTS.md`, `src/AGENTS.md`, and three passages of this
  document described already-implemented work as pending. Documentation that
  contradicts the implementation is a Phase-1 defect, and these were corrected.

## Phase 2 handoff

The Phase-2 cleanup list is now fully resolved. Every item below was either
closed by a narrow follow-up task after Phase 1 closure or, for public
headers with no single clear subsystem owner, deliberately and explicitly
kept flat rather than left merely unmigrated. Phase 2 as a whole is
**COMPLETE**.

Resolved, kept here only as a record: the canonical-to-compact P2P packing
rule and the duplicated factorial helper (internal-duplication cleanup); the
`StaticFmmTopology`/tree-to-plan seam (tree/topology boundary cleanup, see
above); cache entry points as `UniformFmm` members (cache/plan-preparation
boundary cleanup, see above — every cache function is now a free function
taking an explicit identity/payload record, and no `UniformFmm` cache method
remains); and the public/internal API, header ownership, and packaging
cleanup (see above):

- **Public-header homes.** `tensor_dictionary.hpp` moved to
  `plan/p2p/tensor_dictionary.hpp`. `ExecutionBackend` moved to
  `backend/execution.hpp`. The CUDA/oneMKL availability queries moved to
  `backend/cuda/availability.hpp`/`backend/mkl/availability.hpp`.
  `timings.hpp`, `periodic.hpp`, `validation.hpp`, and `uniform_fmm.hpp`
  itself were each audited and kept flat, with the specific reason recorded
  above for each — not because no one had looked, but because none has one
  single clearer subsystem owner. `src/periodic.cpp`, `src/parameter_selection.cpp`,
  and `src/validation.cpp` accordingly stay flat too: relocating an
  implementation file to a subsystem its own public header does not belong to
  would not clarify anything.
- **`parameter_selection.hpp` → `uniform_fmm.hpp` seam.** Resolved:
  `parameter_selection.hpp` now depends only on `backend/execution.hpp` and
  `math/vec3.hpp`.
- **API and internal simplification.** CUDA availability queries relocated;
  a minimal CMake package configuration (`find_package(cdfmm CONFIG)`,
  `cdfmm::cdfmm_c`) added; CUDA/oneMKL/OpenMP dependencies confirmed private
  to the never-installed `cdfmm_core` and absent from `cdfmm_c`'s exported
  interface, so there was no propagation to stop — the exported target
  already carried none of them.
- **Responsibility-review candidates.** `backend/cuda/p2p/plan.cu`,
  `geometry/primitives/tetrahedron.cpp`, and `backend/cuda/fmm/plan.cu` remain
  the largest units and were not touched by this cleanup. Size locates audit
  work; it does not mandate splitting, and this is not a Phase-2 item.
- **Coverage gaps.** CI remains portable CPU only; oneMKL, CUDA, and Fortran
  are validated manually. Most CUDA-gated C++ cases return through `SUCCEED()`
  rather than a true `SKIP()`, so a portable-CPU run reports them as passed;
  only four cases report a real CTest skip. Making that distinction visible
  would make portable-run results easier to read literally. Not a Phase-2
  item; left as a standing observation.

**Obsolete test/doc/example audit and repository pruning**, and the future
`tests/{unit,backend,integration}` and `benchmarks/{direct,p2p,far_field,fmm}`
taxonomies, remain explicitly out of scope and unstarted. They are **not**
the next planned work: the next phase is a dedicated performance-optimization
phase covering both construction/plan-preparation and evaluation/repeated
execution across the CPU, oneMKL, and CUDA paths (profiling, allocation,
memory traffic, cache construction, P2P/M2L/hierarchy passes, kernel
structure, occupancy, launch overhead, transfers, asynchronous overlap,
reuse, and representative end-to-end workloads), because benchmark/test
infrastructure should remain available while that work is done. Repository
pruning follows the optimization phase, not this cleanup. A future `v0.2`
release tag or branch is a separate, explicit step that only follows both;
none was created by this closure.

---

# From `docs/static-p2p.md`: sweep-based recommendation and further experiments

## Sweep-based recommendation

The latest complete sweep covered 19 runnable `(particle count, depth)` cases
from 2^12 through 2^17 particles and depths two through five. CPU SoA won
12 cases, was within 10% of the winner in 15, and had a 1.154 geometric-mean
speedup over canonical AoS. It became the portable CPU default for stored
tensors. Since the Phase-3B CPU evaluation work, point-source / point-target
plans on `CpuStatic` recompute their list-1 pairs from the sorted positions
instead (`P2PExecutionPacking::PointGeometry`): the stored-tensor kernels are
DRAM-bound at 29-53 bytes per pair, while the positions of a target leaf's
neighbourhood stay cache resident. Periodic point plans joined that default in
the Phase-3 P2P unification (the executor folds each record's image shift in
and was measured 2.7-5.1x faster than the SoA rows on the periodic near
field). The SoA tensors remain the default for finite near fields (see
`docs/backends.md` and `agent_docs/performance_optimization.md`).

CUDA BSR(3) won 12 cases, was within 10% of the winner in 17, and had a 1.757
geometric-mean kernel speedup over canonical CUDA. Its full blocks use about
35% more persistent P2P storage than canonical rows, so it is the preferred
fixed-identity path subject to the explicit memory budget rather than an
unconditional choice. Canonical CUDA remains the correct default for changing
identity maps and the fallback when BSR is too large.

Leaf-block CUDA won only the high-occupancy `(N=65536, depth=4)` and
`(N=131072, depth=4)` cases in that sweep. Those two points do not yet justify
an occupancy dispatch rule, so leaf blocking remains an experimental packing.
(Phase 3A later made leaf blocks the general CUDA default; see below.)

## Position-based point P2P on CUDA (Phase 3B.5)

The CUDA `PointGeometry` kernel keeps one warp per canonical list-1 record
(target leaf, source leaf, image) and the leaf-block lane layout, but reads
one aligned position (16 bytes in FP32) and one moment per source instead of
six stored tensor components, and recomputes the pair with the shared
point-dipole formula. Measured on the RTX 5090 against the best stored
packing of each plan (`agent_docs/performance_optimization.md`, Phase 3B.5,
FP32, order 6, `cuda-full`):

- random points, depth 3, 8 to 128 points per leaf: the kernel is 1.3x (8 per
  leaf) to 7.5x (128 per leaf) faster than the leaf blocks and the evaluation
  1.0-6.2x faster, with 2-33x less persistent device memory (14-144 MB
  instead of 32-4747 MB);
- regular lattices: in evaluation time it is faster than or equal to the
  dictionary at every measured occupancy (8, 16, 32 and 64 per leaf, 4096 to
  262144 points); the dictionary kernel alone is still up to 2x faster at 8
  points per leaf on a 262k lattice, where the far field dominates the
  evaluation and the two agree within 1.3 %;
- FP64: recomputation is 1.7-3x slower than the leaf blocks and 2x slower than
  the dictionary, because the consumer GPU's FP64 rate is a small fraction of
  its FP32 rate.

The CUDA execution policy therefore selects `PointGeometry` for FP32 plans
with point sources and point targets on any layout, and keeps the leaf blocks
and the lattice dictionary for FP64 point plans and for finite bodies. The
stored packings remain explicit choices (`p2p_packing`,
`use_reduced_symmetry_p2p`). Resident memory of the procedural plan is the
sorted positions (16 or 32 bytes per point), one 32- or 48-byte record per
list-1 leaf pair and the identity map.

## Further experiments

1. Profile the fixed-identity BSR and canonical fallback with Nsight; compare
   kernel time separately from transfers and confirm the depth-five crossover.
2. Tune leaf source-batch size and block size only if the profiler identifies
   shared-memory, occupancy, or register-pressure limits.
3. Test AoSoA and source batching for long CPU rows.
4. Revalidate the unconditional CPU SoA policy on additional architectures
   with fixed affinity, cache, SIMD, and NUMA measurements.
5. Benchmark multiple moment right-hand sides separately; that workload may
   favour dense leaf kernels or GEMM.


---

# From `docs/benchmarks.md`: focused notebook results

## Focused notebook results

The controlled Cartesian/spherical comparisons found the same relative L2
accuracy, to displayed numerical precision, for both bases at every tested
order. This held for the original point-source/point-target comparison at
orders 1--6, 8, and 10 and for the uniform-cube/volume-averaged-cube comparison
over orders 1--6. The result validates the algebraic equivalence of the two
bases for these geometries; it does not imply equal coefficient count, setup
cost, memory, or evaluation time.

On the 512-cube FP64 CUDA-full case at order 6, Cartesian and spherical setup
took 9.44 s and 10.09 s, respectively, while repeated evaluation took 1.50 ms
and 0.87 ms. Spherical retained 49 coefficients instead of Cartesian's 84 and
used about 20.2 MiB instead of 45.7 MiB in total retained host-plus-device
state. These timings were measured on an NVIDIA GeForce RTX 5090 with eight
setup threads and are hardware-specific. They also show that finite-cuboid
operator construction, rather than repeated GPU evaluation, dominated this
small comparison.

The independent P2M/L2P comparison used a 15x15x15 lattice of 10 nm cubes at
30 nm spacing, FP64 spherical CUDA-full execution, orders 4 and 6, and tree
depths 2--5. Exact cuboid-to-cuboid P2P was fixed in all four cases. At order 4,
point and cuboid P2M/L2P produced identical errors. At order 6, enabling either
cuboid far-field endpoint gave no accuracy gain and slightly increased the
measured error. At depth 5 the relative L2 results were:

| Far-field P2M / L2P | Relative L2 error |
|---|---:|
| point / point | 1.3448e-3 |
| cuboid / point | 1.4135e-3 |
| point / cuboid | 1.3946e-3 |
| cuboid / cuboid | 1.4675e-3 |

This is an observation about the tested low orders, not evidence that the
finite-cuboid operators are generally less accurate. For a centred cube, the
degree-three shape correction is proportional to the Laplacian and vanishes
outside the source. The first physical shape correction is degree five and
scales as `O((h/R)^4)`. Order 4 therefore cannot distinguish the cases, while
order 6 includes only the first small correction. Point-geometry error can
also partially cancel FMM truncation error, so adding a physical shape
correction need not improve the total error monotonically at fixed order.
Non-cubic cells, higher orders, and different size-to-separation ratios remain
separate comparisons.

The same run measured construction of its 3,375-by-3,375 exact dense
cuboid-to-cuboid reference at 239.00 s and one portable evaluation at 59.7 ms.
Construction forms 11,390,625 exact finite-volume pair tensors, or about
21 microseconds per pair, and retains six FP64 component matrices totalling
about 521 MiB. The current generic constructor neither parallelises this loop
nor reuses repeated lattice displacements. This regular lattice has only
29^3 = 24,389 distinct displacements, so displacement caching or a specialised
convolution plan is the principal prospective optimisation. The fast repeated
evaluation reflects reuse of the already constructed matrices.

Setting both far-field model selectors to their point variants provides the
tested hybrid model: list1 P2P remains exact prism-to-prism, while far-field
P2M treats each supplied total moment as a point dipole at its prism centre
and L2P samples the local expansion at the target centre. Source and target
sizes select near-field physics; callers must still convert magnetisation to
total dipole moment before evaluation.


---

# `docs/cuda-m2l-performance.md` (2026-08-17)


## Decision

The production CUDA M2L path uses one shared deterministic target-row executor
for `cuda-partial` and `cuda-full`. It derives compact active-row metadata from
the canonical `StaticM2LPlan`, pre-scales each resident multipole coefficient
once, and applies local scaling after the complete interaction/alpha reduction.
The canonical matrices, transfer-class IDs, interaction multiset, coefficient
ordering, FP64 precision, and backend transfer boundaries are unchanged.

The optimised executor is retained because every acceptance case improved by
at least 25.0% in median complete repeated-evaluation time. No order dispatch,
cuBLAS linkage, class-grouped staging, atomic scatter, or evaluation-time
allocation is retained. The simpler grouped alternatives were not promoted to
production after the target-row candidate cleared the 5% gate across orders 4,
6, and 8 without a regression.

## Measurement environment

- Date: 2026-08-17
- Baseline commit: `df9d36259771b1938896174827a02db0a5d7fc5e`
- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0, 32,607 MiB
- Driver: 595.84
- CUDA compiler/toolkit: 13.3.73
- Host compiler: GNU 14.4.0
- Build: `RelWithDebInfo`, native CUDA architecture, OpenMP, oneMKL, NVTX
- Runtime: `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=1`
- Sampling: two warm-ups, 20 evaluations per sample, five samples
- Launch: 256 threads per target-row block on this device

The primary metric is the median complete repeated-evaluation time. M2L is the
CUDA-event parent duration and includes the pre-scaling kernel. Raw CSV,
`.nsys-rep`, SQLite, and attempted `.ncu-rep` files are stored below the ignored
`profiling_results/cuda_m2l_optimisation/` directory.

## Results

| N | Order | Depth | Backend | Baseline eval (ms) | Final eval (ms) | Improvement | Baseline M2L (ms) | Final M2L (ms) |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 10,000 | 4 | 3 | cuda-partial | 1.539 | 0.958 | 37.8% | 1.119 | 0.658 |
| 10,000 | 4 | 3 | cuda-full | 1.185 | 0.889 | 25.0% | 1.059 | 0.604 |
| 50,000 | 4 | 4 | cuda-partial | 7.454 | 4.524 | 39.3% | 4.472 | 1.746 |
| 50,000 | 4 | 4 | cuda-full | 5.209 | 2.577 | 50.5% | 4.559 | 1.951 |
| 50,000 | 6 | 4 | cuda-partial | 28.687 | 15.049 | 47.5% | 20.136 | 7.442 |
| 50,000 | 6 | 4 | cuda-full | 20.432 | 7.867 | 61.5% | 19.468 | 6.894 |
| 100,000 | 6 | 5 | cuda-partial | 162.431 | 71.569 | 55.9% | 139.219 | 48.479 |
| 100,000 | 6 | 5 | cuda-full | 136.788 | 45.692 | 66.6% | 133.580 | 42.575 |
| 20,000 | 8 | 3 | cuda-partial | 15.045 | 8.334 | 44.6% | 10.010 | 3.425 |
| 20,000 | 8 | 3 | cuda-full | 10.935 | 4.231 | 61.3% | 10.455 | 3.729 |
| 50,000 | 8 | 4 | cuda-partial | 86.250 | 42.865 | 50.3% | 69.011 | 25.872 |
| 50,000 | 8 | 4 | cuda-full | 69.378 | 26.294 | 62.1% | 67.251 | 24.258 |

The pre-scaling pass is small relative to the multiply. In the six CUDA-full
cases it measured between 0.004 ms and 0.069 ms and is included in both M2L and
complete-evaluation time.

## Metadata and scratch

The CUDA executor no longer uploads one level integer per interaction. It
stores source and matrix IDs, one compact active-row record, and one level per
tree node. The old size below is reconstructed from the baseline layout; the
new size is reported directly by the candidate binary.

| N/order/depth | Interactions | Active rows | Old metadata (MiB) | New metadata (MiB) | Change | Scaled scratch (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| 10k/4/3 | 56,448 | 576 | 0.65 | 0.44 | -31.9% | 0.16 |
| 50k/4/4 | 640,472 | 4,671 | 7.35 | 4.98 | -32.3% | 1.25 |
| 50k/6/4 | 640,472 | 4,671 | 7.35 | 4.98 | -32.3% | 3.00 |
| 100k/6/5 | 5,154,372 | 33,719 | 59.13 | 39.98 | -32.4% | 24.00 |
| 20k/8/3 | 56,448 | 576 | 0.65 | 0.44 | -31.9% | 0.74 |
| 50k/8/4 | 640,472 | 4,671 | 7.35 | 4.98 | -32.3% | 5.89 |

Pre-scaled scratch is allocated once during plan construction only when the
required buffer fits both 10% of total VRAM and 25% of currently free VRAM.
Otherwise the same executor uses its deterministic no-scratch target-row
fallback. There are no allocations or global/device synchronisations during
evaluation.

## Profiler interpretation

Nsight Systems captured eleven consecutive N=50k, order-4, depth-4 CUDA-full
evaluations. M2L and P2P ran on distinct streams in all eleven. Median M2L/P2P
overlap was 1.479 ms, or 90.5% of the 1.653 ms median P2P duration. The M2L
kernel used 256-thread blocks and 40 registers per thread. CUDA-full still
transfers only changing moments H2D and final fields D2H; CUDA-partial retains
its multipole/local boundary.

Nsight Compute 2026.1.1 was invoked with `--set full`, but the driver rejected
hardware-counter access with `ERR_NVGPUCTRPERM`. Occupancy, SM/FP64 utilisation,
DRAM/L2 counters, instruction counts, and warp-stall counters are therefore not
reported. Enabling NVIDIA performance counters is an administrator policy
change and was deliberately not performed by this work.

## Reproduction

Build and run one acceptance case:

```console
conda activate cdfmm
module load cuda
cmake --fresh --preset profile-all
cmake --build --preset profile-all --target benchmark_uniform_fmm -j

OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 \
./build-profile-all/benchmarks/benchmark_uniform_fmm \
  --backend cuda-full \
  --sources 50000 --targets 50000 \
  --depth 4 --order 6 --threads 8 \
  --warmups 2 --evaluations 20 --samples 5 \
  --no-workload-comparison --no-direct \
  --output profiling_results/cuda_m2l_optimisation/final_n50000_p6_d4_full.csv
```

Capture the ten-evaluation scheduling trace:

```console
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output=profiling_results/cuda_m2l_optimisation/final_n50000_p4_d4_full \
  ./build-profile-all/benchmarks/benchmark_uniform_fmm \
    --backend cuda-full \
    --sources 50000 --targets 50000 \
    --depth 4 --order 4 --threads 8 \
    --profile
```

After an administrator enables GPU performance counters, collect one M2L
launch with:

```console
ncu --set full \
  --kernel-name-base function \
  --kernel-name regex:apply_scaled_m2l_rows_kernel \
  --launch-count 1 \
  --force-overwrite \
  --export profiling_results/cuda_m2l_optimisation/final_n50000_p6_d4_full \
  ./build-profile-all/benchmarks/benchmark_uniform_fmm \
    --backend cuda-full \
    --sources 50000 --targets 50000 \
    --depth 4 --order 6 --threads 8 \
    --profile
```

## Validation

- C++: 60/60 tests passed with the RTX 5090 available.
- Python: 81/81 tests passed against the CUDA-enabled extension.
- Isolated CUDA M2L: orders 2, 4, 6, and 8; depths 2, 3, and 5; 7,776
  coefficient assertions passed.
- `compute-sanitizer --tool memcheck`: zero errors for isolated M2L and a
  complete CUDA-full repeated-evaluation case.
- Existing complete-FMM tests cover repeated states, separate targets,
  source-point identities, empty geometry, transfer counts, and static uploads.

---

# `docs/pre-refactor-baseline-v0.1.0.md`


This is a compact, deterministic reference record for comparison during the
architecture refactor. It is not a publication benchmark and should not be
used for cross-machine performance claims.

## Measurement identity

- Source commit: `01ad4cd9635d9bdff98a28d9ac0ca6f4869a5c9f`
- Host: Intel Core i9-14900KF, 32 logical CPUs (24 physical cores reported)
- OS/kernel: Linux 7.0.0-31-generic, x86_64
- Compiler: Conda-forge GNU g++ 15.3.0
- Build: Release, C++20, native architecture, OpenMP, CUDA compiled, oneMKL
  compiled (`2026.0.1`)
- CUDA toolkit: 13.3; no CUDA device was available during this run
- Cache policy: disabled for the Python geometry/timing smoke tests; the
  point benchmark used the repository cache and records setup separately
- Python geometry samples used deterministic `numpy.random.default_rng`
  seeds and ten repeated evaluations after one warm-up evaluation.

## Point benchmark

Command shape:

```console
build-release-validation/benchmarks/benchmark_uniform_fmm --profile \
  --sources 512 --targets 512 --order 4 --depth 3 --threads 4 \
  --backend <backend> --precision float32 --output <csv>
```

All rows use seed `314159`, spherical order 4, depth 3 for FMM rows, ten
evaluations, one sample, and 4 OpenMP threads. `evaluation_median` is the
median seconds per evaluation; setup is one-time plan construction.

| Backend | Geometry | Setup (s) | Median/eval (ms) | Evaluations/s | Static plan (bytes) | Nodes | Occupied source leaves | M2L translations | Near-field pairs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cpu-direct` | point dipole, direct all-to-all | 0.00000005 | 0.560416 | 1784.39 | 0 | 0 | 0 | 0 | 261632 |
| `cpu-static-matrix` | point dipole, spherical FMM | 0.033555 | 5.08655 | 196.597 | 3105596 | 585 | 314 | 23640 | 11800 |
| `cpu-static-matrix-mkl` | point dipole, spherical FMM | 0.0225651 | 4.51621 | 221.424 | 8211836 | 585 | 314 | 23640 | 11800 |

The direct row has no tree or static plan. CUDA status from this executable was
`compiled=1`, `available=0`, `direct=0`, `partial=0`, `full=0`; oneMKL was
available.

## Geometry and adaptive smoke samples

These modest-N samples use the Python bindings from the same source commit and
FP64 portable execution. They exercise the persistent dense geometry plan and
the prebuilt adaptive-topology path; they are included to anchor functionality,
not to compare against the point benchmark's workload.

| Case | Deterministic setup | Setup (s) | Median/eval (us) | Readily available memory | Result |
|---|---|---:|---:|---:|---|
| Dense `prism -> prism` | 8 source/target representatives, common `0.12 x 0.10 x 0.08` prism, explicit identity map, 10 evaluations | 0.016919448 | 1.094 | 3072-byte six-tensor plan | finite, including finite self terms |
| Dense `tetrahedron -> tetrahedron` | 8 source/target representatives, common four-vertex tetrahedron, explicit identity map, 10 evaluations | 0.019572416 | 0.535 | 3072-byte six-tensor plan | finite, including finite self terms |
| `AdaptiveTree` + static FMM | 64 clustered points, seed `20260909`, leaf capacity 8, maximum depth 3, Cartesian FP64 order 4, CPU static, explicit identities | tree wall 0.008536769; FMM setup 0.087753865 | 1174.909 | topology 68080 B; static plan 3765338 B | finite field output |

The adaptive sample produced 28 nodes, 23 leaves, reached level 3, 136 M2L
interactions, and 309 P2P leaf records. The topology was passed directly to
the same static FMM plan used by the uniform path.

## Validation caveat at capture time

The rebuilt C++ release-validation binary passed all 175 CTest cases; three
CUDA-runtime tests were skipped because no device was available. During the
initial audit, the Python memory test found a 21,504-byte accounting mismatch:
the notebook estimate was `1,099,986` bytes while the constructed plan reported
`1,121,490` bytes because the retained M2M/L2L maps were omitted from the
estimate. This was corrected during Phase 0 by aligning the estimate with the
plan's actual ownership and disabling cache-dependent assumptions in the
comparison test. The post-correction focused Python geometry and adaptive
checks passed; final full-suite counts belong in the release report.
