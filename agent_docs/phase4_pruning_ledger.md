# Phase 4 pruning ledger (closed)

This is the audit ledger that governed Phase 4 (repository pruning, test
rationalisation, notebook consolidation, documentation reduction, source
readability), moved here from the worktree root when the phase closed on
2026-09-20. Every removal, merge and rename below was carried out as
recorded unless a row says otherwise; the outcome is summarised in
`project_progress.md` ("Phase 4 repository pruning and documentation
cleanup"). Two rows were resolved differently from their plan: the
`test_uniform_fmm.cpp` upward-pass ladder and convergence cases were left as
separate cases after re-reading their bodies (they share a helper, not a
body; a shared `relative_rms_field_error` helper replaced two inline
lambdas), and the procedural expansion exactness test was replaced by a
kernel-level test at every compiled order plus a small plan-level loop rather
than merely shrunk. A third row resolved to its own fallback: section 2's
classifier row said "M only if the merge can be made byte-identical ...;
otherwise D with a note". The merge was byte-identical on the persisted
plans, but it broke the GCC 13 LTO link that CI uses, so the row ends as
**D**, with the note in `project_progress.md` and the warning in `AGENTS.md`.

Worktree-local audit ledger for Phase 4 (repository pruning, test
rationalisation, notebook consolidation, documentation reduction, source
readability). Starting HEAD `ad48459` on `refactor/architecture-v0.2`; work
branch `phase4-pruning`.

Every entry records: path/symbol, current responsibility, evidence of
overlap or deadness, replacement/canonical owner, action, reason, validation
required. Nothing is removed before its row exists here.

Legend: **K** keep, **M** merge/parameterise, **R** remove, **D** defer.

## 0. Preserved without discussion (Article1 boundary)

| Item | Action | Reason |
|---|---|---|
| Portable CPU, oneMKL, `CudaPartial`, `CudaFull`, dense direct CPU/oneMKL/CUDA | K | benchmarkable alternatives for Article1 |
| `P2PExecutionPacking::{CanonicalAos, ParticleRowSoa, TensorDictionary, CudaBsr3, LeafBlock, PointGeometry}` + signed dictionary + `use_reduced_symmetry_p2p` + `cuda_dictionary_*` flags | K | explicit forceable representations |
| `PointExpansionExecution::{Precomputed, Procedural}` | K | both point P2M/L2P representations |
| FP32 and FP64 plans everywhere | K | precision paths |
| Spherical and Cartesian bases | K | both formulations |
| `benchmarks/*` drivers, runners, analysers, `baselines/phase3d/` | K | Article1 reproduction; only the stale `benchmarks/AGENTS.md` tree is corrected |
| All flat headers under `include/cdfmm/`, `src/operators.cpp`, `DenseDirectPlan::evaluate()` dispatch | K | installed public compatibility façades / approved exceptions |
| `agent_docs/*` | K (current-state summaries corrected, Phase-4 record added) | provenance |

## 1. Public documentation (4D)

Current: 32 files in `docs/` (~350 KB), flat toctree of 28 pages mixing user
guide, maths, current architecture, refactor history, benchmark history and
roadmap. Target: 12 user/developer pages + 3 mathematical pages + API
reference. Every unique mathematical statement is kept; development history
moves to `agent_docs/`.

| Current file | Responsibility today | Overlap / status | Action | New owner |
|---|---|---|---|---|
| `index.md` | toctree only | duplicates `overview.md` intro | M | `index.md` becomes overview + "where to start" + structured toctree |
| `overview.md` | purpose, capabilities, boundary, architecture pointer | overlaps `index.md`, `fmm-overview.md`, `roadmap.md` | M | `index.md` (capabilities/boundary), `architecture.md` (pointer) |
| `getting-started.md` | C++/Python direct + static FMM snippets | thin; no lifecycle explanation | K (expand) | `getting-started.md`: install pointer, first solve, static lifecycle, repeated evaluate, units/shapes, link to tutorials |
| `installation.md` | presets, env, troubleshooting | current; FMM3D/MagTense subsections describe comparison envs | K (trim external-comparison subsections to a note) | `installation.md` |
| `examples.md` | catalogue of C++ demos + old notebook tables | notebook tables will be stale | M (rewrite) | `examples.md`: C++ demos, Python script, canonical tutorial table |
| `fortran-interface.md` | C ABI + Fortran wrapper use | current, unique | K (rename to `c-and-fortran.md`) | `c-and-fortran.md` |
| `geometry-models.md` | source/target geometry, models, prism/tetra records | unique user content; finite semantics scattered in `math.md`, `spherical-expansions.md` (comparison flags), `fmm-overview.md` | M | `geometry.md`: records, nine pairs, four model selectors, finite averaging, self semantics, comparison flags, moments `m = V M` |
| `backends.md` | operator placement, CUDA/CPU policy, point expansion execution, data flows | current, authoritative; capability matrix lives in `static-p2p.md` | K (fold in P2P capability matrix + explicit-selection caveat, precision summary pointer) | `backends.md` |
| `precision.md` | FP32/FP64 boundary table, root-width scaling, dense-plan precision | unique; root-width scaling is mathematical | M | user table -> `backends.md` "Precision"; root-width scaling -> `math/conventions.md` |
| `caching.md` | cache layout, keys, precompute tool | unique | M | `caching-and-periodicity.md` |
| `periodic-boundaries.md` | periodic API, wrapped topology, Ewald root periodiser, Python setup | unique (Ewald formula is the one normative statement) | M | `caching-and-periodicity.md` (keeps the Ewald formula; `math/conventions.md` links to it) |
| `parameter-selection.md` | advisers | unique | M | `trees-and-parameter-selection.md` |
| `uniform-tree.md` | Morton, flat ordering, permutations, list1/list2 | reference content; overlaps `static-architecture.md` | M | `trees-and-parameter-selection.md` (tree semantics, adaptive tree, depth/order/layout hints, diagnostics) |
| `fmm-overview.md` | traversal steps, two bases, geometry boundary | overlaps `static-architecture.md`, `operators.md`, `getting-started.md` | M | evaluation steps -> `architecture.md`; boundary -> `geometry.md`/`index.md` |
| `static-architecture.md` | static-plan data ownership and flow | overlaps `architecture.md` | M | `architecture.md` "Static plan lifecycle" |
| `architecture.md` (85 KB) | layers, rules, taxonomy, P2P invariant, plus Phase 1/2 closure narratives, per-cleanup audits, validation matrices | ~70 % history | M (rewrite concise) | `architecture.md` (~15 KB): layers, ownership, lifecycle, P2P invariant, precomputed/procedural, compatibility surface rules, validation map. History verbatim -> `agent_docs/architecture_history.md` |
| `static-p2p.md` | invariant, capability matrix, dictionary packing, Phase-3 sweep history, reproduction | mixed | M | matrix + explicit selection -> `backends.md`; packing description -> `architecture.md`; reproduction commands -> `benchmarks.md`; sweep history -> `agent_docs/architecture_history.md` |
| `benchmarks.md` (41 KB) | tools map, measurement rules, historical notebook results, driver docs, CSV schema | ~35 % history | M (trim) | `benchmarks.md`: how to run, drivers/runners, timing categories, CSV fields, forced-option controls, baselines; history -> `agent_docs/architecture_history.md` |
| `profiling.md` | perf/Nsight workflow | unique, current | M | `benchmarks.md` "Profiling" section |
| `validation.md` | what tests cover, CUDA manual validation | developer content; overlaps `tests/AGENTS.md` | M | `architecture.md` "Validation" (short) |
| `cuda-m2l-performance.md` | 2026-08-17 M2L tuning report | history; decision now in `backends.md` | R (archive) | `agent_docs/architecture_history.md` |
| `pre-refactor-baseline-v0.1.0.md` | v0.1.0 baseline numbers | history | R (archive) | `agent_docs/architecture_history.md` |
| `roadmap.md` | implemented checklist + future work | checklist duplicates capabilities; "later work" is the useful part | M | `index.md` "Capability boundary and future work" |
| `math.md` | kernel, SH convention, cuboid/tetra maths, Cartesian operators, static maps, root normalisation, M2L reuse | normative; finite geometry section is large | M (split) | `math/conventions.md` + `math/finite-geometry.md` |
| `spherical-expansions.md` | SH basis, static construction, backends/limits | normative + user | M | maths -> `math/spherical-expansions.md`; comparison flags -> `geometry.md` |
| `cartesian-expansions.md` | multi-index ordering, factorial normalisation | normative | M | `math/conventions.md` |
| `laplace-derivatives.md` | Taylor-jet derivative generation | normative | M | `math/conventions.md` |
| `operators.md` | per-operator I/O reference | duplicates `math.md` definitions | M | `math/conventions.md` (operator definitions) + `getting-started.md` (flat operator API mention) |
| `api.rst`, `conf.py`, `Doxyfile`, `requirements.txt`, `AGENTS.md` | build | current | K (update AGENTS.md structure) | — |

Target hierarchy:

```text
docs/
  index.md                          overview, capability boundary, where to start
  installation.md
  getting-started.md
  geometry.md
  backends.md                       backends, execution options, precision, P2P matrix
  caching-and-periodicity.md
  trees-and-parameter-selection.md
  examples.md                       C++ demos + canonical notebook tutorials
  c-and-fortran.md
  benchmarks.md                     running benchmarks, timing categories, profiling
  architecture.md                   concise current architecture + validation map
  api.rst
  math/conventions.md               kernel, signs, bases, operators, static maps, scaling
  math/spherical-expansions.md
  math/finite-geometry.md           prism, tetrahedron, averaging, self, safeguards
```

Validation: `sphinx-build -W --keep-going -b html docs docs/_build/html`
with zero warnings; grep for stale links to removed files across the repo.

## 2. Source (4A)

Reference evidence from a read-only audit (grep over `src/ include/ tests/
python/ fortran/ benchmarks/ docs/`), re-verified by the lead before each edit.

| Symbol | Where | Evidence | Action | Validation |
|---|---|---|---|---|
| `UniformFmm::use_cuboid_p2m_`, `use_cuboid_l2p_` | `include/cdfmm/uniform_fmm.hpp:601-602`; assigned `src/fmm/execution_setup.cpp:679,695,703,755,763` | 2 declarations + 5 writes, zero reads anywhere; not part of any installed binary layout (`UniformFmm` is behind `unique_ptr` in the opaque `cdfmm_plan`, `cdfmm_core` is never installed) | R | dev build + ctest; `sizeof(UniformFmm)` recorded before/after (7272 -> new) |
| `CudaExecutionPolicyInputs::periodic`, `bsr_estimate_bytes`, `bsr_budget_bytes` | `src/backend/cuda/execution_policy.hpp:43,65,66`; written `src/fmm/execution_setup.cpp:445,469-474` | no rule in `execution_policy.cpp:127-275` reads them; `diagnostics.cpp:244-264` never prints them | R (+ fix the `docs/backends.md` sentence that says they are carried for diagnostics) | dev + cuda builds, `test_cuda_backend` policy cases |
| 3-arg `far_field_stream_priority(size_t,size_t,int)` | `execution_policy.hpp:148-150`, `.cpp:84-94` | no production caller; called by `tests/test_cuda_backend.cpp:1015,1017`; carries the rule's prose | R: move the prose to the 5-arg overload; port the test to the 5-arg form with explicit packing/precision | `test_cuda_backend` "CUDA execution policy resolves..." on the cuda tree |
| `classify_endpoint_operators` (`plan_preparation.cpp:135-172`) vs `classify_exact_operators` (`exact_operator_reuse.hpp:152-191`) | callers `:555,:1028` vs `p2p.cpp:305,510,731`, `dense.cpp:478` | same algorithm shape (first-seen numbering, 32-bit guard, sample gate) but different key type (vector vs fixed array), gate (4096 vs 65536 samples) and `max_classes` | M only if the merge can be made byte-identical (generic key type + explicit gate at the endpoint call sites); otherwise D with a note | byte-compare persisted geometry plans before/after (`test_p2p_exact_reuse`, plus a cache-file `sha256sum` probe over prism/tetra exact far-field models) |
| `cuda_policy::resolve_cuda_execution_policy` also decides the CPU dictionary | `execution_setup.cpp:396,421,428` | naming only; `cuda_policy.*` strings are printed only for CUDA backends and no test/python reads them | K; explain the ownership in a comment | — |
| everything else under `src/` | — | sampled sweep found no further unreferenced symbol; FP32/FP64 overload pairs in `p2p/plan.cu` are not duplicates | K | — |

## 3. Tests (4B)

`tests/AGENTS.md` still says "do not prune or consolidate them in the
architecture steps"; Phase 4 is the explicitly authorised pruning step, so the
sentence is replaced by the tier description below.

Findings: the C++ suite is already layered deliberately. The identity/finite-
self contract is pinned once per implementation layer (canonical P2P reuse,
dense reuse, cuboid, tetrahedron static, `UniformFmm`, CUDA packings) and each
protects a different builder or executor; the eight header/stub files must stay
one-per-translation-unit because per-header self-containment *is* the
contract. What is redundant is inside `test_procedural_point_expansion.cpp`
(two scenes of 2000 points for an exactness check) and the seven-case
upward-pass ladder plus three convergence cases in `test_uniform_fmm.cpp`.

| Test / block | Category | Overlap | Action | Justification |
|---|---|---|---|---|
| `test_procedural_point_expansion.cpp:148` "procedural point P2M and L2P reproduce the precomputed rows" (~1102 s) | 3 | none | K, shrink `make_scene(2000)`; keep backend x precision x order {1,3,6,10} loops | bitwise reproduction, not convergence: N buys no coverage |
| `:202` "procedural point L2P reproduces the precomputed potential" | 3 | subset of :148 | M into :148 as an `OutputFlags::Both` branch | removes a second 2000-point scene |
| `:229` "procedural point expansion requests are validated" (~364 s) | 6 | none | K, shrink scene after timing it | validation paths, not accuracy |
| `:287` cold/warm cache agreement | 7 | none | K, shrink N | exactness |
| `test_uniform_fmm.cpp` upward-pass ladder (7 cases, 1090-1246) | 5 | shared harness | M into one TEST_CASE with SECTIONs, one per concern, after re-reading each body | same pattern as the geometry matrix |
| `test_uniform_fmm.cpp` "converges to exact dense direct" (cuboid/tetra/generic) | 5 | shared harness | M into one geometry-parameterised case if bodies are parallel | — |
| `test_p2p_geometry_matrix.cpp` (5 cases) | 3/4 | none — each varies one axis | K unchanged | the 9-pair x backend x precision x packing contract |
| `test_p2p_exact_reuse.cpp`, `test_dense_direct_exact_reuse.cpp` | 3 | analogues for two builders | K | different builders (Phase 3C vs 3C.5) |
| `test_cuda_backend.cpp` (20 cases) | 4/5 | overlaps are by layer or by defect | K; add cross-reference comments | includes regression tests for named defects |
| `test_precision.cpp`, `test_cuboid.cpp`, `test_rectangular_prism_magtense.cpp`, `test_tetrahedron*.cpp`, `test_static_m2l.cpp` | 1/2/3 | trust anchors | K | ground truth for the finite pairs |
| header/stub tests (8 files) | 9/4 | none | K | per-TU self-containment |
| Python: notebook-contract tests for removed notebooks (`test_fmm_memory.py:100`, `test_fmm3d_comparison.py:143`, `test_periodic_fmm.py:53`, `test_parameter_selection.py:76`, `test_precision_notebook.py`, `test_spherical_fmm.py:74,100`, `test_adaptive_notebook.py:42`) | 9 | tied to removed notebooks | R, replaced by `test_tutorial_notebooks.py` executing every retained tutorial | no regression tests for removed notebooks |
| Python: `fmm_memory.py` helper + 5 numerical tests | 5 | none | K as a test support module (`python_tests/memory_model.py`) | protects plan byte accounting |
| Python: `adaptive_showcase.py` + 7 numerical tests | 5 | none | K (helper moves with the tutorial) | substantive adaptive-tree tests |
| Python: FMM3D helper unit tests, installer tests | 8 | none | K (paths updated to `benchmarks/external/fmm3d/`) | Article1 utilities |
| Python: MagTense notebook/env/preset tests | 6/9 | none | K (paths updated to `examples/validation/`) | external validation retained |
| Python: benchmark-runner tests | 8 | none | K | tooling |

Tiers (documented in `tests/AGENTS.md`): fast portable CI (everything not
gated); optional CUDA/oneMKL cases (skip or `SUCCEED()` when the build lacks
them); benchmark/regression-tool tests under `python_tests/`.

## 4. Notebooks and examples (4C)

Before: 29 notebooks in three directories (16 numbered, 12 `simple_*`, 1
matrix), README lists 22 of them, three have no Markdown at all, one calls the
removed `cdfmm.CuboidSize`, one never imports `cdfmm`.

Target: `examples/tutorials/01..06_*.ipynb` (six canonical tutorials, CPU-
executable, CUDA/oneMKL sections guarded), `examples/validation/` (two
MagTense comparison notebooks, external environment), and Article1 raw
material under `benchmarks/external/fmm3d/`.

| Old notebook | Role | Action | Feeds |
|---|---|---|---|
| `notebooks/00`-`07` (P2P, P2M, M2M, M2P, M2L, L2L, L2P, chain) | teaching | M | `06_operator_chain` |
| `notebooks/08_uniform_tree` | teaching | M | `05_trees_and_parameters` |
| `notebooks/09`, `10` (upward/downward pass) | teaching | M | `01_getting_started` (lifecycle, decomposition) |
| `notebooks/11_fmm3d_comparison` + `fmm3d_comparison.py` + `install_fmm3d.sh` | external comparison (exploratory) | move | `benchmarks/external/fmm3d/` (not a tutorial) |
| `notebooks/12_cuda_memory_usage` + `fmm_memory.py` | memory investigation | R; helper -> `python_tests/memory_model.py` | statistics section of `03_backends` |
| `notebooks/13_parameter_selection` | focused | M | `05_trees_and_parameters` |
| `notebooks/14_periodic_fmm_direct_compare` | focused | M | `04_cache_and_periodic` |
| `notebooks/15_adaptive_tree` + `adaptive_showcase.py` | teaching (advanced) | M (helper moves) | `05_trees_and_parameters` |
| `simple_fmm_single_update`, `simple_fmm_multiple_update` | getting started (no Markdown) | M | `01_getting_started` |
| `simple_fmm_fmm3d_compare` | duplicate of 11, unguarded import | R | — |
| `simple_cuboid_fmm_direct_compare`, `simple_cuboid_target_fmm_direct_compare`, `simple_cuboid_p2m_l2p_direct_compare` | focused finite geometry | M | `02_finite_geometry` |
| `simple_cuboid_magtense_compare`, `simple_geometry_magtense_compare` | external validation (MagTense env) | move | `examples/validation/` |
| `simple_dense_direct_precision_compare`, `simple_cartesian_spherical_fmm_compare`, `ReducedSymmetryP2P` (fix `CuboidSize`) | backends/precision/basis/packing | M | `03_backends_and_execution` |
| `simple_persistent_cache_reuse` | cache | M | `04_cache_and_periodic` |
| `matrix_notebooks/quadtree_near_interaction_heatmap` | no `cdfmm` | R | — |
| `examples/plot_uniform_tree.py` | duplicate of tree tutorial, untested | R | `05_trees_and_parameters` |
| `example_utils.py` | plotting helpers | K (moves to `examples/tutorials/`) | all |
| `single_box_demo.cpp`, `operator_convergence_demo.cpp`, `fortran/magtense_style_demag.f90` | C++/Fortran examples, built by presets | K | — |

Regression: `python_tests/test_tutorial_notebooks.py` executes each tutorial
with `nbclient` on the portable build (kernel `python3`, per-notebook timeout),
so CI runs the retained tutorials rather than parsing old ones. `[test]`
extras gain `nbclient` and `ipykernel`.
