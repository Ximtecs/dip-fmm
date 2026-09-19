# Phase-3D engineering regression baseline

**This is an engineering regression baseline, NOT an Article1 publication
benchmark.**

Do not copy these numbers into the FMM article, do not build publication
figures from them, and do not compare them with jaxFMM, FMM3D or any other
external framework. The publication campaign runs after Phase-4 pruning
against a frozen implementation and uses its own measurement protocol. These
rows exist so that a future change which breaks construction, warm loading,
evaluation or memory is noticed.

## What was measured

| File | Driver | Content |
|---|---|---|
| `phase3d_regression.csv` | `benchmarks/run_phase3d_regression.py --suite all` | 158 cases: the workload classes under the automatic policy, and the automatic policy against forced alternatives. Repeated evaluation and its phases, the resolved representation, retained host and device bytes. |
| `phase3d_construction.csv` | `benchmarks/run_construction_matrix.py --preset baseline` | Cold plan construction with the geometry cache disabled for every row, decomposed per stage. |
| `phase3d_startup.csv` | `benchmarks/run_phase3d_startup.py` | One point and two finite geometries built three ways: cache disabled, cold with a cache write, and a warm hit. |

Summarise with:

```bash
python benchmarks/analyse_phase3d_regression.py \
    --input benchmarks/baselines/phase3d/phase3d_regression.csv --view both
```

## How the driver behaves

These rules exist because each one was a review finding against the driver
that produced this baseline.

- **A failed case fails the run.** An invocation that cannot run, or that
  produces no data row, raises, names the case and exits non-zero. The driver
  also checks the number of rows against the number of cases the selected
  suite and backend matrix define. A baseline that silently drops rows cannot
  be told apart from a complete one, which is why `--allow-failures` -- the
  opt-in that restores skipping -- is for exploration only and must never
  produce a retained baseline.
- **A run is fresh by default.** `--resume` reuses the per-case CSVs of an
  interrupted run, but only when `session.json` in the scratch directory
  matches: the revision, the binary's path and SHA-256 (with size and mtime
  recorded beside it), the suite and filter, the evaluation, warm-up, sample
  and thread counts, backend availability, and the column set. Any mismatch
  is refused and the differing field is printed. Reuse used to depend on a
  file merely existing, which quietly mixed commits and binaries into
  something that looked like one session.
- **Comparison groups are data, not parsed names.** Every row records
  `comparison_group` -- everything two rows must share before a ratio between
  them means anything -- and `comparison_variant`, the single axis under
  review. The analyser groups on those columns. The retained CSVs below
  predate them, so the analyser reconstructs their groups from the driver's
  own case generators; a case neither source recognises is printed alone
  rather than compared against a guess.

## Regenerating it safely

The retained CSVs are kept as they were measured. Their case names, their
order and every number in them are unchanged by the closure pass, so
regeneration was not needed and was not done. If a schema change ever forces
it:

```bash
OMP_NUM_THREADS=8 OMP_PROC_BIND=close \
OMP_PLACES='{0},{2},{4},{6},{8},{10},{12},{14}' \
python benchmarks/run_phase3d_regression.py \
    --binary build-bench-all/benchmarks/benchmark_uniform_fmm \
    --output benchmarks/baselines/phase3d/phase3d_regression.csv \
    --suite all
```

Rerun only this internal matrix, from one `benchmark-all` tree, on an
otherwise idle machine. Do not widen it: expanding the workload set turns it
into a publication campaign, which is a separate exercise against a frozen
implementation after Phase-4 pruning.

## Conditions

Intel i9-14900KF with 8 P-cores pinned (`OMP_NUM_THREADS=8`,
`OMP_PLACES={0},{2},{4},{6},{8},{10},{12},{14}`, `OMP_PROC_BIND=close`);
RTX 5090, driver 595.84, CUDA 13.2 runtime, nvcc 13.3.73; g++ 15.3.0 as C++
and CUDA host compiler; oneMKL 2026.1. One `benchmark-all` tree, so every
comparison is same-binary and same-session. Evaluation rows are medians of 5
samples of 20 evaluations after 3 warm-ups.

## Reading them safely

- The comparison number for a policy decision is **repeated evaluation**, and
  for a rule that governs one stage it is **that stage**: a P2M/L2P rule is
  invisible in an M2L-dominated total. The FP64 `CudaFull` expansion rows
  differ by 2 % end to end and by 1.39x on P2M+L2P.
- `fmm_setup_seconds` in `phase3d_regression.csv` is a **warm** number: that
  driver leaves the caches enabled, so the first row of a given order and
  precision pays the universal operator build and caches it for every later
  row. Use `phase3d_construction.csv` for cold construction.
- Absolute times depend on the machine. A regression check should compare
  ratios within one session, not these numbers directly.

## Headline expectations at this HEAD

Automatic policy, all confirmed here:

| Situation | Resolved |
|---|---|
| point P2P, CPU, General | `point-geometry` |
| point P2P, CPU, `RegularGrid` | `tensor-dictionary` |
| point P2P, CUDA FP32 (any layout) | `point-geometry` |
| point P2P, CUDA FP64, General | `leaf-block` |
| point P2P, CUDA FP64, `RegularGrid` | `tensor-dictionary` |
| finite P2P, CPU / CUDA, General | `particle-row-soa` / `leaf-block` |
| finite P2P, `RegularGrid` | `tensor-dictionary` |
| point P2M/L2P, CPU and `CudaFull` FP32 | procedural |
| point P2M/L2P, `CudaFull` FP64 | precomputed |
| finite P2M/L2P | precomputed everywhere |
| CPU M2L | portable (oneMKL is explicit-only) |
| `ExecutionBackend::Auto` | `CpuStatic` |

Derived dictionaries: 248 distinct prism and 187 tetrahedron tensors at FP32
on a 4096-body lattice (one-byte tokens); 172 at 8 points per leaf and 1688 at
64 per leaf on a 32768-point lattice. Irregular geometry builds none.

The full analysis, including every rejected change and the reason it was
rejected, is in `agent_docs/performance_optimization.md`, "Final cross-backend
integration and production policy (Phase 3D)".
