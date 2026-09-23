# Benchmarks and profiling

Benchmarking answers how fast a deterministic workload is; profiling explains
why. This page says how to build and run the drivers, what the reported
quantities mean, and how to read them. The measured record behind the
production policies is `agent_docs/performance_optimization.md`; the retained
engineering baseline is `benchmarks/baselines/phase3d/`. None of these are
publication results.

## Rules for every measurement

- Keep setup, one-time packing/upload, repeated evaluation and transfer
  timing distinct; static-plan reuse is part of the measured contract.
- Reuse one fixed geometry for changing moments, warm the runtime libraries
  before timing, and report accuracy beside runtime.
- Finite bodies default to point far-field models so that a backend or packing
  comparison isolates the near field; `--far-field-model exact` exercises the
  finite P2M/L2P operators.
- Record precision, geometry, expansion, backend, packing, thread/device and
  problem size; CUDA timings and crossover points are hardware specific.
- A construction measurement disables the geometry cache (every row is a cold
  build); a cache measurement says so explicitly.
- Never sum overlapping CUDA stream durations as a critical path.

## Building

```console
conda activate cdfmm
cmake --fresh --preset benchmark          # portable CPU, icpx, OpenMP, LTO
cmake --fresh --preset benchmark-mkl      # plus oneMKL
cmake --fresh --preset benchmark-all      # portable CPU, oneMKL and CUDA in one executable
cmake --fresh --preset profile-all        # benchmark-all with NVTX ranges and symbols
cmake --build --preset <preset> -j
./build-bench-all/benchmarks/benchmark_uniform_fmm --cuda-status
```

Release builds enable LTO and native CPU instruction selection by default
(`CDFMM_ENABLE_NATIVE_ARCH=OFF` for a portable binary). Fast math is off and
is not part of the canonical configuration; `CDFMM_EXTRA_COMPILE_OPTIONS`
appends C++ options only. Record the CMake summary, compiler version,
backend, CUDA device and thread environment with every result.

## Drivers and runners

| Question | Tool |
|---|---|
| direct CPU versus direct CUDA; portable CPU, oneMKL, CUDA partial and CUDA full FMM | `benchmark_uniform_fmm` and `run_benchmarks.py` profiles |
| isolated near-field packings on CPU, oneMKL and CUDA | `benchmark_p2p`, `run_p2p_sweep.py` |
| geometry × backend × packing, periodic cells, CudaPartial/CudaFull crossover | `run_p2p_packing_matrix.py` |
| precomputed versus procedural exact operators | `benchmark_operator_representation`, `run_operator_representation.py`, `analyse_operator_representation.py` |
| cold static-plan construction by phase | `run_construction_matrix.py` |
| dense all-to-all construction and evaluation | `benchmark_dense_direct_construction`, `run_dense_construction_matrix.py` |
| cold versus warm cache setup | `benchmark_cache_initialisation`, `run_phase3d_startup.py` |
| internal cross-backend regression and policy matrix | `run_phase3d_regression.py`, `analyse_phase3d_regression.py` |
| dense-leaf CUDA occupancy study | `run_high_occupancy_p2p.py` |
| external FMM3D comparison material | `benchmarks/external/fmm3d/` |

### `benchmark_uniform_fmm`

The main driver builds one plan and evaluates changing moments on one backend
(`--backend cpu-direct | cuda-direct | cpu-static-matrix | cpu-static-matrix-mkl
| cuda-partial | cuda-full`; the older `cuda-m2l-p2p`, `cuda-m2l` and
`cuda-m2l-static-p2p` spellings alias `cuda-partial`). It is an all-to-all
source-point comparison: sources and targets are the same particles with an
explicit identity map, so `--sources` and `--targets` must be equal.

```console
./build-bench-all/benchmarks/benchmark_uniform_fmm --backend cuda-full \
  --sources 32768 --targets 32768 --depth 4 --order 6 --regular-grid \
  --source-geometry tetrahedron --target-geometry tetrahedron \
  --p2p-packing tensor-dictionary --precision float32 \
  --no-direct --no-workload-comparison --accuracy-targets 0
```

| Option | Meaning |
|---|---|
| `--source-geometry`, `--target-geometry` | `point`, `prism` or `tetrahedron`; finite bodies are centred on the particles with extent `--body-fill` (default 0.9) of the nominal spacing, one common record by default and one per body with `--irregular-bodies` |
| `--far-field-model` | `point` (default) or `exact` for the finite P2M/L2P operators |
| `--p2p-packing` | `canonical-aos`, `particle-row-soa`, `tensor-dictionary`, `leaf-block`, `cuda-bsr3`, `point-geometry` or `auto`; impossible combinations fail at construction with the reason |
| `--point-expansion` | `precomputed`, `procedural` or `auto` |
| `--regular-grid`, `--periodic` | the layout hint; a fully periodic cubic cell equal to the root box |
| `--precision`, `--order`, `--depth`, `--threads` | plan parameters and OpenMP threads |
| `--warmups`, `--evaluations`, `--samples` | measurement policy; `--profile` sets one warm-up, one sample and ten consecutive evaluations for profiler timelines |
| `--direct`, `--accuracy-targets N` | exact-reference validation on all or on `N` deterministically spaced targets; point bodies are compared with the point-dipole sum, finite bodies with the exact FP64 all-pairs sum of their pair tensors (`DenseDirectPlan`, built in blocks), so the error is the FMM's and not the geometry-model difference |
| `CDFMM_CACHE_DIR`, `CDFMM_DISABLE_CACHE` | cache control for the timed construction only; the untimed CUDA runtime warm-up and the workload-comparison constructions never read or write the cache, so an empty cache directory gives a cold `fmm_setup_seconds` on every backend |

Every automated row records two end-to-end workloads (one construction plus
one evaluation, and one construction plus ten evaluations with changing
moments) beside the warmed timed evaluations, and reads every CSV column by
name so added columns never migrate a reader.

### `run_benchmarks.py`

```console
python benchmarks/run_benchmarks.py --profile rough --max-threads 8 \
  --executable build-bench-all/benchmarks/benchmark_uniform_fmm
```

Profiles are `rough` (20 processes: four FMM backends at 20 000 and 30 000
particles, depths three and four, order four, plus CPU and CUDA direct; fails
unless all six backends are available), `quick`, `standard` and `full`. The
runner probes the executable's capabilities, sets `OMP_NUM_THREADS` per case
and `MKL_NUM_THREADS=1`, runs a `parameter_grid` suite over particle count,
order and depth, a `direct` suite (one row per particle count and direct
backend), a `scaling` suite over threads at the largest size, and a
`comparison` suite at one common geometry, and writes a timestamped directory
under `benchmark_results/` with the CSV, machine metadata, a summary and PNG
figures grouped by result type. Existing runs are never overwritten.

### Construction, dense and cache drivers

`run_construction_matrix.py --binary <benchmark_uniform_fmm> --output <csv>
--preset {quick,baseline,full}` sweeps geometry (random and lattice points,
regular and irregular prisms and tetrahedra), size, order, precision and
backend with the cache disabled, records the per-phase `StaticPlanStatistics`
timings, plan bytes, peak resident set and an evaluation median, and resumes
an interrupted CSV. The near-field stage is decomposed into
`p2p_interaction_setup`, `p2p_canonical_operator` and `p2p_derived_packing`;
`precision_conversion` times the FP64-to-FP32 conversion; a warm cache hit
still charges the last two, which is how a warm plan is distinguished from a
rebuilt one.

`benchmark_dense_direct_construction` and `run_dense_construction_matrix.py
--preset {quick,geometry-matrix,sizes,asymmetric,threads,anisotropy}` measure
the exact dense plan's setup (validation, allocation, prepared finite geometry,
exact classification, tensor build and materialisation, and on CUDA host
build, context, allocation and upload) and its repeated evaluation
separately, one process per row so that a row's peak resident set is its own.
`precision_conversion_seconds` is always zero and is kept to say so: FP32
dense plans quantise at the point of store. `--probe-redundancy` counts
distinct exact operator inputs and `--checksum` hashes the matrices so that a
construction change is held to bitwise equality. The workloads (`lattice`,
`lattice-irregular`, `random`, `refined`, `anisotropic`,
`refined-anisotropic`) bracket exact reuse from thirty- to eightyfold down to
none; anisotropy matters because a spacing that is not exactly representable
makes equal index differences round to different displacement bits.

`benchmark_cache_initialisation --depth 3 --grid 8 --backend onemkl
--cache-dir "$(mktemp -d)"` constructs one `p = 6` FP32 prism plan twice into
an initially empty directory and emits `CACHE_BENCH` rows for cold and warm
setup (normalisation, tree, lookup/hash/load, analytical operators, backend
packing, CUDA upload, byte counts); backends are `portable`, `onemkl` and
`cuda-full`. `run_phase3d_startup.py` splits startup into `cold-no-cache`,
`cold-write` and `warm-hit` with per-stage timings.

### Operator representation study

`benchmark_operator_representation` compares, for one operator family, the
production precomputed representation (built with the production builders,
packed into the production layouts and applied with the production apply
functions) with a procedural representation that retains only positions,
body records, topology and the hoisted per-body invariants and calls the
same builder during every update. `--stage p2p|p2m|l2p`, `--source`,
`--target`, `--orders`, `--layout`, `--separation` (`self`, `adjacent`,
`list1`, `small-far`, `far-safeguard`), `--mode hot` (one small set,
cache-resident, single-threaded) or `--mode streaming` (the complete list-1
neighbourhood of a lattice with every thread), `--cuda`, and
`--procedural-budget-seconds` (default 60, recorded as `measured_fraction`).
Both representations are checked against the FP64 canonical operator on the
final moments and every result is checksummed. `run_operator_representation.py`
runs the resumable matrix; `analyse_operator_representation.py` reports the
procedural/precomputed ratio, the retained bytes saved and the amortisation
break-even `K = (T_build - T_procedural_setup) / (T_procedural - T_apply)` in
complete field updates.

### Regression matrix and baseline

`run_phase3d_regression.py --binary <benchmark_uniform_fmm> --output <csv>
--suite {regression,policy,all}` runs the workload classes (random points,
point lattices at low and high occupancy, regular prism and tetrahedron
lattices with and without the layout hint, an irregular finite control) under
the automatic policy on every available backend, and the automatic policy
against its credible forced alternatives. Each row records the resolved
representation (`p2p_packing`, `p2m_execution`, `l2p_execution`,
`static_multiply_backend`), the repeated evaluation and its phases, the
retained host and device bytes, and `comparison_group`/`comparison_variant`
so that the analyser (`analyse_phase3d_regression.py --view
{policy,matrix,both}`) never guesses a group from a display name. A run is
fresh and strict by default: a case that cannot run fails the run and the row
count is checked against the defined matrix; `--allow-failures` only moves
where it stops and must not produce a retained baseline; `--resume` accepts
only a scratch manifest with matching revision, binary SHA-256, suite,
sampling and backend availability. Two rules for reading it: judge a rule
that governs one stage on **that stage**, not the total; and its
`fmm_setup_seconds` is a warm number, since this driver leaves the caches
enabled.

`benchmarks/baselines/phase3d/` retains the accepted output of the three
drivers with a README stating what it is: an engineering regression baseline,
explicitly not an Article1 benchmark.

## Timing levels

Internal timing is opt-in. `UniformFmmOptions::timing_level` (Python
`options.timing_level`, a `TimingLevel` enum) selects how much the solver
measures, and `UniformFmm::set_timing_level` changes it later on the same
plan; results, the resolved backend and packing, cache keys and cache contents
are the same at every level.

| Level | Collected | Cost |
|---|---|---|
| `Off` (default) | Nothing. Only the functional CUDA events remain (two cross-stream dependencies and the completion point, created without timestamps). Every `PhaseTiming` keeps its default and the records say `timing_level == Off`. | The production path and the Article1 configuration; indistinguishable from a build with no instrumentation. |
| `Coarse` | Whole host wall times: `total`, and on the CPU hierarchy `far_field` and `p2p` (`cuda_p2p_wait` for the hybrid backend); at construction `total_setup` and the static-plan `total`. No device event is recorded. | A handful of `steady_clock` reads per evaluation. |
| `Detailed` | Every phase: the host phases, the oneMKL gather/multiply/scatter split, the CUDA device lanes (`CudaEvaluationTimings`, up to sixteen event records and twelve elapsed-time queries per full-plan evaluation) and the construction subphases including cache lookup, load and write, and the breakdown of the two uniform trees the plan builds (`UniformTree::build_timings`, copied into `tree_construction`). | Measured 3–15 % of a sub-millisecond CUDA evaluation, at noise level on the CPU (`agent_docs/performance_optimization.md`, Phase 5). |

What `Off` covers is exactly the code a `UniformFmm` owns: geometry
normalisation, the two uniform trees it constructs, topology, every operator
and static-plan build, cache lookup and load, backend setup and upload, and
every evaluation. The exact dense baselines take the same switch as a
constructor argument (`DenseDirectPlan(..., timing_level)` and
`CudaDenseDirectPlan(..., timing_level)`, default `Off`), which gates their
internal construction records the same way; the Python dense plans are always
`Off`. Two standalone tree objects are outside a plan and keep their own
behaviour: a `UniformTree` you construct yourself collects
`build_timings()` unless `UniformTreeOptions::collect_build_timings` is
false (a plan sets it from its level), and `AdaptiveTree` always records its
two coarse numbers (`tree_seconds`, `interaction_seconds`; three clock reads
per build, measured below 0.1 µs). Cache files carry no timing.

A region above the selected level is never entered by a clock, so a zero in
`last_timings` at `Off` is an uncollected default, not a measurement; read the
record's `timing_level` first. The NVTX ranges
(`CDFMM_ENABLE_PROFILING`) are a separate compile-time mechanism for external
profilers and do not depend on the timing level, so a profiling build can run
`Off` under Nsight. The C ABI exposes the same switch through
`cdfmm_plan_set_timing_level`; `cdfmm_plan_get_last_evaluation_seconds` fails
rather than returning zero while a plan is `Off` (see
[C and Fortran](c-and-fortran.md) for the compatibility status of that
change).

### External clock versus internal timing

Every driver keeps two clocks apart, and a CSV column belongs to exactly one
of them.

- **External benchmark wall clock**: the driver's own `steady_clock` around
  complete constructor or `evaluate`/`evaluate_into` calls. These are the
  headline columns and are measured at every level: in
  `benchmark_uniform_fmm` `fmm_setup_seconds`, `evaluation_median`,
  `evaluation_mean`, `evaluations_per_second`, `amortised_seconds`,
  `direct_seconds`, `accuracy_reference_seconds` and the
  `workload_*_median` columns; in `benchmark_p2p` `setup_s` and
  `host_total_s`; in `benchmark_dense_direct_construction`
  `construction_seconds`, `first_evaluation_seconds` and
  `evaluation_seconds`.
- **Internal solver timing**: values copied from the solver's own records
  (`EvaluationTimings`, `StaticPlanStatistics`, `TreeBuildTimings`,
  `CudaEvaluationTimings`, the dense construction records). They obey the
  timing level exactly and stay zero when the level did not collect them:
  the tree breakdown (`tree_total` … `interaction_lists`), the evaluation
  phases (`moment_permutation` … `result_unpermutation`), the CUDA lanes
  (`cuda_*`), the static-plan phases (`static_plan_seconds` …
  `p2p_tensor_plan_seconds`), the P2P driver's `h2d_s`/`kernel_s`/`d2h_s`,
  and the dense driver's phase columns (that driver requests `Detailed`,
  because the phase split is what it exists to report).

The direct reference backends of `benchmark_uniform_fmm` have no
`UniformFmm`-style collector: the CPU all-to-all reference has none, and the
CUDA direct plan has only its device lanes. Their phase columns are therefore
never filled from the driver's clock, so an `Off` row for `cpu-direct` or
`cuda-direct` carries a measured `evaluation_median` and zeros everywhere
else. The trailing column `internal_timing_source` (`uniform_fmm`,
`cuda_direct_plan` or `none`) names the collector behind a row's internal
columns; it was added after the Phase-3D baseline, whose CSVs and analysers
(which read columns by name) are unaffected.

For a performance measurement, run with timing `Off` and read the external
columns; every driver records the level it ran at (`--timing
off|coarse|detailed`, default `off`, column `timing_level`). The Article1
campaign uses `TimingLevel::Off` and external wall-clock timing of repeated
`evaluate_into` calls only. Use `Detailed` in a separate diagnostic run when
the phase split is the question.

## Reading the numbers

`EvaluationTimings` (Python `last_timings`) reports caller wall time per
phase at `TimingLevel::Detailed` (`total`, `far_field` and `p2p` already at
`Coarse`). `m2l` is a top-level phase; `m2l_scale`, `m2l_gather`, `m2l_multiply`
and `m2l_scatter` partition its work and are excluded from top-level
phase-share normalisation so the same time is not counted twice (CUDA uses
`m2l_scale` for its pre-scaling pass; grouped CPU execution uses gather,
multiply and scatter). For hybrid CUDA the multipole H2D and local D2H are in
that nested partition; `cuda_kernel` is a diagnostic sum, not another phase.
Both CUDA backends report their independently scheduled near field in the
CUDA P2P lane (`cuda_p2p_h2d`, `cuda_p2p_kernel`, `cuda_p2p_d2h`,
`cuda_p2p_wait`); the full backend keeps moments and near field
device-resident, so only `cuda_p2p_kernel` is non-zero. These lanes overlap
the far field and are never added to obtain wall time; compare them with
`total`. Timers surround complete OpenMP regions, so they report caller
elapsed time, not summed thread time. CSV rows carry `execution_backend`,
compile/runtime CUDA status, device, setup bytes, per-evaluation H2D/D2H bytes
and upload counts, persistent device bytes, the runtime-selected
`oneMKL`/`portable` implementation, and accuracy columns (reference target
count, sampled reference time, mean/RMS/maximum relative field error). Large
automated cases disable the quadratic reference explicitly rather than
reporting a misleading zero error.

Evaluation is internally OpenMP-parallel, but concurrent calls on one
`UniformFmm` are unsupported because expansions, scratch and timers are
mutable; separate objects may be evaluated from separate threads. Morton
sorting is serial and complete-tree storage grows geometrically with depth.

The `magtense` preset deliberately has `CDFMM_ENABLE_OPENMP=OFF`: MagTense
uses Intel OpenMP and MKL's Intel threading layer, so a parallel integration
must be a separate IntelLLVM/libiomp5 build rather than a mix of `libgomp` and
`libiomp5` in one process.

## Profiling

`benchmark_uniform_fmm --profile` keeps the driver's geometry, moments, seed,
backend, depth, order and thread controls and changes only the measurement
policy (one warm-up, one sample, ten back-to-back evaluations, validation
off; `--evaluations N` changes the count, `--direct` or `--accuracy-targets`
restores validation). Use an optimised build with symbols:

```console
cmake -S . -B build-profile -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCDFMM_BUILD_BENCHMARKS=ON -DCDFMM_ENABLE_NATIVE_ARCH=ON     # CPU
cmake --fresh --preset profile-all && cmake --build --preset profile-all -j   # CUDA + oneMKL + NVTX
```

With profiling disabled the range abstraction is an empty object: evaluation
makes no NVTX calls, allocations, registry lookups or extra synchronisation.

```console
# CPU
perf stat   build-profile/benchmarks/benchmark_uniform_fmm --backend cpu-static-matrix \
    --sources 50000 --targets 50000 --depth 4 --order 6 --threads 8 --profile
perf record --call-graph dwarf build-profile/benchmarks/benchmark_uniform_fmm ... --profile

# CUDA timeline, then one kernel
nsys profile --trace=cuda,nvtx,osrt --sample=cpu --output=cdfmm_cuda_full \
    build-profile-all/benchmarks/benchmark_uniform_fmm --backend cuda-full \
    --sources 50000 --targets 50000 --depth 4 --order 6 --profile
ncu --set full --kernel-name regex:.*m2l.* --launch-count 1 \
    build-profile-all/benchmarks/benchmark_uniform_fmm --backend cuda-full ... --profile

# isolated near-field packings side by side
nsys profile --trace=cuda,osrt --sample=cpu --output=cdfmm_static_p2p \
    build-profile-all/benchmarks/benchmark_p2p --cuda --depth 2 --occupancy 8 --evaluations 20
```

The stable NVTX names begin with `cdfmm/evaluate`, with `cdfmm/far_field` and
`cdfmm/near_field` branches and operator-specific children. Check in a
`cuda-full` timeline that repeated evaluation transfers changing moments in
and the final field out, without persistent geometry; near an advised depth
compare the two branches against `T_eval ≈ max(T_near, T_far) + T_overhead`.
Suggested source-point cases with the default seed `314159`: 10 000 points at
order 4, 50 000 at orders 6 and 8, 100 000 at order 6, each at the adviser's
depth. Record the Git commit, compiler, flags, CPU/GPU, thread count, backend,
particle count, depth, order, seed and tool version with every capture; the
profiling-mode summary prints the effective workload values and whether NVTX
was compiled in. Nsight Compute needs GPU performance counters enabled by an
administrator.
