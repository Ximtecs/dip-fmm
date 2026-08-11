# Performance benchmarks

## CUDA measurements

Build CUDA benchmarks with `cmake --fresh --preset cuda` and
`cmake --build --preset cuda`, then run:

```console
python benchmarks/run_benchmarks.py --profile quick --max-threads 4
```

The runner prefers the CUDA-preset executable when it exists and probes it
before planning any measurements. A CPU-only executable runs every ordinary
geometry with `cpu-static`. A CUDA-compiled executable must have an accessible
device and runs every parameter-grid, scaling, and comparison geometry three times:
`cpu-static`, `cuda-m2l`, and `cuda-full`, using the same deterministic geometry
and moment states. It never silently omits CUDA measurements from a CUDA build.
The deliberately expensive `cpu-reference` traversal remains confined to the
single dedicated comparison workload.

CSV rows identify `execution_backend`, compile/runtime CUDA status, device,
setup bytes, per-evaluation H2D/D2H bytes, and persistent device bytes. CUDA
rows also report device-stream H2D, kernel, and D2H times measured with CUDA
events; these phases appear in the same phase plots as the CPU traversal
phases. Caller wall time remains the primary end-to-end measurement. CUDA
results are hardware-specific and are never generated in GitHub Actions. Do
not report crossover or amortisation claims without measured GPU output.

The performance path uses Intel oneAPI `icpx`, Release optimisation, native CPU
code generation, IPO where supported, and OpenMP. Configure it from the
repository root:

```console
conda activate cdfmm
cmake --preset benchmark
cmake --build --preset benchmark
```

To benchmark the grouped static M2L path with oneMKL DGEMM, use the dedicated
preset and executable:

```console
conda env update -n cdfmm -f environment.yml
conda activate cdfmm
cmake --preset benchmark-mkl
cmake --build --preset benchmark-mkl
python benchmarks/run_benchmarks.py --profile quick --max-threads 4 \
  --executable build-bench-mkl/benchmarks/benchmark_uniform_fmm
```

For repeated static geometry, construct one evaluator per backend and generate
all moment states before timing. Compare the default against
`M2LBackend::Reference`, reporting `static_plan_statistics` separately from
first and median evaluations. Runtime instrumentation separates `m2l_gather`,
`m2l_multiply`, and `m2l_scatter`; plan statistics report operator, interaction
map, and reusable scratch bytes. Cumulative static time is `setup + N *
median_static`, while reference time is `N * median_reference`; their first
crossing is the break-even count.

The standard preset exercises the same static plan with its portable dense
multiply. The `benchmark-mkl` preset sets `CDFMM_ENABLE_MKL=ON`; CSV output
identifies the selected implementation as `oneMKL` or `portable`. Record the
compiler/MKL versions plus `OMP_NUM_THREADS` and `MKL_NUM_THREADS`. Run the
quick reproducible portable sweep and figures with:

```console
python benchmarks/run_benchmarks.py --profile quick --max-threads 4
```

The preset sets `CMAKE_CXX_COMPILER=icpx`, `CDFMM_ENABLE_OPENMP=ON`, and
`CDFMM_ENABLE_IPO=ON`. A portable serial build remains available with
`-DCDFMM_ENABLE_OPENMP=OFF`. OpenMP uses the standard runtime controls:

```console
OMP_NUM_THREADS=16 ./build-bench/benchmarks/benchmark_uniform_fmm \
  --sources 10000 --targets 10000 --depth 4 --order 6 \
  --threads 16 --evaluations 100 --warmups 2 --samples 10 \
  --output result.csv
```

`--threads` calls the OpenMP runtime directly and takes precedence for the
process. The CSV records compiler and OpenMP metadata, geometry work counters,
direct-reference accuracy, tree phases, and evaluation phases. Median elapsed
wall time across samples is the primary result. Input generation, reporting,
and file output are outside timed regions.

Each automated profile adds one comparison case at its smallest particle count.
That case records two end-to-end workloads for each evaluation strategy:

- direct all-to-all P2P over every source-target pair;
- the independent reference FMM traversal;
- the static grouped FMM traversal, using oneMKL when the executable was built
  with `CDFMM_ENABLE_MKL=ON`.

The workloads are exactly one construction plus one evaluation and one
construction plus ten evaluations. Construction, evaluation, and combined
median times are stored separately. Direct all-to-all P2P has no reusable
geometry object, so its construction time is zero. Both FMM paths report error
metrics against the same direct all-to-all state. `backend_workloads.png` and
the generated summary
compare performance and accuracy for that case at the available expansion order
nearest four. Other sweep cases skip both the reference traversal and the
additional one-versus-ten construction comparison.

The benchmark is an all-to-all source-point comparison: the source and target
arrays contain the same particle positions, and both FMM and direct P2P use the
same explicit identity map to exclude only each particle's singular self-pair.
Consequently `--sources` and `--targets` must be equal.

## Automated profiles

```console
python benchmarks/run_benchmarks.py --profile quick --max-threads 8
python benchmarks/run_benchmarks.py --profile standard --max-threads 8
python benchmarks/run_benchmarks.py --profile full --max-threads 8
```

`--max-threads` caps the thread count used by every parameter-grid case. It also
limits the scaling sweep; for example, `--max-threads 8` tests 1, 2, 4, and 8
threads even when the machine reports more logical CPUs. Without the option,
the runner uses all logical CPUs reported by the operating system.

Profiles contain a full `parameter_grid` suite in `results.csv`, covering every
configured particle-count, expansion-order, and tree-depth combination. The
`scaling` suite varies OpenMP threads at the largest problem size, and the
`comparison` suite performs the dedicated direct/reference/static 1- and
10-evaluation workloads. The driver
creates a timestamped directory below `benchmark_results/` containing the CSV,
machine metadata (including the thread cap), a measured summary, and PNG
figures. Figure titles state their particle count, order, depth, backend, and
thread configuration. The phase table in `summary.md` deliberately uses the
CPU-static row at the fastest depth for the largest particle count and tested
expansion order nearest four; per-run phase plots cover static and CUDA
backends. The comparison outputs cover the reference traversal. Generated
results are ignored by Git and existing runs are not overwritten.

The terminal reports the exact suite, source/target count, order, depth,
backend, and thread count before every case. It shows overall case progress,
while the C++
executable updates completed samples and timed evaluations after each sample.
Progress printing occurs outside the measured interval. Figures are grouped by
result type under `figures/runtime/`, `accuracy/`, `work/`, `phases/`,
`setup/`, `scaling/`, and `comparison/`. `figures/combined_overview.png`
combines runtime, accuracy, M2L translations, and near-field pairs for the full
grid. Every recorded row has individual phase, setup/evaluation, and work plots
in the corresponding `per_run/` directory. Per-configuration depth sweeps and
per-order/depth particle sweeps provide less crowded detailed views.

## Setup and repeated evaluation

`UniformFmm` construction is geometry setup: it determines the root, sorts
positions, builds topology/ranges/interactions, and caches occupied traversal
indices. `evaluate()` and allocation-free `evaluate_into()` reuse this geometry
for changing moments. `tree().build_timings()` exposes setup timings;
`last_timings()`, `aggregate_timings()`, and `reset_timings()` expose evaluation
wall times. Timers surround complete OpenMP regions and therefore report caller
elapsed time rather than summed thread CPU time.

Evaluation is internally OpenMP-parallel, but concurrent calls on the same
`UniformFmm` instance are unsupported because expansions, scratch results, and
timing accumulators are mutable. Separate objects may be evaluated by separate
calling threads, although oversubscription must then be managed by the caller.

The static backend caches grouped M2L translation matrices and reusable gather
and scatter buffers. Near-field tensors are not cached. Morton stable sorting
remains serial, and complete-tree storage can become expensive at excessive
depth.
