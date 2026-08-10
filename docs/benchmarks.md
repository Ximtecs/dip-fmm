# Performance benchmarks

The performance path uses Intel oneAPI `icpx`, Release optimisation, native CPU
code generation, IPO where supported, and OpenMP. Configure it from the
repository root:

```console
conda activate cdfmm
cmake --preset benchmark
cmake --build --preset benchmark
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

`--max-threads` caps the thread count used by every size/order and tree-depth
case. It also limits the scaling sweep; for example, `--max-threads 8` tests 1,
2, 4, and 8 threads even when the machine reports more logical CPUs. Without
the option, the runner uses all logical CPUs reported by the operating system.

Profiles contain three explicit suites in `results.csv`: `size_order` varies
particle count and expansion order, `depth` tests every profile tree depth at a
fixed particle count and order, and `scaling` varies OpenMP threads. The driver
creates a timestamped directory below `benchmark_results/` containing the CSV,
machine metadata (including the thread cap), a measured summary, and PNG
figures. Figure titles state their particle count, order, depth, and thread
configuration. The phase table in `summary.md` uses the largest particle count
and the tested expansion order nearest four. Generated results are ignored by
Git and existing runs are not overwritten.

The terminal reports the exact suite, source/target count, order, depth, and
thread count before every case. It shows overall case progress, while the C++
executable updates completed samples and timed evaluations after each sample.
Progress printing occurs outside the measured interval. A representative phase
plot remains at `figures/evaluation_breakdown.png`; phase plots for every case
are stored under `figures/evaluation_breakdowns/` with names such as
`depth_n300_p4_d2_t8.png`.

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

The current implementation deliberately does not cache translation matrices or
near-field tensors. Morton stable sorting remains serial, and complete-tree
storage can become expensive at excessive depth. These limitations are inputs
to later profiling and the separate static-geometry research milestone.
