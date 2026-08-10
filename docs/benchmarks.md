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

## Automated profiles

```console
python benchmarks/run_benchmarks.py --profile quick
python benchmarks/run_benchmarks.py --profile standard
python benchmarks/run_benchmarks.py --profile full
```

Profiles sweep particle count, tree depth, expansion order, and available
power-of-two thread counts. The driver creates a timestamped directory below
`benchmark_results/` containing `results.csv`, machine metadata, a measured
summary, and PNG figures for runtime scaling, direct/FMM comparison,
order/accuracy tradeoffs, phase breakdowns, setup amortisation, and OpenMP
scaling. Generated results are ignored by Git and existing runs are not
overwritten.

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
