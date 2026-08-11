# Performance benchmarks

## Combined CPU, CUDA, and oneMKL measurements

Use the combined preset to place all four implemented strategies in one
executable:

```console
conda activate cdfmm
cmake --fresh --preset benchmark-all
cmake --build --preset benchmark-all
python benchmarks/run_benchmarks.py --profile quick --max-threads 4 \
  --executable build-bench-all/benchmarks/benchmark_uniform_fmm
```

The runner prefers this combined executable and probes its capabilities before
planning measurements. A complete run compares `cpu-direct`, `cuda-direct`,
`cpu-static-matrix`, and `cpu-static-matrix-mkl`. CUDA direct uses a persistent
plan: source and target positions are uploaded once during construction, while
each subsequent evaluation uploads only changing dipole moments and downloads
the requested results. It is an O(N^2) direct P2P implementation, not CUDA M2L
or a CUDA FMM. Expansion order and tree depth are retained on both direct rows
only to identify their paired FMM geometry.

CSV rows identify `execution_backend`, compile/runtime CUDA status, device,
setup bytes, per-evaluation H2D/D2H bytes, and persistent device bytes. CUDA
rows also report device-stream H2D, kernel, and D2H times measured with CUDA
events; these phases appear in the same phase plots as the CPU traversal
phases. Caller wall time remains the primary end-to-end measurement. CUDA
results are hardware-specific and are never generated in GitHub Actions. Do
not report crossover or amortisation claims without measured GPU output.

The CPU-only performance preset uses Intel oneAPI `icpx`, Release optimisation,
native CPU code generation, IPO where supported, and OpenMP. The combined
CUDA/oneMKL preset uses `g++` as the CUDA-compatible host compiler. Configure
the CPU-only portable build from the repository root with:

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

The two FMM rows retain the uniform tree, Morton permutations, source/target
ranges, list1/list2 interaction lists, and one dense Cartesian M2L coefficient
matrix `T(R)` for every occupied
`(level, dx, dy, dz)` transfer class. They also retain the interaction index
maps and gather/translated scratch buffers used for grouped multiplication.
`cpu-static-matrix` uses the portable nested-loop multiply;
`cpu-static-matrix-mkl` selects oneMKL DGEMM at runtime from the same binary.

Only M2L coefficient matrices are currently cached. There are no stored P2M,
M2M, L2L, or L2P matrices. CSV columns report the M2L strategy, number and bytes
of cached matrices, interaction-map bytes, scratch bytes, and total static-plan
storage. Runtime instrumentation separates `m2l_gather`, `m2l_multiply`, and
`m2l_scatter` for the matrix strategy. The older per-interaction M2L traversal
is omitted because it makes complete sweeps prohibitively slow.

Gathering and matrix application are scheduled across independent transfer
classes with OpenMP. Each worker processes a complete grouped matrix operation;
the oneMKL path gives it a single-threaded DGEMM call. Scattering remains serial
because different transfer classes can contribute to the same target local.
This avoids nested MKL teams and repeated synchronisation for every small
matrix while retaining the optimised MKL kernel.

The `benchmark-mkl` and `benchmark-all` presets set `CDFMM_ENABLE_MKL=ON`, which
includes rather than replaces the portable implementation. CSV output
identifies the runtime-selected implementation as `oneMKL` or `portable`. Record the
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

The runner sets `OMP_NUM_THREADS` to each case's thread count,
`MKL_NUM_THREADS=1`, and configures the OpenMP runtime directly with
`--threads`. The
CSV records compiler and OpenMP metadata, geometry work counters,
direct-reference accuracy, tree phases, and evaluation phases. Median elapsed
wall time across samples is the primary result. Input generation, reporting,
and file output are outside timed regions.

`m2l` is a top-level parent phase. `m2l_gather`, `m2l_multiply`, and
`m2l_scatter` partition its static-matrix work and appear in a separate nested
breakdown. They are excluded from top-level phase-share normalisation so the
same M2L time is not counted twice.

Every automated benchmark row records two end-to-end workloads: exactly one
construction plus one evaluation, and exactly one construction plus ten
evaluations with changing dipole moments. Construction, evaluation, and total
median times are stored separately. This applies independently to CPU direct,
GPU direct, portable static-matrix FMM, and oneMKL static-matrix FMM. In the
GPU 1+10 workload the positions remain device-resident and only moments and
results cross the PCIe boundary between evaluations. OpenMP, CUDA, and oneMKL
runtime initialisation is warmed before these workloads are timed.

The comparison suite uses one representative geometry to produce two backend
workload figures. `backend_workloads_with_creation.png` includes backend
construction, while `backend_workloads_evaluation_only.png` excludes it and
shows the total time spent in one or ten evaluations. Both use a logarithmic
time axis so direct and FMM results remain legible when their setup costs differ
substantially. The per-run setup figure further separates tree construction,
static M2L matrix-plan construction, other backend setup, and one evaluation.
In particular, a large static-matrix setup time must not be interpreted as tree
construction time.

CPU direct is already the all-to-all CPU P2P reference, so it is not repeated
as a direct-versus-FMM figure. Other backends are compared with the CPU direct
reference where direct validation was requested.

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
`comparison` suite presents the backend workload comparison at one common
geometry. The driver
creates a timestamped directory below `benchmark_results/` containing the CSV,
machine metadata (including the thread cap), a measured summary, and PNG
figures. Figure titles state their particle count, order, depth, backend, and
thread configuration. The phase table in `summary.md` deliberately uses the
portable static-matrix row at the fastest depth for the largest particle count
and tested expansion order nearest four; per-run phase plots cover direct,
static-matrix, and CUDA phases. Generated
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
