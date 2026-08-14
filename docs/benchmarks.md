# Performance benchmarks

## Complete reset, build, test, and rough benchmark

Run from the repository root. This resets all build directories but preserves
existing benchmark results:

```bash
conda env update -n cdfmm -f environment.yml
conda env update -n cdfmm -f environment-cuda.yml
conda activate cdfmm

rm -rf \
  build \
  build-release \
  build-cuda \
  build-bench \
  build-bench-mkl \
  build-bench-all

cmake --fresh --preset cuda
cmake --build --preset cuda -j"$(nproc)"

ctest --preset cuda --timeout 120
python -m pytest python_tests -v

cmake --fresh --preset benchmark-all
cmake --build --preset benchmark-all -j"$(nproc)"

./build-bench-all/benchmarks/benchmark_uniform_fmm --cuda-status

python benchmarks/run_benchmarks.py \
  --profile rough \
  --max-threads 8 \
  --executable build-bench-all/benchmarks/benchmark_uniform_fmm
```

Before benchmarking, confirm the capability output includes:

```text
cuda_available=1
cuda_direct_available=1
cuda_m2l_available=1
cuda_full_available=1
one_mkl_available=1
```

To also remove all previous generated benchmark results, explicitly run before
the benchmark:

```bash
rm -rf benchmark_results
```

That last command permanently removes the old CSV files and figures.

## Combined CPU, CUDA, and oneMKL measurements

Use the combined preset to place all six implemented strategies in one
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
`cuda-partial`, `cuda-full`, `cpu-static-matrix`, and
`cpu-static-matrix-mkl`. The older `cuda-m2l-p2p`, `cuda-m2l`, and
`cuda-m2l-static-p2p` CLI spellings remain aliases for `cuda-partial`. CUDA direct
uses a persistent
plan: source and target positions are uploaded once during construction, while
each subsequent evaluation uploads only changing dipole moments and downloads
the requested results. It is an O(N^2) direct P2P implementation, not CUDA M2L
or a CUDA FMM. Each direct backend is measured exactly once per configured
particle count because all-to-all P2P does not construct or depend on a tree.
Direct CSV rows use `suite=direct` and record order and depth as zero to mark
them as not applicable.

CSV rows identify `execution_backend`, compile/runtime CUDA status, device,
setup bytes, per-evaluation H2D/D2H bytes, upload counts, and persistent device
bytes. Accuracy columns include the number of exact-reference targets, sampled
reference time, and mean/RMS/maximum relative field errors. CUDA rows report
separate device timings. For `cuda-partial`, the P2P lane
overlaps the dependent P2M/M2M/M2L/L2L/L2P chain and is therefore plotted
separately rather than added to its phase total. Caller wall time remains the
primary end-to-end measurement. For `cuda-full`, the aggregate CUDA kernel
timer is excluded from phase totals because the individual device P2M, M2M,
M2L, L2L, L2P, and P2P timers already partition that work. CUDA
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

The static FMM rows retain the uniform tree, Morton permutations, source/target
ranges, list1/list2 interaction lists, and one dense Cartesian M2L coefficient
matrix `T(R)` for every occupied
`(level, dx, dy, dz)` transfer class. They also retain the interaction index
maps and gather/translated scratch buffers used for grouped multiplication.
`cpu-static-matrix` uses the portable nested-loop multiply;
`cpu-static-matrix-mkl` selects oneMKL DGEMM at runtime from the same binary.

The static plan caches sparse P2M maps per occupied source leaf, shared
triangular M2M and L2L maps, dense M2L matrices per transfer class, fixed L2P
rows per target, and sparse list-1 P2P tensor blocks. CSV columns report
construction time and storage across these operators. Runtime instrumentation
separates `m2l_gather`, `m2l_multiply`, and `m2l_scatter` for the dense M2L
strategy. The older per-interaction M2L traversal is retained only as a
validation reference.

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
same M2L time is not counted twice. For hybrid CUDA M2L/P2P, multipole H2D and
local-coefficient D2H are included in this nested partition as well;
`cuda_kernel` is a diagnostic sum and is not counted as another top-level
phase. The hybrid backend additionally reports `cuda_p2p_h2d`,
`cuda_p2p_kernel`, `cuda_p2p_d2h`, and the single residual
`cuda_p2p_wait`. These form an independent overlapping lane and are excluded
from sequential top-level phase-share normalisation.

Every automated benchmark row records two end-to-end workloads: exactly one
construction plus one evaluation, and exactly one construction plus ten
evaluations with changing dipole moments. Construction, evaluation, and total
median times are stored separately. This applies independently to CPU direct,
GPU direct, portable static-matrix FMM, oneMKL static-matrix FMM, and hybrid
CUDA M2L/P2P FMM. In the
GPU-direct 1+10 workload, positions remain device-resident and only moments
and results cross the PCIe boundary between evaluations. Hybrid CUDA M2L/P2P keeps
static M2L matrices and sparse P2P tensors resident while transferring packed
multipoles, raw M2L locals, changing moments, identities, and near fields per
evaluation. OpenMP, CUDA, and oneMKL
runtime initialisation is warmed before these workloads are timed. This removes
process-global driver and library first-use costs. Timed construction still
includes everything owned by an evaluator: tree and static-plan construction,
CUDA handle and stream creation, device allocation, and static operator or
geometry uploads.

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

For a deliberately coarse six-backend comparison, use the `rough` profile:

```console
python benchmarks/run_benchmarks.py --profile rough --max-threads 8 \
  --executable build-bench-all/benchmarks/benchmark_uniform_fmm
```

It runs exactly 20 processes: the four FMM backends at 20,000 and 30,000
particles, depths three and four, and expansion order four (16 runs), plus one
CPU-direct and one CUDA-direct run at each particle count (four runs). Each
process records
one warmed timed evaluation with one sample, plus independent 1+1 and 1+10
construction/evaluation workloads. The profile deliberately omits thread
scaling, the extra comparison suite, and a full all-target accuracy reference.
Instead, each process checks 128 deterministically spaced targets against exact
CPU P2P over all sources. The target count and RMS/maximum relative errors are
included in `summary.md`. Accuracy sampling is outside all timed benchmark
regions. The profile still
produces construction-inclusive, evaluation-only, and setup-amortisation plots
for every geometry, including projections through 10,000 evaluations. It fails
explicitly unless all six backends are available, including `cuda-full`.

```console
python benchmarks/run_benchmarks.py --profile quick --max-threads 8
python benchmarks/run_benchmarks.py --profile standard --max-threads 8
python benchmarks/run_benchmarks.py --profile full --max-threads 8
```

`--max-threads` caps the thread count used by every parameter-grid case. It also
limits the scaling sweep; for example, `--max-threads 8` tests 1, 2, 4, and 8
threads even when the machine reports more logical CPUs. Without the option,
the runner uses all logical CPUs reported by the operating system.

Profiles contain a full FMM `parameter_grid` suite in `results.csv`, covering
every configured particle-count, expansion-order, and tree-depth combination.
The separate `direct` suite contains one CPU-direct row and, when available,
one CUDA-direct row per particle count. The `scaling` suite varies threads at
the largest problem size for every backend, including CPU and CUDA direct.
Its maximum-thread direct endpoint reuses the corresponding `direct` row, so
that configuration is not executed twice. The
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

The terminal reports the exact suite, source/target count, applicable tree
parameters, backend, and thread count before every case. It shows overall case
progress,
while the C++
executable updates completed samples and timed evaluations after each sample.
Progress printing occurs outside the measured interval. Figures are grouped by
result type under `figures/runtime/`, `accuracy/`, `work/`, `phases/`,
`setup/`, `scaling/`, and `comparison/`. `figures/combined_overview.png`
combines runtime, accuracy, M2L translations, and near-field pairs for the full
grid. Every recorded row has individual phase and setup/evaluation plots;
tree-based rows also have work plots in the corresponding `per_run/`
directory. Per-configuration depth sweeps and
per-order/depth FMM particle sweeps provide less crowded detailed views. Direct
runtime particle sweeps contain one point per configured particle count and do
not carry an order or depth label.

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

The static backend caches all geometry-dependent operator maps, including
grouped M2L translation matrices, reusable gather/scatter buffers, and sparse
near-field tensors. Morton stable sorting remains serial, and complete-tree
storage can become expensive at excessive depth.

## Reproducible optimisation configuration

A portable Release build uses the compiler's normal optimised Release flags and
produces a binary suitable for other compatible machines:

```console
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release
```

For measurements confined to the build workstation, native instruction
selection and CMake's checked interprocedural optimisation can be enabled:

```console
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCDFMM_ENABLE_NATIVE_ARCH=ON \
    -DCDFMM_ENABLE_LTO=ON
```

`CDFMM_ENABLE_NATIVE_ARCH` is off by default because such a binary may fail on
older processors. `CDFMM_ENABLE_FAST_MATH` is also off by default and is not
part of the canonical benchmark configuration: it can alter IEEE behaviour,
reproducibility, and numerical error, so scientific results require independent
validation when it is used. Advanced users may append semicolon-separated C++
options with `CDFMM_EXTRA_COMPILE_OPTIONS`; these do not replace build-type or
standard CMake compiler flags and are deliberately not forwarded to CUDA.
Record the configuration summary printed by CMake, compiler version, backend,
CUDA architecture/device where applicable, and thread environment with every
reported benchmark result.
