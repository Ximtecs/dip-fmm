# Performance benchmarks

Benchmark timing answers how fast a deterministic workload is. To investigate
why it takes that long with `perf`, NVIDIA Nsight, and the same benchmark
driver, see the dedicated [profiling guide](profiling.md).

The isolated fixed-geometry near-field benchmark and current results are in
the [static P2P execution study](static-p2p.md). It compares canonical rows,
source-only SoA, compact leaf blocks, portable BSR(3), and oneMKL BSR(3).
In a CUDA-enabled build, `--cuda` adds canonical CUDA, CUDA SoA, the
shared-memory leaf kernel, and cuSPARSE BSR(3), with transfers and kernel time
reported separately.

## Comparison map and measurement rules

Different tools answer different comparison questions:

| Comparison | Current tool |
|---|---|
| Direct CPU versus direct CUDA | `benchmark_uniform_fmm` / benchmark driver |
| Portable CPU, oneMKL, CUDA partial, and CUDA full FMM | benchmark driver profiles |
| Cartesian versus spherical finite cuboids | `simple_cartesian_spherical_fmm_compare.ipynb` |
| FP32 versus FP64 | `simple_dense_direct_precision_compare.ipynb` |
| Static spherical FMM versus FMM3D | `11_fmm3d_comparison.ipynb` |
| Cartesian cuboid FMM versus exact cuboid direct | `simple_cuboid_fmm_direct_compare.ipynb` |
| Spherical cuboid P2M and L2P isolation | `simple_cuboid_p2m_l2p_direct_compare.ipynb` |
| Direct cuboid convention versus MagTense | `simple_cuboid_magtense_compare.ipynb` |

All reported FMM results must keep setup separate from repeated evaluation,
reuse one fixed geometry for changing moments, warm runtime libraries before
timing, and report accuracy beside runtime. CUDA timings and crossover points
are hardware-specific. Cartesian/spherical order, FMM3D tolerance, and
precision are independent controls and must not be presented as intrinsically
accuracy-equivalent. Stored notebook output is historical measurement data and
is not rewritten without rerunning the notebook.

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

Setting both `use_cuboid_p2m=false` and `use_cuboid_l2p=false` provides the
tested hybrid model: list1 P2P remains exact cuboid-to-cuboid, while far-field
P2M treats each supplied total moment as a point dipole at its cuboid centre
and L2P samples the local expansion at the target centre. Source and target
sizes select near-field physics; callers must still convert magnetisation to
total dipole moment before evaluation.

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
  build-bench-all \
  build-profile-all

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
ranges, list1/list2 interaction lists, and one dense basis-specific,
level-independent M2L coefficient matrix for every used integer displacement
class. Degree-dependent box-width scaling restores each physical level. They
also retain interaction index maps and gather/translated scratch buffers used
for grouped multiplication.
`cpu-static-matrix` uses the portable nested-loop multiply;
`cpu-static-matrix-mkl` selects oneMKL SGEMM or DGEMM at runtime from the same
binary.

The static plan caches sparse P2M maps per occupied source leaf, shared
triangular M2M and L2L maps, dense M2L matrices per transfer class, fixed L2P
rows per target, and sparse list-1 P2P tensor blocks. CSV columns report
construction time and storage across these operators. Runtime instrumentation
separates `m2l_scale`, `m2l_gather`, `m2l_multiply`, and `m2l_scatter` for
dense M2L strategies. CUDA uses `m2l_scale` for its device pre-scaling pass;
CPU grouped execution uses gather, multiply, and scatter. The older
per-interaction M2L traversal is retained only as a validation reference.

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
`CDFMM_ENABLE_LTO=ON`. LTO is enabled by default for Release builds. A
portable serial build remains available with
`-DCDFMM_ENABLE_OPENMP=OFF`. OpenMP uses the standard runtime controls:

```console
OMP_NUM_THREADS=16 ./build-bench/benchmarks/benchmark_uniform_fmm \
  --sources 10000 --targets 10000 --depth 4 --order 6 \
  --threads 16 --evaluations 100 --warmups 2 --samples 10 \
  --output result.csv
```

### Signed tensor-dictionary CPU P2P

The experimental signed tensor-dictionary path is selected explicitly with
`UniformFmmOptions::use_reduced_symmetry_p2p`; it is not the default. Its
OpenMP target tile is independently configurable with
`signed_p2p_target_tile_size` (valid range 1--128, default 32). The inner SIMD
microtile remains fixed at the compiled native width (eight FP32 or four FP64
targets for AVX2), while OpenMP statically owns disjoint `(target leaf, target
tile)` work items. Recommended runtime affinity is:

```console
OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=8 \
  ./build-bench-all/benchmarks/benchmark_p2p --depth 2 --occupancy 32 \
  --evaluations 10 --signed-target-tile 32
```

The benchmark reports particle-row SoA, the former whole-target-tile signed
kernel, and the production SIMD-microtile signed kernel in both FP64 and FP32.
Generate GCC vectorization diagnostics and retain inspectable non-LTO object
code with:

```console
cmake -S . -B build-p2p-vector -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCDFMM_BUILD_TESTS=OFF -DCDFMM_BUILD_EXAMPLES=OFF \
  -DCDFMM_BUILD_PYTHON=OFF -DCDFMM_BUILD_BENCHMARKS=ON \
  -DCDFMM_ENABLE_CUDA=OFF -DCDFMM_ENABLE_MKL=OFF \
  -DCDFMM_ENABLE_LTO=OFF -DCDFMM_ENABLE_VECTORIZATION_REPORTS=ON
cmake --build build-p2p-vector --target benchmark_p2p -j
less build-p2p-vector/cdfmm-static-operators.vec
objdump -dC build-p2p-vector/libcdfmm_core.a | \
  grep -E 'apply_signed_microtile_avx2|vgather|vfmadd'
```

The `magtense` preset deliberately has `CDFMM_ENABLE_OPENMP=OFF`: consequently
the signed P2P executor is SIMD-vectorized but single-threaded in that build.
MagTense itself uses Intel OpenMP and MKL's Intel threading layer, whereas the
inherited cdfmm notebook build uses GNU C++; simply turning cdfmm OpenMP on
would mix `libgomp` and `libiomp5` in one process. A parallel integration
should therefore be configured only as a separate, fully IntelLLVM/libiomp5
build after validating the complete MagTense link, rather than by overriding
the existing preset.

The runner sets `OMP_NUM_THREADS` to each case's thread count,
`MKL_NUM_THREADS=1`, and configures the OpenMP runtime directly with
`--threads`. The
CSV records compiler and OpenMP metadata, geometry work counters,
direct-reference accuracy, tree phases, and evaluation phases. Median elapsed
wall time across samples is the primary result. Input generation, reporting,
and file output are outside timed regions.

`m2l` is a top-level parent phase. `m2l_scale`, `m2l_gather`, `m2l_multiply`,
and `m2l_scatter` partition its static-matrix work and appear in a separate nested
breakdown. They are excluded from top-level phase-share normalisation so the
same M2L time is not counted twice. For hybrid CUDA M2L/P2P, multipole H2D and
local-coefficient D2H are included in this nested partition as well;
`cuda_kernel` is a diagnostic sum and is not counted as another top-level
phase. Both CUDA FMM backends report their independently scheduled near-field
work in the CUDA P2P lane. The hybrid backend includes `cuda_p2p_h2d`,
`cuda_p2p_kernel`, `cuda_p2p_d2h`, and the single residual `cuda_p2p_wait`.
The full CUDA backend keeps the moments and near-field result device-resident,
so only `cuda_p2p_kernel` is non-zero. These values are excluded from
sequential top-level phase-share normalisation because P2P can overlap the
far-field hierarchy.

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
FMM row uses the executable's spherical default unless
`--expansion-basis cartesian` is supplied in a direct executable invocation.
Each process records
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

Persistent-cache setup can be measured separately with the release benchmark:

```console
cache_dir=$(mktemp -d)
benchmark_cache_initialisation --depth 3 --grid 8 --backend onemkl \
    --cache-dir "$cache_dir"
```

The cache directory must initially be empty. The program constructs the same
spherical `p=6`, FP32, uniform-cuboid plan twice and emits `CACHE_BENCH` CSV
rows for cold and warm setup, with normalization, tree, lookup/hash/load,
analytical operator, backend-packing, CUDA-upload, and byte-count columns.
Backends are `portable`, `onemkl`, and `cuda-full`.

## Reproducible optimisation configuration

Release builds enable LTO and native CPU instruction selection by default:

```console
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release
```

The resulting binary is optimised for the build machine and may not run on an
older or otherwise incompatible processor. Disable native instruction selection
when a portable binary is required:

```console
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCDFMM_ENABLE_NATIVE_ARCH=OFF
```

Fast math remains off by default. Enable it explicitly with:

```console
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCDFMM_ENABLE_FAST_MATH=ON
```

Fast math is not part of the canonical benchmark configuration: it can alter
IEEE behaviour, reproducibility, and numerical error, so scientific results
require independent validation when it is used. Advanced users may append
semicolon-separated C++ options with `CDFMM_EXTRA_COMPILE_OPTIONS`; these do
not replace build-type or standard CMake compiler flags and are deliberately
not forwarded to CUDA.
Record the configuration summary printed by CMake, compiler version, backend,
CUDA architecture/device where applicable, and thread environment with every
reported benchmark result.
