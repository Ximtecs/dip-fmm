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

Setting both far-field model selectors to their point variants provides the
tested hybrid model: list1 P2P remains exact prism-to-prism, while far-field
P2M treats each supplied total moment as a point dipole at its prism centre
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

## Operator representation: precomputed versus procedural

`benchmark_operator_representation` answers one question for a single operator
family: is it faster to build the exact operator once and stream it on every
moment update, or to reconstruct it from the retained geometry during every
update? It measures the two representations separately from construction:

- **precomputed** builds the operator with the production builders
  (`build_static_p2p_operator`, `operators::p2m::build_cuboid` /
  `build_tetrahedron`, `operators::l2p::build_cuboid` / `build_tetrahedron`),
  derives the production execution packings (particle-row SoA, dense leaf
  blocks, the signed tensor dictionary, the dense `PackedP2M` / `PackedL2P`
  rows, and on CUDA the leaf-block and dictionary plans) and applies them with
  the production apply functions; and
- **procedural** retains only the sorted positions, the body records, the
  list-1 topology and the per-body invariants a builder already hoists
  (prepared tetrahedra, polyhedron surfaces), then calls the *same* per-pair
  or per-body builder during every update and applies the result at once.
  No operator is stored.

Both are checked against the FP64 canonical operator on the final moments of
every run, every update uses different moments and every result is
checksummed, so no work can be elided. Construction is timed separately and
never enters an update time.

Two modes bracket the memory behaviour:

- `--mode hot` repeats one small set (`--pairs`, default 512 pairs or 64
  bodies) single-threaded, so the stored operator stays in L1/L2 and the
  result is the intrinsic arithmetic cost of the representation; and
- `--mode streaming` traverses the complete list-1 neighbourhood of a lattice
  with every thread, as the FMM near field does. `--depth 3
  --bodies-per-leaf-axis 2` gives 4096 bodies and 681k pairs; depth 4 gives
  32768 bodies and 6.2M pairs, whose stored tensors exceed the last-level
  cache and the GPU L2.

`--separation` selects the numerical branch of a hot P2P set: `self` (the
physical finite demagnetisation tensor at zero displacement), `adjacent`,
`list1`, `small-far` (quarter-size bodies at list-1 offsets) and
`far-safeguard`, which places the bodies ten summed circumradii apart, beyond
the `polyhedron_far_separation_factor` switch to the Gauss-averaged source
tensor. A point source has no self pair: it is the singular one the identity
map excludes.

```console
# one finite pair type, both representations, hot and streaming
./build-cuda/benchmarks/benchmark_operator_representation \
  --stage p2p --source prism --target tetrahedron --mode streaming \
  --layout irregular --output results.csv

# the exact finite expansions at several orders
./build-cuda/benchmarks/benchmark_operator_representation \
  --stage p2m --source tetrahedron --orders 4,6,8 --mode streaming \
  --output results.csv

# the CUDA stored representations and the procedural prism device kernel
./build-cuda/benchmarks/benchmark_operator_representation \
  --stage p2p --source prism --target point --mode streaming --cuda \
  --no-cpu-precomputed --no-cpu-procedural --output results.csv
```

A procedural finite update can take minutes, so
`--procedural-budget-seconds` (default 60) bounds one precision's complete
procedural measurement: the benchmark first reduces the evaluations per
sample and only then truncates the leaf traversal, recording the fraction it
measured in `measured_fraction` and scaling the reported update time by it.

`benchmarks/run_operator_representation.py --binary <executable> --output
<csv> --suite {p2p-hot,p2p-streaming,p2p-large,p2p-cuda,p2m,l2p,all}` runs the
whole matrix and appends every row to one CSV. It resumes: a case whose rows
are already in the CSV is skipped unless `--no-resume` is given.
`benchmarks/analyse_operator_representation.py <csv> [--markdown out.md]
[--summary-csv out.csv]` pairs each procedural row with the best precomputed
representation of the same configuration and reports the ratio, the persistent
bytes saved and the amortisation break-even

```text
K_break_even = (T_build - T_procedural_setup) / (T_procedural - T_apply)
```

in complete field updates. The row CSV carries the reproduction metadata
(`commit`, `compiler`, `compiler_version`, `build_type`, `cpu_model`,
`gpu_model`, `threads`), the configuration (`stage`, `source_geometry`,
`target_geometry`, `layout`, `separation`, `mode`, `precision`, `order`,
`backend`, `bodies`, `leaves`, `items`), the timings (`build_seconds`,
`build_seconds_per_item`, `update_seconds`, `ns_per_item`,
`measured_fraction`), the memory breakdown (`geometry_bytes`,
`topology_bytes`, `operator_bytes`, `index_bytes`, `metadata_bytes`,
`invariant_bytes`, `scratch_bytes`, `total_persistent_bytes`,
`unique_operators`) and the verification (`checksum`,
`max_relative_error`, `note`). The measured result is in
`agent_docs/performance_optimization.md`, "Exact finite-geometry procedural vs
precomputed operators (Phase 3B.5b)".

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

### Geometry, packing, and periodic controls

`benchmark_uniform_fmm` benchmarks finite bodies and forced P2P packings on
the same fixed-geometry, repeated-moment workload as the point cases:

```console
./build-cuda/benchmarks/benchmark_uniform_fmm --backend cuda-full \
  --sources 32768 --targets 32768 --depth 4 --order 6 --regular-grid \
  --source-geometry tetrahedron --target-geometry tetrahedron \
  --p2p-packing tensor-dictionary --precision float32 \
  --no-direct --no-workload-comparison --accuracy-targets 0
```

- `--source-geometry` / `--target-geometry` select `point`, `prism`, or
  `tetrahedron`; finite bodies are centred on the particle positions with
  extent `--body-fill` (default 0.9) of the nominal spacing, one common record
  by default and one varying record per body with `--irregular-bodies`.
  Finite bodies use point far-field models by default, so every backend shares
  the identical hierarchy and the comparison isolates the stored-tensor P2P.
  `--exact-cuboid-p2p` is the legacy spelling of the regular prism workload.
- `--far-field-model` selects `point` (the default above) or `exact`. `exact`
  restores the `UniformFmmOptions` default of exact finite far-field models,
  which is what exercises the finite P2M and L2P moment operators, including
  the tetrahedron barycentric expansion. The near field is exact geometry
  either way. The CSV records the choice as `far_field_model`.
- `--p2p-packing` forces `canonical-aos`, `particle-row-soa`,
  `tensor-dictionary`, `leaf-block`, `cuda-bsr3`, or `point-geometry`
  (`auto` keeps the backend policy); an impossible combination fails at
  construction with its reason. `point-geometry` runs on the CPU and, since
  Phase 3B.5, on both CUDA backends.
- `--point-expansion` forces `precomputed` or `procedural` point P2M/L2P
  (`auto` keeps the measured policy; see `docs/backends.md`, "Point P2M and
  L2P execution").
- `--periodic` evaluates a fully periodic cubic cell equal to the root box.
- The CSV gains `source_geometry`, `target_geometry`, `irregular_bodies`,
  `body_fill`, `periodic`, `p2p_packing_requested`, the dictionary size and
  token width, the canonical/dictionary/near-field operator bytes,
  `cuda_p2p_geometry_bytes` (resident positions of the CUDA position-based
  P2P executor), `far_field_model`, `point_expansion_requested`, the resolved
  `p2m_execution` / `l2p_execution`, and `p2m_operator_bytes` /
  `l2p_operator_bytes`. Every column is read by name, so the added column does
  not migrate an existing reader.

`benchmarks/run_p2p_packing_matrix.py --binary <benchmark_uniform_fmm>
--output <csv> --suite {periodic,finite,cuda-finite,crossover,all}` runs the
matrix used by the Phase-3 P2P unification study (periodic point geometry
versus SoA rows, CPU rows versus dictionary on regular and irregular prism and
tetrahedron grids, every CUDA packing on finite bodies, and the
CudaPartial/CudaFull crossover cases) and collects one row per case.

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
spherical `p=6`, FP32, rectangular-prism plan twice and emits `CACHE_BENCH` CSV
rows for cold and warm setup, with normalization, tree, lookup/hash/load,
analytical operator, backend-packing, CUDA-upload, and byte-count columns.
Backends are `portable`, `onemkl`, and `cuda-full`.

### Cold construction matrix

Construction and repeated evaluation are measured separately.
`benchmarks/run_construction_matrix.py --binary <benchmark_uniform_fmm>
--output <csv> --preset {quick,baseline,full}` sweeps geometry (random and
lattice points, regular and irregular prisms, regular and irregular
tetrahedra), size, expansion order, precision, and backend, and records the
per-phase `StaticPlanStatistics` timings with the plan's byte accounting. It
sets `CDFMM_DISABLE_CACHE` for every row, so each one is a cold build, and
keeps an evaluation median per row as the regression guard. Rows are appended
as they complete and an existing CSV is resumed, so an interrupted sweep can
be restarted and an expensive row skipped without losing the matrix.

The near-field stage is reported as `p2p_tensor_plan` and decomposed into
`p2p_interaction_setup`, `p2p_canonical_operator` and `p2p_derived_packing`;
`precision_conversion` times the FP64-to-FP32 conversion. A warm
geometry-cache hit records `p2p_derived_packing` and `precision_conversion`,
so the cost it still pays is attributable rather than visible only in the
total; the construction timers themselves stay at zero calls on a hit, which
is how a warm plan is distinguished from a rebuilt one.

### Dense all-to-all construction matrix

`DenseDirectPlan` is the exact dense baseline, and
`benchmarks/benchmark_dense_direct_construction` measures its setup and its
repeated evaluation as separate quantities. `CudaDenseDirectPlan` builds the
same host plan and retains only its device copy, so the two share one
construction path; the portable CPU and oneMKL backends share it as well and
differ only in `evaluate()`. Construction is therefore measured once per
geometry, not once per backend.

Setup is decomposed with the internal construction records into validation,
matrix allocation, prepared finite geometry, exact classification, tensor
build and materialisation, and on CUDA additionally host build, context
creation, device allocation and upload. The CSV also carries retained matrix
bytes, transient class-map and unique-tensor bytes, persistent device bytes
and the process peak resident set, with ns per pair for both halves so a setup
cost can be amortised over K evaluations.

`precision_conversion_seconds` is always zero and is kept as a column to say
so: an FP32 plan quantises each tensor at the point of store, so no complete
FP64 matrix is ever materialised and there is no separate conversion pass.
When classification is abandoned the build and the scatter are one fused loop,
reported entirely under `tensor_build_seconds` with `materialisation_seconds`
zero; a row is fused exactly when `classified` is 0. Separating them there
would need a complete `PairTensor` array, which at forty-eight bytes a pair is
larger than the matrices themselves.

Seven workloads bracket exact redundancy, and the choice matters more than it
looks. `lattice` shares one body record on a uniform grid and reaches
thirty- to eightyfold reuse; `lattice-irregular` and `random` give every body
its own record and reach none at all, because in an all-to-all plan every pair
holds a unique pair of records. `refined` replaces one octant's cells by their
eight children, and `anisotropic` stretches the spacing and the bodies; both
land between the extremes, and `refined-anisotropic` at about fivefold.
Anisotropy is not cosmetic: the far-separation switch keys on a body's
circumradius, so elongating a body moves the boundary between the analytical
surface integrals and the quadrature, and a spacing that is not exactly
representable makes equal index differences round to different displacement
bits, which raises the distinct-operator count severalfold. A measurement
taken only on an isotropic unit lattice overstates what exact reuse is worth.

`--probe-redundancy` counts distinct exact operator inputs with the production
key and no sampling gate, so a geometry's redundancy is measured rather than
assumed. `--checksum` hashes the six matrices, which is how a construction
change is held to bitwise equality rather than to a tolerance.

`benchmarks/run_dense_construction_matrix.py --binary <binary> --output <csv>
--preset {quick,geometry-matrix,sizes,asymmetric,threads,anisotropy}` drives
the sweep one process per row, so a row's peak resident set belongs to that
row alone and a row that exhausts memory cannot take the sweep with it. It
resumes an existing CSV and refuses a row whose matrices exceed
`--max-matrix-bytes`, because dense storage is `6 * Ns * Nt * scalar_bytes`
and reaches 1.6 GB at `Ns = Nt = 4096` in FP64. Sources and targets are
generated independently, so `--targets` differing from `--sources` is an
ordinary row; without an identity map the two sets are placed half a cell
apart, since a point source has no self field and an unmapped coincident
point pair is a singular configuration rather than a measurement.

### Internal cross-backend regression matrix (Phase 3D)

`benchmarks/run_phase3d_regression.py --binary <benchmark_uniform_fmm>
--output <csv> --suite {regression,policy,all}` is the internal
policy-validation and regression matrix, and
`benchmarks/analyse_phase3d_regression.py --input <csv> --view
{policy,matrix,both}` summarises it. The `regression` suite runs the workload
classes -- random points, point lattices at low and high occupancy, regular
prism and tetrahedron lattices with and without the layout hint, an irregular
finite control -- under the automatic policy on every available backend; the
`policy` suite runs the automatic policy against the credible forced
alternatives. Each row records the resolved representation (`p2p_packing`,
`p2m_execution`, `l2p_execution`, `static_multiply_backend`) beside repeated
evaluation, its phases, and the retained host and device bytes, so a stale
production rule is visible without reading the source.

Each row also records `comparison_group` and `comparison_variant`. The group
is everything two rows must share before a ratio between them means anything
-- workload class, geometry, lattice or irregular placement, size and
precision -- and the variant is the single axis under review. The analyser
groups on those columns rather than on the display name, whose token
positions shift whenever a case family gains or loses a dimension; for a CSV
written before the columns existed it reconstructs the grouping from the
driver's own case generators, and prints a case neither source recognises
alone rather than comparing it against a guess.

A run is fresh by default and strict: a case that cannot run, or that yields
no data row, names itself and fails the run with a non-zero exit, and the row
count is checked against the number of cases the selected suite and backend
matrix define. `--allow-failures` only moves where the run stops -- it
attempts the remaining cases rather than aborting at the first -- and still
exits non-zero on an incomplete matrix; it is for exploration and must not
produce a retained baseline. `--resume` reuses the per-case CSVs of an
interrupted run only when the scratch directory's `session.json` matches --
revision, binary path and SHA-256, suite and filter, evaluation, warm-up,
sample and thread counts, backend availability and the column set -- so
"same-session, same-binary" is a checked property rather than a convention.
A mismatch is refused and names the differing field.

Two rules for reading it. The comparison number for a rule that governs one
stage is **that stage**, not the total: a P2M/L2P rule is invisible inside an
M2L-dominated evaluation. And this driver leaves the caches enabled, so its
`fmm_setup_seconds` is a warm number that catches only a gross setup
regression -- cold construction is `run_construction_matrix.py`.

`benchmarks/run_phase3d_startup.py --binary <binary> --output <csv>` splits
startup into `cold-no-cache` (`CDFMM_DISABLE_CACHE`), `cold-write` (an empty
cache directory, so the build also pays the write) and `warm-hit` (the same
directory again), keeping the per-stage timings. It selects a cache directory
and changes no cache format or key. The difference between the last two is
what a returning caller saves; the stages still charged on a warm hit are what
is left to optimise.

`benchmarks/baselines/phase3d/` retains the accepted-HEAD output of all three,
with a README stating what it is. It is an engineering regression baseline and
explicitly not an Article1 benchmark: its numbers must not enter the article,
become publication figures, or be compared with external frameworks.

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
