# Validation and testing

## Manual CUDA validation

CUDA tests are intentionally not executed by GitHub Actions. They must be run
manually on an NVIDIA CUDA-capable system:

```console
cmake --fresh --preset cuda
cmake --build --preset cuda
ctest --preset cuda
python -m pytest python_tests -v
```

Run the fixed-geometry problem-size and expansion-order sweep (including
`cuda-m2l` only when capability probing succeeds) with:

```console
cmake --build --preset cuda --target benchmark_uniform_fmm
python benchmarks/run_benchmarks.py --profile standard
```

For a focused repeated-evaluation comparison, use identical arguments and
moment generation for each backend:

```console
build-cuda/benchmarks/benchmark_uniform_fmm --backend cpu-static-matrix --sources 20000 --targets 20000 --depth 4 --order 6 --evaluations 20 --samples 5 --output cpu.csv
build-cuda/benchmarks/benchmark_uniform_fmm --backend cuda-m2l --sources 20000 --targets 20000 --depth 4 --order 6 --evaluations 20 --samples 5 --output cuda-m2l.csv
```

The C++ suite selects CPU reference/static backends explicitly and compares
`cuda_direct_p2p_reference` with the CPU direct reference whenever
`cuda_direct_available()` is true. Manual CUDA tests also compare `CudaM2L`
with `CpuStatic`; the full-FMM capability remains false.

The hybrid path performs CPU moment permutation, P2M and M2M; GPU gather,
static M2L cuBLAS multiplication and scatter; then CPU L2L, L2P and near-field
P2P. Repeated evaluation transfers packed multipole coefficients H2D and raw
M2L local coefficients D2H. Static matrices and interaction metadata are
uploaded only during plan construction, and all levels share those two dynamic
transfers.

The CUDA direct convenience evaluator is an O(N^2) validation reference, not
an FMM backend. It uploads geometry when called and must not be used to claim a
moment-only repeated far-field transfer contract.

`EvaluationTimings` records CUDA H2D, gather, multiply, scatter, and D2H durations
with CUDA events. Their sum excludes host packing and result-copy overhead;
compare it with the complete caller wall time in `total` when diagnosing
launch, synchronisation, and other host-side costs.

The project validates small mathematical pieces analytically before composing
them.  Unit tests cover multi-index counts and ordering, Taylor-jet algebra,
Laplace derivatives, output modes, axial and transverse direct dipoles, tree
topology, sorting, ranges, and interaction lists.

The default exact static M2L backend is checked against the independently
retained `m2l_add()` traversal. Select that validation path explicitly with
`UniformFmmOptions::m2l_backend = M2LBackend::Reference`; static-plan matrices,
interaction maps, and scratch storage are constructed only once.

Operator accuracy tests compare these paths with direct P2P:

- **P2M + M2P** checks source coefficients and kernel differentiation;
- **P2M + M2L + L2P** checks conversion to and evaluation of local expansions;
- **P2M + M2M + M2P** checks child-to-parent translation;
- direct pair tests compare against analytic dipole configurations.

The reusable `direct_p2p_reference` helper evaluates multiple targets, while
`compute_error_metrics` reports mean, RMS, and maximum relative errors plus
mean and maximum absolute errors.  The relative denominator is floored for
near-zero reference fields.

Run the C++ suite after configuring with tests enabled:

```console
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run installed Python bindings against their tests with:

```console
python -m pytest python_tests -v
```

Complete-traversal tests compare `UniformFmm` with direct P2P across expansion
orders and check the convergence trend. They also cover depth-zero direct
evaluation, target unsorting, repeated magnetic states, empty complete-tree
boxes, output modes, and explicit index-based source self exclusion. Python
tests exercise the same compiled evaluator, array shapes, ordering, and reset
behaviour.

## Parallel correctness and benchmark accuracy

When OpenMP is enabled, automated tests compare deterministic complete-FMM
results obtained with one and multiple threads and verify repeated-evaluation
state reset. The benchmark optionally evaluates the same state with parallel
direct P2P and writes mean, RMS, and maximum relative field errors beside the
phase timings. Large automated cases disable the quadratic direct reference
explicitly rather than reporting a misleading zero error.
# Static list1 P2P validation

The static CPU path is checked against the independent geometry-recomputing
list1 traversal for coincident source/target populations, independent target
sets, explicit self maps, multiple neighbours, and empty rows. The CUDA test is
manual and compares the persistent custom six-value block executor with the CPU
static result when a device is available. Geometry and tensor values are
uploaded once; evaluation traffic contains moments, identities, and fields.

MagTense's six separate component matrices provide the architectural reference.
Here a single row structure and one compact block kernel reuse indices across
all components, use one launch, and load six rather than nine tensor values.
A generic 3-by-3 BSR form would store nine values and was therefore not retained
without platform benchmark evidence that compensates for its extra bandwidth.
