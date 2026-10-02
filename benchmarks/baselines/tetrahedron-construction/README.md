# Kuhn-mesh construction optimisation, 2 October 2026

Engineering measurements, not Article1 or jaxFMM comparison results. The
fixture is the jaxFMM campaign's Kuhn mesh: six low-to-high diagonal simplices
per Cartesian cell, unit spacing, vertices relative to each centroid, with
the six coordinate permutations in campaign order. It does not use the
synthetic tetrahedron lattice of `benchmark_uniform_fmm`.

Starting source: `aa9d75fc37a35fa3846619bc0e13a99d4f117b67` on
`refactor/architecture-v0.2`. Portable baseline: a copy of the newly built,
unchanged Python extension under `/tmp/dipfmm-construction-baseline`.
CUDA baseline: the existing Article1 runtime build of that commit. Final
builds: `dev` and `cuda`, with benchmarks enabled, examples disabled,
warnings as errors and the cdfmm environment's Python 3.11. CUDA installation
was disabled; no installed or Article1 runtime binary was replaced.

Host compiler: conda-forge GCC 15.3.0; CUDA compiler: nvcc 13.3.73; GPU:
NVIDIA GeForce RTX 5090. Release, native architecture, OpenMP and LTO enabled,
fast math disabled. Every launch used `OMP_NUM_THREADS=8` and
`taskset -c 0,2,4,6,8,10,12,14`. The original baseline driver recorded the
calling thread's affinity after OpenMP had pinned it, so some baseline JSON
files show `[0]`; that is not the worker team's launch mask. The final driver
captures the mask before construction.

Every sample disables **all** persistent caches, including the universal
operator bank. Exact finite source and target models, spherical basis,
general spatial layout, automatic P2P packing, detailed construction timing.
The external wall clock surrounds the complete constructor. Geometry and
moment generation, evaluation and snapshot writing are outside that clock.
Baseline rows are one cold sample; final rows are medians of three cold
samples. Some baseline runs overlapped compilation, so these are diagnostic
speedups rather than precision estimates of runtime ratios.

## Same-fixture results

750 bodies (`n=5`), depth 2:

| Backend / precision / order | Cold wall before → after | P2M before → after | L2P before → after | Relative L2 field change |
|---|---:|---:|---:|---:|
| CPU FP32 p10 | 171.33 s → 0.447 s | 25.475 s → 1.639 ms | 59.655 s → 2.320 ms | 0, bit-identical |
| CUDA-full FP32 p10 | 162.72 s → 0.465 s | 19.956 s → 1.675 ms | 48.133 s → 2.367 ms | 6.47e-8 |
| CPU FP64 p6 | 2.911 s → 0.364 s | see raw rows | see raw rows | 5.04e-17 |
| CUDA-full FP64 p6 | 3.146 s → 0.375 s | see raw rows | see raw rows | 1.25e-16 |

The CPU order-10 FP32 far-field, P2P and total fields are all bit-identical.
The FP64 far-field changes are about `7e-16` relative L2; total fields change
less because the near field dominates. CUDA baseline and final builds also
show rounding-level P2P field differences; no P2P formula or executor was
changed. Coefficient summation order differs from the old implementation,
so bit identity is not a general promise.

The earlier retained Article1 construction profile reported 68.40 s total,
including 19.62 s P2M, 48.01 s L2P and 0.44 s P2P. Those endpoint timings
reproduce. Its reported universal-bank time was only 0.069 s; our entirely
cold baseline measured 80–94 s for that bank. The old profile's total must
therefore not be compared with our entirely cold total without this caveat.
The final changes include the scoped constant-time multi-index lookup and
Laplace recurrence already developed in `dc32d1e` and `abe6aa8` on
`perf/universal-operator-build`; the rest of that branch was not imported.
The final p10 bank builds in about 0.064 s.

## Larger meshes and higher orders

CUDA-full FP32, three cold samples per configuration:

| Mesh / order / depth | Bodies | Cold wall | P2M | L2P | P2P | Universal bank |
|---|---:|---:|---:|---:|---:|---:|
| n8 / p10 / d3 | 3,072 | 0.199 s | 0.608 ms | 0.783 ms | 0.0666 s | 0.0644 s |
| n10 / p10 / d3 | 6,000 | 2.038 s | 10.19 ms | 11.79 ms | 1.763 s | 0.0644 s |
| n5 / p15 / d2 | 750 | 1.397 s | 10.14 ms | 17.70 ms | 0.357 s | 0.882 s |
| n5 / p20 / d2 | 750 | 7.208 s | 31.77 ms | 52.67 ms | 0.354 s | 6.412 s |

Exact-bit geometry reuse is much stronger on the dyadic `n=8` mesh than on
`n=5` or `n=10`; size alone does not predict cold construction time.
P2P is the remaining bottleneck on the larger order-10 mesh. At order 20,
assembling the universal translation matrices dominates instead. No geometry
snapping, approximate near-field equivalence, or P2P policy change was made.

`summary.csv` collects the raw JSON medians and comparison metrics. The NPZ
files retain the deterministic random-state far-field, near-field and total
field snapshots. Higher-order/larger rows are construction and execution
smoke checks; they are not accuracy comparisons against the old solver.

## Reproduction

Run the driver once for each build; `--compare` uses the previous JSON and its
adjacent NPZ file. Choose new output names to preserve existing evidence.

```bash
module load cuda
OMP_NUM_THREADS=8 taskset -c 0,2,4,6,8,10,12,14 \
  /home/mihaa/.conda/envs/cdfmm/bin/python \
  benchmarks/benchmark_kuhn_construction.py \
  --build build-cuda --backend cuda_full --grid 5 --depth 2 --order 10 \
  --precision float32 --repetitions 3 \
  --compare benchmarks/baselines/tetrahedron-construction/kuhn-before-cuda-p10.json \
  --output /tmp/kuhn-construction-recheck.json
```

Validation: portable and CUDA CTest 265/265 (four portable optional skips,
one CUDA oneMKL skip). Focused Python suites import `PYTHONPATH=build` or
`build-cuda`: 41 passed / 7 skipped and 46 passed / 2 skipped. New tests
compare moments and complete Cartesian/spherical endpoint operators with the
unchanged scalar builder through order 10, analytic simplex moments and
independent Duffy volume quadrature through order 20, and Laplace derivatives
with the old Taylor-jet construction through orders 20 and 30. GCC 13,
oneMKL, Fortran and Windows were not exercised in this session.
