# Pre-refactor baseline (`v0.1.0`)

This is a compact, deterministic reference record for comparison during the
architecture refactor. It is not a publication benchmark and should not be
used for cross-machine performance claims.

## Measurement identity

- Source commit: `01ad4cd9635d9bdff98a28d9ac0ca6f4869a5c9f`
- Host: Intel Core i9-14900KF, 32 logical CPUs (24 physical cores reported)
- OS/kernel: Linux 7.0.0-31-generic, x86_64
- Compiler: Conda-forge GNU g++ 15.3.0
- Build: Release, C++20, native architecture, OpenMP, CUDA compiled, oneMKL
  compiled (`2026.0.1`)
- CUDA toolkit: 13.3; no CUDA device was available during this run
- Cache policy: disabled for the Python geometry/timing smoke tests; the
  point benchmark used the repository cache and records setup separately
- Python geometry samples used deterministic `numpy.random.default_rng`
  seeds and ten repeated evaluations after one warm-up evaluation.

## Point benchmark

Command shape:

```console
build-release-validation/benchmarks/benchmark_uniform_fmm --profile \
  --sources 512 --targets 512 --order 4 --depth 3 --threads 4 \
  --backend <backend> --precision float32 --output <csv>
```

All rows use seed `314159`, spherical order 4, depth 3 for FMM rows, ten
evaluations, one sample, and 4 OpenMP threads. `evaluation_median` is the
median seconds per evaluation; setup is one-time plan construction.

| Backend | Geometry | Setup (s) | Median/eval (ms) | Evaluations/s | Static plan (bytes) | Nodes | Occupied source leaves | M2L translations | Near-field pairs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cpu-direct` | point dipole, direct all-to-all | 0.00000005 | 0.560416 | 1784.39 | 0 | 0 | 0 | 0 | 261632 |
| `cpu-static-matrix` | point dipole, spherical FMM | 0.033555 | 5.08655 | 196.597 | 3105596 | 585 | 314 | 23640 | 11800 |
| `cpu-static-matrix-mkl` | point dipole, spherical FMM | 0.0225651 | 4.51621 | 221.424 | 8211836 | 585 | 314 | 23640 | 11800 |

The direct row has no tree or static plan. CUDA status from this executable was
`compiled=1`, `available=0`, `direct=0`, `partial=0`, `full=0`; oneMKL was
available.

## Geometry and adaptive smoke samples

These modest-N samples use the Python bindings from the same source commit and
FP64 portable execution. They exercise the persistent dense geometry plan and
the prebuilt adaptive-topology path; they are included to anchor functionality,
not to compare against the point benchmark's workload.

| Case | Deterministic setup | Setup (s) | Median/eval (us) | Readily available memory | Result |
|---|---|---:|---:|---:|---|
| Dense `prism -> prism` | 8 source/target representatives, common `0.12 x 0.10 x 0.08` prism, explicit identity map, 10 evaluations | 0.016919448 | 1.094 | 3072-byte six-tensor plan | finite, including finite self terms |
| Dense `tetrahedron -> tetrahedron` | 8 source/target representatives, common four-vertex tetrahedron, explicit identity map, 10 evaluations | 0.019572416 | 0.535 | 3072-byte six-tensor plan | finite, including finite self terms |
| `AdaptiveTree` + static FMM | 64 clustered points, seed `20260909`, leaf capacity 8, maximum depth 3, Cartesian FP64 order 4, CPU static, explicit identities | tree wall 0.008536769; FMM setup 0.087753865 | 1174.909 | topology 68080 B; static plan 3765338 B | finite field output |

The adaptive sample produced 28 nodes, 23 leaves, reached level 3, 136 M2L
interactions, and 309 P2P leaf records. The topology was passed directly to
the same static FMM plan used by the uniform path.

## Validation caveat at capture time

The rebuilt C++ release-validation binary passed all 175 CTest cases; three
CUDA-runtime tests were skipped because no device was available. During the
initial audit, the Python memory test found a 21,504-byte accounting mismatch:
the notebook estimate was `1,099,986` bytes while the constructed plan reported
`1,121,490` bytes because the retained M2M/L2L maps were omitted from the
estimate. This was corrected during Phase 0 by aligning the estimate with the
plan's actual ownership and disabling cache-dependent assumptions in the
comparison test. The post-correction focused Python geometry and adaptive
checks passed; final full-suite counts belong in the release report.
