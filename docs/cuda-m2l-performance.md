# CUDA M2L performance report

## Decision

The production CUDA M2L path uses one shared deterministic target-row executor
for `cuda-partial` and `cuda-full`. It derives compact active-row metadata from
the canonical `StaticM2LPlan`, pre-scales each resident multipole coefficient
once, and applies local scaling after the complete interaction/alpha reduction.
The canonical matrices, transfer-class IDs, interaction multiset, coefficient
ordering, FP64 precision, and backend transfer boundaries are unchanged.

The optimised executor is retained because every acceptance case improved by
at least 25.0% in median complete repeated-evaluation time. No order dispatch,
cuBLAS linkage, class-grouped staging, atomic scatter, or evaluation-time
allocation is retained. The simpler grouped alternatives were not promoted to
production after the target-row candidate cleared the 5% gate across orders 4,
6, and 8 without a regression.

## Measurement environment

- Date: 2026-08-17
- Baseline commit: `df9d36259771b1938896174827a02db0a5d7fc5e`
- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0, 32,607 MiB
- Driver: 595.84
- CUDA compiler/toolkit: 13.3.73
- Host compiler: GNU 14.4.0
- Build: `RelWithDebInfo`, native CUDA architecture, OpenMP, oneMKL, NVTX
- Runtime: `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=1`
- Sampling: two warm-ups, 20 evaluations per sample, five samples
- Launch: 256 threads per target-row block on this device

The primary metric is the median complete repeated-evaluation time. M2L is the
CUDA-event parent duration and includes the pre-scaling kernel. Raw CSV,
`.nsys-rep`, SQLite, and attempted `.ncu-rep` files are stored below the ignored
`profiling_results/cuda_m2l_optimisation/` directory.

## Results

| N | Order | Depth | Backend | Baseline eval (ms) | Final eval (ms) | Improvement | Baseline M2L (ms) | Final M2L (ms) |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 10,000 | 4 | 3 | cuda-partial | 1.539 | 0.958 | 37.8% | 1.119 | 0.658 |
| 10,000 | 4 | 3 | cuda-full | 1.185 | 0.889 | 25.0% | 1.059 | 0.604 |
| 50,000 | 4 | 4 | cuda-partial | 7.454 | 4.524 | 39.3% | 4.472 | 1.746 |
| 50,000 | 4 | 4 | cuda-full | 5.209 | 2.577 | 50.5% | 4.559 | 1.951 |
| 50,000 | 6 | 4 | cuda-partial | 28.687 | 15.049 | 47.5% | 20.136 | 7.442 |
| 50,000 | 6 | 4 | cuda-full | 20.432 | 7.867 | 61.5% | 19.468 | 6.894 |
| 100,000 | 6 | 5 | cuda-partial | 162.431 | 71.569 | 55.9% | 139.219 | 48.479 |
| 100,000 | 6 | 5 | cuda-full | 136.788 | 45.692 | 66.6% | 133.580 | 42.575 |
| 20,000 | 8 | 3 | cuda-partial | 15.045 | 8.334 | 44.6% | 10.010 | 3.425 |
| 20,000 | 8 | 3 | cuda-full | 10.935 | 4.231 | 61.3% | 10.455 | 3.729 |
| 50,000 | 8 | 4 | cuda-partial | 86.250 | 42.865 | 50.3% | 69.011 | 25.872 |
| 50,000 | 8 | 4 | cuda-full | 69.378 | 26.294 | 62.1% | 67.251 | 24.258 |

The pre-scaling pass is small relative to the multiply. In the six CUDA-full
cases it measured between 0.004 ms and 0.069 ms and is included in both M2L and
complete-evaluation time.

## Metadata and scratch

The CUDA executor no longer uploads one level integer per interaction. It
stores source and matrix IDs, one compact active-row record, and one level per
tree node. The old size below is reconstructed from the baseline layout; the
new size is reported directly by the candidate binary.

| N/order/depth | Interactions | Active rows | Old metadata (MiB) | New metadata (MiB) | Change | Scaled scratch (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| 10k/4/3 | 56,448 | 576 | 0.65 | 0.44 | -31.9% | 0.16 |
| 50k/4/4 | 640,472 | 4,671 | 7.35 | 4.98 | -32.3% | 1.25 |
| 50k/6/4 | 640,472 | 4,671 | 7.35 | 4.98 | -32.3% | 3.00 |
| 100k/6/5 | 5,154,372 | 33,719 | 59.13 | 39.98 | -32.4% | 24.00 |
| 20k/8/3 | 56,448 | 576 | 0.65 | 0.44 | -31.9% | 0.74 |
| 50k/8/4 | 640,472 | 4,671 | 7.35 | 4.98 | -32.3% | 5.89 |

Pre-scaled scratch is allocated once during plan construction only when the
required buffer fits both 10% of total VRAM and 25% of currently free VRAM.
Otherwise the same executor uses its deterministic no-scratch target-row
fallback. There are no allocations or global/device synchronisations during
evaluation.

## Profiler interpretation

Nsight Systems captured eleven consecutive N=50k, order-4, depth-4 CUDA-full
evaluations. M2L and P2P ran on distinct streams in all eleven. Median M2L/P2P
overlap was 1.479 ms, or 90.5% of the 1.653 ms median P2P duration. The M2L
kernel used 256-thread blocks and 40 registers per thread. CUDA-full still
transfers only changing moments H2D and final fields D2H; CUDA-partial retains
its multipole/local boundary.

Nsight Compute 2026.1.1 was invoked with `--set full`, but the driver rejected
hardware-counter access with `ERR_NVGPUCTRPERM`. Occupancy, SM/FP64 utilisation,
DRAM/L2 counters, instruction counts, and warp-stall counters are therefore not
reported. Enabling NVIDIA performance counters is an administrator policy
change and was deliberately not performed by this work.

## Reproduction

Build and run one acceptance case:

```console
conda activate cdfmm
module load cuda
cmake --fresh --preset profile-all
cmake --build --preset profile-all --target benchmark_uniform_fmm -j

OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 \
./build-profile-all/benchmarks/benchmark_uniform_fmm \
  --backend cuda-full \
  --sources 50000 --targets 50000 \
  --depth 4 --order 6 --threads 8 \
  --warmups 2 --evaluations 20 --samples 5 \
  --no-workload-comparison --no-direct \
  --output profiling_results/cuda_m2l_optimisation/final_n50000_p6_d4_full.csv
```

Capture the ten-evaluation scheduling trace:

```console
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output=profiling_results/cuda_m2l_optimisation/final_n50000_p4_d4_full \
  ./build-profile-all/benchmarks/benchmark_uniform_fmm \
    --backend cuda-full \
    --sources 50000 --targets 50000 \
    --depth 4 --order 4 --threads 8 \
    --profile
```

After an administrator enables GPU performance counters, collect one M2L
launch with:

```console
ncu --set full \
  --kernel-name-base function \
  --kernel-name regex:apply_scaled_m2l_rows_kernel \
  --launch-count 1 \
  --force-overwrite \
  --export profiling_results/cuda_m2l_optimisation/final_n50000_p6_d4_full \
  ./build-profile-all/benchmarks/benchmark_uniform_fmm \
    --backend cuda-full \
    --sources 50000 --targets 50000 \
    --depth 4 --order 6 --threads 8 \
    --profile
```

## Validation

- C++: 60/60 tests passed with the RTX 5090 available.
- Python: 81/81 tests passed against the CUDA-enabled extension.
- Isolated CUDA M2L: orders 2, 4, 6, and 8; depths 2, 3, and 5; 7,776
  coefficient assertions passed.
- `compute-sanitizer --tool memcheck`: zero errors for isolated M2L and a
  complete CUDA-full repeated-evaluation case.
- Existing complete-FMM tests cover repeated states, separate targets,
  source-point identities, empty geometry, transfer counts, and static uploads.
