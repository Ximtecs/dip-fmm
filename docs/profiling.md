# Profiling deterministic FMM workloads

## Purpose

Benchmarking measures **how fast** a workload is; profiling explains **why** it
takes that long. `benchmark_uniform_fmm --profile` uses the benchmark's normal
geometry, moment states, seed, backend, depth, order, thread controls, and
production evaluator. It changes only the measurement policy: unless explicitly
overridden, one warm-up, one sample, and one evaluation are run, while direct
validation and the 1-versus-10 workload comparison are disabled. `--direct` or
`--accuracy-targets N` explicitly restores validation.

Use an optimised build with symbols. CPU profiling does not require NVTX:

```console
cmake -S . -B build-profile -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCDFMM_BUILD_BENCHMARKS=ON -DCDFMM_ENABLE_NATIVE_ARCH=ON
cmake --build build-profile -j
```

For CUDA timelines, enable CUDA and the optional, compile-time NVTX ranges:

```console
cmake -S . -B build-cuda-profile -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCDFMM_BUILD_BENCHMARKS=ON -DCDFMM_ENABLE_CUDA=ON \
    -DCDFMM_ENABLE_NATIVE_ARCH=ON -DCDFMM_ENABLE_PROFILING=ON
cmake --build build-cuda-profile -j
```

With profiling disabled the range abstraction is an empty object: evaluation
makes no NVTX calls, allocations, registry lookups, or additional
synchronisation. Existing `EvaluationTimings` remain the lightweight numerical
phase report; profiler ranges are timeline annotations rather than new timers.

## CPU: perf

Hardware-counter summaries and sampled call stacks can use the same medium
case:

```console
perf stat build-profile/benchmarks/benchmark_uniform_fmm \
    --backend cpu-static-matrix --sources 50000 --targets 50000 \
    --depth 4 --order 6 --threads 8 --profile

perf record --call-graph dwarf \
    build-profile/benchmarks/benchmark_uniform_fmm \
    --backend cpu-static-matrix --sources 50000 --targets 50000 \
    --depth 4 --order 6 --threads 8 --profile
perf report
```

Inspect samples in P2M, M2M, M2L, L2L, L2P and P2P, memory stalls, cache
misses, branch behaviour, and OpenMP imbalance. `perf` is not a build or test
dependency.

## CUDA: NVIDIA Nsight

Nsight Systems shows NVTX phase hierarchy, CUDA streams, kernels, transfers,
launch gaps and synchronisation:

```console
nsys profile --trace=cuda,nvtx,osrt --sample=cpu \
    --output=cdfmm_cuda_full \
    build-cuda-profile/benchmarks/benchmark_uniform_fmm \
    --backend cuda-full --sources 50000 --targets 50000 \
    --depth 4 --order 6 --profile
```

The stable names begin with `cdfmm/evaluate`, with `cdfmm/far_field` and
`cdfmm/near_field` branches and operator-specific children. CUDA kernel names
and CUDA API rows distinguish execution from H2D/D2H traffic. In particular,
check that repeated CUDA-full evaluation transfers changing moments to the
device and the final field back, without transferring persistent geometry.

Use Nsight Compute only after Systems identifies a kernel of interest. Kernel
filtering avoids the very expensive full metric set being replayed for every
kernel:

```console
ncu --set full --kernel-name regex:.*m2l.* --launch-count 1 \
    build-cuda-profile/benchmarks/benchmark_uniform_fmm \
    --backend cuda-full --sources 50000 --targets 50000 \
    --depth 4 --order 6 --profile
```

Adjust the kernel-name expression for P2P or another production kernel. Inspect
DRAM throughput, L1/L2 behaviour, occupancy, register pressure, warp and
instruction efficiency, and arithmetic intensity. CUDA capture validation is
manual and requires suitable NVIDIA hardware and tools.

## Representative cases and near/far balance

Suggested source-point cases use the fixed default seed `314159`:

| Case | Particles | Order | Depth |
|---|---:|---:|---|
| small | 10,000 | 4 | adviser suggestion or 3 |
| medium | 50,000 | 6 | adviser suggestion or 4 |
| higher-order | 50,000 | 8 | adviser suggestion or 4 |
| large | 100,000 | 6 | adviser suggestion or 5 |

Prefer the [parameter adviser](parameter-selection.md) depth, but record an
explicit depth in every capture. For concurrent branches, compare the timeline
against

```text
T_eval ~= max(T_near, T_far) + T_overhead.
```

Near an advised depth, inspect whether `T_near ~= T_far`; traces are an
independent check of the heuristic, not evidence that it must be correct. Also
inspect the dominant phase, actual near/far overlap, memory bandwidth, GPU
occupancy, kernel launch overhead, synchronisation and all transfers.

## Reproducibility checklist

Record the Git commit, compiler and version, optimisation flags, CPU and GPU,
thread count, backend, particle count, depth, order, seed, and profiler/tool
version. The profiling-mode configuration summary prints the effective workload
values and whether NVTX was compiled in.
