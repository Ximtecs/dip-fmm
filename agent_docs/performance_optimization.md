# Performance optimization log

Phase 3 of the `v0.2` roadmap. Each section records one optimization task:
hardware, the fixed starting SHA, workloads, methodology, baseline, profiler
findings, accepted changes with measured evidence, rejected experiments, final
numbers, and the bottlenecks that remain. Raw `.nsys-rep`/`.ncu-rep` captures
are not committed.

## GPU evaluation optimization (3A)

### Environment

- Machine: `mihaa-workstation`, Ubuntu 26.04.1, kernel 7.0.0-31, Intel
  i9-14900KF (8 P-cores + 16 E-cores; the conda environment pins
  `OMP_NUM_THREADS=8`, `OMP_PLACES={0},{2},...,{14}`, `OMP_PROC_BIND=close`,
  i.e. one thread per P-core), 184 GB RAM.
- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0 (`sm_120`), 170 SMs,
  32 GB GDDR7, 96 MB L2, driver 595.84 (CUDA 13.2 runtime).
- Toolchain (conda env `cdfmm`): nvcc 13.3.73 with host `g++` 15.3.0,
  CMake 4.4.3, Ninja 1.13.2. Both baseline and candidate builds use
  `Release`, `CMAKE_CUDA_ARCHITECTURES=native` (`sm_120`), LTO on,
  `-march=native`, `CDFMM_ENABLE_PROFILING=ON` (NVTX ranges) and
  `-lineinfo`; no `-G`, no fast-math, oneMKL off.
- Profilers: Nsight Systems 2025.6.3 (from `/usr/local/cuda-13.2`) works for
  CUDA/NVTX tracing. Nsight Compute 2026.1.1 is installed but **every metric
  collection fails with `ERR_NVGPUCTRPERM`** (GPU performance counters are
  restricted to admin users; `perf_event_paranoid=4` also disables nsys CPU
  sampling). No driver/kernel setting was changed, so kernel classification
  below relies on nsys timelines, CUDA-event phase timings, arithmetic
  intensity estimates, and controlled experiments instead of hardware
  counters. Compute Sanitizer 2026.1.1 works.
- Clocks could not be locked (`nvidia-smi -lgc` needs root); the CPU governor
  is `powersave`. All numbers are medians of 7 samples x 20 evaluations after
  3 warm-up evaluations, and the same binary re-run gives the noise floor
  recorded below.

### Starting point

`GPU_PERF_BASELINE = 293144bf248423d1fd273f87e1298e2e2820a348`
(`docs: record public API and packaging cleanup, close Phase 2`) on
`refactor/architecture-v0.2`, built in a detached worktree with the identical
configuration.

### Workloads

`benchmark_uniform_fmm`, spherical basis, default seed 314159, fixed
geometry with 23 changing moment states, fixed identity map
(`fixed_target_source_indices`), field output, `--no-direct
--no-workload-comparison --warmups 3 --evaluations 20 --samples 7
--accuracy-targets 128` (the accuracy columns compare 128 sampled targets
against the exact point-dipole reference):

| Case | Geometry | N | Order | Depth |
|---|---|---:|---:|---:|
| S | random points | 10,000 | 4 | 3 |
| M | random points | 50,000 | 6 | 4 |
| H | random points | 50,000 | 8 | 4 |
| L | random points | 100,000 | 6 | 4 |
| XL | random points | 200,000 | 6 | 5 |
| F | 16^3 regular grid of cubes, exact cuboid P2P | 4,096 | 6 | 3 |

Each case runs `cuda-partial` and `cuda-full` in `float32` and `float64`.
The originally planned 32^3 finite case was dropped: its 56M exact
cuboid pair tensors take about 20 minutes per construction, so it measures
construction, not repeated evaluation. The F accuracy column is not
meaningful (the sampled reference is the point-dipole formula while the
near field uses exact cuboid tensors) and is ignored.

With the fixed identity map and point geometry, both CUDA backends select the
cuSPARSE BSR(3) P2P packing by default; the finite case also selects BSR.

### Baseline (293144bf)

Device phase timings are CUDA-event intervals on the far-field stream; the
`M2M` column of `cuda-full` is inflated by the concurrent P2P kernel sharing
the GPU (both streams run simultaneously), see the Nsight Systems findings.

| Case | Backend | Precision | eval median [us] | P2P dev [us] | M2L dev [us] | P2M [us] | M2M [us] | L2L [us] | L2P [us] | H2D [us] | D2H [us] | max rel err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S (10k, p=4, d=3) | cuda-full | float32 | 485 | 170 | 261 | 16 | 85 | 8 | 10 | 10 | 6 | 8.9e-03 |
| S (10k, p=4, d=3) | cuda-full | float64 | 760 | 269 | 481 | 18 | 154 | 8 | 11 | 15 | 8 | 8.9e-03 |
| S (10k, p=4, d=3) | cuda-partial | float32 | 619 | 157 | 233 | 68 | 103 | 80 | 21 | 13 | 17 | 8.9e-03 |
| S (10k, p=4, d=3) | cuda-partial | float64 | 915 | 263 | 464 | 79 | 126 | 156 | 36 | 22 | 29 | 8.9e-03 |
| M (50k, p=6, d=4) | cuda-full | float32 | 2397 | 508 | 1181 | 71 | 478 | 70 | 96 | 42 | 28 | 6.5e-03 |
| M (50k, p=6, d=4) | cuda-full | float64 | 4275 | 902 | 2717 | 66 | 870 | 88 | 88 | 74 | 53 | 6.5e-03 |
| M (50k, p=6, d=4) | cuda-partial | float32 | 8138 | 431 | 1151 | 1390 | 2193 | 1956 | 823 | 116 | 144 | 6.5e-03 |
| M (50k, p=6, d=4) | cuda-partial | float64 | 11196 | 786 | 2726 | 1857 | 2148 | 2075 | 1441 | 230 | 251 | 6.5e-03 |
| H (50k, p=8, d=4) | cuda-full | float32 | 4986 | 550 | 3271 | 103 | 663 | 191 | 235 | 42 | 28 | 1.6e-03 |
| H (50k, p=8, d=4) | cuda-full | float64 | 8694 | 1004 | 6517 | 129 | 1132 | 236 | 218 | 74 | 53 | 1.6e-03 |
| H (50k, p=8, d=4) | cuda-partial | float32 | 21206 | 431 | 3229 | 2895 | 6227 | 6177 | 1504 | 172 | 190 | 1.6e-03 |
| H (50k, p=8, d=4) | cuda-partial | float64 | 27013 | 783 | 6395 | 3776 | 6197 | 6216 | 2642 | 311 | 328 | 1.6e-03 |
| L (100k, p=6, d=4) | cuda-full | float32 | 4307 | 1766 | 1227 | 110 | 1671 | 70 | 220 | 74 | 52 | 5.8e-03 |
| L (100k, p=6, d=4) | cuda-full | float64 | 7146 | 3214 | 2718 | 151 | 3067 | 88 | 228 | 144 | 103 | 5.8e-03 |
| L (100k, p=6, d=4) | cuda-partial | float32 | 10990 | 1641 | 1175 | 2786 | 2229 | 1984 | 1735 | 166 | 181 | 5.8e-03 |
| L (100k, p=6, d=4) | cuda-partial | float64 | 16510 | 3030 | 2748 | 4311 | 2385 | 2278 | 3121 | 328 | 302 | 5.8e-03 |
| XL (200k, p=6, d=5) | cuda-full | float32 | 15504 | 948 | 10479 | 1090 | 770 | 594 | 513 | 148 | 104 | 1.3e-02 |
| XL (200k, p=6, d=5) | cuda-full | float64 | 24402 | 1755 | 18378 | 1910 | 963 | 735 | 518 | 283 | 205 | 1.3e-02 |
| XL (200k, p=6, d=5) | cuda-partial | float32 | 59503 | 907 | 10462 | 7017 | 17124 | 15370 | 4131 | 684 | 700 | 1.3e-02 |
| XL (200k, p=6, d=5) | cuda-partial | float64 | 76299 | 1673 | 18307 | 9293 | 16905 | 16365 | 6581 | 1373 | 1370 | 1.3e-02 |
| F (4096 cubes 16^3 grid, exact cuboid P2P, p=6, d=3) | cuda-full | float32 | 512 | 39 | 408 | 6 | 23 | 10 | 10 | 8 | 5 | 5.6e+01 |
| F (4096 cubes 16^3 grid, exact cuboid P2P, p=6, d=3) | cuda-full | float64 | 925 | 49 | 820 | 7 | 29 | 14 | 10 | 9 | 6 | 5.6e+01 |
| F (4096 cubes 16^3 grid, exact cuboid P2P, p=6, d=3) | cuda-partial | float32 | 1142 | 26 | 399 | 50 | 329 | 257 | 19 | 13 | 20 | 5.6e+01 |
| F (4096 cubes 16^3 grid, exact cuboid P2P, p=6, d=3) | cuda-partial | float64 | 1668 | 32 | 798 | 57 | 326 | 329 | 28 | 22 | 37 | 5.6e+01 |

Noise floor (same baseline binary run twice, evaluation medians): `cuda-full`
cases agree within +-0.2 %; `cuda-partial` cases within +-1 % except
L (100k) FP32/FP64 at -6.6 %/+8.9 %, because that path is dominated by
CPU-side P2M/M2M/L2L/L2P. Speedups below 1.02x on `cuda-full` or below
1.10x on `cuda-partial` are therefore not claimed.

### Nsight Systems findings (baseline)

Captures: `nsys profile --trace=cuda,nvtx,osrt` of `benchmark_uniform_fmm
--profile --evaluations 10` for S/M/L in both backends and precisions plus
XL and F in FP32. Per-evaluation numbers below come from the GPU trace of
the last traced evaluation (`cuda_gpu_trace`), so construction-time
uploads/allocations are excluded.

`cuda-full`, M FP32 (2375 us per evaluation from the benchmark clock):

| Device op (far-field stream unless noted) | Duration [us] |
|---|---:|
| H2D moments (600 KB, pinned) | 34 |
| `permute_moments_kernel` | 2 |
| memset multipoles + P2M `apply_entries_kernel` | 48 |
| P2P `cusparse bsrmv_tiny_core` (near-field stream) | 466 |
| M2M `apply_shared_translation_kernel` x4 levels | 88 (stalled behind P2P) + 17 + 13 + 13 |
| memset locals + `scale_m2l_multipoles_kernel` | 2 |
| M2L `apply_scaled_m2l_rows_kernel` | 1133 |
| L2L `apply_shared_translation_kernel` x4 levels | 17 + 13 + 14 + 23 |
| memset far fields + L2P `apply_entries_kernel` | 93 |
| `combine_order_kernel` | 2 |
| D2H fields (600 KB, pinned) | 23 |
| device span / union busy / idle gaps | 1928 / 1907 / 21 |
| host turnaround (previous D2H end to next first op) | 417 |

Answers to the profiling questions:

1. Dominant kernels: the M2L target-row kernel (59 % of GPU time at M, 92 %
   at XL where depth 5 makes 4.9M translations), then cuSPARSE BSR P2P
   (24 % at M, 58 % at L where the near field is 65M pairs), then the
   COO-atomic P2M/L2P kernels (7 %) and the per-level M2M/L2L kernels
   (10 %, of which most is one launch stalled behind P2P).
2. P2P: 466 us at M FP32 (16.3M pairs, 587 MB of 9-value BSR blocks, i.e.
   about 1.3 TB/s including indices, roughly 70 % of DRAM peak); 1.64 ms at
   L FP32 (65M pairs, 2.35 GB, 1.47 TB/s). It is DRAM-bandwidth bound; the
   only lever is bytes per pair.
3. Far field: M2L 1133 us + P2M/L2P 139 us + M2M/L2L about 110 us at M FP32.
   FP32 M2L runs at 2.6 TFLOP/s (640k translations x 49^2 x 2), FP64 M2L at
   1.07 TFLOP/s which is about 65 % of the RTX 5090's FP64 peak, so FP64 M2L
   is compute-bound while FP32 M2L is bound by streaming each 49x49 matrix
   from L2 once per interaction.
4. Gaps between device ops: 16-40 us per evaluation in every case; launch
   overhead is not significant.
5. CPU/GPU overlap, `cuda-full`: the host issues all work and blocks in one
   `cudaEventSynchronize`; the next evaluation's host preparation cannot
   start until the previous D2H completes. Host turnaround is 84 us (S),
   417 us (M), 884 us (L), 1826 us (XL) FP32 and 336/791 us (M/L) FP64:
   12-20 % of each evaluation. It is the serial O(N) host passes (identity
   gather, moment scaling with nine divisions per source in FP32, copy into
   pinned memory, copy out of pinned memory, result conversion, plus a
   per-call `std::vector<FloatPotentialField>` allocation for FP32 plans
   evaluated through the FP64 API).
6. H2D/D2H: 34 + 23 us at M (600 KB each way at 18-26 GB/s pinned); 3 % or
   less everywhere. Unavoidable per evaluation.
7. Repeated transfers: none. Only moments in and user-ordered fields out
   cross PCIe per evaluation; geometry, matrices, topology and scratch stay
   resident (one H2D and one D2H per evaluation in the trace).
8. Synchronisation: exactly one `cudaEventSynchronize` per evaluation on the
   hot path (`cuda-full`); the hybrid path has two (M2L D2H and P2P D2H).
   No `cudaDeviceSynchronize`, no synchronous `cudaMemcpy`.
9. Events: 13 records per evaluation, all in-stream, 1.2 us each; they do not
   serialise the streams.
10. Allocations: no `cudaMalloc`/`cudaFree`/`cudaMallocHost` inside the
    evaluation loop (all 52 `cudaMalloc` calls of the M capture belong to
    the two constructions).
11. Launch overhead: 15 launches per evaluation at M (8 + 2 x depth), about
    19 us of host API time per evaluation; the per-level M2M/L2L launches
    each scan the complete interaction array and discard other levels, so
    they cost 13-23 us of GPU time each even where a level has 8 nodes.
12. `cuda-full` stays device-resident (see 7).
13. `cuda-partial` overlap: the P2P stream does run concurrently with the CPU
    hierarchy and its final wait is 0-150 us, but the CPU P2M/M2M/L2L/L2P
    chain (about 6.4 ms at M) dominates the 8.1 ms evaluation and the GPU
    is idle for most of it. The M2L round trip additionally uses pageable
    staging vectors and a blocking wait. GPU-side work is 1.6 ms of the
    8.1 ms, so this task's changes can only shorten the GPU portion; the
    CPU hierarchy belongs to task 3B.

In `cuda-full` the P2P and far-field streams overlap only nominally: the BSR
kernel fills every SM, so the far-field kernels launched behind it wait
(the M2M level-4 launch above starts 370 us after it was issued). The
evaluation time is therefore approximately P2P + far-field chain + host
turnaround, and the hidden `cudaEventElapsedTime` phase for M2M absorbs the
P2P duration in the CSV phase columns.

Nsight Compute: unavailable (`ERR_NVGPUCTRPERM`), so no counter-level
classification was possible. Kernel classification below uses arithmetic
intensity (bytes streamed per FMA) and controlled experiments.

### Accepted changes

Each change was measured on the same builds and workloads as the baseline;
numbers are evaluation medians. Compute Sanitizer memcheck and racecheck
(FP32 and FP64, both backends, 3,000 particles, depth 3, order 6) reported no
errors after every kernel change, and the CUDA-related CTest cases passed.

#### 1. Transfer-class grouped M2L (`721fe93`, `perf(cuda-m2l)`)

- Bottleneck: `apply_scaled_m2l_rows_kernel` owned one `(target, beta)`
  output per thread and streamed every `C x C` matrix from L2 once per
  interaction — one coalesced load per FMA, 2.6 TFLOP/s in FP32.
- Change: pairs are counting-sorted by transfer class at construction and cut
  into blocks of `slots x 8` pairs (`slots = 256 / C`). One block stages one
  matrix through shared memory in 16-row alpha slabs and reuses each value for
  every pair; threads own one beta and eight pairs (vector loads of the eight
  multipole values). Multipole level scaling is applied while staging, local
  scaling before the final `atomicAdd`, so the scratch buffer and the separate
  scaling kernel disappear. The target-row kernels stay as the fallback for
  `C > 256`. Accumulation order changes (atomics across classes), which is
  already the case for the atomic P2M/L2P/M2M/L2L kernels.
- M2L kernel time (FP32 / FP64): S 261 -> 25 us / 481 -> 88 us;
  M 1181 -> 208 / 2717 -> 1786; H 3271 -> 469 / 6517 -> 4755;
  L 1227 -> 207 / 2719 -> 1783. FP64 remains compute-bound at about
  1.6 TFLOP/s, the RTX 5090's FP64 rate.
- End-to-end `cuda-full` (FP32 / FP64): S 485 -> 278 us (1.75x) /
  760 -> 367 (2.07x); M 2397 -> 1406 (1.70x) / 4275 -> 3346 (1.28x);
  H 4986 -> 2145 (2.33x) / 8694 -> 6929 (1.25x); L 4307 -> 3306 (1.30x) /
  7146 -> 6220 (1.15x). `cuda-partial` shares the executor: S 1.54x/1.46x,
  H 1.15x/1.07x, M and L within noise because the CPU hierarchy dominates.

#### 2. Host turnaround around `CudaFull` (`1d92a28`, `perf(cuda-fmm)`)

- Bottleneck: 12-20 % of every evaluation was serial host work between the
  D2H completion and the next first device op (see Nsight Systems findings).
- Change: `CudaFullPlan` exposes its pinned staging spans and skips its own
  copies when handed those spans; `UniformFmm` scales user-order moments
  straight into pinned memory and reads user-order fields straight from it in
  OpenMP loops, reuses the sorted identity map gathered at construction when
  the map is fixed, precomputes the inverse volume scale (the FP32 path did
  nine divisions per source), and keeps a persistent FP32 result scratch for
  the FP64 result API. No device work changed.
- End-to-end `cuda-full` on top of change 1 (FP32 / FP64): S 278 -> 234 us
  (1.18x) / 367 -> 349 (1.05x); M 1406 -> 1116 (1.26x) / 3346 -> 3374
  (0.99x); L 3306 -> 2651 (1.25x) / 6220 -> 5998 (1.04x); XL FP32
  cumulative 15504 -> 5414 (2.86x).

### P2P variant study (`benchmark_p2p --cuda --irregular --depth 3`, FP64, 20 evaluations)

Kernel time per evaluation and interactions per second of the existing CUDA
P2P executors at the baseline SHA, random points, 512 target leaves:

| Particles per leaf | canonical AoS | particle-row SoA | leaf block (block per leaf, shared memory) | cuSPARSE BSR(3) |
|---:|---:|---:|---:|---:|
| 12 (6k targets) | 442 us, 3.5 G/s | 350 us, 4.4 G/s | 240 us, 6.4 G/s | 87 us, 17.8 G/s |
| 24 (12k targets) | 1126 us, 5.5 G/s | 1031 us, 6.0 G/s | 843 us, 7.3 G/s | 330 us, 18.7 G/s |
| 64 (33k targets) | 6181 us, 7.1 G/s | 4659 us, 9.4 G/s | 2309 us, 18.9 G/s | 2101 us, 20.8 G/s |

The one-thread-per-target kernels are starved of parallelism at realistic
occupancies (6k-12k threads on 170 SMs) and the block-per-leaf kernel puts
only 32-64 threads per leaf in flight, so cuSPARSE — which parallelises inside
each row — wins by 3-5x despite streaming 9 values per pair instead of 6. At
64 particles per leaf, where the leaf kernel finally has enough threads, it
matches BSR with 33 % fewer tensor bytes. Conclusion: the leaf-block layout
(dense source-major rectangles, six SoA components) is the right data layout;
it needed a warp-per-block execution mapping, not a new representation.

#### 3. Atomic-free far-field kernels (`8312083`, `perf(cuda-fmm)`)

- Bottleneck: P2M/L2P as one atomic per COO entry (L2P's target-major order
  serialised the atomics of one target inside a warp); M2M/L2L as one
  full-size launch per level over all interactions, discarding the other
  levels, with two degree gathers and an `ldexp` per entry.
- Change: P2M/L2P become CSR by output row with 8-lane groups per row and a
  shuffle reduction; M2M/L2L use level-scaled child-class matrices in CSR by
  output and interactions grouped by (level, target), one right-sized launch
  per level, lane groups per output. Deterministic, no atomics, fewer bytes.
- Rejected intermediate: one thread per CSR row. It made P2M slower (S FP64
  13 -> 51 us, L FP64 215 -> 524 us) because 15k-230k threads each walked a
  36-75-entry row serially, so memory-level parallelism collapsed; the lane
  groups restore it.
- Device phases (FP32 / FP64, measured together with change 5): L2L M
  72 -> 29 us / 88 -> 41 us, XL 596 -> 103 / 736 -> 213; L2P M 96 -> 36 /
  89 -> 53, XL 513 -> 208 / 517 -> 448; M2M XL 771 -> 224 / 964 -> 358.

#### 4. Pinned staging in the standalone M2L plan (`e508382`, `perf(cuda-m2l)`)

- Bottleneck (hybrid path): pageable `std::vector` staging made the
  per-evaluation multipole H2D and local D2H driver-staged synchronous copies.
- Change: `cudaMallocHost` staging buffers of the same size; copies become
  asynchronous on the M2L stream.
- M2L phase of `cuda-partial` (H2D + kernels + D2H): M FP32 505 -> 437 us,
  L FP32 505 -> 438, XL FP32 3918 -> 3503. End-to-end stays inside the
  hybrid path's noise floor because its CPU hierarchy dominates.

#### 5. Warp-per-block leaf P2P as the point-source default (`9ac2c58`, `perf(cuda-p2p)`)

- Bottleneck: cuSPARSE BSR(3) streamed 9 values + 1 index per pair (40 B
  FP32) at about 1.3-1.5 TB/s; the custom kernels with 6-value layouts were
  parallelism-starved (see the variant study above).
- Change: the dense leaf packing (`StaticP2PLeafPlan`, six SoA components,
  source-major inside each block) is executed one warp per
  (target leaf, source leaf) block — `stride` targets x `32/stride` source
  slots per warp, contiguous component loads, shuffle reduction over source
  slots, one atomic add per block and target. Point sources (non-periodic)
  select it in both backends with fixed or dynamic identities; the new
  `P2PExecutionPacking::LeafBlock` / `LEAF_BLOCK` reports it. Finite
  sources keep BSR(3)/canonical and the dictionary opt-in is unchanged.
- P2P kernel (FP32 / FP64): S 164 -> 93 us / 263 -> 170; M 474 -> 310 /
  884 -> 544 (16.3M pairs, 391 MB at 1.3 TB/s); L 1640 -> 1205 FP32.
  Standalone FP64 kernel vs BSR: 1.9x at 12/leaf, 1.45x at 24/leaf, 1.36x
  at 64/leaf.
- End-to-end `cuda-full` for changes 3+5 together (FP32 / FP64):
  S 234 -> 199 us (1.18x) / 349 -> 321 (1.09x); M 1116 -> 793 (1.41x) /
  3374 -> 2916 (1.16x); L 2651 -> 1963 (1.35x) / 5998 -> 4777 (1.26x);
  XL 5414 -> 3426 (1.58x) / 20998 -> 18797 (1.12x).

Nsight Systems on this state (M FP32, 793 us): leaf P2P 281 us (at the
DRAM limit for 24 B per pair), grouped M2L 208 us, four M2M launches 80 us,
host turnaround 102 us (was 417), P2M 36, L2P 33, L2L 26, H2D+D2H 57,
device gaps 19 us.

#### 6. Workload-sized M2L pair blocks and translation lane groups (`86f82d9`, `perf(cuda)`)

- Experiments against change 5: 16 M2L pairs per thread (FP32 M2L kernel
  210 -> 180 us at M/L, 1757 -> 1503 us at XL; but FP64 +2 % and S +15 %),
  a whole warp per M2M output (S -11 us, XL +100 us), 8-lane L2L groups
  (slightly worse everywhere). Kept adaptively: FP32 plans with >= 250k
  translations use 16 pairs per thread, others 8; translation levels with
  <= 65536 outputs use 32 lanes per output, larger levels 4; L2L back to 4.
  The slot count is bounded by the 48 KiB shared-memory limit (the plain
  16-pair build failed to launch for small coefficient counts).
- End-to-end `cuda-full` (FP32 / FP64): S 199 -> 188 us (1.06x) /
  321 -> 302 (1.06x); M 793 -> 758 (1.05x) / 2916 -> 2840 (1.03x);
  L 1963 -> 1932 (1.02x) / 4777 -> 4744 (1.01x).
- `benchmark_uniform_fmm` gained `--reduced-symmetry-p2p`,
  `--dictionary-target-owned` and `--dictionary-power2-microtiles`.

### Rejected experiments

- One thread per CSR output row for P2M/L2P/M2M/L2L (first version of
  change 3): P2M slower by up to 4x for FP64 because of lost memory-level
  parallelism; replaced by lane groups.
- 16 M2L pairs per thread unconditionally: FP64 and small plans regress
  (see change 6); kept only for large FP32 plans.
- 32 lanes per translation output unconditionally: +100 us at XL; kept only
  for small levels.
- 8-lane L2L groups: no gain (one parent per child); reverted to 4.

### Reduced-symmetry dictionary P2P versus leaf block (regular grids)

Requested comparison of the experimental signed tensor-dictionary CUDA
executors (`use_reduced_symmetry_p2p`, default source-warp kernel;
`cuda_dictionary_target_owned`; `cuda_dictionary_power2_microtiles`) against
the new leaf-block default on symmetric point lattices (`--regular-grid`,
spherical p=6, `cuda-full`, medians of 7 x 20 evaluations). Evaluation
median [us] with the P2P device phase in parentheses:

| Case | Pairs | Precision | leaf block | dictionary (warp) | dictionary target-owned | dictionary power-of-two |
|---|---:|---|---:|---:|---:|---:|
| 16^3, d3 (8/leaf, P2P negligible) | 0.9M | FP32 | 123 (9) | 122 (51) | 119 (41) | 118 (28) |
| | | FP64 | 293 (18) | 389 (336) | 304 (138) | 288 (45) |
| 32^3, d3 (64/leaf, P2P dominant) | 43.6M | FP32 | 809 (656) | 611 (517) | **537 (440)** | 820 (725) |
| | | FP64 | 1706 (1324) | 2658 (2341) | **1342 (1149)** | 2821 (2449) |
| 32^3, d4 (8/leaf) | 5.5M | FP32 | 512 (117) | 383 (69) | 371 (64) | **367 (38)** |
| | | FP64 | 2382 (215) | 2611 (452) | 2298 (333) | 2301 (184) |
| 64^3, d4 (64/leaf, P2P dominant) | 453M | FP32 | 7210 (5825) | 2448 (1554) | **2331 (1460)** | 2636 (1703) |
| | | FP64 | 15185 (11579) | 9376 (6485) | 9374 (6684) | 9456 (6658) |
| 64^3, d5 (8/leaf) | 57M | FP32 | 3696 (981) | 3325 (436) | **3082 (204)** | 3119 (192) |
| | | FP64 | 20476 (1809) | 21858 (3104) | 19652 (957) | 19712 (844) |

Findings:

- Where P2P is negligible (16^3 at depth 3, M2L dominates) every executor is
  within 5 % of the leaf block; the choice does not matter.
- Where P2P dominates on a lattice, the dictionary is the better
  representation: its 1-2 byte tokens replace 24 (FP32) or 48 (FP64) bytes
  of tensors per pair, so the target-owned executor is 1.5x faster than the
  leaf block at 32^3/depth 3 in FP32 (1.27x FP64) and 3.1x (FP32) / 1.6x
  (FP64) at 64^3/depth 4 where the leaf kernel is at 1.9 TB/s, the DRAM
  limit. Even at 8 particles per leaf the dictionary P2P phase is 2-5x
  shorter, worth 10-30 % end to end.
- Among the dictionary executors the target-owned kernel is the most
  consistent; the default source-warp kernel loses badly in FP64 at 64
  particles per leaf (2.3 ms vs 1.1 ms), and the power-of-two microtiles are
  best only at 8 particles per leaf.
- The dictionary is only applicable to point sources with a fixed identity
  map and non-periodic plans, and it depends on the geometry being a
  lattice (few distinct displacement tensors); for the random point cases
  above it would degenerate to one variant per pair. It therefore stays an
  explicit opt-in; users with regular lattices and P2P-heavy settings should
  enable `use_reduced_symmetry_p2p` with `cuda_dictionary_target_owned`.
  Making target-owned the dictionary default is a separate option-semantics
  decision recorded here, not made.

### Final results (final HEAD versus `293144bf`)

Same builds (identical configuration), hardware, workloads, repeat counts,
precision and parameters as the baseline table; medians of 7 samples x 20
evaluations; sampled accuracy against the exact point reference is
unchanged in every case (the F column is the meaningless cuboid-vs-point
comparison discussed above).

| Case | Backend | Precision | Baseline [us] | Final [us] | Speedup | P2P dev base -> final [us] | M2L dev base -> final [us] | max rel err |
|---|---|---|---:|---:|---:|---|---|---:|
| S (10k pts, p4, d3) | cuda-full | float32 | 485 | 189 | 2.57x | 170 -> 93 | 261 -> 15 | 8.9e-03 |
| S (10k pts, p4, d3) | cuda-full | float64 | 760 | 302 | 2.51x | 269 -> 171 | 481 -> 58 | 8.9e-03 |
| S (10k pts, p4, d3) | cuda-partial | float32 | 619 | 395 | 1.57x | 157 -> 79 | 233 -> 14 | 8.9e-03 |
| S (10k pts, p4, d3) | cuda-partial | float64 | 915 | 477 | 1.92x | 263 -> 170 | 464 -> 57 | 8.9e-03 |
| M (50k pts, p6, d4) | cuda-full | float32 | 2397 | 755 | 3.17x | 508 -> 310 | 1181 -> 180 | 6.5e-03 |
| M (50k pts, p6, d4) | cuda-full | float64 | 4275 | 2922 | 1.46x | 902 -> 546 | 2717 -> 1785 | 6.5e-03 |
| M (50k pts, p6, d4) | cuda-partial | float32 | 8138 | 7396 | 1.10x | 431 -> 284 | 1151 -> 184 | 6.5e-03 |
| M (50k pts, p6, d4) | cuda-partial | float64 | 11196 | 10480 | 1.07x | 786 -> 496 | 2726 -> 1785 | 6.5e-03 |
| H (50k pts, p8, d4) | cuda-full | float32 | 4986 | 1085 | 4.59x | 550 -> 335 | 3271 -> 379 | 1.6e-03 |
| H (50k pts, p8, d4) | cuda-full | float64 | 8694 | 6107 | 1.42x | 1004 -> 581 | 6517 -> 4759 | 1.6e-03 |
| H (50k pts, p8, d4) | cuda-partial | float32 | 21206 | 17626 | 1.20x | 431 -> 287 | 3229 -> 380 | 1.6e-03 |
| H (50k pts, p8, d4) | cuda-partial | float64 | 27013 | 25275 | 1.07x | 783 -> 495 | 6395 -> 4760 | 1.6e-03 |
| L (100k pts, p6, d4) | cuda-full | float32 | 4307 | 1924 | 2.24x | 1766 -> 1274 | 1227 -> 180 | 5.8e-03 |
| L (100k pts, p6, d4) | cuda-full | float64 | 7146 | 4749 | 1.50x | 3214 -> 2144 | 2718 -> 1787 | 5.8e-03 |
| L (100k pts, p6, d4) | cuda-partial | float32 | 10990 | 10165 | 1.08x | 1641 -> 1211 | 1175 -> 185 | 5.8e-03 |
| L (100k pts, p6, d4) | cuda-partial | float64 | 16510 | 15432 | 1.07x | 3030 -> 2044 | 2748 -> 1786 | 5.8e-03 |
| XL (200k pts, p6, d5) | cuda-full | float32 | 15504 | 3164 | 4.90x | 948 -> 717 | 10479 -> 1502 | 1.3e-02 |
| XL (200k pts, p6, d5) | cuda-full | float64 | 24402 | 18812 | 1.30x | 1755 -> 1376 | 18378 -> 15583 | 1.3e-02 |
| XL (200k pts, p6, d5) | cuda-partial | float32 | 59503 | 50193 | 1.19x | 907 -> 569 | 10462 -> 1509 | 1.3e-02 |
| XL (200k pts, p6, d5) | cuda-partial | float64 | 76299 | 78129 | 0.98x | 1673 -> 1146 | 18307 -> 15559 | 1.3e-02 |
| F (16^3 cubes, exact cuboid P2P, p6, d3) | cuda-full | float32 | 512 | 183 | 2.80x | 39 -> 39 | 408 -> 24 | 5.6e+01 |
| F (16^3 cubes, exact cuboid P2P, p6, d3) | cuda-full | float64 | 925 | 286 | 3.23x | 49 -> 44 | 820 -> 173 | 5.6e+01 |
| F (16^3 cubes, exact cuboid P2P, p6, d3) | cuda-partial | float32 | 1142 | 787 | 1.45x | 26 -> 26 | 399 -> 25 | 5.6e+01 |
| F (16^3 cubes, exact cuboid P2P, p6, d3) | cuda-partial | float64 | 1668 | 983 | 1.70x | 32 -> 32 | 798 -> 173 | 5.6e+01 |

CUDA system changes per `cuda-full` evaluation (M FP32, Nsight Systems,
baseline -> final): device ops 20 -> 20 (15 kernels; M2M/L2L stay one launch
per level but each now covers only its level; the M2L scaling kernel is
gone), H2D 1 -> 1 (34 us), D2H 1 -> 1 (26 us), memsets 3 -> 4 (the leaf
kernel accumulates, so the near-field buffer is cleared), inter-op gaps
21 -> 19 us, one `cudaEventSynchronize` and 13 event records unchanged, no
allocations in the loop, host turnaround 417 -> 95 us, device span
1928 -> 681 us. The P2P and far-field streams still overlap only nominally
because both fill the GPU.

Kernel results (M FP32 / FP64, device phase): P2P 466/884 -> 279/498 us
(cuSPARSE BSR -> warp-per-block leaf); M2L 1133/2717 -> 178/1785 us
(grouped); P2M 47/66 -> 36/62 us; L2P 92/89 -> 33/50 us; L2L
67/88 -> 28/40 us; M2M (uncontaminated levels) about 50/70 -> 35/45 us.

### Correctness

- CTest: all CUDA/precision/P2P/M2L cases pass on every accepted commit;
  at the final HEAD the full CTest of the CUDA build passes 199/199 (one
  oneMKL-only case skipped) and the CUDA + oneMKL integration build
  (`CDFMM_ENABLE_MKL=ON`, `one_mkl_available=1`) passes 199/199.
- Python: `PYTHONPATH=build-gpu-perf python -m pytest python_tests` at the
  final HEAD: 141 passed, 3 skipped, module imported from `build-gpu-perf`
  with `cuda_full_available() == True`.
- Compute Sanitizer memcheck/racecheck: no errors after every kernel change.
- New regression test: "CUDA FP32 point plans build without a geometry
  cache" (found and fixed a real defect in the first leaf-packing commit).

### Remaining GPU bottlenecks (`cuda-full`)

- FP64 M2L is compute-bound at about 1.6 TFLOP/s (the RTX 5090's FP64
  rate): 1.8 ms of 2.9 ms at M, 15.6 ms of 18.8 ms at XL. Only fewer
  operations help (exploiting operator structure/symmetry), which is a new
  numerical algorithm and out of scope here.
- Random-point P2P is DRAM-bound at 24 B per pair (1.3-1.9 TB/s); the only
  remaining lever is fewer bytes per pair, i.e. a dictionary-like
  representation for irregular geometry.
- Host turnaround (95 us at M, 220 us at L, FP32) is now the OpenMP scale /
  widen passes plus the synchronous API boundary; pipelining consecutive
  evaluations would need an asynchronous evaluate API.
- H2D + D2H (60 us at M, 140 us at L) are unavoidable per evaluation.
- FP32 M2L at 14-17 TFLOP/s could gain from register tiling over beta as
  well; the M2M launches at coarse levels remain latency-bound (3-13 us
  each).

### Out-of-scope findings

Construction (task 3C): exact cuboid pair tensors cost about 21 us per pair
single-threaded (32^3 cubes at depth 3 need about 20 minutes); leaf-plan
promotion/quantisation for FP32 CUDA plans copies the P2P operator once
more; the dictionary microtile schedule is built and uploaded even when the
default dictionary executor is selected.

CPU / oneMKL (task 3B): the hybrid path is dominated by the CPU hierarchy —
at 50k particles P2M 1.4 ms, M2M 2.2 ms, L2L 2.0 ms, L2P 0.8 ms of 7.4 ms,
against 1.6 ms of GPU work — so `cuda-partial` is 3-5x slower than
`cuda-full` at every size above 10k. Level-serialised M2M/L2L with a barrier
per level and per-target scalar L2P are the first candidates.

## CUDA execution policy and regular-grid hint (3A follow-up)

Starting HEAD `a187c76`. Goal: one deterministic internal policy module that
owns the CUDA strategy choices found in Phase 3A, plus a public
`SpatialLayout` hint so lattices get the reduced-symmetry dictionary without
the experimental flags. No kernel changed; no cache format or key changed.

### Module

`src/backend/cuda/execution_policy.{hpp,cpp}` (compiled in every build, host
only). `resolve_cuda_execution_policy(inputs)` maps plan facts to concrete
choices; `m2l_pairs_per_thread(precision, translations)` and
`translation_lanes_for_outputs(outputs)` are queried by the M2L and
far-field executors so the rules live in one place. Inputs come from the
constructed topology and the options in `UniformFmm::resolve_cuda_execution_policy()`
(called at the end of `initialise_p2p_policy`, before any packing is derived):
precision, `spatial_layout`, CUDA backend selected, effective point source,
periodic, fixed identity available, the three explicit dictionary options,
source/target counts, order, coefficient count, depth, occupied target leaf
count, mean targets per occupied target leaf, list-1 pair count (from
`p2p_leaf_records`), M2L translation count, BSR(3) size estimate and budget.

Rules (in precedence order):

1. explicit `use_reduced_symmetry_p2p` and the dictionary is valid
   (non-periodic; point sources need a fixed identity map) -> signed
   dictionary; executor from the explicit flags with their documented
   meaning (`cuda_dictionary_target_owned` > `cuda_dictionary_power2_microtiles`
   > source-warp);
2. `SpatialLayout::RegularGrid`, CUDA backend, non-periodic point sources
   with a fixed identity map -> signed dictionary; executor = explicit flag
   if set, else power-of-two microtiles when the mean occupancy is below
   48 targets per leaf, target-owned otherwise;
3. non-periodic point sources -> leaf block (Phase 3A default);
4. finite sources or fixed-identity points, non-periodic, BSR estimate within
   `cuda_p2p_bsr_max_bytes` -> cuSPARSE BSR(3);
5. otherwise canonical rows.

M2L: 16 pairs per thread for FP32 plans with >= 250k translations, else 8.
M2M/L2L: 32 lanes per output for levels with <= 65536 outputs, else 4.
If a dictionary is selected but its plan cannot be derived, the plan falls
back to the General rules (3-5). CPU backends ignore the hint and keep
`use_reduced_symmetry_p2p` as their only dictionary switch.

### Dictionary executor calibration (regular lattices, `cuda-full`, p=6)

P2P device phase [us] (evaluation median in parentheses); `occN` is the mean
number of targets per leaf; N = 131072/d5, 32768/d4, 65536/d4, 131072/d4,
262144/d4, 65536/d3.

| Occupancy | Precision | source-warp | target-owned | power-of-two | best |
|---|---|---:|---:|---:|---|
| 4 | FP32 | 212 (2587) | **65 (2426)** | 80 (2471) | owned |
| 4 | FP64 | 1523 (19566) | **373 (18272)** | 411 (18483) | owned |
| 8 | FP32 | 69 (388) | 59 (368) | **39 (365)** | pow2 |
| 8 | FP64 | 453 (2655) | 347 (2295) | **184 (2281)** | pow2 |
| 16 | FP32 | 312 (673) | 233 (588) | **193 (590)** | pow2 |
| 16 | FP64 | 904 (3258) | 955 (2894) | **652 (2843)** | pow2 |
| 32 | FP32 | 600 (1184) | 526 (1072) | **525 (1168)** | tie |
| 32 | FP64 | **1783 (4396)** | 2139 (4249) | 1980 (4593) | warp |
| 64 | FP32 | 1551 (2455) | **1440 (2307)** | 1701 (2621) | owned |
| 64 | FP64 | **6500 (9220)** | 6680 (9402) | 6667 (9398) | warp (3 %) |
| 128 | FP32 | **974 (1214)** | 1339 (1599) | 1679 (1962) | warp |
| 128 | FP64 | **4605 (5131)** | 7646 (8234) | 6987 (7664) | warp |

Together with the Phase-3A study (32^3/d3, 64 per leaf: target-owned 440 vs
warp 517 vs pow2 725 us FP32; 1149 vs 2341 vs 2449 us FP64) the simple rule
"power-of-two microtiles below 48 targets per leaf, target-owned above" is
best or within a few percent of best from 8 to 64 per leaf, which covers the
depths the parameter adviser produces. Known limits, recorded rather than
encoded: at 4 per leaf target-owned is 15 % faster on a P2P phase that is
3 % of the evaluation (0.6 % end to end), and at 128 per leaf (a shallow
tree the adviser would not choose) the source-warp kernel is 1.4-1.7x faster
than target-owned; that executor remains reachable through the explicit
`use_reduced_symmetry_p2p` option.

### Benchmarks (policy commit versus `a187c76`, identical builds)

General random points (policy resolves to the same choices as before;
this is a no-regression check), evaluation medians [us], before -> after:

| Case | cuda-full FP32 | cuda-full FP64 | cuda-partial FP32 | cuda-partial FP64 |
|---|---|---|---|---|
| S 10k p4 d3 | 189 -> 188 | 302 -> 302 | 393 -> 400 | 477 -> 478 |
| M 50k p6 d4 | 758 -> 758 | 2919 -> 2909 | 7276 -> 7384 | 11230 -> 10093 |
| L 100k p6 d4 | 1924 -> 1924 | 4738 -> 4808 (see note) | 10337 -> 10578 | 15316 -> 15543 |

All `cuda-full` differences are inside the +-0.3 % noise floor except
L FP64 (+1.5 % in the matrix run). The resolved policy is identical there
(leaf block, 8 pairs per thread); eight back-to-back alternating runs gave
before 4.709/4.728/4.690/4.735 ms and after 4.743/4.748/4.717/4.739 ms, an
overlapping 1 % scatter, so no regression is attributed to the change. `cuda-partial`
differences are inside that path's 1-9 % CPU-dominated noise.

Regular lattices, `cuda-full`, `--spatial-layout regular-grid` with no
explicit dictionary options, against the best explicit executor of the
Phase-3A study and the leaf-block default:

| Case | Occupancy | Precision | Automatic (executor chosen) | Best explicit | Leaf block (General) |
|---|---:|---|---:|---:|---:|
| 32^3 d4 | 8 | FP32 | **366 us** (pow2) | 367 (pow2) | 512 |
| 32^3 d4 | 8 | FP64 | **2284** (pow2) | 2298 (owned) / 2301 (pow2) | 2382 |
| 32^3 d3 | 64 | FP32 | **532** (owned) | 537 (owned) | 809 |
| 32^3 d3 | 64 | FP64 | **1345** (owned) | 1342 (owned) | 1706 |
| 64^3 d5 | 8 | FP32 | **3125** (pow2) | 3082 (owned) / 3119 (pow2) | 3696 |
| 64^3 d5 | 8 | FP64 | **19703** (pow2) | 19652 (owned) / 19712 (pow2) | 20476 |
| 64^3 d4 | 64 | FP32 | **2304** (owned) | 2331 (owned) | 7210 |
| 64^3 d4 | 64 | FP64 | **9252** (owned) | 9374 (owned) | 15185 |

The automatic choice is within 1.4 % of the best explicit executor in every
case (and 1.4-3.1x faster than the General leaf block where P2P dominates);
it is 0.2-1.4 % slower than target-owned at 8 per leaf on the 64^3 lattice,
where P2P is under 6 % of the evaluation.

### Cache and correctness

- `SpatialLayout` is not part of the persistent cache identity: the cached
  geometry payload is the canonical operator, and the dictionary/leaf/BSR
  packings are derived after loading (`use_reduced_symmetry_p2p` stays in
  the key as before; no format or key changed, no cache invalidated).
- CTest: 71/71 policy/CUDA/precision/P2P/M2L cases with the cache disabled,
  202/202 full run with the cache (one oneMKL-only skip); new cases
  "CUDA execution policy resolves the P2P packing from layout and options",
  "regular-grid layout hint selects the dictionary on CUDA plans" (both
  backends, depths 1 and 2, explicit overrides, dynamic identities, FP32
  agreement), "regular-grid layout hint keeps finite and periodic CUDA
  policies".
- Python: `SpatialLayout.GENERAL/REGULAR_GRID`, `UniformFmmOptions.spatial_layout`,
  `UniformFmm.spatial_layout`; new tests for the round trip and the CUDA
  dictionary selection.
- Compute Sanitizer memcheck on the automatic regular-grid dictionary
  (8 per leaf -> microtiles, 64 per leaf -> target-owned, FP32 and FP64) and
  racecheck on the hybrid path: no errors.
- Defect found while validating: the FP32 CUDA path only quantised the
  dictionary plan when the explicit flag was set, so the layout-selected
  FP32 dictionary silently fell back to leaf blocks; fixed in the same
  commit and covered by the FP32 assertion of the new test.

## Regular-grid dictionary policy at high occupancy (3A closure)

Starting HEAD `a0a7003`. The two-regime rule above (power-of-two microtiles
below 48 targets per leaf, target-owned otherwise) was known to be wrong at
128 per leaf, where the source-warp kernel is 1.4-1.7x faster. This task
calibrates the upper crossover and adds the third regime. No kernel changed.

### Method

`benchmark_uniform_fmm --backend cuda-full --regular-grid --order 6`, FP32
and FP64, explicit `--reduced-symmetry-p2p` with the source-warp default,
`--dictionary-target-owned`, and `--dictionary-power2-microtiles`; medians of
5 samples x 20 evaluations after 3 warm-ups, same `build-gpu-perf` binary and
environment as the Phase-3A tables (RTX 5090, driver 595.84, CUDA 13.2,
g++ 15.3, `-march=native`, LTO). The lattice generator now accepts counts of
the form `odd * 2^k` (the odd factor stretches the shortest axis) so that
non-power-of-two occupancies are exact and uniform; the printed grid shape is
recorded below. Every depth-3 case has 512 occupied leaves, so
occupancy = N / 512.

| Case | N | Grid | Targets per leaf |
|---|---:|---|---:|
| occ48d3 | 24,576 | 48 x 16 x 32 | 48 |
| occ64d3 | 32,768 | 32 x 32 x 32 | 64 |
| occ80d3 | 40,960 | 80 x 16 x 32 | 80 |
| occ96d3 | 49,152 | 48 x 32 x 32 | 96 |
| occ128d3 | 65,536 | 32 x 32 x 64 | 128 |
| occ160d3 | 81,920 | 80 x 32 x 32 | 160 |
| occ192d3 | 98,304 | 96 x 32 x 32 | 192 |
| occ48d4 | 196,608 | 96 x 32 x 64 | 48 (4096 leaves) |

### Results

P2P device phase [us], evaluation median in parentheses:

| Occupancy | Precision | source-warp | target-owned | power-of-two | best | owned / best | warp / best |
|---:|---|---:|---:|---:|---|---:|---:|
| 48 | FP32 | 463 (553) | **330 (418)** | 789 (901) | owned | 1.00 | 1.40 |
| 64 | FP32 | 521 (611) | **440 (535)** | 724 (823) | owned | 1.00 | 1.18 |
| 80 | FP32 | **490 (601)** | 616 (727) | 1534 (1707) | warp | 1.26 | 1.00 |
| 96 | FP32 | **600 (748)** | 745 (916) | 1244 (1447) | warp | 1.24 | 1.00 |
| 128 | FP32 | **974 (1243)** | 1348 (1624) | 1668 (1938) | warp | 1.38 | 1.00 |
| 160 | FP32 | **1514 (1814)** | 2217 (2549) | 2443 (2773) | warp | 1.46 | 1.00 |
| 192 | FP32 | **2844 (3207)** | 2926 (3315) | 3016 (3387) | warp | 1.03 | 1.00 |
| 48 | FP64 | 1815 (2072) | **899 (1058)** | 1441 (1629) | owned | 1.00 | 2.02 |
| 64 | FP64 | 2341 (2616) | **1148 (1342)** | 2436 (2697) | owned | 1.00 | 2.04 |
| 80 | FP64 | **2079 (2343)** | 2245 (2642) | 4119 (4579) | warp | 1.08 | 1.00 |
| 96 | FP64 | **2567 (2991)** | 4772 (5330) | 5032 (5575) | warp | 1.86 | 1.00 |
| 128 | FP64 | **4612 (5145)** | 7701 (8345) | 6950 (7534) | warp | 1.67 | 1.00 |
| 160 | FP64 | **4787 (5377)** | 9825 (10480) | 9194 (9968) | warp | 2.05 | 1.00 |
| 192 | FP64 | **6217 (6918)** | 12194 (13139) | 11320 (12097) | warp | 1.96 | 1.00 |
| 48 (depth 4, 196k) | FP32 | 1438 (2037) | **816 (1617)** | 1633 (2228) | owned | 1.00 | 1.76 |
| 48 (depth 4, 196k) | FP64 | 5089 (7652) | **4517 (6461)** | 6039 (7079) | owned | 1.00 | 1.13 |

The crossover lies between 64 and 80 targets per leaf in both precisions and
is monotone on either side: target-owned is best at 48 and 64 (by 1.18-2.04x
over source-warp), source-warp is best at every point from 80 to 192 (by
1.08-2.05x over target-owned; the 192/FP32 point is a 3 % margin). The
power-of-two microtile kernel is never best above 32 per leaf and collapses
at non-power-of-two occupancies (80: 2.5-3.1x slower than the best). The
128-thread target-owned tile also explains its loss at 80 and 96 (37 % and
25 % of the lanes of the last tile idle), and above 128 its per-target
serial source loop simply falls behind the warp-cooperative kernel.

### Decision

Three regimes, thresholds on the mean targets per occupied target leaf:

```text
occupancy <  48  -> power-of-two microtiles
48 <= occ  < 72  -> target-owned
occupancy >= 72  -> source-warp
```

72 is the midpoint of the measured 64/80 crossover. The lower regime and the
very-low-occupancy behaviour (4 per leaf, 0.6 % end to end) are unchanged.
Explicit `cuda_dictionary_target_owned` / `cuda_dictionary_power2_microtiles`
still override the automatic choice, `use_reduced_symmetry_p2p` keeps its
source-warp default, `SpatialLayout::General` is untouched, and nothing in
the cache identity or format changes (the executor is chosen after the
canonical operator is loaded).

### Automatic policy versus best explicit executor (policy commit)

`--spatial-layout regular-grid` with no explicit dictionary options, same
lattices, evaluation median [us] (P2P device phase in parentheses), against
the fastest explicit executor of the calibration table:

| Occupancy | Precision | Automatic (chosen) | Best explicit | auto / best |
|---:|---|---:|---:|---:|
| 48 | FP32 | 417 (328) target-owned | 418 (330) owned | 1.00 |
| 48 | FP64 | 1057 (900) target-owned | 1058 (899) owned | 1.00 |
| 64 | FP32 | 533 (440) target-owned | 535 (440) owned | 1.00 |
| 64 | FP64 | 1365 (1150) target-owned | 1342 (1148) owned | 1.02 |
| 80 | FP32 | 601 (489) source-warp | 601 (490) warp | 1.00 |
| 80 | FP64 | 2331 (2072) source-warp | 2343 (2079) warp | 0.99 |
| 96 | FP32 | 748 (600) source-warp | 748 (600) warp | 1.00 |
| 96 | FP64 | 3024 (2566) source-warp | 2991 (2567) warp | 1.01 |
| 128 | FP32 | 1199 (971) source-warp | 1243 (974) warp | 0.96 |
| 128 | FP64 | 5122 (4604) source-warp | 5145 (4612) warp | 1.00 |
| 192 | FP32 | 3203 (2843) source-warp | 3207 (2844) warp | 1.00 |
| 192 | FP64 | 6921 (6225) source-warp | 6918 (6217) warp | 1.00 |

The automatic choice is within +-2 % (run-to-run noise) of the best explicit
executor at every point; at 128 per leaf it is now 1.34x (FP32) and 1.63x
(FP64) faster than the previous two-regime rule. Below 48 per leaf nothing
changed (the 8-32 per leaf results of the previous section stand).

### Validation

- `ctest` on `build-gpu-perf`: 30/30 policy/regular-grid/dictionary/CUDA
  cases and 65/65 precision/P2P/M2L/FMM/layout cases pass; the policy unit
  test now asserts all three regimes and both boundary values, and the
  lattice integration test asserts the exact automatic executor per depth.
- No kernel changed, so no new sanitizer campaign was run; the FP32 and FP64
  agreement assertions of the lattice test cover the automatic executors.

**Phase 3A GPU evaluation is closed.**

## CPU / oneMKL evaluation optimization (3B)

### Environment

- Same machine as 3A: `mihaa-workstation`, Linux 7.0.0-31, Intel
  i9-14900KF (8 P-cores + 16 E-cores, no AVX-512; AVX2/FMA), 184 GB DDR5,
  CPU governor `powersave`, clocks not locked.
- OpenMP policy from the `cdfmm` conda environment: `OMP_NUM_THREADS=8`,
  `OMP_PLACES={0},{2},{4},{6},{8},{10},{12},{14}` (one thread per P-core),
  `OMP_PROC_BIND=close`. Thread-scaling runs use `--threads N` (which calls
  `omp_set_num_threads`) with the same places list, so N <= 8 threads always
  sit on P-cores; E-cores were not used.
- Toolchain: conda `g++` 15.3.0 (`-O3 -march=native`, LTO on, OpenMP on, no
  fast-math), CMake 4.4.3, Ninja 1.13.2; oneMKL 2026.1.0 (lp64, gnu_thread)
  from the conda environment (`MKLROOT` is unset; `MKL_DIR` points at the
  environment's cmake files). `icpx` 2026.1.1 exists but the portable
  production builds (CI, `dev`, `release`) use g++, so g++ is the benchmark
  compiler.
- Profilers: `perf` is unusable (`perf_event_paranoid=4`, no sudo); VTune,
  Advisor, llvm-mca and gperftools are not installed. Evidence therefore
  comes from the repository phase timers (`EvaluationTimings`, per-phase
  means in the benchmark CSV), wall-clock medians, thread-scaling runs,
  byte-count arithmetic, and three read-only source audits.

### Starting point

`CPU_PERF_BASELINE = e4f1c795cd49d5a3e73e7236f573a63bc4b0cb63`
(`fix(cuda-stub): define the pinned staging accessors in non-CUDA builds`).
This is the Task-A HEAD (`c7f7dac`) plus one link fix: since the Phase-3A
host-turnaround commit (`1d92a28`) every non-CUDA build that links the
evaluation path (tests, benchmarks, Python module) failed with undefined
`CudaFullPlan::pinned_*` symbols, so the portable baseline could not be
built without it. The fix adds four stub accessors that are never executed
and has no effect on CPU performance. Baseline binaries were built in a
detached worktree at that SHA (`build-cpu`: oneMKL off; `build-mkl`: oneMKL
on), candidates in `build-cpu-perf` / `build-mkl-perf` of the working tree
with the identical configuration (`-DCMAKE_BUILD_TYPE=Release
-DCMAKE_CXX_COMPILER=g++ -DCDFMM_ENABLE_LTO=ON -DCDFMM_ENABLE_NATIVE_ARCH=ON
-DCDFMM_ENABLE_OPENMP=ON -DCDFMM_ENABLE_CUDA=OFF`, plus the MKL variables for
the oneMKL trees).

### Workloads

`benchmark_uniform_fmm`, spherical basis, random points (seed 314159), fixed
identity map, field output, `--no-direct --no-workload-comparison --warmups 3
--evaluations 20 --samples 7 --accuracy-targets 128`; the S/M/H/L cases of
Phase 3A (S 10k p4 d3; M 50k p6 d4; H 50k p8 d4; L 100k p6 d4), backends
`cpu-static-matrix` (portable) and `cpu-static-matrix-mkl`, FP32 and FP64,
8 threads; thread scaling on M at 1/2/4/8 threads. `cuda-partial` from the
CUDA build (`build-gpu-perf`, same compiler and flags plus CUDA) is the
secondary end-to-end metric.

### Baseline (e4f1c79), 8 threads, evaluation median and per-phase means [us]

| Case | Backend | Prec | eval | P2M | M2M | M2L (gather / multiply / scatter) | L2L | L2P | P2P | max rel err |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| S | portable | FP32 | 4270 | 125 | 101 | 1797 | 82 | 91 | 2004 | 8.9e-03 |
| S | portable | FP64 | 6478 | 188 | 137 | 1942 | 114 | 207 | 3856 | 8.9e-03 |
| S | oneMKL | FP32 | 3070 | 124 | 118 | 453 (68 / 130 / 255) | 86 | 104 | 2150 | 8.9e-03 |
| S | oneMKL | FP64 | 5600 | 176 | 100 | 722 (106 / 225 / 390) | 97 | 195 | 4233 | 8.9e-03 |
| M | portable | FP32 | 99684 | 1822 | 2692 | 84578 | 2452 | 1203 | 8916 | 6.5e-03 |
| M | portable | FP64 | 115401 | 2416 | 2802 | 92537 | 2509 | 2245 | 15784 | 6.5e-03 |
| M | oneMKL | FP32 | 35657 | 1765 | 2440 | 18693 (4279 / 7057 / 7356) | 2203 | 1111 | 8800 | 6.5e-03 |
| M | oneMKL | FP64 | 62849 | 2299 | 2726 | 36816 (7794 / 14570 / 14451) | 2305 | 2183 | 15577 | 6.5e-03 |
| H | portable | FP32 | 263430 | 3076 | 6640 | 236388 | 6263 | 1799 | 8904 | 1.6e-03 |
| H | portable | FP64 | 325641 | 4139 | 6714 | 290934 | 6669 | 3658 | 15602 | 1.6e-03 |
| H | oneMKL | FP32 | 58170 | 3045 | 6297 | 31534 (6982 / 12352 / 12199) | 5849 | 1808 | 8908 | 1.6e-03 |
| H | oneMKL | FP64 | 98036 | 3905 | 6390 | 61384 (11676 / 24811 / 24895) | 6151 | 3584 | 15548 | 1.6e-03 |
| L | portable | FP32 | 127956 | 3532 | 2727 | 83774 | 2446 | 2178 | 34139 | 5.8e-03 |
| L | portable | FP64 | 164951 | 4609 | 2999 | 90764 | 2481 | 4068 | 61424 | 5.8e-03 |
| L | oneMKL | FP32 | 64911 | 3531 | 2618 | 18687 (4315 / 7090 / 7282) | 2251 | 2154 | 34341 | 5.8e-03 |
| L | oneMKL | FP64 | 113283 | 4451 | 2964 | 36955 (7821 / 14611 / 14522) | 2342 | 4036 | 61143 | 5.8e-03 |

Moment permutation, resets and result unpermutation are 0.2-0.5 ms in total
at every size. Noise floor (same code, baseline versus candidate tree,
8 threads): S/H/L agree within +-2 %, the two M cases within 5 %
(99.7 -> 104.8 ms, 115.4 -> 120.1 ms), so 8-thread speedups below 1.05x are
not claimed.

Thread scaling, M case, evaluation median [ms] (M2L phase in parentheses):

| Threads | portable FP32 | portable FP64 | oneMKL FP32 (gather/mult/scatter) | oneMKL FP64 |
|---:|---:|---:|---:|---:|
| 1 | 669.6 (599.8) | 748.0 (671.0) | 113.3 (8.8 / 28.1 / 7.0) | 164.3 (16.7 / 57.0 / 14.0) |
| 2 | 345.6 (306.9) | 381.7 (332.3) | 67.8 (5.8 / 15.0 / 7.0) | 106.0 (10.9 / 30.6 / 14.0) |
| 4 | 179.0 (155.3) | 200.0 (166.7) | 44.1 (4.6 / 8.7 / 7.2) | 74.3 (8.9 / 17.6 / 14.2) |
| 8 | 99.7 (84.6) | 115.4 (92.5) | 35.7 (4.3 / 7.1 / 7.4) | 62.8 (7.8 / 14.6 / 14.5) |

Other phases 1 -> 8 threads (M FP64): P2M 8.2 -> 2.4 ms (3.4x), M2M
15.3 -> 2.8 (5.5x), L2L 14.4 -> 2.5 (5.7x), L2P 7.0 -> 2.2 (3.1x), P2P
31.7 -> 15.8 (2.0x; FP32 29.1 -> 8.9, 3.3x).

### Dominant bottlenecks (baseline)

1. **Portable M2L** is 85 % of the standalone portable evaluation at M/L
   and 90 % at H: one thread owns one `(target, beta)` output and re-walks
   the target's whole interaction row per beta with stride-`C` matrix reads
   (`matrix[matrix_id*C*C + alpha*C + beta]`), re-applying the source-level
   scaling inside the innermost loop (3 flops per useful FMA, 4.6 GFLOP per
   evaluation at M instead of 3.1). It is scalar and cache-line amplified
   (one line fetched per useful matrix value); it scales well with threads
   (7.1x) precisely because it is latency-bound, not bandwidth-bound.
2. **oneMKL M2L**: the scatter is serial (7-14.5 ms, flat across thread
   counts, 40 % of the phase at 8 threads); gather and scatter both scan
   every column of every group and skip other levels (about 75 % wasted
   iterations at depth 4); `mkl_set_num_threads_local` is called twice per
   GEMM (about 1800 calls per evaluation); the gather/translated scratch is
   sized for all levels (502 MB FP64 at M).
3. **CPU P2P** streams 53 B (FP64) / 29 B (FP32) per pair from DRAM: 16.3M
   pairs at M are 875 / 478 MB per evaluation, i.e. 0.34-0.62 flop/B, and the
   phase scales 2-3.3x on 8 threads, consistent with the DRAM roof. The FP32
   compact kernel also lacks the `omp simd` reduction of the FP64 kernel.
4. **P2M / L2P** stream 16-byte COO entries (48C B per source: 118 MB at M,
   p=6) and four separately allocated rows per target (32C B per target,
   heap order): both are bandwidth/pointer-chasing bound at 0.125-0.25 flop/B.
5. **M2M / L2L** call `std::ldexp` and two bounds-checked degree lookups per
   sparse entry (about 1500 entries per translation at p=6), scalar scatter
   accumulation; L2L also forks one parallel region per level.
6. The remaining serial passes (identity gather per evaluation even with a
   fixed map, near-field combine, resets) are 0.3-0.5 ms per evaluation.

### Accepted changes

Each change was measured on the same builds (identical configuration, g++
15.3, LTO, `-march=native`), workloads and repeat counts as the baseline;
numbers are per-evaluation means of the phase timers and evaluation medians
at 8 threads. Correctness: the full portable and oneMKL CTest suites and the
Python tests pass after every commit (see "Validation"), and a 20k-point
FP64/FP32 field comparison against the baseline module quantifies each
accumulation-order change.

#### 1. Portable M2L per target with unit-stride matrix columns (`82c4b35`, `perf(cpu-m2l)`)

- Evidence: 85 % of the portable evaluation (M), scalar stride-C matrix
  reads, 3 flops per useful FMA (source scaling inside the innermost loop),
  7.1x thread scaling (latency-bound, not bandwidth-bound).
- Change: one target per iteration; each interaction scales its source once
  and streams its `C x C` matrix exactly once as `C` unit-stride axpy updates
  of a stack accumulator (vectorised, no scatter); local scaling once per
  target; interaction order per target unchanged. Plans above 512
  coefficients keep the per-output kernel.
- M2L phase: M 84.6 -> 10.1 ms (FP32), 92.5 -> 22.9 (FP64); H 236 -> 33.0 /
  291 -> 67.7; L 83.8 -> 10.5 / 90.8 -> 22.9; S 1.80 -> 0.31 / 1.94 -> 0.51.
- Numerics: FP64 fields differ from the baseline by at most 1e-19 of the
  field scale (product association only), FP32 by 1e-10; medians 0.

#### 2. Dense level-scaled far-field packing (`a5d8d05`, `perf(cpu-far-field)`)

- Evidence: P2M streamed 118 MB of 16-byte COO entries per evaluation at M
  (1.8 ms, 82 GB/s, i.e. the DRAM roof); L2P streamed four heap-allocated rows
  per target (78 MB, pointer chasing); M2M/L2L executed `std::ldexp` and two
  bounds-checked degree lookups per sparse entry (about 1500 per translation at
  p=6) with scalar scatter accumulation.
- Change: `backend/cpu/far_field/packing.{hpp,cpp}` builds once at
  construction, for every backend that runs the hierarchy on the CPU, dense
  `[source][component][C]` P2M rows (24C B per source instead of 48C),
  per-(child level, class) level-scaled translation column banks over each
  input column's contiguous output range (no degree lookup, no `ldexp`,
  unit-stride axpy), and flat `[target][3][C]` field rows plus separate
  potential rows for L2P (field-only evaluations stream 24C B per target).
  The canonical per-source/per-target maps are released after packing
  (nothing reads them once the cache is written and no CUDA-full plan is
  built); the eight canonical translation operators stay resident. New
  `StaticPlanStatistics::far_field_packing` timing; P2M/L2P bytes now report
  the packing. Cache format, keys and CUDA paths untouched.
- Phases at M (FP32 / FP64): P2M 1.82 / 2.42 -> 0.51 / 1.14 ms; M2M
  2.69 / 2.80 -> 0.32 / 0.78; L2L 2.45 / 2.51 -> 0.17 / 0.43; L2P
  1.20 / 2.25 -> 0.63 / 1.16. Hierarchy total 8.2 / 10.0 -> 1.6 / 3.5 ms.
- Numerics: identical products (the level factor is an exact power of two);
  the Cartesian basis changes the input summation order of M2M/L2L only.

#### 3. oneMKL M2L: level ranges, parallel deterministic scatter, hoisted thread setting (`ed67134`, `perf(mkl)`)

- Evidence: serial scatter flat at 7.0 ms (FP32) / 14.0 ms (FP64) from 1 to 8
  threads (40 % of the phase at 8 threads); gather and scatter scanned every
  column of every group per level (75 % wasted iterations at depth 4);
  `mkl_set_num_threads_local` twice per GEMM (about 1800 calls per
  evaluation); scratch sized for all levels (502 MB FP64 at M).
- Change: per-group per-level column ranges; gather/GEMM visit only the
  level's columns; one thread-setting bracket per thread per level; scratch
  for one level; scatter parallel over targets through a per-level schedule
  that visits each target's (group, column) contributions in the former
  serial order, so the accumulation is bit-identical.
- oneMKL M2L at M: 18.7 -> 12.0 ms FP32 (gather 4.3 -> 2.8, multiply
  7.1 -> 6.5, scatter 7.4 -> 2.7), 36.8 -> 25.8 ms FP64 (7.8 -> 6.4,
  14.6 -> 14.1, 14.5 -> 5.3); H 31.5 -> 21.8 / 61.4 -> 43.8; L 18.7 -> 12.3 /
  37.0 -> 25.9. Scratch 502 -> 219 MB (FP64, M).

#### 4. Transfer-class-sorted M2L block schedule for the portable executor

- Evidence: after change 1 the kernel streamed each `C x C` matrix from L3
  once per interaction (316 x 19 KB = 6 MB of FP64 matrices at p=6 exceed the
  2 MB L2): 10.1 / 22.9 ms at M for 3.07 GFLOP, i.e. 300 / 134 GFLOP/s.
- Change: `backend/cpu/m2l/schedule.{hpp,cpp}` cuts each level's targets
  into Morton blocks (accumulators of at most 4096 values, at least four
  blocks per thread per level) and stably sorts a block's interactions by
  transfer class, so one matrix stays in L1 across a run of interactions.
  One thread owns a block; accumulation per target is deterministic but in
  class order instead of row order. Measured kernel split: FP32 rows are
  fastest as plain unit-stride axpy updates, FP64 gains from register-blocking
  16/8/4 outputs across alpha (FP32 with the same chunks was 25 % slower:
  M 8.6 -> 10.3 ms). The schedule is built only when the matrix set exceeds
  1 MiB (below that the row kernel already runs from L2: S FP32 lost 20 % of
  its 0.3 ms M2L with the schedule).
- M2L phase (FP32 / FP64): M 10.1 / 22.9 -> 8.2 / 12.0 ms; H 33.0 / 67.7 ->
  18.8 / 34.4; L 10.5 / 22.9 -> 8.2 / 12.0; S unchanged (row kernel).
- Rejected variants on the way: 128-target blocks with a fixed count
  (S regressed 25 %: four blocks per level); register chunks for FP32.

#### 5. Position-based list-1 P2P for point sources and targets (`P2PExecutionPacking::PointGeometry`)

- Evidence: after changes 1-4 the stored-tensor P2P was 40-70 % of the
  portable evaluation (M 8.6 / 15.5 ms, L 34 / 61 ms FP32 / FP64) and sat
  at the DRAM roof: 29 B (FP32) / 53 B (FP64) per pair, 478 / 875 MB per
  evaluation at M, 55 GB/s, 0.34-0.62 flop/B, 2.0-3.3x thread scaling. Only
  fewer bytes per pair can help; for point sources the pair tensor is a
  closed formula of two resident positions.
- Change: `backend/cpu/p2p/geometry.{hpp,cpp}` sweeps the canonical list-1
  leaf records from the sorted positions: per target leaf the neighbourhood
  (positions with image shift, moments, sorted index) is gathered once into a
  per-thread SoA scratch sized at construction, then every target of the leaf
  runs one contiguous vectorised sweep with the self pair excluded by index.
  The pair formula lives once in `operators/p2p_point_kernel.hpp`, which
  `operators::p2p::evaluate_pair` now also calls, so the backend defines no
  second P2P. Selected automatically on `CpuStatic` for non-periodic plans
  with point sources and point targets (geometry or near-field model) and no
  explicit reduced-symmetry request; finite and periodic near fields keep the
  particle-row SoA tensors, CUDA backends are unaffected. The canonical and
  row P2P operators are released after the cache is written, so no pair
  tensors stay resident (statistics report zero near-field operator bytes
  and the scratch); construction still builds the canonical operator for the
  cache (a 3C item). The FP32 plan computes in FP32 from the FP64 positions;
  the FP32 potential path sweeps the same records.
- P2P phase (FP32 / FP64): S 1.92 / 3.76 -> 1.50 / 1.59 ms; M 8.56 / 15.5 ->
  5.61 / 5.87; H 8.84 / 15.5 -> 5.79 / 6.11; L 34.0 / 60.9 -> 21.5 / 22.7.
  Resident P2P memory at M: 875 MB (FP64 compact) -> 0.5 MB of scratch.
- Tests: the packing name asserted by the C++ and Python API tests changed
  from `ParticleRowSoa` to `PointGeometry` for point plans; finite-geometry
  tests keep `ParticleRowSoa`.

#### 6. Vectorised position sweep (part of the P2P commit)

- Evidence: the first version of the sweep ran at about 13 cycles per pair
  per thread in both precisions (5.6 / 5.9 ms at M) and the linked binary
  contained only scalar `vsqrtsd`/`vdivsd`: with `-fmath-errno` (GCC's
  default) `std::sqrt` is a call with an errno branch, and the masked
  self-pair select counts as trapping control flow, so GCC refused to
  vectorise the loop ("unsupported control flow in loop").
- Change: the sweep excludes the self pair without control flow (unit
  displacement plus zero weight) and `cdfmm_core` is compiled and LTO-linked
  with `-fno-math-errno -fno-trapping-math`. Both plan precisions evaluate
  the pair in FP64 (measured to cost nothing over FP32 arithmetic) and round
  once, which brings the FP32 field back to the accuracy of the former
  FP64-computed tensors (max difference to the baseline 3.5e-7 of the field
  scale instead of 5.1e-5 with FP32 pair arithmetic).
- P2P phase at M: 5.6 / 5.9 -> 3.1 / 3.1 ms (FP32 / FP64); the branch-free
  loop without the flags was slower (7.9 ms) because it stayed scalar with
  extra selects. A per-source-file `COMPILE_OPTIONS` property was tried first
  and had no effect on the LTO-linked code.

### Rejected experiments

- Register-blocked M2L accumulator chunks for FP32 (kept for FP64 only):
  M 8.6 -> 10.3 ms, H 20.1 -> 26.0 ms; plain axpy rows win for FP32.
- Fixed 128-target M2L blocks: only four blocks per level at S, 25 % slower
  than the row kernel there; replaced by at least four blocks per thread per
  level.
- M2L block schedule for small matrix sets (S FP32, 0.8 MB of matrices):
  0.31 -> 0.37 ms M2L; gated on 1 MiB.
- FP32 pair arithmetic in the position-based P2P: no faster than FP64
  (13 cycles per pair either way) and 500x larger FP32 differences; FP64
  arithmetic kept.
- Per-source `-fno-math-errno` compile option: ignored by the LTO link; the
  target-level compile and link options are used instead.
- The one-thread-per-output M2L kernel (the baseline) is retained only as the
  fallback for more than 512 coefficients.

### Out-of-scope findings (construction, task 3C)

- Construction still builds the canonical P2P operator for every point plan
  (about 5 s of the S case's setup at 10k points and the dominant part of the
  L case's setup) although the CPU backends now release it immediately after
  the cache is written; a CPU point plan with the cache disabled could skip
  it entirely, and the cache payload could carry the topology instead of
  16.3M tensors.
- The far-field packing and the M2L block schedule are derived at
  construction (about 0.04 ms at 24 particles, tens of milliseconds at M);
  the packing could be written by the geometry-cache load directly instead of
  building the sparse maps first.
- `benchmark_uniform_fmm --regular-grid` still needs power-of-two-times-odd
  counts.
- The installed `libcdfmm_c.so` in the conda environment shadows the freshly
  built one at test time (the conda `LDFLAGS` put `$PREFIX/lib` first in
  the executables' `RPATH` with `--disable-new-dtags`), so the C-ABI test
  process aborted with a heap corruption when the solver layout changed and
  then hung inside the abort handler for 20 minutes. All CTest runs of this
  task were made with `LD_PRELOAD=<build>/libcdfmm_c.so`; the build should
  either strip that rpath for test executables or install after building.
  Not fixed here (build-system change).

### Remaining CPU bottlenecks (final HEAD)

- Portable M2L is now the largest phase again: 8.2 / 12.0 ms at M (FP32 /
  FP64) for 3.07 GFLOP, i.e. 370 / 250 GFLOP/s, about 30-40 % of the AVX2
  FMA peak of eight P-cores. The kernel loads one matrix column per FMA
  vector from L1; a true register-tiled micro-GEMM over several interactions
  sharing a matrix (the schedule already provides the runs) is the next step.
- The position-based P2P (3.1 ms at M, 16.3M pairs, about 5 cycles per pair
  per thread) is bound by the packed sqrt/div throughput; an `rsqrt`
  approximation with one Newton step would halve it but changes rounding and
  was not tried.
- oneMKL M2L (12.0 / 25.8 ms at M) is now slower than the portable executor
  at every measured size; its gather/translated scratch traffic (about 1 GB
  per evaluation at M) is the limit, and the `cblas_?gemm` calls are 55 % of
  the phase. A blocked gather/GEMM/scatter over the same class-sorted
  schedule would remove most of that traffic (not done: the portable path
  already wins).
- P2M/L2P/M2M/L2L together are 1.6 / 3.5 ms at M; P2M and L2P stream
  24C bytes per particle and are at the DRAM roof again.
- Serial passes (identity gather with a fixed map, near-field combine,
  resets) remain 0.3-0.5 ms at M; not attacked.

### Final results (final HEAD `cad666d` versus `CPU_PERF_BASELINE = e4f1c79`)

Same builds (identical configuration), hardware, workloads and repeat counts
as the baseline table; evaluation medians and per-phase means at 8 threads
[us]; the sampled accuracy against the exact point reference is unchanged in
every case.

| Case | Backend | Prec | Baseline | Final | Speedup | P2M | M2M | M2L (gather / multiply / scatter) | L2L | L2P | P2P |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| S (10k, p4, d3) | portable | FP32 | 4270 | 1239 | 3.45x | 33 | 26 | 296 | 26 | 19 | 804 |
| S | portable | FP64 | 6478 | 1341 | 4.83x | 34 | 30 | 408 | 26 | 22 | 777 |
| S | oneMKL | FP32 | 3070 | 1186 | 2.59x | 33 | 29 | 248 (56 / 144 / 47) | 25 | 21 | 794 |
| S | oneMKL | FP64 | 5600 | 1376 | 4.07x | 33 | 31 | 383 (83 / 227 / 73) | 26 | 47 | 777 |
| M (50k, p6, d4) | portable | FP32 | 99684 | 12682 | 7.86x | 166 | 296 | 8789 | 125 | 499 | 3246 |
| M | portable | FP64 | 115401 | 18010 | 6.41x | 206 | 730 | 12387 | 162 | 1047 | 3128 |
| M | oneMKL | FP32 | 35657 | 16631 | 2.14x | 188 | 289 | 11952 (2798 / 6486 / 2668) | 115 | 623 | 3113 |
| M | oneMKL | FP64 | 62849 | 32357 | 1.94x | 605 | 745 | 25754 (6343 / 14145 / 5264) | 287 | 1150 | 2914 |
| H (50k, p8, d4) | portable | FP32 | 263430 | 25508 | 10.33x | 685 | 822 | 18777 | 348 | 959 | 3114 |
| H | portable | FP64 | 325641 | 44165 | 7.37x | 1037 | 1702 | 35533 | 674 | 1953 | 3168 |
| H | oneMKL | FP32 | 58170 | 27651 | 2.10x | 531 | 815 | 21557 (5059 / 12000 / 4497) | 291 | 955 | 3070 |
| H | oneMKL | FP64 | 98036 | 53342 | 1.84x | 1068 | 1711 | 43922 (10624 / 24588 / 8708) | 529 | 1893 | 2979 |
| L (100k, p6, d4) | portable | FP32 | 127956 | 23930 | 5.35x | 554 | 305 | 8951 | 254 | 1189 | 12047 |
| L | portable | FP64 | 164951 | 30620 | 5.39x | 948 | 736 | 12181 | 523 | 2211 | 11268 |
| L | oneMKL | FP32 | 64911 | 26996 | 2.40x | 544 | 298 | 12207 (2828 / 6608 / 2771) | 160 | 1156 | 11174 |
| L | oneMKL | FP64 | 113283 | 43840 | 2.58x | 941 | 745 | 25784 (6342 / 14156 / 5285) | 312 | 2225 | 11050 |

(The M2M column of the final rows is the per-evaluation mean; small phases
fluctuate by tens of microseconds between runs.)

Thread scaling, M case, final evaluation median [ms] (baseline in
parentheses):

| Threads | portable FP32 | portable FP64 | oneMKL FP32 | oneMKL FP64 |
|---:|---:|---:|---:|---:|
| 1 | 85.7 (669.6) | 115.0 (748.0) | 70.5 (113.3) | 117.2 (164.3) |
| 2 | 44.4 (345.6) | 59.1 (381.7) | 41.5 (67.8) | 69.5 (106.0) |
| 4 | 23.3 (179.0) | 31.4 (200.0) | 23.3 (44.1) | 41.0 (74.3) |
| 8 | 12.7 (99.7) | 18.0 (115.4) | 16.6 (35.7) | 32.4 (62.8) |

The portable path scales 6.8x (FP32) / 6.4x (FP64) from 1 to 8 threads;
single-thread it is 7.8x / 6.5x faster than the baseline, so the gains are
per-core, not a parallelisation artefact.

**Portable versus oneMKL.** At the baseline oneMKL won every case above S
(2.8x at M FP32). At the final HEAD the portable executor wins at every size
and precision (M: 12.7 vs 16.6 ms FP32, 18.0 vs 32.4 ms FP64; H: 25.5 vs
27.7 / 44.2 vs 53.3; L: 23.9 vs 27.0 / 30.6 vs 43.8; S within 5 %). The
oneMKL M2L itself is 1.4x (FP32) to 2.1x (FP64) slower than the portable
class-sorted kernel because its gather/translated scratch traffic dominates
(about 1 GB per evaluation at M). `StaticMatrixBackend::Portable` is already
the default, so no automatic selection changed; recorded for Phase 3D: there
is no longer a size at which oneMKL should be preferred for M2L on this
machine.

**oneMKL call structure (final).** Per evaluation at M: 4 levels x up to 316
`cblas_?gemm` calls (about 900 with non-empty groups), M = K = 49,
N = 13-1766 columns; two `mkl_set_num_threads_local` calls per thread per
level; the gather is 23 %, the multiply 54 % and the (now parallel) scatter
22 % of the phase.

**`cuda-partial` (secondary metric, CUDA build with the same compiler and
flags plus CUDA, baseline built at `e4f1c79`).** Evaluation medians [us] with
the CPU hierarchy phases:

| Case | Prec | Baseline | Final | Speedup | P2M base -> final | M2M | L2L | L2P | CPU hierarchy base -> final |
|---|---|---:|---:|---:|---|---|---|---|---|
| S | FP32 | 380 | 211 | 1.80x | 62 -> 14 | 94 -> 47 | 79 -> 17 | 23 -> 14 | 258 -> 92 |
| S | FP64 | 508 | 368 | 1.38x | 73 -> 19 | 94 -> 177 (contains the M2L round trip wait) | 80 -> 22 | 55 -> 192 | see note |
| M | FP32 | 7114 | 1714 | 4.15x | 1469 -> 378 | 2034 -> 408 | 1922 -> 374 | 882 -> 78 | 6307 -> 1238 |
| M | FP64 | 10549 | 4906 | 2.15x | 2056 -> 972 | 2056 -> 2197 (contains the M2L wait) | 1975 -> 872 | 1773 -> 690 | see note |
| H | FP32 | 17276 | 3008 | 5.74x | 2792 -> 722 | 5839 -> 725 | 5832 -> 718 | 1451 -> 83 | 15914 -> 2248 |
| H | FP64 | 25863 | 10226 | 2.53x | 3770 -> 1762 | 6342 -> 5431 (M2L wait) | 6364 -> 1632 | 3262 -> 692 | see note |
| L | FP32 | 10117 | 3249 | 3.11x | 3167 -> 952 | 2054 -> 457 | 1966 -> 957 | 1836 -> 153 | 9023 -> 2519 |
| L | FP64 | 16083 | 7595 | 2.12x | 4310 -> 2080 | 2342 -> 2208 (M2L wait) | 2134 -> 1993 | 3774 -> 2386 | see note |

Note: in the hybrid FP64 rows the CPU hierarchy is now shorter than the
device M2L (2.2 ms at M, 5.4 ms at H), so the M2M/L2P columns absorb the
wait for the GPU; the FP64 hybrid is GPU-bound and its remaining time is the
FP64 M2L kernel discussed in 3A. The FP32 hybrid is 1.2-1.5x slower than
`cuda-full` (M 1.71 vs 0.77 ms) instead of 9x.

### GPU regression check

`cuda-full` at the final HEAD, same build tree, against the policy-commit
numbers: S 189 / 301 us (188 / 302), M 767 / 2925 us (758 / 2909),
32^3 lattice depth 3 with `--spatial-layout regular-grid` 532 / 1343 us
(532 / 1345) with the dictionary packing still selected; all within the
+-1 % noise floor. No CUDA kernel, plan or policy changed in 3B.

### Validation

- CTest, full suites with the freshly built `libcdfmm_c.so` preloaded (see
  the RPATH finding above): portable 202/202, oneMKL 202/202, CUDA
  (CUDA + CPU side) 202/202 at the final HEAD; the statistics-based cases
  (`oneMKL grouped executor retains canonical levels and scratch`, `cold and
  warm binary caches preserve complete plan results`, `FMM initialisation
  reports requested and resolved options`, the C++/Python packing
  assertions) were updated to the new accounting and packing names.
- Python: 138 passed / 8 skipped (portable module), 140 / 6 (oneMKL),
  143 / 3 (CUDA); the notebook storage estimator models the new resident
  containers exactly (`test_host_storage_estimate_matches_constructed_cpu_static_plan`).
- Numerics against the baseline module on a 20k-point p=6 depth-3 plan:
  FP64 field max difference 8e-16 of the field scale (median 4e-16), FP32
  3.5e-7 (median 3.6e-7); potential 5e-18 / 8e-7. Benchmark accuracy columns
  identical in every case.
- `git diff --check` clean.

**Phase 3B CPU / oneMKL evaluation is complete.**

### All backends at 8 threads, final state (evaluation medians [us])

`cuda-full` did not change in 3B; its S and M values are the regression
check above at the final HEAD, its H and L values are the Phase-3A final
table (state `bf511ce`, identical CUDA code). The CPU and hybrid columns are
the 3B final table; every case is 8 P-core threads, spherical basis, random
points, field output, fixed identity map.

| Case | Prec | portable | oneMKL | cuda-partial | cuda-full | portable / cuda-full |
|---|---|---:|---:|---:|---:|---:|
| S (10k, p4, d3) | FP32 | 1239 | 1186 | 211 | 189 | 6.6x |
| S | FP64 | 1341 | 1376 | 368 | 301 | 4.5x |
| M (50k, p6, d4) | FP32 | 12682 | 16631 | 1714 | 767 | 16.5x |
| M | FP64 | 18010 | 32357 | 4906 | 2925 | 6.2x |
| H (50k, p8, d4) | FP32 | 25508 | 27651 | 3008 | 1085 | 23.5x |
| H | FP64 | 44165 | 53342 | 10226 | 6107 | 7.2x |
| L (100k, p6, d4) | FP32 | 23930 | 26996 | 3249 | 1924 | 12.4x |
| L | FP64 | 30620 | 43840 | 7595 | 4749 | 6.4x |

For reference, `cuda-full` at the Phase-3A GPU baseline (`293144bf`) was
S 485 / 760, M 2397 / 4275, H 4986 / 8694, L 4307 / 7146 us (FP32 / FP64),
and the CPU paths at the 3B baseline (`e4f1c79`) are in the table above.

## P2P execution unification, geometry/backend coverage, and CudaPartial crossover (Phase 3 follow-up)

### Scope and starting point

Starting HEAD `af50b69` (`docs(agent): add the all-backend 8-thread comparison
including cuda-full`) on `refactor/architecture-v0.2`; work committed on the
worktree branch `worktree-p2p-unification`. Same machine, toolchain and
measurement rules as 3A/3B (RTX 5090, i9-14900KF with eight pinned P-core
threads, conda `cdfmm` with g++ 15.3 and nvcc 13.3, Release, medians of five
samples of twenty evaluations after three warm-ups, fixed geometry with
changing moments, field output, fixed identity map for point sources). All
numbers in this section come from one CUDA-enabled build tree
(`build-cuda`, tests + Python + benchmarks) so CPU and CUDA rows are from the
same binary. Construction time is reported separately and never enters the
evaluation medians.

Architectural outcome (details in `docs/static-p2p.md` and
`docs/architecture.md`): every one of the nine point/prism/tetrahedron
source-target pairs builds canonical tensors (prism/tetrahedron pairs via the
tetrahedron pair's polyhedron surface formulation); the dense leaf packing
carries the canonical identity marker so every stored-tensor executor (CPU
SoA/leaf/dictionary, CUDA canonical/compact/leaf/BSR/dictionary) obeys the
metadata instead of the geometry; periodic image records pack into leaf
blocks (image ordinals), merged BSR blocks and dictionary tokens; and
`UniformFmmOptions::p2p_packing` forces any packing a backend can execute.
`P2PExecutionPacking::PointGeometry` is the one deliberate geometry-specific
executor (point sources and point targets, CPU).

### Periodic point-geometry P2P (CPU, 8 threads)

Periodic point plans previously kept the SoA rows because the position-based
executor was policy-disabled. Its implementation already folds each leaf
record's image shift and identity marker into the gathered neighbourhood; the
periodic geometry-matrix and a dedicated field/potential test (dynamic and
fixed identity maps, both precisions) confirmed it against the SoA rows, so it
was enabled explicitly and then, on this evidence, in the automatic policy.
Fully periodic cubic cell equal to the root box, random points:

| Case | Precision | SoA rows: eval / P2P [us] | PointGeometry: eval / P2P [us] | P2P speedup |
|---|---|---:|---:|---:|
| S (10k, p4, d3), 4.47M pairs | FP32 | 3399 / 2504 | 1725 / 930 | 2.7x |
| S | FP64 | 5927 / 4826 | 1900 / 945 | 5.1x |
| M (50k, p6, d4), 16.3M pairs | FP32 | 22320 / 8764 | 16257 / 3151 | 2.8x |
| M | FP64 | 37462 / 16121 | 23569 / 3182 | 5.1x |

The periodic near field has 2.7x more pairs than the free-space one at S
(4.47M versus 1.62M) because every leaf sees 26 neighbours through the images,
so the DRAM-bound SoA rows (29-53 bytes per pair) lose even more against the
cache-resident positions than in free space. `selects_point_geometry_p2p()`
no longer excludes periodic plans; the periodic potential path uses the
executor's own record sweep.

### Finite-geometry tensor dictionary on the CPU (8 threads)

Exact prism->prism and tetrahedron->tetrahedron near fields, point far-field
models (identical hierarchy for every row), order 6, `benchmark_uniform_fmm`
via `benchmarks/run_p2p_packing_matrix.py --suite finite`. "regular" is a
lattice of identical bodies (body extent 0.9 of the spacing); "irregular"
gives every body its own record (size 0.6-1.0 of the regular one, permuted
tetrahedron axes) on the same lattice for N = 4096 and on random positions
for N = 32768. The dictionary column also gives the unique tensor count over
the pair count and the resident bytes of the dictionary versus the SoA rows.

| Case | Pairs | Prec | SoA rows: eval / P2P [us] | Dictionary: eval / P2P [us] | P2P speedup | Unique tensors | Dictionary / SoA bytes |
|---|---:|---|---:|---:|---:|---:|---:|
| prism regular 16^3, d3 | 681k | FP32 | 1058 / 139 | 965 / 66 | 2.1x | 248 | 0.95 MB / 28.0 MB |
| prism regular 16^3, d3 | 681k | FP64 | 1778 / 251 | 1496 / 84 | 3.0x | 344 | 1.6 MB / 52.5 MB |
| prism irregular 16^3, d3 | 681k | FP32 | 1083 / 161 | 1037 / 131 | 1.2x | 391775 (57 %) | 12.8 MB / 32.3 MB |
| prism irregular 16^3, d3 | 681k | FP64 | 1858 / 370 | 1869 / 366 | 1.0x | 774080 (100 %) | 40.6 MB / 60.6 MB |
| prism regular 32^3, d4 | 6.23M | FP32 | 12021 / 3014 | 9405 / 547 | 5.5x | 248 | 8.7 MB / 256 MB |
| prism regular 32^3, d4 | 6.23M | FP64 | 19703 / 5714 | 14392 / 763 | 7.5x | 344 | 14.9 MB / 480 MB |
| prism irregular random 32768, d4 | 6.98M | FP32 | 12403 / 3395 | 12019 / 2768 | 1.2x | 3.47M (50 %) | 114 MB / 286 MB |
| prism irregular random 32768, d4 | 6.98M | FP64 | 19994 / 6367 | 20253 / 6325 | 1.0x | 6.88M (99 %) | 361 MB / 537 MB |
| tetrahedron regular 16^3, d3 | 681k | FP32 | 1051 / 136 | 974 / 66 | 2.1x | 187 | 1.0 MB / 28.0 MB |
| tetrahedron regular 16^3, d3 | 681k | FP64 | 1783 / 270 | 1496 / 85 | 3.2x | 281 | 1.6 MB / 52.5 MB |
| tetrahedron irregular 16^3, d3 | 787k | FP32 | 1084 / 162 | 1035 / 133 | 1.2x | 391774 (50 %) | 12.8 MB / 32.3 MB |
| tetrahedron irregular 16^3, d3 | 787k | FP64 | 1865 / 371 | 1657 / 195 | 1.9x | 391973 (50 %) | 22.2 MB / 60.6 MB |

Observations: on a lattice of identical bodies the dictionary sees only the
displacement classes (248 prism / 187 tetrahedron variants for 681k-6.2M
pairs), the token stream (1 byte per pair) replaces 41-77 bytes of SoA row
per pair, and the CPU signed-dictionary kernel is 2-7.5x faster than the SoA
rows with 20-30x less resident memory. With one record per body the exact
tensors are almost all distinct (50 % unique in FP32 where quantisation merges
near-equal values, up to 100 % in FP64), yet the dictionary is never slower:
its token-plus-dictionary bytes are still below the SoA rows and the executor
is the same register-tiled kernel. The dictionary therefore behaves exactly as
predicted by the invariant: it compresses the *tensors*, and whether they came
from points, prisms or tetrahedra is irrelevant to its execution.

| tetrahedron regular 32^3, d4 | 6.23M | FP32 | 12113 / 3058 | 9441 / 547 | 5.6x | 187 | 8.7 MB / 256 MB |
| tetrahedron regular 32^3, d4 | 6.23M | FP64 | 19978 / 5734 | 14473 / 792 | 7.2x | 281 | 14.9 MB / 480 MB |
| tetrahedron irregular random 32768, d4 | 6.98M | FP32 | 12457 / 3405 | 12041 / 2758 | 1.2x | 3.47M (50 %) | 114 MB / 286 MB |
| tetrahedron irregular random 32768, d4 | 6.98M | FP64 | 20712 / 6430 | 19210 / 5184 | 1.2x | 3.47M (50 %) | 197 MB / 537 MB |
| point regular 32^3, d4 (SoA / PointGeometry / dictionary) | 6.23M | FP32 | 12104 / 2985 | PointGeometry 10185 / 1267; dictionary 9476 / 550 | 2.4x / 5.4x | 172 | 8.7 MB / 256 MB |

The 32^3 tetrahedron rows reproduce the prism rows exactly (same lattice, same
displacement classes), and on the point lattice the dictionary is 2.3x faster
than the position-based executor, so `SpatialLayout::RegularGrid` now selects
the dictionary on the CPU as well (point lattices with a fixed identity map,
finite lattices unconditionally), guarded by the built plan's token width
(see the policy section below).

Construction (recorded for Phase 3C, not optimised here): the 32^3 prism
plans took 598-653 s to construct because the canonical builder's generic
prism/point loop is serial, whereas the parallel tetrahedron pair loop built
the 32^3 tetrahedron plans in 121-170 s (16^3: 15-22 s); deriving the
dictionary itself costs 1-3.5 s at 6-7M pairs.

### CUDA packings on finite bodies (N = 32768, depth 4, 8 bodies per leaf, FP32)

Same lattices as above, every CUDA packing forced explicitly, both CUDA
backends. "P2P" is the device kernel time; evaluation medians include the
whole FMM (the `cuda-partial` total is dominated by its CPU hierarchy and is
the same for every packing).

| Bodies | Backend | canonical rows | leaf block | BSR(3) | dictionary source-warp | dictionary target-owned | dictionary microtiles | Unique tensors / token bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| prism regular | cuda-full eval / P2P | 769 / 536 | 505 / 117 | 568 / 191 | 379 / 73 | 363 / 66 | 357 / 37 | 248 / 1 |
| prism regular | cuda-partial eval / P2P | 1093 / 240 | 1096 / 102 | 1088 / 172 | 1075 / 58 | 1070 / 43 | 1076 / 31 | 248 / 1 |
| prism irregular | cuda-full eval / P2P | 713 / 424 | 526 / 142 | 595 / 217 | 569 / 276 | 507 / 266 | 604 / 510 | 3.47M / 4 |
| prism irregular | cuda-partial eval / P2P | 1149 / 251 | 1157 / 121 | 1117 / 197 | 1114 / 225 | 1148 / 216 | 1158 / 347 | 3.47M / 4 |
| tetrahedron regular | cuda-full eval / P2P | 761 / 520 | 507 / 117 | 572 / 191 | 380 / 69 | 364 / 65 | 359 / 39 | 187 / 1 |
| tetrahedron irregular | cuda-full eval / P2P | 721 / 430 | 525 / 142 | 599 / 217 | 569 / 278 | 506 / 266 | 603 / 509 | 3.47M / 4 |

(The `cuda-partial` P2P kernel times are shorter than `cuda-full`'s because
the hybrid's kernel does not share the GPU with a concurrent far field; the
tetrahedron `cuda-partial` rows equal the prism rows within noise.)
Persistent device bytes for the prism lattice: canonical 379 MB, leaf 232 MB,
BSR 329 MB, dictionary 88 MB (`cuda-full`).

Three conclusions, all geometry-independent as the invariant predicts:

1. Leaf blocks beat BSR(3) on finite bodies (117 vs 191 us regular, 142 vs
   217 us irregular) exactly as they do on points, and use 30 % less device
   memory. The "finite sources take BSR(3)" default was therefore a
   geometry-based rule without a measured basis; leaf blocks are now the
   general default for every geometry and BSR(3) is an explicit packing.
2. On a lattice of identical bodies the dictionary is 3x faster than leaf
   blocks (37-39 us with the power-of-two microtile executor at 8 bodies per
   leaf, the same executor the point calibration selects below 48 per leaf)
   and needs 2.6x less device memory, so the `RegularGrid` hint now selects
   it for finite bodies too. The dictionary executor calibration transfers
   because execution sees only tokens.
3. A dictionary that does not compress is slower than leaf blocks on CUDA
   (3.47M variants, four-byte tokens: 266-510 us versus 142 us), unlike on
   the CPU where it stayed neutral. The layout hint is therefore a prediction
   that the built plan verifies: a hint-selected dictionary is kept only when
   its token width is at most two bytes (at most 65535 variants, about 1.5 MB
   of dictionary), otherwise the plan falls back to the general default. A
   wrong hint now costs construction time only.

### CudaPartial versus CudaFull crossover (FP32, order 6)

`cuda-partial` = CPU P2M/M2M/L2L/L2P with GPU M2L and GPU P2P on separate
streams; `cuda-full` = everything on the device. Random points with a fixed
identity map, Auto policy (leaf blocks on both backends) and matched forced
packings; the `p2m`/`m2m`/`m2l`/`l2l`/`l2p` columns are per-evaluation means
of the host-side phases (for `cuda-partial` the M2L column includes the wait
for the device M2L, for `cuda-full` the M2M column absorbs the concurrent P2P
kernel).

| Points per leaf (N, depth 3, 512 leaves) | Pairs | cuda-partial eval [us] (P2P kernel / M2L incl. wait / P2M / L2P) | cuda-full eval [us] (P2P kernel) | full / partial |
|---|---:|---|---|---:|
| 8 (4096) | 0.75M | 209 (14 / 64 / 12 / 13) | 121 (12) | 0.58 |
| 16 (8192) | 3.0M | 236 (31 / 64 / 15 / 19) | 139 (25) | 0.59 |
| 32 (16384) | 12.0M | 488 (285 / 241 / 27 / 33) | 407 (293) | 0.83 |
| 48 (24576) | 26.9M | 860 (606 / 512 / 55 / 63) | 728 (616) | 0.85 |
| 64 (32768) | 48.0M | 1476 (1058 / 892 / 117 / 124) | 1220 (1077) | 0.83 |
| 96 (49152) | 108M | 2848 (2314 / 1750 / 350 / 303) | 2587 (2358) | 0.91 |
| 128 (65536) | 192M | 4641 (3949 / 2980 / 612 / 494) | 4542 (4031) | 0.98 |

Matched forced packings (leaf block on both, BSR(3) on both) change neither
ordering nor gap: with BSR(3) both backends are 8-25 % slower than with leaf
blocks and `cuda-partial` stays behind by the same margin (for example 64 per
leaf: 1594 vs 1468 us with BSR, 1476 vs 1220 us with leaf blocks).

| Other workloads | cuda-partial [us] | cuda-full [us] | full / partial |
|---|---:|---:|---:|
| regular lattice 32^3, d4, dictionary (both) | 1105 | 382 | 0.35 |
| regular lattice 32^3, d4, leaf block (both) | 1109 | 510 | 0.46 |
| regular lattice 64^3 (262144), d5, dictionary (both) | 12993 | 3346 | 0.26 |
| regular lattice 64^3, d5, leaf block (both) | 13007 | 3750 | 0.29 |
| random 100k, d4, Auto | 3237 | 1932 | 0.60 |
| random 200k, d5, Auto | 11331 | 3163 | 0.28 |
| exact prism 32^3, d4, leaf block (both) | 1100 | 506 | 0.46 |
| exact prism 32^3, d4, BSR(3) (both) | 1086 | 575 | 0.53 |
| exact tetrahedron 32^3, d4, leaf block (both) | 1092 | 505 | 0.46 |

`cuda-partial` does not win any measured regime. It approaches parity only
where the device P2P kernel dominates everything (96-128 points per leaf: the
gap shrinks to 9 % and 2 %), because there both backends are limited by the
same P2P kernel and the hybrid's CPU hierarchy (P2M 612 us, L2P 494 us,
against 154 and 53 us on the device) hides behind it. Wherever the far field
matters (deep trees, lattices, small occupancy) the hybrid is 1.7-4x slower.
The matched-packing rows isolate far-field placement: the ordering is a
property of where P2M/M2M/L2L/L2P run, not of the P2P packing.

### CudaPartial and CudaFull timelines (Nsight Systems, FP32)

Captures with `nsys profile --trace=cuda,nvtx,osrt` of `benchmark_uniform_fmm
--profile` (ten consecutive evaluations) from an NVTX-enabled Release build
(`build-profile`): M random (50k, p6, d4), 128 points per leaf (65536, d3),
the 32^3 lattice with the dictionary, and the 32^3 prism lattice with leaf
blocks, each on `cuda-partial` and on `cuda-full`. Per-evaluation device
operations, from the GPU trace of the steady-state evaluations (offsets from
the P2P upload that starts the evaluation):

`cuda-partial`, M random (period 1.82 ms, benchmark median 1.83 ms):

| Offset [us] | Stream | Operation | Duration [us] |
|---:|---|---|---:|
| 0 | P2P | H2D moments 600 KB | 40 |
| 50 | P2P | `leaf_p2p_kernel` | 277 |
| 330 | P2P | D2H near field 600 KB | 30 |
| 600 | M2L | H2D multipoles 917 KB | 50 |
| 660 | M2L | `apply_grouped_m2l_kernel` | 182 |
| 850 | M2L | D2H locals 917 KB | 36 |
| 1820 | P2P | next evaluation's H2D | |

The GPU is busy for about 0.66 ms of the 1.82 ms period. The P2P kernel is
finished 0.36 ms into the evaluation, long before the host needs it, so the
final wait is zero; P2P and M2L never overlap on the device (the M2L upload
starts 0.6 ms in, after the host P2M/M2M). The critical path is the host
hierarchy: P2M + M2M (about 0.6 ms) -> M2L round trip with the host blocked
(H2D + kernel + D2H + launch gaps, about 0.3 ms) -> L2L + L2P + near/far
combination + the next evaluation's preparation (about 0.9 ms). PCIe traffic
is 3 MB per evaluation (about 160 us of transfer time, all of it overlapped
except the M2L round trip's share); no transfer is repeated or avoidable.

`cuda-partial`, 128 points per leaf (period 4.6 ms, median 4.8 ms):

| Offset [us] | Stream | Operation | Duration [us] |
|---:|---|---|---:|
| 0 | P2P | H2D moments 786 KB | 46 |
| 50 | P2P | `leaf_p2p_kernel` | 3919 |
| 650 | M2L | H2D multipoles 115 KB | 9 |
| 2590 | M2L | memset (queued behind the P2P kernel) | 13 |
| 3250 | M2L | `apply_grouped_m2l_kernel` (stretched from ~180) | 295 |
| 3550 | M2L | D2H locals | 5 |
| 3930 | P2P | D2H near field 786 KB | 42 |
| 4570 | P2P | next evaluation's H2D | |

Here the device is the bottleneck for both backends (P2P kernel 3.92 ms of
a 4.6 ms period). The M2L work is *starved* behind the P2P kernel: its memset
and kernel start about 2 ms after the host issued them and the kernel runs
1.6x slower while sharing the SMs, so the host waits until +3.55 ms for the
locals and only then runs L2L + L2P (about 0.6 ms) plus the combination,
finishing at +4.57 ms even though the near field was ready at +3.97 ms. The
`cuda-full` capture of the same case shows the same contention from the
other side: its far-field kernels (`translate_targets`, `apply_grouped_m2l`)
stretch 2-6x while the 3.96 ms P2P kernel runs, and the far field finishes
just after the P2P, so both backends land at 4.3-4.6 ms per evaluation.

`cuda-partial`, 32^3 lattice with the dictionary (period 1.17 ms): P2P kernel
53 us, M2L kernel 173 us, transfers about 0.15 ms; the GPU is busy a quarter
of the time and the host hierarchy (P2M 133, M2M 124, L2L 110, L2P 132 us plus
the M2L round trip and preparation) is the whole critical path. The prism
lattice with leaf blocks behaves the same with a 103 us P2P kernel.

Answers to the timeline questions: GPU P2P overlaps the CPU upward work
completely in every case (final P2P wait is 0 everywhere); GPU M2L overlaps
P2P only when P2P is long enough to still be running, and then the two
*contend* rather than cooperate (M2L is delayed and stretched); the CPU
downward work starts as soon as the locals arrive; PCIe transfers are 1-3 MB
per evaluation and never on the critical path; serial host preparation and
combination are 0.2-0.4 ms per evaluation (about 15-20 % at M). The
"GPU P2P + fully CPU far field" variant was not built: the M2L round trip
costs 0.3 ms at M where the portable CPU M2L costs 8.8 ms, and at S the CPU
M2L (0.3 ms) already exceeds the whole device round trip, so no regime exists
in which it could win.

### Policy changes made on this evidence

All in `src/backend/cuda/execution_policy.cpp` / `src/fmm/execution_setup.cpp`;
none touches a kernel, the cache format or a cache key.

1. **Periodic point plans use `PointGeometry` on the CPU** (2.7-5.1x faster
   near field than the SoA rows; see the periodic table).
2. **Leaf blocks are the general CUDA default for every geometry.** The
   "finite sources take BSR(3) within `cuda_p2p_bsr_max_bytes`" rule was a
   geometry name standing in for a measurement; measured, leaf blocks beat
   BSR(3) on finite bodies by the same margin as on points. BSR(3) and
   canonical rows stay available through `p2p_packing`; the budget option is
   retained for compatibility and no longer steers the policy.
3. **`SpatialLayout::RegularGrid` selects the dictionary for any geometry on
   every backend** (point sources still need a fixed identity map), and the
   hint is verified on the built plan: a layout-selected dictionary is kept
   only when its token width is at most two bytes (at most 65535 variants),
   otherwise it is released and the general default applies. The CUDA
   dictionary executor keeps the Phase-3A occupancy calibration, which the
   finite-body rows reproduce (microtiles best at 8 per leaf).
4. **Periodicity no longer restricts any packing** (image ordinals in the
   leaf pairs, merged BSR blocks), so periodic point plans on CUDA moved from
   canonical rows to leaf blocks / the dictionary.
5. **Stream priority** (below).

`ExecutionBackend::Auto` still resolves to `CpuStatic`; no backend
auto-selection changed. Candidate future policy for Phase 3D, from the
crossover table: `cuda-full` should be preferred over `cuda-partial` in every
measured regime; the hybrid only reaches parity when the P2P kernel exceeds
about 4 ms per evaluation (128 or more points per leaf), where it is 1-2 %
ahead after the stream-priority change below.

### Accepted CudaPartial / CudaFull change: far-field stream priority

The timelines showed the hybrid's M2L stream starved behind the P2P kernel
whenever the two overlap (the host then waits for the locals before it can
run L2L/L2P), and the full backend's short far-field kernels stretched 2-6x
behind the same kernel. `src/backend/cuda/common/stream.hpp` creates a stream
at the device's greatest priority; the hybrid's M2L stream always uses it (a
pure win: the host is waiting on exactly that stream), and the full backend's
far-field stream uses it when the execution policy estimates the far field at
less than three times the P2P kernel (`far_field_stream_priority`, 18 ps per
pair versus 0.22 ps per M2L multiply-add), because a far field several times
longer than P2P otherwise starves the small P2P kernel to the very end
(+3 % at 200k points, depth 5). Same binary before/after, FP32, medians:

| Case | cuda-partial before -> after [us] | cuda-full before -> after [us] |
|---|---|---|
| S random (10k, p4, d3) | 211 -> 211 | 189 -> 180 (-5 %) |
| M random (50k, p6, d4) | 1714 -> 1684 (-2 %) | 767 -> 694 (-10 %) |
| 32 per leaf (16384, d3) | 488 -> 444 (-9 %) | 407 -> 398 (-2 %) |
| 64 per leaf (32768, d3) | 1476 -> 1312 (-11 %) | 1220 -> 1215 |
| 96 per leaf (49152, d3) | 2848 -> 2684 (-6 %) | 2587 -> 2574 |
| 128 per leaf (65536, d3) | 4641 -> 4442 (-4 %) | 4542 -> 4518 (-1 %) |
| 160 per leaf (81920, d3), after only | 6960 | 7043 |
| random 100k, d4 | 3237 -> 3244 | 1932 -> 1850 (-4 %) |
| random 200k, d5 | 11331 -> 11316 | 3163 -> 3260 (+3 %) with unconditional priority; equal priority restored by the rule |
| lattice 32^3, d4, dictionary | 1105 -> 1109 | 382 -> 361 (-5 %) with unconditional priority; equal priority under the rule (far field 3x the P2P estimate) |

Confirmation run with the conditional rule compiled in (`build-cuda/bench/priority3/`,
same settings, medians; the diagnostics print `cuda_policy.far_field_stream_priority`
so each row's decision is known):

| Case | Rule decision | cuda-full [us] | cuda-partial [us] |
|---|---|---|---|
| random 200k, d5 | equal priority | 3176 (baseline 3163, within noise; unconditional 3260) | - |
| lattice 32^3, d4, dictionary | equal priority | 363 (so the earlier 382 -> 361 was run-to-run noise, not priority) | - |
| S random (10k, p4, d3) | priority | 179 | - |
| M random (50k, p6, d4) | priority | 695 | 1714 |
| random 100k, d4 | priority | 1865 | - |
| 128 per leaf (65536, d3) | priority | 4562 | 4451 |
| M random FP64 (50k, p6, d4) | priority | 2833 | 4950 |

The deep-far-field regression is gone and the `cuda-full` M / S / 100k gains
are kept; the 128-per-leaf crossover (hybrid 2 % ahead, 4451 vs 4562)
reproduces. The hybrid's -2 % at M did not reproduce (1714, equal to its
baseline): the hybrid's reproducible gains are the 32-128 per leaf rows, where
the M2L stream actually competes with a long P2P kernel. The FP64 rows are the
first FP64 timing of the two CUDA backends in this study: `cuda-full` is
1.75x faster than `cuda-partial` there too.

The hybrid's M2L wait shrank from 892/1750/2980 us to 690/1372/2379 us at
64/96/128 per leaf (it is still delayed: block priority only takes effect as
P2P blocks retire). With the change, `cuda-partial` is 1-2 % ahead of
`cuda-full` at 128 and 160 points per leaf (4442 vs 4518, 6960 vs 7043 us) and
behind everywhere else; that is the only crossover found, it is within a few
percent, and it appears only where the near field alone takes several
milliseconds per evaluation.

### Rejected / not pursued

- *Pipelining the hybrid's M2L round trip level by level* so L2L on upper
  levels overlaps M2L on lower levels: bounded by the 0.3 ms round trip at M
  (17 % of the hybrid's 1.8 ms, still 2x behind `cuda-full`); a scheduler for
  that gain is not warranted.
- *GPU P2P with a fully CPU far field*: not built; the profile shows the M2L
  round trip (0.3 ms at M) is far below the portable CPU M2L (8.8 ms), and at
  S the CPU M2L alone already exceeds the device round trip.
- *Unconditional far-field priority in `cuda-full`*: +3 % on the deep
  far-field-dominated case; replaced by the conditional rule above.
- *Dictionary for irregular bodies by default*: neutral on the CPU, 1.9-3.6x
  slower than leaf blocks on CUDA; hence the token-width guard instead of a
  geometry rule.

### Validation (final HEAD of `worktree-p2p-unification`)

Four fresh trees, all with the conda `g++` 15.3 as C++ and CUDA host
compiler (`-DCMAKE_CXX_COMPILER`, `-DCMAKE_CUDA_HOST_COMPILER`): portable CPU
(`build-cpu`), CUDA (`build-cuda`), oneMKL (`build-mkl`), CUDA + oneMKL
(`build-cuda-mkl`). Full CTest passes 222/222 on each (tests of a backend
that is not compiled are skipped); `pytest python_tests` passes 140 / 145 /
142 / 147 with the rest skipped. `compute-sanitizer --tool memcheck` over the
`[identity]`, `[packing]` and `[geometry-matrix]` cases (every changed CUDA
P2P kernel path): 0 errors in 12 test cases; `--tool racecheck` over
`[identity]` and `[geometry-matrix]`: 0 hazards in 8 cases (no barrier
changed; the identity skip sits inside a barrier-free source loop).
The whitespace check is clean. `src/cache/` is untouched since `af50b69`:
the serialised records are the canonical operator's (whose per-block identity
flag already existed), the leaf/dictionary/BSR plans are derived after
loading, so cache format and keys are unchanged; the C ABI and Fortran
interface are untouched.

Toolchain observation (not fixed, recorded): the conda environment exports
`CXX=icpx`, `CC=icx` and `NVCC_PREPEND_FLAGS=-ccbin=icpx`. A oneMKL tree
configured without pinning the compiler is built by `icpx` 2026.1.1 and then
fails one test, `static triangular translations match M2M and L2L
references` (bitwise `m2m_add` versus static-operator equality; a last-bit
difference from icpx's default fast floating-point model), and nvcc 13.x
rejects icpx/clang 22 as a host compiler, so the CUDA + oneMKL configure
fails outright. Both trees pass fully once g++ is pinned, which is the
documented toolchain.

## Procedural vs precomputed point operators (Phase 3B.5)

### Scope, starting point and method

Question: for the analytically cheap point operators (P2P, P2M, L2P), when is
it faster to reconstruct the operator during every evaluation than to stream
a precomputed representation? Starting HEAD `38f1b98` (`docs(agent): record
the final four-tree validation and the icpx toolchain note`) on
`worktree-p2p-unification`, which `refactor/architecture-v0.2` was
fast-forwarded to at the start of the task. Same machine, toolchain and rules
as the previous Phase-3 stages: RTX 5090 (driver 595.84), i9-14900KF with
eight pinned P-core threads, conda `g++` 15.3 as C++ and CUDA host compiler
(the environment's `icpx`/`-ccbin=icpx` overridden), nvcc 13.x, Release, LTO,
`-march=native`; `benchmark_uniform_fmm` with the spherical basis, random
points (seed 314159) or the `--regular-grid` lattice, fixed identity map,
field output, `--no-direct --no-workload-comparison --warmups 3 --evaluations
20 --samples 5 --accuracy-targets 128`, medians of the five samples;
construction is reported separately and never enters the evaluation medians.
All CUDA-only rows come from one build tree (`build-cuda`) and were taken with
no other load; the CPU and hybrid rows were re-measured on an idle machine
after a first pass had been perturbed by concurrent host work (the perturbed
pass is not reported). Nsight Compute counters remain unavailable
(`ERR_NVGPUCTRPERM`); evidence is CUDA-event kernel timings, the phase
timers, byte counts and controlled variants.

Conceptual result (now in `docs/static-p2p.md` and `docs/architecture.md`):
the near-field invariant "geometry builds tensors, executors apply tensors"
describes where an operator is defined, not how an executor must hold it.
Every operator has a *precomputed* representation (tensor / coefficient rows
/ compressed packing built once) and, for point geometry, a *procedural* one
(the identical operator reconstructed from resident positions every
evaluation). Precomputation is an execution choice. Finite tiles keep it;
three point operators now have a measured procedural alternative.

### Part A: CUDA position-based point P2P (`PointGeometry` on CUDA)

Design: `src/backend/cuda/p2p/plan.cu`, `point_geometry_p2p_kernel`. One
warp per canonical list-1 record (target leaf, source leaf, image) with the
leaf-block lane layout (`stride` consecutive targets x `32/stride` source
slots), so nothing about the topology, ranges, identity handling or periodic
records changed. A lane reads one aligned position (16 bytes FP32 / 32 bytes
FP64) and one moment per source instead of six tensor components, recomputes
the pair with the shared `operators/p2p_point_kernel.hpp` formula (now
`__host__ __device__`, with an inverse-radius overload so an executor may
choose its inverse square root), removes the self pair without control flow
like the CPU executor, reduces the source slots with shuffles and adds the
record atomically. Uploaded per plan: the sorted positions (aliased when
sources and targets are the same points), one 32-byte (FP32) / 48-byte (FP64)
record per list-1 leaf pair, and the identity map; no canonical tensor is
read, and the host canonical/row operators are released after the cache is
written (as on the CPU). `P2PExecutionPacking::PointGeometry` is the public
selector on every backend; internally `cuda_policy::CudaP2PPacking::
PointGeometry`. `CudaPlanStatistics::p2p_geometry_bytes` reports the resident
positions.

Accuracy: the first version formed `x_t - x_s` from root-frame FP32
coordinates and lost about five digits to cancellation: the FP32 field
differed from the stored FP32 tensors by 9.4e-5 of the field scale, and its
error against the FP64 direct reference on 512 sampled targets was 5.2e-5
against 1.6e-6 for the stored tensors (50k points, depth 4, p 6). Storing
every position relative to its own leaf centre and folding the (exactly
representable) source-minus-target leaf-centre difference plus the image
shift into each record fixed it at zero run-time cost: the difference to the
stored tensors is now 4.6e-6 (50k, d4), 3.2e-6 (32768, d3) and 4.8e-6
(hybrid, 50k) of the scale, and the error against the direct reference is
1.582e-6 versus 1.583e-6 for the stored tensors (identical to three digits);
FP64 differs by 5.6e-16 (summation order). `tests/test_p2p_geometry_matrix.cpp`
now runs `PointGeometry` on both CUDA backends for every point pair,
free-space and periodic, both precisions, both layouts, against the FP64
dense reference at the unchanged tolerances (2e-4 FP32, 1e-10 FP64).

Random points, depth 3 (512 leaves), FP32, order 6, `cuda-full`, P2P kernel /
evaluation median [us] / persistent device MB:

| Points per leaf (N) | Pairs | leaf block | position-based | canonical rows | BSR(3) | kernel speedup vs leaf |
|---|---:|---|---|---|---|---:|
| 8 (4096) | 0.75M | 12.4 / 122 / 32 | 9.7 / 121 / 14 | 120 / 172 / 50 | 47 / 124 / 44 | 1.3x |
| 16 (8192) | 3.0M | 26.6 / 137 / 95 | 17.5 / 136 / 23 | | | 1.5x |
| 32 (16384) | 12.0M | 291 / 397 / 328 | 48 / 160 / 40 | 646 / 734 / 615 | 410 / 488 / 519 | 6.1x |
| 48 (24576) | 26.9M | 616 / 725 / 703 | 80 / 196 / 58 | | | 7.7x |
| 64 (32768) | 48.0M | 1071 / 1215 / 1225 | 126 / 248 / 75 | | | 8.5x |
| 96 (49152) | 108M | 2352 / 2587 / 2702 | 258 / 432 / 110 | | | 9.1x |
| 128 (65536) | 192M | 4031 / 4501 / 4747 | 536 / 723 / 144 | 8248 / 8779 / 9351 | 5002 / 5509 / 7816 | 7.5x |

Bytes per pair streamed: 24 (leaf block FP32) / 36 + index (BSR) / 24 + 12
index (canonical) versus about 28 bytes per *source* per record for the
position-based kernel, i.e. 28 / `stride` bytes per pair from L2 (all
positions of 65k points are 1 MB). The kernel is compute-bound: 2.8 ps per
pair at 128 per leaf (about 25 flops plus one inverse square root), against
18 ps for the DRAM-bound leaf blocks.

Other random workloads (FP32, kernel / evaluation / device MB, leaf block ->
position-based): S 10k d3 p4: 93 / 178 / 120 -> 23 / 114 / 13; 100k d4 p6:
1500 / 1853 / 1799 -> 505 / 791 / 237; 200k d5 p6: 721 / 3161 / 1466 -> 404 /
2865 / 613.

FP64 (random, depth 3): leaf block -> position-based, kernel / evaluation
[us]: 16 per leaf 116 / 395 -> 347 / 497; 64 per leaf 2090 / 2327 -> 3674 /
4066; 128 per leaf 7059 / 7657 -> 12300 / 13094. Recomputation is 1.7-3x
*slower*: the RTX 5090's FP64 rate is a small fraction of its FP32 rate, so
the compute-bound kernel loses to the bandwidth-bound tensors. FP64 keeps the
stored packings.

Regular lattices, FP32, order 6, `cuda-full` (`--regular-grid`; the
dictionary rows use the `RegularGrid` hint with its calibrated executor),
kernel / evaluation [us] / device MB:

| Lattice | Per leaf | Leaves | dictionary | position-based | leaf block |
|---|---:|---:|---|---|---|
| 16^3, d3 | 8 | 512 | 29.6 / 119 | 7.7 / 120 | |
| 32^3, d4 | 8 | 4096 | 38.8 / 362 / 88 | 36.7 / 372 / 83 | 117 / 513 / 232 |
| 64^3, d5 | 8 | 32768 | 210 / 3138 / 705 | 466 / 3179 / 663 | 990 / 3715 / 1931 |
| 32x16x16, d3 | 16 | 512 | 85 / 145 | 13.9 / 134 | |
| 64x32x32, d4 | 16 | 4096 | 196 / 592 | 254 / 575 | |
| 32x32x16, d3 | 32 | 512 | 211 / 285 | 29.8 / 158 | |
| 64x64x32, d4 | 32 | 4096 | 525 / 1206 | 548 / 917 | |
| 32^3, d3 | 64 | 512 | 440 / 536 / 158 | 82 / 215 / 71 | 654 / 802 / 1117 |
| 64^3, d4 | 64 | 4096 | 1607 / 2397 / 1339 | 1122 / 1885 / 546 | 6185 / 7137 / 10110 |

The dictionary kernel alone stays ahead only at 8 (and marginally 16) points
per leaf on lattices with thousands of leaves (2.2x at 64^3 depth 5), where
the position-based kernel's per-record reduction and atomics dominate its 64
pairs per record; in evaluation time the two are within 1.3 % there because
the far field dominates, and everywhere else the position-based kernel wins
(up to 7x kernel, 1.9x evaluation). Its resident memory is never larger. FP64
lattices: dictionary 183 / 2273 (32^3 d4) and 1148 / 1340 (32^3 d3) versus
position-based 366 / 2530 and 2484 / 2904 -> the FP64 dictionary stays.

Hybrid backend (`cuda-partial`, FP32, idle machine, kernel / evaluation /
device MB): M 50k d4 p6 leaf 287 / 1686 / 416 -> position-based 60 / 1689 /
26 (the CPU hierarchy is the critical path, so the evaluation is unchanged);
128 per leaf 3971 / 4452 / 4610 -> 392 / 1661 / 7.9.

Policy (`src/backend/cuda/execution_policy.cpp`): FP32 plans with point
sources and point targets select `PointGeometry` on both CUDA backends, on
any layout (the `RegularGrid` hint no longer brings the dictionary back for
FP32 points, which measured equal or faster in evaluation time with equal or
less memory); FP64 point plans and finite bodies keep the Phase-3A/3B rules
(leaf blocks, lattice dictionary). Explicit `p2p_packing` and
`use_reduced_symmetry_p2p` keep precedence. The stream-priority rule now uses
the measured cost per pair of the resolved packing and precision
(`p2p_picoseconds_per_pair`: leaf 18 / 37, canonical 43 / 86, BSR 26 / 52,
dictionary 8 / 26, position-based 3 / 64 ps, FP32 / FP64) instead of the
leaf-block constant, so with the cheaper kernel the far field is prioritised
less often. `CudaExecutionPolicyInputs` gained `effective_point_target`.

### Parts B and C: procedural point P2M and L2P

Audit of the retained operators (spherical, `C = (p+1)^2 = 49` at p 6):

- CPU packing (`backend/cpu/far_field/packing.hpp`): dense P2M rows of
  `3 C` scalars per source (12 C bytes FP32, 24 C FP64: 588 / 1176 bytes at
  p 6) and dense L2P rows of `4 C` scalars per target (field plus potential:
  784 / 1568 bytes), streamed every evaluation at the DRAM roof (0.125-0.25
  flop per byte). Point, prism and tetrahedron plans retain exactly the same
  bytes per point because the packing is dense (measured: 588 / 784 bytes per
  point on a 16^3 lattice for all three geometries).
- CUDA (`backend/cuda/far_field/executor.cu`): a CSR-by-output copy of the
  sparse canonical entries, about `3 C` (value, input) pairs per point, i.e.
  24 C bytes (1176 bytes at p 6, FP32) plus row offsets, read by eight-lane
  row groups.
- The mathematics is `M_lm = 1/(4 pi) sum m . grad R_lm(d)` and
  `H = -sum L_lm grad R_lm(dx)` with the real regular solid harmonics; the
  existing `regular_solid_harmonics` evaluates them mode by mode from
  Cartesian polynomial tables with heap allocation and is not usable per
  point in a kernel.

Implementation:

- `src/math/solid_harmonic_recurrence.hpp`: allocation-free host/device
  recurrence in the factorial normalisation `Q_l^m = r^l P_l^m e^{im phi} /
  (l+m)!`, whose three-term, diagonal and gradient relations have rational
  coefficients only (`d/dz Q_l^m = Q_{l-1}^m`, `(d/dx +- i d/dy) Q_l^m =
  +-Q_{l-1}^{m+-1}`); the repository's real mode is `f_{l,m}` times the real
  or imaginary part, with `f_{l,m} = sqrt((l-m)!(l+m)!) sqrt(2) (-1)^m`
  applied once per leaf from a `C`-entry table. Generic over the lane type
  (float, double or a SIMD pack). `tests/test_spherical_harmonics.cpp` checks
  values and gradients against the polynomial basis at orders 0-12 to 1e-12.
- `src/operators/point_expansion_kernel.hpp`: the P2M accumulation, the L2P
  field and potential dot products, and the only copies of the constants
  (`f/(4 pi)`, `-f`, `f`). Both executors call these kernels.
- CPU (`backend/cpu/far_field/procedural.{hpp,cpp}`, `lanes.hpp`): packs of
  four FP64 or eight FP32 points run the recurrence as SIMD lanes (a
  fixed-width pack type whose operators are constant-length loops), tails are
  zero-padded, P2M reduces the pack per mode, L2P scales the leaf's locals
  once. Orders 1-10 are compiled (`dispatch_order`).
- CUDA (`backend/cuda/far_field/procedural.cuh`): lane groups of
  `procedural_lanes_per_leaf(mean occupancy)` lanes (a power of two, 1-32)
  own one leaf each, so small leaves do not idle the warp; P2M lanes stride
  through the sources keeping the `C` Q-normalised sums in registers, reduce
  with shuffles and one lane scales into the leaf's multipole slot (no
  atomics); L2P lanes evaluate whole targets from the leaf's pre-scaled
  locals. Displacements from the leaf centre are uploaded once (16 / 32 bytes
  per point), so the far-field executor holds no coefficient rows for a
  procedural stage. Orders 1-10 instantiated.
- Selection: `UniformFmmOptions::point_expansion_execution`
  (`PointExpansionExecution::Auto / Precomputed / Procedural`, mirrored in
  Python), resolved once per stage in `UniformFmm::
  resolve_point_expansion_execution` (`src/fmm/execution_setup.cpp`) before
  the packings are built; `p2m_execution()` / `l2p_execution()` and the
  initialisation summary report the result; a procedural stage skips its row
  packing / CSR upload, so `p2m_operator_bytes` / `l2p_operator_bytes` drop to
  the factor tables (588 bytes in total at p 6 FP32). Finite far-field models
  (prism / tetrahedron P2M or L2P) keep their exact rows in every mode; the
  Cartesian basis keeps its rows (a procedural Cartesian evaluator would need
  the multi-index table on the device and was not needed for the production
  default). The cache still stores the canonical operators: format and keys
  are unchanged.

Measurements, precomputed -> procedural, per-evaluation means of the P2M and
L2P phases and the evaluation median [us]; device MB for `cuda-full`:

`cuda-full`, FP32:

| Case | P2M | L2P | evaluation | device MB |
|---|---|---|---|---|
| S 10k d3 p4 | 11.4 -> 13.9 | 7.5 -> 6.1 | 180 -> 167 | 120 -> 111 |
| M 50k d4 p6 | 40.2 -> 12.9 | 36.5 -> 8.2 | 689 -> 634 | 521 -> 421 |
| H 50k d4 p8 | 63.5 -> 18.7 | 57.4 -> 8.2 | 1120 -> 1019 | 603 -> 429 |
| L 100k d4 p6 | 75.8 -> 28.9 | 160 -> 87 | 1885 -> 1754 | 1799 -> 1599 |
| X 50k d4 p10 | 94.9 -> 38.1 | 88.2 -> 12.3 | 1672 -> 1536 | 728 -> 428 |
| M FP64 | 59.3 -> 110.7 | 94.1 -> 103.9 | 2911 -> 2933 | |

(These rows were taken with the leaf-block P2P forced on both variants so
that only the far-field stages differ; the device MB column includes the leaf
tensors.)

`cpu-static-matrix`, 8 threads, idle machine:

| Case | P2M | L2P | evaluation |
|---|---|---|---|
| S 10k d3 p4 FP32 | 13.8 -> 15.7 | 18.1 -> 15.9 | 1252 -> 1240 |
| M 50k d4 p6 FP32 | 436 -> 103 | 459 -> 91 | 12409 -> 11838 |
| H 50k d4 p8 FP32 | 887 -> 292 | 963 -> 331 | 25374 -> 24249 |
| L 100k d4 p6 FP32 | 1000 -> 212 | 1130 -> 170 | 22861 -> 20738 |
| X 50k d4 p10 FP32 | 1436 -> 380 | 1545 -> 507 | 63286 -> 60934 |
| M 50k d4 p6 FP64 | 1009 -> 175 | 1028 -> 198 | 17733 -> 16090 |

(The precomputed P2M at M FP32 measured 436 us in this session against 166 us
in the Phase-3B record on the same code path; the comparison above is
same-session and same-binary. The packed rows stream 588 bytes per point from
DRAM, the procedural stages run from cache-resident positions at 8-16 flops
per byte.)

`cuda-partial` (CPU hierarchy, device M2L and P2P), FP32, idle machine:

| Case | P2M | L2P | evaluation |
|---|---|---|---|
| S 10k d3 p4 | 14.1 -> 18.5 | 16.8 -> 15.1 | 210 -> 208 |
| M 50k d4 p6 | 359 -> 93 | 342 -> 76 | 1691 -> 1101 |
| H 50k d4 p8 | 710 -> 141 | 708 -> 111 | 2960 -> 1746 |
| L 100k d4 p6 | 941 -> 137 | 944 -> 111 | 3247 -> 2006 |
| X 50k d4 p10 | 1186 -> 207 | 1148 -> 162 | 5059 -> 3037 |
| M 50k d4 p6 FP64 | 950 -> 121 | 872 -> 101 | 4937 -> 3303 |

Accuracy (`accuracy_check.py` on 50k / 20k random points): procedural versus
precomputed fields differ by 5e-8 to 1.5e-7 of the field scale in FP32 (both
CUDA and CPU, p 4-10) and by 9e-17 (CUDA) / 7e-20 (CPU) in FP64; both have
the same error against the direct reference to four digits.
`tests/test_procedural_point_expansion.cpp` compares the two on every static
backend, both precisions, orders 1, 3, 6 and 10 (field; potential on the
CPU), checks the selection rules (Cartesian and out-of-range orders rejected
for an explicit request, finite far-field models kept precomputed) and the
warm cache.

Policy (`resolve_point_expansion_execution`): the CPU hierarchy (`CpuStatic`
and the CPU stages of `CudaPartial`) recomputes point P2M and L2P in both
precisions at orders 1-10 (equal within microseconds at p 4 with the rows
gone; 3-6x faster stages at p >= 6); `CudaFull` recomputes them for FP32 plans
(3-7x faster kernels from p 6, +2.5 / -1.4 us at p 4, about 100 MB less
device memory at M) and keeps the streamed rows for FP64, where the device
recurrence is slower (P2M 59 -> 111 us). No dependence on N or the layout
was needed.

### Automatic policy, final state (`Auto` everywhere, idle machine)

| Case | Backend | Prec | Before (38f1b98) [us] | After [us] | Speedup | Device MB before -> after |
|---|---|---|---:|---:|---:|---|
| S 10k d3 p4 | cuda-full | FP32 | 178 | 116 | 1.54x | 120 -> 4.0 |
| M 50k d4 p6 | cuda-full | FP32 | 689 | 450 | 1.53x | 521 -> 30.5 |
| H 50k d4 p8 | cuda-full | FP32 | 1120 | 704 | 1.59x | 603 -> 38.3 |
| L 100k d4 p6 | cuda-full | FP32 | 1885 | 657 | 2.87x | 1799 -> 36.5 |
| 128 per leaf (65536, d3) | cuda-full | FP32 | 4501 | 654 | 6.9x | 4747 -> 13.4 |
| 200k d5 p6 | cuda-full | FP32 | 3161 | 2613 | 1.21x | 1466 -> 208 |
| lattice 32^3 d4 (hint) | cuda-full | FP32 | 362 (dictionary) | 362 (position-based) | 1.0x | 88 -> 28.5 |
| lattice 64^3 d5 (hint) | cuda-full | FP32 | 3138 (dictionary) | 2901 (position-based) | 1.08x | 705 -> 225 |
| M 50k d4 p6 | cuda-full | FP64 | 2911 | 2823 | 1.03x (noise; leaf block, rows kept) | 972 -> 972 |
| lattice 32^3 d4 (hint) | cuda-full | FP64 | 2273 | 2276 (dictionary kept) | 1.0x | 123 -> 123 |
| M 50k d4 p6 | cuda-partial | FP32 | 1686 | 1081 | 1.56x | 416 -> 25.7 |
| 128 per leaf | cuda-partial | FP32 | 4452 | 906 | 4.9x | 4610 -> 7.9 |
| M 50k d4 p6 | cpu-static | FP32 | 12409 | 11713 | 1.06x | (host rows 59 MB -> 588 B) |
| lattice 32^3 d4 (hint) | cpu-static | FP32 | 9476 (record) | 9175 (dictionary kept) | | |

The M FP32 `cuda-full` evaluation is now 450 us (P2P 58 us, P2M 12 us, L2P
6 us, M2L 174 us): 1.7x faster than the Phase-3B all-backend table (767 us)
with 17x less device memory.

### CudaPartial check

With procedural P2M/L2P on the hybrid's CPU hierarchy and the position-based
P2P on both backends, `cuda-full` beats `cuda-partial` everywhere measured:
M 450 vs 1081 us, 128 per leaf 654 vs 906 us (FP32). The 1-2 % crossover in
favour of the hybrid at 128-160 points per leaf recorded in the previous
stage no longer exists: the procedural P2P shortened the device near field
seven-fold, so the hybrid's CPU hierarchy (P2M 76 + L2P 56 us at 128 per
leaf, plus the M2L round trip) is the critical path again. Phase-3D input:
prefer `cuda-full` in every measured regime.

### Regular-grid check

FP32 point lattices moved from the dictionary to the position-based kernel by
policy: evaluation 362 -> 362 us (32^3 d4) and 3138 -> 2901 us (64^3 d5) with
3x less device memory; the dictionary kernel itself is faster only at 8
points per leaf on the large lattice (210 vs 260-466 us), where the far field
dominates. FP64 lattices keep the dictionary (2276 us, unchanged), the CPU
lattice rule is unchanged (dictionary, 9175 us), and finite lattices are
untouched. The dictionary remains explicit for FP32 points
(`use_reduced_symmetry_p2p`, `p2p_packing`), verified by the updated
lattice tests.

### Finite-geometry check and retained-operator audit

Prism and tetrahedron paths are unchanged: every finite pair still executes
through stored tensors (the position-based packing is rejected for finite
near-field geometry on every backend), and finite far-field models keep
their exact P2M/L2P rows under every `point_expansion_execution` value
(`tests/test_procedural_point_expansion.cpp`, geometry-matrix tests). No
construction was optimised. Retained operators on a 16^3 lattice (4096
bodies, depth 3, 681k list-1 pairs, spherical p 6, `CpuStatic`):

| Geometry | P2M per source | L2P per target | P2P (SoA rows) | Shared M2M / M2L / L2L |
|---|---|---|---|---|
| point, precomputed | 588 B (FP32) / 1176 B (FP64) | 784 B / 1568 B | 0 (position-based; 93 kB scratch) | 281 kB / 3.04 MB / 281 kB (FP32) |
| point, procedural | 588 B in total (three tables) | included | 0 | same |
| prism, exact | 588 B / 1176 B | 784 B / 1568 B | 89 B per pair (60.7 MB) FP32, 165 B (112.5 MB) FP64 | same |
| tetrahedron, exact | 588 B / 1176 B | 784 B / 1568 B | 89 B per pair FP32, 165 B FP64 | same |

Construction times seen in that audit (Phase 3C input, not optimised): exact
prism P2P 64.5 s (the serial prism pair loop) against 15 s for the
tetrahedra; exact tetrahedron P2M 5.0 s and L2P 11.1 s for 4096 bodies.

### M2M / M2L / L2L audit

Left unchanged. The translation operators are a small set reused by every
interaction: eight level-scaled M2M and L2L class banks (281 kB each at p 6
FP32) and 316 M2L class matrices (3.04 MB), applied through the class-sorted
schedule so that one matrix stays in L1 across a run of interactions. No
operator is streamed once per interaction; recomputing a 49 x 49 transfer
matrix per interaction would cost far more than the cache-resident read, and
the M2L phase is FMA-bound (Phase 3B). No procedural M2L experiment was run.

### Rejected experiments and observations

- Root-frame FP32 displacement in the CUDA point kernel: correct but 30x less
  accurate than the stored FP32 tensors (5.2e-5 vs 1.6e-6 of the scale
  against the reference); replaced by leaf-relative positions, kept.
- Procedural point P2P in FP64 on CUDA: 1.7-3x slower than leaf blocks at
  every occupancy; remains an explicit packing, not selected.
- Procedural point P2M/L2P in FP64 on `CudaFull`: P2M 1.9x slower, L2P 1.1x
  slower than the streamed rows; remains explicit, not selected.
- Dictionary for FP32 point lattices: faster kernel only at 8-16 points per
  leaf on thousands of leaves, equal or slower evaluation, never less memory
  than the position-based kernel; the FP32 lattice rule was replaced. The
  FP64 lattice rule and finite lattices keep the dictionary.
- A procedural Cartesian P2M/L2P was not implemented (production default is
  spherical; it would need the multi-index table on the device).
- The precomputed CPU P2M at M measured 436 us against 166 us in the 3B
  record; not investigated here (same code path, same-session comparison
  used).

### Correctness, sanitizer and validation (final HEAD of `worktree-p2p-unification`)

- Correctness: `tests/test_p2p_geometry_matrix.cpp` (every geometry pair,
  every packing of every available backend including the CUDA `PointGeometry`
  kernel, both precisions, both layouts, free-space and periodic, against the
  FP64 dense reference), `tests/test_procedural_point_expansion.cpp`
  (procedural versus precomputed P2M/L2P on CpuStatic, CudaPartial and
  CudaFull, FP32 and FP64, orders 1/3/6/10, field and potential, selection
  rules, warm cache), the recurrence test at orders 0-12, and the updated
  policy, packing, precision and Python tests. Arithmetic differences are
  quantified above (FP32 P2P 3-5e-6 of the field scale, FP32 P2M/L2P
  5e-8-1.5e-7, FP64 at most 1e-15); no tolerance was loosened (FP32
  geometry-matrix tolerance 2e-4, FP64 1e-10; the new procedural test uses
  5e-5 FP32 / 1e-11 FP64 against the precomputed rows).
- `compute-sanitizer --tool memcheck` and `--tool racecheck` over the
  position-based P2P kernel (`[packing]` cases, both backends) and over the
  procedural P2M/L2P kernels through the benchmark driver (FP32 and FP64,
  orders 4, 6 and 10, `cuda-full` and `cuda-partial`): 0 errors, 0 hazards.
- Four fresh trees with the conda `g++` 15.3 pinned as C++ and CUDA host
  compiler (`PYTHON_EXECUTABLE` pinned to the environment's Python 3.11 so a
  fresh tree does not pick the base-environment interpreter): portable CPU,
  CUDA, oneMKL, CUDA + oneMKL. Full CTest 227/227 on each (tests of an
  absent backend are skipped); `pytest python_tests` 140 / 145 / 142 / 147
  passed with the rest skipped. `git diff --check` clean. `src/cache/`,
  `include/cdfmm/c_api.h`, `src/bindings/` and `fortran/` are untouched, so
  cache format and keys, the C ABI and the Fortran interface are unchanged;
  `CudaPlanStatistics`, `StaticExecutionPlan` consumers and
  `UniformFmmOptions` gained fields without changing existing members.

## Exact finite-geometry procedural vs precomputed operators (Phase 3B.5b)

### Question, starting point and method

Question: the previous stage showed that the analytically cheap *point*
operators are often faster reconstructed during every evaluation than streamed
from precomputed rows. Is precomputation actually faster for the exact
operators of uniformly magnetised prisms and tetrahedra, which are the primary
scientific use case? Starting HEAD `551c790` (`docs(agent): record the
Phase-3B.5 correctness, sanitizer and four-tree validation`) on
`worktree-p2p-unification`, a clean fast-forward of
`refactor/architecture-v0.2` (`38f1b98`); that branch was checked out in the
main working copy, so the work continued on `worktree-p2p-unification` and
`refactor/architecture-v0.2` was fast-forwarded at the end. Same machine and
rules as the earlier Phase-3 stages: i9-14900KF with eight pinned P-core
threads (`OMP_NUM_THREADS=8`, `OMP_PLACES={0},{2},...,{14}`,
`OMP_PROC_BIND=close`), RTX 5090 (driver 595.84, 32 GB, 96 MB L2), conda `g++`
15.3.0 as C++ and CUDA host compiler (the environment's `icx`/`icpx` and
`NVCC_PREPEND_FLAGS` overridden per tree), nvcc 13.3, Release, LTO,
`-march=native`. Fresh pinned trees `build-3b5b-cpu` (portable) and
`build-3b5b-cuda` (CUDA); the machine was otherwise idle.

Two representations of every exact finite operator were measured with the new
`benchmark_operator_representation` (`docs/benchmarks.md`, "Operator
representation: precomputed versus procedural"):

- *precomputed*: the production builders construct the operator once
  (`build_static_p2p_operator`; `operators::p2m::build_cuboid` /
  `build_tetrahedron`; `operators::l2p::build_cuboid` / `build_tetrahedron`),
  the production packings derive their execution form (particle-row SoA, dense
  leaf blocks, the signed tensor dictionary, the dense `PackedP2M` /
  `PackedL2P` rows, and on CUDA the leaf-block and dictionary plans) and the
  production apply functions stream it on every update;
- *procedural*: only the sorted positions, the body records, the list-1
  topology and the per-body invariants a builder already hoists
  (`PreparedTetrahedron`, `PolyhedronBody` surfaces) are retained, and the
  *same* per-pair or per-body builder the canonical construction calls
  reconstructs the operator during every update and applies it at once.
  Nothing per pair or per body is stored.

Both are compared against the FP64 canonical operator on the final moments of
every run (`max_relative_error`); the moments (or local coefficients) change
on every update and every result is checksummed, so no work can be elided.
Construction is timed separately and never enters an update time. Medians of
five samples of twenty updates; a procedural run that would exceed a 60 s
budget per precision first reduces its evaluations per sample and only then
truncates the leaf traversal, recording the measured fraction (the procedural
paths are compute-bound, so the truncation does not change the per-pair cost).

Two modes bracket the memory behaviour. *Hot* repeats one small set (512 pairs
or 64 bodies) single-threaded, so the stored operator stays in L1/L2 and the
number is the intrinsic arithmetic cost. *Streaming* traverses the complete
list-1 neighbourhood of a lattice with all eight threads, as the FMM near
field does: depth 3 with 2^3 bodies per leaf (4096 bodies, 512 leaves, 681k
pairs) and, for the pairs whose construction allows it, depth 4 (32768 bodies,
6.2M pairs), whose stored tensors reach 176 MB and so exceed both the 36 MB
LLC and the 96 MB GPU L2. The hot P2P sets cover the numerical branches of the
exact kernels: the finite self interaction (the physical demagnetisation
tensor at zero displacement), face-adjacent bodies, random list-1 lattice
offsets in a regular and a jittered layout, quarter-size bodies at list-1
offsets, and the far-separation safeguard at ten summed circumradii, beyond
the `polyhedron_far_separation_factor = 8` switch to the 216-point Gauss
average of the source point tensor. A point source has no self pair: it is the
singular one the identity map excludes, exactly as in production.

### Part A: what the exact finite operators cost to build and to hold

| Operator | Builder | Formulation | Transcendentals / pair | Stored form | Device-callable as written? |
|---|---|---|---|---|---|
| point <-> prism P2P | `rectangular_prism_point_tensor` (both directions by reciprocity) | MagTense f/g/h corner sums and F/F log ratios, `long double` | about 24 atan + 24 log + hypot | 6 scalars: 24 B FP32 / 48 B FP64 per pair | now yes, through the shared precision-generic kernel |
| prism -> prism P2P | `rectangular_prism_rectangular_prism_tensor` | averaged F1/F2 antiderivatives, 8 x 8 corner alternating sums, `long double` | a few hundred atan/log | same 6 scalars | mechanically, but ruled out by cost |
| tetrahedron <-> point P2P | `tetrahedron_point_tensor` / `point_tetrahedron_tensor` | four oriented face contributions: frames, cancellation-free atanh edge terms, Van Oosterom solid angle | about 16 log/atanh + 4 atan2 + 12 acos | same 6 scalars | no: `std::sort` frames, exceptions |
| tetrahedron -> tetrahedron P2P | `tetrahedron_tetrahedron_tensor_prepared` | analytical Galerkin triangle-pair Laplace integral over 16 face pairs, 10 when coincident; 6^3 Gauss average of the source point tensor beyond eight summed circumradii | hundreds | same 6 scalars | no: recursion, `std::sort`, exceptions |
| prism <-> tetrahedron P2P | `polyhedron_pair_tensor` | the same double surface integral over 12 x 4 = 48 face pairs | several hundred | same 6 scalars | no: `std::variant`, heap surfaces, recursion |
| prism P2M / L2P | `build_cuboid` | exact prism-averaged monomials `J_beta(d,h)` contracted with the Cartesian polynomial table of `R_lm` and its gradient | none; `terms x 3` monomial averages, `O(p)` each (172 terms at p 6, 470 at p 10) | dense `3 C` / `4 C` rows | logic yes, heap and exceptions no |
| tetrahedron P2M / L2P | `build_tetrahedron` | exact simplex Dirichlet moments through a barycentric expansion of every monomial | none; hundreds of `O(n^4)` expansions per body, each heap-allocating | dense `3 C` / `4 C` rows | no: heap allocation per call, exceptions |

Retained precomputed bytes per body for P2M / L2P are `12 C` / `16 C` (FP32)
and `24 C` / `32 C` (FP64) with `C = (p+1)^2`: 300 / 400 B at p 4, 588 / 784 B
at p 6, 972 / 1296 B at p 8 and 1452 / 1936 B at p 10 in FP32. They do not
depend on the geometry, only their values do, so a finite source costs exactly
what a point source costs to hold.

Two construction properties matter for the amortisation below and are
recorded, not changed (they are Phase-3C material). First, only two of the
canonical pair loops are parallel: the mixed prism/tetrahedron path
(`src/operators/p2p.cpp:239`) and the tetrahedron pair path (`:388`). The
generic loop that serves every point/prism combination and both
point/tetrahedron directions is serial, as is the per-leaf P2M and per-target
L2P plan construction, which matches `src/fmm/plan_preparation.cpp`. A
construction time below is therefore serial for the point/prism families and
eight-threaded for the polyhedron families. Second, the production prism
tensor evaluates its MagTense formulas in `long double`: reconstructing the
identical operator through the shared kernel in `double` is 2.7x faster and
still agrees with the `long double` result to 4.5e-15 of the field scale, and
in `float` 6.2x faster at 3.5e-6 (hot prism -> point, FP64 row: 2.83 us,
1.03 us and 0.45 us per pair respectively).

### Part B and C: exact finite P2P, hot and streaming

Hot, one cache-resident operator set, single thread, FP64; the stored column
is the range over the separation classes of the fastest stored packing:

| Pair | stored apply | self | adjacent | list1 | small-far | far-safeguard |
|---|---|---|---|---|---|---|
| point->point | 2.35-7.71 ns | n/a | 19.8 ns | 31 ns | 14.8 ns | 11.4 ns |
| point->prism | 2.02-2.36 ns | n/a | 1.66 us | 2.53 us | 2.46 us | 2.77 us |
| prism->point | 1.96-2.29 ns | 1.64 us | 1.66 us | 2.53 us | 2.46 us | 2.76 us |
| prism->prism | 1.96-2.57 ns | 92.4 us | 93.9 us | 99.4 us | 95.8 us | 93.9 us |
| point->tetrahedron | 2.09-2.36 ns | n/a | 2.17 us | 2.02 us | 2.12 us | 2.02 us |
| tetrahedron->point | 1.97-2.53 ns | 2.51 us | 2.17 us | 2.03 us | 2.12 us | 2.03 us |
| prism->tetrahedron | 2.26-2.30 ns | 691 us | 807 us | 952 us | 675 us | 580 us |
| tetrahedron->prism | 2.18-2.27 ns | 690 us | 802 us | 949 us | 571 us | 434 us |
| tetrahedron->tetrahedron | 1.96-2.27 ns | 30.2 us | 139 us | 277 us | 427 us | 434 us |

Point -> point is the control, and it is not the production procedural point
path: it runs the same generic per-pair reconstruction loop as the finite
pairs, so it measures what that loop costs when the operator itself is nearly
free. At 3x to 8x it separates the cost of the loop structure from the cost of
the exact geometry, and the finite pairs add another two to five orders of
magnitude on top of it. (The production point procedural executor is the fused
`PointGeometry` sweep of Phase 3B.5a, a different implementation that beats
the stored tensors; it is not what this row measures.) Every finite pair is
between 670x and 540,000x slower reconstructed, and the ordering follows the
mathematics of Part A: one
analytical prism or tetrahedron point tensor costs about 2 us, a
target-averaged prism pair about 95 us, a tetrahedron pair 30 us coincident to
434 us through the far-separation quadrature, and a prism/tetrahedron pair,
whose 48 face pairs each run the recursive Galerkin integral, up to 1.03 ms.
The far-separation safeguard is not a cheap path: at ten summed circumradii
the 216-point Gauss average of the source point tensor costs 434-580 us for a
mixed pair, more than the analytical integral it replaces for a tetrahedron
pair at contact (30 us).

Streaming, complete list-1 neighbourhood, eight threads. `K_break_even` is
`(T_build - T_procedural_setup) / (T_procedural - T_apply)` in complete field
updates; the procedural setup is zero because the procedural path retains
nothing:

FP32:

| Pair | layout | pairs | T_build | best stored | apply/pair | procedural/pair | ratio | stored bytes | K_break_even |
|---|---|---|---|---|---|---|---|---|---|
| point->point | regular | 677,376 | 82.1 ms | tensor-dictionary-1B | 0.0933 ns | 1.37 ns | 14.7x | 951 kB | 95 |
| point->point | irregular | 677,376 | 126 ms | tensor-dictionary-4B | 0.113 ns | 1.38 ns | 12.2x | 11.1 MB | 1.5e+02 |
| point->prism | regular | 677,376 | 1.75 s | tensor-dictionary-1B | 0.0937 ns | 309 ns | 3.29e3x | 952 kB | 8.4 |
| point->prism | regular | 6,196,736 | 16.2 s | tensor-dictionary-1B | 0.0802 ns | 308 ns | 3.83e3x | 8.65 MB | 8.5 |
| point->prism | irregular | 6,196,736 | 20.5 s | tensor-dictionary-4B | 0.42 ns | 355 ns | 844x | 176 MB | 9.3 |
| prism->point | regular | 681,472 | 1.75 s | tensor-dictionary-1B | 0.0926 ns | 308 ns | 3.32e3x | 952 kB | 8.4 |
| prism->point | regular | 6,229,504 | 16.2 s | tensor-dictionary-1B | 0.0794 ns | 307 ns | 3.86e3x | 8.65 MB | 8.5 |
| prism->point | irregular | 681,472 | 2.15 s | tensor-dictionary-4B | 0.113 ns | 357 ns | 3.17e3x | 19.2 MB | 8.8 |
| prism->point | irregular | 6,229,504 | 20.4 s | tensor-dictionary-4B | 0.426 ns | 354 ns | 830x | 176 MB | 9.3 |
| prism->prism | regular | 681,472 | 66.9 s | tensor-dictionary-2B | 0.0929 ns | 12.3 us | 1.33e5x | 953 kB | 8 |
| prism->prism | irregular | 681,472 | 64.5 s | tensor-dictionary-4B | 0.113 ns | 11.8 us | 1.05e5x | 11.1 MB | 8 |
| point->tetrahedron | regular | 677,376 | 1.46 s | tensor-dictionary-2B | 0.0946 ns | 254 ns | 2.68e3x | 1.64 MB | 8.5 |
| point->tetrahedron | regular | 6,196,736 | 13.5 s | tensor-dictionary-2B | 0.0817 ns | 252 ns | 3.08e3x | 14.9 MB | 8.6 |
| point->tetrahedron | irregular | 677,376 | 1.6 s | tensor-dictionary-4B | 0.119 ns | 257 ns | 2.16e3x | 19.2 MB | 9.2 |
| point->tetrahedron | irregular | 6,196,736 | 15.6 s | tensor-dictionary-4B | 0.44 ns | 255 ns | 579x | 176 MB | 9.9 |
| tetrahedron->point | regular | 681,472 | 1.46 s | tensor-dictionary-2B | 0.0948 ns | 254 ns | 2.68e3x | 1.64 MB | 8.4 |
| tetrahedron->point | regular | 6,229,504 | 13.4 s | tensor-dictionary-2B | 0.0821 ns | 252 ns | 3.07e3x | 14.9 MB | 8.6 |
| tetrahedron->point | irregular | 681,472 | 1.6 s | tensor-dictionary-4B | 0.101 ns | 258 ns | 2.56e3x | 19.3 MB | 9.1 |
| tetrahedron->point | irregular | 6,229,504 | 15.6 s | tensor-dictionary-4B | 0.43 ns | 256 ns | 596x | 176 MB | 9.8 |
| prism->tetrahedron | regular | 681,472 | 80.7 s | tensor-dictionary-2B | 0.0959 ns | 120 us (0.50) | 1.26e6x | 1.64 MB | 0.98 |
| prism->tetrahedron | irregular | 681,472 | 88.9 s | tensor-dictionary-4B | 0.109 ns | 133 us (0.45) | 1.21e6x | 19.3 MB | 0.98 |
| tetrahedron->prism | regular | 681,472 | 80.7 s | tensor-dictionary-2B | 0.0959 ns | 121 us (0.51) | 1.26e6x | 1.64 MB | 0.98 |
| tetrahedron->prism | irregular | 681,472 | 88.9 s | tensor-dictionary-4B | 0.103 ns | 133 us (0.45) | 1.28e6x | 19.3 MB | 0.98 |
| tetrahedron->tetrahedron | regular | 681,472 | 14.3 s | tensor-dictionary-2B | 0.0931 ns | 33.4 us | 3.59e5x | 952 kB | 0.63 |
| tetrahedron->tetrahedron | irregular | 681,472 | 18.3 s | tensor-dictionary-4B | 0.107 ns | 43.1 us | 4.05e5x | 11.1 MB | 0.62 |

FP64:

| Pair | layout | pairs | T_build | best stored | apply/pair | procedural/pair | ratio | stored bytes | K_break_even |
|---|---|---|---|---|---|---|---|---|---|
| point->point | regular | 677,376 | 82.1 ms | tensor-dictionary-1B | 0.124 ns | 1.12 ns | 9.06x | 956 kB | 1.2e+02 |
| point->point | irregular | 677,376 | 126 ms | tensor-dictionary-4B | 0.155 ns | 1.12 ns | 7.26x | 19.2 MB | 1.9e+02 |
| point->prism | regular | 677,376 | 1.75 s | tensor-dictionary-1B | 0.124 ns | 308 ns | 2.48e3x | 958 kB | 8.4 |
| point->prism | regular | 6,196,736 | 16.2 s | tensor-dictionary-1B | 0.111 ns | 307 ns | 2.78e3x | 8.66 MB | 8.5 |
| point->prism | irregular | 6,196,736 | 18.5 s | leaf-block | 0.852 ns | 354 ns | 416x | 301 MB | 8.4 |
| prism->point | regular | 681,472 | 1.75 s | tensor-dictionary-1B | 0.124 ns | 308 ns | 2.49e3x | 958 kB | 8.4 |
| prism->point | regular | 6,229,504 | 16.2 s | tensor-dictionary-1B | 0.109 ns | 307 ns | 2.81e3x | 8.66 MB | 8.5 |
| prism->point | irregular | 681,472 | 2.15 s | tensor-dictionary-4B | 0.198 ns | 357 ns | 1.80e3x | 35.5 MB | 8.8 |
| prism->point | irregular | 6,229,504 | 20.4 s | tensor-dictionary-4B | 0.851 ns | 354 ns | 416x | 325 MB | 9.3 |
| prism->prism | regular | 681,472 | 66.9 s | tensor-dictionary-2B | 0.125 ns | 12.3 us | 9.87e4x | 1.65 MB | 8 |
| prism->prism | irregular | 681,472 | 64.5 s | tensor-dictionary-4B | 0.195 ns | 11.8 us | 6.07e4x | 35.2 MB | 8 |
| point->tetrahedron | regular | 677,376 | 1.46 s | tensor-dictionary-2B | 0.125 ns | 254 ns | 2.02e3x | 1.65 MB | 8.5 |
| point->tetrahedron | regular | 6,196,736 | 13.5 s | tensor-dictionary-2B | 0.113 ns | 252 ns | 2.23e3x | 14.9 MB | 8.6 |
| point->tetrahedron | irregular | 677,376 | 1.6 s | tensor-dictionary-4B | 0.202 ns | 257 ns | 1.27e3x | 35.5 MB | 9.2 |
| point->tetrahedron | irregular | 6,196,736 | 13.5 s | leaf-block | 0.859 ns | 255 ns | 297x | 301 MB | 8.6 |
| tetrahedron->point | regular | 681,472 | 1.46 s | tensor-dictionary-2B | 0.127 ns | 254 ns | 2.00e3x | 1.65 MB | 8.4 |
| tetrahedron->point | regular | 6,229,504 | 13.4 s | tensor-dictionary-2B | 0.112 ns | 252 ns | 2.25e3x | 14.9 MB | 8.6 |
| tetrahedron->point | irregular | 681,472 | 1.6 s | tensor-dictionary-4B | 0.204 ns | 258 ns | 1.26e3x | 35.5 MB | 9.1 |
| tetrahedron->point | irregular | 6,229,504 | 15.6 s | tensor-dictionary-4B | 0.845 ns | 256 ns | 303x | 325 MB | 9.8 |
| prism->tetrahedron | regular | 681,472 | 80.7 s | tensor-dictionary-2B | 0.126 ns | 120 us (0.50) | 9.54e5x | 1.65 MB | 0.98 |
| prism->tetrahedron | irregular | 681,472 | 88.9 s | tensor-dictionary-4B | 0.196 ns | 133 us (0.45) | 6.78e5x | 35.5 MB | 0.98 |
| tetrahedron->prism | regular | 681,472 | 80.7 s | tensor-dictionary-2B | 0.127 ns | 121 us (0.51) | 9.52e5x | 1.65 MB | 0.98 |
| tetrahedron->prism | irregular | 681,472 | 88.9 s | tensor-dictionary-4B | 0.197 ns | 133 us (0.45) | 6.73e5x | 35.5 MB | 0.98 |
| tetrahedron->tetrahedron | regular | 681,472 | 14.3 s | tensor-dictionary-2B | 0.126 ns | 33.4 us | 2.66e5x | 1.64 MB | 0.63 |
| tetrahedron->tetrahedron | irregular | 681,472 | 18.3 s | tensor-dictionary-4B | 0.16 ns | 43.1 us | 2.70e5x | 19.3 MB | 0.62 |

Three things follow. The stored representations are at the memory roof and the
procedural ones are compute-bound, so the depth-4 lattice is the fair test:
with a 176 MB irregular operator, four times the LLC, the stored apply slows
from 0.11 to 0.43 ns per pair while the procedural cost does not move, and the
gap narrows only from about 3200x to about 840x. The regular lattice
compresses to one- or two-byte dictionary tokens (8.7-14.9 MB for 6.2M pairs)
and stays cache resident at 0.08 ns per pair. And the break-even is one to ten
complete field updates for every finite pair: a serial construction costs
about eight parallel procedural updates (the thread count), a parallel
construction costs about one, because construction *is* one pass of the same
arithmetic. Even a single-shot calculation that evaluates the field ten times
is already better off precomputing.

### Part D: CUDA finite P2P

Classification of the three distinct mathematical families, from the Part A
audit and the CPU costs above:

| Family | Device-callable with modest work? | What it would take |
|---|---|---|
| prism analytical point tensor (point <-> prism) | yes | the formulas are fixed-size arithmetic with `atan`/`log`/`hypot` and one `std::optional` branch; made precision-generic in `src/geometry/primitives/rectangular_prism_point_kernel.hpp` and instantiated on the device |
| prism pair (target-averaged F1/F2) | yes, mechanically | same shape, 64-corner sums; not ported because the CPU cost (95 us per pair) already rules it out by five orders of magnitude |
| tetrahedron point tensor | no | `std::sort` face frames, `std::domain_error` singularity detection |
| tetrahedron pair and mixed polyhedron | no | recursive `triangle_i3 -> i2s/i2t -> i1 -> i0` dimensional reduction, `std::variant` bodies, heap-backed prism surfaces, exceptions |

The prism family was ported, because it is exactly the "small shared-math
refactor" that makes a real measurement possible: the MagTense f/g/h corner
sums and F/F log ratios now live once in a precision-generic header that
production instantiates in `long double` (bit-identical to the previous
implementation, verified over 20k tensors covering the face, edge, vertex and
symmetry-plane branches) and that the benchmark instantiates in `double` and
`float`, on the host and inside a CUDA kernel
(`benchmarks/benchmark_operator_representation_cuda.cu`, one thread per
(target, source range) work item, no stored tensors).

The fastest stored packing of every case and every procedural row; the
complete 178-row set, including the slower stored packings, is in the
CSV.

| Pair | layout | pairs | prec | representation | kind | update/pair | kernel | device bytes | error |
|---|---|---|---|---|---|---|---|---|---|
| point->point | irregular | 677,376 | fp32 | leaf-block | precomputed | 0.0521 ns | 10 us | 16.7 MB | 1.8e-07 |
| point->point | irregular | 677,376 | fp64 | leaf-block | precomputed | 0.0858 ns | 24 us | 33 MB | 8.8e-16 |
| point->point | regular | 677,376 | fp32 | leaf-block | precomputed | 0.0522 ns | 10 us | 16.7 MB | 3.2e-07 |
| point->point | regular | 677,376 | fp64 | leaf-block | precomputed | 0.086 ns | 24 us | 33 MB | 1.1e-15 |
| point->prism | irregular | 677,376 | fp32 | leaf-block | precomputed | 0.0523 ns | 9 us | 16.7 MB | 2.3e-07 |
| point->prism | irregular | 677,376 | fp32 | procedural-cuda-fp64-math | procedural | 3.64 ns | 2.44 ms | 0 B | 2.0e-07 |
| point->prism | irregular | 677,376 | fp32 | procedural-cuda-fp32-math | procedural | 0.135 ns | 72 us | 0 B | 5.3e-06 |
| point->prism | irregular | 677,376 | fp64 | leaf-block | precomputed | 0.0845 ns | 26 us | 33 MB | 9.4e-16 |
| point->prism | irregular | 677,376 | fp64 | procedural-cuda | procedural | 3.64 ns | 2.44 ms | 0 B | 1.3e-14 |
| point->prism | irregular | 6,196,736 | fp32 | leaf-block | precomputed | 0.0292 ns | 98 us | 152 MB | 2.8e-07 |
| point->prism | irregular | 6,196,736 | fp32 | procedural-cuda-fp64-math | procedural | 3.57 ns | 22 ms | 0 B | 2.3e-07 |
| point->prism | irregular | 6,196,736 | fp32 | procedural-cuda-fp32-math | procedural | 0.103 ns | 557 us | 0 B | 9.7e-06 |
| point->prism | irregular | 6,196,736 | fp64 | leaf-block | precomputed | 0.0616 ns | 193 us | 302 MB | 1.2e-15 |
| point->prism | irregular | 6,196,736 | fp64 | procedural-cuda | procedural | 3.57 ns | 22 ms | 0 B | 2.0e-14 |
| point->prism | regular | 677,376 | fp32 | leaf-block | precomputed | 0.0528 ns | 10 us | 16.7 MB | 1.6e-07 |
| point->prism | regular | 677,376 | fp32 | procedural-cuda-fp64-math | procedural | 3.64 ns | 2.44 ms | 0 B | 2.3e-07 |
| point->prism | regular | 677,376 | fp32 | procedural-cuda-fp32-math | procedural | 0.137 ns | 72 us | 0 B | 2.2e-06 |
| point->prism | regular | 677,376 | fp64 | leaf-block | precomputed | 0.0859 ns | 24 us | 33 MB | 1.2e-15 |
| point->prism | regular | 677,376 | fp64 | procedural-cuda | procedural | 3.64 ns | 2.44 ms | 0 B | 3.7e-15 |
| point->prism | regular | 6,196,736 | fp32 | tensor-dictionary-1B-target-owned | precomputed | 0.0187 ns | 45 us | 8.82 MB | 7.9e-07 |
| point->prism | regular | 6,196,736 | fp32 | procedural-cuda-fp64-math | procedural | 3.57 ns | 22 ms | 0 B | 2.8e-07 |
| point->prism | regular | 6,196,736 | fp32 | procedural-cuda-fp32-math | procedural | 0.105 ns | 555 us | 0 B | 2.6e-06 |
| point->prism | regular | 6,196,736 | fp64 | tensor-dictionary-1B-target-owned | precomputed | 0.0515 ns | 134 us | 8.82 MB | 0.0e+00 |
| point->prism | regular | 6,196,736 | fp64 | procedural-cuda | procedural | 3.57 ns | 22 ms | 0 B | 4.0e-15 |
| point->tetrahedron | irregular | 677,376 | fp32 | leaf-block | precomputed | 0.0502 ns | 9 us | 16.7 MB | 1.7e-07 |
| point->tetrahedron | irregular | 677,376 | fp64 | leaf-block | precomputed | 0.0859 ns | 25 us | 33 MB | 9.2e-16 |
| point->tetrahedron | regular | 677,376 | fp32 | leaf-block | precomputed | 0.0542 ns | 9 us | 16.7 MB | 2.6e-07 |
| point->tetrahedron | regular | 677,376 | fp64 | leaf-block | precomputed | 0.0842 ns | 24 us | 33 MB | 1.4e-15 |
| prism->point | irregular | 681,472 | fp32 | leaf-block | precomputed | 0.0527 ns | 9 us | 16.7 MB | 2.8e-07 |
| prism->point | irregular | 681,472 | fp32 | procedural-cuda-fp64-math | procedural | 3.62 ns | 2.45 ms | 0 B | 2.5e-07 |
| prism->point | irregular | 681,472 | fp32 | procedural-cuda-fp32-math | procedural | 0.138 ns | 74 us | 0 B | 1.9e-06 |
| prism->point | irregular | 681,472 | fp64 | leaf-block | precomputed | 0.0851 ns | 24 us | 33 MB | 1.1e-15 |
| prism->point | irregular | 681,472 | fp64 | procedural-cuda | procedural | 3.62 ns | 2.44 ms | 0 B | 3.7e-15 |
| prism->point | irregular | 6,229,504 | fp32 | leaf-block | precomputed | 0.0288 ns | 100 us | 152 MB | 3.0e-07 |
| prism->point | irregular | 6,229,504 | fp32 | procedural-cuda-fp64-math | procedural | 3.55 ns | 22 ms | 0 B | 2.7e-07 |
| prism->point | irregular | 6,229,504 | fp32 | procedural-cuda-fp32-math | procedural | 0.105 ns | 565 us | 0 B | 2.8e-06 |
| prism->point | irregular | 6,229,504 | fp64 | leaf-block | precomputed | 0.0605 ns | 192 us | 302 MB | 1.4e-15 |
| prism->point | irregular | 6,229,504 | fp64 | procedural-cuda | procedural | 3.55 ns | 22 ms | 0 B | 4.9e-15 |
| prism->point | regular | 681,472 | fp32 | leaf-block | precomputed | 0.0528 ns | 9 us | 16.7 MB | 2.5e-07 |
| prism->point | regular | 681,472 | fp32 | procedural-cuda-fp64-math | procedural | 3.61 ns | 2.44 ms | 0 B | 2.5e-07 |
| prism->point | regular | 681,472 | fp32 | procedural-cuda-fp32-math | procedural | 0.134 ns | 72 us | 0 B | 1.8e-06 |
| prism->point | regular | 681,472 | fp64 | leaf-block | precomputed | 0.0854 ns | 24 us | 33 MB | 1.2e-15 |
| prism->point | regular | 681,472 | fp64 | procedural-cuda | procedural | 3.62 ns | 2.44 ms | 0 B | 3.0e-15 |
| prism->point | regular | 6,229,504 | fp32 | tensor-dictionary-1B-target-owned | precomputed | 0.0184 ns | 48 us | 8.82 MB | 8.2e-07 |
| prism->point | regular | 6,229,504 | fp32 | procedural-cuda-fp64-math | procedural | 3.53 ns | 21.9 ms | 0 B | 2.6e-07 |
| prism->point | regular | 6,229,504 | fp32 | procedural-cuda-fp32-math | procedural | 0.104 ns | 559 us | 0 B | 1.8e-06 |
| prism->point | regular | 6,229,504 | fp64 | tensor-dictionary-1B-target-owned | precomputed | 0.0508 ns | 141 us | 8.82 MB | 0.0e+00 |
| prism->point | regular | 6,229,504 | fp64 | procedural-cuda | procedural | 3.56 ns | 22 ms | 0 B | 2.7e-15 |
| prism->prism | irregular | 681,472 | fp32 | leaf-block | precomputed | 0.0513 ns | 9 us | 16.7 MB | 2.8e-07 |
| prism->prism | irregular | 681,472 | fp64 | leaf-block | precomputed | 0.0851 ns | 24 us | 33 MB | 1.2e-15 |
| prism->prism | regular | 681,472 | fp32 | leaf-block | precomputed | 0.0519 ns | 9 us | 16.7 MB | 2.0e-07 |
| prism->prism | regular | 681,472 | fp64 | leaf-block | precomputed | 0.0853 ns | 24 us | 33 MB | 1.3e-15 |
| prism->tetrahedron | irregular | 681,472 | fp32 | leaf-block | precomputed | 0.0537 ns | 17 us | 16.7 MB | 3.8e-07 |
| prism->tetrahedron | irregular | 681,472 | fp64 | leaf-block | precomputed | 0.0855 ns | 24 us | 33 MB | 1.5e-15 |
| prism->tetrahedron | regular | 681,472 | fp32 | leaf-block | precomputed | 0.0549 ns | 17 us | 16.7 MB | 2.1e-07 |
| prism->tetrahedron | regular | 681,472 | fp64 | leaf-block | precomputed | 0.0852 ns | 24 us | 33 MB | 1.1e-15 |
| tetrahedron->point | irregular | 681,472 | fp32 | leaf-block | precomputed | 0.0542 ns | 9 us | 16.7 MB | 2.3e-07 |
| tetrahedron->point | irregular | 681,472 | fp64 | leaf-block | precomputed | 0.0845 ns | 24 us | 33 MB | 1.8e-15 |
| tetrahedron->point | regular | 681,472 | fp32 | leaf-block | precomputed | 0.0516 ns | 9 us | 16.7 MB | 4.5e-07 |
| tetrahedron->point | regular | 681,472 | fp64 | leaf-block | precomputed | 0.0844 ns | 24 us | 33 MB | 1.7e-15 |
| tetrahedron->prism | irregular | 681,472 | fp32 | leaf-block | precomputed | 0.0522 ns | 9 us | 16.7 MB | 2.3e-07 |
| tetrahedron->prism | irregular | 681,472 | fp64 | leaf-block | precomputed | 0.0847 ns | 24 us | 33 MB | 1.2e-15 |
| tetrahedron->prism | regular | 681,472 | fp32 | leaf-block | precomputed | 0.0543 ns | 17 us | 16.7 MB | 2.2e-07 |
| tetrahedron->prism | regular | 681,472 | fp64 | leaf-block | precomputed | 0.0854 ns | 24 us | 33 MB | 1.8e-15 |
| tetrahedron->tetrahedron | irregular | 681,472 | fp32 | leaf-block | precomputed | 0.0506 ns | 9 us | 16.7 MB | 2.9e-07 |
| tetrahedron->tetrahedron | irregular | 681,472 | fp64 | leaf-block | precomputed | 0.0849 ns | 24 us | 33 MB | 1.3e-15 |
| tetrahedron->tetrahedron | regular | 681,472 | fp32 | leaf-block | precomputed | 0.0516 ns | 9 us | 16.7 MB | 3.7e-07 |
| tetrahedron->tetrahedron | regular | 681,472 | fp64 | leaf-block | precomputed | 0.0848 ns | 24 us | 33 MB | 1.4e-15 |

The device narrows the gap by two and a half orders of magnitude and still
loses. Reconstructing the prism point tensor in FP32 arithmetic costs 0.10 to
0.14 ns per pair against 0.018 to 0.053 ns for the best stored packing, a
factor of 2.5 to 5.7, and it retains no operator at all against 8.8 MB (a
one-byte dictionary on the 32768-body lattice) to 152 MB (leaf blocks on the
irregular one). Reconstructing in FP64 arithmetic, which is what an FP64 plan
would need, costs 3.5 ns per pair: 68x to 196x the stored apply, because the
consumer GPU's FP64 transcendental rate is a small fraction of its FP32 one.
Against the same reconstruction on eight pinned CPU cores the device is 376x
to 559x faster in float arithmetic, 31x to 41x faster in double, and 2300x to
3400x faster than the production `long double` host path, which is the
measurement the feasibility question needed.

Accuracy is not equivalent either. The FP32-math reconstruction sits at
1.8e-6 to 2.8e-6 of the field scale against 2.5e-7 to 8.2e-7 for the stored
FP32 tensors, and the same FP32 reconstruction degrades to 4.5e-4 in the
far-separation hot class, where the corner sums of the analytical formula
cancel. The FP64-math reconstruction matches the stored FP32 tensors (2.5e-7)
and the FP64 ones to 3.0e-15.

The amortisation is the one place where this family is genuinely interesting,
and it is worth stating precisely because it is the closest any finite
operator came. Because the stored apply is so fast on the device and the
construction is a serial host loop, the break-even is not one to ten updates
as on the CPU but 30,000 to 38,000:

| Case | T_build | stored apply | procedural | ratio | bytes saved | K_break_even |
|---|---|---|---|---|---|---|
| 681k pairs, regular | 1.7 s | 0.036 ms | 0.091 ms | 2.5x | 16.7 MB | 31,200 |
| 681k pairs, irregular | 2.0 s | 0.036 ms | 0.094 ms | 2.6x | 16.7 MB | 34,300 |
| 6.2M pairs, regular | 15.9 s | 0.115 ms | 0.649 ms | 5.7x | 8.8 MB | 29,700 |
| 6.2M pairs, irregular | 18.2 s | 0.180 ms | 0.657 ms | 3.7x | 152 MB | 38,200 |

A single run shorter than about thirty thousand field updates would therefore
finish sooner with a procedural prism near field on the GPU. That is not a
reason to change production, for three reasons that the table does not show.
The persistent geometry cache already pays the construction once per geometry,
so from the second run onwards the stored path wins from the first update.
The break-even compares against a construction that is serial host code, which
Phase 3C may change, and every second removed there moves the break-even
further out of reach. And the procedural path exists for one of the nine
geometry pairs, in FP32 arithmetic only, at nine times the error of the stored
FP32 tensors. No procedural finite path was therefore integrated into
`UniformFmm`, and no complete FMM measurement was run; the numbers above are
recorded so the threshold can be re-examined if device memory rather than time
ever becomes the binding constraint.

The three families that were not ported are recorded instead, as the brief
allows. Their procedural cost on eight CPU cores is 12.3 us (prism pair),
33.4 us (tetrahedron pair) and 120 us (mixed pair) per pair, against 0.05 to
0.09 ns for a stored apply on the device, so a device implementation would
have to be between 2x10^5 and 2x10^6 times faster than the eight-thread CPU to
reach parity. The measured prism port achieves 376x to 559x in float
arithmetic and 31x to 41x in double. The parallelism is ample (one pair per
thread) but
the arithmetic is not: the tetrahedron pair integral is recursive with
data-dependent branches per face pair, which is why no such subsystem was
built for this experiment.

### Parts E and F: exact finite P2M and L2P

P2M (streaming, eight threads):

| geometry | layout | p | prec | bodies | T_build/source | row bytes | stored apply | procedural | ratio | K_break_even |
|---|---|---|---|---|---|---|---|---|---|---|
| prism | irregular | 4 | fp32 | 4,096 | 5.7 us | 300 B | 3.23 ns | 686 ns | 213x | 8.3 |
| prism | irregular | 4 | fp64 | 4,096 | 5.65 us | 600 B | 4.46 ns | 685 ns | 153x | 8.3 |
| prism | irregular | 6 | fp32 | 4,096 | 28.3 us | 588 B | 3.99 ns | 3.6 us | 902x | 7.9 |
| prism | irregular | 6 | fp64 | 4,096 | 28.1 us | 1176 B | 5.87 ns | 3.57 us | 608x | 7.9 |
| prism | irregular | 8 | fp32 | 4,096 | 89.6 us | 972 B | 5.19 ns | 11.4 us | 2.20e3x | 7.8 |
| prism | irregular | 8 | fp64 | 4,096 | 89.2 us | 1944 B | 8.01 ns | 11.4 us | 1.42e3x | 7.8 |
| prism | irregular | 10 | fp32 | 4,096 | 224 us | 1452 B | 6.34 ns | 31.4 us | 4.96e3x | 7.1 |
| prism | irregular | 10 | fp64 | 4,096 | 224 us | 2904 B | 9.37 ns | 28.6 us | 3.06e3x | 7.8 |
| prism | regular | 4 | fp32 | 4,096 | 5.8 us | 300 B | 2.57 ns | 688 ns | 268x | 8.5 |
| prism | regular | 4 | fp64 | 4,096 | 5.71 us | 600 B | 4.52 ns | 680 ns | 151x | 8.4 |
| prism | regular | 6 | fp32 | 4,096 | 28.4 us | 588 B | 3.84 ns | 3.56 us | 926x | 8 |
| prism | regular | 6 | fp64 | 4,096 | 28.2 us | 1176 B | 6.13 ns | 3.55 us | 578x | 8 |
| prism | regular | 8 | fp32 | 4,096 | 89.6 us | 972 B | 5.04 ns | 11.4 us | 2.25e3x | 7.9 |
| prism | regular | 8 | fp64 | 4,096 | 89.1 us | 1944 B | 8.31 ns | 11.3 us | 1.36e3x | 7.9 |
| prism | regular | 10 | fp32 | 4,096 | 224 us | 1452 B | 6.33 ns | 28.8 us | 4.54e3x | 7.8 |
| prism | regular | 10 | fp64 | 4,096 | 223 us | 2904 B | 10.1 ns | 31.1 us | 3.08e3x | 7.2 |
| tetrahedron | irregular | 4 | fp32 | 4,096 | 74.6 us | 300 B | 3.21 ns | 9.67 us | 3.01e3x | 7.7 |
| tetrahedron | irregular | 4 | fp64 | 4,096 | 74.5 us | 600 B | 4.49 ns | 9.61 us | 2.14e3x | 7.8 |
| tetrahedron | irregular | 6 | fp32 | 4,096 | 1.21 ms | 588 B | 4.03 ns | 165 us | 4.08e4x | 7.4 |
| tetrahedron | irregular | 6 | fp64 | 4,096 | 1.21 ms | 1176 B | 5.9 ns | 163 us | 2.76e4x | 7.5 |
| tetrahedron | irregular | 8 | fp32 | 4,096 | 16.8 ms | 972 B | 5.14 ns | 2.26 ms (1.00) | 4.40e5x | 7.4 |
| tetrahedron | irregular | 8 | fp64 | 4,096 | 16.8 ms | 1944 B | 7.61 ns | 2.27 ms (1.00) | 2.98e5x | 7.4 |
| tetrahedron | regular | 4 | fp32 | 4,096 | 73.1 us | 300 B | 3.13 ns | 9.47 us | 3.03e3x | 7.7 |
| tetrahedron | regular | 4 | fp64 | 4,096 | 73 us | 600 B | 4.48 ns | 9.45 us | 2.11e3x | 7.7 |
| tetrahedron | regular | 6 | fp32 | 4,096 | 1.23 ms | 588 B | 4.01 ns | 166 us | 4.14e4x | 7.4 |
| tetrahedron | regular | 6 | fp64 | 4,096 | 1.23 ms | 1176 B | 6.07 ns | 165 us | 2.72e4x | 7.4 |
| tetrahedron | regular | 8 | fp32 | 4,096 | 16.9 ms | 972 B | 4.98 ns | 2.28 ms (0.99) | 4.58e5x | 7.4 |
| tetrahedron | regular | 8 | fp64 | 4,096 | 16.9 ms | 1944 B | 8.35 ns | 2.27 ms (0.99) | 2.72e5x | 7.4 |

L2P (streaming, eight threads):

| geometry | layout | p | prec | bodies | T_build/target | row bytes | stored apply | procedural | ratio | K_break_even |
|---|---|---|---|---|---|---|---|---|---|---|
| prism | irregular | 4 | fp32 | 4,096 | 9.53 us | 400 B | 2.7 ns | 1.17 us | 432x | 8.2 |
| prism | irregular | 4 | fp64 | 4,096 | 9.47 us | 800 B | 3.26 ns | 1.16 us | 356x | 8.2 |
| prism | irregular | 6 | fp32 | 4,096 | 42.5 us | 784 B | 3.23 ns | 5.53 us | 1.71e3x | 7.7 |
| prism | irregular | 6 | fp64 | 4,096 | 42.4 us | 1568 B | 4.44 ns | 5.45 us | 1.23e3x | 7.8 |
| prism | irregular | 8 | fp32 | 4,096 | 130 us | 1296 B | 3.89 ns | 17.4 us | 4.48e3x | 7.5 |
| prism | irregular | 8 | fp64 | 4,096 | 130 us | 2592 B | 5.93 ns | 16.8 us | 2.83e3x | 7.7 |
| prism | irregular | 10 | fp32 | 4,096 | 322 us | 1936 B | 5 ns | 43.1 us | 8.62e3x | 7.5 |
| prism | irregular | 10 | fp64 | 4,096 | 322 us | 3872 B | 7 ns | 44.1 us | 6.31e3x | 7.3 |
| prism | regular | 4 | fp32 | 4,096 | 9.55 us | 400 B | 2.73 ns | 1.16 us | 424x | 8.3 |
| prism | regular | 4 | fp64 | 4,096 | 9.46 us | 800 B | 3.24 ns | 1.16 us | 358x | 8.2 |
| prism | regular | 6 | fp32 | 4,096 | 42.5 us | 784 B | 3.23 ns | 5.45 us | 1.69e3x | 7.8 |
| prism | regular | 6 | fp64 | 4,096 | 42.4 us | 1568 B | 4.48 ns | 5.38 us | 1.20e3x | 7.9 |
| prism | regular | 8 | fp32 | 4,096 | 130 us | 1296 B | 3.88 ns | 16.7 us | 4.31e3x | 7.8 |
| prism | regular | 8 | fp64 | 4,096 | 130 us | 2592 B | 6.01 ns | 17.3 us | 2.89e3x | 7.5 |
| prism | regular | 10 | fp32 | 4,096 | 322 us | 1936 B | 5.05 ns | 44.2 us | 8.74e3x | 7.3 |
| prism | regular | 10 | fp64 | 4,096 | 322 us | 3872 B | 6.85 ns | 43.4 us | 6.34e3x | 7.4 |
| tetrahedron | irregular | 4 | fp32 | 4,096 | 171 us | 400 B | 2.66 ns | 22.3 us | 8.37e3x | 7.7 |
| tetrahedron | irregular | 4 | fp64 | 4,096 | 171 us | 800 B | 3.34 ns | 21.7 us | 6.51e3x | 7.9 |
| tetrahedron | irregular | 6 | fp32 | 4,096 | 2.72 ms | 784 B | 3.23 ns | 367 us | 1.14e5x | 7.4 |
| tetrahedron | irregular | 6 | fp64 | 4,096 | 2.72 ms | 1568 B | 4.42 ns | 365 us | 8.26e4x | 7.5 |
| tetrahedron | irregular | 8 | fp32 | 4,096 | 35.6 ms | 1296 B | 3.89 ns | 4.87 ms (0.47) | 1.25e6x | 7.3 |
| tetrahedron | irregular | 8 | fp64 | 4,096 | 35.6 ms | 2592 B | 5.97 ns | 4.86 ms (0.47) | 8.14e5x | 7.3 |
| tetrahedron | regular | 4 | fp32 | 4,096 | 168 us | 400 B | 2.71 ns | 22 us | 8.12e3x | 7.7 |
| tetrahedron | regular | 4 | fp64 | 4,096 | 168 us | 800 B | 3.28 ns | 22.5 us | 6.86e3x | 7.5 |
| tetrahedron | regular | 6 | fp32 | 4,096 | 2.74 ms | 784 B | 3.25 ns | 369 us | 1.14e5x | 7.4 |
| tetrahedron | regular | 6 | fp64 | 4,096 | 2.74 ms | 1568 B | 4.39 ns | 368 us | 8.38e4x | 7.4 |
| tetrahedron | regular | 8 | fp32 | 4,096 | 35.5 ms | 1296 B | 3.89 ns | 4.86 ms (0.47) | 1.25e6x | 7.3 |
| tetrahedron | regular | 8 | fp64 | 4,096 | 35.5 ms | 2592 B | 5.96 ns | 4.84 ms (0.47) | 8.12e5x | 7.3 |

P2M (hot, one thread, 64 bodies):

| geometry | layout | p | prec | bodies | T_build/source | row bytes | stored apply | procedural | ratio | K_break_even |
|---|---|---|---|---|---|---|---|---|---|---|
| prism | irregular | 4 | fp32 | 64 | 9.9 us | 300 B | 6.59 ns | 5.37 us | 814x | 1.8 |
| prism | irregular | 4 | fp64 | 64 | 14.7 us | 600 B | 8.81 ns | 5.37 us | 610x | 2.7 |
| prism | irregular | 6 | fp32 | 64 | 33.4 us | 588 B | 9.11 ns | 27.6 us | 3.04e3x | 1.2 |
| prism | irregular | 6 | fp64 | 64 | 44 us | 1176 B | 14.9 ns | 27.7 us | 1.85e3x | 1.6 |
| prism | irregular | 8 | fp32 | 64 | 96.1 us | 972 B | 13.6 ns | 88.2 us | 6.48e3x | 1.1 |
| prism | irregular | 8 | fp64 | 64 | 117 us | 1944 B | 24.9 ns | 88.1 us | 3.54e3x | 1.3 |
| prism | irregular | 10 | fp32 | 64 | 235 us | 1452 B | 22.6 ns | 223 us | 9.86e3x | 1.1 |
| prism | irregular | 10 | fp64 | 64 | 265 us | 2904 B | 35.3 ns | 223 us | 6.30e3x | 1.2 |
| prism | regular | 4 | fp32 | 64 | 9.59 us | 300 B | 6.63 ns | 5.35 us | 807x | 1.8 |
| prism | regular | 4 | fp64 | 64 | 14.4 us | 600 B | 8.75 ns | 5.35 us | 612x | 2.7 |
| prism | regular | 6 | fp32 | 64 | 32.1 us | 588 B | 9.92 ns | 27.5 us | 2.78e3x | 1.2 |
| prism | regular | 6 | fp64 | 64 | 44.3 us | 1176 B | 14.8 ns | 27.6 us | 1.86e3x | 1.6 |
| prism | regular | 8 | fp32 | 64 | 96.2 us | 972 B | 15.7 ns | 88.1 us | 5.60e3x | 1.1 |
| prism | regular | 8 | fp64 | 64 | 117 us | 1944 B | 22.9 ns | 88.1 us | 3.85e3x | 1.3 |
| prism | regular | 10 | fp32 | 64 | 234 us | 1452 B | 19.4 ns | 222 us | 1.14e4x | 1.1 |
| prism | regular | 10 | fp64 | 64 | 266 us | 2904 B | 35.5 ns | 222 us | 6.26e3x | 1.2 |
| tetrahedron | irregular | 4 | fp32 | 64 | 80.7 us | 300 B | 6.58 ns | 74.4 us | 1.13e4x | 1.1 |
| tetrahedron | irregular | 4 | fp64 | 64 | 85.5 us | 600 B | 8.7 ns | 74.5 us | 8.56e3x | 1.1 |
| tetrahedron | irregular | 6 | fp32 | 64 | 1.23 ms | 588 B | 9.3 ns | 1.22 ms | 1.31e5x | 1 |
| tetrahedron | irregular | 6 | fp64 | 64 | 1.24 ms | 1176 B | 14.8 ns | 1.22 ms | 8.22e4x | 1 |
| tetrahedron | irregular | 8 | fp32 | 64 | 16.9 ms | 972 B | 13.4 ns | 16.8 ms | 1.25e6x | 1 |
| tetrahedron | irregular | 8 | fp64 | 64 | 16.9 ms | 1944 B | 23 ns | 16.9 ms | 7.34e5x | 1 |
| tetrahedron | irregular | 10 | fp32 | 64 | 185 ms | 1452 B | 19.3 ns | 185 ms | 9.62e6x | 1 |
| tetrahedron | irregular | 10 | fp64 | 64 | 185 ms | 2904 B | 35 ns | 185 ms | 5.28e6x | 1 |
| tetrahedron | regular | 4 | fp32 | 64 | 78.4 us | 300 B | 6.51 ns | 72.9 us | 1.12e4x | 1.1 |
| tetrahedron | regular | 4 | fp64 | 64 | 82.5 us | 600 B | 8.76 ns | 73.1 us | 8.35e3x | 1.1 |
| tetrahedron | regular | 6 | fp32 | 64 | 1.23 ms | 588 B | 9.48 ns | 1.22 ms | 1.29e5x | 1 |
| tetrahedron | regular | 6 | fp64 | 64 | 1.24 ms | 1176 B | 14.8 ns | 1.22 ms | 8.24e4x | 1 |
| tetrahedron | regular | 8 | fp32 | 64 | 16.9 ms | 972 B | 13.6 ns | 16.9 ms | 1.24e6x | 1 |
| tetrahedron | regular | 8 | fp64 | 64 | 16.9 ms | 1944 B | 23 ns | 16.9 ms | 7.33e5x | 1 |
| tetrahedron | regular | 10 | fp32 | 64 | 185 ms | 1452 B | 19.3 ns | 185 ms | 9.61e6x | 1 |
| tetrahedron | regular | 10 | fp64 | 64 | 185 ms | 2904 B | 35 ns | 185 ms | 5.29e6x | 1 |

L2P (hot, one thread, 64 bodies):

| geometry | layout | p | prec | bodies | T_build/target | row bytes | stored apply | procedural | ratio | K_break_even |
|---|---|---|---|---|---|---|---|---|---|---|
| prism | irregular | 4 | fp32 | 64 | 39.7 us | 400 B | 5.87 ns | 9.02 us | 1.54e3x | 4.4 |
| prism | irregular | 4 | fp64 | 64 | 36 us | 800 B | 11 ns | 9.03 us | 819x | 4 |
| prism | irregular | 6 | fp32 | 64 | 71.5 us | 784 B | 8.82 ns | 42 us | 4.76e3x | 1.7 |
| prism | irregular | 6 | fp64 | 64 | 70.3 us | 1568 B | 18.7 ns | 42 us | 2.25e3x | 1.7 |
| prism | irregular | 8 | fp32 | 64 | 179 us | 1296 B | 13.2 ns | 129 us | 9.78e3x | 1.4 |
| prism | irregular | 8 | fp64 | 64 | 174 us | 2592 B | 24 ns | 129 us | 5.37e3x | 1.3 |
| prism | irregular | 10 | fp32 | 64 | 395 us | 1936 B | 18.6 ns | 321 us | 1.73e4x | 1.2 |
| prism | irregular | 10 | fp64 | 64 | 391 us | 3872 B | 36.9 ns | 322 us | 8.72e3x | 1.2 |
| prism | regular | 4 | fp32 | 64 | 38 us | 400 B | 5.71 ns | 8.99 us | 1.57e3x | 4.2 |
| prism | regular | 4 | fp64 | 64 | 35.5 us | 800 B | 9.53 ns | 8.99 us | 943x | 3.9 |
| prism | regular | 6 | fp32 | 64 | 70.6 us | 784 B | 8.86 ns | 41.9 us | 4.73e3x | 1.7 |
| prism | regular | 6 | fp64 | 64 | 70.5 us | 1568 B | 18.7 ns | 41.9 us | 2.24e3x | 1.7 |
| prism | regular | 8 | fp32 | 64 | 180 us | 1296 B | 13.2 ns | 129 us | 9.81e3x | 1.4 |
| prism | regular | 8 | fp64 | 64 | 172 us | 2592 B | 24 ns | 129 us | 5.38e3x | 1.3 |
| prism | regular | 10 | fp32 | 64 | 400 us | 1936 B | 18.6 ns | 322 us | 1.73e4x | 1.2 |
| prism | regular | 10 | fp64 | 64 | 398 us | 3872 B | 35.4 ns | 322 us | 9.09e3x | 1.2 |
| tetrahedron | irregular | 4 | fp32 | 64 | 204 us | 400 B | 5.77 ns | 170 us | 2.94e4x | 1.2 |
| tetrahedron | irregular | 4 | fp64 | 64 | 199 us | 800 B | 9.57 ns | 167 us | 1.75e4x | 1.2 |
| tetrahedron | irregular | 6 | fp32 | 64 | 2.77 ms | 784 B | 8.93 ns | 2.72 ms | 3.05e5x | 1 |
| tetrahedron | irregular | 6 | fp64 | 64 | 2.77 ms | 1568 B | 16.1 ns | 2.7 ms | 1.68e5x | 1 |
| tetrahedron | irregular | 8 | fp32 | 64 | 35.6 ms | 1296 B | 13.3 ns | 35.5 ms | 2.68e6x | 1 |
| tetrahedron | irregular | 8 | fp64 | 64 | 35.6 ms | 2592 B | 24 ns | 35.4 ms | 1.47e6x | 1 |
| tetrahedron | irregular | 10 | fp32 | 64 | 358 ms | 1936 B | 18.7 ns | 358 ms | 1.92e7x | 1 |
| tetrahedron | irregular | 10 | fp64 | 64 | 358 ms | 3872 B | 36.8 ns | 358 ms | 9.71e6x | 1 |
| tetrahedron | regular | 4 | fp32 | 64 | 202 us | 400 B | 5.71 ns | 169 us | 2.97e4x | 1.2 |
| tetrahedron | regular | 4 | fp64 | 64 | 198 us | 800 B | 11.1 ns | 167 us | 1.50e4x | 1.2 |
| tetrahedron | regular | 6 | fp32 | 64 | 2.78 ms | 784 B | 8.86 ns | 2.74 ms | 3.10e5x | 1 |
| tetrahedron | regular | 6 | fp64 | 64 | 2.77 ms | 1568 B | 18.6 ns | 2.72 ms | 1.46e5x | 1 |
| tetrahedron | regular | 8 | fp32 | 64 | 35.6 ms | 1296 B | 13.2 ns | 35.6 ms | 2.70e6x | 1 |
| tetrahedron | regular | 8 | fp64 | 64 | 35.6 ms | 2592 B | 24.6 ns | 35.5 ms | 1.44e6x | 1 |
| tetrahedron | regular | 10 | fp32 | 64 | 358 ms | 1936 B | 18.7 ns | 358 ms | 1.91e7x | 1 |
| tetrahedron | regular | 10 | fp64 | 64 | 358 ms | 3872 B | 35.2 ns | 358 ms | 1.02e7x | 1 |

The prism expansions are the cheapest finite operators in the study and still
lose by two to three orders of magnitude, because a stored row is a dense
`3 C` or `4 C` dot product at the memory roof while the builder walks the
Cartesian polynomial table of every mode and evaluates one exact
volume-averaged monomial per term and axis (172 terms at p 6, 470 at p 10).
The tetrahedron expansions are far worse: `tetrahedron_averaged_monomial`
expands each monomial in barycentric coordinates with a heap-allocating
polynomial product per (mode, term, axis), so one source costs 9.5 us at p 4,
166 us at p 6 and 2.3 ms at p 8, and one target 0.17 ms at p 4 rising to
358 ms at p 10.

### Hybrid representations

Not pursued, and the profile says why. The brief allowed a hybrid of small
precomputed shape invariants plus a procedural position-dependent recurrence
only if the separation is mathematically clean and the geometry-only part
dominates. It is clean: a prism's per-axis even moments
`h^gamma / (2^gamma (gamma+1)!)` and a tetrahedron's barycentric simplex
moments `6 prod(n_i!) / (3 + |n|)!` depend only on the body, and the
displacement enters through a binomial shift. But the measured gap is 150x to
1,250,000x per update, while the hoistable part is a handful of scalars per
monomial: hoisting it cannot close three to five orders of magnitude, because
what remains is the `O(C x terms)` walk itself. A hybrid would also have to
retain a per-body invariant table of `(p+1)(p+2)(p+3)/6` monomials (286 at
p 10), which is the same order as the `4 C` row it would replace (484 at
p 10). No hybrid was implemented.

### Parts H and I: the full-FMM gate

Every finite operator family fails the gate by three to six orders of
magnitude in the realistic streaming microbenchmark, so no procedural finite
path was integrated into `UniformFmm` and no complete FMM measurement was run;
per the brief, the amortisation is recorded instead and the family stops
there. No forced-procedural FMM path, no cache change: the persistent
geometry-cache format and keys are untouched, and the question of whether a
production procedural path would need a cache-key change never arose.

### Part K: memory

Persistent bytes per representation, from the `*_bytes` columns of the CSV
(depth-3 lattice, 681k pairs, 4096 bodies):

| Stage | representation | geometry | topology | operator | index/metadata | invariants | total persistent |
|---|---|---|---|---|---|---|---|
| P2P prism->prism regular FP32 | tensor-dictionary-2B | 98.3 kB | 89.3 kB | 5.95 kB | 947 kB | 0 B | 1.14 MB |
| P2P prism->prism regular FP32 | procedural | 98.3 kB | 89.3 kB | 0 B | 0 B | 0 B | 188 kB |
| P2P prism->prism irregular FP32 | tensor-dictionary-4B | 197 kB | 89.3 kB | 8.13 MB | 2.99 MB | 0 B | 11.4 MB |
| P2P prism->prism irregular FP32 | procedural | 197 kB | 89.3 kB | 0 B | 0 B | 0 B | 286 kB |
| P2P tetra->tetra regular FP32 | tensor-dictionary-2B | 98.4 kB | 89.3 kB | 4.34 kB | 947 kB | 0 B | 1.14 MB |
| P2P tetra->tetra regular FP32 | procedural | 98.4 kB | 89.3 kB | 0 B | 0 B | 496 B | 188 kB |
| P2M prism p 6 FP32 | packed-rows | 98.3 kB | 20.5 kB | 2.41 MB | 0 B | 0 B | 2.53 MB |
| P2M prism p 6 FP32 | procedural-builder | 98.3 kB | 20.5 kB | 0 B | 0 B | 0 B | 119 kB |
| L2P prism p 6 FP32 | packed-rows | 98.3 kB | 20.5 kB | 3.21 MB | 0 B | 0 B | 3.33 MB |
| L2P prism p 6 FP32 | procedural-builder | 98.3 kB | 20.5 kB | 0 B | 0 B | 0 B | 119 kB |
| P2M tetra p 6 FP32 | packed-rows | 98.4 kB | 20.5 kB | 2.41 MB | 0 B | 0 B | 2.53 MB |
| P2M tetra p 6 FP32 | procedural-builder | 98.4 kB | 20.5 kB | 0 B | 0 B | 0 B | 119 kB |
| L2P tetra p 6 FP32 | packed-rows | 98.4 kB | 20.5 kB | 3.21 MB | 0 B | 0 B | 3.33 MB |
| L2P tetra p 6 FP32 | procedural-builder | 98.4 kB | 20.5 kB | 0 B | 0 B | 0 B | 119 kB |

The procedural representations retain the geometry and topology they would
need anyway plus the hoisted per-body invariants, and no operator: 0 bytes of
tensors against 0.95 MB (regular, one-byte dictionary tokens) to 19.3 MB
(irregular, 98% unique tensors) for P2P, and 0 bytes of rows against 1.2 MB
(P2M) and 1.6 MB (L2P) at p 4 FP32 rising to 5.9 MB and 7.9 MB at p 10. On a
regular lattice the dictionary already compresses the near field by 12x to
36x, which is the representation procedural execution would have to beat.

### Correctness, sanitizer and validation

Every one of the 1003 CPU and 178 CUDA measurement rows carries the maximum
relative error of its own result against the FP64 canonical operator on the
final moments of that run. All procedural FP64 reconstructions agree with the
canonical operator to 5e-15 or better and most to 1e-16; the FP32 rows sit at
2e-7 to 3e-6 beside 2e-7 to 1e-6 for the quantised stored packings. Four rows
of the whole set exceed 1e-4, all of them the deliberate float-arithmetic
prism variant in the far-separation hot class (4.5e-4): reconstructing the
analytical prism tensor in `float` loses four digits where its corner sums
cancel, which is a further argument against the one finite family the device
could execute.

The production prism refactor is bit-identical, not merely close. A one-off
probe linked the pre-refactor implementation from the parent commit beside the
refactored one and compared
the IEEE bit patterns of 20,680 tensors over five prisms and 4136
displacements, including every face, edge, vertex and symmetry plane, the
coincident self tensor, the two-sided `getF_limit` branch and the far field:
zero differences, zero exceptions, and matching exception behaviour. The
invariant is now locked at the lowest layer by a new case in
`tests/test_rectangular_prism_magtense.cpp`, which requires the `long double`
instantiation of the shared kernel to equal the public tensor exactly and the
`double` one to stay within 1e-12.

`compute-sanitizer` 2026.1.1 over the new procedural prism device kernel and
the stored CUDA plans it is compared against, both pair directions:

| Tool | prism -> point | point -> prism |
|---|---|---|
| memcheck | 0 errors | 0 errors |
| racecheck | 0 hazards | 0 hazards |
| initcheck | 0 errors | 0 errors |
| synccheck | 0 errors | 0 errors |

racecheck matters here because the kernel accumulates each target's field from
several work items with atomics. One defect was found and fixed during the
study by the benchmark's own correctness column, not by the sanitizer: the
first device kernel applied a point source's coincident self pair instead of
excluding it, which the canonical operator marks `skip_for_identity`. It
showed up as an order-unity error on every point-source row, the affected
measurements were discarded and the prism cases re-measured with the corrected
kernel.

Four fresh pinned trees on the final state, each a clean configure, build,
full CTest and Python suite with conda `g++` 15.3.0 as C++ and CUDA host
compiler:

| Tree | CTest | pytest |
|---|---|---|
| portable CPU | 228 / 228 | 140 passed, 8 skipped |
| oneMKL | 228 / 228 | 142 passed, 6 skipped |
| CUDA | 228 / 228 | 145 passed, 3 skipped |
| CUDA + oneMKL | 228 / 228 | 147 passed, 1 skipped |

The skips are the optional backends absent from each configuration. The new
kernel case is test 94, "precision-generic prism kernel is the production
point tensor". The whitespace check is clean. `src/cache/`,
`include/cdfmm/c_api.h`, `src/bindings/` and `fortran/` are untouched, so the
cache format and keys, the C ABI and the Fortran interface are unchanged, and
no public header, option or enumeration gained or lost a member.

### Production decision

Outcome A of the brief: every finite procedural path loses, and the finite
operators stay precomputed on every backend. This is now a measured statement
rather than an assumption, and `docs/static-p2p.md` and
`docs/architecture.md` say so with the numbers. No production policy, option
or executor was added; `PointExpansionExecution` keeps its point-only
semantics.

Accepted from this study:

- `src/geometry/primitives/rectangular_prism_point_kernel.hpp`, the
  precision-generic MagTense prism point tensor that production instantiates
  in `long double` (bit-identical) and that made the CUDA measurement
  possible without a second copy of the mathematics; and
- the benchmark infrastructure (`benchmark_operator_representation`, its
  device kernel, `run_operator_representation.py`,
  `analyse_operator_representation.py`), which reproduces every table above
  from the final `refactor/architecture-v0.2` code.

Rejected: procedural finite P2P on the CPU (all eight pairs), procedural
finite P2P on CUDA (prism family measured, others not ported with the reason
recorded), procedural finite P2M and L2P for prisms and tetrahedra at orders
4-10 in both precisions, and the invariant/recurrence hybrid.

Observations for Phase 3C, recorded and not acted on: the point/prism pair
loops and the per-leaf P2M and per-target L2P plan construction are serial
while the polyhedron pair loops are parallel; the prism tensor's `long double`
arithmetic costs 2.7x its `double` equivalent for 4.5e-15 of agreement; and
`tetrahedron_averaged_monomial` heap-allocates a barycentric polynomial per
(mode, term, axis), which dominates exact tetrahedron P2M/L2P construction.

## Static-plan construction and plan-preparation optimization (Phase 3C)

### Question, starting point and method

Phase 3A and 3B optimised repeated evaluation; 3B.5 and 3B.5b settled which
operator representation to execute. None of them touched the cost of *making*
a plan. This phase asks where cold plan construction spends its time and
memory, and which of that work can be parallelised, reused, deduplicated or
avoided without slowing repeated evaluation.

Starting HEAD `a2af367` ("docs(perf): record the exact finite procedural
versus precomputed study") on `refactor/architecture-v0.2`, which
`origin/refactor/architecture-v0.2` already pointed at; the local branch was
fast-forwarded to it at the start of the session. Work was done on the
worktree branch `phase3c-construction` and fast-forwarded back.

Hardware: Intel i9-14900KF, eight P-cores used for every controlled
measurement (`OMP_NUM_THREADS=8`,
`OMP_PLACES={0},{2},{4},{6},{8},{10},{12},{14}`, `OMP_PROC_BIND=close`);
NVIDIA RTX 5090 (sm_120, 32 GB, 96 MB L2). Toolchain pinned explicitly rather
than inherited: g++ 15.3.0 from the `cdfmm` conda environment for C++ and as
the `nvcc` host compiler, CUDA 13.2, oneMKL 2026.1. The environment's `icpx`
/ `icx` / `NVCC_PREPEND_FLAGS` defaults are unset for every build in this
study.

Method. Construction and evaluation are measured separately.
`benchmarks/run_construction_matrix.py` sweeps geometry, size, order,
precision and backend with `CDFMM_DISABLE_CACHE` set, so every row is a cold
build, and records the per-phase `StaticPlanStatistics` timings. Correctness
is checked by byte-comparing the persisted geometry plan, which serialises the
canonical P2P blocks, the P2M plans and the L2P evaluators: if the file is
identical, the operators are identical bit for bit, which is a far stronger
statement than a tolerance comparison of fields.

Two measurement faults were found and corrected during the study, and both
changed conclusions:

- `benchmark_uniform_fmm` hard-coded point far-field models for finite
  bodies, so the exact finite P2M/L2P operators -- and with them the
  tetrahedron barycentric expansion that Phase 3B.5b flagged -- were never
  exercised by any benchmark. `--far-field-model exact` restores the
  `UniformFmmOptions` default and is what makes Lead 5 measurable at all.
- One killed benchmark process survived and competed for the cores during an
  early exact far-field sweep, inflating those rows by roughly 2x. Those
  numbers were discarded and re-measured on an idle machine.

### Baseline (`a2af367`), cold, cache disabled, 4096 bodies, p = 6, FP32, CpuStatic

| case | total setup | P2P stage | P2P share | unique tensors |
|---|---|---|---|---|
| point, random | 1.448 s | 0.062 s | 4.3 % | n/a |
| point, lattice | 1.488 s | 0.087 s | 5.8 % | 172 |
| prism lattice | 66.81 s | 65.41 s | 97.9 % | 248 |
| prism irregular | 66.11 s | 64.70 s | 97.9 % | 24,947 |
| tetrahedron lattice | 15.87 s | 14.48 s | 91.2 % | 187 |
| tetrahedron irregular | 19.74 s | 18.35 s | 93.0 % | n/a |

For every finite case the canonical near-field build is 91-98 % of cold
construction. For point cases the floor is `universal_operator_build`
(1.33 s at p = 6): the 316 M2L class matrices, which depend only on basis,
order and precision, are already OpenMP-parallel, and are served from the
universal cache in normal use.

### Dominant cost 1: the near field built every pair, and built it serially

Two independent causes, found by audit and confirmed by measurement.

The generic pair loop (`src/operators/p2p.cpp`) serving the point/prism
combinations and both point/tetrahedron directions -- six of the nine pairs --
was serial. Nothing forced that: the block vector is sized and ordered before
any tensor is built and every iteration writes one distinct entry. It was
simply the one loop that had never been given the OpenMP exception
scaffolding the two polyhedron loops already carried.

Far larger: every pair was evaluated although almost all of them describe the
same operator. A pair tensor is a pure function of the displacement and the
participating body records, and a periodic image shift is folded into the
displacement before any tensor call, so pairs whose inputs agree bit for bit
are the same interaction. The decisive fact is that they agree *exactly*
rather than approximately, because normalisation puts body centres on a
canonical grid. Measured on the baseline with a temporary probe over the
exact inputs, 4096 bodies at depth 3:

| case | near-field pairs | distinct exact inputs | redundancy |
|---|---|---|---|
| point, random | 745,226 | 741,130 | 1.006x |
| point, lattice | 681,472 | 342 | 1992x |
| prism lattice | 681,472 | 343 | 1986x |
| prism irregular | 681,472 | 34,643 | 19.7x |

The irregular row is the informative one: 343 distinct displacements survive
even when every body has its own size record, so the reuse comes from the
lattice geometry and not from the bodies being identical.

The cost asymmetry decides where the classification is worth its own price.
The near-field stage costs about 96 us per pair on the prism lattice
(65.4 s / 681,472) against 78 ns per pair on random points (0.058 s /
745,226), and the point figure is itself dominated by sorting the interaction
list rather than by the tensor. A hash lookup is therefore invisible beside a
finite tensor and would be the whole cost of a point pair, which -- as the
table shows -- has no duplicates to find anyway. Point pairs build directly.

### Accepted: `perf(p2p): build each exact near-field operator once and in parallel`

All three pair loops now classify their pairs by exact operator inputs, build
one tensor per class in parallel, and scatter. Design points that matter:

- the key holds raw bit patterns and is never compared with a tolerance;
  `-0.0` stays distinct from `+0.0`, which can at worst repeat one build,
  where merging them would assume a continuity the exact corner formulas do
  not have;
- classes are numbered in first-seen order, so the classification, the
  representatives and every built value are independent of thread count and
  of hash iteration order;
- classification is abandoned when an initial sample shows too few duplicates
  to repay it, which bounds both the table and the wasted lookups on
  irregular geometry and only ever changes performance;
- the tetrahedron pair classifies among its reciprocity owners alone, so the
  reciprocal pair keeps sharing one owner's bits exactly as before, and the
  coincident-geometry selector is part of the key because it selects a
  different algorithm;
- a failure is now reported from the lowest failing pair index in all three
  loops, which is the pair a serial build would have reached first, so the
  reported cause no longer depends on scheduling.

Cold construction, 4096 bodies, p = 6, FP32, eight threads, cache disabled:

| case | P2P before | P2P after | P2P speedup | total before | total after |
|---|---|---|---|---|---|
| prism lattice | 65.41 s | 0.119 s | 550x | 66.81 s | 1.515 s |
| prism irregular | 64.70 s | 0.547 s | 118x | 66.11 s | 1.946 s |
| tetrahedron lattice | 14.48 s | 0.150 s | 96.6x | 15.87 s | 1.537 s |
| tetrahedron irregular | 18.35 s | 14.71 s | 1.25x | 19.74 s | 16.09 s |
| point random | 0.062 s | 0.061 s | 1.02x | 1.448 s | 1.439 s |

Irregular tetrahedra have no exact duplicates to find -- every body carries
its own four vertices -- so classification abandons as designed and the gain
there is the rebalanced schedule alone. That row is the honest limit of the
mechanism, not a defect in it.

### Dominant cost 2: the finite endpoint operators, once the near field was fixed

With the near field fixed, the remaining serial work showed up only under
exact finite far-field models, which no benchmark had been exercising. At
4096 bodies, p = 6, the tetrahedron case spent 32.6 s of its 34.4 s
construction in the two serial endpoint loops.

A finite P2M or L2P operator is a pure function of the expansion basis, the
body's shape record and its displacement from its leaf centre. Leaf P2M
entries are indexed by leaf-local source, so two leaves whose bodies sit at
the same offsets with the same shapes produce identical entries; a target's
L2P rows depend on nothing but its own displacement and shape. Both are
therefore reusable on exactly the same bitwise terms as the near field, and
both loops are embarrassingly parallel -- the only shared state was the byte
accounting inside the loop, which is now summed afterwards.

### Accepted: `perf(plan): reuse and parallelise the finite endpoint operators`

Measured against the same tree with the accepted P2P change but serial
endpoint loops, 4096 bodies, exact far-field models, cache disabled:

| geometry | p | P2M before -> after | L2P before -> after | total setup |
|---|---|---|---|---|
| prism | 4 | 0.0228 s -> 0.0008 s (28x) | 0.0384 s -> 0.0009 s (43x) | 0.321 s -> 0.260 s |
| tetrahedron | 4 | 0.3002 s -> 0.0014 s (214x) | 0.6857 s -> 0.0022 s (312x) | 1.266 s -> 0.284 s |
| prism | 6 | 0.1153 s -> 0.0021 s (55x) | 0.1742 s -> 0.0014 s (124x) | 1.811 s -> 1.523 s |
| tetrahedron | 6 | 5.0147 s -> 0.0115 s (436x) | 11.0745 s -> 0.0228 s (486x) | 17.62 s -> 1.576 s |
| prism | 8 | 0.3646 s -> 0.0040 s (91x) | 0.5315 s -> 0.0028 s (190x) | 12.94 s -> 11.99 s |

With point far-field models at 32,768 bodies the same change gives 4-7x
rather than hundreds, and that number is now correct rather than
disappointing: what remains is the unavoidable copy of one materialised plan
per leaf and per target (about 51 MB of L2P rows at p = 6), not the build.
Sharing that storage instead of copying it would need a plan-representation
and cache-format change and was not attempted.

Parallelism alone, measured separately before reuse was added, gave 7.4-8.6x
on both loops, so the two mechanisms compose rather than overlap.

### Rejected and not pursued

**Prism `long double` arithmetic (Phase-3B.5b lead 6).** Not changed, and no
numerical audit was needed, because the premise no longer holds. 3B.5b
measured the shared exact prism point formula 2.7x faster in `double` than in
`long double`, agreeing to 4.5e-15 of field scale, when that formula ran once
per near-field pair. It now runs once per *distinct* operator. On the 4096-body
prism lattice the exact tensor evaluation is a small fraction of a near-field
stage that is itself under a tenth of construction, so a 2.7x there is
invisible: of the 0.0685 s canonical stage on the 4096-body prism lattice, the
343 exact tensors are about 4 ms. The irregular prism case is the one where
the arithmetic still dominates its own stage -- roughly 0.42 s of a 0.49 s
canonical stage -- and even there a 2.7x would return about 0.26 s of a
1.95 s cold build, some 13 %, bought by changing the numerical contract of the
production exact tensor. That fails the phase's own rule that construction
speed alone does not justify a precision reduction, so production keeps
`long double` and the numerical audit the change would have required was not
needed.

**Sharing endpoint operator storage instead of copying it.** After reuse, what
remains in the point far-field endpoint stages is the copy of one materialised
P2M plan per leaf and one L2P evaluator per target. Referencing a shared
operator instead of copying it would remove that, but `P2MPlan` and
`StaticL2PEvaluator` own their storage and both are serialised by the
geometry cache, so it is a plan-representation and cache-format change. Out of
scope for this phase; recorded as the next endpoint opportunity.

**Skipping the canonical near-field build for point plans (lead 7). STOPPED
at the cache boundary, not attempted.** A CPU plan whose resolved packing is
`PointGeometry` never executes the canonical near-field tensors: the compact
plan is already skipped and `release_stored_p2p_tensors()` frees them once the
CPU far-field packing is built. The build itself is still paid, and it is
worth real time -- 1.16 s of the 3.15 s cold construction of a 32,768-point
random plan at order 6.

Every in-process consumer was classified and none of them blocks it. The
compact plan (`src/fmm/plan_preparation.cpp`), the signed tensor dictionary
(`src/fmm/execution_setup.cpp`, which early-returns unless the policy chose
`SignedDictionary`), the CUDA BSR/canonical/leaf plans, and the `CanonicalAos`
evaluation branch are all unreachable under `PointGeometry`, and an explicit
`UniformFmmOptions::p2p_packing` request for any other packing already turns
the predicate off before the build. Backend resolution happens before plan
preparation, so nothing re-chooses the packing afterwards.

The cache does block it, in two ways, and this is the phase's declared stop
condition rather than a judgement call:

- `write_geometry_cache` serialises the canonical blocks unconditionally
  (`src/cache/geometry.cpp`, called from `src/fmm/plan_preparation.cpp`), so a
  skipped build would write a structurally valid but empty near-field section.
  The on-disk *format* would be unchanged; the *contents* would not.
- `CacheIdentityInputs` (`src/cache/internal.hpp`) carries basis, precision,
  order, geometries, near and far models, `use_reduced_symmetry_p2p`, periodic
  options, the tree, the size and tetrahedron records and the fixed identity
  map -- but neither `p2p_packing` nor the backend. One geometry therefore has
  one cache entry shared by every packing and every backend, and a warm read
  reconstructs the operator from the file rather than rebuilding it. An entry
  written by a skipped build would silently give a later `CanonicalAos`,
  `ParticleRowSoa` or CUDA plan an empty near field: wrong results, not an
  error.

One exposed statistic would also change: `p2p_interactions` is the only
operator-derived counter that `release_stored_p2p_tensors()` does not zero, so
it would fall to zero for point plans. It has a topology-derived equivalent
already computed for the CUDA policy, so that part is not the obstacle.

Two related wastes were found on the same path and are recorded, not fixed:
even under `PointGeometry` the FP32 conversion still quantises the whole
canonical operator into an FP32 copy that nothing reads, and the FP32 BSR plan
is built on CPU plans with a fixed identity map although only CUDA consumes
it.

The opportunity is real and worth taking, but every safe framing of it -- a
cache key that distinguishes a plan without a stored near field, or a header
flag saying the section is unpopulated -- is a persistent-cache change. Phase
3C is not authorised to make one, so the subtask stops here.

### Cache behaviour

Three states of the same plan, spherical, FP32, eight threads: cold with
`CDFMM_DISABLE_CACHE`, cold into an empty cache directory, and a warm hit on
what that wrote.

| case | cold, no cache | cold + write | warm hit |
|---|---|---|---|
| point, 4096, p = 6 | 1.479 s | 1.540 s | 0.110 s |
| prism, 4096, p = 6 | 1.515 s | 1.576 s | 0.107 s |
| tetrahedron, 4096, p = 6 | 1.534 s | 1.610 s | 0.106 s |
| prism, 4096, p = 8 | 11.99 s | -- | -- |
| prism, 32768, p = 6 | 3.066 s | 3.407 s | 0.978 s |

Writing the geometry plan costs 0.035 s at 4096 bodies and 0.298 s at 32,768.
A warm hit is 14x faster than a cold build at 4096 bodies and 3.1x at 32,768.

Two things the warm rows make explicit, both of which the new timers were
needed to see:

- The `universal_operator_build` floor disappears entirely on a warm hit
  (1.33 s at p = 6, 11.79 s at p = 8, to zero). It depends only on basis,
  order and precision, so one file serves every geometry ever built at that
  configuration. The cold-cache-disabled numbers elsewhere in this section
  therefore overstate what a user pays in normal operation.
- What remains on a warm hit is mostly the derived execution packings, which
  are deliberately not persisted: 0.469 s of the 0.978 s warm setup at 32,768
  bodies is the near-field derived packing being rebuilt from the loaded
  canonical operator. After this phase that is the largest single warm-cache
  cost and the clearest remaining lead.

Cache format, cache keys, and the ability to load an existing compatible file
are unchanged; no cache code was touched.

### Backend preparation

Backend-specific setup at 4096 bodies, p = 6, FP32, over all four backends:
`backend_packing` 0.006-0.043 s, `far_field_packing` at most 0.003 s and
correctly zero for `CudaFull` which does not build it, `cuda_upload`
0.0015-0.039 s, and `precision_conversion` 0.017-0.053 s. None of these is a
bottleneck once geometry construction is fixed, and none was changed. The
largest of them, the FP64-to-FP32 conversion, is also the one that still
converts a canonical near-field operator that a `PointGeometry` plan will
never read; see lead 7 above.

### Memory

Peak resident set over the whole 42-row matrix stayed below 0.6 GiB, and the
largest rows are the CUDA ones, where the device staging dominates rather than
the classification.

The classification's own transient cost is small and bounded by construction.
It holds one `uint32_t` class per pair, allocated up front, plus one key per
distinct class: 2.7 MB and about 77 kB respectively for the 4096-body prism
lattice. When classification is abandoned the table can never exceed the
65,536-pair sample, so an irregular plan pays the per-pair class array and a
bounded table and nothing more. No configuration trades setup time for a large
transient allocation.

### Final stage decomposition

Cold, cache disabled, regular prism lattice, 4096 bodies, p = 6, FP32,
CpuStatic, eight threads, after both accepted changes:

| stage | seconds | share |
|---|---|---|
| `universal_operator_build` | 1.3355 | 87.95 % |
| `p2p_tensor_plan` | 0.1209 | 7.96 % |
| -- `p2p_interaction_setup` | 0.0036 | 0.24 % |
| -- `p2p_canonical_operator` | 0.0685 | 4.51 % |
| -- `p2p_derived_packing` | 0.0488 | 3.21 % |
| `precision_conversion` | 0.0483 | 3.18 % |
| `backend_packing` | 0.0334 | 2.20 % |
| `topology_construction` | 0.0035 | 0.23 % |
| `tree_construction` | 0.0025 | 0.16 % |
| `far_field_packing` | 0.0013 | 0.09 % |
| `m2m` / `l2l` / `m2l` / `p2m` / `l2p` / `geometry_hash` / `normalisation` | each under 0.002 | under 0.1 % each |
| total setup | 1.5184 | 100 % |

`precision_conversion` and `backend_packing` overlap by the FP32 near-field
packing, which both timers see; they are not additive.

Two things follow.

First, what is left of the near-field stage is no longer the exact
mathematics. Of the 0.0685 s canonical stage, the 343 distinct exact tensors
account for about 4 ms on eight threads, and the classification for roughly
another 12 ms -- the difference between this stage and the point case, which
does no classifying, is only some 20 ns per pair. The remainder is the
`std::sort` of the 681,472-entry interaction list and the row-offset pass,
both of which predate this phase and are now the largest single item in the
stage. That is an estimate by subtraction rather than a direct measurement,
and it names the next lever: the interaction list is generated in a structured
order from the leaf records, so it could be produced already sorted, or sorted
in parallel, instead of being sorted serially after the fact.

Second, everything above the near field is now dominated by the
geometry-independent universal operator bank, which the cache removes
entirely.

### Threading

Construction thread scaling at 4096 bodies, p = 6, FP32, on one, two, four and
eight pinned P-cores. The near-field stage is shown, plus the endpoint stages
for the exact far-field case.

| builder | 1 | 2 | 4 | 8 | 8-thread speedup |
|---|---|---|---|---|---|
| tetrahedron irregular near field, no reuse possible | 115.26 s | 58.35 s | 28.73 s | 14.69 s | 7.84x |
| prism irregular near field, 34,643 classes | 3.417 s | 1.772 s | 0.961 s | 0.549 s | 6.22x |
| prism lattice near field, 343 classes | 0.152 s | 0.132 s | 0.124 s | 0.120 s | 1.27x |
| tetrahedron exact far field, P2M | 0.0116 s | 0.0109 s | 0.0104 s | 0.0103 s | 1.13x |
| tetrahedron exact far field, L2P | 0.0232 s | 0.0228 s | 0.0226 s | 0.0228 s | 1.02x |

The two mechanisms divide the work exactly as intended. Where reuse cannot
help -- irregular tetrahedra, whose four vertices differ per body -- the
parallel build carries the case and scales almost linearly. Where reuse has
already collapsed millions of pairs to a few hundred operators, thread count
barely matters, because what is left is the serial classification pass and the
serial sort of the interaction list rather than the parallel build. Neither
case regresses in the other's regime, which is why both were kept.

The schedules are `dynamic` throughout the construction builders, because an
exact finite operator's cost varies by an order of magnitude between a
coincident pair, an adjacent pair and a far-separated one, and because a
classified build has few, very unequal items. The M2L class loop keeps the
`dynamic` schedule it already had. Nothing nests: construction runs before any
evaluation parallel region and the endpoint and near-field loops are
sequential with respect to one another.

### Final results (accepted HEAD versus the Phase-3C baseline `a2af367`)

Cold construction, cache disabled, 4096 bodies, p = 6, FP32, CpuStatic, eight
P-cores, identical builds and settings:

| case | baseline setup | final setup | speedup | baseline P2P | final P2P |
|---|---|---|---|---|---|
| point, random | 1.448 s | 1.436 s | 1.0x | 0.062 s | 0.062 s |
| point, lattice | 1.488 s | 1.478 s | 1.0x | 0.087 s | 0.086 s |
| prism lattice | 66.81 s | 1.518 s | 44.0x | 65.41 s | 0.121 s |
| prism irregular | 66.11 s | 1.950 s | 33.9x | 64.70 s | 0.552 s |
| tetrahedron lattice | 15.87 s | 1.535 s | 10.3x | 14.48 s | 0.151 s |
| tetrahedron irregular | 19.74 s | 16.08 s | 1.2x | 18.35 s | 14.70 s |

Under exact finite far-field models, where the endpoint operators are real
work, the order-6 tetrahedron case falls from 17.62 s to 1.576 s with P2M 436x
and L2P 486x (measured against the same tree carrying the accepted near-field
change but serial endpoint loops, because the baseline benchmark could not
select exact far-field models at all).

The baseline was measured at the primary comparison point the phase specified
-- p = 6, 4096 bodies, FP32, CpuStatic, all six geometries -- and not extended
to the secondary rows (the order sweep, the backend spot checks and FP64).
The final matrix covers all of those, so those rows have a final number but no
baseline twin. Nothing in the conclusions rests on them: the near-field stage
is geometry work shared by every backend and is almost independent of the
expansion order, as the final matrix shows directly (the prism lattice P2P
stage is 0.121 s at p = 4, 6 and 8 alike).

What this does and does not change. The *cold* path improved by up to 44x; the
*warm* path is unchanged, because a geometry-cache hit returns before any of
these builders runs and loads the operators instead. That is visible in the
cache table above, where the warm hit costs the same 0.107 s for point, prism
and tetrahedron geometry even though their cold builds differed by a factor of
forty. What this phase fixed is therefore exactly what a first-ever geometry,
a cache miss, a changed geometry or a cache-disabled run pays -- which is also
what every benchmark, test and parameter sweep pays.

### Correctness

The construction changes are allowed to change construction order, scheduling
and temporary representation, and nothing else. That is checked directly
rather than inferred, by byte-comparing the persisted geometry plan against a
binary built without the change: the file serialises the canonical P2P blocks,
the P2M plans and the L2P evaluators, so identical bytes mean identical
operators. A tolerance comparison of evaluated fields would not have been
able to make that statement.

Thirty-nine configurations are byte identical:

- the near-field change, 17 configurations: all three geometries against
  themselves on a regular lattice, the same with per-body records, and a
  general layout; the six mixed geometry pairs; FP32 and FP64; orders 4, 6 and
  8; free-space and fully periodic evaluation.
- the endpoint change, 22 configurations: three geometries times exact and
  point far-field models times three layouts, plus FP64 at order 4 for prisms
  and tetrahedra, a periodic prism lattice, and the Cartesian basis.

`tests/test_p2p_exact_reuse.cpp` pins the reuse contract itself rather than
its consequences, because a byte comparison can only say that today's two
implementations agree. It checks that a repeated displacement really does
carry repeated bits; that a batched build matches a pair-by-pair reference
bitwise for every geometry combination, offering the reciprocal partner where
the tetrahedron pair shares one owner's bits; that a prism half-width or a
tetrahedron vertex one ULP away never aliases; that a displacement one ULP
away never aliases; that per-body sizes reach their own operators; that
periodic images landing on one displacement legitimately share and a different
image does not; that the finite self tensor survives and is not confused with
the point identity exclusion; and that a coincident point pair keeps its
undefined tensor and its marker.

Two of those tests were wrong when first written and were corrected, not the
code. A tetrahedron pair does not match a single-pair reference bitwise,
because the pre-existing reciprocity rule makes `K_{t<-s}(r)` and
`K_{s<-t}(-r)^T` share whichever owner computed them, and the identity marker
carries the interaction's own request rather than meaning "this is a self
pair".

Cache compatibility is confirmed in both directions. The byte comparisons
above show that a cache file written after the change is identical to one
written before it, and a separate check has the baseline binary load a cache
written by the changed binary and report both a universal and a geometry hit.
No file under `src/cache/` was modified, and the cache identity inputs are
unchanged.

### Construction determinism

Five loops became parallel in this phase, so a race would show as a plan that
differs between runs or between thread counts. Six cases -- prism,
tetrahedron and point geometry, each as a regular lattice with exact
far-field models and as an irregular general layout -- were each built at one,
two, four and eight threads and once more at eight, and every persisted plan
byte-compared against the single-threaded one. All 24 comparisons are
identical. That is the host-side counterpart of `racecheck`, and it also
confirms that numbering classes in first-seen order really does make the
result independent of scheduling.

No CUDA source was changed in this phase, so no `compute-sanitizer` run was
warranted; the CUDA and CUDA-plus-oneMKL trees below exercise the device
paths, and the host plan they upload is byte-identical to the one the previous
implementation produced.

### Tests and validation

Four fresh trees at the accepted HEAD, each configured with g++ 15.3.0 pinned
explicitly for C++ and as the `nvcc` host compiler:

| tree | build | CTest | Python |
|---|---|---|---|
| portable CPU | ok | 237/237 passed | 140 passed, 8 skipped |
| oneMKL | ok | 237/237 passed | 142 passed, 6 skipped |
| CUDA | ok | 237/237 passed | 145 passed, 3 skipped |
| CUDA + oneMKL | ok | 237/237 passed | 147 passed, 1 skipped |

`git diff --check` is clean. The skip count falls from eight to one as
capability is added, which is the expected shape: the combined tree runs the
CUDA-only and oneMKL-only cases that the portable build skips.

One real regression was caught here and nowhere else. The warm-cache
instrumentation initially recorded its work under `p2p_tensor_plan` as well,
which broke the contract in `tests/test_cache.cpp` that every construction
timer reports zero calls on a geometry-cache hit -- the way a warm plan is
distinguished from a rebuilt one. Byte-comparing plans could not have found
it, because timings are not serialised. Only the new `p2p_derived_packing` and
`precision_conversion` timers record the warm path now.

### Repeated-evaluation regression gate

Eleven cases at 32,768 bodies, order 6, across CpuStatic, oneMKL, CudaFull and
CudaPartial, 1000 timed evaluations per binary per case. Both binaries share
one geometry cache and each case is warmed first, so both measured runs see
the same cache state: a cold build and a warm load can resolve their execution
packing differently, and comparing one against the other would read as a
regression that is not there.

| case | baseline | current | ratio |
|---|---|---|---|
| cpu-static point M fp32 | 12.249 ms | 13.710 ms | 1.119 (see below) |
| cpu-static point M fp64 | 15.387 ms | 15.194 ms | 0.988 |
| cpu-static prism lattice fp32 | 9.600 ms | 9.341 ms | 0.973 |
| cpu-static tetra lattice fp32 | 9.831 ms | 9.525 ms | 0.969 |
| onemkl point M fp32 | 13.921 ms | 13.828 ms | 0.993 |
| cuda-full point M fp32 | 0.37807 ms | 0.37871 ms | 1.002 |
| cuda-full point M fp64 | 2.3223 ms | 2.3193 ms | 0.999 |
| cuda-full point L fp32, P2P heavy | 0.24404 ms | 0.24557 ms | 1.006 |
| cuda-full prism lattice fp32 | 0.36122 ms | 0.36226 ms | 1.003 |
| cuda-full tetra lattice fp32 | 0.36016 ms | 0.36037 ms | 1.001 |
| cuda-partial point M fp32 | 0.99161 ms | 0.99097 ms | 0.999 |

The one row that moved was the first measured, immediately after another stage
finished, and it does not reproduce. Re-measured on a settled machine with the
binary order alternated, four rounds give 1.0137, 1.0202, 1.0299 and 0.9635 --
a mean of 1.007, with the current binary faster in one round -- and both
binaries drop from 12-14 ms to 10.6-11.3 ms, which is what shows the original
row was taken while the machine was still busy. The run-to-run spread of about
3.5 % is the noise floor of that case.

Nothing here is a regression, which is what the mechanism predicts: the plans
are byte-identical and no execution packing, kernel or policy was touched.

### Measurement provenance

Two contamination sources were identified and handled rather than absorbed.
A benchmark process survived a cancelled sweep and competed for the cores,
inflating an early exact far-field measurement roughly twofold; those rows were
discarded and re-measured. Separately, an unrelated task ran on the
workstation from about 09:23; every construction timing in this section was
recorded before that, between 07:30 and 09:10, and the only stage that
overlapped it was the four-tree validation, which is a correctness check that
contention can slow but not invalidate.

### Remaining bottlenecks

In rough order of what a user would notice.

1. **The universal operator bank.** `universal_operator_build` is 1.33 s at
   p = 6 and 11.79 s at p = 8, which is 88 % and 99 % of a cold
   cache-disabled build of every geometry measured here. It is the 316 M2L
   class matrices, it depends only on basis, order and precision, it is
   already OpenMP-parallel over the classes, and the universal cache removes
   it entirely after the first build at a given configuration. Shrinking it
   would mean exploiting symmetry between the 316 classes, which is M2L
   operator mathematics and outside this phase.

2. **Irregular tetrahedra.** 14.7 s at 4096 bodies and p = 6, and 134 s at
   32,768 bodies and p = 8, is exact Galerkin face-pair integration with no
   duplicate to find: every body carries its own four vertices. The build is
   parallel and scales 7.8x on eight cores, so the only levers left are in
   the tetrahedron pair mathematics itself.

3. **The near-field interaction sort.** With the exact tensors reduced to a
   few hundred builds, the largest single item inside the canonical stage on
   a lattice is the serial `std::sort` of the interaction list and the
   row-offset pass -- of a 0.0685 s stage, roughly 40 ms by subtraction. The
   list is generated in a structured order from the leaf records, so it could
   be produced already sorted or sorted in parallel. Neither predates nor was
   introduced by this phase.

4. **Derived packings on a warm cache.** 0.469 s of the 0.978 s warm setup at
   32,768 bodies rebuilds the near-field derived packing from the loaded
   canonical operator. The derived forms are deliberately not persisted, so
   this is the largest warm-cache cost and the clearest next lead.

5. **Work a point plan never reads.** The canonical near-field operator, and
   its FP32 conversion, are built for point plans whose executor is
   `PointGeometry` and never reads them: 1.16 s of a 3.15 s cold build at
   32,768 points. No in-process consumer blocks skipping it; the geometry
   cache does, because it serialises the operator under a key that
   distinguishes neither packing nor backend. A cache change, and therefore
   explicit approval, is required.

6. **Endpoint plan copying.** After reuse, the point far-field P2M and L2P
   stages are dominated by copying one materialised plan per leaf and per
   target -- about 51 MB of L2P rows at 32,768 bodies and p = 6 -- rather
   than by building anything. Sharing that storage is a plan-representation
   and cache-format change.

7. **A cheaper classification key.** The key carries the shape record even
   when the whole build shares one, which is the common case. Dropping it
   there would shorten the key from nine words to three for a common-size
   prism plan. Worth a small part of the canonical stage and not taken,
   because the stage is no longer dominated by the classification.

## Dense/all-to-all construction optimization (Phase 3C.5)

### Question, starting point and method

Phase 3C optimised static-plan construction for the FMM hierarchy. It did not
touch `DenseDirectPlan`, the exact dense all-to-all baseline, which builds six
immutable `Nt x Ns` matrices from the analytical pair tensors and applies them
with nine GEMVs per evaluation. This phase asks how quickly that plan can be
made, and how much of the Phase-3C construction strategy actually transfers to
it.

Starting HEAD `cefc975` ("docs(perf): state what the Phase 3C baseline does
and does not cover"), a fast-forward descendant of
`refactor/architecture-v0.2` carrying the nine Phase-3C commits. Work was done
on the worktree branch `phase3c5-dense-construction`.

Hardware: Intel i9-14900KF, eight P-cores for every controlled measurement
(`OMP_NUM_THREADS=8`, `OMP_PLACES={0},{2},{4},{6},{8},{10},{12},{14}`,
`OMP_PROC_BIND=close`); NVIDIA RTX 5090 (sm_120, 32 GB). Toolchain pinned
rather than inherited: g++ 15.3.0 from the `cdfmm` conda environment for C++
and as the `nvcc` host compiler, CUDA 13.2/13.3, oneMKL 2026.1, with the
environment's `icpx`/`icx`/`NVCC_PREPEND_FLAGS` defaults unset for every build
tree.

Method. A pinned worktree at the pre-change commit was built in the same
session and with the same benchmark source, so every baseline number here was
measured rather than remembered, and the comparison is baseline binary against
final binary on an otherwise idle machine. Correctness is bitwise: the
benchmark hashes the raw bytes of the six matrices, so a construction change
that reorders the work must leave that hash unchanged.

### Constructor architecture (the audit that decided the work)

Ten questions were asked of the baseline before anything was changed.

1. The six components are computed inline in the constructor, one `PairTensor`
   per `(target, source)` pair, dispatched by geometry to
   `operators::p2p::build_pair` or one of the five tetrahedron entry points.
2. **CPU and CUDA share host construction.** `CudaDenseDirectPlan` constructs a
   complete host `DenseDirectPlan` and retains only its device copy. The
   portable CPU and oneMKL backends share it too and differ only in
   `evaluate()`. There is therefore one construction path for all three
   backends, and construction is measured once per geometry rather than once
   per backend.
3. **FP32 does not build FP64 first.** Each tensor is quantised at the point of
   store with `static_cast<Scalar>`; no complete FP64 matrix is ever
   materialised and there is no separate conversion pass to optimise. The
   benchmark keeps a `precision_conversion_seconds` column, always zero, to
   record that.
4. Self and identity semantics: a pair is an omitted self interaction only when
   the explicit identity map names it *and* the effective source geometry is a
   point dipole. Finite self interactions keep their physical tensor. Nothing
   is inferred from coordinate equality.
5. **Geometry was prepared per pair, not per body.** A tetrahedron pair called
   `prepare_tetrahedron` on both bodies every time; a mixed prism/tetrahedron
   pair called `prepare_polyhedron_body` on both, which heap-allocates four
   vectors. All of it is a pure function of the record.
6. The pair loop was already OpenMP-parallel, `schedule(static)` above 256
   pairs. Parallelism was not the missing piece.
7. Storage is six separate target-major vectors, entry `target * Ns + source`.
8. Temporaries were one stack `PairTensor` per pair plus the per-pair heap
   allocations of item 5.
9. CUDA uploads exactly the final dense representation, six `cudaMemcpyAsync`
   calls from pageable host memory.
10. oneMKL and portable CPU do not differ in construction at all.

### Exact redundancy

An exact dense pair tensor is a pure function of the displacement, the two
body records, and whether the pair is an omitted point self interaction.
Counted with the production key and no sampling gate:

| workload | pairs | distinct exact inputs | reuse |
|---|---|---|---|
| lattice, point->point 512^2 | 262,144 | 3,375 | 77.7x |
| lattice, prism->prism 256^2 | 65,536 | 1,723 | 38.0x |
| lattice, tetra->tetra 192^2 | 36,864 | 1,243 | 29.7x |
| anisotropic, prism->prism 256^2 | 65,536 | 4,595 | 14.3x |
| refined, prism->prism 256^2 | 65,536 | 5,418 | 12.1x |
| refined, tetra->tetra 192^2 | 36,864 | 3,646 | 10.1x |
| refined-anisotropic, prism->prism 256^2 | 65,536 | 11,614 | 5.6x |
| refined-anisotropic, tetra->tetra 192^2 | 36,864 | 6,734 | 5.5x |
| lattice-irregular, any pair | 65,536 | 65,536 | 1.0x |
| random, any pair | 65,536 | 65,536 | 1.0x |

Two rows of that table matter more than the headline ones.

**Irregular geometry has no reuse at all, and that is arithmetic rather than a
defect.** In an all-to-all plan every pair holds a unique combination of source
record and target record, so a per-body record makes every key unique by
construction. This differs from the FMM near field, where Phase 3C still found
19.7x on irregular prisms, because a near-field list pairs only nearby bodies.
The mechanism's limit here is sharper than it was there.

**Anisotropy costs reuse, through floating-point rounding rather than
geometry.** Stretching the lattice raises the distinct-input count of a point
plan 4.5x. The cause is exact: on a unit lattice, integer coordinates subtract
exactly, so fifteen index differences give fifteen distinct displacement bit
patterns; with a spacing of 0.4 or 2.2, `spacing * a - spacing * b` rounds
differently with magnitude, and the same fifteen index differences give
thirty-three and thirty-one distinct patterns. The measured 15,345 distinct
displacements is exactly `15 * 33 * 31`. Reuse stays exact and the results stay
bit-identical; it simply finds less. `DenseDirectPlan` has no coordinate
normalisation to hide this — the FMM path's canonical grid is a property of
tree construction, not of the dense plan — so bitwise agreement for a dense
plan rests entirely on how the caller generated its positions. A measurement
taken only on an isotropic unit lattice overstates what exact reuse is worth,
and this study would have reported such a number had the anisotropic workloads
not been added.

### Accepted: `perf(direct): build each exact dense pair tensor once`

The constructor now prepares each distinct finite record once, classifies
pairs by exact operator inputs, builds one tensor per class in parallel with a
`dynamic` schedule, and scatters with a `static` one into the entries each
pair already owned. Both paths share a single tensor-valued lambda, so the
classified and unclassified builds cannot drift apart.

Two policy decisions, both measured.

**Point-to-point plans never classify.** A point pair costs about 4 ns; the
cheapest finite pair — point-to-prism or point-to-tetrahedron — costs about
260 ns; a key lookup costs tens of nanoseconds. Classification would be most
of a point plan's build and is negligible beside any finite one. The supported
geometries separate by two orders of magnitude with nothing in between, so
"some side is finite" is a measured predicate rather than an arbitrary rule
about geometry. Point plans keep 77x redundancy unexploited on purpose: taking
it would cost more than the arithmetic it saves.

**The gate caps transient tensor storage in bytes, not as a reuse ratio.** The
distinct tensors are held until they are scattered, so what must be bounded is
their storage; the cap is half the matrix bytes the plan retains anyway, which
is `pairs/4` classes in FP32 and `pairs/2` in FP64.

That distinction was not cosmetic, and getting it wrong was this phase's one
substantive design error. The gate was first written as a fixed eightfold
reuse requirement, justified by the observation that the then-available
workloads sat either at no reuse or above thirtyfold, "never in between". The
locally refined anisotropic grid — a regular grid with one octant subdivided,
which is what a real discretisation looks like — sits at 5.5x, and the
eightfold gate discarded it: an exact prism-to-tetrahedron build stayed at
5.11 s when 0.91 s was available. The byte budget keeps those cases and still
provides the bound the ratio was there to provide. The lesson is recorded
because the failure mode is general: a threshold interpolated between two
extremes is a guess about the middle, and the middle is where real geometry
lives.

### Accepted: `perf(geometry): derive a tetrahedron's point field once per record`

`tetrahedron_magnetisation_tensor` re-derived the volume, the largest edge
length and all four face frames on every evaluation, although each is a pure
function of the record, and the far-separation quadrature evaluates one source
at 216 nodes. The derivation is now split from the evaluation
(`PreparedTetrahedronPointField`, `prepare_tetrahedron_point_field`,
`tetrahedron_point_tensor_prepared`), the quadrature prepares its source once,
and the record-level entry point is the two composed, so its cost and its
result are unchanged.

This was predicted to be the dominant remaining cost for tetrahedron sources
and it is not: it is worth about 9% of a tetrahedron-source far pair. The
216-node quadrature is dominated by the target-dependent edge and solid-angle
primitives, not by the frame construction. It is pure code motion, bit-
identical and free, so it stays, but the prediction was wrong and the
measurement is what settled it. The prepared-body hoisting in the dense
constructor is similarly worth 0-4% on its own.

### Accepted: `refactor(operators): share the exact operator classification`

The exact-equivalence key, its first-seen classification, its sampling gate
and the lowest-index failure report moved from the anonymous namespace of
`src/operators/p2p.cpp` to `src/operators/exact_operator_reuse.hpp`, so the
near-field and dense builders cannot drift apart on what "the same operator"
means. The header owns the equivalence and owns no mathematics and no storage
layout. The gate's sample size, reuse factor and class cap became parameters,
because the right values depend on the caller's per-pair cost. The
endpoint-operator copy in `src/fmm/plan_preparation.cpp` is keyed on a
leaf-relative offset and was deliberately left alone: folding it in would be
an abstraction refactor beyond this phase.

### Rejected

**Pinned staging for the CUDA upload.** The upload runs from pageable host
memory at about 14.5 GB/s; pinned staging would reach roughly 25 GB/s. It is
not worth it. Measured below, the upload is 1% or less of setup for every
finite geometry, where setup actually costs something, and 38% only for
point-to-point, whose entire setup is 17 ms. Buying a few milliseconds there
would cost either a 96 MB pinned staging buffer or a chunked upload, against
Phase M's instruction not to grow an already-large plan's peak memory.

**Reciprocity between the triangular halves.** Measured, not assumed, and
rejected on the measurement. Exact-class reuse keys on the displacement
vector, so `d` and `-d` are separate classes; if the pair tensor were even in
the displacement, half of every classified build would be redundant. Sampling
400 random displacements per geometry and comparing `T(-d)` against `T(d)`:

| pair | bitwise equal | equal to 1e-12 | worst relative difference |
|---|---|---|---|
| point -> point | 400/400 | 400/400 | 0 |
| prism -> point | 12/400 | 400/400 | 9.9e-15 |
| prism -> prism, equal records | 1/400 | 158/400 | 7.2e-11 |
| prism -> prism, unequal records | 1/400 | 232/400 | 8.6e-11 |
| tetra -> tetra, equal records | 0/400 | 27/400 | 2.8e-9 |
| tetra -> point | 0/400 | 0/400 | 3.2 |
| prism -> tetra | 0/400 | 0/400 | 8.7e-1 |

Three different answers, and none of them helps. The symmetry is exact and
bitwise only for point-to-point — precisely the case that deliberately never
classifies, because its arithmetic is cheaper than a key lookup. For the
centrosymmetric prism geometries it holds mathematically but *not* bitwise:
the `long double` corner sums round differently under `d -> -d`, so
exploiting it would trade this phase's bitwise contract for a factor of two on
a build that exact reuse has already reduced 67x. And for any tetrahedron it
does not hold at all — a tetrahedron is not centrosymmetric, so its field at
`-d` is a genuinely different tensor, by a relative 3.2 for the point case.
The one place a halving would still have been worth real time, the
prism-tetrahedron combinations, is the place the symmetry does not exist.

**Splitting build from scatter when classification is abandoned.** It would
need a complete `PairTensor` array, forty-eight bytes a pair, larger than the
FP32 matrices themselves. Those rows stay a fused loop and the benchmark says
so rather than reporting a zero.

**Changing the prism `long double` policy.** Out of scope and unnecessary: for
the same reason Phase 3C gave, the exact prism arithmetic now runs once per
distinct operator on regular geometry, and on irregular geometry the phase's
rule against buying construction speed with a numerical-contract change still
holds.

### CPU results

Cold construction, eight P-cores, baseline binary against final binary in one
session. Every row is bit-identical between the two.

Every figure below was re-measured on a quiet machine after the validation
trees finished, and agrees with the first sweep to within a percent -- except
the first row, whose *baseline* varies between 3.07 s and 3.72 s from run to
run while the final build stays at 0.0457-0.0458 s. Its speedup is therefore
67-82x depending on the baseline sample; the conservative end is quoted.

| case | baseline | final | speedup |
|---|---|---|---|
| prism->prism lattice 512^2 FP32 | 3.069 s | 0.0458 s | 67.0x |
| prism->prism lattice 512^2 FP64 | 3.071 s | 0.0458 s | 67.0x |
| prism->prism asymmetric 1024x512 FP32 | 6.066 s | 0.0715 s | 84.8x |
| tetra->prism lattice 256^2 FP32 | 7.423 s | 0.1283 s | 57.9x |
| prism->tetra lattice 256^2 FP32 | 7.791 s | 0.1586 s | 49.1x |
| tetra->tetra lattice 256^2 FP32 | 3.296 s | 0.0873 s | 37.8x |
| tetra->tetra lattice 256^2 FP64 | 3.298 s | 0.0876 s | 37.6x |
| prism->point lattice 1024^2 FP32 | 0.3378 s | 0.0245 s | 13.8x |
| point->prism lattice 512^2 FP32 | 0.0816 s | 0.0065 s | 12.5x |
| tetra->point lattice 512^2 FP32 | 0.0677 s | 0.0088 s | 7.7x |
| point->tetra lattice 512^2 FP32 | 0.0676 s | 0.0094 s | 7.2x |
| point->point lattice 1024^2 FP32 | 0.0060 s | 0.0060 s | 1.0x |
| point->point lattice 1024^2 FP64 | 0.0123 s | 0.0121 s | 1.0x |

Intermediate redundancy, which is where real discretisations sit:

| case | baseline | final | speedup |
|---|---|---|---|
| prism->tetra refined 384x192 FP64, no identity map | 8.448 s | 0.6353 s | 13.3x |
| prism->prism refined 256^2 FP32 | 0.7753 s | 0.0658 s | 11.8x |
| prism->tetra refined 192^2 FP32 | 4.566 s | 0.3798 s | 12.0x |
| tetra->tetra refined 192^2 FP32 | 1.800 s | 0.1755 s | 10.3x |
| tetra->prism refined-anisotropic 192^2 FP64 | 5.104 s | 0.8910 s | 5.7x |
| prism->tetra refined-anisotropic 192^2 FP32 | 5.111 s | 0.9055 s | 5.6x |
| prism->prism refined-anisotropic 256^2 FP32 | 0.7620 s | 0.1375 s | 5.5x |
| tetra->tetra refined-anisotropic 192^2 FP32 | 1.579 s | 0.2935 s | 5.4x |
| prism->point refined-anisotropic 384^2 FP32 | 0.0479 s | 0.0126 s | 3.8x |

No reuse available, where the gate abandons and only the schedule and the
prepared geometry remain:

| case | baseline | final | speedup |
|---|---|---|---|
| prism->prism lattice-irregular 512^2 FP32 | 3.032 s | 3.035 s | 1.0x |
| prism->prism random 512^2 FP32 | 3.043 s | 3.041 s | 1.0x |
| tetra->tetra lattice-irregular 256^2 FP32 | 3.429 s | 3.118 s | 1.1x |
| prism->tetra lattice-irregular 256^2 FP32 | 7.165 s | 6.879 s | 1.0x |

### CUDA results

`CudaDenseDirectPlan` delegates its exact tensors to a host plan, so it
inherits the host speedup exactly: prism->prism lattice 512^2 FP32 falls from
3.070 s to 0.0467 s (65.8x), tetra->tetra lattice 256^2 FP32 from 3.302 s to
0.0882 s (37.4x), prism->point lattice 1024^2 FP32 from 0.3353 s to 0.0266 s
(12.6x), and point->point is unchanged. Setup decomposition at the final HEAD:

| configuration | total | host build | context | allocation | H2D | device |
|---|---|---|---|---|---|---|
| point->point lattice 2048^2 FP32 | 0.0173 s | 0.0098 s | 0.0003 s | 0.0006 s | 0.0066 s | 96.0 MiB |
| point->point lattice 2048^2 FP64 | 0.0790 s | 0.0590 s | 0.0002 s | 0.0005 s | 0.0135 s | 192.1 MiB |
| prism->prism lattice 1024^2 FP32 | 0.1124 s | 0.1104 s | 0.0002 s | 0.0005 s | 0.0013 s | 24.0 MiB |
| prism->prism lattice 1024^2 FP64 | 0.1167 s | 0.1129 s | 0.0002 s | 0.0006 s | 0.0031 s | 48.0 MiB |
| tetra->tetra lattice 512^2 FP32 | 0.1800 s | 0.1790 s | 0.0001 s | 0.0005 s | 0.0003 s | 6.0 MiB |
| tetra->tetra lattice 512^2 FP64 | 0.1803 s | 0.1791 s | 0.0001 s | 0.0004 s | 0.0006 s | 12.0 MiB |

Phase J's instruction was to optimise host construction first and then measure
the upload, and the table confirms that was the right order: context creation
and device allocation are a fraction of a millisecond throughout, and the
upload only becomes visible for point-to-point, whose host build is trivial.
No CUDA evaluation kernel changed. The only CUDA construction change is that
the upload's stream synchronisation moved inside each precision branch so it
can be timed; it still executes exactly once.

### Thread scaling

Cold construction on one, two, four and eight pinned P-cores, final HEAD:

| builder | 1 | 2 | 4 | 8 | 8-thread |
|---|---|---|---|---|---|
| prism->prism lattice-irregular 512^2, fused | 24.303 s | 12.111 s | 6.062 s | 3.036 s | 8.00x |
| tetra->tetra lattice-irregular 256^2, fused | 24.472 s | 12.193 s | 6.094 s | 3.113 s | 7.86x |
| prism->tetra refined-anisotropic 192^2, classified | 7.124 s | 3.547 s | 1.774 s | 0.901 s | 7.90x |
| prism->prism lattice 512^2, classified | 0.3204 s | 0.1635 s | 0.0855 s | 0.0460 s | 6.97x |
| point->point random 2048^2, fused and cheap | 0.0609 s | 0.0420 s | 0.0323 s | 0.0282 s | 2.16x |

The two mechanisms divide the work as intended. Where reuse cannot help, the
parallel build carries the case and scales essentially linearly. Where reuse
has already collapsed the build, what is left is the serial classification
pass, which caps the classified lattice row at 7.0x — on a build that is
already 67x faster, so the scaling loss costs nothing absolute. The cheap
point row is memory-bandwidth bound: it writes 96 MB of matrices and there is
almost no arithmetic to overlap with that, which is a property of the problem
rather than of the schedule.

### Memory

Retained and transient bytes at the final HEAD. Transient is the class map
plus the distinct built tensors plus the prepared bodies.

| configuration | matrices | class map | tensors | transient | transient/matrices |
|---|---|---|---|---|---|
| prism->prism lattice 2048^2 FP32 | 96.0 MB | 16.06 MB | 0.69 MB | 16.75 MB | 0.17 |
| prism->prism lattice 2048^2 FP64 | 192.0 MB | 16.06 MB | 0.69 MB | 16.75 MB | 0.09 |
| prism->prism refined-anisotropic 1024^2 FP32 | 24.0 MB | 4.33 MB | 3.99 MB | 8.32 MB | 0.35 |
| tetra->tetra lattice 1024^2 FP32 | 24.0 MB | 4.03 MB | 0.33 MB | 4.36 MB | 0.18 |
| prism->prism lattice-irregular 1024^2 FP32 | 24.0 MB | 0 | 0 | 0 | 0.00 |
| point->point lattice 4096^2 FP32 | 384.0 MB | 0 | 0 | 0 | 0.00 |
| point->point lattice 4096^2 FP64 | 768.0 MB | 0 | 0 | 0 | 0.00 |

The byte budget holds with margin: the worst measured ratio is 0.35 against a
cap of 0.5, and the unclassified paths hold nothing transient at all. Peak
resident set tracks the matrices plus a constant ~215 MB of loaded vendor
runtime in the combined CPU/CUDA/oneMKL benchmark binary, so the meaningful
quantity is the delta rather than the absolute figure.

### Repeated-evaluation regression gate

Repeated evaluation must not pay for setup. Baseline against final, same
session, for point->point, prism->point, prism->prism and tetra->tetra on
portable CPU, oneMKL and CUDA in both precisions: every ratio lies in
0.94-1.02x, which is run-to-run variance on this machine.

One row first read 0.40x, which would have been a 2.5x evaluation speedup that
no change in this phase could explain. Repeating it with more warmups showed
baseline and final agreeing at about 0.13 ms; the original figure was a
first-call oneMKL warm-up inside that one process. It is recorded here because
the correct response to an implausibly good number is to re-measure it, not to
report it.

### Correctness

The six matrices are compared **bit for bit** against the pinned starting-SHA
build, over 150 configurations: nine geometry combinations x seven workloads x
FP32 and FP64, plus asymmetric counts, plans with and without an identity map,
and both precisions of each. All identical. Configurations that throw are
compared on their exception message rather than skipped, so the failure paths
are held to the same standard as the successful ones.

`tests/test_dense_direct_exact_reuse.cpp` pins the contract at the lowest
layer: every geometry combination reproduces the per-pair geometry function
bitwise; a one-ULP prism half-size, displacement or tetrahedron vertex still
reaches the tensor so near-equal inputs are never merged; a mapped point self
interaction is omitted while a mapped finite one keeps its physical tensor;
two bodies at one place with different identities are an ordinary coincident
interaction; a partial identity map omits exactly the pairs it names;
asymmetric counts with no identity map behave as the general case; and
repeated construction is bit-identical.

Writing those tests found that an unmapped point target may not sit on a
source — a point source has no self field, so that is a singular pair rather
than an omitted one, and the constructor correctly rejects it. The test says
so explicitly rather than encoding the mistake.

### Tests, sanitizers and validation

Four fresh pinned trees at the accepted HEAD, all with g++ 15.3.0 as the C++
and `nvcc` host compiler:

| tree | result |
|---|---|
| portable CPU (`dev`) | 242/242 pass, 4 skipped for the absent optional backends |
| CUDA (`cuda`) | 242/242 pass, 1 skipped (the oneMKL comparison) |
| oneMKL without CUDA | 242/242 pass, 3 skipped (the CUDA-only tests) |
| CUDA plus oneMKL (`notebooks`) | 242/242 pass, nothing skipped |

Python suite against the just-built portable module (`PYTHONPATH=build-dev`,
no install): 140 passed, 8 skipped. `git diff --check` is clean across the
whole series.

`compute-sanitizer` over the CUDA dense plan, because its setup path changed
(the upload's stream synchronisation moved inside each precision branch so it
could be timed; it still executes exactly once): `memcheck` clean on eight
configurations covering both precisions, the classified and unclassified
builds, and an asymmetric plan with no identity map; `racecheck` clean on
setup plus evaluation in both precisions; `initcheck` and `synccheck` clean.

Compatibility. `include/cdfmm/`, `python/` and `fortran/` are untouched by the
entire series, and the fourteen `extern "C"` ABI symbols are identical between
the baseline and final shared libraries. The library gains two exported
symbols, `cdfmm::detail::prepare_tetrahedron_point_field` and
`cdfmm::detail::tetrahedron_point_tensor_prepared`; both are internal C++
`detail` functions that appear only because the library has no visibility map,
and `src/AGENTS.md` places files under `src/` outside any downstream
compatibility obligation. The C ABI itself is unchanged.

Cache format and keys. Nothing under `src/cache/` was touched, but
`tetrahedron.cpp` is shared with the FMM near field, so the persisted plans
were compared rather than assumed: the geometry and universal cache files are
**byte identical** between the baseline and final trees for all nine FMM
geometry configurations, including irregular tetrahedra and both far-field
models, and for the cache-initialisation benchmark on the portable and oneMKL
backends. A geometry plan serialises the canonical P2P blocks, the P2M plans
and the L2P evaluators, so identical files mean the shared change altered no
FMM operator and no cache content or key.

### Remaining bottlenecks

1. **Irregular and random geometry.** 3-7 s at the sizes measured, with no
   exact duplicate to find, because every pair of an all-to-all plan holds a
   unique pair of records. The build is parallel and scales 7.9-8.0x on eight
   cores, so the only levers left are in the exact tensor mathematics itself —
   principally the 216-node far-separation quadrature and the recursive
   triangle-pair integrals.
2. **Caller-dependent bitwise agreement.** Dense reuse finds less on a lattice
   whose spacing is not exactly representable, purely through rounding. A
   canonical-grid normalisation like the FMM path's would recover it, but that
   changes what a dense plan computes from what the caller supplied and is not
   a construction optimisation; it needs its own decision.
3. **The serial classification pass** caps classified builds at about 7x on
   eight cores. It is a single pass over the pairs building a hash table, and
   it is only visible once the build it enables has become 40-80x faster.
4. **Matrix allocation** is 62% of a point-to-point plan's construction: the
   six `resize` calls zero 96-768 MB that the build then overwrites entirely.
   A non-zeroing allocation would remove it, but `std::vector` cannot express
   that without a custom allocator.
5. **Point plans leave 77x redundancy unexploited**, correctly, because the
   arithmetic is cheaper than the lookup. A cheaper key — the displacement
   alone, three words, when the whole plan shares one record — might change
   that balance, and was not attempted.

## Final cross-backend integration and production policy (Phase 3D)

Starting HEAD `49b5fe3` ("docs(perf): record the Phase 3C.5 validation
evidence"), the tip of `phase3c5-dense-construction`. An earlier note here
said `refactor/architecture-v0.2` still pointed at `a2af367` when this phase
began. That described this checkout's local ref, not the integration branch
itself: the remote branch had already been advanced through Phase 3C/3C.5 to
`49b5fe3`, and the local ref caught up by fast-forward pull. `a2af367` is an
ancestor of `49b5fe3` in either reading, so nothing had diverged and no
history was rewritten. Work happened on `phase3d-final-integration`, based
exactly on `49b5fe3`.

This phase reconciles the measured history of Phase 3 with what the code
actually does. It is an internal regression and policy-validation matrix, not
the Article1 campaign: no external framework comparison, no publication
figures, and no Article1 data were touched.

### Environment

Intel i9-14900KF, 8 P-cores pinned for every CPU measurement
(`OMP_NUM_THREADS=8`, `OMP_PLACES={0},{2},{4},{6},{8},{10},{12},{14}`,
`OMP_PROC_BIND=close`); RTX 5090 (sm_120, 32 GB, driver 595.84); conda `cdfmm`
environment with g++ 15.3.0 as C++ *and* nvcc host compiler, nvcc 13.3.73,
oneMKL 2026.1, Python 3.11.16. The environment's `CC=icx`, `CXX=icpx` and
`NVCC_PREPEND_FLAGS=-ccbin=icpx` were unset for every tree, and Python was not
installed into the conda environment (`CDFMM_INSTALL_PYTHON_TO_ENV=OFF`,
`PYTHONPATH` instead), so the user's environment is unchanged.

Measurements come from one `build-bench-all` tree (CPU + oneMKL + CUDA in one
binary), so every comparison below is same-session and same-binary.

### Integration hygiene

**The exact-reuse index boundary.** `ExactOperatorClasses` stores pair indices
and class numbers as `std::uint32_t`, and `class_of_pair` holds one entry per
pair. A dense all-to-all plan exceeds that range at roughly 65536 bodies per
side (`pair_count = ns * nt`), and
`result.representative.push_back(static_cast<std::uint32_t>(index))` narrowed
silently. Widening both vectors would have doubled the largest transient
allocation of a big build, so the compact words were kept and classification
is abandoned instead when the indices do not fit -- which is already the
established outcome of the sample and class-count gates, and costs only the
reuse, never correctness. The guard runs before the per-pair allocation and
before the first key, so its unit test costs no memory.
`classify_endpoint_operators` in `src/fmm/plan_preparation.cpp` is a separate
helper with the same pattern on leaf/target counts (N-bounded, not N^2) and
takes the same guard. The duplication between the two classifiers is recorded
for Phase 4 rather than merged here.

The dense reuse test's shared comment claimed every section was large enough
to classify; production deliberately never classifies point-to-point plans,
because a point pair costs about 4 ns against about 260 ns for the cheapest
finite pair. That section exercises the unclassified loop, and the same
expectation is correct for both paths because they agree bit for bit. The
comment now says so.

**An explicit FP32 BSR(3) request under a lowered budget (correctness).**
`docs/static-p2p.md` states that `p2p_packing` "takes precedence over
`spatial_layout`, `use_reduced_symmetry_p2p` and the BSR memory budget". The
FP32 path did not honour the last clause. `cuda_p2p_bsr_max_bytes` gates a
speculative BSR prebuild in `quantise_static_plan_to_float()`, while
`build_cuda_p2p_plan()` consumed `p2p_bsr_plan_float_` unconditionally once
the policy resolved to BSR(3). With a budget below the plan's estimate the
prebuild was skipped and the executor received a default-constructed plan:
construction reported `CudaBsr3` and the evaluation then failed with "CUDA
FP32 P2P dimensions are inconsistent". FP32 is the default precision, so only
the budget had to be lowered to reach it; the FP64 branch always rebuilt and
was unaffected. The FP32 branch now builds on demand exactly as FP64 does.

The pre-existing budget test asserted the resolved packing *name*, which a
default-constructed plan satisfies. The new case evaluates the field at a
single level, where every pair is near field, and requires the two budgets to
agree on a field verified to be non-zero. It fails without the change and
passes with it.

**A stale example.** `examples/simple_notebooks/ReducedSymmetryP2P.ipynb` set
`options.use_cuboid_p2m` / `use_cuboid_l2p`; neither exists on
`UniformFmmOptions` nor in the bindings, which declare no dynamic attributes,
so the notebook raised `AttributeError` and could not run. No regression test
executes it. Both flags selected a point far-field model, so they became
`far_field_source_model` / `far_field_target_model`. The notebook's other
pre-Phase-3A assumptions are recorded for Phase 4.

### Phase A -- production policy inventory

Every row was read from the source rather than from the Phase-3 record.

| Decision | Owner |
|---|---|
| execution backend | `execution_setup.cpp:246-257` |
| P2P packing (CPU **and** CUDA) | `cuda_policy::resolve_cuda_execution_policy`, `execution_policy.cpp:169-275` |
| CPU packing resolution | `resolve_cpu_p2p_packing`, `execution_setup.cpp:968-985` |
| CPU point rule | `selects_point_geometry_p2p`, `execution_setup.cpp:948-966` |
| point P2M/L2P execution | `resolve_point_expansion_execution`, `execution_setup.cpp:854-910` |
| M2L implementation | `static_matrix_backend`, applied in `far_field.cpp:48-89` |
| dictionary token guard | `execution_setup.cpp:634-638`, threshold `execution_policy.cpp:117` |
| stream priority | `execution_policy.cpp:69-82`, `backend/cuda/common/stream.hpp` |

Despite its name, `resolve_cuda_execution_policy` runs for *every* backend and
also decides whether the CPU builds the signed dictionary. That is a naming
and ownership wart, not a duplicated rule; it is recorded for Phase 4.

Resolver precedence: explicit packing -> explicit reduced symmetry -> **the
FP32 CUDA point rule** -> the layout dictionary -> the leaf-block default. The
FP32 point rule sits above the layout hint, so an FP32 CUDA point lattice
deliberately ignores `RegularGrid`.

| Situation | Auto | Override | Fallback |
|---|---|---|---|
| CpuStatic point->point, General | `PointGeometry` | `p2p_packing`, `use_reduced_symmetry_p2p` | -- |
| CpuStatic point->point, RegularGrid + fixed identity | `TensorDictionary` | as above | token > 2 B -> `PointGeometry` |
| CpuStatic point->point, RegularGrid, no fixed identity | `PointGeometry` | as above | -- |
| CpuStatic finite, General | `ParticleRowSoa` | `p2p_packing` | -- |
| CpuStatic finite, RegularGrid | `TensorDictionary` | `p2p_packing` | token > 2 B -> `ParticleRowSoa` |
| CUDA FP32 point->point, any layout | `PointGeometry` | `p2p_packing`, `use_reduced_symmetry_p2p` | -- |
| CUDA FP64 point->point, General | `LeafBlock` | as above | -- |
| CUDA FP64 point->point, RegularGrid + fixed identity | `TensorDictionary` | as above | token > 2 B -> General rules |
| CUDA finite, General | `LeafBlock` | `p2p_packing` | -- |
| CUDA finite, RegularGrid | `TensorDictionary` | `p2p_packing` | token > 2 B -> `LeafBlock` |

Point P2M/L2P (`procedural_available` needs the spherical basis, a
non-`CpuReference` backend and order 1-10): the CPU hierarchy -- which
includes `CudaPartial`, whose hierarchy *is* the CPU -- and `CudaFull` FP32
resolve to procedural; `CudaFull` FP64 resolves to precomputed. Precision
enters at one line. Finite far-field models are unconditionally precomputed,
and an explicit `Procedural` request no stage can honour throws.

M2L: `StaticMatrixBackend` has **no** `Auto`; the default is `Portable` and
oneMKL runs only on an explicit request. `ExecutionBackend::Auto` resolves to
`CpuStatic`; **neither CUDA backend is ever auto-selected**.

Reachability: `CudaBsr3` and `CanonicalAos` are explicit-only.
`cuda_p2p_bsr_max_bytes` bounds the speculative FP32 prebuild only.

Mathematical/model choices (they change the field): expansion basis and order,
precision, near/far-field models, periodicity, tree depth and root bounds.
Execution-representation choices (they must not, beyond rounding): P2P
packing, dictionary executor, point expansion execution, `StaticMatrixBackend`,
`ExecutionBackend`, the `SpatialLayout` hint, M2L pairs per thread,
translation lane groups, stream priority, and the BSR byte budget.
### Phase B -- internal regression matrix

`benchmarks/run_phase3d_regression.py` (suite `all`, 158 cases, 0 failures),
`--evaluations 20 --warmups 3 --samples 5 --threads 8`. Every automatic policy
resolved exactly as the Phase-3 record intends, on every backend and both
precisions:

| Workload | CPU | CudaFull FP32 | CudaFull FP64 | CudaPartial |
|---|---|---|---|---|
| A random points | `point-geometry`, procedural | `point-geometry`, procedural | `leaf-block`, precomputed | `point-geometry`/`leaf-block`, **procedural** (CPU hierarchy) |
| B/C point lattice, General | `point-geometry` | `point-geometry` | `leaf-block` | as CudaFull's packing |
| B/C point lattice, `RegularGrid` | `tensor-dictionary` | `point-geometry` (hint deliberately ignored) | `tensor-dictionary` | as CudaFull's packing |
| D/E finite lattice, General | `particle-row-soa` | `leaf-block` | `leaf-block` | `leaf-block` |
| D/E finite lattice, `RegularGrid` | `tensor-dictionary` | `tensor-dictionary` | `tensor-dictionary` | `tensor-dictionary` |
| F irregular finite | `particle-row-soa` | `leaf-block` | -- | `leaf-block` |
| D/E exact far field | precomputed P2M/L2P | -- | -- | -- |

The derived dictionaries reproduce the variant counts recorded in the code:
248 distinct prism tensors and 187 tetrahedron tensors at FP32, one-byte
tokens, 28-33x smaller than the canonical operator. The point lattices reach
172 variants at 8 per leaf (one-byte tokens, 29-55x) and 1688 at 64 per leaf
(two-byte tokens, 20-38x). Irregular geometry builds no dictionary at all and
falls back as designed.

Evaluation medians [ms], automatic policy, 8 threads:

| Case | CPU portable | CPU oneMKL | CudaFull | CudaPartial |
|---|---:|---:|---:|---:|
| A random S 10k p4 FP32 | 1.234 | 1.167 | 0.115 | 0.203 |
| A random M 50k p6 FP32 | 11.787 | 15.655 | 0.448 | 1.091 |
| A random M 50k p6 FP64 | 16.002 | 30.180 | 2.762 | 3.213 |
| B lattice 8/leaf FP32 | 9.798 | 13.728 | 0.361 | 0.942 |
| C lattice 64/leaf FP32 | 8.598 | 8.100 | 0.210 | 0.377 |
| D prism N4096 FP32 | 1.029 | 0.823 | 0.127 | 0.206 |
| E tetrahedron N4096 FP32 | 1.049 | 0.825 | 0.128 | 0.208 |
| F irregular prism FP32 | 1.085 | 0.900 | 0.133 | 0.208 |

### Phase C -- automatic policy against forced alternatives

A policy comparison is decided on the phase the policy governs. Total
evaluation time hides a P2M/L2P rule inside an M2L-dominated evaluation: the
FP64 `CudaFull` expansion rows differ by 2 % end to end but by 1.39x on
P2M+L2P, which is the number that matters. Both are given below.

**Point P2P packing.** P2P phase [ms], then evaluation, then retained bytes.

| Case | Auto | Alternative | P2P ratio | eval ratio | memory |
|---|---|---|---:|---:|---|
| CPU random M FP32 | `point-geometry` 3.044 | `particle-row-soa` 8.300 | 2.73x slower | 1.46x | 1481 MB vs 29 MB host |
| CPU random M FP64 | `point-geometry` 3.025 | `particle-row-soa` 15.249 | 5.04x slower | 1.85x | 2727 MB vs 36 MB host |
| CUDA random M FP32 | `point-geometry` 0.0576 | `leaf-block` 0.4835 | 8.39x slower | 1.48x | 421 MB vs 30 MB device |
| CUDA random M FP64 | `leaf-block` 2.411 | `point-geometry` 3.146 | 1.30x slower | 1.27x | but 192 MB vs 972 MB device |
| CUDA lattice 8/leaf FP32 | `point-geometry` 0.0306 | `tensor-dictionary` 0.0671 | 2.19x slower | 1.03x | 33 MB vs 28 MB device |
| CUDA lattice 8/leaf FP64 | `leaf-block` 0.2438 | hint `tensor-dictionary` 0.1841 | **1.32x faster** | 0.99x | 123 MB vs 416 MB device |

Every automatic choice is confirmed. The FP32/FP64 split on CUDA point pairs
is still exactly right: recomputation wins by 8.4x on the FP32 P2P phase and
loses by 1.30x in FP64. The FP32 rule's precedence over the layout hint is
also still right -- the dictionary is 2.19x slower than recomputation at 8
points per leaf.

**Point P2M/L2P.** P2M+L2P phase [ms]:

| Case | procedural | precomputed | Auto picks | ratio |
|---|---:|---:|---|---:|
| CPU M FP32 | 0.195 | 0.958 | procedural | 4.92x |
| CPU M FP64 | 0.367 | 2.103 | procedural | 5.73x |
| CudaFull M FP32 | 0.0185 | 0.0730 | procedural | 3.95x |
| CudaFull M FP64 | 0.214 | 0.154 | **precomputed** | 1.39x |

The FP64 `CudaFull` P2M alone is 1.86x slower procedurally (0.1105 against
0.0595 ms), reproducing the "P2M 1.9x slower" of Phase 3B.5 almost exactly.
Every point-expansion rule is confirmed on the phase it governs.

**CPU M2L, portable against oneMKL** (evaluation medians [ms]):

| Case | Portable | oneMKL | winner |
|---|---:|---:|---|
| point M 50k p6 FP32 | 11.890 | 15.689 | Portable 1.32x |
| point M 50k p6 FP64 | 16.046 | 30.102 | Portable 1.88x |
| prism N4096 p6 FP32 | 1.048 | 0.824 | **oneMKL 1.27x** |
| prism N4096 p6 FP64 | 1.667 | 2.092 | Portable 1.25x |

**CudaFull against CudaPartial** (evaluation medians [ms]):

| Case | CudaFull | CudaPartial | ratio |
|---|---:|---:|---:|
| random M FP32 | 0.447 | 1.084 | 2.42x |
| random M FP64 | 2.860 | 3.285 | 1.15x |
| 128 points per leaf FP32 | 0.656 | 0.888 | 1.35x |

The 1-2 % crossover in the hybrid's favour at 128-160 points per leaf,
recorded before the procedural point P2P landed, is gone: `CudaFull` is now
1.35x ahead there. `CudaPartial` does retain far less host memory at high
occupancy (8.4 MB against 158.6 MB), which is its remaining reason to exist
besides being an explicit backend.
### Phase D -- accepted and rejected changes

**Accepted.** All four are correctness or clarity; no automatic performance
policy changed, because no measurement asked for one.

1. `fix(operators)` -- the exact-reuse index guard and the corrected dense
   reuse test comment (above).
2. `fix(fmm)` -- an explicit FP32 BSR(3) request is honoured under a lowered
   budget (above). Before: construction reported `CudaBsr3` and the evaluation
   threw "CUDA FP32 P2P dimensions are inconsistent". After: it evaluates, and
   both budgets agree on the field.
3. `fix(examples)` -- the P2P notebook runs again.
4. `docs(cuda)` and `docs(backends)` -- the hybrid's unconditional M2L stream
   priority and the policy's true inputs are recorded where they are read.

**Rejected, with the measurement that rejected each.**

- *Making the CPU lattice dictionary the default for point plans.* At 8 points
  per leaf the dictionary's P2P phase is 2.2x faster than recomputation
  (0.562 against 1.256 ms FP32) and the evaluation 8 % faster, but the plan
  retains 583 MB against 28 MB, because `PointGeometry` keeps no pair tensors
  at all while the dictionary needs the canonical operator built first. A 4-8 %
  evaluation gain does not buy a 20-31x memory increase by default. The hint
  remains the opt-in way to ask for it.
- *Making oneMKL the default M2L.* It wins exactly one of four rows -- the
  small finite FP32 lattice, by 1.27x -- and loses the point workloads by
  1.32x (FP32) and 1.88x (FP64). Portable stays the default, oneMKL stays
  supported and explicit, and the crossover is documented rather than turned
  into a threshold.
- *Removing `CudaPartial`.* `CudaFull` is 1.15-2.42x faster on every measured
  row and the old 128-per-leaf crossover is gone, but the hybrid retains 8.4 MB
  against 158.6 MB of host bytes at 128 points per leaf. It stays a valid
  explicit backend.
- *Switching CUDA FP64 point pairs to `PointGeometry`.* 1.30x slower on the
  P2P phase. It does retain 14x less host and 5x less device memory, so it
  stays available explicitly and that trade is now documented.
- *Switching FP64 `CudaFull` point expansions to procedural.* 1.39x slower on
  P2M+L2P, P2M alone 1.86x. The 2 % end-to-end difference that first suggested
  otherwise is noise inside an M2L-dominated evaluation.
- *Making an explicit dictionary request consult the occupancy calibration.*
  An explicit `p2p_packing = TensorDictionary` with neither executor flag gets
  the source-warp kernel and is 2.59x slower on the P2P phase than the same
  packing chosen by the hint at 8 targets per leaf (0.477 against 0.184 ms,
  FP64); adding `--dictionary-power2-microtiles` recovers it exactly
  (0.183 ms), which confirms the executor is the only difference. Not changed
  for two reasons: the source-warp default of `use_reduced_symmetry_p2p` was a
  deliberate recorded decision of the Phase-3A closure, and "neither flag set"
  is the *only* way to request source-warp, so calibrating it would remove the
  ability to select that kernel below 72 targets per leaf. Documented instead.
- *Removing the dead `use_cuboid_p2m_` / `use_cuboid_l2p_` members, the unused
  three-argument `far_field_stream_priority` overload, and the write-only
  `periodic` / `bsr_*` policy inputs.* All confirmed unused, all Phase-4
  pruning; the members sit in a public header, so removing them changes the
  class layout and belongs in a deliberate pruning step.

### Phase E -- cold and warm startup

`benchmarks/run_phase3d_startup.py`, portable CPU, FP32, per-stage timings
[ms]:

| Stage | point 32768 d4 | | | prism 32768 d4 lattice | | |
|---|---:|---:|---:|---:|---:|---:|
| | cold | +write | warm | cold | +write | warm |
| total | 2355 | 2650 | **373** | 2652 | 2918 | **419** |
| universal operator build | 1337 | 1338 | 0 | 1336 | 1336 | 0 |
| canonical near field | 587 | 579 | 0 | 660 | 660 | 0 |
| derived P2P packing | 0 | 0 | **0** | 167 | 166 | **0** |
| FP32 precision conversion | 221 | 222 | 80 | 315 | 313 | **167** |
| far-field packing | 31 | 30 | 25 | 13 | 12 | 12 |
| geometry cache load | 0 | 0 | 152 | 0 | 0 | 133 |
| geometry cache write | 0 | 287 | 0 | 0 | 249 | 0 |

Two of the four Phase-3C startup observations change.

*The universal operator bank is confirmed as the dominant cold cost and is
fully cached.* The cold construction matrix measures it at 0.083 s (p = 4),
1.34 s (p = 6) and 11.82 s (p = 8), identical to within noise across point,
prism and tetrahedron geometries and across all four backends -- it is
geometry-independent, as documented -- and it is 88-99 % of a cold build for
every geometry except irregular tetrahedra. A warm hit removes it entirely.

*Derived packing on a warm cache no longer reproduces.* Phase 3C recorded
0.469 s of a 0.978 s warm setup at 32,768 bodies as the largest warm cost and
"the clearest next lead". At that same size the warm setup is now 0.419 s and
its derived packing is 0.00 ms, for the point plan and the finite lattice
alike. The timer is still charged on a warm hit -- `precision_conversion`
beside it is -- and the stages account for the total (167 + 133 + 12 ms
against a 321 ms static plan), so the zero is real rather than an uncharged
timer. What replaced it is the FP32 precision conversion, 167 ms or 40 % of
the warm setup.

*The canonical near field a point plan never reads is unchanged*: 587 ms of a
2355 ms cold build at 32,768 points, still built for a plan whose executor is
`PointGeometry`. Skipping it still requires a cache-identity change, which
Phase 3D deliberately does not make.

### Phase F -- remaining bottlenecks, and what was left alone

Ranked by absolute user-facing time rather than by percentage.

1. **The universal operator bank**, 11.8 s at p = 8 and 1.34 s at p = 6 on a
   truly cold build. Geometry-independent and removed completely by the
   universal cache, so it is paid once per (basis, order, precision) per
   machine. Shrinking it is M2L operator mathematics -- symmetry between the
   316 classes -- and outside this phase.
2. **Irregular tetrahedra**, 14.8 s of near-field construction at 4096 bodies
   and p = 6, identical on all four backends because it is shared host work.
   Every pair holds a unique record, so there is no duplicate to find; the
   lever is the tetrahedron pair mathematics.
3. **FP32 precision conversion on a warm reload**, 167 ms of a 419 ms warm
   setup at 32,768 finite bodies. Removing it means persisting the FP32 plan,
   which is a cache-format change. Not taken: Phase 3D's stated default is to
   preserve the cache format and keys, and 167 ms once per process against a
   1 ms evaluation does not justify a format migration. Recorded for post-v0.2.
4. **The canonical near field a procedural point plan discards**, 587 ms of a
   2355 ms cold build. Unchanged for the same cache-identity reason Phase 3C
   recorded.

Nothing here was optimised, and that is the finding: the three items Phase 3C
left as leads are now either cached away, gone, or blocked behind a cache
change that this phase is not authorised to make.

### Phase G -- CUDA execution closure

The resolved policy was captured for representative point and finite rows on
both CUDA backends. `CudaFull` FP32 points resolve to `PointGeometry` with
procedural expansions; FP64 points to `LeafBlock` with precomputed
expansions; finite geometry to `LeafBlock` on a `General` layout and to the
signed dictionary under the hint, with the occupancy-calibrated executor.
`CudaPartial` resolves the same packings and keeps the CPU hierarchy, so its
expansions are procedural in both precisions.

The stream-priority rule survives the much faster point P2P unchanged. It is
still conditional on `CudaFull` and unconditional on the hybrid, and the
reason is now recorded at both call sites. The rule's estimate uses the
resolved packing's cost per pair, so the eightfold cheaper `PointGeometry`
kernel feeds into it correctly rather than being priced as a leaf block. No
stream, synchronisation or device allocation behaviour changed in this phase,
so no new sanitizer campaign was required for a changed kernel; the campaign
below covers the changed FP32 BSR construction path.

### Phase H -- CPU and oneMKL closure

Measured above. The explicit production policy is unchanged and now stated:
**`StaticMatrixBackend::Portable` is the default and there is no `Auto` on
this axis; oneMKL runs only when the caller asks for it.** Portable wins the
point workloads decisively (1.32x FP32, 1.88x FP64 at 50k points, p = 6) and
the FP64 finite lattice (1.25x); oneMKL wins the FP32 finite lattice (1.27x).
One crossover in four rows, in the direction of small finite plans, is not a
reason to change a default, and it is not a reason to remove oneMKL either.
A caller whose workload looks like the finite FP32 row has a documented,
one-line way to take that 1.27x.

### Phase I -- architecture review

Two invariants were checked directly and hold. No file under `src/operators/`,
`src/plan/`, `include/cdfmm/operators/` or `include/cdfmm/plan/` mentions
`ExecutionBackend`, `CudaFull`, `CudaPartial`, `SpatialLayout` or
`StaticMatrixBackend`, so no backend concept has leaked into mathematical
construction. No file under `src/backend/cpu/p2p/` or `src/backend/cuda/p2p/`
mentions `SpatialLayout`, `SourceGeometry` or `TargetGeometry`, so the
executors branch only on flags resolved upstream and no geometry rule has
leaked into execution.

Found and fixed here: the BSR budget was the one policy question answered in
two places with two different rules, and the FP32 half was wrong; the notebook
and the two stale doc/comment statements.

Found and deliberately deferred to Phase 4, all confirmed by reading the code:

- `cuda_policy::resolve_cuda_execution_policy` runs for every backend and also
  decides the CPU dictionary, so its name and its home under
  `src/backend/cuda/` understate its ownership.
- `classify_exact_operators` (`src/operators/exact_operator_reuse.hpp`) and
  `classify_endpoint_operators` (`src/fmm/plan_preparation.cpp`) are
  structurally the same helper; the second is not routed through the shared
  one.
- `UniformFmm::use_cuboid_p2m_` / `use_cuboid_l2p_` are assigned in five
  places and read nowhere.
- `CudaExecutionPolicyInputs::periodic`, `bsr_estimate_bytes` and
  `bsr_budget_bytes` are written and never read by any rule.
- The three-argument `far_field_stream_priority` overload has no caller, but
  carries the rule's authoritative documentation, so removing it means moving
  that text.
- `ReducedSymmetryP2P.ipynb` still uses `cuda_p2p_bsr_max_bytes = 0` as an
  idiom for "do not use BSR", which has not steered the policy since leaf
  blocks became the general default.
### Phase J -- final production decision ledger for v0.2

What executes, why, when it changes, how to override it, and what happens when
the choice cannot be honoured. This table is the engineering truth at the
Phase-3D HEAD and is meant to be read without the development history.

| Operator / backend | Representation | Why | Changes when | Override | Fallback |
|---|---|---|---|---|---|
| point P2P, CPU, General | `PointGeometry` (recomputed from positions) | 2.7x faster P2P and 51x less host memory than SoA rows at 50k points | never automatically | `p2p_packing`, `use_reduced_symmetry_p2p` | none needed; it is always constructible |
| point P2P, CPU, RegularGrid | signed `TensorDictionary` | 2.2x faster P2P at 8 per leaf; needs a fixed identity map | the hint is set **and** the built dictionary has 1-2 byte tokens | `p2p_packing` | token width > 2 B, or no fixed identity -> `PointGeometry` |
| point P2P, CUDA, FP32 | `PointGeometry` | 8.4x faster P2P and 14x less device memory than leaf blocks | never; this rule outranks the layout hint | `p2p_packing`, `use_reduced_symmetry_p2p` | none |
| point P2P, CUDA, FP64 | `LeafBlock` (General), `TensorDictionary` (RegularGrid) | recomputation is 1.30x slower in FP64 on this GPU; the lattice dictionary is 1.32x faster than leaf blocks and uses 3.4x less device memory | layout hint, fixed identity, token width | `p2p_packing` | token width > 2 B -> `LeafBlock` |
| finite P2P, CPU, General | `ParticleRowSoa` | stored exact tensors; procedural finite reconstruction lost by 150x-1,300,000x (Phase 3B.5b) | never automatically | `p2p_packing` | -- |
| finite P2P, CPU, RegularGrid | signed `TensorDictionary` | 248 prism / 187 tetrahedron distinct tensors, 28-33x smaller than canonical | hint set and tokens 1-2 B | `p2p_packing` | token width > 2 B -> `ParticleRowSoa` |
| finite P2P, CUDA, General | `LeafBlock` | measured faster than BSR(3) on finite bodies as well as points | never automatically | `p2p_packing` | -- |
| finite P2P, CUDA, RegularGrid | signed `TensorDictionary`, occupancy-calibrated executor | equal-to-5 % faster with 3.4x less device memory | hint set and tokens 1-2 B | `p2p_packing`, `cuda_dictionary_*` | token width > 2 B -> `LeafBlock` |
| point P2M / L2P, CPU (incl. `CudaPartial`) | procedural | 4.9x (FP32) and 5.7x (FP64) faster than streaming the rows, and 3.4-4.8x less memory | spherical basis, order 1-10, not `CpuReference` | `point_expansion_execution` | unavailable -> precomputed |
| point P2M / L2P, `CudaFull` FP32 | procedural | 3.95x faster than the rows | as above | `point_expansion_execution` | as above |
| point P2M / L2P, `CudaFull` FP64 | **precomputed** | procedural is 1.39x slower on P2M+L2P, P2M alone 1.86x | precision alone | `point_expansion_execution` | -- |
| finite P2M / L2P, every backend | precomputed, always | exact operators are expensive to build and cheap to apply; every procedural variant lost (Phase 3B.5b) | never | none; an explicit `Procedural` request that only a finite stage could honour throws | -- |
| M2L, CPU | portable, class-sorted block schedule | 1.32x (FP32) and 1.88x (FP64) faster than oneMKL on point workloads | never; there is no `Auto` on this axis | `static_matrix_backend = OneMkl` | -- |
| M2L, CUDA | target rows, 16 pairs/thread for FP32 with >= 250k translations else 8 | Phase 3A calibration | translation count and precision | none | -- |
| execution backend | `CpuStatic` | `Auto` never selects a CUDA backend | -- | `backend` | requesting an uncompiled CUDA backend throws |
| dense direct, CPU | exact tensors, one build per exact class | Phase 3C.5; classification is skipped for point-to-point plans because a point pair is ~4 ns against ~260 ns | geometry has exact duplicates | `DenseDirectBackend` | irregular geometry classifies nothing |
| dense direct, CUDA | the same host plan, uploaded | one construction path serves all three backends | -- | -- | -- |

Two things this table deliberately does not promise. `SpatialLayout` is a
hint, not a guarantee: it is verified against the built dictionary's token
width and released when the geometry does not compress. And an explicit
`p2p_packing = TensorDictionary` without an executor flag gets the source-warp
kernel, which is 2.59x slower than the calibrated choice below 48 targets per
leaf; that is the only way to select that kernel, and
`cuda_dictionary_power2_microtiles` recovers the difference exactly.

### Phase K -- retained regression baseline

`benchmarks/baselines/phase3d/` holds the accepted-HEAD numbers as CSV plus a
short Markdown summary, labelled an engineering regression baseline. It is
enough to catch a major future regression in construction, warm loading,
evaluation and memory, and it is explicitly **not** an Article1 benchmark: it
must not be copied into the article, compared with jaxFMM or FMM3D, or turned
into publication figures. The publication campaign runs after Phase-4 pruning
against a frozen implementation.

### Phase L -- deferred

The matched production-versus-forced-precomputed point benchmark, the final
finite operator representation summary, the CPU/CUDA scaling and cold/warm
construction plots, the dense-direct comparison, and the jaxFMM and FMM3D
comparisons are all deferred to Article1 after Phase-4 pruning, as the brief
requires. None was started. When it happens it must use dip-fmm's best
production point path -- which is `PointGeometry` with procedural expansions
in FP32 -- rather than forcing precomputed point tensors.
### Phase M -- validation

Four fresh pinned trees at the accepted HEAD, each configured with the conda
`g++` 15.3 as C++ *and* CUDA host compiler and with the environment's Python
3.11 pinned; the Python module under test is the just-built one, selected
through `PYTHONPATH`, because nothing was installed into the conda
environment.

| Tree | CTest | pytest |
|---|---|---|
| portable CPU (`build-v-cpu`) | 244 / 244 passed, 519 s | 140 passed, 8 skipped |
| oneMKL (`build-v-mkl`) | 244 / 244 passed, 503 s | 142 passed, 6 skipped |
| CUDA (`build-v-cuda`) | 244 / 244 passed, 1137 s | 145 passed, 3 skipped |
| CUDA + oneMKL (`build-v-cuda-mkl`) | 244 / 244 passed, 1144 s | 147 passed, 1 skipped |

Skipped cases are those of a backend the tree does not compile. `git diff
--check` is clean.

**Sanitizers.** No kernel, stream or synchronisation was changed, but the FP32
BSR(3) fix makes a previously-unbuilt plan get allocated and uploaded, so that
path is new device-allocation behaviour and was covered:

| Tool | Scope | Result |
|---|---|---|
| `memcheck` | explicit BSR under a lowered budget | 0 errors |
| `memcheck` | BSR memory budget fallback | 0 errors |
| `memcheck` | `[packing]` (every CUDA P2P executor, 10410 assertions) | 0 errors |
| `racecheck` | explicit BSR under a lowered budget | 0 hazards |
| `initcheck` | explicit BSR under a lowered budget | 0 errors |
| `synccheck` | explicit BSR under a lowered budget | 0 errors |

**Cache compatibility.** `src/cache/` is untouched by this phase, so the
stronger property was checked directly rather than assumed. The starting HEAD
`49b5fe3` was exported with `git archive` and built separately, and a cache
was written from cold by each side for a point plan and for a finite prism
lattice:

- the serialised payloads are **byte-identical** between `49b5fe3` and the
  accepted HEAD for both geometries, so no serialised operator value changed;
  and
- each side reads the other's cache as a hit
  (`cache.geometry.hit: true` in both directions), so no cache key changed.

**ABI and API.** No file under `include/` changed at all (`git diff
49b5fe3..HEAD -- include/` is empty), so `CDFMM_ABI_VERSION` stays 1 and the C
header is untouched; `src/bindings/`, `python/`, `fortran/` and `src/cache/`
are likewise untouched. The Python public API is exercised by the four pytest
runs above. The Fortran interface could **not** be compile-verified: no
Fortran compiler is installed in this environment. Its source and the C ABI it
wraps are both unchanged, so the interface is unchanged by construction, but
that is an argument rather than a build, and it is recorded as such.

**Production diff.** Three source files and 40 lines:
`src/operators/exact_operator_reuse.hpp` (+14),
`src/fmm/plan_preparation.cpp` (+15, of which 5 are comment) and
`src/fmm/execution_setup.cpp` (+11), plus a comment-only change to
`src/backend/cuda/m2l/plan.cu`.

## Pre-pruning closure after Phase 3D

Starting HEAD `51b2434` ("docs(perf): close the Phase 3D cross-backend
integration review"), the tip of `phase3d-final-integration`, on the branch
`phase3d-pre-pruning-closure`. This is not Phase 4: nothing was pruned and no
obsolete file was deleted. It closes the four review findings against Phase
3D, makes every first-party compilation path warning-clean, and repairs a
GitHub Actions workflow that had been failing on every branch for twelve days.

**No production policy changed.** All fourteen automatic policies stand
exactly as Phase 3D measured them. The only behavioural change in `src/` is a
`break` after a call that already threw, and it is unreachable either way.

### The four review findings

**Comparison groups were guessed from a display name.**
`analyse_phase3d_regression.py` derived a group by keeping the first, second
and last slash-separated tokens of `case`. Every
`P-cuda-finite/<kind>/{regular,irregular}/...` case shares those three, so a
regular lattice and an irregular cloud were one group of six, divided by
whichever automatic row the file happened to list first. `layout_hint` could
not break the tie: it records only whether `--spatial-layout` was passed, so
an irregular case and an unhinted lattice both read `general`. Each case now
carries `comparison_group` and `comparison_variant` as recorded data — the
group is everything two rows must share before a ratio means anything, the
variant is the single axis under review. On the retained baseline this splits
21 groups into 23, the two six-member `P-cuda-finite` groups becoming four
regular and two irregular rows.

The retained CSVs predate the columns, so the analyser reconstructs their
groups from the driver's own case generators rather than from a second
parsing rule; a case neither source recognises is printed alone. The
generated case list was checked against the baseline: 158 names, identical
and in the same order, so **no regeneration was needed and none was done.**

**A failed case did not fail the run.** `run_case` returned `None` on a
non-zero exit or an empty result file, and the driver filtered those out and
exited zero, writing a short CSV. A baseline that drops rows silently cannot
be distinguished from a complete one. A failure now raises, names the case
and fails the run, and the row count is checked against the number of cases
the selected suite and backend matrix define. `--allow-failures` restores
skipping for exploration only.

**Resume could mix sessions.** A per-case CSV was reused because the file
existed and was non-empty, which silently mixes commits, binaries and sample
counts into something that looks like one session. A run is now fresh by
default; `--resume` reuses files only when `session.json` matches — revision,
binary path and SHA-256 with size and mtime beside it, suite and filter,
evaluation, warm-up, sample and thread counts, backend availability, and the
column set. A mismatch is refused and names the differing field.

**The integration history was stated wrongly.** Both Phase-3D records said
`refactor/architecture-v0.2` still pointed at `a2af367` when the phase began,
which reads as a claim about the branch. It described this checkout's local
ref. The remote branch had already been advanced through Phase 3C/3C.5 to
`49b5fe3`, and the reflog shows the local ref catching up by fast-forward
pull, not a push. Corrected in both files; no performance result was touched.

Twenty-three new checks in `python_tests/test_phase3d_regression_matrix.py`
pin all three driver properties at the lowest layer that can hold them: the
generators, the analyser's grouping and the driver's exit behaviour are pure
Python, so none of it needs the extension or a device.

### Warning cleanliness

The project set **no warning flags at all**, so roughly 115 `.cpp` and 8 `.cu`
files had never been read by `-Wall -Wextra`. `cdfmm_enable_warnings()` now
applies the project warning set per first-party target at the same twelve call
sites as `cdfmm_enable_ipo()`, which keeps it off Catch2 and pybind11 by
construction — they are never passed to it. Host warnings reach nvcc
translation units through `-Xcompiler`; device diagnostics stay at nvcc's own
default level.

`CDFMM_WARNINGS_AS_ERRORS` is **off by default and on in CI**. That split is
deliberate: a downstream build uses a compiler this project has not seen, and
a future release may add a diagnostic that is not a defect here, so failing
that build would serve nobody; CI runs the one toolchain the project does
support, which is where a new warning should stop a change.

Thirty-three diagnostics were found, all first-party, all fixed at the source.
No `-w`, no blanket `/wd`, no nvcc suppression, no `#pragma` was added, and
the warning level was not lowered anywhere.

- `src/math/solid_harmonic_recurrence.hpp` keyed its unroll hint on
  `__CUDACC__`, which nvcc leaves defined while handing the host half of a
  `.cu` to the host compiler, so g++ parsed a pragma it does not know. It is
  now keyed on `__CUDA_ARCH__`: the device pass keeps `#pragma unroll` and the
  host half takes the same host spelling a `.cpp` already took. Device codegen
  is unchanged.
- `src/fmm/execution_setup.cpp` fell through from the `ParticleRowSoa`
  rejection into the `PointGeometry` case. `reject()` always throws, so the
  path was already unreachable; the `break` states that for the compiler.
- `src/geometry/primitives/tetrahedron.cpp` formed two triangle edges it never
  used and kept a `body_point_tensor` helper with no remaining caller,
  superseded by `SourcePointField`.
- `src/backend/cpu/p2p/near_field.cpp` bound a leaf node and the node array
  without reading either.
- Five by-value structured bindings over pairs now bind by reference
  (`src/fmm/plan_preparation.cpp`, `src/plan/p2p/leaf.cpp` and three tests).
  Only the runner's g++ reported these; neither g++ 15.3 nor a conda g++ 13.4
  emits `-Wrange-loop-construct` for them.
- Benchmarks and tests: an aggregate-initialised `GeometryPlan` gained an
  explicit constructor, the dense-direct volatile checksum sink moved to
  namespace scope because nothing reads it back, `run_p2m_precision` lost a
  parameter it never used, and three tests dropped unused locals, used the
  constants they had declared, and stopped copying an `initializer_list`
  element.

### Warning-clean build matrix

Every tree below is **fresh** (`rm -rf` then configure), built with
`CDFMM_WARNINGS_AS_ERRORS=ON`, tests, examples, benchmarks, tools and the
Python extension all enabled, and reports **0 warnings and 0 errors**:

| Configuration | Compiler | Result |
|---|---|---|
| portable CPU | g++ 15.3.0 | 0 / 0 |
| CPU + oneMKL | g++ 15.3.0 | 0 / 0 |
| CUDA | g++ 15.3.0 + nvcc 13.3.73 | 0 / 0 |
| CUDA + oneMKL | g++ 15.3.0 + nvcc 13.3.73 | 0 / 0 |
| portable CPU, `Debug` | g++ 15.3.0 | 0 / 0 |
| portable CPU | conda g++ 13.4.0 | 0 / 0 |
| portable CPU (GitHub runner) | ubuntu-24.04 g++ | see CI below |

The **Fortran interface could not be built**: no Fortran compiler exists in
this environment. `cdfmm_enable_warnings()` emits only `CXX` and `CUDA`
generator expressions, so it cannot affect a Fortran target, and
`fortran/` is byte-identical to `51b2434`; the interface is therefore
unchanged by construction. That is an argument, not a build, and is recorded
as such. MSVC was likewise not exercised — no Windows machine and no Windows
CI job exists.

### GitHub Actions

CI had been failing on **every** branch since 2026-09-07, including
`refactor/architecture-v0.2` and the Phase-3D head `51b2434`; this was
pre-existing and not caused by the closure. At `51b2434` the build and CTest
steps passed and **pytest failed**. Downloading a run log needs repository
rights, so the workflow was first changed to re-publish its diagnostics as
`::error::` annotations, which are part of the public run summary; every
failure below was then read from the runner's own output.

- **Pre-existing:** `test_adaptive_notebook.py` and
  `test_geometry_magtense_all_to_all_notebook.py` import `nbformat` at module
  scope, and the workflow's hand-written dependency list never included it, so
  pytest failed at *collection* rather than skipping. The list had drifted
  from what the tests import, so `nbformat` joined the `test` extra in
  `pyproject.toml` and the workflow installs `.[test]`. Reproduced in a clean
  interpreter carrying only the workflow's packages: the collection error
  becomes **163 passed, 8 skipped**, matching the local build, and those two
  tests run in CI for the first time.
- **Introduced and fixed here:** warnings-as-errors turned the five by-value
  structured bindings into errors under the runner's compiler.

Both jobs are **green** on the closure head `bbe3926` (run 35468228726):
`first-party warning surface` and `portable CPU build and tests` each
succeeded, which is the first green run on this repository since 2026-09-07.
No job was skipped and no coverage was removed to get there; the run is
strictly wider than the one it replaces, because it now also compiles the
benchmarks and runs the two notebook tests that previously failed collection.

The workflow also moved off the deprecated Node 20 actions
(`actions/checkout@v4` → `@v5`, `actions/setup-python@v5` → `@v6`), pinned
`runs-on` to `ubuntu-24.04` rather than `ubuntu-latest` so an image bump
cannot silently change what warnings-as-errors means, gained a `concurrency`
group, builds with `-k` so one run reports every diagnostic, prints the path
of the `cdfmm` extension it is about to import, and gained a second job that
compiles **every** first-party target — including the benchmarks, which no
automated build had ever compiled.

### Test and sanitizer matrix

Run on the final tree (CUDA + oneMKL, `-Werror`, built from the closure
sources):

| Check | Result |
|---|---|
| `ctest` (CUDA + oneMKL) | **244/244 passed**, 1102 s |
| `pytest python_tests` (build tree) | **171 passed, 1 skipped** |
| `pytest python_tests` (portable CPU tree) | 163 passed, 8 skipped |
| `pytest` in a CI-faithful interpreter | 163 passed, 8 skipped |
| `git diff --check` | clean |

The module the tests import was recorded rather than assumed. An older
`cdfmm` extension *is* installed in this machine's conda environment, and
`PYTHONPATH=<build>` was verified to override it. The extension links
`cdfmm_core` statically and carries no `libcdfmm_c.so` dependency, so the
stale `libcdfmm_c.so` in the same environment cannot be picked up either.
CI prints the path it is about to import for the same reason.

Two pre-existing tests dominate the suite: `procedural point P2M and L2P
reproduce the precomputed rows` at 1102 s and `procedural point expansion
requests are validated` at 364 s, together about 96% of the wall time.
Neither file is touched by this closure; they are recorded for Phase 4
rather than changed here.

Sanitizers, on the same tree. This closure changes no device code -- the only
edit reaching a `.cu` is the unroll-pragma guard, and the device pass still
expands `#pragma unroll` exactly as before -- so a representative matrix was
run rather than a full sweep:

| Tool | Cases | Result |
|---|---|---|
| `memcheck` | six representative CUDA cases: both backends, both precisions, point and finite geometry, stored packings and the position-based kernel | 0 errors each |
| `racecheck` | full-FMM device residency, stored P2P packings | 0 hazards, 0 errors, 0 warnings |
| `initcheck` | the same two | 0 errors |
| `synccheck` | the same two | 0 errors |

The driver was also exercised end to end against the real
`benchmark_uniform_fmm`: a fresh run reported "6 of 6 rows" and populated
both comparison columns, the manifest recorded the revision and the binary's
SHA-256, a matching `--resume` reused every per-case file, and a run with
`--samples` changed was refused with the differing field named.

### API, ABI, cache and baseline

`git diff 51b2434..HEAD -- include/ src/bindings/ src/cache/ python/ fortran/`
is **empty**. No public C++ API, C ABI, Python API or Fortran interface
changed, `CDFMM_ABI_VERSION` stays 1, and no cache format or cache key
changed. `git diff --check` is clean. The whole `src/` change is 44 lines
across six files.

The retained baseline under `benchmarks/baselines/phase3d/` is **unchanged**:
158 rows, the same case names in the same order, the same numbers. Its README
now records the driver's contract and how to regenerate it safely, and keeps
its ENGINEERING REGRESSION BASELINE — NOT ARTICLE1 status.

### Remaining Phase-4 inventory

Carried forward, untouched here and explicitly not started:

- flat public compatibility façades under `include/cdfmm/`, retained
  deliberately and removable only by an approved API change;
- `src/operators.cpp` as thin compatibility delegation, and the
  `DenseDirectPlan::evaluate()` compatibility dispatch;
- `benchmarks/AGENTS.md`'s file tree, which is stale with respect to the
  Phase-3D drivers and `baselines/phase3d/`;
- geometry packing, grain generation/discretisation and prism/tetrahedron
  refinement, all still future work;
- no Windows, MSVC, CUDA, oneMKL or Fortran CI coverage, each needing a
  runner, toolkit, device or compiler that is not available today; and
- two pre-existing tests take about 96% of the C++ suite's wall time
  (`procedural point P2M and L2P reproduce the precomputed rows` at 1102 s
  and `procedural point expansion requests are validated` at 364 s), which is
  what makes the CI test step slow. Neither was touched here.

## Phase 5 — timing overhead and implementation freeze

**Status: FROZEN FOR ARTICLE1 BENCHMARKING** at `62835b5`. Starting HEAD `8f54e6f` (`refactor: defer the endpoint
classifier merge, which GCC 13 LTO rejects`, the tip of `phase4-pruning`);
work on `phase5-timing-freeze`, based exactly on it. Toolchain: g++ 15.3
(conda `cdfmm`), nvcc 13.3, oneMKL, CMake 4.4; CI reproduction with
conda-forge g++ 13.4; hardware: i9-14900KF (8 threads used), RTX 5090
(sm_120, driver 595.84).

Question answered: how much does the always-on instrumentation cost, and can
it be made opt-in without touching results, policy, cache or functional
synchronisation?

### A. Instrumentation inventory (starting HEAD)

Three mechanisms existed and were entangled at three CUDA events:

| Mechanism | Where | Construction / evaluation | Unconditional? |
|---|---|---|---|
| Host `std::chrono::steady_clock` phase clocks | `src/fmm/evaluation.cpp` (total, P2P, unpermutation, hybrid wait), `src/fmm/far_field.cpp` (moment permutation, resets, P2M, M2M, per-level L2L and M2L, L2P), `src/backend/mkl/m2l.cpp` (gather/multiply/scatter) | evaluation | yes: about 14 clock reads per CPU evaluation, 2 per hybrid M2L call, 3 per oneMKL level |
| Host clocks | `src/fmm/construction.cpp`, `plan_preparation.cpp`, `execution_setup.cpp`, `src/cache/{universal,geometry,keys}.cpp` | construction | yes: about 40 reads per cold build |
| CUDA timing events | `CudaFullPlan` 13 events, 13 `cudaEventRecord` + 12 `cudaEventElapsedTime` per evaluation; `CudaP2PPlan` 4 + 3; `CudaM2LPlan` 5 + 4 (one recorded inside the shared M2L `enqueue`); `CudaDirectPlan` 4 + 3; `CudaDenseDirectPlan` none | evaluation | yes; every event created timing-capable |
| NVTX `ProfileRange` | `src/profile.hpp`, used in `evaluation.cpp`, `far_field.cpp`, `cuda/fmm/plan.cu` | evaluation | no: compile-time `CDFMM_ENABLE_PROFILING` (`CDFMM_ENABLE_NVTX`), an empty object otherwise |
| Tree build timings, dense-direct construction statistics | `UniformTree`, `AdaptiveTree`, `src/plan/direct/dense.cpp`, `src/backend/cuda/direct/dense.cu` | construction | yes; outside `UniformFmm`, one-time, unchanged in this phase |

No timer sat inside a per-body or per-leaf loop; every host clock bracketed a
whole OpenMP region or a whole backend call, so the per-thread MagTense timer
was not needed. No `cudaDeviceSynchronize` existed anywhere and none was
added.

**CUDA event classification.** Of `CudaFullPlan`'s 13 events exactly three
carried a functional role: `moments_ready` (the near-field stream waits on it),
`p2p_complete` (the far-field stream waits on it before combining) and
`d2h_complete` (the host's completion point). Each was *also* an
elapsed-time endpoint. The other ten (`evaluation_start`, `p2m_complete`,
`m2m_complete`, `m2l_scale_complete`, `m2l_complete`, `l2l_complete`,
`l2p_complete`, `p2p_start`, `combination_start`, `combination_complete`)
were diagnostic only. In the single-stream plans (`CudaP2PPlan`,
`CudaM2LPlan`, `CudaDirectPlan`) only the terminal `d2h` event is functional
(the `cudaEventSynchronize` completion point); `start`, `h2d`, `scale`,
`kernel` are diagnostic. The C ABI exposed one timing value
(`cdfmm_plan_get_last_evaluation_seconds` = `last_timings().total`); the
Fortran wrapper exposes none.

### B. Measured cost of the old instrumentation

External `steady_clock` around complete `evaluate_into` calls (the driver's
`evaluation_median`; 20 evaluations x 9 samples, 5 warm-ups, 8 threads,
`OMP_PROC_BIND=close OMP_PLACES=cores`, 5 interleaved repetitions per build,
median of the per-run medians). "hard-off" is a temporary tree of the starting
HEAD with every diagnostic record, every `cudaEventElapsedTime` and every host
evaluation clock removed, the three functional events kept and created with
`cudaEventDisableTiming` (`/tmp` experiment, never committed).

| Workload | start (us) | hard-off (us) | hard-off / start | absolute |
|---|---:|---:|---:|---:|
| CPU point 10k, d3, p4, FP32 | 1249.5 | 1252.9 | 1.003 | noise |
| CPU point 50k, d4, p6, FP32 | 11730.0 | 11747.7 | 1.002 | noise |
| oneMKL point 50k, d4, p6, FP32 | 15698.4 | 15590.0 | 0.993 | -108 |
| CudaFull point 4k, d2, p4, FP32 | 101.8 | 86.2 | 0.848 | -15.6 |
| CudaFull point 10k, d3, p4, FP32 | 116.0 | 103.8 | 0.894 | -12.2 |
| CudaFull point 50k, d4, p6, FP32 | 446.2 | 431.4 | 0.967 | -14.8 |
| CudaFull point 50k, d4, p6, FP64 | 2787.1 | 2695.6 | 0.967 | -91.5 |
| CudaFull prism 4k, d3, p6, FP32 | 127.6 | 107.9 | 0.846 | -19.7 |
| CudaPartial point 50k, d4, p6, FP32 | 1072.6 | 1063.8 | 0.992 | -8.8 |
| standalone CUDA P2P canonical AoS (d3, occ 8) | 182.6 | 173.8 | 0.952 | -8.8 |
| standalone CUDA P2P particle-row SoA | 268.9 | 266.1 | 0.990 | -2.8 |
| standalone CUDA P2P leaf block | 48.2 | 38.5 | 0.798 | -9.7 |
| standalone CUDA P2P cuSPARSE BSR(3) | 61.2 | 57.1 | 0.934 | -4.1 |

Run-to-run spread of a case's medians was about 1 us on the CUDA cases, so
these differences are real. The instrumentation cost a `CudaFull` evaluation
12-20 us regardless of size: 15 % of the fastest cases, 3.3 % at 50k points,
and 20 % of the fastest standalone P2P plan. On the CPU it was below the noise
(<0.3 %). Removed per `CudaFull` evaluation: 10 event records, 12
elapsed-time queries and 1 host clock pair; per hybrid evaluation 3 + 3 (P2P)
and 4 + 4 (M2L) plus 14 host clock reads.

### C. Final design

`TimingLevel { Off, Coarse, Detailed }` in `cdfmm/core/timing.hpp`;
`UniformFmmOptions::timing_level` (default `Off`), `UniformFmm::timing_level()`
and `set_timing_level()`. Every timing record carries the level it was
collected at (`EvaluationTimings::timing_level`,
`StaticPlanStatistics::timing_level`, `CudaEvaluationTimings::timing_level`),
so an uncollected zero is never mistaken for a measurement.

- **Off**: no host clock is read during evaluation or construction; the CUDA
  plans record only their functional events, which are now created with
  `cudaEventDisableTiming`; `cudaEventElapsedTime` is never called; the
  device lanes are not copied. Byte, count and cache-hit statistics are
  unchanged.
- **Coarse**: host wall times only. Evaluation: `total`; on the CPU
  hierarchy `far_field` (new field: the whole P2M..L2P branch) and `p2p`;
  the hybrid backend's `cuda_p2p_wait`. Construction: `total_setup` and the
  static-plan `total`. No device event is recorded, because the `CudaFull`
  near and far lanes overlap and a device total would just duplicate the host
  total.
- **Detailed**: the previous depth: every host phase, the oneMKL split, the
  construction subphases including cache lookup/load/write, and the device
  lanes from a *separate diagnostic event graph*: the three functional events
  gain timing twins (`moments_permuted`, `p2p_finished`, `evaluation_end`),
  so the functional graph never changes with the level. A `CudaFull`
  evaluation records 16 events and queries 12 elapsed times at `Detailed`;
  the single-stream plans 5 (4 diagnostic + the functional `d2h`) and 3-4.

Host clocks are one `detail::PhaseStopwatch` (`src/phase_stopwatch.hpp`) per
level: `start()`/`record()` are one predictable branch when the level does
not include the region, following the MagTense rule that the verbosity gate
precedes the clock. CUDA diagnostic records go through
`cuda_detail::record_diagnostic` (`src/backend/cuda/common/diagnostic_event.hpp`).
The shared M2L `enqueue` takes a nullable phase event. The cache functions
gate on `statistics.timing_level`, so no cache signature changed. NVTX is
untouched and independent: a `profile-all` build can run `Off` under Nsight.

**Runtime setter.** `set_timing_level` is supported because no resource
depends on the level: the CUDA plans hold all events permanently and only
the *records* are gated, so switching is a member write plus a propagation
to the owned device plans (`set_timing_level` on `CudaFullPlan`,
`CudaP2PPlan`, `CudaM2LPlan`, `CudaDirectPlan`). It resets the aggregate so
an aggregate never mixes levels.

**Bindings.** Python: `cdfmm.TimingLevel` (`OFF`/`COARSE`/`DETAILED`),
`options.timing_level`, `plan.timing_level`, `plan.set_timing_level`,
`plan.reset_timings`; the timing dictionaries gain `far_field` and
`timing_level`. C ABI: additive `cdfmm_plan_set_timing_level` and
`CDFMM_TIMING_*`; `cdfmm_plan_get_last_evaluation_seconds` now fails with
`CDFMM_ERROR_UNSUPPORTED` while a plan is `Off` instead of returning zero;
options struct and ABI version unchanged. Fortran: unchanged, exposes no
timing (decision recorded). `suggest_parameters_for_accuracy` forces
`Detailed` on its own candidate plans because it splits near and far.

**Not gated, deliberately.** `UniformTree`/`AdaptiveTree` build timings and
the dense-direct construction statistics are one-time construction records
outside `UniformFmm` (about ten clock reads per build); they are copied into
`StaticPlanStatistics::tree_construction` only at `Detailed`.

**Semantic change worth noting.** The FP32 hybrid path's `p2p` used to span
fill, wait and accumulation on the host; it is now the device lane sum, as on
the FP64 path. `reset_timings()` now also records the level in the cleared
aggregate.

### J. Final performance

Same protocol as B, five builds interleaved (times in us; medians of the
five per-run medians; `phaseJ_all_builds.csv`):

| Workload | hard-off | start | Off | Coarse | Detailed | Off/hard-off | Coarse/hard-off | Detailed/Off | start/hard-off |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU point 10k, d3, p4, FP32 | 1251.1 | 1253.0 | 1258.6 | 1254.1 | 1256.8 | 1.006 | 1.002 | 0.999 | 1.002 |
| CPU point 50k, d4, p6, FP32 | 11773.4 | 11786.6 | 11758.7 | 11774.7 | 11753.9 | 0.999 | 1.000 | 1.000 | 1.001 |
| oneMKL point 50k, d4, p6, FP32 | 15732.3 | 15596.3 | 15620.5 | 15619.4 | 15590.4 | 0.993 | 0.993 | 0.998 | 0.991 |
| CudaFull point 4k, d2, p4, FP32 | 86.3 | 102.3 | 86.2 | 86.5 | 91.7 | 0.999 | 1.002 | 1.064 | 1.185 |
| CudaFull point 10k, d3, p4, FP32 | 104.1 | 115.1 | 104.1 | 104.4 | 121.4 | 1.000 | 1.003 | 1.166 | 1.106 |
| CudaFull point 50k, d4, p6, FP32 | 431.0 | 448.1 | 431.2 | 431.7 | 448.7 | 1.000 | 1.002 | 1.041 | 1.040 |
| CudaFull point 50k, d4, p6, FP64 | 2695.6 | 2706.1 | 2698.2 | 2697.7 | 2710.0 | 1.001 | 1.001 | 1.004 | 1.004 |
| CudaFull prism 4k, d3, p6, FP32 | 108.3 | 127.4 | 108.3 | 108.1 | 128.0 | 1.000 | 0.998 | 1.182 | 1.176 |
| CudaPartial point 50k, d4, p6, FP32 | 1067.7 | 1070.9 | 1062.0 | 1063.9 | 1069.9 | 0.995 | 0.996 | 1.007 | 1.003 |
| standalone P2P canonical AoS | 172.2 | 183.0 | 172.6 | 173.9 | 182.6 | 1.002 | 1.010 | 1.058 | 1.063 |
| standalone P2P particle-row SoA | 266.8 | 269.6 | 266.5 | 266.0 | 268.4 | 0.999 | 0.997 | 1.007 | 1.010 |
| standalone P2P leaf block | 40.2 | 48.1 | 45.1 | 38.9 | 47.9 | 1.122 | 0.968 | 1.062 | 1.197 |
| standalone P2P cuSPARSE BSR(3) | 59.4 | 63.5 | 56.8 | 57.5 | 62.6 | 0.956 | 0.968 | 1.102 | 1.069 |

**Off acceptance.** On every FMM workload the final `Off` path is within
0.1-0.6 % of the hard-off lower bound (0.995-1.006), i.e. inside the
run-to-run spread; the fastest CUDA cases agree to 0.1 us. The standalone P2P
leaf-block row (Off 1.12, Coarse 0.97 on the *same* binary and the same
device path, since neither level records a device event) is noise in a
minimum-of-50 at 40 us; a dedicated re-measurement of that driver alone (10
interleaved repetitions, 200 evaluations each,
`p2p_standalone_repeat.log`) gave medians hard-off 43.4 / Off 43.7 /
start 46.5 / Detailed 47.1 us for the leaf block and 173.0 / 173.2 / 181.9 /
182.4 us for canonical AoS, so Off equals hard-off there too.

**Coarse** costs at most 0.3 % anywhere (two to four host clock reads).
**Detailed** reproduces the old always-on cost, now quantified once and
for all: +6 % on the 86 us case, +17-18 % on the 104-108 us cases, +4 % at
50k FP32, +0.4 % at 50k FP64, +6-10 % on the standalone CUDA P2P plans, and
nothing measurable on the CPU. Recorded numbers live in
`benchmarks/baselines/phase5-timing/` (labelled INTERNAL TIMING OVERHEAD
STUDY, NOT ARTICLE1); `benchmarks/baselines/phase3d/` is unchanged.

### K/L. Validation

- Correctness of the timing code: `tests/test_timing_levels.cpp` (Off collects
  nothing; Coarse populates only the coarse fields; Detailed every phase;
  aggregate accumulates; `reset_timings` clears only timing; a run-time
  level change leaves results, `execution_plan`, packing and point-expansion
  resolution unchanged; cache keys *and persisted file bytes* identical
  across levels and a warm load at a third level agrees; CUDA backends agree
  across levels to the backend's own reproducibility, i.e. bitwise when a
  repeat at one level is bitwise, and gate their lanes; the CUDA direct plan
  gates its lanes), `python_tests/test_timing_levels.py`, and the C ABI test
  (`Off` makes `cdfmm_plan_get_last_evaluation_seconds` fail,
  `set_timing_level` succeeds). Tests that read timing call counts now ask
  for `Detailed` explicitly.
- Fresh warning-as-error builds: portable CPU (CI reproduction with
  conda-forge g++ 13.4, Unix Makefiles, LTO, `-Werror`: 252/252 CTest,
  173 passed / 9 skipped pytest), and CUDA + oneMKL (`notebooks` preset plus
  benchmarks, g++ 15.3 / nvcc 13.3, `-Werror`: 252/252 CTest, 181 passed /
  1 skipped pytest including the six executed tutorials); the
  `benchmark-all` preset (the Phase-J binaries). CPU + oneMKL without CUDA
  and CUDA without oneMKL were not built separately in this pass: every
  translation unit they compile is covered by the two configurations above
  (the oneMKL and CUDA sources are the union, and the stubs are compiled by
  the CPU-only build).
- `compute-sanitizer` memcheck, racecheck, initcheck and synccheck over the 35 CUDA- and timing-tagged C++ test cases (41 312 assertions): 0 errors, 0 hazards each.
- `sphinx-build -W` clean; `git diff --check` clean.
- Cache format and keys: unchanged (the key strings still end in `_v04`; the
  timing-level test compares the persisted bytes of an `Off` and a
  `Detailed` cold build and finds them identical). C ABI: additive only
  (`CDFMM_ABI_VERSION` stays 1; `cdfmm_options` unchanged); the accessor's
  new failure while `Off` is a behaviour change for a C caller that never
  selected a level, documented in `docs/c-and-fortran.md`. Fortran wrapper:
  byte-identical to the starting HEAD, no Fortran compiler available here.
  NVTX: `src/profile.hpp` untouched.
- Numerical results: the CPU tests compare bitwise across levels; the CUDA
  levels agree to the backend's reproducibility; the pre-existing
  CPU-versus-CUDA agreement tests pass unchanged.

### Commits and status

Seven commits: `62e4ce8` CUDA event separation, `776d26a` UniformFmm timing levels, `fa870c6` Python/C ABI, `244b341` benchmark drivers and the overhead record, `7688d6a` tests, `3fb36d2` examples, `62835b5` documentation. The implementation is **FROZEN FOR ARTICLE1 BENCHMARKING** at
`62835b5`; the agent-documentation commit follows it. CI: green on `5cdfaa8` (run 35571431509), both jobs and every step, including the C++ and Python test steps. The run on `62835b5` shows as cancelled because the workflow sets `cancel-in-progress` per ref and the documentation push superseded it; `5cdfaa8` contains that commit's tree plus the agent docs.
Next: the Article1 benchmark campaign, with `timing_level = Off` and external
wall-clock timing of repeated `evaluate_into` calls; detailed timing is a
separate diagnostic run.
