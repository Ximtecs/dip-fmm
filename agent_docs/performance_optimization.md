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
