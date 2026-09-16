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
