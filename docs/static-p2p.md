# Static P2P execution study

## Tensor6 dictionary packing

The P2P execution plan retains the exact canonical interaction topology as
dense target/source leaf blocks. Each interaction stores one four-byte token:
a 26-bit Tensor6 dictionary ID and six independent sign bits for
`xx, xy, xz, yy, yz, zz`. The dictionary stores the component-wise absolute
Tensor6 entries once, using exact bit patterns at the selected execution
precision. FP32 is quantised before its dictionary is deduplicated.

This packing does not inspect particle coordinates or attempt to detect a
regular grid. It applies to point and finite cuboid P2P operators alike. A
point self interaction is skipped only when its canonical block requests it;
finite cuboid centre interactions are retained.

The dictionary is experimental and is selected only when
`use_reduced_symmetry_p2p` is explicitly enabled. The reported plan statistics
include tensor counts and canonical/dictionary persistent-memory fields for
benchmark inspection.
interactions), leaf blocking was 1.21x, and oneMKL BSR was 0.80x.

## Sweep-based recommendation

The latest complete sweep covered 19 runnable `(particle count, depth)` cases
from 2^12 through 2^17 particles and depths two through five. CPU SoA won
12 cases, was within 10% of the winner in 15, and had a 1.154 geometric-mean
speedup over canonical AoS. It is therefore the new portable CPU default.

CUDA BSR(3) won 12 cases, was within 10% of the winner in 17, and had a 1.757
geometric-mean kernel speedup over canonical CUDA. Its full blocks use about
35% more persistent P2P storage than canonical rows, so it is the preferred
fixed-identity path subject to the explicit memory budget rather than an
unconditional choice. Canonical CUDA remains the correct default for changing
identity maps and the fallback when BSR is too large.

Leaf-block CUDA won only the high-occupancy `(N=65536, depth=4)` and
`(N=131072, depth=4)` cases in that sweep. Those two points do not yet justify
an occupancy dispatch rule, so leaf blocking remains an experimental packing.

## Further experiments

1. Profile the fixed-identity BSR and canonical fallback with Nsight; compare
   kernel time separately from transfers and confirm the depth-five crossover.
2. Tune leaf source-batch size and block size only if the profiler identifies
   shared-memory, occupancy, or register-pressure limits.
3. Test AoSoA and source batching for long CPU rows.
4. Revalidate the unconditional CPU SoA policy on additional architectures
   with fixed affinity, cache, SIMD, and NUMA measurements.
5. Benchmark multiple moment right-hand sides separately; that workload may
   favour dense leaf kernels or GEMM.

## Reproduction

```console
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./build-bench-all/benchmarks/benchmark_p2p \
  --depth 1 --sweep --evaluations 50

OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores MKL_NUM_THREADS=8 \
  ./build-bench-all/benchmarks/benchmark_p2p \
  --depth 2 --occupancy 32 --evaluations 50

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./build-bench-all/benchmarks/benchmark_p2p \
  --depth 2 --occupancy 8 --irregular --evaluations 50
```

Configure and run the CUDA comparison manually with:

```console
cmake --fresh --preset cuda
cmake --build build-cuda --target benchmark_p2p cdfmm_tests -j

ctest --test-dir build-cuda --output-on-failure \
  -R "CUDA static P2P packings"

./build-cuda/benchmarks/benchmark_p2p \
  --cuda --depth 2 --occupancy 8 --evaluations 50
./build-cuda/benchmarks/benchmark_p2p \
  --cuda --depth 1 --sweep --evaluations 50
./build-cuda/benchmarks/benchmark_p2p \
  --cuda --depth 2 --occupancy 8 --irregular --evaluations 50
```

The CUDA preset clears Conda's `NVCC_PREPEND_FLAGS` and selects `g++` as both
the C++ compiler and nvcc host compiler. This avoids accidentally forwarding
the CPU-only `icpx` benchmark compiler to nvcc.

The CUDA table reports setup, H2D, kernel, D2H, device-total, and host-total
times; speedups relative to canonical CUDA; interaction throughput; and value,
index, row, leaf, identity, scratch, and total persistent device bytes.

## Stored CPU, oneMKL, and CUDA sweep

The sweep runner uses the combined `benchmark-all` executable so every case
contains portable CPU, oneMKL BSR(3), and all CUDA implementations:

```console
cmake --fresh --preset benchmark-all
cmake --build --preset benchmark-all -j
python benchmarks/run_p2p_sweep.py
```

By default it tests exactly 2^12 through 2^17 particles at every depth from 2
through 5. Before launching a case, the runner estimates its particle
interaction count, simultaneous host packing storage, and largest persistent
CUDA plan. A case is recorded in `skipped_cases.csv` rather than launched when
the estimates exceed 50 GiB of host memory or 20 GiB of device memory. A case
is also skipped when currently available memory is lower than the corresponding
configured limit. Override the matrix or limits when needed, for example:

```console
python benchmarks/run_p2p_sweep.py \
  --particles 4096 8192 16384 32768 65536 131072 \
  --depths 2 3 4 5 \
  --gpu-memory-limit-gib 20 \
  --host-memory-limit-gib 50 \
  --evaluations 100 \
  --threads 8
```

Each run creates a timestamped directory below `p2p_sweep_results` containing
raw output per completed case, `cases.csv`, `cpu_results.csv`,
`cuda_results.csv`, `skipped_cases.csv` when a memory guard is triggered, and
four figures covering runtime, persistent memory, interaction throughput, and
the runtime-versus-memory trade-off. CSV files are updated after every case so
completed results survive an interrupted or out-of-memory sweep.
