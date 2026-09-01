# Static P2P execution study

## Regular-grid displacement experiments

High leaf occupancy makes the pairwise tensor stream grow approximately as
`27 N q`.  On a complete Cartesian lattice with coincident sources and targets,
however, the tensor depends only on the integer source-minus-target offset.
The experimental `StaticP2PGridStencilPlan` therefore retains one Tensor6 per
unique offset and makes interaction rows refer to that compact table.  For an
`s x s x s` leaf, a complete 3 x 3 x 3 list1 neighbourhood needs at most
`(4s - 1)^3` offsets: `s=10` needs 59,319 tensors, or 1,423,656 bytes in FP32.

Eligibility detection is deterministic construction-time work.  It currently
requires a complete axis-aligned Cartesian product, constant spacing on every
non-singleton axis, coincident source and target counts, a fixed identity map,
and canonical rows whose indices address that grid.  An irregular, incomplete,
duplicated, rotated, or non-coincident geometry is rejected; production callers
must retain the particle-row SoA or resolved CUDA fallback.  Detection is not
performed during evaluation.

The CPU table executor assigns one target row to an OpenMP iteration and uses
an SIMD reduction over sources.  A second point-only executor obtains physical
displacements from integer offsets and evaluates
`H = [3 r (m.r)/|r|^5 - m/|r|^3]/(4 pi)` on demand.  It avoids coordinate and
Tensor6 traffic and provides the requested table-versus-arithmetic experiment.
Both are derived execution representations; the canonical operator remains the
mathematical reference.

The CUDA experiment uploads the same compact displacement table and launches
one block for each target tile rather than one block for a complete target
leaf. Tile sizes 64, 128, and 256 are supported. Each thread may own one, two,
or four consecutive targets, retaining separate field accumulators while it
walks their rows. Point plans can select either Tensor6 lookup or on-the-fly
integer-displacement arithmetic. The constructor rejects unsupported tuning
values and cuboid use of the point-only arithmetic path. These variants are
available through the isolated benchmark and are not yet production dispatch
defaults.

The current experiment deliberately does **not** claim a performance winner.
In particular, `q=1000` is a useful stress point rather than a target optimum.
The total balance should be interpreted using
`T(q) ~ N [C_p2p q + C_far/q]`, and the optimum must be measured on the target
CPU and GPU. CUDA target tiles and targets-per-thread remain benchmark
parameters to validate before production dispatch; no unmeasured
occupancy-dependent policy is encoded.

The compact-table memory report separates unique tensor bytes, interaction
indices, row metadata, and grid/displacement metadata.  The present portable
prototype still has per-interaction source and displacement indices.  Removing
those indices by deriving them from complete-tree leaf coordinates is the next
required step before the representation can construct the largest occupancy
cases without first materialising canonical rows.

For a small correctness run and a bounded tuning sweep:

```console
./build-cuda/benchmarks/benchmark_p2p --cuda --depth 2 \
  --occupancy 125 --regular-grid-s 5 --evaluations 20 \
  --cuda-target-tile 128 --cuda-targets-per-thread 2

for tile in 64 128 256; do
  for targets in 1 2 4; do
    ./build-cuda/benchmarks/benchmark_p2p --cuda --depth 2 \
      --occupancy 125 --regular-grid-s 5 --evaluations 20 \
      --cuda-target-tile "${tile}" --cuda-targets-per-thread "${targets}"
  done
done
```

Representative Nsight Compute collection can use the same command line:

```console
ncu --set full --kernel-name regex:grid_p2p_kernel \
  ./build-cuda/benchmarks/benchmark_p2p --cuda --depth 2 \
  --occupancy 125 --regular-grid-s 5 --evaluations 2 \
  --cuda-target-tile 128 --cuda-targets-per-thread 2
```

## Existing representation

`UniformTree` constructs `list1` as the clipped, sorted 3 x 3 x 3
neighbourhood of every leaf, including the leaf itself. `UniformFmm` visits
occupied target leaves, then each target, each `list1` leaf, and each source in
that leaf. `build_static_p2p_operator` sorts those pairs by `(target, source)`.

`StaticP2POperator` remains the backend-independent mathematical source of
truth. `row_offsets[target]` identifies a target row. Each
`StaticDipoleBlock` stores two particle indices, three potential-row
coefficients, and six symmetric field-tensor coefficients. CPU evaluation
assigns one target row to one OpenMP iteration. CUDA previously used the same
record with one thread per target. Both avoid atomics. The target index is
redundant during execution because its row already identifies it.

One interaction is a symmetric 3 x 3 matrix-vector product. Its low arithmetic
intensity and large immutable tensor stream make cache and memory traffic the
main structural cost for long rows. Hardware counters are still needed to
quantify the DRAM fraction on a particular processor.

## New execution packings

The canonical operator can generate three immutable portable packings:

```text
StaticP2POperator (canonical AoS)
    +-- StaticP2PCompactPlan: rows + source indices + three potential and six field streams
    +-- StaticP2PLeafPlan: target-leaf rows + source ranges + six streams
    +-- StaticP2PBsrPlan: rows + source indices + full 3 x 3 blocks
```

The particle-row packing retains the complete potential/field operator. The
field-only leaf packing omits particle indices because they are implied by
each dense target/source leaf rectangle. It supports unequal occupancies and
empty neighbours. Construction verifies that the rectangles cover every
canonical entry exactly once, and records minimum, maximum, mean, unique-count,
and uniformity occupancy statistics.

The BSR plan expands each symmetric tensor to a row-major full 3 x 3 block.
Because a sparse-library call cannot apply a changing per-row self mask, self
blocks selected by `target_source_indices` are replaced by zero blocks at plan
construction. The identity map is therefore immutable for BSR execution and a
different map requires rebuilding the plan.

The isolated benchmark compares the portable BSR loop and, with
`CDFMM_ENABLE_MKL=ON`, a oneMKL sparse matrix constructed, hinted, and
optimised once. Known BSR storage is 76 bytes per interaction plus row offsets
and the fixed identity map. Opaque library analysis storage is not included.

## CUDA experiments

`CudaP2PPlan` now accepts each of the four representations. All immutable
arrays are uploaded at construction and remain resident on the device:

- `cuda-canonical-aos` retains the previous one-thread-per-target baseline;
- `cuda-particle-row-soa` removes the redundant target index and reads six
  separate tensor streams;
- `cuda-leaf-block-compact` assigns one CUDA block to an occupied target leaf,
  batches up to 128 source moments through shared memory, and keeps one field
  accumulator per target thread in registers;
- `cuda-cusparse-bsr3` applies the persistent full BSR matrix with
  `cusparseSbsrmv` or `cusparseDbsrmv` and block size three.

The leaf tensor is transposed once during CUDA setup from portable
target-major order to source-major order within every leaf rectangle. This
makes neighbouring target threads read neighbouring tensor coefficients. Leaf
occupancies larger than the selected 32, 64, 128, or 256-thread block are
processed in target batches, so unequal and partially occupied leaves remain
valid. Each block owns its target leaf and no atomics are used.

Passing a fixed identity map to the canonical, SoA, or leaf CUDA constructor
uploads that map once. A repeated evaluation then uploads only moments and
downloads fields. The BSR plan always has fixed identities because its self
blocks were zeroed during construction. Omitting fixed identities from the
other constructors preserves the existing dynamic-identity behaviour used by
`UniformFmm`.

The alternatives are exposed as standalone experiments and by
`benchmark_p2p --cuda`. Production `UniformFmm` derives its execution packing
from the canonical operator once during construction:

- CPU static and oneMKL FMM paths execute the particle-row SoA packing;
- CUDA partial and full paths execute cuSPARSE BSR(3) when
  `fixed_target_source_indices` is supplied and the BSR storage is no larger
  than `cuda_p2p_bsr_max_bytes`;
- CUDA retains canonical AoS rows when identities remain dynamic or the BSR
  plan exceeds that budget.

The BSR budget defaults to 20 GiB. `p2p_execution_packing()` reports the
resolved choice. The canonical operator remains resident as the mathematical
source and fallback; selecting an execution packing never rebuilds the
interaction set.

## CPU results

These Release-build measurements were taken on 20 August 2026. Times are the
best repeated isolated applications; field clearing and setup are excluded.
Checksums agree across implementations. Results are machine specific.

One thread, depth 2, 512 particles, uniform occupancy 8, 64,000 interactions:

| implementation | evaluation | speedup | persistent bytes |
|---|---:|---:|---:|
| canonical AoS | 71.85 us | 1.000 | 3,586,052 |
| particle-row SoA | 79.13 us | 0.908 | 3,330,052 |
| compact leaf block | 102.93 us | 0.698 | 3,088,772 |
| portable BSR(3) | 91.78 us | 0.783 | 4,866,052 |
| oneMKL BSR(3) | 71.58 us | 1.004 | at least 4,866,052 |

The controlled single-thread occupancy sweep from 1 through 64 retained the
canonical AoS lead. At occupancy 64 it took 302.66 us for 262,144 interactions;
SoA took 325.50 us, leaf blocking 357.98 us, and portable BSR 434.59 us. Leaf
storage was about 14% smaller than canonical, while BSR was about 36% larger.

For an irregular depth-2 cloud (512 particles, mean occupancy 8, maximum 13,
12 unique occupancies), canonical took 74.54 us, SoA 82.59 us, leaf blocking
118.27 us, and portable BSR 96.68 us for 66,998 interactions.

Eight-thread results were affinity-sensitive. With `OMP_PROC_BIND=close` and
`OMP_PLACES=cores`, canonical narrowly won at occupancy 16. At occupancy 32,
SoA was 1.47x faster than canonical (0.726 ms versus 1.066 ms for 1,024,000
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
