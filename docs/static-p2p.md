# Static P2P execution study

## Execution invariant: geometry builds tensors, executors apply tensors

Every exact near-field interaction is applied as `H_t += T_{t,s} m_s` with a
pair tensor `T` that is constructed once from the physical geometry. The
construction is the only geometry-specific step:

```text
physical geometry (point, rectangular prism, tetrahedron)
    -> geometry-specific pair-tensor construction (src/operators/p2p.cpp,
       src/geometry/primitives/*)
    -> canonical P2P interaction tensors and topology
       (StaticP2POperator: target row, source index, six tensor components,
        identity marker; StaticP2PLeafRecord: dense leaf pairs, image shift)
    -> geometry-independent execution packings
       (SoA rows, dense leaf blocks, signed tensor dictionary, BSR(3))
    -> CPU / CUDA executors
```

Once the canonical operator exists, no packing builder or executor reads a
`SourceGeometry`, `TargetGeometry`, prism size or tetrahedron record; they see
tensor components, indices, leaf ranges, image ordinals and identity markers
only. Self-interaction semantics travel with the tensors: the canonical rows
mark a pair with `skip_for_identity` when the source is a point dipole (its
singular self pair is omitted whenever the identity map names it), and leave
the marker clear for a finite source, whose coincident self tensor is a
physical demagnetisation field and is applied like any other stored value.
Every dense leaf block carries the same marker, the signed dictionary encodes
its zero variant only for marked blocks, and the BSR builder zeroes only marked
blocks, so an executor never infers identity handling from geometry.

The one deliberate exception is `P2PExecutionPacking::PointGeometry`: a fused
geometry evaluation for point sources and point targets that recomputes the
point-dipole formula from the resident sorted positions instead of streaming
stored tensors. It exists on the CPU (a gathered per-leaf sweep) and, since
Phase 3B.5, on the CUDA backends (one warp per canonical list-1 record, with
the same lane layout as the leaf-block kernel); both call the single formula
in `src/operators/p2p_point_kernel.hpp`. It is not a tensor executor and is
rejected for finite near-field geometry; finite bodies keep precomputed
tensors because evaluating analytical prism or tetrahedron tensors on every
evaluation would cost far more than streaming them.

### Precomputed and procedural representations

The invariant separates the mathematical operator from how an executor holds
it. Every near-field or far-field operator has two possible representations:

```text
physical operator
    |
    +--> precomputed representation   tensor rows, coefficient rows,
    |                                 compressed packings (built once)
    |
    +--> procedural representation    the mathematically identical cheap
                                      point operator reconstructed from the
                                      positions during every evaluation
```

Precomputation is an execution choice, not a mathematical requirement. For
finite tiles it remains the production strategy: the exact prism and
tetrahedron operators are expensive to construct and cheap to stream. For
point sources and point targets the operator is a closed formula (P2P) or a
short recurrence (the regular solid harmonics of P2M and L2P) of positions
that stay cache resident, so recomputing it can beat streaming its stored
form. Three point operators have a procedural representation, each selected
by a measured policy with an explicit override (`docs/backends.md`):

| Operator | Precomputed form | Procedural form | Default |
|---|---|---|---|
| point P2P | pair tensors (24-53 bytes per pair) | `PointGeometry`, positions plus list-1 records | CPU: always; CUDA: FP32 plans |
| point P2M | dense rows, `3 C` scalars per source | recurrence per source, `C`-entry factor table | CPU hierarchy: always; CudaFull: FP32 plans |
| point L2P | dense rows, `3 C` scalars per target (plus `C` for the potential) | recurrence per target, factor tables | CPU hierarchy: always; CudaFull: FP32 plans |

The procedural point P2M/L2P executors (`UniformFmmOptions::
point_expansion_execution`) exist for the spherical basis at orders 1 to 10;
the Cartesian basis and finite far-field models keep their precomputed rows.

### Capability matrix

All nine source/target combinations of point, rectangular prism and
tetrahedron build canonical tensors (`docs/geometry-models.md`). Every
stored-tensor packing then executes any of them; the table lists the
combinations a production backend accepts, with the reason for each
intentional exclusion. "any pair" means all nine geometry pairs, free-space or
periodic.

| Backend | Packing | Geometry pairs | Requirement / reason |
|---|---|---|---|
| `CpuStatic` (Portable, oneMKL) | `CanonicalAos` | any pair | none |
| `CpuStatic` | `ParticleRowSoa` | any pair | none (default for finite near fields on a `General` layout) |
| `CpuStatic` | `TensorDictionary` | any pair | point sources need `fixed_target_source_indices` (the self pair is encoded at construction); automatic on a `RegularGrid` layout whose built dictionary has one- or two-byte tokens |
| `CpuStatic` | `PointGeometry` | point -> point only (free-space or periodic) | the fused point executor and the default for point pairs; finite near-field geometry is rejected explicitly |
| `CudaPartial`, `CudaFull` | `CanonicalAos` | any pair | none (explicit only) |
| `CudaPartial`, `CudaFull` | `LeafBlock` | any pair | none (the general default for every geometry) |
| `CudaPartial`, `CudaFull` | `CudaBsr3` | any pair | point sources need `fixed_target_source_indices` (explicit only; `cuda_p2p_bsr_max_bytes` no longer steers the policy) |
| `CudaPartial`, `CudaFull` | `TensorDictionary` (source-warp, target-owned, power-of-two microtiles) | any pair | point sources need `fixed_target_source_indices`; automatic on a `RegularGrid` layout whose built dictionary has one- or two-byte tokens |
| `CudaPartial`, `CudaFull` | `PointGeometry` | point -> point only (free-space or periodic) | the position-based kernel and the default for FP32 point pairs; fixed or changing identity maps; finite near-field geometry is rejected explicitly |
| `CudaPartial`, `CudaFull` | `ParticleRowSoa` | none | CPU row packing; requested explicitly it fails with that reason |
| `CpuReference` | `Reference` | point -> point only | the reference backend forms dynamic Cartesian contractions and has no finite P2M/L2P; it rejects exact finite stages explicitly, which does not restrict the static operator |

Periodic image records are ordinary stored tensors: the per-pair packings keep
one row entry per image, the dense leaf packings one block per image (a leaf
pair carries its image ordinal among the records sharing its ranges), the BSR
builder sums the images of a `(target, source)` pair into one block, and the
point-geometry executor folds each record's image shift into the gathered
neighbourhood. Periodicity therefore restricts no packing.

### Explicit selection

`UniformFmmOptions::p2p_packing` (default `P2PExecutionPacking::Auto`) forces
one packing and takes precedence over `spatial_layout`,
`use_reduced_symmetry_p2p` and the BSR memory budget; the dictionary executor
follows `cuda_dictionary_target_owned` / `cuda_dictionary_power2_microtiles`.
A request the backend or plan cannot execute throws `std::invalid_argument`
at construction naming the representational reason above. The initialisation
summary prints `p2p_packing.requested` and the resolved `p2p_packing`;
`requested_p2p_packing()` and `p2p_execution_packing()` expose both. The
table above is enforced by `tests/test_p2p_geometry_matrix.cpp`, which runs
every pair through every packing of every available backend in FP32 and FP64
against the FP64 `DenseDirectPlan` reference.

## Signed Tensor6 dictionary packing

The P2P execution plan retains the exact canonical interaction topology as
dense target/source leaf blocks. Each interaction stores a direct ID into an
already-signed Tensor6 execution dictionary. IDs use one, two, or four bytes
according to the signed variant count; no sign reconstruction occurs during
execution. Tokens remain source-major within each leaf pair, and signed
variants are ordered by descending use frequency without changing interaction
or source order. The dictionary is six structure-of-arrays component vectors.
FP32 is quantised and deduplicated independently.

This packing does not inspect particle coordinates or attempt to detect a
regular grid. It applies to point, prism and tetrahedron P2P operators alike,
free-space or periodic. Fixed point self interactions use an exact zero
dictionary variant (encoded only for blocks whose canonical rows carry the
identity marker); finite self interactions are retained.

The dictionary is selected when `p2p_packing` requests it, when
`use_reduced_symmetry_p2p` is enabled, or automatically by the CUDA execution
policy when `UniformFmmOptions::spatial_layout` is `SpatialLayout::RegularGrid`
for a point-source plan with a fixed identity map (see
[backends](backends.md)). The reported plan statistics
include tensor counts and canonical/dictionary persistent-memory fields for
benchmark inspection. Its CPU executor owns disjoint target tiles with static
OpenMP scheduling and traverses sources using register-resident SIMD
microtiles. The OpenMP tile defaults to 32 targets and is runtime-configurable;
the SIMD width is independent of that tile.

When reduced symmetry is explicitly requested for `cuda_m2l_p2p` or
`cuda_full`, the same signed host dictionary is uploaded as six contiguous SoA
component ranges followed by the active one-, two-, or four-byte token stream.
CUDA builds their own 128-target tiles rather than reusing the CPU tile size.
One 128-thread block owns each `(target leaf, target tile)` item, one thread
owns one target, and source moments are staged in shared memory in batches of
128. The dictionary kernel consumes the already-encoded point-self zero
variant (or physical cuboid self tensor), so it has no runtime identity or
geometry branch. Without the explicit reduced-symmetry option or the regular-grid layout hint,
the CUDA leaf-block/BSR/canonical selection policy is unchanged.

## Sweep-based recommendation

The latest complete sweep covered 19 runnable `(particle count, depth)` cases
from 2^12 through 2^17 particles and depths two through five. CPU SoA won
12 cases, was within 10% of the winner in 15, and had a 1.154 geometric-mean
speedup over canonical AoS. It became the portable CPU default for stored
tensors. Since the Phase-3B CPU evaluation work, point-source / point-target
plans on `CpuStatic` recompute their list-1 pairs from the sorted positions
instead (`P2PExecutionPacking::PointGeometry`): the stored-tensor kernels are
DRAM-bound at 29-53 bytes per pair, while the positions of a target leaf's
neighbourhood stay cache resident. Periodic point plans joined that default in
the Phase-3 P2P unification (the executor folds each record's image shift in
and was measured 2.7-5.1x faster than the SoA rows on the periodic near
field). The SoA tensors remain the default for finite near fields (see
`docs/backends.md` and `agent_docs/performance_optimization.md`).

CUDA BSR(3) won 12 cases, was within 10% of the winner in 17, and had a 1.757
geometric-mean kernel speedup over canonical CUDA. Its full blocks use about
35% more persistent P2P storage than canonical rows, so it is the preferred
fixed-identity path subject to the explicit memory budget rather than an
unconditional choice. Canonical CUDA remains the correct default for changing
identity maps and the fallback when BSR is too large.

Leaf-block CUDA won only the high-occupancy `(N=65536, depth=4)` and
`(N=131072, depth=4)` cases in that sweep. Those two points do not yet justify
an occupancy dispatch rule, so leaf blocking remains an experimental packing.
(Phase 3A later made leaf blocks the general CUDA default; see below.)

## Position-based point P2P on CUDA (Phase 3B.5)

The CUDA `PointGeometry` kernel keeps one warp per canonical list-1 record
(target leaf, source leaf, image) and the leaf-block lane layout, but reads
one aligned position (16 bytes in FP32) and one moment per source instead of
six stored tensor components, and recomputes the pair with the shared
point-dipole formula. Measured on the RTX 5090 against the best stored
packing of each plan (`agent_docs/performance_optimization.md`, Phase 3B.5,
FP32, order 6, `cuda-full`):

- random points, depth 3, 8 to 128 points per leaf: the kernel is 1.3x (8 per
  leaf) to 7.5x (128 per leaf) faster than the leaf blocks and the evaluation
  1.0-6.2x faster, with 2-33x less persistent device memory (14-144 MB
  instead of 32-4747 MB);
- regular lattices: in evaluation time it is faster than or equal to the
  dictionary at every measured occupancy (8, 16, 32 and 64 per leaf, 4096 to
  262144 points); the dictionary kernel alone is still up to 2x faster at 8
  points per leaf on a 262k lattice, where the far field dominates the
  evaluation and the two agree within 1.3 %;
- FP64: recomputation is 1.7-3x slower than the leaf blocks and 2x slower than
  the dictionary, because the consumer GPU's FP64 rate is a small fraction of
  its FP32 rate.

The CUDA execution policy therefore selects `PointGeometry` for FP32 plans
with point sources and point targets on any layout, and keeps the leaf blocks
and the lattice dictionary for FP64 point plans and for finite bodies. The
stored packings remain explicit choices (`p2p_packing`,
`use_reduced_symmetry_p2p`). Resident memory of the procedural plan is the
sorted positions (16 or 32 bytes per point), one 32- or 48-byte record per
list-1 leaf pair and the identity map.

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
