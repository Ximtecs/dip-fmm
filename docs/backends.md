# Execution backends

All FMM production backends consume one canonical CPU-built static plan.
Backend selection changes where and how an operator is applied; it does not
change the tree, interaction partition, operator mathematics, or requested
precision. `ExecutionBackend::Auto` resolves to `CpuStatic` and never replaces
an FMM request with an all-to-all direct calculation.

Every `UniformFmm` construction prints the requested options and resolved
execution choices to standard output. Because this happens in the C++ core,
the same initialisation summary is emitted for direct C++, Python, C ABI, and
Fortran callers. In particular, it reports the resolved per-operator executors
and P2P packing rather than leaving `Auto` ambiguous.

## Operator placement

| Public selection | P2M | M2M | M2L | L2L | L2P | P2P | Device residency |
|---|---|---|---|---|---|---|---|
| `CpuReference` | CPU reference | CPU reference | CPU reference | CPU reference | CPU reference | CPU direct `list1` | Host |
| `CpuStatic` + `Portable` | CPU static (procedural for point models) | CPU static | CPU class-sorted blocks | CPU static | CPU static (procedural for point models) | CPU point geometry (point sources and targets, free-space or periodic) / SoA tensor | Host |
| `CpuStatic` + `OneMkl` | CPU static (procedural for point models) | CPU static | oneMKL SGEMM/DGEMM | CPU static | CPU static (procedural for point models) | CPU point geometry (point sources and targets, free-space or periodic) / SoA tensor | Host |
| `CudaPartial` | CPU static (procedural for point models) | CPU static | CUDA target rows | CPU static | CPU static (procedural for point models) | CUDA point geometry (FP32 point sources and targets) / CUDA static tensor | Static GPU data; expansion state crosses at the M2L boundary |
| `CudaFull` | CUDA static (procedural for FP32 point models) | CUDA static | CUDA target rows | CUDA static | CUDA static (procedural for FP32 point models) | CUDA point geometry (FP32 point sources and targets) / CUDA static tensor | Operators and coefficient state remain on device |
| `DenseDirectPlan` | — | — | — | — | — | CPU dense exact | Host geometry tensors |
| `CudaDirectPlan` | — | — | — | — | — | CUDA dense exact | Persistent device geometry and scratch |

### CUDA execution policy

Within a selected CUDA backend the implementation strategy is resolved once,
deterministically, by `src/backend/cuda/execution_policy.{hpp,cpp}` from
facts of the constructed plan — precision, source geometry, periodicity,
whether a fixed identity map exists, the number of occupied target leaves and
the mean targets per leaf, the list-1 pair and M2L translation counts, the
BSR(3) size estimate and budget — together with the options. Nothing is
measured at run time, and the choice never changes the mathematical result.

| Situation | List-1 P2P packing | Dictionary executor |
|---|---|---|
| explicit `p2p_packing` (valid for the plan) | the requested packing | `cuda_dictionary_target_owned` > `cuda_dictionary_power2_microtiles` > source-warp |
| explicit `use_reduced_symmetry_p2p` (valid) | signed tensor dictionary | `cuda_dictionary_target_owned` > `cuda_dictionary_power2_microtiles` > source-warp |
| FP32 plan, point sources and point targets (geometry or near-field model), any layout | position-based point kernel (`PointGeometry`: sorted positions and list-1 records, no pair tensors) | — |
| `spatial_layout = RegularGrid`, any other plan (FP64 points need a fixed identity map), built dictionary with one- or two-byte tokens | signed tensor dictionary | explicit executor option if set; otherwise power-of-two microtiles below 48 targets per leaf, target-owned from 48 to below 72, source-warp from 72 upwards |
| `RegularGrid` whose built dictionary needs four-byte tokens (more than 65535 variants) | falls back to the `General` rule below | — |
| `General`, any geometry | dense leaf blocks (one warp per leaf pair) | — |

Periodicity enters no row: the leaf packing carries the canonical identity
marker, so finite self tensors and point self exclusion execute through the
same stored-tensor kernels, and periodic image records are ordinary dense leaf
pairs (tagged with their image ordinal), merged BSR blocks, additional row
entries or, for the position-based kernel, records carrying their image
shift. Geometry enters only through the point-pair rule: recomputing the
point-dipole formula from positions is a representation available to point
sources and point targets alone, and it is selected for FP32 plans because it
was measured faster than every stored packing there (1.3-7.5x faster than
leaf blocks on random points from 8 to 128 per leaf, faster than or equal to
the lattice dictionary in evaluation time) with 2-33x less device memory,
while FP64 recomputation is 1.7-3x slower than streaming on this GPU (see
`docs/static-p2p.md`).
 cuSPARSE BSR(3) and canonical target rows
remain explicit packings; `cuda_p2p_bsr_max_bytes` is retained for source
compatibility and no longer steers the automatic policy (leaf blocks were
measured faster than BSR(3) on finite bodies as well as on points). A packing
that cannot execute a plan (BSR or the dictionary for point sources without
`fixed_target_source_indices`, CPU-only packings on CUDA) is rejected at
construction with that reason when requested explicitly. Explicit options
take precedence over the layout hint, and the hint only fills in what was
left unspecified; a hint on geometry that does not compress costs
construction time only. `SpatialLayout::RegularGrid`
is a performance hint for lattices of identical bodies (points, prisms or
tetrahedra) with repeated displacement tensors. On irregular coordinates the
derived dictionary would need four-byte tokens, so the plan releases it and
keeps the general default. CPU backends apply the same lattice rule: a
`RegularGrid` point plan with a fixed identity map or a finite lattice runs
the CPU signed dictionary instead of the position-based or SoA executors. The same
module owns the grouped-M2L pairs-per-thread rule (16 for FP32 plans with at
least 250k translations, otherwise 8) and the M2M/L2L lane-group rule (32
lanes per output for levels with at most 65536 outputs, otherwise 4). The
resolved choices appear in the initialisation summary as `spatial_layout`
and `cuda_policy.*`. Neither the hint nor the derived packing enters the
persistent geometry cache; the cached canonical operator is shared.

### CPU execution packing

`CpuStatic` executes the far-field hierarchy from a packing derived once at
construction from the canonical operators (dense P2M rows, level-scaled M2M/L2L
column banks, flat L2P rows). The portable M2L applies the canonical transfer
matrices through a block schedule sorted by transfer class when the matrix set
exceeds 1 MiB, and per target row otherwise. For plans with point sources and
point targets (geometry or near-field model), free-space or periodic, and no
explicit reduced-symmetry request, list-1 P2P is
`P2PExecutionPacking::PointGeometry`: pairs are recomputed from the sorted
positions (with each periodic record's image shift) using the same
point-dipole formula as the reference kernel, and no pair tensors are kept
resident. Finite near fields keep the particle-row SoA tensors by default; the
signed dictionary is the explicit `use_reduced_symmetry_p2p` choice, and
`p2p_packing` can force `CanonicalAos`, `ParticleRowSoa`, `TensorDictionary`
(any geometry) or `PointGeometry` (point pairs). None
of these packings enters the persistent cache; the cached canonical operators
are unchanged. See [static P2P](static-p2p.md) for the capability matrix and
the execution invariant behind it.

### Point P2M and L2P execution

The point-source P2M and point-target L2P operators have two execution
strategies, selected by `UniformFmmOptions::point_expansion_execution`
(`PointExpansionExecution::Auto`, `Precomputed`, `Procedural`). `Precomputed`
streams the coefficient rows built at construction (`3 C` scalars per source
and per target, `C = (p + 1)^2`; 588 bytes per point at `p = 6` in FP32).
`Procedural` recomputes the operator from the sorted positions during every
evaluation with the allocation-free solid-harmonic recurrence in
`src/math/solid_harmonic_recurrence.hpp` and the shared kernels in
`src/operators/point_expansion_kernel.hpp`; the plan then retains three
`C`-entry factor tables instead of the rows. It exists for the spherical basis
at orders 1 to 10; the Cartesian basis keeps its precomputed rows, and a stage
whose far-field model is a finite body (prism or tetrahedron P2M or L2P) keeps
its exact precomputed rows in every mode. An explicit `Procedural` request
that no stage can honour throws `std::invalid_argument` at construction.

`Auto` follows the Phase-3B.5 measurements (`agent_docs/
performance_optimization.md`): the CPU hierarchy (`CpuStatic` and the CPU
stages of `CudaPartial`) recomputes both operators in FP32 and FP64, where
the procedural stages were faster from `p = 4` upwards and several times
faster at `p >= 6`; `CudaFull` recomputes them for FP32 plans (3-7x faster
P2M/L2P kernels from `p = 6`, equal within a few microseconds at `p = 4`,
about 100 MB less device memory at 50k points and `p = 6`) and keeps the
streamed rows for FP64 plans, whose device recurrence is slower than the
rows. The initialisation summary prints `point_expansion.requested`,
`p2m_execution` and `l2p_execution`; `UniformFmm::p2m_execution()` and
`l2p_execution()` report the resolved choice, and
`StaticPlanStatistics::p2m_operator_bytes` / `l2p_operator_bytes` report the
factor tables for a procedural stage. The result is identical for every
strategy up to rounding; the persistent cache stores the canonical operators
regardless, so cache format and keys are unchanged.

`Portable` and `OneMkl` in the table are values of `StaticMatrixBackend`.
oneMKL accelerates M2L only: interactions sharing a normalised transfer matrix
are gathered into columns, multiplied with SGEMM or DGEMM, and scattered to
target locals. The other stages retain the portable static executors.
The persistent grouping and scratch plus all oneMKL calls are internal to
`src/backend/mkl`; high-level P2M/M2M/M2L-dispatch/L2L/L2P sequencing remains
in `src/fmm`.

`CpuReference` is the independent Cartesian mathematical traversal used for
validation and education. `CpuStatic` is the portable production default.
oneMKL, CUDA partial, and CUDA full are production-capable optional builds and
also remain explicit benchmark selections. The two direct plans are exact
$O(N^2)$ references, not FMM backends.

## Basis and geometry support

Point-dipole Cartesian and real spherical plans support CPU static, oneMKL,
CUDA partial, and CUDA full execution in FP32 or FP64. `CpuReference` is
Cartesian-only because it forms dynamic Cartesian derivative contractions
instead of consuming the spherical static payload.

`UniformFmm` supports point, rectangular-prism, and tetrahedron sources and
point, rectangular-prism, and tetrahedron targets with either expansion basis
on static plans; all nine near-field pairs execute on `CpuStatic`,
`CudaPartial`, and `CudaFull` in FP32 and FP64 through every stored-tensor
packing (`tests/test_p2p_geometry_matrix.cpp`).
Exact P2P follows the selected physical geometry, while the comparison flags
can substitute point P2M or point L2P. `DenseDirectPlan` provides the matching
exact geometries as an independent reference.

Fully periodic cubic zero-`k=0` plans use the same static CPU and CUDA
placements. The CPU-reference backend is unavailable because it does not
consume image-aware list1/list2 plans or the Ewald root periodiser. See [Fully
periodic boundary conditions](periodic-boundaries.md).

CUDA-full is the field-only device-resident FMM path. CPU static and the hybrid
path retain the supported potential calculation; hybrid potential uses the CPU
near-field calculation because its cached CUDA P2P tensor stores field rows.
Unsupported combinations fail explicitly.

## CUDA partial data flow

```text
CPU: moments -> sort -> P2M -> M2M
                              |
                              +-> multipoles H2D -> CUDA M2L -> locals D2H

moments H2D -> CUDA P2P -> near field D2H       (independent stream)

CPU: raw locals -> L2L -> L2P -> far field
CPU: far field + near field -> target unsorting
```

Static M2L matrices, interaction metadata, scaling tables, and the selected P2P
packing are uploaded during plan construction. M2L and P2P may overlap; their
phase timings therefore are not a sequential sum. The M2L stream is created at
the device's greatest stream priority: when the two overlap the CPU is waiting
for exactly the M2L result, so its short kernel is scheduled ahead of the
SM-saturating P2P kernel instead of behind it (measured 4-11 % faster
evaluations at 32-128 points per leaf).

## CUDA full data flow

```text
changing moments -> H2D
    -> sort -> P2M -> M2M -> M2L -> L2L -> L2P
                    +-------------------------> P2P
    -> far + near -> unsort -> final field -> D2H
```

Geometry-dependent operators, permutations, identity metadata, scaling tables,
and persistent scratch are uploaded once. A repeated evaluation uploads only
the changing moments and downloads only the final user-ordered field. The
fixed target/source identity map is part of the plan; changing it requires a
new evaluator. The far-field stream outranks the near-field stream when the
execution policy estimates the far field at less than three times the P2P
kernel (`cuda_policy.far_field_stream_priority` in the initialisation
summary; the P2P estimate uses the measured cost per pair of the resolved
packing and precision), so the many short far-field kernels are not stretched
behind the P2P kernel; when a deep tree makes the far field dominate, both streams keep
equal priority so the small P2P kernel can hide inside it.

## Compatibility names and availability

`CudaM2LP2P` is the canonical enum value behind `CudaPartial`.
`CudaM2L` and `CudaM2LStaticP2P` are compatibility aliases for that same
hybrid implementation, not separate backends. `cuda_m2l_available()` is
likewise retained as an alias for `cuda_m2l_p2p_available()`.

CUDA compilation and runtime device availability are separate. Capability
queries report both, and requesting an unavailable backend raises an error.
The standalone CUDA direct reference remains separately named so its
quadratic algorithm cannot be mistaken for a fallback FMM.

The C ABI and native Fortran wrapper expose `cdfmm_one_mkl_available()` so
host applications can implement an explicit oneMKL-to-portable CPU fallback
before constructing a plan.
