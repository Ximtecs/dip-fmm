# Execution backends, precision and execution options

All FMM production backends consume one canonical CPU-built static plan.
Backend selection changes where and how an operator is applied; it does not
change the tree, the interaction partition, the operator mathematics, or the
requested precision. `ExecutionBackend::Auto` resolves to `CpuStatic` and
never replaces an FMM request with an all-to-all direct calculation. Tutorial
3 measures every choice on this page.

Every `UniformFmm` construction prints the requested options and the resolved
execution choices to standard output. Because this happens in the C++ core,
the same initialisation summary is emitted for direct C++, Python, C ABI, and
Fortran callers; it reports the resolved per-operator executors and P2P
packing rather than leaving `Auto` ambiguous, and the same values are
available through `backend()`, `execution_plan()`, `p2p_execution_packing()`,
`p2m_execution()`, `l2p_execution()` and the statistics.

## Operator placement

| Public selection | P2M | M2M | M2L | L2L | L2P | P2P | Device residency |
|---|---|---|---|---|---|---|---|
| `CpuReference` | CPU reference | CPU reference | CPU reference | CPU reference | CPU reference | CPU direct `list1` | host |
| `CpuStatic` + `Portable` | CPU static (procedural for point models) | CPU static | CPU class-sorted blocks | CPU static | CPU static (procedural for point models) | CPU point geometry (point pairs, free-space or periodic) / SoA tensor | host |
| `CpuStatic` + `OneMkl` | CPU static (procedural for point models) | CPU static | oneMKL SGEMM/DGEMM | CPU static | CPU static (procedural for point models) | as above | host |
| `CudaPartial` | CPU static (procedural for point models) | CPU static | CUDA target rows | CPU static | CPU static (procedural for point models) | CUDA point geometry (FP32 point pairs) / CUDA static tensor | static GPU data; expansion state crosses at the M2L boundary |
| `CudaFull` | CUDA static (procedural for FP32 point models) | CUDA static | CUDA target rows | CUDA static | CUDA static (procedural for FP32 point models) | CUDA point geometry (FP32 point pairs) / CUDA static tensor | operators and coefficient state stay on the device |
| `DenseDirectPlan` | — | — | — | — | — | CPU dense exact (portable or oneMKL GEMV) | host geometry tensors |
| `CudaDenseDirectPlan`, `CudaDirectPlan` | — | — | — | — | — | CUDA dense exact | persistent device geometry and scratch |

`Portable` and `OneMkl` are values of `StaticMatrixBackend`. oneMKL
accelerates M2L only: interactions sharing a normalised transfer matrix are
gathered into columns, multiplied with SGEMM or DGEMM, and scattered to target
locals; the other stages keep the portable executors, and the grouping,
scratch and vendor calls are internal to `src/backend/mkl`. Portable CPU M2L
is the default; oneMKL is explicit because it wins on some problem shapes and
not on others.

`CpuReference` is the independent Cartesian mathematical traversal used for
validation and teaching: it regenerates the kernel derivatives of every M2L
interaction during the evaluation, so one evaluation costs seconds where a
static plan costs a millisecond. It is Cartesian-only, accepts point geometry
only and has no periodic plan. `CudaM2LP2P` is the canonical enum value behind
`CudaPartial`; `CudaM2L` and `CudaM2LStaticP2P` are compatibility aliases of
the same hybrid backend, and `cuda_m2l_available()` is an alias of
`cuda_m2l_p2p_available()`. CUDA compilation and runtime device availability
are separate; the capability queries (`cuda_compiled`, `cuda_available`,
`cuda_direct_available`, `cuda_m2l_p2p_available`, `cuda_full_available`,
`one_mkl_available`, also exposed through the C ABI and Fortran) report both,
and requesting an unavailable backend raises. The standalone CUDA direct
reference is separately named so its quadratic algorithm cannot be mistaken
for a fallback FMM.

`CudaFull` is the field-only device-resident path; CPU static and the hybrid
path retain the scalar potential (hybrid potential uses the CPU near-field
calculation). Unsupported combinations fail explicitly at construction.

## Data flow

```text
CUDA partial
CPU: moments -> sort -> P2M -> M2M
                              +-> multipoles H2D -> CUDA M2L -> locals D2H
moments H2D -> CUDA P2P -> near field D2H       (independent stream)
CPU: raw locals -> L2L -> L2P -> far field
CPU: far field + near field -> target unsorting

CUDA full
changing moments -> H2D
    -> sort -> P2M -> M2M -> M2L -> L2L -> L2P
                    +-------------------------> P2P
    -> far + near -> unsort -> final field -> D2H
```

Static M2L matrices, interaction metadata, scaling tables and the selected P2P
packing are uploaded during construction. In the partial backend M2L and P2P
may overlap, so their phase timings are not a sequential sum; the device
phase timings of both CUDA backends are recorded only at
`TimingLevel::Detailed`, and below it only the three functional events (the
two cross-stream dependencies and the completion point, created without
timestamps) remain ([Benchmarks and profiling](benchmarks.md)); its M2L stream
is created at the greatest stream priority because the CPU waits for exactly
that result (measured 4–11 % faster at 32–128 points per leaf). In the full
backend a repeated evaluation uploads only the changing moments and downloads
only the user-ordered field; the far-field stream outranks the near-field
stream when the policy estimates the far field at less than three times the
P2P kernel (`cuda_policy.far_field_stream_priority` in the summary), otherwise
both keep equal priority so the small P2P kernel can hide inside the far
field. A fixed identity map is part of a full plan; changing it needs a new
evaluator.

## Precision

`StaticPrecision::Float32` is the default; `Float64` selects double
precision. For `UniformFmm` this is an execution choice, not a storage-only
choice:

| Boundary | FP32 plan | FP64 plan |
|---|---|---|
| caller input | FP32 or FP64 arrays | FP32 or FP64 arrays |
| analytical construction | FP64 temporaries | FP64 |
| retained static operators | FP32 only | FP64 only |
| multipole/local state, scratch, near/far fields | FP32 | FP64 |
| CPU arithmetic / BLAS | typed loops / SGEMM | typed loops / DGEMM |
| CUDA buffers and arithmetic | FP32 | FP64 |
| Python result and coefficient dtype | `numpy.float32` | `numpy.float64` |

Positions, root metadata and analytical operator construction are formed in
FP64; an FP32 plan quantises each completed operator once and releases its
FP64 temporaries, retaining no hidden double-precision table. Moments are
converted once at the evaluation boundary. The C++ `evaluate`, `multipole`,
`local` and `root_multipole` compatibility methods return double-valued
objects and widen FP32 results at the public boundary; the explicitly typed
`*_float32`/`*_float64` counterparts reject a plan of the wrong precision, and
Python exposes the native dtype. FP32 plans scale coordinates and moments by
the root width internally so that nanometre geometry stays representable
([Mathematical conventions](math/conventions.md)). `DenseDirectPlan` applies
the same scalar to its six matrices and quantises each tensor at the point of
store, so an FP32 dense plan never materialises an FP64 matrix; its analytical
prism construction stays in extended precision because its logarithm, inverse
hyperbolic sine, arctangent and square-root formulas are cancellation
sensitive. There is deliberately no mixed FP32-static/FP64-state mode. Host
and device statistics report the actual retained scalar widths
(`scalar_bytes`).

## Near-field packings

The exact near-field operator is one canonical set of pair tensors; a packing
is the derived layout an executor streams
([Architecture](architecture.md)). Every stored-tensor packing executes any
of the nine geometry pairs, free-space or periodic; the table lists what each
production backend accepts and why the exclusions exist.

| Backend | Packing | Geometry pairs | Requirement / reason |
|---|---|---|---|
| `CpuStatic` (portable, oneMKL) | `CanonicalAos` | any | none |
| `CpuStatic` | `ParticleRowSoa` | any | none; the default for finite near fields on a `General` layout |
| `CpuStatic` | `TensorDictionary` | any | point sources need `fixed_target_source_indices` (the self pair is encoded at construction); automatic on `RegularGrid` when the built dictionary has one- or two-byte tokens |
| `CpuStatic` | `PointGeometry` | point → point only | the fused point executor and the default for point pairs, free-space or periodic; finite geometry is rejected; the plan builds no pair tensors, so construction memory does not grow with the list-1 pair count |
| `CudaPartial`, `CudaFull` | `CanonicalAos` | any | explicit only |
| `CudaPartial`, `CudaFull` | `LeafBlock` | any | the general default for every geometry (one warp per leaf pair) |
| `CudaPartial`, `CudaFull` | `CudaBsr3` | any | point sources need `fixed_target_source_indices`; explicit only |
| `CudaPartial`, `CudaFull` | `TensorDictionary` (source-warp, target-owned or power-of-two microtiles) | any | point sources need `fixed_target_source_indices`; automatic on `RegularGrid` with one- or two-byte tokens |
| `CudaPartial`, `CudaFull` | `PointGeometry` | point → point only | the position-based kernel and the default for FP32 point pairs; fixed or changing identity maps |
| `CudaPartial`, `CudaFull` | `ParticleRowSoa` | none | a CPU row packing; requested explicitly it fails with that reason |
| `CpuReference` | `Reference` | point → point only | dynamic Cartesian contractions, no finite operators |

### The automatic policy

Within a selected backend the representation is resolved once,
deterministically, from facts of the constructed plan (precision, effective
point sources and targets, whether a fixed identity map exists, the mean
targets per occupied target leaf, the list-1 pair and M2L translation counts)
and the options. Nothing is measured at run time and no choice changes the
result. Neither periodicity nor the BSR(3) memory budget is a policy input:
periodic image records pack like any other pair, and the automatic policy has
not selected BSR(3) since leaf blocks became the general default.

| Situation | List-1 P2P packing | Dictionary executor |
|---|---|---|
| explicit `p2p_packing` (valid for the plan) | the requested packing | `cuda_dictionary_target_owned` > `cuda_dictionary_power2_microtiles` > source-warp |
| explicit `use_reduced_symmetry_p2p` (valid) | signed tensor dictionary | as above |
| FP32 point pairs on CUDA, any layout | `PointGeometry`: sorted positions and list-1 records, no pair tensors | — |
| `RegularGrid` layout, any other plan (point sources need a fixed identity map), built dictionary with one- or two-byte tokens | signed tensor dictionary | explicit executor flag if set; otherwise power-of-two microtiles below 48 targets per leaf, target-owned from 48 to below 72, source-warp from 72 upwards |
| `RegularGrid` whose built dictionary needs four-byte tokens (more than 65535 variants) | falls back to the `General` rule | — |
| point sources and point targets on the CPU, otherwise (`General`, no fixed identity map, or four-byte dictionary tokens) | `PointGeometry`: sorted positions and list-1 records, no pair tensors | — |
| `General`, finite geometry or FP64 points on CUDA | dense leaf blocks | — |
| `General`, finite geometry on the CPU | particle-row SoA tensors | — |

The reasons are measured, not assumed: recomputing point pairs from positions
was 1.3–7.5× faster than leaf blocks on random FP32 points with 2–33× less
device memory and faster than or equal to the lattice dictionary, while FP64
recomputation is 1.7–3× slower than streaming on a consumer GPU; leaf blocks
beat BSR(3) on finite bodies once the leaf packing carried the identity
metadata; and a dictionary is only profitable while it compresses to narrow
tokens (32768 irregular bodies gave 3.47 M variants and 1.9–3.6× slower
kernels, a lattice 187–344 variants and 3× faster). `SpatialLayout::RegularGrid`
is therefore a **hint** for lattices of identical bodies: the result is the
same either way, and a hint on geometry that does not compress costs
construction time only. The same module owns the grouped-M2L pairs-per-thread
rule (16 for FP32 plans with at least 250 k translations, otherwise 8) and the
M2M/L2L lane-group rule (32 lanes per output for levels with at most 65536
outputs, otherwise 4); the resolved choices appear as `spatial_layout` and
`cuda_policy.*` in the summary. Neither the hint nor the derived packing
enters the persistent cache. The measurements are recorded in
`agent_docs/performance_optimization.md`.

### Construction and resident memory

The near field is built one chunk of consecutive target leaves at a time
(about 4.2 M list-1 pairs per chunk), and each chunk is turned straight into
the representations the resolved plan keeps; no representation the plan
does not read is materialised. Chunking never changes a value: every pair
tensor is a pure function of its displacement and body records, and the rows
are ordered by target, so the chunked result is bitwise the one-shot result
(tetrahedron self-systems, which share each tensor with its reciprocal pair,
are handled by building the reverse pairs a chunk depends on).

| Resolved near field | Built during construction | Resident afterwards |
|---|---|---|
| `PointGeometry` (CPU or CUDA) | nothing per pair | nothing per pair |
| `TensorDictionary` | tokens, chunk by chunk | the dictionary (CPU), nothing on the host (CUDA); point plans also keep the rows their potential output reads |
| CPU `ParticleRowSoa` | SoA rows in the plan's precision | the SoA rows |
| CPU `CanonicalAos` | canonical rows in the plan's precision | the canonical rows |
| CUDA `LeafBlock` | leaf blocks in the plan's precision | nothing on the host |
| CUDA `CanonicalAos` / `CudaBsr3` | canonical rows in the plan's precision | nothing on the host |

A point near field additionally keeps the SoA rows its `OutputFlags::Potential`
evaluation reads (FP32 always, FP64 on a periodic plan) except on `CudaFull`,
which is field-only; exact finite near fields reject potential output. With
the geometry cache enabled, the FP64 canonical records are streamed into the
cache file as they are built rather than held in full, and a dictionary plan
persists the dictionary it executes (in its own precision) with the rows
beside it, so a warm construction loads every stored representation and
builds no pair tensor. A procedural point-expansion stage reads no P2M or
L2P map: those maps are built only for a shared cache file and are not kept;
`CudaFull` releases its host far-field maps after upload. The plan statistics
count what the plan keeps, and are identical for a cold, a warm and an
uncached construction.

### Explicit selection

`UniformFmmOptions::p2p_packing` (default `Auto`) forces one packing and takes
precedence over `spatial_layout`, `use_reduced_symmetry_p2p` and the BSR
memory budget; the dictionary executor follows `cuda_dictionary_target_owned`
/ `cuda_dictionary_power2_microtiles`. A request the backend or plan cannot
execute throws `std::invalid_argument` at construction naming the reason. The
summary prints `p2p_packing.requested` and the resolved `p2p_packing`;
`requested_p2p_packing()` and `p2p_execution_packing()` expose both.
`cuda_p2p_bsr_max_bytes` is retained for source compatibility and no longer
steers the automatic policy.

One caveat for an explicit dictionary request on CUDA: the layout hint picks
the dictionary executor from the measured occupancy calibration, but an
explicit `p2p_packing = TensorDictionary` or `use_reduced_symmetry_p2p` with
neither executor flag set keeps the source-warp kernel. That is the documented
meaning of "neither flag" and the only way to select that kernel, but below
the first calibration threshold it costs real time (2.59× on the P2P phase of
a 32768-point FP64 lattice at 8 targets per leaf; `cuda_dictionary_power2_microtiles`
recovers it exactly). Set the executor flag when requesting the dictionary
explicitly at low occupancy, or use `spatial_layout` and let the calibration
choose.

### The signed tensor dictionary

The dictionary packing keeps the exact canonical interaction topology as dense
target/source leaf blocks and stores, per interaction, a one-, two- or
four-byte id into an already-signed six-component dictionary; no sign
reconstruction occurs during execution, tokens are source-major within a leaf
pair, and variants are ordered by descending frequency. It inspects no
coordinates and applies to point, prism and tetrahedron operators alike. Fixed
point self interactions use an exact zero variant (encoded only for blocks
whose canonical rows carry the identity marker); finite self interactions are
retained. Its CPU executor owns disjoint target tiles with static OpenMP
scheduling (`signed_p2p_target_tile_size`, 1–128, default 32) and traverses
sources with register-resident SIMD microtiles; the CUDA executors upload the
same dictionary as six SoA component ranges plus the token stream and stage
source moments in shared memory in batches of 128.

## Point P2M and L2P execution

The point-source P2M and point-target L2P operators have two execution
strategies, selected by `UniformFmmOptions::point_expansion_execution`
(`Auto`, `Precomputed`, `Procedural`). `Precomputed` streams the coefficient
rows built at construction (`3 C` scalars per source and per target,
`C = (p+1)^2`; 588 bytes per point at `p = 6` in FP32). `Procedural`
recomputes the operator from the sorted positions during every evaluation with
the allocation-free solid-harmonic recurrence and retains three `C`-entry
factor tables instead of the rows. It exists for the spherical basis at orders
1 to 15 on the static backends (orders 11 to 15 only on explicit request: the
`Auto` policy below was measured to order 10 and stays precomputed above it);
the Cartesian basis keeps its precomputed
rows, a stage whose far-field model is a finite body keeps its exact
precomputed rows in every mode, and an explicit `Procedural` request that no
stage can honour throws `std::invalid_argument` at construction.

`Auto` follows the measurements: the CPU hierarchy (`CpuStatic` and the CPU
stages of `CudaPartial`) recomputes both operators in FP32 and FP64, where the
procedural stages were faster from `p = 4` upwards and several times faster at
`p >= 6`; `CudaFull` recomputes them for FP32 plans (3–7× faster P2M/L2P
kernels from `p = 6`, equal within microseconds at `p = 4`, about 100 MB less
device memory at 50 k points and `p = 6`) and keeps the streamed rows for FP64
plans, whose device recurrence is slower than the rows. The summary prints
`point_expansion.requested`, `p2m_execution` and `l2p_execution`;
`StaticPlanStatistics::p2m_operator_bytes` / `l2p_operator_bytes` report the
factor tables for a procedural stage. The result is identical for every
strategy up to rounding, and the cache stores the canonical operators
regardless.

## Basis and geometry support

Point-dipole Cartesian and real spherical plans support `CpuStatic`, oneMKL,
`CudaPartial` and `CudaFull` in FP32 or FP64. All nine near-field pairs of
point, rectangular-prism and tetrahedron sources and targets execute on every
production backend in both precisions through every stored-tensor packing;
the exact P2P follows the physical geometry while the model selectors can
substitute point P2M or point L2P ([Geometry](geometry.md)). Fully periodic
cubic plans use the same CPU and CUDA placements
([Caching and periodicity](caching-and-periodicity.md)).

## Which options to leave alone

| Option | Recommendation |
|---|---|
| `backend` | choose for the machine: `CPU_STATIC` (the `AUTO` resolution), or `CUDA_FULL` for repeated field evaluation on a GPU |
| `precision` | choose for the accuracy needed; FP32 halves memory and traffic and is the default |
| `expansion_basis` | keep spherical |
| `spatial_layout` | set `REGULAR_GRID` for lattices of identical bodies |
| `static_matrix_backend` | keep portable unless a measurement on the target problem favours oneMKL |
| `p2p_packing`, `point_expansion_execution`, `use_reduced_symmetry_p2p`, `cuda_dictionary_*` | leave on the automatic policy; force only to validate or benchmark an alternative |
