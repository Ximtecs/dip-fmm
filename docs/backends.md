# Execution backends

All FMM production backends consume one canonical CPU-built static plan.
Backend selection changes where and how an operator is applied; it does not
change the tree, interaction partition, operator mathematics, or requested
precision. `ExecutionBackend::Auto` resolves to `CpuStatic` and never replaces
an FMM request with an all-to-all direct calculation.

## Operator placement

| Public selection | P2M | M2M | M2L | L2L | L2P | P2P | Device residency |
|---|---|---|---|---|---|---|---|
| `CpuReference` | CPU reference | CPU reference | CPU reference | CPU reference | CPU reference | CPU direct `list1` | Host |
| `CpuStatic` + `Portable` | CPU static | CPU static | CPU loops | CPU static | CPU static | CPU SoA tensor | Host |
| `CpuStatic` + `OneMkl` | CPU static | CPU static | oneMKL SGEMM/DGEMM | CPU static | CPU static | CPU SoA tensor | Host |
| `CudaPartial` | CPU static | CPU static | CUDA target rows | CPU static | CPU static | CUDA static tensor | Static GPU data; expansion state crosses at the M2L boundary |
| `CudaFull` | CUDA static | CUDA static | CUDA target rows | CUDA static | CUDA static | CUDA static tensor | Operators and coefficient state remain on device |
| `DenseDirectPlan` | — | — | — | — | — | CPU dense exact | Host geometry tensors |
| `CudaDirectPlan` | — | — | — | — | — | CUDA dense exact | Persistent device geometry and scratch |

`Portable` and `OneMkl` in the table are values of `StaticMatrixBackend`.
oneMKL accelerates M2L only: interactions sharing a normalised transfer matrix
are gathered into columns, multiplied with SGEMM or DGEMM, and scattered to
target locals. The other stages retain the portable static executors.

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

`UniformFmm` uniform-cuboid sources require the Cartesian basis and a static
M2L plan. Targets remain points. The lower-level `DenseDirectPlan` independently
supports point or uniform-cuboid sources and point or volume-averaged-cuboid
targets. See [Mathematical formulation](math.md) for that distinction.

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
phase timings therefore are not a sequential sum.

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
new evaluator.

## Compatibility names and availability

`CudaM2LP2P` is the canonical enum value behind `CudaPartial`.
`CudaM2L` and `CudaM2LStaticP2P` are compatibility aliases for that same
hybrid implementation, not separate backends. `cuda_m2l_available()` is
likewise retained as an alias for `cuda_m2l_p2p_available()`.

CUDA compilation and runtime device availability are separate. Capability
queries report both, and requesting an unavailable backend raises an error.
The standalone CUDA direct reference remains separately named so its
quadratic algorithm cannot be mistaken for a fallback FMM.
