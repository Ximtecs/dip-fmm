# Execution backends

All production backends consume the same CPU-built static operators and use
the same uniform-tree interaction lists. Backend selection changes where an
operator is applied, not its mathematical definition or traversal order.
`Auto` resolves conservatively to `CpuStatic`.

## Operator placement

| Backend | P2M | M2M | M2L | L2L | L2P | P2P |
|---|---|---|---|---|---|---|
| CPU portable (`CpuStatic`) | CPU | CPU | CPU | CPU | CPU | CPU |
| CPU oneMKL | CPU | CPU | oneMKL SGEMM/DGEMM on CPU | CPU | CPU | CPU |
| CUDA partial (`CudaPartial`) | CPU | CPU | CUDA | CPU | CPU | CUDA |
| CUDA full (`CudaFull`) | CUDA | CUDA | CUDA | CUDA | CUDA | CUDA |

The CPU portable path applies compact sparse P2M/M2M/L2L/L2P/P2P maps and the
canonical target-row M2L plan without a BLAS dependency. `CpuReference` is an
independent educational/validation traversal rather than the production
static path.

The oneMKL option accelerates **M2L only**. Interactions sharing a normalised
transfer matrix are gathered into columns, multiplied by SGEMM or DGEMM, then scattered
to target locals. P2M, M2M, L2L, L2P and P2P remain portable CPU operations.
The oneMKL executor uses the same matrices and level scaling as the portable
executor.

Every row supports FP32 and FP64. Backend placement does not change scalar
precision: an FP32 plan uses four-byte operators, coefficient state, scratch,
device buffers, and evaluation transfers throughout. Geometry positions remain
FP64-capable and are converted only where a completed static operator is stored.

## CUDA partial data flow

```text
CPU: sort moments -> P2M -> M2M
                         |
                         +-- multipoles -> GPU -> M2L -> raw locals -> CPU

moments -> GPU -> P2P -> near field -> CPU       (independent CUDA stream)

CPU: raw locals -> L2L -> L2P -> far field
CPU: far field + near field -> user-order result
```

Static M2L matrices, interaction metadata, scaling tables, and the sparse P2P
tensor are uploaded when the evaluator is initialised. During evaluation the
M2L and P2P transfers and kernels run independently; the host waits for P2P
only when final assembly needs the near field. Potential output uses the CPU
reference near-field calculation because the static CUDA P2P tensor is
field-only.

## CUDA full data flow

Geometry-dependent operators, permutations, interaction metadata, scaling
tables, and persistent buffers are built on the CPU and uploaded once during
initialisation. A repeated field evaluation is:

```text
moments -> GPU
    sort -> P2M -> M2M -> M2L -> L2L -> L2P
              moments -> P2P
          far field + near field -> unsort
field -> CPU
```

The full path keeps intermediate multipoles, locals, and fields resident on
the device. M2M must complete a child level before its parent level consumes
it, and L2L must complete a parent level before its children consume it; the
single CUDA stream supplies those dependencies. The explicit self-identity map
becomes part of this persistent plan on its first use and cannot change without
rebuilding the evaluator.

CUDA compilation and runtime device availability are separate. Requesting an
unavailable backend raises an error; it never silently changes the algorithm
or falls back to direct all-to-all evaluation. `cuda_direct_p2p_reference` is
a separate $O(N^2)$ validation facility, not an FMM backend.
