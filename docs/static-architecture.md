# Static-geometry architecture

`UniformFmm` is designed for stationary source and target positions with many
changing dipole-moment states. Initialisation is deliberately allowed to be
comparatively expensive: it amortises tree construction, coefficient-map
construction, interaction packing, and optional device upload over repeated
evaluations.

```text
INITIALISATION

physical positions, cuboid sizes
   |
   v
canonical root normalisation
   |
   v
uniform tree and Morton permutations
   |
   v
validated universal/geometry cache lookup
   |
   v
load, or construct, static P2M / M2M / M2L / L2L / L2P / P2P
   |
   v
canonical packed execution plans
   |
   +----> optional CUDA upload and persistent buffers


REPEATED EVALUATION

moments -> Morton order -> P2M -> M2M -> M2L -> L2L -> L2P -> far field
    |
    +---------------------------> P2P ---------------------> near field

far field + near field -> target unsorting -> target field
```

## Data ownership and lifetime

The `UniformTree` owns positions in Morton order, permutations, flat nodes,
and `list1`/`list2`. These are immutable after construction. `UniformFmm` owns
the static operator plan derived from that geometry and mutable per-evaluation
arrays: sorted moments, one multipole and local vector per node, near/far
result scratch, and timings. Calls on one evaluator are therefore not
concurrent even though stages may use OpenMP internally.

Nodes are stored level by level. This makes the dependency order explicit:
P2M writes occupied leaves; M2M visits levels from leaf to root; M2L reads
same-level source multipoles and adds to target locals; L2L visits root to leaf;
and L2P reads leaf locals. P2P is independent of the far-field chain until the
two contributions are assembled.

## Canonical packed plans

Sparse coefficient maps store triples `(output, input, value)`. Dense M2L
matrices are column-major and shared by transfer class. M2L interactions use
target-row offsets plus parallel source-node, matrix-ID, and level arrays.
That representation gives each target row a contiguous interaction range and
lets portable CPU and CUDA executors consume identical mathematical data. The
oneMKL executor derives a gather/multiply/scatter packing from it without
changing the canonical plan.

P2P follows the same separation. `StaticP2POperator` is the exact canonical
target-row tensor. Execution experiments derive source-only SoA, compact dense
leaf rectangles, or full BSR(3) blocks without changing it. Standalone CUDA
plans keep each packing resident on the device and compare one-thread-per-row,
one-block-per-leaf, and cuSPARSE execution. Production CPU execution uses SoA.
Production hybrid and full CUDA execution use BSR(3) when a fixed self-identity
map is supplied and its configured memory budget permits; otherwise they use
canonical rows. See the
[static P2P execution study](static-p2p.md) for storage, correctness
constraints, and measured dispatch recommendations.

The word *plan* is reserved here for immutable, precomputed execution
descriptions reused across evaluations. Mutable coefficient and result arrays
are scratch/state rather than plans. CUDA plan objects additionally own device
copies and persistent buffers required to execute a plan.

The far-field coefficient count is basis-dependent: Cartesian plans store
`(p+1)(p+2)(p+3)/6` coefficients and real spherical plans store `(p+1)^2`.
Both use the same canonical static operator and interaction representations;
the near-field plan and tree are independent of this selection.
