# Core technology and invariants

## Mathematical contract

The implementation uses the Laplace Green function

```text
G(r) = 1 / (4*pi*|r|)
phi(x) = sum_j m_j . (x - x_j) / (4*pi*|x - x_j|^3)
H(x) = -grad(phi)
H_ij = 1/(4*pi) * [3*r_ij*(m_j.r_ij)/|r_ij|^5 - m_j/|r_ij|^3]
```

where `r_ij = x_i - x_j`. A source-point singularity is excluded only by an
explicit target-to-source identity map; equal coordinates alone never identify
a self pair. Moments are total dipole moments. For finite geometry, the source
and target records select physical integration elements, while near-field and
far-field source/target approximations remain independent options.

Real spherical harmonics use the documented Condon–Shortley real tesseral
convention: coefficients are ordered by increasing degree and then
`m = -l..+l`, degree-one modes in `(-1,0,1)` order are `(y,z,x)`, and order
`p` stores `(p+1)^2` coefficients. Cartesian total-degree order `p` stores
`(p+1)(p+2)(p+3)/6` coefficients. Both bases represent the same potential and
field contract but have independent operator construction and validation paths.

## Persistent FMM lifecycle

`UniformFmm` turns fixed geometry into reusable execution state:

```text
physical positions and geometry
  -> root normalisation and tree/topology
  -> canonical P2M/M2M/M2L/L2L/L2P/P2P operators
  -> immutable static plans and derived packings
  -> optional persistent oneMKL/CUDA resources
  -> repeated evaluations with changing moments
```

Construction owns geometry, Morton permutations, interaction lists, operator
maps, cache identity, and backend resources. Evaluation replaces moments,
clears mutable multipole/local/result state, executes the upward/downward and
near-field chains, combines near/far fields, and restores caller target order.
One evaluator is not re-entrant; OpenMP parallelism is internal to an
evaluation. Empty complete-tree nodes remain explicit and zero-valued.

In the current architecture, the mathematical maps are constructed under the
operator layer, immutable canonical and derived representations under the plan
layer, and portable static-plan application under the CPU backend boundary.
This separation is structural: it does not change the coefficient, ordering,
identity, precision, or accumulation contracts below.

Canonical mathematical data is the backend-independent truth. Portable CPU SoA,
oneMKL gather/GEMM/scatter, CUDA canonical/BSR, tensor-dictionary, and other
packings are derived execution representations and must not redefine operators.
Level barriers preserve P2M, deep-to-shallow M2M, level-local M2L, shallow-to-deep
L2L, L2P, and independent list-1 P2P dependencies.

## Precision and scaling

`StaticPrecision::Float32` is the default. Analytical geometry/operator
construction uses FP64 temporaries, then an FP32 plan retains only FP32 static
operators, expansion state, scratch, fields, and device buffers. FP64 plans
retain and execute those objects as FP64. Public compatibility accessors may
widen FP32 inspection/results, while typed APIs and Python preserve native
precision.

FP32 FMM plans normalise coordinates by the physical root-box width and divide
moments by its cube at the input boundary; potential is restored by one root
width at output. This internal transformation is separate from Cartesian
factorial normalisation, spherical harmonic normalisation, and level-dependent
M2L degree scaling.

## Backends and optional systems

`CpuReference` is the independent Cartesian traversal. `CpuStatic` is the
portable production static plan. oneMKL accelerates only grouped static M2L
(`SGEMM`/`DGEMM`) over the canonical plan. `CudaPartial` keeps sorting, P2M,
M2M, L2L, and L2P on the CPU while CUDA executes static M2L and list-1 P2P.
`CudaFull` keeps the complete repeated field path and static state on the
device. `DenseDirectPlan` and `CudaDirectPlan` are exact O(N^2) references.

For repeated CUDA-full evaluation, geometry/operators/scratch are uploaded at
construction; only changing moments cross H2D and only the final user-ordered
field crosses D2H. CUDA partial has separate M2L and P2P streams, so phase
timings are not a sequential sum. Availability means both compile-time support
and runtime device capability; requests must fail explicitly when unavailable.

Fully periodic plans use image-aware topology and Ewald-derived root operators
for a cubic cell under the `ZeroK0` convention. Partial periodicity and other
macroscopic conventions are intentionally unsupported.

## Persistent caches

Universal translation banks and geometry plans are validated binary artifacts.
Keys include basis, order, precision, geometry/model selections, topology and
cache/math versions; physical translation and scale are excluded after root
normalisation. Headers, checksums, temporary-file writes, `fsync`, and atomic
rename make missing/corrupt/incompatible files rebuildable without changing an
evaluation result. `CDFMM_CACHE_DIR` and `CDFMM_DISABLE_CACHE` control the
default cache behaviour; `cdfmm-precompute` creates universal operator files.
