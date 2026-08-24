# Numerical precision

`StaticPrecision::Float32` is the default; `StaticPrecision::Float64` selects
double precision. For `UniformFmm`, this is an execution choice rather than a
static-storage-only choice.

| Boundary | FP32 plan | FP64 plan |
|---|---|---|
| Caller input | accepts FP32 or FP64 arrays | accepts FP32 or FP64 arrays |
| Analytical construction | FP64 temporaries | FP64 |
| Retained static operators | FP32 only | FP64 only |
| Multipole/local state | FP32 | FP64 |
| Scratch and near/far fields | FP32 | FP64 |
| CPU arithmetic / BLAS | typed loops / SGEMM | typed loops / DGEMM |
| CUDA buffers and arithmetic | FP32 | FP64 |
| Python result and coefficient dtype | `numpy.float32` | `numpy.float64` |

Positions, root metadata, and analytical operator construction are parsed or
formed in FP64. An FP32 plan quantises each completed operator once and releases
its FP64 construction temporaries; it does not retain a hidden double-precision
operator table. Moments are converted once at the evaluation boundary, after
which P2M, M2M, M2L, L2L, L2P, P2P, coefficient state, scratch, transfers, and
device buffers use the selected scalar throughout.

The C++ `evaluate`, `multipole`, `local`, and `root_multipole` compatibility
methods return double-valued objects and therefore widen FP32 results at the
public boundary. Their explicitly typed counterparts reject a plan with the
wrong precision. Python exposes the native selected dtype.

## Root-width scaling in FP32 FMM plans

FP32 `UniformFmm` operators use coordinates divided by the physical root-box
width. Moments are divided by the cube of that width during boundary
conversion. This paired scaling keeps high-order coefficients and inverse-box
M2L factors representable for physical scales such as nanometres without
introducing FP64 expansion state. The field is invariant under the scaling;
potential is restored with one root-width factor at the output boundary.

This is mandatory internal behaviour of FP32 FMM plans, not a public optional
global coordinate-normalisation switch. FP64 plans retain physical-coordinate
operator construction. Both paths accept and return physical coordinates,
moments, potential, and field.

The root-width transformation is also distinct from:

- factorial-normalised Cartesian Taylor monomials;
- orthonormal spherical-harmonic and solid-harmonic normalisation; and
- degree/box-width factors that make one M2L transfer matrix reusable across
  tree levels.

Those mathematical conventions are defined in
[Mathematical formulation](math.md).

## Direct plans and memory reporting

`DenseDirectPlan` applies the same selected storage/execution scalar to its six
immutable pair-tensor matrices and reusable component staging. Portable GEMV
and oneMKL SGEMV/DGEMV are alternative executors over those same matrices.
Analytical cuboid construction remains FP64 because its logarithm, inverse
hyperbolic sine, arctangent, square-root, and cancellation-sensitive formulas
need that setup precision.

Host and device statistics report actual retained scalar widths.
`scalar_bytes` identifies the representation; operator, expansion-state,
near-field, tree, and persistent-device bytes are separate so FP32 savings are
not obscured by basis-independent metadata.

This design deliberately excludes a mixed FP32-static/FP64-state mode.
