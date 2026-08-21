# Numerical precision

`DenseDirectPlan` and `UniformFmm` separate three precision boundaries:

* **Input precision** is the dtype supplied by the caller. The Python adapter
  accepts C-contiguous NumPy `float32` and `float64` arrays and converts their
  values during shape-checked ingestion.
* **Static storage precision** is selected with `static_precision="float32"`
  (the default) or `"float64"`. The six geometry-dependent tensor matrices
  have exactly that scalar type. Construction evaluates point and analytical
  cuboid tensors in double precision and quantises only the completed values;
  an FP32 plan does not retain an FP64 matrix copy.
* **Execution precision** follows static storage. Portable evaluation uses
  typed loops, and oneMKL uses `cblas_sgemv` for FP32 or `cblas_dgemv` for
  FP64. FP32 input and output component buffers are retained by the plan and
  reused. Public C++ results remain `Vec3` (double) and are converted at the
  boundary.

For the complete FMM, select `UniformFmmOptions::precision` in C++ or
`options.precision` in Python. `StaticPrecision::Float32`/`FLOAT32` is the
default. FP32 plans accept FP32 or FP64 position and moment arrays. Positions
and analytical geometry construction are parsed in FP64; moments are converted
once at the evaluation boundary. P2M, M2M, M2L, L2L, L2P, P2P, multipoles,
locals, near/far fields, scaling, scratch, transfers, and device buffers then
use FP32 exclusively.

FP32 plans express operator geometry in coordinates normalised by the root-box
width. Input moments are correspondingly divided by the cube of that width
during their FP64-capable boundary conversion. This keeps high-order
multipoles and inverse-length M2L scalings representable at physical scales
such as nanometres without introducing FP64 expansion state. The magnetic
field is invariant under this paired scaling; scalar potential is multiplied
by one root-width factor at the output boundary. The root centre and width are
analytical geometry metadata and remain FP64-capable.

Python `evaluate`, `multipole`, `local`, and `root_multipole` arrays have the
selected NumPy dtype. C++ retains the historical double-returning evaluation
and coefficient methods as explicit widening adapters. The typed
`evaluate_float32`/`evaluate_float64`, `multipole_float32`/`multipole_float64`,
and corresponding local/root methods reject a plan with the other precision.

CPU portable uses typed loops; oneMKL uses SGEMM or DGEMM. CUDA partial and
CUDA full use typed kernels and transfers, and CUDA BSR uses `Sbsrmv` or
`Dbsrmv`. `scalar_bytes` in host/device statistics identifies the selected
representation. Host plan memory and persistent device memory remain separate
domains.

`tensor_memory_bytes` reports only the six immutable matrices and is therefore
exactly half as large in FP32 mode. Analytical geometry construction remains
double precision to protect the logarithm, inverse hyperbolic sine, arctangent,
square-root, and cancellation-sensitive cuboid calculations.

FP32 plans do not retain parallel FP64 operator tables. The completed
geometry-dependent operators are quantised once, after which their analytical
FP64 construction temporaries are released. This deliberately excludes a
mixed FP32-static/FP64-state mode.
