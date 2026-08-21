# Numerical precision

`DenseDirectPlan` separates three precision choices:

* **Input precision** is the dtype supplied by the caller. The Python adapter
  accepts C-contiguous NumPy `float32` and `float64` arrays and converts their
  values during shape-checked ingestion.
* **Static storage precision** is selected with `static_precision="float32"`
  or `"float64"` (the default). The six geometry-dependent tensor matrices
  have exactly that scalar type. Construction evaluates point and analytical
  cuboid tensors in double precision and quantises only the completed values;
  an FP32 plan does not retain an FP64 matrix copy.
* **Execution precision** follows static storage. Portable evaluation uses
  typed loops, and oneMKL uses `cblas_sgemv` for FP32 or `cblas_dgemv` for
  FP64. FP32 input and output component buffers are retained by the plan and
  reused. Public C++ results remain `Vec3` (double) and are converted at the
  boundary.

`tensor_memory_bytes` reports only the six immutable matrices and is therefore
exactly half as large in FP32 mode. Analytical geometry construction remains
double precision to protect the logarithm, inverse hyperbolic sine, arctangent,
square-root, and cancellation-sensitive cuboid calculations.

The uniform FMM and CUDA plans currently remain FP64. They do not silently
claim FP32 storage: extending typed static storage through their coefficient,
M2L, L2P, P2P/BSR, and device representations remains future work. A mixed
FP32-static/FP64-state mode is likewise not provided because ordinary BLAS has
no direct mixed-type GEMM/GEMV equivalent.
