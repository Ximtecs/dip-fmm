# Fortran interface

## MagTense-style use

The native Fortran wrapper is the recommended interface. It hides the C plan
pointer and ABI structure, while retaining component-separated arrays and the
persistent-plan lifecycle:

```fortran
use, intrinsic :: iso_c_binding
use cdfmm_fortran

type(cdfmm_plan_t) :: fmm
type(cdfmm_options_t) :: options
integer(c_int) :: ierr

options = cdfmm_options()
options%order = 6
options%depth = 4
options%basis = CDFMM_BASIS_SPHERICAL
options%precision = CDFMM_PRECISION_FLOAT32
options%backend = CDFMM_BACKEND_CUDA_FULL

call cdfmm_create_uniform_cuboids(fmm, x, y, z, cell_size, options, ierr)
if (ierr /= CDFMM_SUCCESS) error stop cdfmm_last_error()

call cdfmm_evaluate(fmm, mx, my, mz, hx, hy, hz, ierr)
if (ierr /= CDFMM_SUCCESS) error stop cdfmm_last_error()

call fmm%destroy()
```

`cdfmm_evaluate` selects the existing FP32 or FP64 C evaluator from the array
kind. The kind must match `options%precision`; a mismatch returns
`CDFMM_ERROR_INVALID_ARGUMENT` and a descriptive `cdfmm_last_error()` message.
The wrapper neither converts precision nor packs temporary Nx3 arrays.

The constructor represents coincident source and target positions, uniformly
magnetised cuboid sources, volume-averaged cuboid targets, and one common
`cell_size(3)`. Geometry and backend resources are created once and reused by
every evaluation. Periodicity is configured through `options%periodic`,
`periodic_cell_center`, `periodic_cell_lengths`, and `periodic_tolerance`.
The current C ABI does not yet expose periodic persistent-plan construction, so
requesting it returns `CDFMM_ERROR_UNSUPPORTED`; the calling workflow will not
change when that support is added.

Inputs are **total physical magnetic moments**, not magnetisation density:

```text
moment_i = volume_i * M_i = volume_i * Ms_i * mhat_i
```

The wrapper deliberately performs no volume or `Ms` scaling because these may
vary by cell. The result is the signed, volume-averaged demagnetising field
`H = -grad(phi)`.

## Advanced C ABI access

The boundary beneath the convenience layer remains `ISO_C_BINDING` Fortran ->
the versioned plain C ABI in `cdfmm/c_api.h` -> `cdfmm::UniformFmm`. Advanced
callers may use `cdfmm_c_options_t`, `cdfmm_default_options`, the constructors
with separate source and target arrays, and explicit `cdfmm_evaluate_f32` or
`cdfmm_evaluate_f64`. The C header and its symbols remain unchanged.

No exception crosses this boundary. Calls return integer status values and a
thread-local C error is exposed through `cdfmm_last_error()`. One plan is
non-reentrant; separate plans may be used independently.

## Building

```sh
cmake -S . -B build -DCDFMM_BUILD_FORTRAN_INTERFACE=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/magtense_style_demag
```

After installation to `$prefix`, a manual build is:

```sh
gfortran -c $prefix/share/cdfmm/fortran/cdfmm_fortran.f90
gfortran cdfmm_fortran.o my_solver.f90 -L$prefix/lib -lcdfmm_c -lstdc++ \
  -Wl,-rpath,$prefix/lib -o my_solver
```
