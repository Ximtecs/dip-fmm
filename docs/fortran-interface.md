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
options%backend = CDFMM_BACKEND_CPU_STATIC
if (cdfmm_one_mkl_available()) then
    options%static_matrix_backend = CDFMM_STATIC_MATRIX_ONE_MKL
else
    options%static_matrix_backend = CDFMM_STATIC_MATRIX_PORTABLE
end if

call cdfmm_create_uniform_cuboids(fmm, x, y, z, cell_size, options, ierr)
if (ierr /= CDFMM_SUCCESS) error stop cdfmm_last_error()

call cdfmm_evaluate(fmm, mx, my, mz, hx, hy, hz, ierr)
if (ierr /= CDFMM_SUCCESS) error stop cdfmm_last_error()

call fmm%destroy()
```

Plan creation prints a complete initialisation summary to standard output,
including requested options, resolved backend, per-operator executors, P2P
packing, tree geometry, physical source/target geometry, and periodic settings.
The summary is produced by the shared C++ core and is therefore identical for
Fortran, C, C++, and Python callers.

`cdfmm_one_mkl_available()` reports whether the linked library contains the
optional oneMKL executor. This lets an application prefer oneMKL without
turning its absence into a plan-construction error. The equivalent additive C
ABI query is also named `cdfmm_one_mkl_available()` and returns zero or one.

`cdfmm_evaluate` selects the existing FP32 or FP64 C evaluator from the array
kind. The kind must match `options%precision`; a mismatch returns
`CDFMM_ERROR_INVALID_ARGUMENT` and a descriptive `cdfmm_last_error()` message.
The wrapper neither converts precision nor packs temporary Nx3 arrays.

The constructor represents coincident source and target positions, uniformly
magnetised cuboid sources, volume-averaged cuboid targets, and one common
`cell_size(3)`. Geometry and backend resources are created once and reused by
every evaluation. Periodicity is configured through `options%periodic`,
`periodic_cell_center`, `periodic_cell_lengths`, and `periodic_tolerance`.
The same high-level constructor selects the fully periodic, cubic, zero-`k=0`
plan internally. Partial periodicity and non-cubic periodic cells are not
currently supported; invalid cell settings return an error through `ierr` and
`cdfmm_last_error()`.

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
`cdfmm_evaluate_f64`. Existing version-one C options and constructor symbols
remain unchanged; periodic same-cuboid plans use an additive C entry point.

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
