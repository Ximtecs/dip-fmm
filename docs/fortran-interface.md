# C and Fortran integration interface

The boundary is `ISO_C_BINDING` Fortran -> the versioned plain C ABI in
`cdfmm/c_api.h` -> `cdfmm::UniformFmm`. ABI version 1 exposes no C++ classes,
exceptions, containers, CUDA handles, or tree objects. The opaque `cdfmm_plan`
owns fixed geometry, static operators, backend resources, and reusable adapter
buffers.

## Lifecycle, layout, and physical convention

Create a plan once, call `cdfmm_plan_evaluate_f32` or
`cdfmm_plan_evaluate_f64` for changing states, then destroy it. Coordinates are
double precision and component separated. Evaluation consumes contiguous
`mx/my/mz` and writes `hx/hy/hz`; adapter AoS and result staging is allocated at
construction and reused. Evaluation precision must match the plan and output is
field-only.

The C adapter passes `FloatVec3` staging to the native FP32 UniformFmm overload,
so component FP32 input does not take a float-to-double-to-float round trip.

Inputs are **total physical moments**, not magnetisation density:

```
moment_i = volume_i * M_i = volume_i * Ms * mhat_i
```

The result is the signed demagnetising field, `H = -grad(phi)`. A uniformly
magnetised cube at its centre has `H = -M/3`; a future adapter must not retain
an external minus sign from an already-signed tensor implementation.

Point plans support spherical and Cartesian expansions. Uniform cuboid sources
require Cartesian expansions and use finite cuboid P2M, exact list1
cuboid-to-point P2P, and the finite self field. This is **point-target**
evaluation, not the volume-averaged cuboid target required for complete
MagTense cell-averaged equivalence. The API does not fake that missing
capability and leaves room for a later constructor.

CUDA-full selects the existing `ExecutionBackend::CudaFull`; geometry, static
operators, streams, library resources, and buffers persist in UniformFmm.
Changing moments are staged and transferred each call without re-uploading
geometry. An unavailable CUDA request returns a CUDA-unavailable status.

No exception crosses the boundary. Failures return integer status values and a
thread-local message is available through `cdfmm_get_last_error` or the Fortran
`cdfmm_last_error()`. One plan is non-reentrant; different plans may be used
independently and internal OpenMP remains permitted.

## Building and using Fortran

`fortran/cdfmm_fortran.f90` provides `cdfmm_plan_t`, defaults, point,
same-point and uniform-cuboid constructors, FP32/FP64 evaluation, explicit
destruction, and finalisation safety. The example builds one eight-cell plan,
converts `Ms*mhat` into total moment outside the library, evaluates three
states, times them, and destroys the plan.

```sh
cmake -S . -B build -DCDFMM_BUILD_FORTRAN_INTERFACE=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/magtense_style_demag
```

After installation to `$prefix`, a concise manual build is:

```sh
gfortran -c $prefix/share/cdfmm/fortran/cdfmm_fortran.f90
gfortran cdfmm_fortran.o my_solver.f90 -L$prefix/lib -lcdfmm_c -lstdc++ \
  -Wl,-rpath,$prefix/lib -o my_solver
```

Set `LD_LIBRARY_PATH=$prefix/lib` instead if an rpath is not embedded. CMake is
preferred because additional platform dependencies can vary.

## Future MagTense contract

The later MagTense PR should create one plan during
`SolveLandauLifshitzEquation` initialisation, reuse or convert its component
arrays in every `updateDemagfield`, call `cdfmm_evaluate_f32`, write
`HmX/HmY/HmZ`, and destroy the plan during cleanup. It must verify the current
call site's `Mfact` scaling and sign rather than coupling this generic ABI to
MagTense global variables or unverified semantics.
