# Fortran interface guidance

Inherit `../AGENTS.md`. The Fortran layer is a thin ISO_C_BINDING wrapper over
the public C ABI.

## Current structure

```text
fortran/
`-- cdfmm_fortran.f90    native wrapper module

tests/test_fortran_api.f90
examples/fortran/magtense_style_demag.f90
include/cdfmm/c_api.h
src/c_api.cpp
```

## Rules and validation

- Keep kinds, shapes, ownership, error propagation, enum values, and lifetime
  semantics aligned with `c_api.h`; do not bypass the C boundary.
- Do not implement numerical or planning logic in Fortran.
- Preserve the wrapper as optional while the C ABI remains unconditional.
- Update the smoke test and example when a supported C/Fortran contract changes.

Configure a suitable preset or manual Release build with
`-DCDFMM_BUILD_FORTRAN_INTERFACE=ON`, build, and run CTest. This requires an
available Fortran compiler; record a precise skip when one is unavailable.
