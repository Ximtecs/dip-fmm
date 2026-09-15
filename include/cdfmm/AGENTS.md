# Public interface guidance

Inherit the repository contract from `../../AGENTS.md`. This directory is the
installed C++/C interface, not a convenient home for shared implementation.

## Current structure

The foundational public headers now have canonical subsystem homes, while
legacy flat paths remain as forwarding compatibility headers:

```text
include/cdfmm/
|-- cdfmm.hpp                         umbrella include
|-- core/                             precision, output flags, basic timing
|-- math/                             vectors, indices, derivatives, expansions
|-- geometry/                         models and exact physical primitives
|-- tree/                             common, uniform, and adaptive interfaces
|-- *.hpp                             legacy shims and deferred higher layers
|-- operators.hpp,
|   static_operators.hpp              operators and static plans
|-- uniform_fmm.hpp                   solver API and options
|-- backend/cuda/{direct,dense_direct,p2p,m2l}.hpp
|                                      canonical CUDA direct, P2P, and M2L APIs
|-- backend/cpu/{p2p,m2l,far_field}.hpp
|                                      canonical portable static-plan APIs
|-- cuda_direct.hpp, cuda_p2p.hpp,
|   cuda_cuboid.hpp                   legacy CUDA-facing compatibility façades
|-- cuboid.hpp                        compatibility façade; the prism record,
|                                      averaged monomial, pair tensor, and
|                                      dense-direct plan are canonical below
|-- timings.hpp, validation.hpp,
|   parameter_selection.hpp           diagnostics and utilities
`-- c_api.h                           stable C boundary
```

Remaining target structure, introduced only as cohesive code moves in later
steps:

```text
include/cdfmm/
|-- cdfmm.hpp
|-- core/
|-- math/
|-- geometry/
|-- tree/
|-- operators/
|-- plan/
|-- backend/
`-- fmm/
```

## Rules

- Add a header here only for an intentionally supported downstream interface.
- Keep internal headers, CUDA `.cuh` files, device/launch helpers, cache
  implementation, and backend resource ownership in `src/`.
- Prefer forward declarations or narrow lower-layer includes. A lower layer
  must not include a higher-level solver header for convenience.
- `cdfmm.hpp` is the user umbrella; subsystem headers should remain usable
  directly and must not depend on the umbrella.
- Document public types and functions with Doxygen, including ownership,
  units, precision, indexing, self-interaction, and lifetime constraints.
- Preserve public names and include compatibility during the behaviour-
  preserving refactor. Moving implementation is not an API redesign.
- Every flat header here was an installed public path at `v0.1.0`. They are
  intentional compatibility façades, not leftovers, and are retained. A façade
  must also reproduce the pre-v0.2 transitive surface of its path, because
  downstream code relied on what it pulled in.
- Compatibility flows one way. A canonical subsystem header under `core/`,
  `math/`, `geometry/`, `tree/`, `operators/`, `plan/`, or `backend/` includes
  only canonical headers; the flat façades include those, not the reverse. The
  remaining flat includes from canonical headers are `cdfmm/timings.hpp` and
  `cdfmm/periodic.hpp`, which are substantive public headers awaiting a
  subsystem home rather than façades.
- A flat header that still owns declarations (`operators.hpp`, `periodic.hpp`,
  `timings.hpp`, `uniform_fmm.hpp`, `validation.hpp`, `parameter_selection.hpp`,
  `tensor_dictionary.hpp`) is supported public API whose canonical home is
  deferred. Do not treat it as debt to delete, and do not migrate it as part of
  an unrelated step.

## Validation

Compile the public headers through the normal `dev` build, run CTest, and run
Python tests when a binding-visible interface changes. C ABI changes also need
`test_c_api`; optional Fortran-facing changes need the Fortran smoke test.
