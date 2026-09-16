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
|-- operators/                        canonical P2M/M2M/M2L/L2L/L2P/M2P/P2P
|-- plan/                             canonical static plans and P2P packings
|   `-- p2p/tensor_dictionary.hpp     Tensor6 canonicalisation/token packing
|-- *.hpp                             legacy shims and deferred higher layers
|-- operators.hpp,
|   static_operators.hpp              flat operator/static-plan compatibility
|-- uniform_fmm.hpp                   solver API and options
|-- backend/execution.hpp             ExecutionBackend (narrow, backend-selection)
|-- backend/cuda/availability.hpp,
|   backend/mkl/availability.hpp      CUDA/oneMKL capability queries
|-- backend/cuda/{direct,dense_direct,p2p,m2l}.hpp
|                                      canonical CUDA direct, P2P, and M2L APIs
|-- backend/cpu/{p2p,m2l,far_field}.hpp
|                                      canonical portable static-plan APIs
|-- cuda_direct.hpp, cuda_p2p.hpp,
|   cuda_cuboid.hpp                   legacy CUDA-facing compatibility façades
|-- cuboid.hpp                        compatibility façade; the prism record,
|                                      averaged monomial, pair tensor, and
|                                      dense-direct plan are canonical below
|-- tensor_dictionary.hpp             compatibility façade; canonical home is
|                                      plan/p2p/tensor_dictionary.hpp above
|-- timings.hpp, validation.hpp,
|   parameter_selection.hpp           diagnostics and utilities
`-- c_api.h                           stable C boundary
```

The `core`, `math`, `geometry`, `tree`, `operators`, `plan`, and `backend`
homes are implemented. Only `fmm/` remains a target directory, introduced
solely when a cohesive public solver interface moves into it:

```text
include/cdfmm/
|-- cdfmm.hpp
|-- core/         implemented
|-- math/         implemented
|-- geometry/     implemented
|-- tree/         implemented
|-- operators/    implemented
|-- plan/         implemented
|-- backend/      implemented
`-- fmm/          target only; uniform_fmm.hpp remains a flat public header
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
  `cdfmm/periodic.hpp`. Both were audited during the public-header/packaging
  cleanup and kept flat deliberately, not left merely because no one had moved
  them yet: `timings.hpp` aggregates `fmm`-, `plan`-, and CUDA-backend-owned
  diagnostic/statistics types that are read together by one
  `UniformFmm`/diagnostics surface, so no single subsystem below `fmm` owns
  all of it; `periodic.hpp` is depended on symmetrically by the sibling `tree`
  and `operators` subsystems (topology box-list construction and M2L/tree
  construction respectively), so giving it to either would create a
  sibling-to-sibling edge the layering diagram deliberately avoids. See
  "Public/internal API, header ownership, and packaging cleanup" in
  `docs/architecture.md` for the full audit.
- A flat header that still owns declarations (`operators.hpp`, `periodic.hpp`,
  `timings.hpp`, `uniform_fmm.hpp`, `validation.hpp`, `parameter_selection.hpp`)
  is supported public API whose canonical home is deferred or, for
  `timings.hpp`/`periodic.hpp`/`validation.hpp`, was audited and found to have
  no single clearer owner. Do not treat it as debt to delete, and do not
  migrate it as part of an unrelated step. `tensor_dictionary.hpp` no longer
  belongs to this list: its Tensor6 canonicalisation/token-packing primitives
  moved to `plan/p2p/tensor_dictionary.hpp`, the P2P-packing subsystem that is
  their only consumer, and the flat path is now a plain forwarding façade.
  `ExecutionBackend` similarly moved out of `uniform_fmm.hpp` to the narrow
  `backend/execution.hpp`, and the CUDA/oneMKL capability queries moved to
  `backend/cuda/availability.hpp`/`backend/mkl/availability.hpp`;
  `uniform_fmm.hpp` includes all three and keeps re-exporting every name.

## Validation

Compile the public headers through the normal `dev` build, run CTest, and run
Python tests when a binding-visible interface changes. C ABI changes also need
`test_c_api`; optional Fortran-facing changes need the Fortran smoke test.
