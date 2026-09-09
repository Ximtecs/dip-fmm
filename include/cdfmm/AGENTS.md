# Public interface guidance

Inherit the repository contract from `../../AGENTS.md`. This directory is the
installed C++/C interface, not a convenient home for shared implementation.

## Current structure

The public headers are currently flat:

```text
include/cdfmm/
|-- cdfmm.hpp                         umbrella include
|-- vec3.hpp, precision.hpp, ...      core and mathematics
|-- geometry.hpp, cuboid.hpp,
|   rectangular_prism.hpp,
|   tetrahedron.hpp                   geometry and exact models
|-- tree_node.hpp, uniform_tree.hpp,
|   adaptive_tree.hpp,
|   static_topology.hpp               trees and topology
|-- operators.hpp,
|   static_operators.hpp              operators and static plans
|-- uniform_fmm.hpp                   solver API and options
|-- cuda_direct.hpp, cuda_p2p.hpp,
|   cuda_cuboid.hpp                   supported CUDA-facing APIs
|-- timings.hpp, validation.hpp,
|   parameter_selection.hpp           diagnostics and utilities
`-- c_api.h                           stable C boundary
```

Target structure, introduced only as cohesive code moves in later steps:

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

## Validation

Compile the public headers through the normal `dev` build, run CTest, and run
Python tests when a binding-visible interface changes. C ABI changes also need
`test_c_api`; optional Fortran-facing changes need the Fortran smoke test.
