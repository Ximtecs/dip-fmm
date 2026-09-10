# Mathematical operator interfaces

This directory contains the canonical public-facing homes for P2M, M2M, M2L,
L2L, L2P, M2P, and P2P mathematical interfaces. Operators describe mappings
and static construction; M2P also provides a direct multipole reference
evaluation. They do not own execution buffers, vendor descriptors, CUDA
streams, or FMM orchestration.

Plan data used by static construction comes from `cdfmm/plan`; the legacy
`cdfmm/static_operators.hpp` header is only a forwarding umbrella. New
declarations should use the nested `cdfmm::operators` namespaces and preserve
the established coefficient order, sign conventions, geometry dispatch, and
explicit self-interaction policy.

Current tree:

```text
operators/
|-- AGENTS.md
|-- l2l.hpp
|-- l2p.hpp
|-- m2l.hpp
|-- m2m.hpp
|-- m2p.hpp
|-- p2m.hpp
|-- p2p.hpp
`-- operators.hpp
```
