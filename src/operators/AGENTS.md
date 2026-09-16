# Mathematical operator implementations

Implementations here define mathematical maps or static mathematical data.
Keep plans, backend resources, vendor libraries, and repeated-evaluation
orchestration out of this tree. These files are the authoritative homes for
operator construction and dynamic application, including the M2P reference
evaluation. Compatibility names and structured names must share these
implementations. Do not create a second P2P definition with different self or
geometry semantics.

`p2p.cpp` owns the authoritative point/rectangular-prism pair-tensor
construction as `cdfmm::operators::p2p::build_pair`. The flat
`cdfmm::build_pair_tensor` is defined in the same translation unit as a thin
delegation and is the only compatibility spelling of that mathematics; the
former `src/cuboid.cpp` home is removed. Tetrahedral pairs stay with the
tetrahedron operator path. `p2p_point_kernel.hpp` holds the inline
point-dipole pair formula that `evaluate_pair` and the CPU position-based
P2P executor both call; change the formula there and nowhere else.

Current tree:

```text
operators/
|-- AGENTS.md
|-- l2l.cpp
|-- l2p.cpp
|-- m2l.cpp
|-- m2m.cpp
|-- m2p.cpp
|-- p2m.cpp
|-- p2p.cpp
`-- spherical_cartesian_conversion.hpp
```
