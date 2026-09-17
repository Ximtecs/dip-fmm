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
point-dipole pair formula that `evaluate_pair` and the CPU and CUDA
position-based P2P executors all call; change the formula there and nowhere
else. `point_expansion_kernel.hpp` holds the procedural point P2M/L2P kernels
(the spherical operators of `p2m.cpp`/`l2p.cpp` reconstructed from the
recurrence in `math/solid_harmonic_recurrence.hpp`) and the only copies of
their constants; the stored construction remains the authoritative
definition the procedural kernels are tested against.

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
|-- p2p_point_kernel.hpp
|-- point_expansion_kernel.hpp
`-- spherical_cartesian_conversion.hpp
```
