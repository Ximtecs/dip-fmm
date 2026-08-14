# Overview

## Purpose

`dip-fmm` develops a clear CPU reference for the Laplace fast multipole method
(FMM) specialised to point dipoles in Cartesian coordinates.  General-purpose
FMM packages support more kernels and geometries; this project instead keeps
the magnetic convention, coefficient representation, and validation paths
explicit for dipole-field applications.

Cartesian total-degree expansions avoid spherical harmonics and expose every
translation as a contraction over three-dimensional multi-indices.  That makes
the implementation compact, auditable, and suitable as a correctness baseline
for later optimisation.

## Current capabilities

- a total-degree `MultiIndexSet` and linear coefficient storage;
- algebraic Taylor jets and raw Cartesian derivatives of
  $G(r)=1/(4\pi|r|)$;
- CPU P2M, M2M, M2L, L2L, L2P, M2P, and direct P2P operators;
- optional potential and/or magnetic-field output;
- a complete, non-adaptive, Morton-sorted uniform octree with source and target
  ranges plus `list1` and `list2` interactions;
- a functional static `UniformFmm` with upward and downward passes, direct
  list1 near fields, target unsorting, and inspectable node expansions;
- C++ validation helpers, tests, examples, a small benchmark, and Python
  bindings for direct evaluation and tree inspection.

## Limitations

The tree is uniform rather than adaptive, and MagTense/Fortran integration is
not implemented. Portable CPU, oneMKL M2L, partial CUDA, and full CUDA
execution are available when their optional build dependencies are enabled.
See the [backend guide](backends.md) and [roadmap](roadmap.md).

## Architecture

Public C++ declarations live in `include/cdfmm`, with reference
implementations in `src`.  `python/bindings.cpp` supplies the experimental
pybind11 module.  The uniform tree owns geometry and ordering metadata only;
`UniformFmm` owns expansion state separately from its fixed tree geometry;
the mathematical operators remain independent and testable in isolation.
[Mathematical formulation](math.md) defines
the shared conventions, while the [API reference](api.rst) exposes the public
headers through Doxygen and Breathe.

For initialisation/evaluation data flow and ownership boundaries, start with
[Static-geometry architecture](static-architecture.md). Exact operator
equations and M2L scaling conventions are in the
[Mathematical formulation](math.md).
