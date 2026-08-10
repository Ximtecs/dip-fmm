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
- a reference `UniformFmm` upward pass with Morton moment permutation, leaf
  P2M, bottom-up M2M, and read-only node multipoles;
- C++ validation helpers, tests, examples, a small benchmark, and Python
  bindings for direct evaluation and tree inspection.

## Limitations

Only the upward portion of the tree traversal exists: callers cannot currently
request a complete FMM field evaluation.  M2L, downward L2L/L2P, and near-field
P2P traversal are not assembled.  The adaptive tree, persistent geometry plan, CUDA
kernels, and MagTense/Fortran interface are also planned rather than
implemented.  See the [roadmap](roadmap.md) for sequencing.

## Architecture

Public C++ declarations live in `include/cdfmm`, with reference
implementations in `src`.  `python/bindings.cpp` supplies the experimental
pybind11 module.  The uniform tree owns geometry and ordering metadata only;
`UniformFmm` owns upward-pass state separately from its fixed tree geometry;
the mathematical operators remain independent and testable in isolation.
[Mathematical formulation](math.md) defines
the shared conventions, while the [API reference](api.rst) exposes the public
headers through Doxygen and Breathe.
