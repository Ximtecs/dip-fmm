# Overview

## Purpose

`dip-fmm` implements a Laplace FMM specialised to magnetic dipole fields on
fixed free-space or fully periodic geometry. Expensive geometry and operator
construction is amortised across changing dipole-moment states. The magnetic
field $H=-\nabla\phi$ is the primary result; scalar potential is optional on
supported execution paths.

## Current capabilities

- real spherical-harmonic expansions by default and a complete independent
  Cartesian Taylor formulation;
- a complete non-adaptive, Morton-sorted uniform octree with `list1` near and
  `list2` far interactions;
- reusable static P2M, M2M, dense M2L, L2L, L2P, and exact P2P operators;
- true FP32 and FP64 operator storage, state, CPU execution, and CUDA execution;
- portable CPU, class-grouped oneMKL M2L, hybrid CUDA M2L/P2P, and full
  device-resident CUDA FMM;
- exact direct CPU and CUDA references plus retained reference/alternative
  operator packings for validation and benchmarking;
- point-dipole and uniform-cuboid sources, plus point and
  volume-averaged-cuboid targets, in both bases;
- free-space and fully periodic cubic zero-`k=0` evaluation; and
- C++ tests, Python bindings, benchmarks, notebooks, and plan/timing/memory
  inspection.

## Capability boundary

The tree is complete and uniform rather than adaptive. CUDA-full is the
repeated field path. Fully three-dimensional periodic
magnetostatics is available for explicit cubic cells with the zero-`k=0`
convention on static CPU and CUDA plans. Partial periodicity and rectangular
periodic cells are not implemented. A stable C ABI, Fortran wrapper, and
MagTense integration are not implemented.

## Architecture

Public C++ declarations live in `include/cdfmm`; implementation files live in
`src`; `python/bindings.cpp` supplies the pybind11 module. `UniformTree` owns
immutable geometry and ordering metadata. `UniformFmm` owns the selected basis,
canonical static plan, typed mutable coefficient/result state, and optional
device plan.

Start with [Static-geometry architecture](static-architecture.md) for ownership
and data flow, [FMM overview](fmm-overview.md) for traversal,
[Mathematical formulation](math.md) for normative conventions, and
[Execution backends](backends.md) for operator placement.
