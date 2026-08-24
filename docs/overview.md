# Overview

## Purpose

`dip-fmm` implements the free-space Laplace FMM specialised to magnetic dipole
fields on fixed geometry. Expensive geometry and operator construction is
amortised across changing dipole-moment states. The magnetic field
$H=-\nabla\phi$ is the primary result; scalar potential is optional on
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
- point-dipole sources in both bases and Cartesian uniform-cuboid sources to
  point targets;
- lower-level direct point/cuboid source and point/volume-averaged-cuboid target
  tensors; and
- C++ tests, Python bindings, benchmarks, notebooks, and plan/timing/memory
  inspection.

## Capability boundary

The tree is complete and uniform rather than adaptive. Spherical
`UniformFmm` plans support point sources only. Cartesian `UniformFmm` supports
uniform-cuboid sources but still evaluates point targets; lower-level
volume-averaged target support is not yet wired through the complete FMM.
CUDA-full is the repeated field path. A stable C ABI, Fortran wrapper, MagTense
integration, and periodic magnetostatics are not implemented.

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
