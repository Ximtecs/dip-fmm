# Roadmap

ABI version 1 provides the persistent C/Fortran plan boundary for a future
MagTense adapter. Full cuboid-source to volume-averaged-cuboid-target FMM
evaluation remains necessary for final cell-averaged demag equivalence.

This roadmap records the present capability boundary and the next meaningful
development stages. Mathematical details, execution layouts, and historical
benchmark investigations live in the dedicated documentation pages rather
than here.

## Implemented foundation

### Core and reference FMM

- [x] Laplace point-dipole potential and field with explicit self identity.
- [x] Complete non-adaptive uniform octree, Morton ordering, `list1`, and
  `list2`.
- [x] P2M, M2M, M2L, L2L, L2P, M2P, and exact P2P reference operators.
- [x] Field-only, potential-only, and combined output on supported CPU paths.
- [x] Independent direct CPU and CUDA references.

### Static geometry architecture

- [x] Immutable P2M maps for fixed sources.
- [x] Shared M2M and L2L maps for the eight child-offset classes per used
  level.
- [x] Level-independent, box-width-normalised M2L transfer classes.
- [x] Fixed-target L2P rows.
- [x] Exact cached `list1` P2P tensors and canonical, SoA, leaf-grouped, and
  BSR(3) execution packings.
- [x] Detailed construction, operator, state, tree, near-field, transfer, and
  persistent-memory statistics.

The static plan is the normal repeated-evaluation path. Geometry-dependent
work is performed during construction wherever practical; subsequent calls
primarily replace the dipole moments.

### Expansion bases

- [x] Complete Cartesian Taylor FMM.
- [x] Real spherical basis, P2M, M2M, dense static M2L, L2L, and L2P.
- [x] FP32 and FP64 operator storage, state, and execution for both point-source
  bases.
- [x] Python basis selection and basis/order/coefficient-count inspection.
- [x] Cartesian-versus-spherical and FMM3D comparison notebooks.

Real spherical harmonics are the default because their `(p+1)^2` coefficients
remove the Laplacian redundancy present in the Cartesian total-degree basis.
That is not a universal performance claim: runtime and accuracy still depend
on order, depth, geometry, precision, and hardware. The Cartesian backend
remains a complete independent formulation, a spherical regression reference,
the implemented basis for finite-cuboid FMM sources, and a useful static-plan
comparison.

`SphericalM2LBackend::StaticDense` is implemented. FMM3D-style exponential or
plane-wave M2L is not a dependency and is only a possible future optimisation:
profile dense spherical M2L at relevant workloads first, and add the extra
representation only if measurements justify its complexity.

### CPU, CUDA, and precision

- [x] Independent CPU reference traversal.
- [x] Portable CPU static execution and class-grouped oneMKL M2L.
- [x] Hybrid CUDA M2L/P2P execution with CPU far-field stages around it.
- [x] Full device-resident CUDA P2M, M2M, M2L, L2L, L2P, and P2P.
- [x] Persistent CUDA geometry, operators, metadata, and scratch.
- [x] Moments-only H2D and final-field-only D2H for repeated CUDA-full field
  evaluation.
- [x] True FP32 and FP64 execution, including operators, expansion state,
  scratch, transfers, and device buffers.

Compatibility aliases and slower execution paths remain intentionally
available for validation, regression testing, and comparative benchmarking.
See [Execution backends](backends.md) and [Numerical precision](precision.md).

## Current capability boundary

Point dipole to point evaluation is supported by Cartesian and spherical
`UniformFmm` plans. Cartesian `UniformFmm` also supports axis-aligned uniform
cuboid sources evaluated at point targets, including finite-cuboid P2M and
exact cuboid-to-point `list1` tensors. Spherical plans reject uniform cuboid
sources.

The lower-level analytical and dense-direct APIs additionally support point or
uniform-cuboid sources and point or volume-averaged-cuboid targets. This does
not yet constitute end-to-end volume-averaged target support in `UniformFmm`.

The tree is complete and uniform, not adaptive. CUDA-full currently provides
the repeated field path; potential output remains on the supported CPU and
hybrid paths. The source/target geometry and any fixed identity map belong to
the plan and require reconstruction when changed.

## Current consolidation work

- [ ] Extend validation from synthetic and comparison cases to representative
  micromagnetic workloads.
- [ ] Improve evidence-based selection of order, depth, precision, and backend
  without changing explicit user choices.
- [ ] Continue measuring Cartesian and spherical accuracy, setup cost,
  repeated runtime, and memory across relevant geometries and hardware.
- [ ] Keep documentation, memory accounting, and manual CUDA validation aligned
  with the implemented plans.

## Next integration milestone

MagTense integration targets its fixed-geometry micromagnetic use case; it no
longer depends on first implementing adaptive trees or basic CUDA support. The
intended sequence is:

```text
complete cuboid -> volume-averaged-cuboid FMM
    -> small stable C ABI
    -> ISO_C_BINDING Fortran wrapper
    -> experimental MagTense demagnetisation backend
    -> field-level validation against existing MagTense tensors
    -> full micromagnetic simulation validation
```

The existing MagTense notebooks and tests are numerical comparisons, not an
integration API.

## Later work

- Complete cuboid-source to volume-averaged-cuboid-target FMM evaluation.
- Add the stable C ABI, Fortran wrapper, and MagTense backend described above.
- Investigate adaptive or sparse trees only where non-uniform populations show
  a practical benefit over the complete tree.
- Investigate exponential spherical M2L, translation symmetry, compression,
  CUDA Graphs, or other execution changes only after profiling identifies a
  bottleneck and end-to-end measurements justify the added complexity.
- Consider versioned disk caches for problem-independent translation operators
  if setup measurements show worthwhile reuse across processes.
- Define and validate true periodic magnetostatics, including the lattice-sum
  convention and magnetic boundary condition, before adding a periodic API.
- Expand Linux/Windows CI, packaging, wheels, release automation, and clean
  installation validation.

These items are research and productisation directions, not current API
promises.
