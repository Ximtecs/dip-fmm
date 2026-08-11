# Roadmap

The roadmap deliberately keeps the straightforward CPU mathematics as the
correctness baseline before introducing traversal or performance complexity.

## 1. CPU operator layer — complete

- [x] Cartesian multi-index and coefficient representation.
- [x] Taylor-jet Laplace derivatives.
- [x] P2M, M2M, M2L, L2L, L2P, and direct P2P.
- [x] M2P as an independent multipole-validation path.
- [x] Analytic, consistency, and convergence tests.
- [x] Initial Python bindings for direct evaluation.

## 2. Complete uniform-tree geometry — complete

- [x] Cubic root determination and complete level allocation.
- [x] Morton encoding, flat level-wise indexing, parents, and children.
- [x] Independent source and target sorting with forward and inverse
  permutations.
- [x] Leaf assignment and hierarchical point ranges.
- [x] Uniform `list1` near neighbours and `list2` M2L interactions.

The tree organises geometry and interaction lists used by the reference FMM.

## 3. Functional reference uniform FMM — complete

Connect the existing operators and geometry without introducing optimisation:

- [x] leaf P2M;
- [x] upward M2M pass;
- [x] M2L accumulation using each target node's `list2`;
- [x] downward L2L pass;
- [x] leaf L2P;
- [x] direct near-field P2P using `list1`;
- [x] permutation of target results back to user ordering;
- [x] end-to-end validation against direct P2P;
- [x] end-to-end examples and smoke benchmark.

This milestone should establish edge cases such as coincident source/target
sets, empty boxes, tree depth zero, output modes, and self-interaction policy.

## 4. CPU performance and reference benchmarking — complete

Implemented repeatable, OpenMP-aware phase and end-to-end benchmarks for:

- [x] P2M, M2M, M2L, L2L, L2P, and parallel direct P2P;
- [x] timed, partially parallel tree construction;
- [x] complete uniform-FMM evaluation, repeated geometry, accuracy, and scaling sweeps;
- [x] machine-readable results and automatically generated figures.

Sweep expansion order, source count, tree depth, and repeated evaluations on
unchanged geometry.  Record accuracy alongside time where truncation order is
varied.  For repeated-use experiments, report two distinct quantities:

```text
setup time
per-evaluation time
```

This distinction is essential: a more expensive one-time geometry plan can be
valuable if it substantially reduces thousands of later magnetic-field
evaluations.  Kernel profiles and end-to-end measurements should determine
which optimisation work proceeds.

## 5. Persistent geometry plan / static-geometry optimisation — in progress

- [x] immutable geometry-dependent M2L plan;
- [x] exact per-level Cartesian M2L matrices;
- [x] integer transfer classes and persistent interaction maps;
- [x] grouped contiguous gather/DGEMM/scatter execution;
- [x] portable dense fallback, reference mode, and oneMKL integration;
- [x] setup, memory, gather, multiply, and scatter instrumentation;
- [ ] static P2M, M2M, L2L, and fixed-target L2P;
- [ ] symmetry and optional tolerance-controlled compression.

The target use case repeatedly evaluates a fixed geometry for changing dipole
moments.  Source positions do not change; target positions normally do not
change; topology, interaction lists, and expansion order remain fixed.  Only
the three components of each dipole moment change between evaluations.

Dense static M2L is now the default repeated-geometry path. The remaining
operators and alternative representations remain investigations.

### Static P2M operator

For fixed source offsets within a leaf, P2M is linear in moment components.
Investigate precomputing geometry coefficients so repeated evaluation becomes

$$M=P\,m,$$

where $m$ contains the changing dipole components.  Compare an explicit matrix
with blocked or generated loops, including its setup and storage costs.

### Precomputed M2M translations

A complete uniform octree has only eight child-to-parent relative positions per
level.  Investigate reusable M2M translation matrices instead of repeatedly
forming monomial and factorial terms.  Determine whether scale-normalised
coefficients permit reuse across levels without harming conditioning.

### Precomputed M2L translations

M2L uses

$$R=c_{target}-c_{source},\qquad
L_\beta\mathrel{+}=\sum_\alpha T_{\beta,\alpha}(R)M_\alpha,
\qquad T_{\beta,\alpha}(R)=D_{\alpha+\beta}G(R).$$

A fixed uniform tree has a finite set of relative integer box offsets.
Investigate:

- precomputed Laplace derivative tensors and explicit translation matrices;
- caches indexed by relative integer box offset;
- reuse by all box pairs with the same relative geometry;
- Laplace-kernel homogeneity to avoid unnecessary cross-level duplication;
- symmetry reductions where their complexity is worthwhile.

### Precomputed L2L translations

Investigate reusable parent-to-child matrices for the eight displacement
classes, including the same scale-normalisation question as M2M.

### Fixed-target L2P

When target offsets within leaves are fixed, investigate precomputing the
geometry-dependent map from local coefficients to target potential and field.

### Optional near-field geometry caching

For fixed positions a P2P pair is linear in its source moment through a
geometry-dependent $3\times3$ dipole interaction tensor.  Caching those tensors
may reduce repeated arithmetic, but storing every near-field pair can have a
significant memory cost.  Pursue it only if profiling shows an end-to-end
benefit.

### Optimised linear algebra

Benchmark alternative application strategies rather than assuming one BLAS
GEMV per translation is faster.  Candidate approaches include:

- specialised C++ loops;
- SIMD/vectorised kernels;
- BLAS GEMV or batched GEMV;
- BLAS GEMM after grouping translations that share an operator;
- an optional linear-algebra backend.

Cartesian matrices are relatively small, so call overhead and memory traffic
may dominate.  Grouping many boxes with the same operator into matrix-matrix
work may improve arithmetic intensity and make mature BLAS implementations
effective.  The choice must remain a benchmark result, not a design assumption.

### Reference implementation requirement

Retain the straightforward mathematical operators as a reference path where
practical.  Test every optimised operator numerically against that path, and
support performance changes with kernel and end-to-end benchmarks.

## 6. Later capabilities

After the reference and persistent-geometry investigations:

1. adaptive-tree support, with interaction rules and balancing designed rather
   than inherited accidentally from the complete tree;
2. CUDA acceleration for profiled P2P and M2L bottlenecks;
3. MagTense integration and a Fortran interface.

These phases are intentionally outside the current CPU reference scope.

## Cross-platform build and packaging

Linux is the current development path.  Windows is intended support, but a
clean Windows environment and toolchain have not yet been validated and there
are currently no Linux/Windows CI claims.  Support will mean continuous
validation rather than a one-off successful build:

- [ ] Verify clean Conda environment creation on Linux.
- [ ] Verify clean Conda environment creation on Windows.
- [ ] Verify C++20 compilation on Linux.
- [ ] Verify C++20 compilation on Windows.
- [ ] Verify the C++ test suite on both platforms.
- [ ] Verify Python extension build and installation on both platforms.
- [ ] Verify the Python tests on both platforms.
- [ ] Verify documentation builds on both platforms where practical.
- [ ] Add GitHub Actions jobs for Linux and Windows.
- [ ] In each OS job, configure, build, run C++ tests, build Python, and run
  Python tests.
- [ ] Keep paths and documented commands platform-independent where possible.
- [ ] Document unavoidable Windows SDK or toolchain prerequisites.
- [ ] Test installation on completely clean machines or CI runners.

## Distribution and releases

Once source-build portability is continuously tested, investigate:

- [ ] Linux and Windows Python wheels built automatically in CI;
- [ ] whether `cibuildwheel` is appropriate for the native extension;
- [ ] tagged GitHub releases and version/release automation;
- [ ] publication of the Python package;
- [ ] a Conda package and conda-forge feedstock if user demand warrants them.

These are future shipping goals, not capabilities of the current source-only
development workflow.
# CUDA acceleration

- [x] optional `CDFMM_ENABLE_CUDA` build and supplemental Conda environment
- [x] explicit CPU-reference, CPU-static, CUDA-M2L, and CUDA-full selection API
- [x] persistent device geometry, input, output, stream, and pinned staging buffers
- [x] minimal field-only steady-state transfer accounting
- [ ] transfer-class grouped CUDA M2L and resident Cartesian operators
- [ ] device P2M, M2M, L2L, and L2P kernels
- [x] device near-field/direct dipole kernel and user-order output
- [ ] device Morton permutations
- [ ] custom-versus-cuBLAS M2L microbenchmarks
- [ ] CUDA Graph investigation and retention if measurements justify it
- [ ] CPU-static versus hybrid versus full-CUDA crossover study
- [x] manual CUDA validation workflow (CUDA remains excluded from GitHub CI)

The first CUDA landing establishes the optional toolchain, persistent ownership,
strict field-only moment/result transfer path, backend API, and exact device
dipole kernel. The unchecked items are deliberately not represented as complete:
the direct device path is a correctness baseline while the Cartesian FMM stages
and genuine M2L-only transfer path are implemented and profiled.

## Future milestone: true periodic magnetostatics

Periodic boundaries must cover one periodic axis, two-dimensional slabs, and
fully three-dimensional cells. A finite collection of manually replicated image
cells is useful for tests but is **not** true infinite periodicity. Before an API
or implementation is accepted, it must define the unit-cell lattice vectors,
periodic axes, lattice-sum convention, zero/far mode, macroscopic magnetic
boundary condition, and treatment of mean magnetisation or net cell dipole.
Electrostatic neutrality rules must not be assumed to apply unchanged to
magnetostatics.

The leading architecture to investigate is a small near-image region evaluated
by the existing free-space FMM plus a precomputed operator representing every
remaining image. That operator would act as a root/local M2L-like correction
before the ordinary downward pass. It depends on lattice, periodic axes, kernel,
and order but not changing moments, making it suitable for the CPU static plan,
persistent GPU residency, and repeated application. The Cartesian dipole/Taylor
operator must be derived rather than copied from a kernel-independent formula.

Validation must include small explicit image sums, independent Ewald/reference
calculations, analytic and symmetry cases, translation invariance within the
cell, matching across opposite faces, 1-D/2-D/3-D cases, convergence in FMM
order, and convergence of the far-periodic construction. Micromagnetic tests
must detect accidental finite-sample shape effects when true periodicity is
requested.

Literature starting points are:

- Lambert, Darden & Board, *JCP* 126 (1996), DOI `10.1006/jcph.1996.0137`.
- Challacombe, White & Head-Gordon, *JCP* 107 (1997), DOI `10.1063/1.474150`.
- Kudin & Scuseria, *CPL* 283 (1998), DOI `10.1016/S0009-2614(97)01329-8`;
  and *JCP* 121 (2004), DOI `10.1063/1.1771634`.
- Apalkov & Visscher, periodic micromagnetic FMM with a Taylor representation
  of the distant infinite array, *IEEE Trans. Magn.* 39 (2003), DOI
  `10.1109/TMAG.2003.819461`.
- Yan & Shelley, periodic M2L separation, arXiv `1705.02043`.
- Pei, Askham, Greengard & Jiang, arbitrary 2-D lattices, *JCP* 474 (2023),
  DOI `10.1016/j.jcp.2022.111792`.
- Lebecki, Donahue & Gutowski, *J. Phys. D* 41 (2008), DOI
  `10.1088/0022-3727/41/17/175005`; Wysocki & Antropov, *JMMM* 428 (2017),
  DOI `10.1016/j.jmmm.2016.11.128`; and Bruckner et al., *Scientific Reports*
  11 (2021), DOI `10.1038/s41598-021-88541-9`, as independent micromagnetic
  periodic validation approaches.
