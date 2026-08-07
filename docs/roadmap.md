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

The tree currently organises geometry only; it does not claim to be a complete
FMM evaluator.

## 3. Functional reference uniform FMM — immediate next milestone

Connect the existing operators and geometry without introducing optimisation:

- [ ] leaf P2M;
- [ ] upward M2M pass;
- [ ] M2L accumulation using each target node's `list2`;
- [ ] downward L2L pass;
- [ ] leaf L2P;
- [ ] direct near-field P2P using `list1`;
- [ ] permutation of target results back to user ordering;
- [ ] end-to-end validation against direct P2P;
- [ ] end-to-end examples and benchmarks.

This milestone should establish edge cases such as coincident source/target
sets, empty boxes, tree depth zero, output modes, and self-interaction policy.

## 4. Reference benchmarking — before static-geometry optimisation

Replace the current single P2P smoke timing with repeatable benchmarks for:

- P2M, M2M, M2L, L2L, L2P, and P2P;
- tree construction;
- complete reference FMM evaluation.

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

## 5. Persistent geometry plan / static-geometry optimisation — research

The target use case repeatedly evaluates a fixed geometry for changing dipole
moments.  Source positions do not change; target positions normally do not
change; topology, interaction lists, and expansion order remain fixed.  Only
the three components of each dipole moment change between evaluations.

This is an investigation after the reference FMM is correct and benchmarked,
not an implemented feature or predetermined matrix architecture.

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
