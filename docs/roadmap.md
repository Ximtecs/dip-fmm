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

### Better timing strategy

- [x] keep end-to-end wall time as the primary performance measurement;
- [x] report nested M2L gather, multiply, and scatter timings separately from
  the parent M2L phase without double-counting them in phase shares;
- [ ] quantify timer and OpenMP-region overhead for small transfer classes;
- [ ] record per-sample distributions rather than only aggregate medians;
- [ ] add optional CPU affinity, frequency, and hardware-counter metadata;
- [ ] cross-check internal phase totals against an external profiler on
  representative CPU, oneMKL, and CUDA workloads.

Nested phase timings are diagnostic partitions, not additional top-level work.
Future timing changes should preserve that hierarchy explicitly in CSV schemas,
plots, and generated summaries.

## 5. Persistent geometry plan / static-geometry optimisation — in progress

- [x] static P2M;
- [x] static M2M;
- [x] immutable geometry-dependent M2L plan;
- [x] exact per-level Cartesian M2L matrices;
- [x] canonical backend-independent M2L matrix construction and direct
  operator-level validation against `m2l_add`;
- [x] integer transfer classes and persistent interaction maps;
- [x] grouped contiguous gather/DGEMM/scatter execution;
- [x] OpenMP transfer-class scheduling with single-threaded per-worker DGEMM
  calls;
- [x] portable dense fallback, reference mode, and oneMKL integration;
- [x] setup, memory, gather, multiply, and scatter instrumentation;
- [x] static L2L;
- [x] fixed-target static L2P;
- [ ] static near-field P2P tensor;
- [ ] disk-stored problem-independent M2M, M2L, and L2L matrices indexed by
  expansion order and tree depth;
- [ ] further symmetry and optional tolerance-controlled compression.

The target use case repeatedly evaluates a fixed geometry for changing dipole
moments.  Source positions do not change; target positions normally do not
change; topology, interaction lists, and expansion order remain fixed.  Only
the three components of each dipole moment change between evaluations.

The complete far-field static pipeline is now the default repeated-geometry
path.  The independent reference path remains available for validation.

### Static P2M operator

For fixed source offsets within a leaf, P2M is linear in moment components.
The plan precomputes compact geometry coefficients so repeated evaluation becomes

$$M=P\,m,$$

where $m$ contains the changing dipole components.

### Precomputed M2M translations

A complete uniform octree has only eight child-to-parent relative positions per
level.  Compact M2M translation maps are reused by displacement class within
each level instead of repeatedly forming monomial and factorial terms.

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

Reusable compact parent-to-child maps cover the eight displacement classes at
each level.

### Disk-stored problem-independent translations

M2M, M2L, and L2L translation matrices for a complete uniform tree are
independent of the particle setup.  After normalising the root domain to a
unit cube, their translation classes and scale factors depend only on the
expansion order and tree depth.  Investigate generating these operators once,
serialising them in a versioned binary format, and loading them in later runs
instead of rebuilding them during plan construction.  The cache design should
define precision, coefficient ordering, compatibility metadata, validation,
and portable fallback regeneration when a matching file is unavailable.

P2M and P2P data may also be serialised for repeated runs of exactly the same
geometry, but these operators are problem-dependent and are therefore separate
from the reusable order-and-depth translation cache.

### Fixed-target L2P

When target offsets within leaves are fixed, the plan precomputes the
geometry-dependent rows from local coefficients to target potential and field.

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

### Spherical-expansion alternative

Investigate spherical multipole and local expansions as a future alternative
to, rather than a replacement for, the current Cartesian implementation.  A
decision must compare coefficient count at a given expansion order, accuracy
against Cartesian expansions, M2M/M2L/L2L complexity, CPU and CUDA performance,
static-operator memory, and suitability for the existing fixed-geometry,
static-matrix architecture.  Spherical expansions are roadmap work only; the
directly executable Cartesian operators remain the implemented formulation.

After the reference and persistent-geometry investigations:

1. adaptive-tree support, with interaction rules and balancing designed rather
   than inherited accidentally from the complete tree;
2. CUDA acceleration for profiled P2P and M2L bottlenecks;
3. MagTense integration and a Fortran interface.

### Uniformly magnetised cubes

Extend the present point-dipole source model to uniformly magnetised cubes.
This requires deriving and validating cube-source near- and far-field
operators, including limiting and touching-cell cases, rather than treating a
cube as an undocumented point approximation.  Preserve the dipole operators
as an independently selectable reference capability.

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
- [x] truthful CPU-reference and CPU-static selection API
- [x] persistent device geometry, input, output, stream, and pinned staging buffers
- [x] direct all-to-all CUDA dipole reference
- [x] static M2L matrices and interaction metadata uploaded once
- [x] CUDA M2L gather, cuBLAS multiply, and race-free atomic scatter
- [x] static list-1 P2P tensors uploaded once and evaluated on CUDA
- [x] independent CUDA P2P stream overlapped with the far-field chain
- [x] genuine hybrid CUDA-M2L/P2P FMM and CPU-static comparison benchmarks
- [x] static CUDA P2M, M2M, L2L, and L2P operator application
- [x] full device-resident CUDA FMM
- [x] moments-only H2D repeated evaluation
- [x] field-only D2H repeated evaluation
- [x] persistent CUDA direct-P2P buffers and user-order output
- [x] device source and target permutations
- [ ] custom-versus-cuBLAS M2L microbenchmarks
- [ ] CUDA Graph investigation and retention if measurements justify it
- [x] CPU-static versus CUDA-M2L/P2P problem-size and order sweep infrastructure
- [x] manual CUDA validation workflow (CUDA remains excluded from GitHub CI)

`CudaPartial` (`CudaM2LP2P` compatibility alias) is deliberately hybrid: CPU
P2M/M2M feed packed multipoles to the
resident grouped CUDA M2L plan, downloaded raw locals feed CPU L2L/L2P, and an
independent CUDA stream applies the resident sparse list-1 P2P tensor. `Auto`
remains CPU static. `CudaFull` keeps P2M, M2M, M2L, L2L, L2P, P2P and all
intermediate expansions device-resident. Repeated field evaluations perform one
moments-only H2D transfer and one final-field-only D2H transfer. Static identity
metadata is fixed at initial evaluation and a changed map requires plan rebuild.
`CudaM2L` is retained as a compatibility alias.

All production modes now resolve the same canonical, CPU-built static operator
plan through a per-operator executor selection. Each operator has a portable
CPU executor, an optional oneMKL executor where measurements justify it, and a
CUDA executor. `CudaPartial` selects CUDA only for M2L and P2P and therefore
uses precisely the normal CPU P2M, M2M, L2L, and L2P executors (including the
same future oneMKL improvements when those stages benefit). `CudaFull` selects
CUDA for all six stages. Both CUDA
modes upload the same normalised M2L class table (at most 316 matrices), level
scaling and interaction IDs, and the same canonical sparse P2P tensor; neither
backend derives operator mathematics independently. Device-friendly packing is
an executor concern and does not create a second mathematical plan.

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
# Static-geometry operator roadmap

- [x] static P2M
- [x] static M2M
- [x] static M2L
- [x] static L2L
- [x] static L2P
- [x] static near-field P2P tensor
- [x] CPU static sparse P2P
- [x] CUDA static sparse P2P (custom six-component block kernel)
- [ ] stage-level global sparse P2M (candidate; benchmark required)
- [ ] level sparse M2M (candidate; benchmark required)
- [ ] global sparse M2L comparison (class reuse currently avoids duplication)
- [ ] level sparse L2L (candidate; benchmark required)
- [ ] global sparse L2P (candidate; benchmark required)

## Future research milestone: uniformly magnetised rectangular cells

Feasibility is currently unknown. No source-model architecture should be
committed until both the near- and far-field mathematics have been studied and
validated. The intended research boundary is:

```text
SOURCE MODEL
├── Point dipole
│   ├── near field: dipole tensor
│   └── far field: current dipole P2M
│
└── Uniform rectangular prism
    ├── near field
    │   ├── standard prism tensor
    │   └── volume-averaged prism tensor
    │
    └── far field
        └── finite-prism multipole construction [research required]
```

For the near field, investigate replacing the point-dipole 3 x 3 interaction
tensor with a separate rectangular-prism demagnetisation tensor. The MagTense
`develop` branch is the reference implementation: examine
`source/TileDemagTensor/TileRectangularPrismTensor.f90` for the standard tensor,
which evaluates the field of a uniformly magnetised source prism at a target
position. This is the closest analogue of current P2P. Also examine
`source/TileDemagTensor/TileRectangularPrismAvgTensor.f90` for the recently
added receiving-volume-averaged tensor. The latter implements the definite
integral formulation, including `F1` and `F2`, from Fukushima et al., *Volume
Average Demagnetizing Tensor of Rectangular Prisms*, and may better represent a
finite receiving micromagnetic cell. Research must decide which target
quantity this library and eventual MagTense integration require.

Changing P2P alone must not be assumed sufficient. Far-field research must ask:

- whether the Cartesian expansion represents a finite cuboid exactly to a
  requested order through a modified P2M;
- which uniformly magnetised cuboid multipole moments are non-zero and whether
  they can be calculated analytically during static initialisation;
- whether M2M, M2L, L2L, and L2P remain unchanged after that construction;
- how finite-cell convergence compares with the point-dipole model;
- whether a volume-averaged target also requires a modified L2P; and
- at what separation the point-dipole approximation is already sufficient.

The common M2M/M2L/L2L machinery should ideally remain source-model independent,
but this is a hypothesis to verify mathematically, not a guaranteed design.
No cube or prism operator is implemented by this milestone description.
