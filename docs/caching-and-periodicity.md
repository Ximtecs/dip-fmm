# Caching and periodic boundary conditions

Two features concern the lifetime and the physical model of a plan rather
than a single evaluation: the **persistent cache**, which makes a later
process reconstruct an identical plan from disk, and **fully periodic cubic
cells**, which evaluate the field of an infinite lattice of copies of the
cell. Tutorial 4 demonstrates both.

## What a plan is made of

| State | Depends on | Persisted | Changes between evaluations |
|---|---|---|---|
| universal operator bank: eight M2M and eight L2L templates, the 316-class M2L bank | basis, order, precision | yes | never |
| periodic root operator | basis, order, precision, cell tolerance | yes | never |
| geometry plan: tree topology and permutations, self metadata, P2M, M2L connectivity and level scaling, L2P, exact canonical `list1` P2P tensors | normalised geometry, records, model selectors, identities, depth, periodicity, basis, order, precision | yes | never |
| derived execution packing (SoA rows, leaf blocks, dictionary, BSR), CPU far-field packing, device uploads | backend and packing options | no: rebuilt per process | never |
| moments, expansion coefficients, near/far scratch, results | the `evaluate` call | no | every call |

## The persistent cache

`UniformFmm` normalises every problem to the root cube $[-\tfrac12,\tfrac12]^3$
before constructing its static operators, so the universal bank is
independent of physical placement and scale and one geometry plan is shared by
translated or uniformly scaled copies of a geometry.

The cache is enabled by default. Set `options.enable_cache = false` in C++ or
Python, or `CDFMM_DISABLE_CACHE=1`, to force analytical reconstruction. The
default base directory is the source repository's `caches/` (compiled in as an
absolute path so that other processes on the same machine share it, and
ignored by Git); `CDFMM_CACHE_DIR` overrides it and is read at every plan
construction. Files use this structure:

```text
v1/
  universal/operators_spherical_p06_f32_m2m-m2l-l2l_v02.bin
  periodic/periodic_spherical_p06_f32_zerok0_tol1e-12_v02.bin
  plans/plan_spherical_p06_d05_f32_<sha256>_v02.bin
```

Universal files contain the eight M2M templates, the standard 316-class M2L
bank, and the eight L2L templates. Periodic root operators are separate
because their setup tolerance is independently selectable. Geometry files
contain tree topology and Morton permutations, self metadata, P2M, M2L
connectivity and level scaling, L2P, and exact canonical P2P data; they do not
duplicate the universal matrices. The selected execution packing is
deliberately not stored: each process derives it for its backend, and CUDA
uploads that derived representation to the device. FP32 geometry files are
decoded directly into canonical FP32 operators, without widening to FP64.

The geometry SHA-256 covers canonical source and target geometry, prism
dimensions, representative-relative tetrahedron vertices, all four near/far
model selectors, self identities, depth, periodicity, basis, order, precision,
and cache/math versions. Physical root centre, root length, and changing
moments are excluded. Complete regular grids use a validated compact
descriptor (dimensions, canonical origin, spacing, enumeration layout) instead
of hashing every coordinate. A plan built on a supplied `AdaptiveTree`
topology is not cached.

Each binary file has a validated header (magic, schema/math versions,
endianness and data-model markers, kind, basis, order, precision, applicable
depth, one section offset and size, the exact cache key, the geometry digest
where applicable, and a fast 64-bit payload checksum). Writers use a unique
temporary file, `fsync`, and an atomic rename, so independent processes may
race for the same file. Missing, truncated, corrupt, or incompatible files are
ignored and rebuilt; **a cache failure cannot alter an evaluation result**, and
cold and warm plans report the same memory and statistics. The persistent
format and the key strings are compatibility contracts
(`src/cache/AGENTS.md`).

### Inspecting and precomputing

Construction diagnostics and Python's `static_plan_statistics` report phase
times, hit states (`universal_cache_hit`, `periodic_cache_hit`,
`geometry_cache_hit`) and validated bytes read or written. Cache keys are
exposed as `universal_cache_key`, `periodic_cache_key`, and
`geometry_cache_key`.

The installed `cdfmm-precompute` utility populates universal files without a
physical experiment:

```console
cdfmm-precompute --basis spherical --orders 4,6,8 --precision f32
cdfmm-precompute --basis spherical --orders 6 --precision f32 \
  --periodic --periodic-tolerance 1e-12
```

The universal bank is the dominant truly cold cost (about 1.3 s at $p=6$,
12 s at $p=8$ and 75 s at $p=10$ on eight cores) and is removed entirely by
its cache; on a warm hit the remaining setup cost is the derived packing and,
for FP32 plans, the precision conversion.

## Fully periodic cubic cells

The periodic API requires an explicit fundamental cell. The supported mode is
deliberately narrow: all three axes are periodic and the cell is a cube.
Enabling periodicity makes the cell centre and side length the FMM root
centre and width; the period is never inferred from particle bounds, and every
position must lie inside the cell (positions are wrapped into the half-open
fundamental cell during construction). Partial periodicity and rectangular
cells are rejected so that later extensions can be added without giving
existing inputs ambiguous meanings.

```python
options.periodic.enabled = True
options.periodic.axes = [True, True, True]
options.periodic.centre = cdfmm.Vec3(0.0, 0.0, 0.0)
options.periodic.lengths = cdfmm.Vec3(L, L, L)
options.periodic.convention = cdfmm.PeriodicConvention.ZeroK0
options.periodic.setup_tolerance = 1.0e-12
```

### The zero-$k$ convention

A three-dimensional dipolar lattice sum is conditionally convergent, so the
word "periodic" is not by itself a physical specification. The
`PeriodicConvention::ZeroK0` convention omits the reciprocal $k=0$ term and
adds no macroscopic surface or demagnetising term. Its physical content is
that the cell-average field vanishes:

- an exactly uniform magnetisation represented by **finite bodies filling the
  cell** has $H = 0$ (the $-M/3$ self field of each cube cancels the lattice
  term), the "no demagnetising field" property of the convention;
- the same magnetisation represented by **point dipoles at the sites of a
  cubic lattice** gives the Lorentz local field $H = +M/3$ at every site,
  which is why point and finite discretisations of a continuum must not be
  mixed up.

Tutorial 4 reproduces both numbers to four digits.

### Wrapped topology

Periodic boxes are represented by a central-tree node and a three-component
integer image shift. An unwrapped coordinate $q$ at level $l$ is converted by
mathematical floor division into a coordinate modulo $2^l$ and an image shift;
the complete pair is the identity, so node indices alone must not be used for
deduplication, particularly in shallow trees. `list1` is the periodic
$3\times3\times3$ neighbourhood and `list2` is
`children(parent list1) - node list1`; the tree and particle arrays are not
replicated, and child displacement classes stay in the ordinary $[-3,3]^3$
set.

The static plan consumes these image identities directly. `list1` constructs
exact shifted point or finite-body P2P tensors; `list2` maps wrapped boxes onto
the ordinary translation-class matrices. A point self identity removes only
the singular zero-shift pair; every non-zero image of the same particle
remains physical, and finite self fields remain included. Periodic image
records are ordinary stored tensors to every execution packing.

### The Ewald root periodiser

Wrapped traversal covers the central root and its 26 neighbouring images. All
remaining images are represented by one dense root-multipole-to-root-local
matrix. For cell volume $V=L^3$ the setup uses

$$
G_{\mathrm{per}}(r) =
\sum_{n\in\mathbb{Z}^3}
\frac{\operatorname{erfc}(\alpha|r+nL|)}{4\pi|r+nL|}
+ \frac{1}{V}\sum_{k\ne0}
\frac{e^{-|k|^2/(4\alpha^2)}\cos(k\cdot r)}{|k|^2}
- \frac{1}{4\alpha^2V},
\qquad \alpha=\frac{\sqrt{\pi}}{L}.
$$

At coincidence, the singular central real-space term is replaced by its
regular part $-\operatorname{erf}(\alpha r)/(4\pi r)$. Setup constructs the
required Cartesian derivatives of this periodic Laplace Green function with
balanced real and reciprocal Ewald sums, omits reciprocal $k=0$, subtracts the
26 explicitly traversed free-space root translations, and projects the result
into the selected Cartesian or real spherical basis. `setup_tolerance`
controls the exponentially decaying Ewald cut-offs. The matrix is normalised
to unit root width, so the ordinary M2L degree scaling applies to any physical
cell size; the root contribution is accumulated before L2L propagation. FP32
plans build it in FP64 and quantise it with the other static operators; it is
cached under the periodic key.

### Backends and output

The canonical periodic plan is shared by portable CPU, oneMKL, `CudaPartial`
and `CudaFull` execution. `CudaFull` is field-only; select the static CPU or
hybrid backend when the scalar potential is required. The `CpuReference`
backend is unavailable for periodic plans because it has no image-aware
operator plan. `dipole_moments` remain total moments, including for finite
sources.
