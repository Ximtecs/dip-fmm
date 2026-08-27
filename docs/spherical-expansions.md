# Real spherical-harmonic expansions

Real spherical harmonics are the default far-field basis for `UniformFmm`
plans. The normative basis definition, solid-harmonic
normalisation, coefficient indexing, P2M and evaluation signs, translation
directions, and M2L box-width scaling are in
[Mathematical formulation](math.md).

An order-`p` plan stores `(p+1)^2` real coefficients ordered by increasing
degree and then `m=-l,...,+l`. `R_00=1`, and the degree-one modes are
`(y,z,x)`. The Cartesian Taylor basis remains available with
`ExpansionBasis::Cartesian` or by assigning `"cartesian"` to the Python
`options.expansion_basis` property.

## Static operator construction

Regular harmonics are generated as homogeneous Cartesian polynomials. Setup
uses the exact identity

$$I_{lm}(r)=\frac{4\pi(-1)^l}{(2l-1)!!}R_{lm}(\nabla)G(r)$$

to project validated Cartesian translation and Laplace-derivative tensors onto
the minimal harmonic subspace. This produces the completed real P2M, M2M, M2L,
L2L, and L2P operators directly. It is algebraically equivalent to a
rotate--axial-shift--rotate-back spherical translation, but neither Cartesian
expansion state nor complex rotation data is retained at runtime.

Point P2M and L2P use analytic regular-harmonic values and gradients. For an
axis-aligned uniform cuboid, setup analytically averages those finite
Cartesian polynomials and their gradients over the source or target volume.
The resulting P2M and L2P operators have spherical width and are stored
directly; repeated evaluation performs neither cuboid integration nor a
Cartesian/spherical conversion. M2M and L2L store eight universal child
templates with exact power-of-two level scaling. M2L stores one dense real matrix
per used integer displacement class and combines it with degree-dependent
box-width scaling. No geometry-dependent harmonic construction or numerical
field sampling occurs during `evaluate()`.

FP64 setup temporaries are quantised and released for an FP32 plan. The
retained static operators, multipoles, locals, scratch, transfers, and CUDA
buffers use the selected execution scalar.

## Backends and limits

Spherical point-dipole and uniform-cuboid sources, with point or
volume-averaged cuboid targets, support FP32 and FP64 CPU static, oneMKL, CUDA
partial, and CUDA full plans. The ordinary spherical M2M, M2L, and L2L
operators are shared by every geometry. Exact near-field P2P tensors are
basis-independent and use the same canonical builder as Cartesian and dense
direct plans. The independent dynamic CPU reference traversal remains
Cartesian-only.

For controlled finite-cell comparisons, `use_cuboid_p2m=false` substitutes
point-dipole P2M and `use_cuboid_l2p=false` substitutes point-evaluation L2P.
Source and target geometry remain unchanged, so exact cuboid P2P physics is
identical across the comparison cases.

The comparison flags also provide a useful hybrid execution model. With
uniform-cuboid sources, volume-averaged cuboid targets, and both flags false,
list1 uses exact cuboid-to-cuboid tensors while the far field uses point P2M
and centre-sampled L2P. Evaluation input remains total dipole moment rather
than magnetisation; no volume scaling is performed implicitly.

`SphericalM2LBackend::StaticDense` is the implemented M2L strategy. FMM3D
v2.1.0 is used only as an external validation and performance comparison.
FMM3D's exponential/plane-wave representation is a possible later optimisation,
not a required component: it should be considered only if profiling relevant
orders and problem sizes shows dense static M2L to be the limiting stage.
