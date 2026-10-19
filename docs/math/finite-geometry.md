# Finite geometry: prisms, tetrahedra, averaging and self fields

`cdfmm` treats a uniformly magnetised axis-aligned rectangular prism and a
uniformly magnetised tetrahedron as **exact** sources and targets. This page
defines the finite operators, the volume-averaged target, the analytical pair
tensors, the safeguards that keep them well conditioned, and the
self-interaction semantics. Point-dipole definitions are in
[Mathematical conventions](conventions.md).

## Moments and records

Runtime data always remains the **total magnetic moment** $m=VM$ of a body of
volume $V$ and uniform magnetisation $M$; geometry plans absorb source-volume
normalisation, so every direct evaluation has the common form $H=Km$ and no
volume scaling is performed implicitly.

An axis-aligned rectangular prism is described by its representative position
(its centre) and full side lengths $h=(h_x,h_y,h_z)$, volume $V=h_xh_yh_z$. A
tetrahedron is described by its representative position and four vertices
relative to it; the representative is normally the centroid, so one record can
be shared by every body of a regular mesh. Records may be supplied once for
all bodies or once per body in user order ([Geometry](../geometry.md)).

## Finite P2M and L2P (Cartesian form)

For $\beta\in\mathbb N_0^3$, define the factorial-normalised prism average

$$J_\beta(d,h)=\frac1V\int_V\frac{(d+u)^\beta}{\beta!}\,du.$$

Its exact finite sum contains only component-wise even $\gamma\le\beta$:

$$J_\beta=\sum_\gamma\frac{d^{\beta-\gamma}}{(\beta-\gamma)!}
\prod_q\frac{h_q^{\gamma_q}}{2^{\gamma_q}(\gamma_q+1)!}.$$

Prism P2M replaces each point monomial in the P2M equation by
$J_{\alpha-e_k}$; volume-averaged L2P similarly replaces its potential row by
$J_\beta$ and its field row by $-J_{\beta-e_k}$. M2M, M2L and L2L are
unchanged because translations act on the resulting expansion coefficients.
Tetrahedral P2M and volume-averaged L2P use the exact simplex moments in the
same way. The spherical static operators are obtained from these Cartesian
averages by the projection described in
[Real spherical-harmonic expansions](spherical-expansions.md), so both bases
support finite bodies at every far-field stage.

Construction reuses these moments across all operator rows for a body record.
For a prism, the implementation tabulates each one-dimensional average once
and combines the three axes in the same order as the scalar formula; the table
matches that formula bit for bit. Tetrahedra use a corresponding simplex
moment table, also shared by Cartesian and spherical endpoints.

A **point target** evaluates the finite source at the receiving representative
position; a **finite target** receives the exact average of the field over its
own volume through the target-geometry operators.

## Exact pair tensors

Direct geometry stores exactly the six symmetric Cartesian components
$K_{xx},K_{xy},K_{xz},K_{yy},K_{yz},K_{zz}$ of the pair tensor, each an
$N_t\times N_s$ matrix in the dense plan; nine GEMVs apply these six matrices
to the three packed moment components. The same tensors are the `list1`
near-field operator of the FMM.

### Rectangular prisms

The prism corner formulas and finite-volume Newell primitives are direct
adaptations of MagTense's `getN_prism_3D` and `getAvgN_prism_3D`/`F1`/`F2`
analytical formulas, without copying GPL source code. In MagTense's
demagnetisation-tensor notation $N$ the source-volume operator is $K=N/V_s$;
because callers provide the total moment $m=V_sM$, the stored tensor maps $m$
directly to the field. MagTense stores its tensor and applies the physical
minus sign during the matrix-vector operation; `cdfmm`'s tensor maps total
moments directly to the signed field $H=-\nabla\phi$ with $1/(4\pi)$
normalisation. CUDA consumes the precomputed tensors and performs no prism
integration at runtime. The prism point tensor is implemented once in a
precision-generic kernel that production evaluates in `long double`; its
logarithm, inverse hyperbolic sine, arctangent and square-root primitives are
cancellation sensitive, which is why the analytical construction stays in
extended precision even for FP32 plans.

The prism-to-prism tensor is the target-box integral of the source's corner
sum: along each axis the `F1`/`F2` primitives are taken at
$d \pm h_t \mp h_s$ with weight $t(-s)$ for the target and source corner
signs $t, s$, and a point's weight is the product over the three axes. At
each point the six components share one distance, three arctangents and six
logarithms, which are evaluated once. When the two prisms have the same
half-width along an axis, two of that axis's four points coincide at $d$, so
the axis reduces to three points with the second-difference weights
$(1, -2, 1)$: 27 point evaluations for two equal prisms instead of 64. Both
forms are the same sum with identical terms combined before rounding rather
than after; they agree to `double` rounding (about $10^{-15}$ of the largest
field component) and are not bitwise identical.

For a centred cube the degree-three correction is proportional to
$D_x^2+D_y^2+D_z^2=\nabla^2$ and vanishes outside the source. The first
physical shape correction of a cube is therefore multipole degree five, with
relative scale $O((h/R)^4)$; a general non-cubic prism does not have this
cancellation. This is why, for cubes, finite P2M/L2P cannot improve on the
point operators below order five.

### Tetrahedra and mixed pairs

Finite tetrahedron P2M and L2P construction integrates all required Cartesian
monomials together. For representative-relative vertices $v_a$, displacement
$d$, and $n=|\alpha|$, the Dirichlet simplex formula gives

$$J_\alpha(d)=\frac{6}{(n+3)!}[t^\alpha]
\prod_{a=0}^{3}\frac{1}{1-(d+v_a)\cdot t}.$$

Each factor updates its coefficients in total-degree order using
$C_{\mathrm{new},\alpha}=C_{\mathrm{old},\alpha}+
\sum_k(d+v_a)_k C_{\mathrm{new},\alpha-e_k}$, with negative-index terms
absent. This computes the complete table in $O(p^3)$ time and memory.
Construction accumulates in `long double` and converts the final averages
to `double` before assembling the existing Cartesian or spherical operators.
The public scalar `tetrahedron_averaged_monomial` remains the independent
barycentric expansion reference. Both compute the same exact volume average;
their floating-point summation orders differ.

The tetrahedron-to-tetrahedron, prism-to-tetrahedron and tetrahedron-to-prism
pairs are evaluated during dense or static-plan construction from the same
exact polyhedron surface formulation: applying the divergence theorem to both
uniformly magnetised bodies gives

$$K=-\frac{1}{4\pi V_sV_t}\sum_{f_t}\sum_{f_s}\hat n_t\hat n_s^{T}
\iint\frac{dS_t\,dS_s}{|x_t-x_s|},$$

where the prism contributes its twelve boundary triangles and the tetrahedron
its four; the triangle-pair integral is the analytical Galerkin integral of
Gumerov, Kaneko and Duraiswami (2024). Tetrahedron-to-point and
point-to-tetrahedron tensors use the analytical point field of a uniformly
magnetised polyhedron.

Four numerical safeguards keep these analytical kernels well conditioned:

- the point-field edge primitives use the closed form of the atanh
  difference in which the logarithms of the perpendicular edge distance
  cancel, and the normal component is the signed solid angle of the face
  (Van Oosterom–Strackee), so evaluation points on the line through an edge
  outside the body are regular;
- from 1.5 summed circumradii of separation a body pair averages the exact
  source point tensor over the target by collapsed Gauss quadrature instead
  of forming the surface integrals, with seven points per direction from 1.5,
  six from 2 and five from 4 circumradii. The analytical reduction is
  ill-conditioned for separated irregular bodies well before it loses all
  digits: on random irregular tetrahedra its error grows from a median of
  3e-11 at 1.5 circumradii to 2e-8 at eight, with outliers up to 1.5e-5 where
  edges are nearly parallel, whereas each rule of the ladder stays below
  about 2e-10 in its range (6e-11 measured over random pairs from three to
  eight circumradii). The switch is continuous to that precision and is
  exercised by the tests;
- the triangle-pair reduction is exact in exact arithmetic, but its terms
  carry coefficients that grow without bound when the geometry is nearly
  degenerate (nearly parallel planes, nearly collinear or coplanar edge
  vectors), and it drops or snaps those directions with fixed tolerances. In
  double precision this lost every digit for parallel faces with non-aligned
  edges (the top and bottom faces of an unstructured film mesh: errors up to
  24 times the tensor against an independent Sauter–Schwab quadrature on a
  440-pair validation set). A geometric screen (the smallest non-exact
  measure among the angle between the normals, between an edge and the other
  plane, between two edges, and the height of a vertex over the other plane)
  leaves well-conditioned face pairs, those with every such measure above
  1e-2, to the former double-precision reduction, which is accurate to
  4e-11 on them. All other face pairs are reduced in double-double
  arithmetic (about 106 bits, with the transcendental functions in
  `_Float128`; `long double` where the compiler has no `_Float128`) with
  tolerances chosen for that precision: coefficients below 1e-22 drop their term,
  heights below 1e-12 of the pair's scale are zero, an expansion direction
  whose squared independent part is below 1e-23 of the basis scale is
  dependent, and face pairs within 1e-8 of parallel are projected onto
  parallel planes and evaluated by the parallel branch. A conforming mesh
  whose coordinates the FMM rounds still evaluates as the exactly conforming
  one; and
- where the reduction is unreliable whatever the precision, the face pair is
  integrated numerically in double precision instead (the integrand is
  positive, so these quadratures do not cancel): a rank decision whose
  squared independent part lies within three decades of the rank tolerance
  sends a touching face pair to Sauter–Schwab quadrature (Sauter and Schwab,
  *Boundary Element Methods*, Sec. 5.2; 24 Gauss points per direction of the
  transformed four-dimensional domain) and a separated one to Gauss
  quadrature with splitting; separated faces in nearly parallel planes
  (1e-14 to 1e-2) go to the latter directly. Vertices within 1e-6 of the
  larger edge count as common, and faces count as separated only when every
  vertex pair is further apart than 1e-3 of it, so a rounded conforming mesh
  keeps the closed form for its nearly touching faces. On the validation set
  the worst tensor error is 5e-9, limited by the reference. Cold
  construction costs the same as before on a regular lattice, whose
  degeneracies are exact, and about nine times as much on an irregular film
  mesh, most of it in the `_Float128` transcendental functions.

For a tetrahedron point evaluation on a face, the analytical boundary value
uses the MagTense-compatible one-sided limiting convention; edge and vertex
coincidences remain singular.

## Self interaction

Point self interactions are singular and are zeroed **only** through the
explicit identity map: coincident coordinates without a map are two different
particles. Finite self interactions are physical and are always included; a
cube gives $H=-M/3$ at its own centre and as its volume average. The
canonical near-field operator marks a pair `skip_for_identity` only when the
source is a point dipole, so every execution packing applies the same rule
without inspecting geometry ([Architecture](../architecture.md)).

## Where these formulas live

| Quantity | Implementation |
|---|---|
| prism average $J_\beta$ and prism P2M/L2P | `src/geometry/primitives/rectangular_prism.cpp`, `src/operators/{p2m,l2p}.cpp` |
| prism point and prism-to-prism tensors | `src/geometry/primitives/rectangular_prism_point_kernel.hpp`, `rectangular_prism.cpp` |
| tetrahedron scalar moments, point field, pair tensors | `src/geometry/primitives/tetrahedron.cpp` |
| tetrahedron moment table for finite P2M/L2P | `src/geometry/primitives/tetrahedron_moments.cpp` |
| pair dispatch for all nine combinations | `src/operators/p2p.cpp` |
| dense all-to-all plan | `src/plan/direct/dense.cpp` |

The trust anchors are `tests/test_rectangular_prism_magtense.cpp` (an
independent re-derivation of the MagTense F1/F2 primitives) and
`tests/test_tetrahedron.cpp` (reciprocity, symmetry, singular limits,
quadrature cross-checks and far-separation continuity).
