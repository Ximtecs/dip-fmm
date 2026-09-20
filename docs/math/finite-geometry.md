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

For a centred cube the degree-three correction is proportional to
$D_x^2+D_y^2+D_z^2=\nabla^2$ and vanishes outside the source. The first
physical shape correction of a cube is therefore multipole degree five, with
relative scale $O((h/R)^4)$; a general non-cubic prism does not have this
cancellation. This is why, for cubes, finite P2M/L2P cannot improve on the
point operators below order five.

### Tetrahedra and mixed pairs

The tetrahedron-to-tetrahedron, prism-to-tetrahedron and tetrahedron-to-prism
pairs are evaluated exactly during dense or static-plan construction from the
same polyhedron surface formulation: applying the divergence theorem to both
uniformly magnetised bodies gives

$$K=-\frac{1}{4\pi V_sV_t}\sum_{f_t}\sum_{f_s}\hat n_t\hat n_s^{T}
\iint\frac{dS_t\,dS_s}{|x_t-x_s|},$$

where the prism contributes its twelve boundary triangles and the tetrahedron
its four; the triangle-pair integral is the analytical Galerkin integral of
Gumerov, Kaneko and Duraiswami (2024). Tetrahedron-to-point and
point-to-tetrahedron tensors use the analytical point field of a uniformly
magnetised polyhedron.

Two numerical safeguards keep these analytical kernels well conditioned:

- the point-field edge primitives use the closed form of the atanh
  difference in which the logarithms of the perpendicular edge distance
  cancel, and the normal component is the signed solid angle of the face
  (Van Oosterom–Strackee), so evaluation points on the line through an edge
  outside the body are regular; and
- beyond eight summed circumradii of separation a body pair averages the
  exact source point tensor over the target with a six-point Gauss rule
  instead of cancelling large face integrals (the surface form loses about
  five digits at fifty body sizes and all of them at one hundred). The switch
  is continuous to the working precision and is exercised by the tests.

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
| tetrahedron moments, point field, pair tensors | `src/geometry/primitives/tetrahedron.cpp` |
| pair dispatch for all nine combinations | `src/operators/p2p.cpp` |
| dense all-to-all plan | `src/plan/direct/dense.cpp` |

The trust anchors are `tests/test_rectangular_prism_magtense.cpp` (an
independent re-derivation of the MagTense F1/F2 primitives) and
`tests/test_tetrahedron.cpp` (reciprocity, symmetry, singular limits,
quadrature cross-checks and far-separation continuity).
