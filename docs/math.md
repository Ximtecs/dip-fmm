# Mathematical formulation

This page is the normative statement of the signs, displacement directions,
and coefficient normalisation used by the implementation.

## Kernel, dipole potential, and field

The Laplace Green function is

$$G(r)=\frac{1}{4\pi |r|}.$$

For source position $x_j$, dipole moment $m_j$, and target $x$, let
$r_j=x-x_j$.  The scalar potential and magnetic field are

$$\phi(x)=\sum_j\frac{m_j\cdot r_j}{4\pi |r_j|^3},
\qquad H(x)=-\nabla\phi(x).$$

Thus one direct pair contributes

$$H_{ij}=\frac{1}{4\pi}\left[
\frac{3r_{ij}(m_j\cdot r_{ij})}{|r_{ij}|^5}
-\frac{m_j}{|r_{ij}|^3}\right],
\qquad r_{ij}=x_i-x_j.$$

The kernel is singular at zero separation.  A source-point evaluation must
explicitly skip its self-interaction.

## Finite cuboid geometry

An axis-aligned uniformly magnetised cuboid has side lengths
$h=(h_x,h_y,h_z)$ and volume $V=h_xh_yh_z$. Runtime data always remains the
**total magnetic moment** $m=VM$; geometry plans absorb source-volume
normalisation, so every direct evaluation has the common form $H=Km$.
Sources and targets are selected independently as point dipoles or uniform
cuboids, and points or volume-averaged cuboids, respectively.

For $\beta\in\mathbb N_0^3$, define the factorial-normalised cuboid average

$$J_\beta(d,h)=\frac1V\int_V\frac{(d+u)^\beta}{\beta!}\,du.$$

Its exact finite sum contains only component-wise even $\gamma\le\beta$:

$$J_\beta=\sum_\gamma\frac{d^{\beta-\gamma}}{(\beta-\gamma)!}
\prod_q\frac{h_q^{\gamma_q}}{2^{\gamma_q}(\gamma_q+1)!}.$$

Cuboid P2M replaces each point monomial in the P2M equation by
$J_{\alpha-e_k}$. Volume-averaged L2P similarly replaces its potential row by
$J_\beta$ and field row by $-J_{\beta-e_k}$. M2M, M2L and L2L are unchanged
because translations act on the resulting Cartesian expansion coefficients.

Direct geometry stores exactly the six symmetric Cartesian components
$K_{xx},K_{xy},K_{xz},K_{yy},K_{yz},K_{zz}$, each an $N_t\times N_s$ matrix.
Nine GEMVs apply these six matrices to the three packed moment components.
Point self interactions are zeroed only through explicit identity; finite
cuboid self interactions are included (a cube gives $H=-M/3$). The prism
corner formulas and finite-volume Newell primitives are independently
implemented from published analytical results and are validated against the
MagTense `getN_prism_3D` and `getAvgN_prism_3D` conventions without copying
GPL source code.

For a centred cube the degree-three correction is proportional to
$D_x^2+D_y^2+D_z^2=\nabla^2$ and vanishes outside the source. The first
physical shape correction is therefore multipole degree five, with relative
scale $O((h/R)^4)$; a general non-cubic cuboid does not have this cancellation.

## Cartesian multi-indices

For $\alpha=(\alpha_x,\alpha_y,\alpha_z)\in\mathbb N_0^3$,

$$|\alpha|=\alpha_x+\alpha_y+\alpha_z,\quad
\alpha!=\alpha_x!\alpha_y!\alpha_z!,\quad
r^\alpha=r_x^{\alpha_x}r_y^{\alpha_y}r_z^{\alpha_z}.$$

$e_x,e_y,e_z$ denote the Cartesian unit multi-indices.  Expansions use every
$|\alpha|\le p$ in the [linear order defined by `MultiIndexSet`](cartesian-expansions.md).
Taylor monomials are factorial normalised as $r^\alpha/\alpha!$.

## Taylor jets and derivatives

A jet coefficient is $c_\alpha=D_\alpha f(r_0)/\alpha!$, so

$$f(r_0+h)=\sum_{|\alpha|\le p}c_\alpha h^\alpha.$$

`laplace_derivatives_raw` converts the normalised jet back to
$D_\alpha G(r)$ before returning it.  The derivation is described in
[Laplace derivative generation](laplace-derivatives.md).

## P2M

**Purpose.** P2M replaces all dipoles in a leaf by one multipole expansion
about the leaf centre. It is the first stage of the upward pass.

For source expansion centre $c_s$ and $d_j=x_j-c_s$,

$$M_\alpha(c_s)=(-1)^{|\alpha|}
\sum_j\sum_{k\in\{x,y,z\}\atop \alpha_k>0}
m_{j,k}\frac{d_j^{\alpha-e_k}}{(\alpha-e_k)!}.$$

The excluded negative-index terms make $M_0=0$ for pure dipoles. Here $c_s$
is the source-box centre, $x_j$ and $m_j$ are the position and moment of source
$j$, $d_j$ points **from the centre to the source**, $k$ is a Cartesian
component, and $M_\alpha$ is the output multipole coefficient. For fixed
sources, $d_j$ is fixed and this is the sparse map $M=P m$ described below.

## M2M

**Purpose.** M2M shifts a child's multipole expansion to its parent so that
the parent represents every source in its subtree. Levels are processed from
the leaves towards the root.

For $d=c_{\mathrm{parent}}-c_{\mathrm{child}}$,

$$M_\alpha(c_{\mathrm{parent}})\mathrel{+}=
\sum_{\gamma\le\alpha}\frac{d^\gamma}{\gamma!}
M_{\alpha-\gamma}(c_{\mathrm{child}}).$$

The additive form permits accumulation from all eight children. Here $d$
points **from child centre to parent centre**, $\gamma\leq\alpha$ means
component-wise inequality, and the input and output are the child and parent
$M$ coefficients. A uniform tree has eight child-offset classes per level, so
fixed geometry permits exact maps to be shared by children in a class.

## M2L

**Purpose.** M2L converts the multipole expansion of a separated source box
into a local expansion about a target box. Applying every target's `list2`
interactions produces that level's far-field local contribution.

For $R=c_{\mathrm{target}}-c_{\mathrm{source}}$,

$$L_\beta(c_{\mathrm{target}})\mathrel{+}=
\sum_{|\alpha|\le p}M_\alpha(c_{\mathrm{source}})
D_{\alpha+\beta}G(R).$$

Terms reach derivative degree $2p$.  M2L is valid only for separated boxes;
near boxes are handled by P2P. Here $M_\alpha$ is a source multipole
coefficient, $L_\beta$ is a target local coefficient, $\alpha$ and $\beta$
are Cartesian multi-indices, and $D^{\alpha+\beta}G$ is the corresponding
Cartesian derivative of $G(R)=1/(4\pi|R|)$. Crucially, $R$ points **from the
source centre to the target centre**. Fixed centres make these derivatives a
reusable dense matrix $T(R)$.

## L2L

**Purpose.** L2L shifts an accumulated parent local expansion to a child.
Levels are processed from root towards leaves; a child retains its own M2L
contribution while inheriting its parent field.

For $d=c_{\mathrm{child}}-c_{\mathrm{parent}}$,

$$L_\beta(c_{\mathrm{child}})\mathrel{+}=
\sum_{\gamma:\,|\beta+\gamma|\le p}
\frac{d^\gamma}{\gamma!}L_{\beta+\gamma}(c_{\mathrm{parent}}).$$

Here $d$ points **from parent centre to child centre**, the higher-degree local
is the parent input, and $L_\beta$ is the child output. The degree bound
prevents coefficients outside order $p$. Fixed uniform geometry again reduces
the shifts to eight reusable classes per level.

## L2P

**Purpose.** L2P evaluates a leaf local expansion at each target and produces
the far-field potential and/or field.

With $dx=x-c_t$,

$$\phi(x)=\sum_\beta L_\beta\frac{dx^\beta}{\beta!},$$

$$H_k(x)=-\sum_{\beta_k>0}L_\beta
\frac{dx^{\beta-e_k}}{(\beta-e_k)!}.$$

Here $c_t$ is the target leaf centre, $dx$ points **from that centre to the
target**, $L_\beta$ is the input local coefficient, and $k$ selects a field
component. The minus sign implements $H=-\nabla\phi$. Fixed targets yield one
immutable potential row and three immutable field rows.

## M2P

M2P directly evaluates a source multipole at $R=x-c_s$:

$$\phi_{\mathrm{far}}(x)=\sum_\alpha M_\alpha D_\alpha G(R),
\qquad
H_k(x)=-\sum_\alpha M_\alpha D_{\alpha+e_k}G(R).$$

Field evaluation therefore generates kernel derivatives through order $p+1$.

## Static geometry-dependent linear maps

Every far-field stage is linear in the changing moments or in the expansion
coefficients.  For P2M, concatenate the moment components as
$m=(m_{1x},m_{1y},m_{1z},m_{2x},\ldots)^T$.  The P2M equation above then gives

$$M=P m,\qquad
P_{\alpha,(j,k)}=(-1)^{|\alpha|}
\begin{cases}
d_j^{\alpha-e_k}/(\alpha-e_k)!,&\alpha_k>0,\\
0,&\alpha_k=0.
\end{cases}$$

The two tree shifts are likewise exact linear maps:

$$M_{\mathrm{parent}}\mathrel{+}=A(d)M_{\mathrm{child}},\qquad
A_{\alpha,\eta}(d)=
\begin{cases}
d^{\alpha-\eta}/(\alpha-\eta)!,&\eta\le\alpha,\\
0,&\text{otherwise},
\end{cases}$$

$$L_{\mathrm{child}}\mathrel{+}=B(d)L_{\mathrm{parent}},\qquad
B_{\beta,\eta}(d)=
\begin{cases}
d^{\eta-\beta}/(\eta-\beta)!,&\beta\le\eta,\\
0,&\text{otherwise}.
\end{cases}$$

For M2L,

$$L=T(R)M,\qquad T_{\beta,\alpha}(R)=D_{\alpha+\beta}G(R),
\qquad R=c_{\mathrm{target}}-c_{\mathrm{source}}.$$

Finally a fixed target offset has evaluation rows

$$\phi=E_\phi(dx)L,\qquad H=E_H(dx)L,$$

$$E_{\phi,\beta}=dx^\beta/\beta!,\qquad
(E_H)_{k,\beta}=
\begin{cases}
-dx^{\beta-e_k}/(\beta-e_k)!,&\beta_k>0,\\
0,&\beta_k=0.
\end{cases}$$

> When positions, expansion centres, tree structure and expansion order are
> fixed, all of these matrices/operators depend only on geometry. Therefore
> they can be constructed once and reused for every new dipole-moment state.

The implementation stores compact non-zero entry lists for the triangular and
sparse maps even though matrix notation is convenient mathematically.  M2L
retains its grouped dense representation.

```text
changing quantities:
    dipole moments
        ↓ static P2M
    multipoles
        ↓ static M2M
    coarser multipoles
        ↓ static M2L
    local expansions
        ↓ static L2L
    leaf locals
        ↓ static L2P
    far-field H
```

Operator setup occurs once per geometry; operator application occurs once per
moment state.

## P2P

**Purpose.** P2P applies the pair formulas in the first section and sums them
without approximation. It supplies the near-field contribution in an FMM and the
reference answer used by current validation tests. Its input is each source
moment $m_j$ and its output is a target contribution $(\phi_i,H_i)$;
$r_{ij}=x_i-x_j$ points **from source to target**. `list1` selects near boxes.
An explicit identity map excludes $i=j$ rather than treating coincident
coordinates as identity. For fixed geometry the field is the reusable tensor
map $H_{\mathrm{near}}=D_{\mathrm{near}}m$ below; potential retains the direct
scalar calculation.

# Static near-field tensor

For fixed geometry, each list1 pair can be written as

\[
\mathbf H_{ij}=D_{ij}\mathbf m_j,\qquad
D_{ij}=\frac{1}{4\pi}\left(
\frac{3\mathbf r_{ij}\mathbf r_{ij}^{T}}{|\mathbf r_{ij}|^5}
-\frac{I}{|\mathbf r_{ij}|^3}\right).
\]

The tensor is symmetric, so the static plan stores only `Dxx`, `Dxy`, `Dxz`,
`Dyy`, `Dyz`, and `Dzz`. The three output rows still require all appropriate
products with the moment components; symmetry reduces storage, not the apply
to six scalar products. Collecting only list1 blocks gives
`H_near = D_near m`. It is not an all-to-all demagnetisation matrix.
The compact tensor accelerates field output. Potential-only work, and the
potential part of combined output, deliberately continues to use the
independent list1 reference calculation.

# Stage-level static linear forms

Fixed geometry also permits the hierarchical forms `M_leaf = P m`,
`M_parent = A_level M_child`, `L_raw = T M`,
`L_child = B_level L_parent`, and `H_far = E L_leaf`. P2M and L2P are natural
global sparse candidates, while M2M and L2L retain one dependency-ordered
application per level. M2L currently benefits from transfer-class matrix reuse;
a conventional global sparse matrix would duplicate those dense values.
Consequently these alternatives require benchmarks before adoption. Explicitly
composing all stages is avoided because fill-in would approach an all-to-all
operator and discard the FMM hierarchy and scaling.

## M2L normalisation and cross-level reuse

For a same-level M2L interaction, let the physical box width be $h_\ell$ and
write $R=h_\ell t$, where $\ell$ is the **interaction level** and the integer
transfer vector $t$ identifies a uniform-octree translation class. The Laplace
Green function is homogeneous, $G(h r)=h^{-1}G(r)$, and hence

\[
D^\gamma G(h r)=h^{-(|\gamma|+1)}D^\gamma G(r).
\]

Since the Cartesian M2L entry is
(T_{\beta,\alpha}(R)=D^{\alpha+\beta}G(R)), it follows exactly that

\[
T^{(\ell)}_{\beta,\alpha}(t)=
 h_\ell^{-(|\alpha|+|\beta|+1)}
 \widehat T_{\beta,\alpha}(t), \qquad
\widehat T_{\beta,\alpha}(t)=D^{\alpha+\beta}G(t).
\]

Thus $T^{(\ell)}=D_L(h_\ell)\widehat T D_M(h_\ell)$, with
$D_M[\alpha,\alpha]=h_\ell^{-|\alpha|}$ and
$D_L[\beta,\beta]=h_\ell^{-(|\beta|+1)}$. The **normalised transfer
matrix** $\widehat T(t)$ is level independent. **Multipole scaling** $D_M$
converts physical multipoles before multiplication; **local scaling** $D_L$
restores the units and degree of local coefficients. The degree scalings are
precomputed once per level and applied while gathering and scattering.  Raw
Cartesian M2L matrices at two levels are not numerically identical because the
physical box width changes.  The Laplace kernel and all of its derivatives are
homogeneous, allowing the level dependence to be factored into diagonal degree
scalings.  Consequently one normalised M2L matrix per integer transfer vector is
sufficient for all levels. Each interaction stores a **transfer class ID**
selecting $\widehat T(t)$ and its interaction level selecting the scaling
rows. A standard octree list2 displacement lies in the $7\times7\times7$
parent-neighbour stencil but outside the $3\times3\times3$ near stencil. It
therefore has at most $7^3-3^3=316$ transfer classes, irrespective of depth.

For M2M and L2L, a child centre differs from its parent centre by
(d=h\delta), where (h) is the child box width and every component of
(delta) is either (-1/2) or (+1/2).  There are therefore only eight
child-offset classes.  Static plans store shared operators for those classes
(at most eight per used level) and compact child-to-operator identifiers,
rather than duplicating coefficient values for every parent-child edge.
