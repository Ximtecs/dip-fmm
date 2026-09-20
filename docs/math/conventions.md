# Mathematical conventions

This page is the normative statement of the kernel, the signs, the
displacement directions, the coefficient normalisation and the definition of
every operator used by the implementation. Real spherical harmonics are the
default `UniformFmm` basis; the complete Cartesian formulation is an explicit
independent option. The Cartesian definitions are given first because the
static spherical operators are constructed from them
([Real spherical-harmonic expansions](spherical-expansions.md)); finite
sources and targets are in [Finite geometry](finite-geometry.md).

## Kernel, dipole potential, and field

The Laplace Green function is

$$G(r)=\frac{1}{4\pi |r|}.$$

For source position $x_j$, dipole moment $m_j$, and target $x$, let
$r_j=x-x_j$. The scalar potential and magnetic field are

$$\phi(x)=\sum_j\frac{m_j\cdot r_j}{4\pi |r_j|^3},
\qquad H(x)=-\nabla\phi(x).$$

Thus one direct pair contributes

$$H_{ij}=\frac{1}{4\pi}\left[
\frac{3r_{ij}(m_j\cdot r_{ij})}{|r_{ij}|^5}
-\frac{m_j}{|r_{ij}|^3}\right],
\qquad r_{ij}=x_i-x_j.$$

The kernel is singular at zero separation. A source-point evaluation must
explicitly skip its self-interaction; the library does this only through an
explicit identity map, never through coordinate equality.

## Displacement conventions

| Operator | Displacement | Direction |
|---|---|---|
| P2P | $r_{ij}=x_i-x_j$ | from source to target |
| P2M | $d_j=x_j-c_s$ | from the source-box centre to the source |
| M2M | $d=c_{\mathrm{parent}}-c_{\mathrm{child}}$ | from child centre to parent centre |
| M2L | $R=c_{\mathrm{target}}-c_{\mathrm{source}}$ | from source centre to target centre |
| L2L | $d=c_{\mathrm{child}}-c_{\mathrm{parent}}$ | from parent centre to child centre |
| L2P | $dx=x-c_t$ | from the target-box centre to the target |
| M2P | $R=x-c_s$ | from the source centre to the target |

## Cartesian multi-indices and coefficient storage

For $\alpha=(\alpha_x,\alpha_y,\alpha_z)\in\mathbb N_0^3$,

$$|\alpha|=\alpha_x+\alpha_y+\alpha_z,\quad
\alpha!=\alpha_x!\alpha_y!\alpha_z!,\quad
r^\alpha=r_x^{\alpha_x}r_y^{\alpha_y}r_z^{\alpha_z},$$

and $e_x,e_y,e_z$ denote the Cartesian unit multi-indices. `MultiIndexSet(p)`
contains every non-negative $\alpha$ with $|\alpha|\le p$; its size is

$$N_p=\binom{p+3}{3}=\frac{(p+1)(p+2)(p+3)}{6}.$$

Entries are grouped by total degree. At fixed degree the implementation loops
over $\alpha_x$ and then $\alpha_y$ in ascending order; $\alpha_z$ is the
remaining degree. `index(alpha)` maps a mathematical index to linear storage
and `operator[](i)` performs the reverse lookup. `CoeffVector` uses precisely
this basis order for both multipole and local coefficients.

Taylor monomials are factorial normalised as $r^\alpha/\alpha!$. This makes
Taylor translations and jet products free of explicit multinomial factors.
Multipole coefficients are paired with **raw** derivatives $D_\alpha G$,
whereas local coefficients are raw potential derivatives paired with these
normalised monomials.

For dipoles, $M_{(0,0,0)}=0$: P2M requires a term $\alpha-e_k$ for a dipole
component $m_k$, and no coordinate direction is valid when $\alpha=0$. This
expresses the absence of net monopole charge; it does not mean local
expansions omit their degree-zero potential coefficient.

## Taylor jets and Laplace derivatives

A jet coefficient is $c_\alpha=D_\alpha f(r_0)/\alpha!$, so

$$f(r_0+h)=\sum_{|\alpha|\le p}c_\alpha h^\alpha.$$

M2L and M2P require raw derivatives $D_\alpha G(R)$ at orders beyond those
stored in the source expansion. `laplace_derivatives_raw` obtains them through
truncated Cartesian Taylor algebra rather than finite differences. Coordinate
jets represent $x=r_x+h_x$, $y=r_y+h_y$, and $z=r_z+h_z$; the implementation
composes

$$\rho^2=x^2+y^2+z^2,\qquad G=\frac{1}{4\pi}(\rho^2)^{-1/2}.$$

Products use the multi-index Cauchy product. The inverse square root is solved
coefficient by coefficient from $y^2\rho^2=1$ in increasing total degree, so
each unknown depends only on coefficients already found. Finally the jet
coefficient is multiplied by $\alpha!$ to return the raw derivative expected
by the operators. This is analytic/algebraic to the requested truncation
order: it avoids finite-difference step-size selection and cancellation
noise, and a hand-maintained table of high-order Cartesian derivatives. The
evaluation point must be away from the kernel singularity at $R=0$.

## The operators

Every operator is an exact linear map on coefficients; only the truncation at
order $p$ introduces error.

### P2M: particle to multipole

P2M replaces all dipoles in a leaf by one multipole expansion about the leaf
centre; it is the first stage of the upward pass. For source expansion centre
$c_s$ and $d_j=x_j-c_s$,

$$M_\alpha(c_s)=(-1)^{|\alpha|}
\sum_j\sum_{k\in\{x,y,z\}\atop \alpha_k>0}
m_{j,k}\frac{d_j^{\alpha-e_k}}{(\alpha-e_k)!}.$$

The excluded negative-index terms make $M_0=0$ for pure dipoles. For fixed
sources $d_j$ is fixed and this is the sparse map $M=Pm$ below.

### M2M: multipole to multipole

M2M shifts a child's multipole expansion to its parent so that the parent
represents every source in its subtree; levels are processed from the leaves
towards the root. For $d=c_{\mathrm{parent}}-c_{\mathrm{child}}$,

$$M_\alpha(c_{\mathrm{parent}})\mathrel{+}=
\sum_{\gamma\le\alpha}\frac{d^\gamma}{\gamma!}
M_{\alpha-\gamma}(c_{\mathrm{child}}),$$

where $\gamma\le\alpha$ is component-wise. The additive form permits
accumulation from all eight children. A uniform tree has eight child-offset
classes: one canonical level-one bank plus exact power-of-two degree scaling
serves every depth.

### M2L: multipole to local

M2L converts the multipole expansion of a separated source box into a local
expansion about a target box. For $R=c_{\mathrm{target}}-c_{\mathrm{source}}$,

$$L_\beta(c_{\mathrm{target}})\mathrel{+}=
\sum_{|\alpha|\le p}M_\alpha(c_{\mathrm{source}})
D_{\alpha+\beta}G(R).$$

Terms reach derivative degree $2p$. M2L is valid only for separated boxes
(calling it at $R=0$ is invalid); near boxes are handled by P2P. Fixed centres
make these derivatives a reusable dense matrix $T(R)$.

### L2L: local to local

L2L shifts an accumulated parent local expansion to a child; levels are
processed from the root towards the leaves, and a child retains its own M2L
contribution while inheriting its parent field. For
$d=c_{\mathrm{child}}-c_{\mathrm{parent}}$,

$$L_\beta(c_{\mathrm{child}})\mathrel{+}=
\sum_{\gamma:\,|\beta+\gamma|\le p}
\frac{d^\gamma}{\gamma!}L_{\beta+\gamma}(c_{\mathrm{parent}}).$$

The degree bound prevents coefficients outside order $p$; only
$|\beta+\gamma|\le p$ terms survive truncation.

### L2P: local to particle

L2P evaluates a leaf local expansion at each target and produces the
far-field potential and/or field. With $dx=x-c_t$,

$$\phi(x)=\sum_\beta L_\beta\frac{dx^\beta}{\beta!},\qquad
H_k(x)=-\sum_{\beta_k>0}L_\beta
\frac{dx^{\beta-e_k}}{(\beta-e_k)!}.$$

The minus sign implements $H=-\nabla\phi$. Fixed targets yield one immutable
potential row and three immutable field rows.

### M2P: multipole to particle

M2P directly evaluates a source multipole at $R=x-c_s$:

$$\phi_{\mathrm{far}}(x)=\sum_\alpha M_\alpha D_\alpha G(R),
\qquad
H_k(x)=-\sum_\alpha M_\alpha D_{\alpha+e_k}G(R).$$

Field evaluation therefore generates kernel derivatives through order $p+1$.
M2P is not a stage of the FMM (targets receive the far field through local
expansions), but it validates P2M and M2M without introducing local
expansions.

### P2P: particle to particle

P2P applies the pair formula of the first section and sums it without
approximation. It supplies the near-field contribution over `list1` and the
reference answer used by the validation tests. For fixed geometry the field is
the reusable tensor map $H_{\mathrm{near}}=D_{\mathrm{near}}m$ below;
potential retains the direct scalar calculation.

## Static geometry-dependent linear maps

Every far-field stage is linear in the changing moments or in the expansion
coefficients. For P2M, concatenate the moment components as
$m=(m_{1x},m_{1y},m_{1z},m_{2x},\ldots)^T$; the P2M equation then gives

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

When positions, expansion centres, tree structure and expansion order are
fixed, all of these operators depend only on geometry, so they are constructed
once and reused for every new dipole-moment state:

```text
changing quantities:
    dipole moments
        | static P2M
    multipoles
        | static M2M
    coarser multipoles
        | static M2L
    local expansions
        | static L2L
    leaf locals
        | static L2P
    far-field H
```

The implementation stores compact non-zero entry lists for the triangular and
sparse maps even though matrix notation is convenient mathematically; M2L
retains its grouped dense representation. Explicitly composing all stages is
avoided because the fill-in would approach an all-to-all operator and discard
the FMM hierarchy and scaling.

### Static near-field tensor

For fixed geometry each `list1` pair can be written as

$$\mathbf H_{ij}=D_{ij}\mathbf m_j,\qquad
D_{ij}=\frac{1}{4\pi}\left(
\frac{3\mathbf r_{ij}\mathbf r_{ij}^{T}}{|\mathbf r_{ij}|^5}
-\frac{I}{|\mathbf r_{ij}|^3}\right).$$

The tensor is symmetric, so the static plan stores only `Dxx`, `Dxy`, `Dxz`,
`Dyy`, `Dyz`, and `Dzz`; symmetry reduces storage, not the apply to six scalar
products. Collecting only `list1` blocks gives `H_near = D_near m`, which is
not an all-to-all demagnetisation matrix. The compact tensor accelerates field
output; potential-only work, and the potential part of combined output,
deliberately continues to use the independent `list1` reference calculation.
Finite sources replace $D_{ij}$ by the exact body tensors of
[Finite geometry](finite-geometry.md).

## Normalisation terminology

Four distinct ideas appear in this documentation:

1. Cartesian Taylor monomials use the factorial factor $r^\alpha/\alpha!$.
2. The real spherical basis uses the orthonormal-harmonic and
   $\sqrt{4\pi/(2l+1)}$ solid-harmonic factors of
   [Real spherical-harmonic expansions](spherical-expansions.md).
3. Cartesian and spherical M2L plans factor degree-dependent powers of box
   width out of each physical-level matrix so displacement classes are reused
   (below).
4. FP32 FMM plans additionally scale physical coordinates by the root-box
   width internally for numerical range (below).

### Global root normalisation

Public coordinates remain physical. Internally the physical root centre $c$
and side length $L$ define

$$r'=(r-c)/L,\qquad h'=h/L,\qquad m'=m/L^3.$$

Every internal tree therefore has centre zero and half-width $1/2$. For a
free-space cuboid problem the bounds used to determine $c$ and $L$ include
each full interval $r_i-h_i/2$ through $r_i+h_i/2$; for a periodic problem the
explicit cubic cell supplies both values. The magnetic dipole tensor obeys
$T(Lr')=T(r')/L^3$, so evaluating $m'$ in canonical coordinates returns the
physical field directly: $H_{\mathrm{physical}}=H_{\mathrm{internal}}$, with
no field post-scaling. Scalar potential has one fewer inverse power of length
and is returned as $\phi_{\mathrm{physical}}=L\phi_{\mathrm{internal}}$.

Normalised coordinates are further canonicalised to a $10^{-9}$ grid of the
root side before operators are built, which is what lets translated or
uniformly scaled copies of one geometry share a cache key and what makes
equivalent exact operators agree bit for bit
([Architecture](../architecture.md)).

### Root-width scaling in FP32 plans

FP32 `UniformFmm` operators use coordinates divided by the physical root-box
width and moments divided by the cube of that width during boundary
conversion. This paired scaling keeps high-order coefficients and inverse-box
M2L factors representable for physical scales such as nanometres without
introducing FP64 expansion state. The field is invariant under the scaling;
potential is restored with one root-width factor at the output boundary. This
is mandatory internal behaviour of FP32 FMM plans, not a public optional
coordinate transform; FP64 plans retain physical-coordinate operator
construction, and both paths accept and return physical coordinates, moments,
potential and field.

### M2L normalisation and cross-level reuse

For a same-level M2L interaction, let the physical box width be $h_\ell$ and
write $R=h_\ell t$, where $\ell$ is the **interaction level** and the integer
transfer vector $t$ identifies a uniform-octree translation class. The Laplace
Green function is homogeneous, $G(h r)=h^{-1}G(r)$, and hence

$$D^\gamma G(h r)=h^{-(|\gamma|+1)}D^\gamma G(r).$$

Since the Cartesian M2L entry is $T_{\beta,\alpha}(R)=D^{\alpha+\beta}G(R)$,
it follows exactly that

$$T^{(\ell)}_{\beta,\alpha}(t)=
 h_\ell^{-(|\alpha|+|\beta|+1)}
 \widehat T_{\beta,\alpha}(t), \qquad
\widehat T_{\beta,\alpha}(t)=D^{\alpha+\beta}G(t).$$

Thus $T^{(\ell)}=D_L(h_\ell)\widehat T D_M(h_\ell)$, with
$D_M[\alpha,\alpha]=h_\ell^{-|\alpha|}$ and
$D_L[\beta,\beta]=h_\ell^{-(|\beta|+1)}$. The **normalised transfer matrix**
$\widehat T(t)$ is level independent; **multipole scaling** $D_M$ converts
physical multipoles before multiplication and **local scaling** $D_L$ restores
the units and degree of local coefficients. The degree scalings are
precomputed once per level and applied while gathering and scattering.
Consequently one normalised M2L matrix per integer transfer vector is
sufficient for all levels; each interaction stores a **transfer class ID**
selecting $\widehat T(t)$ and its interaction level selecting the scaling
rows. A standard octree `list2` displacement lies in the $7\times7\times7$
parent-neighbour stencil but outside the $3\times3\times3$ near stencil, so
there are at most $7^3-3^3=316$ transfer classes irrespective of depth.

For M2M and L2L, a child centre differs from its parent centre by
$d=h\delta$, where $h$ is the child box width and every component of $\delta$
is $-1/2$ or $+1/2$. There are therefore only eight child-offset classes;
static plans store eight level-one templates and apply exact powers of two
derived from coefficient degree at later levels. The stored M2M, M2L, and L2L
banks are independent of tree depth, which is why they are cached
independently of any geometry
([Caching and periodicity](../caching-and-periodicity.md)).

The periodic root operator and the zero-$k$ convention are defined with the
periodic API in [Caching and periodicity](../caching-and-periodicity.md).
