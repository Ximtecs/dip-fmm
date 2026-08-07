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

For source expansion centre $c_s$ and $d_j=x_j-c_s$,

$$M_\alpha(c_s)=(-1)^{|\alpha|}
\sum_j\sum_{k\in\{x,y,z\}\atop \alpha_k>0}
m_{j,k}\frac{d_j^{\alpha-e_k}}{(\alpha-e_k)!}.$$

The excluded negative-index terms make $M_0=0$ for pure dipoles.

## M2M

For $d=c_{\mathrm{parent}}-c_{\mathrm{child}}$,

$$M_\alpha(c_{\mathrm{parent}})\mathrel{+}=
\sum_{\gamma\le\alpha}\frac{d^\gamma}{\gamma!}
M_{\alpha-\gamma}(c_{\mathrm{child}}).$$

The additive form permits accumulation from all eight children.

## M2L

For $R=c_{\mathrm{target}}-c_{\mathrm{source}}$,

$$L_\beta(c_{\mathrm{target}})\mathrel{+}=
\sum_{|\alpha|\le p}M_\alpha(c_{\mathrm{source}})
D_{\alpha+\beta}G(R).$$

Terms reach derivative degree $2p$.  M2L is valid only for separated boxes;
near boxes are handled by P2P.

## L2L

For $d=c_{\mathrm{child}}-c_{\mathrm{parent}}$,

$$L_\beta(c_{\mathrm{child}})\mathrel{+}=
\sum_{\gamma:\,|\beta+\gamma|\le p}
\frac{d^\gamma}{\gamma!}L_{\beta+\gamma}(c_{\mathrm{parent}}).$$

## L2P

With $dx=x-c_t$,

$$\phi(x)=\sum_\beta L_\beta\frac{dx^\beta}{\beta!},$$

$$H_k(x)=-\sum_{\beta_k>0}L_\beta
\frac{dx^{\beta-e_k}}{(\beta-e_k)!}.$$

## M2P

M2P directly evaluates a source multipole at $R=x-c_s$:

$$\phi_{\mathrm{far}}(x)=\sum_\alpha M_\alpha D_\alpha G(R),
\qquad
H_k(x)=-\sum_\alpha M_\alpha D_{\alpha+e_k}G(R).$$

Field evaluation therefore generates kernel derivatives through order $p+1$.

## P2P

P2P applies the pair formulas in the first section and sums them without
approximation.  It supplies the near-field contribution in an FMM and the
reference answer used by current validation tests.
