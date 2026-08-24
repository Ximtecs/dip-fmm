# Real spherical-harmonic expansions

`UniformFmm` uses a minimal real spherical-harmonic basis by default. The
Cartesian Taylor basis remains available by selecting `ExpansionBasis::Cartesian`
in C++ or assigning `"cartesian"` to `options.expansion_basis` in Python. The
tree, list1 near field, list2 interaction classes, point-dipole convention, and
exact P2P tensor are shared.

## Convention and ordering

Let `Y_l^m` be the orthonormal complex spherical harmonic whose associated
Legendre function includes the Condon--Shortley phase. For `m>0`, the real
tesseral basis is

\[
Y^R_{l0}=Y_l^0,\quad
Y^R_{lm}=\sqrt{2}(-1)^m\operatorname{Re}Y_l^m,\quad
Y^R_{l,-m}=\sqrt{2}(-1)^m\operatorname{Im}Y_l^m.
\]

Regular and irregular solid harmonics are

\[
R_{lm}(r)=\sqrt{\frac{4\pi}{2l+1}}r^lY^R_{lm}(\hat r),\qquad
I_{lm}(r)=\sqrt{\frac{4\pi}{2l+1}}
\frac{Y^R_{lm}(\hat r)}{r^{l+1}}.
\]

Thus `R_00=1`; the `l=1` modes ordered by `m=-1,0,1` are `y,z,x`.
Modes are stored by increasing `l`, then `m=-l,...,+l`, so `(l,m)` has index
`l*l+m+l`. An order-`p` expansion has `(p+1)^2` real coefficients.

The addition theorem is

\[
\frac{1}{|x-d|}=\sum_{l=0}^{\infty}\sum_{m=-l}^{l}
R_{lm}(d)I_{lm}(x),\qquad |d|<|x|.
\]

For `G=1/(4*pi*r)`, point-dipole P2M is

\[
M_{lm}=\frac{1}{4\pi}\sum_j m_j\mathbin{\cdot}\nabla R_{lm}(x_j-c_s).
\]

Multipole and local evaluation are `phi=sum M_lm I_lm` and
`phi=sum L_lm R_lm`; both use `H=-grad(phi)`.

## Static translations

Regular harmonics are generated as homogeneous Cartesian polynomials. Setup
uses the exact identity

\[
I_{lm}(r)=\frac{4\pi(-1)^l}{(2l-1)!!}R_{lm}(\nabla)G(r)
\]

to project the validated Cartesian M2M and L2L translations and Laplace
derivative tensors onto the minimal harmonic subspace. This is algebraically
equivalent to spherical rotate--axial-shift--rotate-back translation, but
produces the final real operators directly. No Cartesian expansion state or
rotation is retained at runtime.

M2M and L2L store eight child classes per used level. M2L stores one dense
matrix per used integer displacement class and uses degree-based box-width
scaling. No geometry-dependent harmonic work occurs in `evaluate()`.

FMM3D v2.1.0 provides direct spherical routines `l3dmpmp`, `l3dmploc`, and
`l3dlocloc`. Its exponential/plane-wave representation is a separate M2L
optimisation. This implementation deliberately starts with reusable dense M2L
matrices so their performance under fixed geometry can be measured first.

## Supported combinations

Spherical point-dipole sources and point targets support FP32 and FP64 CPU
static, oneMKL, CUDA-partial, and CUDA-full plans. FP32 uses root-width
normalised coordinates and retains no FP64 operator copies. Uniform cuboid
sources and the Cartesian dynamic-reference traversal are rejected.
