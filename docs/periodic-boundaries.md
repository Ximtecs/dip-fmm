# Fully periodic boundary conditions

The periodic API requires an explicit fundamental cell.  The first supported
mode is deliberately narrow: all three axes are periodic and the cell is a
cube.  Enabling periodicity makes the cell centre and side length the FMM root
centre and width; the period is never inferred from particle bounds.  Partial
periodicity and rectangular cells are rejected so that later extensions can be
added without giving existing inputs ambiguous meanings.

A three-dimensional dipolar lattice sum is conditionally convergent, so the
word “periodic” is not by itself a physical specification.  The
`PeriodicConvention::ZeroK0` convention omits reciprocal `k=0` and adds no
macroscopic surface or demagnetising term.  Consequently, an exactly uniform
magnetisation has `H_demag = 0`.

## Wrapped topology

Periodic boxes are represented by a central-tree node and a three-component
integer image shift.  An unwrapped coordinate `q` at level `l` is converted by
mathematical floor division into a coordinate modulo `2^l` and an image shift.
The complete pair is the identity: node indices alone must not be used for
deduplication, particularly in shallow trees.

The topology helpers construct list1 as the periodic 3-by-3-by-3 neighbourhood
and list2 as `children(parent list1) - node list1`.  They do not replicate the
tree or particle arrays.  Child displacement classes remain in the ordinary
uniform-tree `[-3,3]^3` list2 class set.

The production static evaluator consumes these image identities directly.
List1 constructs exact shifted point/cuboid P2P tensors. List2 maps wrapped
boxes back onto the ordinary translation-class matrices without duplicating
particle or expansion state. A point self identity removes only the singular
zero-shift pair; every non-zero image of that same particle remains physical.
Finite cuboid self fields remain included.

## Ewald root periodiser

Wrapped traversal covers the central root and its 26 neighbouring images. All
remaining images are represented by one dense root-multipole-to-root-local
matrix. For cell volume $V=L^3$, the setup uses

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
balanced real and reciprocal Ewald sums, omits
reciprocal `k=0`, subtracts the 26 explicitly traversed free-space root
translations, and projects the result into the selected Cartesian or real
spherical basis. `setup_tolerance` controls the exponentially decaying Ewald
cut-offs. The matrix is normalised to unit root width, so the ordinary M2L
degree scaling applies to any physical cell size.

The root contribution is accumulated before L2L propagation. The canonical
static plan is shared by portable CPU, oneMKL, CUDA-partial, and CUDA-full
execution; periodic plans deliberately avoid BSR P2P packing because central
identity exclusion differs from non-zero self-image handling. FP32 plans build
the Ewald matrix in FP64 and quantise it with the other static operators.

Input positions are wrapped into the half-open fundamental cell during plan
construction. Source and target ordering, total-moment input, cuboid P2M/L2P
selection, and exact cuboid-to-cuboid list1 behaviour otherwise match
free-space plans. Dynamic CPU-reference traversal is rejected because it has
no image-aware operator plan.

## Python setup

Periodicity is selected when the reusable plan is constructed:

```python
import cdfmm

options = cdfmm.UniformFmmOptions()
options.backend = cdfmm.ExecutionBackend.CUDA_FULL
options.expansion_basis = cdfmm.ExpansionBasis.SPHERICAL
options.expansion_order = 6
options.tree.max_level = 4
options.periodic.enabled = True
options.periodic.centre = cdfmm.Vec3(0.0, 0.0, 0.0)
options.periodic.lengths = cdfmm.Vec3(1.0, 1.0, 1.0)
options.periodic.convention = cdfmm.PeriodicConvention.ZeroK0
options.periodic.setup_tolerance = 1.0e-12

plan = cdfmm.UniformFmm(source_positions, target_positions, options)
field = plan.evaluate(dipole_moments)["H"]
```

`dipole_moments` contains total source moments, including for cuboid sources.
CUDA-full remains field-only; select the static CPU or hybrid CUDA backend when
scalar potential is also required.
