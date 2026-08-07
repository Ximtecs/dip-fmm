# Operator reference

All expansion operators use a common `MultiIndexSet`, add into destination
coefficients where appropriate, and follow the displacement signs in
[Mathematical formulation](math.md).

## P2M: particle to multipole

**Inputs:** source positions and matching dipole moments, a source expansion
centre $c_s$, and order-$p$ basis. **Output:** a newly initialised multipole
vector $M$.  For each $\alpha$, P2M accumulates dipole components through
$\alpha-e_k$, applies $(-1)^{|\alpha|}$, and leaves the monopole coefficient
zero.  It belongs at occupied source leaves before the upward pass.

## M2M: multipole to multipole

**Inputs:** child multipole, parent destination, and
$d=c_{parent}-c_{child}$. **Output:** contributions added to the parent
multipole.  It contracts factorial-normalised monomials with coefficients at
$\alpha-\gamma$.  Eight child calls form one parent in the upward pass.

## Reference upward composition

`UniformFmm` composes these two operators without reproducing their
mathematics.  It calls P2M once for each non-empty leaf range, then visits
parent levels from deepest to root and calls `m2m_add` for each populated
child.  Destination vectors start at zero on every evaluation.  The M2M sign
is exactly `d = parent centre - child centre`, so a node's final coefficients
describe every source in its subtree about that node centre.

The expansion order is fixed in `UniformFmmOptions` at construction.  The
complete node-vector layout follows flat tree indices and `MultiIndexSet`
coefficient order; `multipole(node_index)` exposes a read-only view for
inspection.  This is only the upward portion of the future traversal.

## M2L: multipole to local

**Inputs:** source multipole, target local destination, and
$R=c_{target}-c_{source}$. **Output:** contributions added to $L$.  The kernel
tensor is $D_{\alpha+\beta}G(R)$ and is generated through order $2p$ for each
call.  A uniform traversal will invoke it for source boxes in a target box's
`list2`.  Calling it at $R=0$ is invalid.

## L2L: local to local

**Inputs:** parent local expansion, child destination, and
$d=c_{child}-c_{parent}$. **Output:** contributions added to the child local
expansion.  Only $|\beta+\gamma|\le p$ terms survive truncation.  It propagates
incoming far fields during the downward pass.

## L2P: local to particle

**Inputs:** local coefficients, their target-box centre, a target position,
and requested output flags. **Output:** one `PotentialField`.  Potential uses
$L_\beta dx^\beta/\beta!$ with $dx=x-c_t$; field differentiates that polynomial
and applies $H=-\nabla\phi$.  It is the far-field evaluation at a target leaf.

## M2P: multipole to particle

**Inputs:** multipole coefficients, their source centre, target position, and
output flags. **Output:** one `PotentialField`.  It contracts $M_\alpha$ with
raw derivatives at $R=x-c_s$; field needs derivative order $p+1$.  M2P is not
a stage of the intended dual-tree flow here, but is a valuable validation path
for P2M and M2M without introducing local expansions.

## P2P: particle to particle

`p2p_dipole_pair` takes target position, source position, moment, and output
flags, and applies the exact dipole formula.  `p2p_dipole_sum` adds that result
over equally sized source and moment arrays.  Its optional `self_index` skips
one singular pair for source-point evaluation.  In a complete FMM, P2P supplies
the near field from `list1`; today it also supplies full reference evaluations.
