# Uniform tree

`UniformTree` is a geometry container for a **complete**, non-adaptive octree.
All $8^\ell$ boxes at every level $\ell\le L$ exist, even when empty.  This
assumption makes topology and interaction lists easy to audit, at the cost of
memory that grows geometrically with depth.

## Root and coordinates

Unless overridden, the root centre is the midpoint of the bounding box of all
sources and targets.  Its half-width is the largest distance from that centre
to any bounding-box face, producing a cube.  User-supplied centres and positive
half-widths are accepted if every point lies inside the resulting closed cube.

At level $\ell$, each axis has $n=2^\ell$ boxes with integer coordinates
$0\le i_x,i_y,i_z<n$.  The root has level zero and coordinate $(0,0,0)$.
Each box half-width is the root half-width divided by $2^\ell$.

Points exactly on an internal plane are assigned to the box on its positive
side.  Points on the root's upper face are clamped into the final box.  A small
tolerance accommodates floating-point round-off at a requested root boundary;
points genuinely outside raise an exception.

## Morton and flat node ordering

`morton_encode` interleaves coordinate bits in x, y, z order.  For example, a
child's local coordinate bits form its child slot as
`dx + 2*dy + 4*dz`.  `morton_decode` reverses this mapping.

Nodes are stored by complete levels:

```text
flat storage: [ root ][ 8 level-1 boxes ][ 64 level-2 boxes ] ...
               0      1 ... 8              9 ... 72
```

The first index of level $\ell$ is

$$o_\ell=1+8+\cdots+8^{\ell-1}=\frac{8^\ell-1}{7},$$

and a node's flat index is `o_l + morton_encode(ix, iy, iz)`.  Dividing each
coordinate by two gives its parent coordinate.  Doubling the parent coordinate
and adding one of the eight local bit triples gives its children.  Leaves use
`-1` child sentinels.

## Point sorting and permutations

Sources and targets are sorted independently by their leaf Morton keys.  The
sort is stable, so points in one leaf preserve user order.  Four arrays make
the convention explicit:

- `source_permutation[sorted_index] = original_index`;
- `source_inverse_permutation[original_index] = sorted_index`;
- and the analogous two target arrays.

Consequently, later target results in sorted order must use
`target_permutation` to return to user order.  The tree does not yet perform
that result permutation itself.

Each sorted point also records its flat leaf index.  Occupied leaves receive a
half-open range `[begin,end)` into the sorted population; empty leaves receive
an empty range.  Ranges are propagated from children to parents.  Morton order
makes a node's descendants contiguous, so the minimum child begin and maximum
child end describe its complete subtree range.

## Interaction lists

`list1` is the clipped same-level $3\times3\times3$ neighbourhood, including
the node itself.  It represents boxes whose particles are too close for M2L
and must ultimately interact through direct P2P.  Interior boxes have 27
entries; physical boundary boxes have fewer because the tree is not periodic.

For a non-root node, `list2` starts with every child of every box in the
parent's `list1`, then removes the node's own `list1`.  Its entries are
therefore well separated at the current level although their parents touch.
These are the classical uniform FMM M2L partners.  The root has an empty
`list2`; the set-based construction also gives deterministic increasing flat
indices and removes duplicates.

Interaction lists include empty nodes because the complete topology is
materialised.  A future traversal can skip work by consulting source and
target counts.  The current `include_empty_nodes` and `cubic_root_box` options
describe the intended configuration but sparse or non-cubic trees are not yet
implemented.
