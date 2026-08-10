# FMM overview

An FMM replaces most particle pairs by hierarchical expansions.  The intended
uniform-tree evaluation consists of:

1. **Tree construction:** enclose sources and targets in a complete octree,
   sort them by leaf Morton index, and form interaction lists.
2. **P2M:** accumulate dipoles in every occupied source leaf into a multipole
   expansion about that leaf centre.
3. **M2M upward pass:** translate child multipoles to each parent and add them,
   proceeding towards the root.
4. **M2L:** for each target box, translate multipoles from its well-separated
   `list2` boxes into a local expansion.
5. **L2L downward pass:** translate each parent's accumulated local expansion
   to its children.
6. **L2P:** evaluate the leaf local expansion at each target.
7. **P2P near field:** directly accumulate sources in each target leaf's
   touching `list1` boxes, excluding singular self-pairs where applicable.
8. **Unsorting:** map results from Morton order back to the user's target order.

All seven mathematical operators exist and the tree provides the required
geometry.  `UniformFmm` now implements steps 2 and 3 as a reference upward
pass.  Its construction fixes the source geometry, while each `upward_pass`
call accepts a new dipole state in user order.  The remaining steps 4--8 are
**not yet assembled into a traversal**; direct P2P remains the only complete
many-source field-evaluation path.

The implemented algorithm is:

```text
Leaf stage:
    M_leaf = P2M(particles in leaf)

Upward stage:
    for level = leaf_level - 1 ... 0:
        for parent at level:
            for populated child in parent.children:
                M_parent += M2M(M_child)
```

Before the leaf stage, input moments are permuted to the source positions'
Morton order.  Every node coefficient vector is cleared on each call, so empty
subtrees stay zero and repeated dipole states cannot contaminate each other.
For M2M the displacement is `parent centre - child centre`.  Consequently each
stored node multipole represents the union of all sources below that node,
expanded about its centre.  In particular, the hierarchical root agrees to
round-off with direct P2M of all sources about the root centre.

See [Operator reference](operators.md) for each translation and
[Uniform tree](uniform-tree.md) for the geometric lists.
