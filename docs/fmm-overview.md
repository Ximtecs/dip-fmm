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

All seven mathematical operators exist and `UniformFmm` assembles steps 2--8
as a functional reference traversal. Construction fixes independent source
and target geometry, while each `evaluate` call accepts a new dipole state in
user source order. The upward and downward passes remain separately callable,
and node multipole and local coefficients remain inspectable.

The implemented algorithm is:

```text
Leaf stage:
    M_leaf = P2M(particles in leaf)

Upward stage:
    for level = leaf_level - 1 ... 0:
        for parent at level:
            for populated child in parent.children:
                M_parent += M2M(M_child)

Downward stage:
    clear every local expansion
    for level = 1 ... leaf_level:
        for occupied target box at level:
            L_target += L2L(L_parent)
            for populated source box in target.list2:
                L_target += M2L(M_source)

Evaluation stage:
    for target in each occupied target leaf:
        value = L2P(L_leaf, target)
        for source box in leaf.list1:
            value += P2P(sources in source box, target)
    unpermute values to user target order
```

Before the leaf stage, input moments are permuted to the source positions'
Morton order.  Every node coefficient vector is cleared on each call, so empty
subtrees stay zero and repeated dipole states cannot contaminate each other.
For M2M the displacement is `parent centre - child centre`.  Consequently each
stored node multipole represents the union of all sources below that node,
expanded about its centre.  In particular, the hierarchical root agrees to
round-off with direct P2M of all sources about the root centre.

M2L uses `target centre - source centre`, and L2L uses `child centre -
parent centre`. `list2` supplies the far-field partition and leaf `list1`
supplies the direct near field, so no pair belongs to both paths. Self
interactions are excluded only through an explicit target-to-source index map;
equal coordinates are not treated as particle identity. Public results are
always unpermuted to the original target ordering.

This is deliberately a readable CPU reference. It does not cache translation
operators, optimise fixed geometry, use an adaptive tree, or provide CUDA.

See [Operator reference](operators.md) for each translation and
[Uniform tree](uniform-tree.md) for the geometric lists.
