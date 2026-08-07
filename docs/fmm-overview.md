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
geometry.  Steps 2--8 are **not yet assembled into a traversal**.  Present
tests compose operators manually on controlled boxes; direct P2P is the only
complete many-source evaluation path.  Completing this reference uniform FMM
is the [immediate roadmap milestone](roadmap.md).

See [Operator reference](operators.md) for each translation and
[Uniform tree](uniform-tree.md) for the geometric lists.
