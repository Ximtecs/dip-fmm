# Trees and parameter selection

An FMM plan is organised by an octree. This page describes the complete
uniform tree, the geometry-only adaptive tree, what the expansion order and
depth trade against each other, and the two advisers that measure candidate
settings on a user's own data. Tutorial 5 runs every item interactively.

## The uniform tree

`UniformTree` is a geometry container for a **complete**, non-adaptive
octree: all $8^\ell$ boxes at every level $\ell\le L$ exist, even when empty.
This makes topology and interaction lists easy to audit, at the cost of memory
that grows geometrically with depth.

**Root and coordinates.** Unless overridden, the root centre is the midpoint
of the bounding box of all sources and targets (including the extents of
finite bodies) and its half-width is the largest distance from that centre to
any bounding-box face, producing a cube. User-supplied centres and positive
half-widths are accepted if every point lies inside the resulting closed cube;
a small tolerance accommodates round-off at the boundary, and points
genuinely outside raise. At level $\ell$ each axis has $n=2^\ell$ boxes with
integer coordinates $0\le i_x,i_y,i_z<n$; the root has level zero, and each
box half-width is the root half-width divided by $2^\ell$. Points exactly on
an internal plane are assigned to the box on its positive side; points on the
root's upper face are clamped into the final box.

**Morton and flat ordering.** `morton_encode` interleaves coordinate bits in
x, y, z order, so a child's local coordinate bits form its child slot as
`dx + 2*dy + 4*dz`. Nodes are stored by complete levels, the first index of
level $\ell$ being $o_\ell=(8^\ell-1)/7$, and a node's flat index is
`o_l + morton_encode(ix, iy, iz)`. Leaves use `-1` child sentinels.

**Sorting and permutations.** Sources and targets are sorted independently
and stably by their leaf Morton keys. Four arrays make the convention
explicit: `source_permutation[sorted] = original`,
`source_inverse_permutation[original] = sorted`, and the two target arrays.
`UniformFmm` accepts moments in original order, permutes them internally, and
returns results in original target order; a caller must not pre-sort. Each
occupied leaf owns a half-open range `[begin, end)` of the sorted population,
propagated to parents so that a node's subtree is one contiguous range.

**Interaction lists.** For a free-space plan, `list1` is the clipped
same-level $3\times3\times3$ neighbourhood including the node itself (27 boxes
in the interior): boxes whose particles are too close for M2L and interact
through exact P2P. For a non-root node, `list2` is every child of every box in
the parent's `list1` minus the node's own `list1`: well separated at the
current level although their parents touch, the classical uniform-FMM M2L
partners. The root has an empty `list2`. Periodic plans derive wrapped lists
without changing this topology ([Caching and periodicity](caching-and-periodicity.md)).
Interaction lists include empty nodes; the plan skips operator work for empty
occupancy while keeping zero coefficient vectors, so one moment state never
contaminates the next.

The static topology (`StaticFmmTopology`, exposed in Python as
`plan.topology` or `uniform_topology(tree)`) is the compact form both trees
produce: nodes with centres and half-widths, leaf ranges, M2M/L2L edges, the
M2L interaction rows with their transfer classes and levels, and the P2P
leaf-pair records with image shifts.

## The adaptive tree

`AdaptiveTree` splits a box only while it holds more than
`max_particles_per_leaf` bodies, up to `max_depth`, with an explicit root
centre and half-width. It produces the same static topology type as the
uniform adapter, and `AdaptiveTree.build_fmm(options)` or
`cdfmm.build_static_fmm(topology, options)` constructs the ordinary static
plan on it; every backend and packing applies unchanged.

Two things differ from the uniform tree. Leaves at different levels interact,
so the plan contains cross-level M2L interactions and unequal-size near-field
pairs, both served by the same universal operators and exact tensors. And the
far-field admissibility is the convergence-safe enclosing-sphere criterion
$(r_s+r_t)/d\le0.75$ on non-touching boxes rather than the classical `list2`,
so the adaptive tree keeps more pairs in the exact near field and typically
reaches a lower far-field error at the same order. Leaf capacity is a split
trigger, not a guarantee at the depth cap. Plans on a supplied topology are
not cached, and the geometry records keep their physical user units.

## Order, depth and layout

| Parameter | Increases | Decreases | Typical range |
|---|---|---|---|
| `expansion_order` $p$ | accuracy; coefficient count ($(p+1)^2$ spherical); M2L cost $\propto (p+1)^4$ per translation | truncation error, roughly geometrically | 4–8 |
| `tree.max_level` | boxes ($8^\ell$), M2L translations, small leaves | exact near-field pairs | 2–5 for $10^3$–$10^5$ bodies |
| `spatial_layout = REGULAR_GRID` | nothing in the result | near-field memory and time on lattices of identical bodies (compressed tensor dictionary) | lattices only |

Depth moves work between the exact near field (pairs in `list1`) and the
approximate far field (M2L translations); the optimum balances them and
depends on the occupancy, the backend and the order. At depth 0 or 1
everything is near field and the evaluation is exact but quadratic. On a
regular lattice, choose a root or lattice offset that keeps bodies off the box
boundaries: a source at a box corner is the slowest-converging position for
its expansion.

## The advisers

Both advisers construct ordinary `UniformFmm` plans for every candidate on the
caller's geometry and moments; they change no default and never construct the
production evaluator. Their output is an empirical suggestion for one problem
on one machine, not an error bound or a universally optimal setting. For a
repeated static problem, run an adviser once during setup and copy the
returned order and depth into the production options.

```python
performance = cdfmm.suggest_depth_for_performance(
    sources, targets, moments, order=6,
    backend=cdfmm.ExecutionBackend.CPU_STATIC,
    candidate_depths=[2, 3, 4], repetitions=3,
)
accuracy = cdfmm.suggest_parameters_for_accuracy(
    sources, targets, moments, desired_accuracy=1e-4,
    candidate_orders=range(2, 9), candidate_depths=range(1, 6),
    sample_size=128, repetitions=3,
)
```

`suggest_depth_for_performance` performs a warm-up and reports the median
near-field (complete P2P branch), far-field (P2M–L2P branch) and complete wall
time per candidate depth, ranking sequential backends by wall time. The
partial CUDA backend overlaps its two branches and additionally reports the
heuristic `max(near_seconds, far_seconds)` and their balance ratio; wall time
is always retained. Failed or conservatively memory-limited candidates stay in
the diagnostics with a reason.

`suggest_parameters_for_accuracy` evaluates the exact direct point-dipole
field once at a deterministic, configurable subset of targets, compares every
candidate at the same indices (preserving explicit identities), reports mean,
RMS and maximum relative error plus mean and maximum absolute field error, and
recommends the **fastest** measured candidate whose sampled RMS relative error
meets the tolerance rather than the most accurate one.

## Diagnostics

`plan.static_plan_statistics` reports the tree and topology bytes, the retained
bytes of every operator, the near-field pair count and unique tensors, and the
construction time of every phase (including cache lookups and loads).
`plan.last_timings` and `plan.aggregate_timings` report the phases of the last
and of all evaluations; `plan.cuda_plan_statistics` reports device residency
and per-evaluation transfer bytes. `UniformTree.build_timings` and
`AdaptiveTree.tree_seconds`/`interaction_seconds` time the tree construction
itself. Tutorial 3 shows the tables these produce.
