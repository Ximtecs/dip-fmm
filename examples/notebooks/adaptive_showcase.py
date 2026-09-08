"""Geometry, plotting and measurement helpers for notebook 15.

Material subdivision and FMM subdivision are independent. All physics in this
example uses point dipoles with volume-weighted total moments.
"""
from time import perf_counter
import numpy as np
import cdfmm

try:
    from example_utils import draw_box_3d, finish_3d_axes, vec3_to_array
except ModuleNotFoundError:
    from examples.notebooks.example_utils import draw_box_3d, finish_3d_axes, vec3_to_array


def generate_material(n_particles=15000, seed=42, base_grid=4):
    """Split non-overlapping dyadic cubes, conserving volume and moment."""
    if n_particles < base_grid**3:
        raise ValueError("particle target must cover the base grid")
    rng = np.random.default_rng(seed)
    axis = (np.arange(base_grid) + 0.5) / base_grid - 0.5
    centres = list(np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T)
    sides = [1.0 / base_grid] * len(centres)
    directions = rng.normal(size=(len(centres), 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    magnetisations = list(directions)
    levels = [0] * len(centres)
    initial_moment = np.sum(directions, axis=0) / base_grid**3
    offsets = np.array([(x, y, z) for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)])
    while len(centres) < n_particles:
        positions = np.asarray(centres)
        # A broad left-hand region is refined; the right-hand region retains
        # base cubes, giving genuinely mixed-depth FMM leaves at depth three.
        eligible = (positions[:, 0] < 0) & (np.asarray(levels) < 7)
        weights = np.where(eligible, np.asarray(sides)**1.5, 0.0)
        weights *= 0.2 + np.exp(-np.sum((positions - [-0.25, 0.0, 0.0])**2, axis=1) / 0.08)
        if not np.any(weights):
            raise ValueError("material refinement cap exhausted")
        selected = int(rng.choice(len(centres), p=weights / weights.sum()))
        centre, side = centres[selected], sides[selected]
        direction, level = magnetisations[selected], levels[selected]
        for values in (centres, sides, magnetisations, levels):
            values[selected] = values[-1]
            values.pop()
        centres.extend(centre + offsets * side / 4.0)
        sides.extend([side / 2.0] * 8)
        magnetisations.extend([direction.copy() for _ in range(8)])
        levels.extend([level + 1] * 8)
    positions, sides = np.asarray(centres), np.asarray(sides)
    magnetisations = np.asarray(magnetisations)
    moments = magnetisations * sides[:, None]**3
    np.testing.assert_allclose(np.sum(sides**3), 1.0, atol=1e-14)
    np.testing.assert_allclose(moments.sum(axis=0), initial_moment, atol=1e-13)
    return dict(positions=positions, sides=sides, magnetisations=magnetisations,
                moments=moments, material_levels=np.asarray(levels))


def generate_random_grains(n_grains=5, base_grid=5, n_refine=3, seed=42):
    """Generate a small, deterministic Voronoi-style grain geometry.

    The domain is ``[-.5, .5] ** 3`` and starts as ``base_grid**3`` equal
    cubes.  A cube is recursively split into its eight children when its
    corners belong to more than one nearest grain.  ``n_refine`` counts
    subdivisions beyond the base grid, so the returned ``material_levels``
    range from zero to ``n_refine``.  No Voronoi package is needed: nearest
    grain labels are computed directly for the eight cube corners.

    Each final cube contributes one point dipole at its centre.  Grain
    magnetisations are seeded by the same RNG and are inherited by all
    descendants.  The returned moments are volume weighted, which makes
    refinement conserve both volume and total magnetic moment.
    """
    if n_grains <= 0:
        raise ValueError("n_grains must be positive")
    if base_grid <= 0:
        raise ValueError("base_grid must be positive")
    if n_refine < 0:
        raise ValueError("n_refine must be non-negative")

    rng = np.random.default_rng(seed)
    grain_centres = rng.uniform(-0.5, 0.5, size=(n_grains, 3))
    grain_magnetisations = rng.normal(size=(n_grains, 3))
    grain_magnetisations /= np.linalg.norm(grain_magnetisations, axis=1)[:, None]

    # Signs are ordered deterministically so repeated runs produce identical
    # leaf ordering and particle permutations.
    corner_signs = np.asarray([(x, y, z)
                               for z in (-1.0, 1.0)
                               for y in (-1.0, 1.0)
                               for x in (-1.0, 1.0)])
    child_signs = corner_signs
    leaves = []

    def nearest_grain(points):
        distances = np.sum((points[:, None, :] - grain_centres[None, :, :])**2, axis=2)
        return np.argmin(distances, axis=1)

    def visit(centre, half_width, level):
        corners = centre[None, :] + half_width * corner_signs
        labels = nearest_grain(corners)
        boundary = np.any(labels != labels[0])
        if boundary and level < n_refine:
            child_half_width = half_width / 2.0
            # A child centre is displaced by half the parent half-width.
            for sign in child_signs:
                visit(centre + sign * child_half_width,
                      child_half_width, level + 1)
            return

        # At the refinement cap a boundary cube is assigned by its centre;
        # interior cubes use the common corner label.
        label = int(nearest_grain(centre[None, :])[0] if boundary else labels[0])
        leaves.append((centre.copy(), 2.0 * half_width, label, level))

    axis = (np.arange(base_grid, dtype=float) + 0.5) / base_grid - 0.5
    half_width = 0.5 / base_grid
    for z in axis:
        for y in axis:
            for x in axis:
                visit(np.asarray([x, y, z]), half_width, 0)

    positions = np.asarray([leaf[0] for leaf in leaves], dtype=float)
    sides = np.asarray([leaf[1] for leaf in leaves], dtype=float)
    grain_labels = np.asarray([leaf[2] for leaf in leaves], dtype=int)
    levels = np.asarray([leaf[3] for leaf in leaves], dtype=int)
    magnetisations = grain_magnetisations[grain_labels]
    moments = magnetisations * sides[:, None]**3

    np.testing.assert_allclose(np.sum(sides**3), 1.0, atol=1e-14)
    return dict(positions=positions, sides=sides, magnetisations=magnetisations,
                moments=moments, material_levels=levels, grain_labels=grain_labels,
                grain_centres=grain_centres,
                grain_magnetisations=grain_magnetisations)


def build_trees(positions, capacity=10, max_depth=3):
    """Build both geometry plans before allocating any numerical operators."""
    options = cdfmm.AdaptiveTreeOptions()
    options.max_particles_per_leaf = capacity
    options.max_depth = max_depth
    options.root_centre = cdfmm.Vec3(0, 0, 0)
    options.root_half_width = 0.5
    start = perf_counter()
    adaptive = cdfmm.AdaptiveTree(positions, options)
    adaptive_seconds = perf_counter() - start
    uniform_options = cdfmm.UniformTreeOptions()
    uniform_options.max_level = adaptive.topology.maximum_level
    uniform_options.root_centre = options.root_centre
    uniform_options.root_half_width = options.root_half_width
    start = perf_counter()
    uniform = cdfmm.UniformTree(positions, positions, uniform_options)
    uniform_tree_seconds = perf_counter() - start
    start = perf_counter()
    uniform_topology = cdfmm.uniform_topology(uniform)
    uniform_topology_seconds = perf_counter() - start
    return adaptive, uniform, dict(adaptive=adaptive.topology, uniform=uniform_topology), dict(
        adaptive=adaptive_seconds, uniform=uniform_tree_seconds + uniform_topology_seconds,
        adaptive_tree=adaptive.tree_seconds, adaptive_interactions=adaptive.interaction_seconds,
        uniform_tree=uniform_tree_seconds, uniform_topology=uniform_topology_seconds)


def summarise(topology, capacity, max_depth):
    nodes = topology.nodes
    occupancies = np.array([leaf.count for leaf in topology.source_leaves])
    levels = [nodes[leaf.node].level for leaf in topology.source_leaves]
    return dict(nodes=len(nodes), leaves=len(levels), reached_depth=topology.maximum_level,
                leaves_by_level={level: levels.count(level) for level in sorted(set(levels))},
                occupancy_min=int(occupancies.min()) if len(occupancies) else 0,
                occupancy_median=float(np.median(occupancies)) if len(occupancies) else 0,
                occupancy_max=int(occupancies.max(initial=0)),
                depth_limited=sum(nodes[leaf.node].level == max_depth and leaf.count > capacity
                                  for leaf in topology.source_leaves),
                m2l=len(topology.m2l_interactions),
                cross_level_m2l=sum(i.source_level != i.target_level for i in topology.m2l_interactions),
                p2p_pairs=sum(nodes[p.source_leaf].source_count * nodes[p.target_leaf].target_count
                              for p in topology.p2p_leaf_records),
                topology_bytes=topology.memory_bytes)


def interaction_categories(topology, selected, include_ancestors=True):
    nodes = topology.nodes
    ancestors = {selected}
    parent = nodes[selected].parent
    while include_ancestors and parent >= 0:
        ancestors.add(parent)
        parent = nodes[parent].parent
    groups = {name: [] for name in ("same-level P2P", "unequal-level P2P", "same-level M2L", "cross-level M2L")}
    for pair in topology.p2p_leaf_records:
        if pair.target_leaf == selected:
            same = nodes[pair.source_leaf].level == nodes[selected].level
            groups["same-level P2P" if same else "unequal-level P2P"].append(pair.source_leaf)
    for interaction in topology.m2l_interactions:
        if interaction.target_node in ancestors:
            same = interaction.source_level == interaction.target_level
            groups["same-level M2L" if same else "cross-level M2L"].append(interaction.source_node)
    return groups


def select_leaf(topology):
    deepest = [leaf.node for leaf in topology.target_leaves
               if topology.nodes[leaf.node].level == topology.maximum_level]
    if not deepest:
        return topology.root
    scores = {}
    for node in deepest:
        groups = interaction_categories(topology, node)
        scores[node] = (bool(groups["unequal-level P2P"]), len(groups["cross-level M2L"]))
    return max(deepest, key=lambda node: scores[node])


def matching_node(topology, node):
    centre = vec3_to_array(node.centre)
    return next(n.index for n in topology.nodes
                if n.level == node.level and np.allclose(vec3_to_array(n.centre), centre)
                and np.isclose(n.half_width, node.half_width))


def draw_interactions(axis, topology, selected, include_ancestors=True):
    colours = {"same-level P2P": "tab:blue", "unequal-level P2P": "tab:cyan",
               "same-level M2L": "tab:orange", "cross-level M2L": "tab:purple"}
    nodes = topology.nodes
    groups = interaction_categories(topology, selected, include_ancestors)
    for name, ids in groups.items():
        for slot, node_id in enumerate(ids):
            node = nodes[node_id]
            draw_box_3d(axis, vec3_to_array(node.centre), node.half_width,
                        colour=colours[name], alpha=0.45, linewidth=0.8,
                        label=f"{name} ({len(ids)})" if slot == 0 else None)
        if not ids:
            axis.plot([], [], [], color=colours[name], label=f"{name} (0)")
    node = nodes[selected]
    draw_box_3d(axis, vec3_to_array(node.centre), node.half_width,
                colour="tab:red", linewidth=2.5, label="selected leaf")
    finish_3d_axes(axis, f"node {selected}, depth {node.level}")
    axis.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5))
    axis.view_init(elev=24, azim=-55)
    axis.legend(fontsize=7)


def moment_states(material, count=10, seed=43):
    """Identical seeded global magnetisation rotations for both evaluators."""
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(count):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        states.append(np.ascontiguousarray(material["moments"] @ rotation))
    return states


def make_options(count, backend, order=6):
    options = cdfmm.UniformFmmOptions()
    options.backend = backend
    options.precision = cdfmm.StaticPrecision.FLOAT64
    options.expansion_basis = cdfmm.ExpansionBasis.SPHERICAL
    options.expansion_order = order
    options.use_reduced_symmetry_p2p = True
    options.fixed_target_source_indices = list(range(count))
    options.enable_cache = False
    return options


def benchmark(topologies, states, options):
    plans, setup, measurements, fields = {}, {}, [], {}
    for name, topology in topologies.items():
        start = perf_counter()
        plans[name] = cdfmm.build_static_fmm(topology, options)
        setup[name] = perf_counter() - start
        plan = plans[name]
        if plan.p2p_execution_packing != cdfmm.P2PExecutionPacking.TENSOR_DICTIONARY:
            raise RuntimeError("requested reduced Tensor6 packing was not selected")
        plan.evaluate(states[0])  # Untimed warm-up; evaluate returns synchronised results.
        fields[name] = []
        for run, moments in enumerate(states):
            start = perf_counter()
            fields[name].append(np.asarray(plan.evaluate(moments)["H"]))
            elapsed = perf_counter() - start
            measurements.append(dict(method=name, run=run, seconds=elapsed,
                                     phases=dict(plan.last_timings)))
    return plans, setup, measurements, fields


def direct_field(positions, moments, cuda=True):
    if cuda:
        plan = cdfmm.CudaDirectPlan(positions, positions, list(range(len(positions))))
        return np.asarray(plan.evaluate(moments)["H"])
    return np.array([cdfmm.p2p_dipole_sum(x, positions, moments, self_index=i)["H"]
                     for i, x in enumerate(positions)])


def direct_near(topology, positions, moments):
    """Independent direct sum over exactly this topology's P2P rectangles."""
    result = np.zeros_like(positions)
    nodes = topology.nodes
    sp, tp = np.asarray(topology.source_permutation), np.asarray(topology.target_permutation)
    for pair in topology.p2p_leaf_records:
        source, target = nodes[pair.source_leaf], nodes[pair.target_leaf]
        source_ids = sp[source.source_begin:source.source_end]
        for target_id in tp[target.target_begin:target.target_end]:
            ids = source_ids[source_ids != target_id]
            if len(ids) == 0:
                continue
            displacement = positions[target_id] - positions[ids]
            radius2 = np.sum(displacement**2, axis=1)
            projection = np.sum(displacement * moments[ids], axis=1)
            values = (3 * displacement * (projection / radius2)[:, None] - moments[ids])
            result[target_id] += np.sum(values / (4 * np.pi * radius2**1.5)[:, None], axis=0)
    return result


def field_errors(value, reference):
    difference = np.linalg.norm(value - reference, axis=1)
    norm = np.linalg.norm(reference)
    return dict(relative_l2=float(np.linalg.norm(value - reference) / norm) if norm else float(np.linalg.norm(value - reference)),
                absolute_rms=float(np.sqrt(np.mean(difference**2))),
                absolute_max=float(difference.max(initial=0)))
