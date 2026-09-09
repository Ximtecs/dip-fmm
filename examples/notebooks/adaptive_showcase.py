"""Geometry, plotting and measurement helpers for notebook 15.

Material subdivision and FMM subdivision are independent. The same material
cells can be evaluated as point dipoles or finite uniformly magnetised
cuboids. Runtime inputs are always volume-weighted total moments.
"""
from time import perf_counter
import gc
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
    """Find the same box, or the deepest box containing its centre.

    Equal-depth comparisons return the geometrically identical uniform box.
    When the two trees use different maximum depths, the fallback keeps the
    interaction illustration meaningful instead of relying on matching node
    IDs or requiring both trees to expose the same level.
    """
    centre = vec3_to_array(node.centre)
    exact = [candidate for candidate in topology.nodes
             if candidate.level == node.level
             and np.allclose(vec3_to_array(candidate.centre), centre)
             and np.isclose(candidate.half_width, node.half_width)]
    if exact:
        return exact[0].index

    tolerance = 32.0 * np.finfo(float).eps
    containing = [candidate for candidate in topology.nodes
                  if np.all(np.abs(centre - vec3_to_array(candidate.centre))
                            <= candidate.half_width + tolerance)]
    if not containing:
        raise ValueError("the comparison topology does not cover the selected box")
    return max(containing, key=lambda candidate: candidate.level).index


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
    root = nodes[topology.root]
    root_centre = vec3_to_array(root.centre)
    axis.set(
        xlim=(root_centre[0] - root.half_width,
              root_centre[0] + root.half_width),
        ylim=(root_centre[1] - root.half_width,
              root_centre[1] + root.half_width),
        zlim=(root_centre[2] - root.half_width,
              root_centre[2] + root.half_width),
    )
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


INTERACTION_MODES = ("point-point", "cuboid-point", "cuboid-cuboid")
P2P_MODES = ("canonical", "bsr", "dictionary")
DICTIONARY_EXECUTORS = ("source_warp", "target_owned", "power2")


def _enum_name(value):
    """Return a stable display name for a pybind enum or ordinary value."""
    return getattr(value, "name", str(value).rsplit(".", 1)[-1])


def _cuboid_sizes(sides, indices=None):
    values = np.asarray(sides, dtype=float)
    if indices is not None:
        values = values[np.asarray(indices, dtype=int)]
    return [cdfmm.RectangularPrism(float(side), float(side), float(side))
            for side in values]


def interaction_geometry(interaction_mode):
    """Return source and target geometry enums for a notebook mode."""
    if interaction_mode == "point-point":
        return cdfmm.SourceGeometry.POINT_DIPOLE, cdfmm.TargetGeometry.POINT
    if interaction_mode == "cuboid-point":
        return cdfmm.SourceGeometry.RECTANGULAR_PRISM, cdfmm.TargetGeometry.POINT
    if interaction_mode == "cuboid-cuboid":
        return (cdfmm.SourceGeometry.RECTANGULAR_PRISM,
                cdfmm.TargetGeometry.RECTANGULAR_PRISM)
    raise ValueError(f"unknown interaction mode: {interaction_mode!r}")


def make_options(material, backend, order=6, interaction_mode="point-point",
                 reduced=True, precision=None, p2p_mode=None,
                 dictionary_executor="source_warp"):
    """Create one static-plan configuration for the material cells.

    Cuboid sizes remain in original particle order; the plan applies topology
    permutations. Point self interactions are singular and receive explicit
    identities. Cuboid self fields are finite, so cuboid modes do not.
    """
    if np.isscalar(material):
        count = int(material)
        sides = None
    else:
        count = len(material["positions"])
        sides = material["sides"]
    if p2p_mode is None:
        p2p_mode = "dictionary" if reduced else "canonical"
    if p2p_mode not in P2P_MODES:
        raise ValueError(f"unknown P2P mode: {p2p_mode!r}")
    if dictionary_executor not in DICTIONARY_EXECUTORS:
        raise ValueError(f"unknown dictionary executor: {dictionary_executor!r}")
    source_geometry, target_geometry = interaction_geometry(interaction_mode)
    options = cdfmm.UniformFmmOptions()
    options.backend = backend
    options.precision = precision or cdfmm.StaticPrecision.FLOAT32
    options.expansion_basis = cdfmm.ExpansionBasis.SPHERICAL
    options.expansion_order = order
    options.source_geometry = source_geometry
    options.target_geometry = target_geometry
    options.near_field_source_model = cdfmm.SourceModel.EXACT_GEOMETRY
    options.near_field_target_model = cdfmm.TargetModel.EXACT_GEOMETRY
    options.use_reduced_symmetry_p2p = p2p_mode == "dictionary"
    options.cuda_dictionary_target_owned = dictionary_executor == "target_owned"
    options.cuda_dictionary_power2_microtiles = dictionary_executor == "power2"
    if p2p_mode == "canonical":
        options.cuda_p2p_bsr_max_bytes = 0
    if source_geometry == cdfmm.SourceGeometry.RECTANGULAR_PRISM:
        if sides is None:
            raise ValueError("cuboid modes require material cell sides")
        options.source_sizes = _cuboid_sizes(sides)
        options.far_field_source_model = cdfmm.SourceModel.EXACT_GEOMETRY
    else:
        options.far_field_source_model = cdfmm.SourceModel.POINT_DIPOLE
    if target_geometry == cdfmm.TargetGeometry.RECTANGULAR_PRISM:
        options.target_sizes = _cuboid_sizes(sides)
        options.far_field_target_model = cdfmm.TargetModel.EXACT_GEOMETRY
    else:
        options.far_field_target_model = cdfmm.TargetModel.POINT
    if interaction_mode == "point-point":
        options.fixed_target_source_indices = list(range(count))
    options.enable_cache = False
    return options


def _resolved_p2p_executor(plan):
    """Read the resolved executor across dict- and object-style bindings."""
    execution = getattr(plan, "execution_plan", None)
    if callable(execution):
        execution = execution()
    if execution is None:
        return "unavailable"
    if isinstance(execution, dict):
        return _enum_name(execution.get("p2p", "unavailable"))
    return _enum_name(getattr(execution, "p2p", "unavailable"))


def p2p_topology_statistics(topology):
    """Summarise canonical leaf rectangles independently of any backend."""
    nodes = topology.nodes
    source_counts = {leaf.node: leaf.count for leaf in topology.source_leaves}
    target_counts = {leaf.node: leaf.count for leaf in topology.target_leaves}
    same_records = unequal_records = same_particles = unequal_particles = 0
    source_occupancies, target_occupancies = [], []
    target_row_work = {}
    for record in topology.p2p_leaf_records:
        source, target = nodes[record.source_leaf], nodes[record.target_leaf]
        source_count = source_counts[record.source_leaf]
        target_count = target_counts[record.target_leaf]
        interactions = source_count * target_count
        source_occupancies.append(source_count)
        target_occupancies.append(target_count)
        target_row_work[record.target_leaf] = (
            target_row_work.get(record.target_leaf, 0) + interactions)
        if source.level == target.level:
            same_records += 1
            same_particles += interactions
        else:
            unequal_records += 1
            unequal_particles += interactions
    row_work = np.asarray(list(target_row_work.values()), dtype=float)
    row_mean = float(np.mean(row_work)) if len(row_work) else 0.0
    return dict(
        p2p_leaf_records=len(topology.p2p_leaf_records),
        p2p_same_level_leaf_records=same_records,
        p2p_unequal_level_leaf_records=unequal_records,
        p2p_particle_rectangles=same_particles + unequal_particles,
        p2p_same_level_particle_rectangles=same_particles,
        p2p_unequal_level_particle_rectangles=unequal_particles,
        p2p_source_occupancy_median=(float(np.median(source_occupancies))
                                     if source_occupancies else 0.0),
        p2p_source_occupancy_max=max(source_occupancies, default=0),
        p2p_target_occupancy_median=(float(np.median(target_occupancies))
                                     if target_occupancies else 0.0),
        p2p_target_occupancy_max=max(target_occupancies, default=0),
        p2p_target_row_work_min=int(row_work.min()) if len(row_work) else 0,
        p2p_target_row_work_median=(float(np.median(row_work))
                                    if len(row_work) else 0.0),
        p2p_target_row_work_p90=(float(np.quantile(row_work, 0.9))
                                 if len(row_work) else 0.0),
        p2p_target_row_work_max=int(row_work.max()) if len(row_work) else 0,
        p2p_target_row_work_cv=(float(np.std(row_work) / row_mean)
                                if row_mean else 0.0),
    )


def plan_diagnostics(plan, topology, tree, interaction_mode, reduced,
                     p2p_mode=None, dictionary_executor="source_warp"):
    """Collect topology, packing and memory evidence for one static plan."""
    statistics = dict(plan.static_plan_statistics)
    interactions = int(statistics["p2p_interactions"])
    token_bytes = int(statistics["p2p_dictionary_token_bytes"])
    token_count = int(statistics.get(
        "p2p_dictionary_tokens", interactions if token_bytes else 0))
    token_width = int(statistics.get(
        "p2p_dictionary_token_width_bytes",
        token_bytes // token_count if token_count else 0,
    ))
    result = dict(
        tree=tree,
        interaction_mode=interaction_mode,
        reduced=bool(reduced),
        p2p_mode=p2p_mode or ("dictionary" if reduced else "canonical"),
        dictionary_executor=dictionary_executor,
        requested_packing=("TensorDictionary" if reduced else
                           ("BSR" if p2p_mode == "bsr" else "canonical")),
        resolved_packing=_enum_name(plan.p2p_execution_packing),
        resolved_p2p_executor=_resolved_p2p_executor(plan),
        p2p_interactions=interactions,
        p2p_unique_tensors=int(statistics["p2p_unique_tensors"]),
        p2p_dictionary_tokens=token_count,
        p2p_dictionary_token_width_bytes=token_width,
        p2p_value_bytes=int(statistics["p2p_value_bytes"]),
        p2p_index_bytes=int(statistics["p2p_index_bytes"]),
        p2p_canonical_total_bytes=int(statistics["p2p_canonical_total_bytes"]),
        p2p_dictionary_token_bytes=token_bytes,
        p2p_dictionary_tensor_bytes=int(statistics["p2p_dictionary_tensor_bytes"]),
        p2p_dictionary_total_bytes=int(statistics["p2p_dictionary_total_bytes"]),
        retained_bytes=int(statistics["total_persistent_bytes"]),
        backend_packing_seconds=float(statistics["backend_packing_seconds"]),
        cuda_upload_seconds=float(statistics["cuda_upload_seconds"]),
        node_count=len(topology.nodes),
        leaf_count=len(topology.target_leaves),
        topology_bytes=int(topology.memory_bytes),
        m2l_interactions=len(topology.m2l_interactions),
        cross_level_m2l_interactions=sum(
            item.source_level != item.target_level
            for item in topology.m2l_interactions),
    )
    result.update(p2p_topology_statistics(topology))
    try:
        cuda_statistics = dict(plan.cuda_plan_statistics)
    except (AttributeError, RuntimeError):
        cuda_statistics = {}
    result.update(
        device_bytes=int(cuda_statistics.get("persistent_device_bytes", 0)),
        cuda_p2p_tensor_bytes=int(cuda_statistics.get("p2p_tensor_bytes", 0)),
        cuda_p2p_index_bytes=int(cuda_statistics.get("p2p_index_bytes", 0)),
        cuda_p2p_row_metadata_bytes=int(cuda_statistics.get("p2p_row_metadata_bytes", 0)),
        cuda_p2p_leaf_metadata_bytes=int(cuda_statistics.get("p2p_leaf_metadata_bytes", 0)),
        cuda_p2p_identity_bytes=int(cuda_statistics.get("p2p_identity_bytes", 0)),
        cuda_p2p_scratch_bytes=int(cuda_statistics.get("p2p_scratch_bytes", 0)),
    )
    return result


TREE_COLOURS = {"uniform": "tab:blue", "adaptive": "tab:red"}

P2P_CASE_STYLES = {
    ("canonical", "source_warp"): dict(marker="o", linestyle="-"),
    ("bsr", "source_warp"): dict(marker="s", linestyle="--"),
    ("dictionary", "source_warp"): dict(marker="^", linestyle="-."),
    ("dictionary", "target_owned"): dict(marker="D", linestyle=":"),
    ("dictionary", "power2"): dict(marker="P", linestyle=(0, (5, 1))),
}


def p2p_case_label(p2p_mode, dictionary_executor="source_warp"):
    """Return a concise label for a benchmark P2P implementation."""
    if p2p_mode == "canonical":
        return "Canonical"
    if p2p_mode == "bsr":
        return "BSR"
    labels = {
        "source_warp": "Reduced source-warp",
        "target_owned": "Reduced target-owned",
        "power2": "Reduced power2",
    }
    return labels.get(dictionary_executor,
                      f"Reduced {dictionary_executor.replace('_', '-')}")


def benchmark_style(tree, p2p_mode, dictionary_executor="source_warp"):
    """Encode tree type by colour and P2P implementation by line and marker."""
    style = dict(P2P_CASE_STYLES.get(
        (p2p_mode, dictionary_executor), dict(marker="x", linestyle="-")))
    style["color"] = TREE_COLOURS.get(tree, "0.35")
    return style


def plan_build_preflight(topologies, p2p_modes, precision, capacities,
                         maximum_depths):
    """Estimate plan scale before materialising particle-pair operators.

    The host estimates describe the current multi-stage construction path,
    where canonical, compact and backend-specific P2P layouts overlap. They
    are deliberately approximate and should be treated as an early warning,
    not as a memory reservation or a measured peak.
    """
    scalar_bytes = (4 if precision == cdfmm.StaticPrecision.FLOAT32 else 8)
    host_bytes_per_pair = {
        "canonical": 224,
        "bsr": 256,
        "dictionary": 320,
    }
    records = []
    for tree, topology in topologies.items():
        topology_statistics = p2p_topology_statistics(topology)
        nodes = topology.nodes
        occupancies = [leaf.count for leaf in topology.source_leaves]
        for p2p_mode in p2p_modes:
            pair_count = topology_statistics["p2p_particle_rectangles"]
            if p2p_mode == "dictionary":
                # Continuous random positions normally retain one tensor for
                # each reciprocal pair and require four-byte tokens.
                device_p2p_bytes = pair_count * (3 * scalar_bytes + 4)
            else:
                # Nine tensor/potential scalars and one source index.
                device_p2p_bytes = pair_count * (9 * scalar_bytes + 4)
            records.append(dict(
                tree=tree,
                p2p_mode=p2p_mode,
                configured_max_depth=int(maximum_depths[tree]),
                reached_depth=int(topology.maximum_level),
                nodes=len(nodes),
                leaves=len(topology.source_leaves),
                depth_limited_leaves=sum(
                    nodes[leaf.node].level == maximum_depths[tree]
                    and leaf.count > capacities[tree]
                    for leaf in topology.source_leaves),
                maximum_leaf_occupancy=max(occupancies, default=0),
                p2p_particle_pairs=int(pair_count),
                estimated_device_p2p_gib=device_p2p_bytes / 2**30,
                estimated_host_build_peak_gib=(
                    pair_count * host_bytes_per_pair[p2p_mode] / 2**30),
            ))
    return records


def benchmark(topologies, states, material, backend, order=6,
              interaction_modes=INTERACTION_MODES, reduced_values=(False, True),
              precision=None, component_runs=(0, -1), p2p_modes=None,
              dictionary_executors=("source_warp",)):
    """Build and time cases sequentially, releasing each static plan.

    Retaining every full-size plan would multiply the dominant immutable P2P
    storage. Fields and selected diagnostic components are copied before the
    plan is released, so the full Cartesian product remains memory bounded.
    """
    setup, measurements, fields, components, diagnostics = {}, [], {}, {}, []
    selected_component_runs = sorted({run % len(states) for run in component_runs})
    if p2p_modes is None:
        cases = [("dictionary" if reduced else "canonical", "source_warp", reduced)
                 for reduced in reduced_values]
    else:
        cases = []
        for p2p_mode in p2p_modes:
            executors = (dictionary_executors if p2p_mode == "dictionary"
                         else ("source_warp",))
            for executor in executors:
                cases.append((p2p_mode, executor, p2p_mode == "dictionary"))
    for interaction_mode in interaction_modes:
        for p2p_mode, dictionary_executor, reduced in cases:
            options = make_options(
                material, backend, order, interaction_mode, reduced, precision,
                p2p_mode, dictionary_executor)
            for tree, topology in topologies.items():
                key = (tree, interaction_mode, p2p_mode, dictionary_executor)
                start = perf_counter()
                plan = cdfmm.build_static_fmm(topology, options)
                setup[key] = perf_counter() - start
                if (reduced and plan.p2p_execution_packing !=
                        cdfmm.P2PExecutionPacking.TENSOR_DICTIONARY):
                    raise RuntimeError("requested reduced Tensor6 packing was not selected")
                diagnostics.append(plan_diagnostics(
                    plan, topology, tree, interaction_mode, reduced,
                    p2p_mode, dictionary_executor))
                plan.evaluate(states[0])
                fields[key] = []
                for run, moments in enumerate(states):
                    start = perf_counter()
                    fields[key].append(np.asarray(plan.evaluate(moments)["H"]))
                    elapsed = perf_counter() - start
                    measurements.append(dict(
                        tree=tree, interaction_mode=interaction_mode,
                        reduced=bool(reduced), p2p_mode=p2p_mode,
                        dictionary_executor=dictionary_executor,
                        run=run, seconds=elapsed,
                        phases=dict(plan.last_timings),
                    ))
                for run in selected_component_runs:
                    parts = plan.evaluate_components(states[run])
                    components[key, run] = {
                        name: np.asarray(value) for name, value in parts.items()
                    }
                del plan
                gc.collect()
    return setup, measurements, fields, diagnostics, components


def benchmark_summary(setup, measurements, diagnostics):
    """Return one plain record per case, including phase medians."""
    records = []
    for diagnostic in diagnostics:
        key = (diagnostic["tree"], diagnostic["interaction_mode"],
               diagnostic["p2p_mode"], diagnostic["dictionary_executor"])
        rows = [row for row in measurements
                if (row["tree"], row["interaction_mode"], row["p2p_mode"],
                    row["dictionary_executor"]) == key]
        seconds = np.asarray([row["seconds"] for row in rows])
        phase_names = sorted({name for row in rows for name in row["phases"]})
        record = dict(diagnostic)
        record.update(
            plan_setup_seconds=setup[key],
            evaluation_median_seconds=float(np.median(seconds)),
            evaluation_min_seconds=float(seconds.min()),
            evaluation_max_seconds=float(seconds.max()),
        )
        for name in phase_names:
            record[f"phase_{name}_median_seconds"] = float(np.median(
                [row["phases"].get(name, 0.0) for row in rows]))
        kernel_seconds = record.get("phase_cuda_p2p_kernel_median_seconds", 0.0)
        if kernel_seconds <= 0.0:
            kernel_seconds = record.get("phase_p2p_median_seconds", 0.0)
        record["p2p_gpair_per_second"] = (
            record["p2p_interactions"] / kernel_seconds / 1.0e9
            if kernel_seconds > 0.0 else float("nan"))
        records.append(record)
    return records


def comparison_ratios(summary):
    """Calculate adaptive/uniform and P2P mode comparison ratios."""
    def ratio(numerator, denominator, field):
        value = denominator.get(field, 0.0)
        return numerator.get(field, 0.0) / value if value else float("nan")

    by_key = {(row["tree"], row["interaction_mode"], row["p2p_mode"],
               row["dictionary_executor"]): row
              for row in summary}
    rows = []
    for mode in sorted({row["interaction_mode"] for row in summary}):
        cases = sorted({(row["p2p_mode"], row["dictionary_executor"])
                        for row in summary})
        for p2p_mode, dictionary_executor in cases:
            adaptive = by_key.get(("adaptive", mode, p2p_mode,
                                   dictionary_executor))
            uniform = by_key.get(("uniform", mode, p2p_mode,
                                  dictionary_executor))
            if adaptive and uniform:
                storage_field = ("p2p_dictionary_total_bytes"
                                 if p2p_mode == "dictionary"
                                 else "p2p_canonical_total_bytes")
                rows.append(dict(
                    comparison="adaptive / uniform", interaction_mode=mode,
                    p2p_mode=p2p_mode,
                    dictionary_executor=dictionary_executor,
                    evaluation_ratio=(adaptive["evaluation_median_seconds"] /
                                      uniform["evaluation_median_seconds"]),
                    p2p_phase_ratio=ratio(
                        adaptive, uniform, "phase_p2p_median_seconds"),
                    cuda_p2p_kernel_ratio=ratio(
                        adaptive, uniform,
                        "phase_cuda_p2p_kernel_median_seconds"),
                    plan_setup_ratio=(adaptive["plan_setup_seconds"] /
                                      uniform["plan_setup_seconds"]),
                    p2p_interaction_ratio=(adaptive["p2p_interactions"] /
                                           uniform["p2p_interactions"]),
                    p2p_storage_ratio=(adaptive[storage_field] /
                                       max(1, uniform[storage_field])),
                ))
        dictionary_executors = sorted({
            row["dictionary_executor"] for row in summary
            if row["interaction_mode"] == mode and
            row["p2p_mode"] == "dictionary"
        })
        for tree in sorted({row["tree"] for row in summary}):
            for ordinary_mode in ("canonical", "bsr"):
                ordinary = by_key.get(
                    (tree, mode, ordinary_mode, "source_warp"))
                for dictionary_executor in dictionary_executors:
                    reduced = by_key.get(
                        (tree, mode, "dictionary", dictionary_executor))
                    if ordinary and reduced:
                        rows.append(dict(
                            comparison=f"{ordinary_mode} / reduced speedup",
                            interaction_mode=mode, tree=tree,
                            p2p_mode="dictionary",
                            dictionary_executor=dictionary_executor,
                            evaluation_ratio=(
                                ordinary["evaluation_median_seconds"] /
                                reduced["evaluation_median_seconds"]),
                            p2p_phase_ratio=ratio(
                                ordinary, reduced,
                                "phase_p2p_median_seconds"),
                            cuda_p2p_kernel_ratio=ratio(
                                ordinary, reduced,
                                "phase_cuda_p2p_kernel_median_seconds"),
                            plan_setup_ratio=(
                                ordinary["plan_setup_seconds"] /
                                reduced["plan_setup_seconds"]),
                            p2p_interaction_ratio=1.0,
                            p2p_storage_ratio=(
                                reduced["p2p_dictionary_total_bytes"] /
                                max(1, ordinary[
                                    "p2p_canonical_total_bytes"])),
                        ))
    return rows


def direct_field(positions, moments, cuda=True):
    """Compatibility wrapper for a full point-to-point direct field."""
    if cuda:
        plan = cdfmm.CudaDirectPlan(positions, positions, list(range(len(positions))))
        return np.asarray(plan.evaluate(moments)["H"])
    return np.asarray(cdfmm.direct_p2p_reference(
        positions, positions, moments,
        target_source_indices=list(range(len(positions))))["H"])


def reference_target_indices(count, sample_size=256, seed=44):
    """Choose a sorted reproducible target sample without replacement."""
    if sample_size is None or sample_size >= count:
        return np.arange(count, dtype=int)
    if sample_size <= 0:
        raise ValueError("reference sample size must be positive")
    return np.sort(np.random.default_rng(seed).choice(count, sample_size, replace=False))


def build_direct_reference(material, interaction_mode="point-point",
                           target_indices=None, cuda=False):
    """Construct reusable exact geometry for a selected target sample."""
    positions = np.asarray(material["positions"])
    if target_indices is None:
        target_indices = np.arange(len(positions), dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)
    source_geometry, target_geometry = interaction_geometry(interaction_mode)
    targets = positions[target_indices]
    if interaction_mode == "point-point":
        identities = target_indices.tolist()
        if cuda:
            if not cdfmm.cuda_direct_available():
                raise RuntimeError("CUDA point direct evaluation is unavailable")
            plan = cdfmm.CudaDirectPlan(positions, targets, identities)
        else:
            # The portable point reference is evaluated without retaining an
            # N-by-N tensor. Keep its immutable geometry in this small record.
            plan = None
        return dict(plan=plan, positions=positions, targets=targets,
                    target_indices=target_indices, identities=identities,
                    interaction_mode=interaction_mode, cuda=bool(cuda))

    source_sizes = _cuboid_sizes(material["sides"])
    target_sizes = (_cuboid_sizes(material["sides"], target_indices)
                    if interaction_mode == "cuboid-cuboid" else [])
    arguments = (positions, targets, source_geometry, target_geometry,
                 source_sizes, target_sizes, [])
    if cuda:
        if not cdfmm.cuda_dense_direct_available():
            raise RuntimeError("CUDA dense cuboid direct evaluation is unavailable")
        plan = cdfmm.CudaDenseDirectPlan(*arguments, static_precision="float64")
    else:
        plan = cdfmm.DenseDirectPlan(*arguments, static_precision="float64")
    return dict(plan=plan, positions=positions, targets=targets,
                target_indices=target_indices, identities=[],
                interaction_mode=interaction_mode, cuda=bool(cuda))


def evaluate_direct_reference(reference, moments):
    """Apply one moment state to reusable exact-reference geometry."""
    if reference["interaction_mode"] == "point-point":
        if reference["cuda"]:
            return np.asarray(reference["plan"].evaluate(moments)["H"])
        return np.asarray(cdfmm.direct_p2p_reference(
            reference["targets"], reference["positions"], moments,
            target_source_indices=reference["identities"])["H"])
    if reference["cuda"]:
        return np.asarray(reference["plan"].evaluate(moments))
    return np.asarray(reference["plan"].evaluate(
        moments, cdfmm.DenseDirectBackend.PORTABLE))


def direct_reference(material, moments, interaction_mode="point-point",
                     target_indices=None, cuda=False):
    """Convenience one-shot exact reference; sweeps should reuse the plan."""
    reference = build_direct_reference(
        material, interaction_mode, target_indices, cuda)
    return evaluate_direct_reference(reference, moments)


def direct_near(topology, material, moments, interaction_mode="point-point",
                target_indices=None):
    """Evaluate this topology's P2P rows directly at selected targets."""
    positions = np.asarray(material["positions"])
    if target_indices is None:
        target_indices = np.arange(len(positions), dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)
    output_slot = {int(particle): slot for slot, particle in enumerate(target_indices)}
    result = np.zeros((len(target_indices), 3), dtype=float)
    nodes = topology.nodes
    sp, tp = np.asarray(topology.source_permutation), np.asarray(topology.target_permutation)
    records_by_target = {}
    for pair in topology.p2p_leaf_records:
        records_by_target.setdefault(pair.target_leaf, []).append(pair.source_leaf)
    source_geometry, target_geometry = interaction_geometry(interaction_mode)
    for target_leaf, source_leaves in records_by_target.items():
        target = nodes[target_leaf]
        target_ids = [int(item) for item in tp[target.target_begin:target.target_end]
                      if int(item) in output_slot]
        if not target_ids:
            continue
        source_ids = np.concatenate([
            sp[nodes[source_leaf].source_begin:nodes[source_leaf].source_end]
            for source_leaf in source_leaves
        ]).astype(int, copy=False)
        if interaction_mode == "point-point":
            for target_id in target_ids:
                ids = source_ids[source_ids != target_id]
                if not len(ids):
                    continue
                displacement = positions[target_id] - positions[ids]
                radius2 = np.sum(displacement**2, axis=1)
                projection = np.sum(displacement * moments[ids], axis=1)
                values = (3 * displacement * (projection / radius2)[:, None] - moments[ids])
                result[output_slot[target_id]] = np.sum(
                    values / (4 * np.pi * radius2**1.5)[:, None], axis=0)
            continue
        target_array = np.asarray(target_ids, dtype=int)
        source_sizes = _cuboid_sizes(material["sides"], source_ids)
        target_sizes = (_cuboid_sizes(material["sides"], target_array)
                        if interaction_mode == "cuboid-cuboid" else [])
        plan = cdfmm.DenseDirectPlan(
            positions[source_ids], positions[target_array], source_geometry,
            target_geometry, source_sizes, target_sizes, [],
            static_precision="float64")
        values = np.asarray(plan.evaluate(
            np.asarray(moments)[source_ids], cdfmm.DenseDirectBackend.PORTABLE))
        for target_id, value in zip(target_ids, values):
            result[output_slot[target_id]] = value
    return result


def field_errors(value, reference):
    difference = np.linalg.norm(value - reference, axis=1)
    norm = np.linalg.norm(reference)
    return dict(relative_l2=float(np.linalg.norm(value - reference) / norm) if norm else float(np.linalg.norm(value - reference)),
                absolute_rms=float(np.sqrt(np.mean(difference**2))),
                absolute_max=float(difference.max(initial=0)))
