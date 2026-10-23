"""Adaptive geometry partition, shared execution and cross-level regression tests."""
import gc
import numpy as np
import pytest
import cdfmm as c


def geometry():
    axis = (np.arange(4) + .5) / 4 - .5
    coarse = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    rng = np.random.default_rng(80)
    return np.r_[coarse, rng.uniform([-.49, -.49, -.49], [-.01, .49, .49], (220, 3))]


def build(points, targets=None, depth=3, capacity=5):
    options = c.AdaptiveTreeOptions()
    options.max_depth = depth
    options.max_particles_per_leaf = capacity
    options.root_centre = c.Vec3(0, 0, 0)
    options.root_half_width = .5
    return c.AdaptiveTree(points, options) if targets is None else c.AdaptiveTree(points, targets, options)


def coverage(topology):
    counts = np.zeros((len(topology.target_permutation), len(topology.source_permutation)), dtype=int)
    nodes = topology.nodes
    for interaction in topology.m2l_interactions:
        s, t = nodes[interaction.source_node], nodes[interaction.target_node]
        counts[t.target_begin:t.target_end, s.source_begin:s.source_end] += 1
    for pair in topology.p2p_leaf_records:
        s, t = nodes[pair.source_leaf], nodes[pair.target_leaf]
        counts[t.target_begin:t.target_end, s.source_begin:s.source_end] += 1
    return counts


def test_adaptive_partition_and_membership():
    x = geometry()
    tree = build(x)
    t = tree.topology
    t.validate()
    assert t.maximum_level == 3
    assert len(t.nodes) < 585
    assert any(n.parent >= 0 and n.level != t.nodes[n.index - 1].level for n in t.nodes)
    np.testing.assert_array_equal(coverage(t), np.ones((len(x), len(x))))
    for population, permutation, leaves, positions in [
        (x, t.source_permutation, t.source_leaves, t.sorted_source_positions),
        (x, t.target_permutation, t.target_leaves, t.sorted_target_positions),
    ]:
        np.testing.assert_array_equal(positions, population[permutation])
        seen = np.zeros(len(population), dtype=int)
        for leaf in leaves:
            node = t.nodes[leaf.node]
            centre = np.array([node.centre.x, node.centre.y, node.centre.z])
            selected = positions[leaf.begin:leaf.begin + leaf.count]
            assert np.all(np.abs(selected - centre) <= node.half_width + 1e-14)
            seen[leaf.begin:leaf.begin + leaf.count] += 1
            assert leaf.count <= 5 or tree.leaf_stop_reasons[leaf.node] == "depth_limit"
        np.testing.assert_array_equal(seen, np.ones(len(population)))
    cross = {(i.source_level, i.target_level) for i in t.m2l_interactions if i.source_level != i.target_level}
    assert cross
    assert any(a < b for a, b in cross) and any(a > b for a, b in cross)
    other = build(x).topology
    assert t.source_permutation == other.source_permutation
    assert [(i.source_node, i.target_node) for i in t.m2l_interactions] == [
        (i.source_node, i.target_node) for i in other.m2l_interactions]


@pytest.mark.parametrize("depth", [0, 3, 8])
def test_empty_and_coincident_geometry(depth):
    empty = np.empty((0, 3))
    t = build(empty, depth=depth).topology
    assert len(t.nodes) == 1 and t.maximum_level == 0
    tree = build(np.zeros((15, 3)), depth=depth)
    assert len(tree.topology.nodes) == depth + 1
    assert tree.topology.maximum_level == depth
    assert list(tree.leaf_stop_reasons.values()) == ["depth_limit"]
    # Inferred zero-extent roots must still have a finite positive width.
    c.AdaptiveTree(np.zeros((15, 3))).topology.validate()


def test_independent_targets_partition():
    x = geometry()
    y = np.random.default_rng(12).uniform(-.5, .5, (35, 3))
    t = build(x, y).topology
    np.testing.assert_array_equal(coverage(t), np.ones((len(y), len(x))))
    assert len(t.sorted_source_positions) == len(x)
    assert len(t.sorted_target_positions) == len(y)


@pytest.mark.parametrize("basis", [c.ExpansionBasis.CARTESIAN, c.ExpansionBasis.SPHERICAL])
@pytest.mark.parametrize("precision", [c.StaticPrecision.FLOAT32, c.StaticPrecision.FLOAT64])
def test_cross_level_convergence_and_shared_lifetime(basis, precision):
    x = geometry()
    tree = build(x)
    topology = tree.topology
    rng = np.random.default_rng(6)
    moments = rng.normal(size=x.shape) * 1e-3
    reference = np.array([c.p2p_dipole_sum(p, x, moments, self_index=i)["H"] for i, p in enumerate(x)])
    errors = []
    for order in (2, 4):
        options = c.UniformFmmOptions()
        options.backend = c.ExecutionBackend.CPU_STATIC
        options.expansion_basis = basis
        options.precision = precision
        options.expansion_order = order
        options.fixed_target_source_indices = list(range(len(x)))
        options.use_reduced_symmetry_p2p = True
        plan = c.build_static_fmm(topology, options)
        assert plan.topology is topology
        assert plan.p2p_execution_packing == c.P2PExecutionPacking.TENSOR_DICTIONARY
        result = np.asarray(plan.evaluate(moments)["H"])
        error = np.linalg.norm(result - reference) / np.linalg.norm(reference)
        errors.append(error)
        components = plan.evaluate_components(moments)
        np.testing.assert_allclose(components["H_total"], result, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(components["H_far"] + components["H_p2p"], result, rtol=2e-6, atol=1e-7)
        del tree
        gc.collect()
        np.testing.assert_allclose(plan.evaluate(2 * moments)["H"], 2 * result, rtol=2e-6, atol=1e-7)
        tree = build(x)
    assert errors[1] < errors[0] * .6
    assert errors[1] < .01


def build_target(points, target, depth=8):
    options = c.AdaptiveTreeOptions()
    options.max_depth = depth
    options.max_particles_per_leaf = target
    options.split_rule = c.AdaptiveSplitRule.Target
    options.root_centre = c.Vec3(0, 0, 0)
    options.root_half_width = .5
    return c.AdaptiveTree(points, options)


def test_target_rule_keeps_a_box_whose_octants_would_be_farther_from_the_target():
    # 64 evenly spread points, target 32: the octants would hold 8 each, |ln(8/32)| > |ln(64/32)|
    axis = (np.arange(4) + .5) / 4 - .5
    x = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    tree = build_target(x, 32)
    assert len(tree.topology.nodes) == 1
    assert list(tree.leaf_stop_reasons.values()) == ["target"]
    # with target 4 the same octants (8 each) are closer than the box (64): it subdivides
    assert len(build_target(x, 4).topology.nodes) > 1


def test_target_rule_partition_and_occupancy():
    rng = np.random.default_rng(23)
    x = np.r_[rng.uniform(-.5, .5, (1500, 3)), rng.normal(0, .05, (2000, 3)).clip(-.49, .49)]
    target = 32
    tree = build_target(x, target)
    t = tree.topology
    t.validate()
    np.testing.assert_array_equal(coverage(t), np.ones((len(x), len(x))))
    counts = np.array([leaf.count for leaf in t.source_leaves])
    assert counts.sum() == len(x)
    assert set(tree.leaf_stop_reasons.values()) <= {"capacity", "target", "depth_limit"}
    # leaf for leaf equal to an independent implementation of the rule (octant: coordinate >= box centre)
    def reference(points, centre, half, level):
        n = len(points)
        if level == 8 or n <= target:
            return [n]
        code = ((points >= centre) * [1, 2, 4]).sum(axis=1)
        kids = [points[code == k] for k in range(8)]
        k = np.array([len(q) for q in kids if len(q)])
        if np.sum(k * np.abs(np.log(k / target))) >= n * abs(np.log(n / target)):
            return [n]
        out = []
        for octant, q in enumerate(kids):
            if len(q):
                sign = np.array([1 if octant & b else -1 for b in (1, 2, 4)])
                out += reference(q, centre + sign * half / 2, half / 2, level + 1)
        return out
    assert sorted(counts.tolist()) == sorted(reference(x, np.zeros(3), .5, 0))
    capacity_counts = np.array([leaf.count for leaf in build(x, depth=8, capacity=target).topology.source_leaves])
    assert counts.mean() > capacity_counts.mean()


def test_target_rule_fmm_accuracy():
    x = geometry()
    tree = build_target(x, 12, depth=3)
    rng = np.random.default_rng(7)
    moments = rng.normal(size=x.shape) * 1e-3
    reference = np.array([c.p2p_dipole_sum(p, x, moments, self_index=i)["H"] for i, p in enumerate(x)])
    options = c.UniformFmmOptions()
    options.backend = c.ExecutionBackend.CPU_STATIC
    options.expansion_basis = c.ExpansionBasis.SPHERICAL
    options.precision = c.StaticPrecision.FLOAT64
    options.expansion_order = 4
    options.fixed_target_source_indices = list(range(len(x)))
    result = np.asarray(c.build_static_fmm(tree.topology, options).evaluate(moments)["H"])
    assert np.linalg.norm(result - reference) / np.linalg.norm(reference) < .01


def build_finite(points, radii, depth=4, capacity=4):
    options = c.AdaptiveTreeOptions()
    options.max_depth = depth
    options.max_particles_per_leaf = capacity
    options.root_centre = c.Vec3(0, 0, 0)
    options.root_half_width = .5
    return c.AdaptiveTree(points, points, radii, radii, options)


def test_finite_support_criterion_holds_for_every_far_field_pair():
    rng = np.random.default_rng(31)
    x = rng.uniform(-.49, .49, (600, 3))
    radii = rng.uniform(0, .06, len(x))          # bodies wider than the depth-4 leaves (half-width 1/32)
    tree = build_finite(x, radii)
    t = tree.topology
    t.validate()
    np.testing.assert_array_equal(coverage(t), np.ones((len(x), len(x))))
    assert tree.support_refinements > 0
    scale = t.coordinate_scale
    pos = np.asarray(t.sorted_source_positions)
    r = radii[np.asarray(t.source_permutation)] / scale
    def support(node):
        centre = np.array([node.centre.x, node.centre.y, node.centre.z])
        sel = slice(node.source_begin, node.source_end)
        return np.max(np.linalg.norm(pos[sel] - centre, axis=1) + r[sel])
    for i in t.m2l_interactions:
        s, q = t.nodes[i.source_node], t.nodes[i.target_node]
        d = np.linalg.norm([q.centre.x - s.centre.x, q.centre.y - s.centre.y, q.centre.z - s.centre.z])
        eta = np.sqrt(3) / 2 if s.level == q.level else .75
        assert support(s) + support(q) <= eta * d * (1 + 1e-9)


def test_zero_support_radii_keep_the_point_topology():
    x = geometry()
    a = build(x).topology
    options = c.AdaptiveTreeOptions()
    options.max_depth, options.max_particles_per_leaf = 3, 5
    options.root_centre, options.root_half_width = c.Vec3(0, 0, 0), .5
    tree = c.AdaptiveTree(x, x, np.zeros(len(x)), np.zeros(len(x)), options)
    b = tree.topology
    assert tree.support_refinements == 0
    assert a.source_permutation == b.source_permutation
    assert [(i.source_node, i.target_node) for i in a.m2l_interactions] == [
        (i.source_node, i.target_node) for i in b.m2l_interactions]
    with pytest.raises(ValueError):
        c.AdaptiveTree(x, x, np.zeros(3), np.zeros(len(x)), options)


def test_invalid_options_and_geometry():
    for key, value in [("max_depth", -1), ("max_depth", 9), ("max_particles_per_leaf", 0)]:
        options = c.AdaptiveTreeOptions()
        setattr(options, key, value)
        with pytest.raises(ValueError):
            c.AdaptiveTree(np.zeros((1, 3)), options)
    with pytest.raises(ValueError):
        c.AdaptiveTree(np.array([[float("nan"), 0, 0]]))
