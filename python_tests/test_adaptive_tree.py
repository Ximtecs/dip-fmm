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


def test_invalid_options_and_geometry():
    for key, value in [("max_depth", -1), ("max_depth", 9), ("max_particles_per_leaf", 0)]:
        options = c.AdaptiveTreeOptions()
        setattr(options, key, value)
        with pytest.raises(ValueError):
            c.AdaptiveTree(np.zeros((1, 3)), options)
    with pytest.raises(ValueError):
        c.AdaptiveTree(np.array([[float("nan"), 0, 0]]))
