"""Focused validation for the adaptive-tree showcase helpers and notebook."""

import json
from pathlib import Path
import sys

import nbformat
import numpy as np
import cdfmm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "examples" / "notebooks"))

from adaptive_showcase import (
    build_trees,
    direct_near,
    generate_material,
    generate_random_grains,
    interaction_categories,
    make_options,
    moment_states,
    benchmark,
)


NOTEBOOK = REPOSITORY_ROOT / "examples" / "notebooks" / "15_adaptive_tree.ipynb"


def _adaptive_options(capacity=10, max_depth=3):
    options = cdfmm.AdaptiveTreeOptions()
    options.max_particles_per_leaf = capacity
    options.max_depth = max_depth
    options.root_centre = cdfmm.Vec3(0.0, 0.0, 0.0)
    options.root_half_width = 0.5
    return options


def test_adaptive_notebook_has_ids_and_compilable_code_cells():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)

    ids = [cell.id for cell in notebook.cells]
    assert len(ids) == len(set(ids)) == len(notebook.cells)
    assert notebook.metadata["kernelspec"]["display_name"] == "cdfmm"

    sources = [
        "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "code"
    ]
    for index, source in enumerate(sources):
        compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec")

    combined = "\n".join(sources)
    assert "cdfmm.AdaptiveTree" in combined or "build_trees" in combined
    assert "ExecutionBackend.CUDA_FULL" in combined
    assert "make_options" in combined
    assert "evaluate_components" in combined
    assert "plan.cuda_plan_statistics" in combined
    assert "plan.cuda_statistics" not in combined


def test_material_generation_is_deterministic_and_conserves_volume_and_moment():
    first = generate_material(300, seed=42, base_grid=4)
    second = generate_material(300, seed=42, base_grid=4)

    assert 300 <= len(first["positions"]) <= 306
    np.testing.assert_array_equal(first["positions"], second["positions"])
    np.testing.assert_array_equal(first["sides"], second["sides"])
    np.testing.assert_array_equal(first["moments"], second["moments"])
    np.testing.assert_allclose(np.sum(first["sides"] ** 3), 1.0, atol=1e-14)
    np.testing.assert_allclose(
        first["moments"].sum(axis=0),
        first["magnetisations"].reshape(-1, 3).T @ (first["sides"] ** 3),
        atol=1e-14,
    )


def test_random_grains_are_deterministic_and_refine_only_boundary_cubes():
    first = generate_random_grains(n_grains=5, base_grid=5, n_refine=3, seed=42)
    second = generate_random_grains(n_grains=5, base_grid=5, n_refine=3, seed=42)

    assert len(first["positions"]) == 15273
    assert first["material_levels"].min() == 0
    assert first["material_levels"].max() == 3
    for key in ("positions", "sides", "material_levels", "grain_labels",
                "grain_centres", "grain_magnetisations", "moments"):
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_allclose(np.sum(first["sides"] ** 3), 1.0, atol=1e-14)

    # Every un-capped leaf has eight corners belonging to one grain.  Leaves
    # at the cap may still straddle a boundary by construction.
    signs = np.asarray([(x, y, z)
                        for z in (-1.0, 1.0)
                        for y in (-1.0, 1.0)
                        for x in (-1.0, 1.0)])
    centres = first["grain_centres"]
    for position, side, level in zip(first["positions"], first["sides"],
                                      first["material_levels"]):
        if level == 3:
            continue
        corners = position + side * signs / 2.0
        labels = np.argmin(np.sum((corners[:, None] - centres[None]) ** 2, axis=2), axis=1)
        assert np.all(labels == labels[0])


def test_adaptive_tree_respects_depth_and_membership_ranges():
    material = generate_material(300, seed=42, base_grid=4)
    tree = cdfmm.AdaptiveTree(material["positions"], _adaptive_options())
    topology = tree.topology

    topology.validate()
    assert topology.root == 0
    assert topology.maximum_level == 3
    assert all(node.index == index for index, node in enumerate(topology.nodes))
    assert all(
        topology.nodes[leaf.node].level == 3 or leaf.count <= 10
        for leaf in topology.source_leaves
    )
    assert any(
        topology.nodes[leaf.node].level == 3 and leaf.count > 10
        for leaf in topology.source_leaves
    )

    ranges = sorted(
        (leaf.begin, leaf.begin + leaf.count) for leaf in topology.source_leaves
    )
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(material["positions"])
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert sorted(topology.source_permutation) == list(range(len(material["positions"])))

    # The seeded geometry deliberately creates both adaptive interaction types.
    groups = interaction_categories(topology, next(leaf.node for leaf in topology.target_leaves
                                                    if topology.nodes[leaf.node].level == 3))
    assert any(item.source_level != item.target_level
               for item in topology.m2l_interactions)
    assert groups["unequal-level P2P"]


def test_empty_adaptive_tree_is_a_valid_depth_zero_topology():
    tree = cdfmm.AdaptiveTree([], _adaptive_options(capacity=1, max_depth=3))
    topology = tree.topology

    topology.validate()
    assert len(topology.nodes) == 1
    assert topology.maximum_level == 0
    assert topology.source_leaves == []
    assert topology.target_leaves == []
    assert topology.p2p_leaf_records == []
    assert topology.m2l_interactions == []


def test_cpu_showcase_benchmark_reuses_topology_and_matches_explicit_near_field():
    material = generate_material(300, seed=42, base_grid=4)
    _, _, topologies, _ = build_trees(
        material["positions"], capacity=10, max_depth=3
    )
    states = moment_states(material, count=10, seed=43)
    options = make_options(len(material["positions"]), cdfmm.ExecutionBackend.CPU_STATIC, order=3)

    plans, setup, measurements, fields = benchmark(topologies, states, options)

    assert set(plans) == {"adaptive", "uniform"}
    assert all(seconds >= 0.0 for seconds in setup.values())
    assert len(measurements) == 20
    assert all(len(values) == 10 for values in fields.values())
    assert all(values[0].shape == (len(material["positions"]), 3)
               for values in fields.values())
    assert all(
        plan.p2p_execution_packing == cdfmm.P2PExecutionPacking.TENSOR_DICTIONARY
        for plan in plans.values()
    )

    for name, plan in plans.items():
        components = plan.evaluate_components(states[0])
        near = direct_near(topologies[name], material["positions"], states[0])
        np.testing.assert_allclose(components["H_p2p"], near, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(
            components["H_far"] + components["H_p2p"],
            components["H_total"],
            rtol=2e-13,
            atol=2e-13,
        )
