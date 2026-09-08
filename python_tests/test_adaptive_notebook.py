"""Focused validation for the adaptive-tree showcase helpers and notebook."""

from pathlib import Path
import sys

import nbformat
import numpy as np
import cdfmm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "examples" / "notebooks"))

from adaptive_showcase import (
    INTERACTION_MODES,
    benchmark,
    benchmark_summary,
    build_trees,
    direct_near,
    direct_reference,
    generate_material,
    generate_random_grains,
    interaction_categories,
    make_options,
    moment_states,
    reference_target_indices,
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
    assert "INTERACTION_MODES" in combined
    assert "REDUCED_VALUES" in combined
    assert "reference_target_indices" in combined
    assert "direct_reference" in combined
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


def test_geometry_options_cover_all_modes_without_excluding_cuboid_self():
    material = generate_random_grains(n_grains=2, base_grid=2, n_refine=0, seed=7)
    for mode in INTERACTION_MODES:
        for reduced in (False, True):
            options = make_options(
                material, cdfmm.ExecutionBackend.CPU_STATIC, order=2,
                interaction_mode=mode, reduced=reduced,
                precision=cdfmm.StaticPrecision.FLOAT64,
            )
            assert options.use_reduced_symmetry_p2p is reduced
            assert bool(options.fixed_target_source_indices) is (mode == "point-point")
            assert (len(options.source_sizes) == len(material["positions"])) is (
                mode != "point-point")
            assert (len(options.target_sizes) == len(material["positions"])) is (
                mode == "cuboid-cuboid")
            assert options.use_cuboid_p2m is (mode != "point-point")
            assert options.use_cuboid_l2p is (mode == "cuboid-cuboid")


def test_cpu_showcase_matrix_preserves_topology_and_matches_direct_fields():
    material = generate_random_grains(n_grains=2, base_grid=2, n_refine=0, seed=7)
    _, _, topologies, _ = build_trees(
        material["positions"], capacity=1, max_depth=1
    )
    fingerprints = {
        name: (
            tuple(topology.source_permutation),
            tuple(topology.target_permutation),
            len(topology.nodes),
            len(topology.m2l_interactions),
            len(topology.p2p_leaf_records),
        )
        for name, topology in topologies.items()
    }
    states = moment_states(material, count=2, seed=43)
    setup, measurements, fields, diagnostics, components = benchmark(
        topologies, states, material, cdfmm.ExecutionBackend.CPU_STATIC,
        order=3, precision=cdfmm.StaticPrecision.FLOAT64,
    )

    case_count = 2 * len(INTERACTION_MODES) * 2
    assert len(setup) == case_count
    assert all(seconds >= 0.0 for seconds in setup.values())
    assert len(measurements) == case_count * len(states)
    assert all(len(values) == len(states) for values in fields.values())
    assert all(values[0].shape == (len(material["positions"]), 3)
               for values in fields.values())
    assert all(row["resolved_packing"] == "TENSOR_DICTIONARY"
               for row in diagnostics if row["reduced"])
    assert {row["interaction_mode"] for row in diagnostics} == set(INTERACTION_MODES)
    assert len(benchmark_summary(setup, measurements, diagnostics)) == case_count
    for name, topology in topologies.items():
        assert fingerprints[name] == (
            tuple(topology.source_permutation),
            tuple(topology.target_permutation),
            len(topology.nodes),
            len(topology.m2l_interactions),
            len(topology.p2p_leaf_records),
        )

    targets = reference_target_indices(len(material["positions"]), sample_size=4, seed=9)
    for mode in INTERACTION_MODES:
        reference = direct_reference(material, states[0], mode, targets, cuda=False)
        for name in topologies:
            ordinary = fields[name, mode, False][0][targets]
            reduced = fields[name, mode, True][0][targets]
            np.testing.assert_allclose(ordinary, reference, rtol=2e-12, atol=2e-12)
            np.testing.assert_allclose(reduced, reference, rtol=2e-12, atol=2e-12)
            np.testing.assert_allclose(reduced, ordinary, rtol=2e-13, atol=2e-13)

    for key in fields:
        name, mode, reduced = key
        parts = components[key, 0]
        near = direct_near(topologies[name], material, states[0], mode)
        np.testing.assert_allclose(parts["H_p2p"], near, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(
            parts["H_far"] + parts["H_p2p"],
            parts["H_total"],
            rtol=2e-13,
            atol=2e-13,
        )


def test_prebuilt_topology_preserves_per_particle_cuboid_size_association():
    positions = np.asarray([
        [-0.31, -0.29, -0.27], [0.33, 0.25, -0.21],
        [-0.23, 0.31, 0.29], [0.27, -0.33, 0.23],
        [0.29, 0.27, 0.31], [-0.25, -0.21, 0.33],
        [0.21, -0.25, -0.31], [-0.33, 0.29, -0.25],
    ])
    sides = np.asarray([0.041, 0.053, 0.067, 0.079, 0.083, 0.097, 0.103, 0.109])
    moments = np.arange(1, 25, dtype=float).reshape(-1, 3) * 1.0e-4
    material = dict(positions=positions, sides=sides, moments=moments)
    tree = cdfmm.AdaptiveTree(positions, _adaptive_options(capacity=1, max_depth=1))
    topology = tree.topology
    assert tuple(topology.source_permutation) != tuple(range(len(positions)))

    reference = direct_reference(material, moments, "cuboid-cuboid", cuda=False)
    for reduced in (False, True):
        options = make_options(
            material, cdfmm.ExecutionBackend.CPU_STATIC, order=3,
            interaction_mode="cuboid-cuboid", reduced=reduced,
            precision=cdfmm.StaticPrecision.FLOAT64,
        )
        actual = np.asarray(cdfmm.build_static_fmm(topology, options).evaluate(moments)["H"])
        np.testing.assert_allclose(actual, reference, rtol=3e-12, atol=3e-12)


def test_cuboid_self_field_is_finite_and_not_removed():
    material = dict(
        positions=np.asarray([[0.125, -0.0625, 0.25]]),
        sides=np.asarray([0.2]),
        moments=np.asarray([[0.003, -0.002, 0.001]]),
    )
    tree = cdfmm.AdaptiveTree(material["positions"],
                              _adaptive_options(capacity=1, max_depth=0))
    for mode in ("cuboid-point", "cuboid-cuboid"):
        reference = direct_reference(material, material["moments"], mode, cuda=False)
        options = make_options(
            material, cdfmm.ExecutionBackend.CPU_STATIC, order=2,
            interaction_mode=mode, reduced=True,
            precision=cdfmm.StaticPrecision.FLOAT64,
        )
        actual = np.asarray(cdfmm.build_static_fmm(
            tree.topology, options).evaluate(material["moments"])["H"])
        assert np.isfinite(actual).all()
        assert np.linalg.norm(actual) > 0.0
        np.testing.assert_allclose(actual, reference, rtol=2e-13, atol=2e-13)
