import json
from pathlib import Path

import numpy as np

import cdfmm
from examples.notebooks.fmm_memory import (
    coefficient_count,
    estimate_source_point_storage,
    p2m_or_l2p_entries_per_particle,
    shift_entries,
)


REPOSITORY_ROOT = Path(__file__).parents[1]
NOTEBOOK = REPOSITORY_ROOT / "examples/notebooks/12_cuda_memory_usage.ipynb"


def test_static_m2l_matrix_matches_independent_operator_columns():
    order = 3
    coefficient_total = coefficient_count(order)
    source_centre = np.array([-0.5, 0.25, 0.0])
    target_centre = np.array([0.75, -0.5, 0.5])
    matrix = cdfmm.static_m2l_matrix(source_centre, target_centre, order)

    assert matrix.shape == (coefficient_total, coefficient_total)
    for column in (0, 3, coefficient_total - 1):
        multipole = np.zeros(coefficient_total)
        multipole[column] = 1.0
        expected = cdfmm.m2l(
            multipole, source_centre, target_centre, order
        )
        np.testing.assert_allclose(matrix[:, column], expected)


def test_combinatorial_operator_counts():
    assert coefficient_count(6) == 84
    assert p2m_or_l2p_entries_per_particle(6) == 168
    assert shift_entries(6) == 924


def test_single_particle_storage_has_manual_counts():
    estimate = estimate_source_point_storage(
        np.array([[0.25, -0.25, 0.25]]), order=0, depth=1
    )

    assert estimate.nodes == 9
    assert estimate.occupied_source_nodes == 1
    assert estimate.occupied_target_nodes == 1
    assert estimate.transfer_classes == 0
    assert estimate.m2l_interactions == 0
    assert estimate.p2p_pairs == 1
    assert estimate.host_static_bytes == 776
    assert estimate.cuda_partial_bytes == 356
    assert estimate.cuda_full_bytes == 724


def test_storage_estimate_detects_far_field_interactions():
    positions = np.array([
        [-0.9, -0.9, -0.9],
        [0.9, 0.9, 0.9],
    ])
    estimate = estimate_source_point_storage(positions, order=2, depth=2)

    assert estimate.m2l_interactions == 2
    assert estimate.p2p_pairs == 2
    assert estimate.transfer_classes == 2
    assert estimate.cuda_full["shared M2L matrices"] > 0
    assert estimate.cuda_full["M2L interaction metadata"] > 0
    assert estimate.cuda_partial["M2L class matrices and indices"] > 0


def test_host_storage_estimate_matches_constructed_cpu_static_plan():
    positions = np.random.default_rng(3).uniform(-0.9, 0.9, size=(24, 3))
    estimate = estimate_source_point_storage(positions, order=3, depth=2)
    options = cdfmm.UniformFmmOptions()
    options.precision = cdfmm.StaticPrecision.FLOAT64
    options.expansion_order = 3
    options.tree.max_level = 2
    options.tree.root_centre = cdfmm.Vec3(0.0, 0.0, 0.0)
    options.tree.root_half_width = 1.0
    options.backend = cdfmm.ExecutionBackend.CPU_STATIC

    plan = cdfmm.UniformFmm(positions, positions, options)

    assert estimate.host_static_bytes == plan.static_plan_statistics["total_bytes"]


def test_memory_notebook_is_valid_json_with_compilable_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["display_name"] == "cdfmm"
    sources = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    combined_source = "\n".join(sources)
    assert "PARTICLE_COUNTS =" in sources[0]
    assert "EXPANSION_ORDERS =" in sources[0]
    assert "TREE_DEPTHS =" in sources[0]
    assert "GB = 1_000_000_000" in combined_source
    assert "MIB =" not in combined_source
    assert "CPU host" in combined_source
    assert "GPU device" in combined_source
    assert "cdfmm.UniformFmm(" in combined_source
    assert "persistent_device_bytes" in combined_source
    assert "m2l_unique_matrix_count" in combined_source
    for index, source in enumerate(sources):
        compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec")
