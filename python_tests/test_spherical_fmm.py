import json
from pathlib import Path

import numpy as np
import pytest

import cdfmm


def test_spherical_configuration_modes_and_statistics():
    modes = cdfmm.spherical_modes(3)
    assert modes.shape == (16, 2)
    np.testing.assert_array_equal(modes[:4], [[0, 0], [1, -1], [1, 0], [1, 1]])

    positions = np.random.default_rng(41).uniform(-0.9, 0.9, size=(48, 3))
    moments = np.random.default_rng(42).normal(size=(48, 3))
    identities = np.arange(len(positions), dtype=np.int32)
    options = cdfmm.UniformFmmOptions()
    assert options.expansion_basis == cdfmm.ExpansionBasis.SPHERICAL
    options.expansion_basis = "cartesian"
    assert options.expansion_basis == cdfmm.ExpansionBasis.CARTESIAN
    options.expansion_basis = "spherical"
    options.expansion_order = 4
    options.precision = cdfmm.StaticPrecision.FLOAT64
    options.tree.max_level = 2
    plan = cdfmm.UniformFmm(positions, positions, options)
    result = plan.evaluate(moments, target_source_indices=identities)

    assert plan.expansion_basis == cdfmm.ExpansionBasis.SPHERICAL
    assert plan.coefficient_count == 25
    assert result["H"].shape == positions.shape
    statistics = dict(plan.static_plan_statistics)
    assert statistics["spherical"] is True
    assert statistics["coefficient_count"] == 25
    assert statistics["p2m_operator_bytes"] > 0
    assert statistics["l2p_operator_bytes"] > 0
    assert statistics["multipole_state_bytes"] == statistics["local_state_bytes"]
    assert statistics["total_persistent_bytes"] > statistics["total_bytes"]


def test_spherical_accepts_cuboids_and_rejects_reference_execution():
    positions = np.zeros((1, 3))
    options = cdfmm.UniformFmmOptions()
    options.expansion_basis = cdfmm.ExpansionBasis.SPHERICAL
    options.source_geometry = cdfmm.SourceGeometry.UNIFORM_CUBOID
    options.source_sizes = [cdfmm.CuboidSize(1.0, 1.0, 1.0)]
    options.target_geometry = cdfmm.TargetGeometry.VOLUME_AVERAGED_CUBOID
    options.target_sizes = options.source_sizes
    options.tree.root_centre = cdfmm.Vec3(0.0, 0.0, 0.0)
    options.tree.root_half_width = 1.0
    plan = cdfmm.UniformFmm(positions, positions, options)
    result = plan.evaluate(np.array([[0.0, 0.0, 1.0]]))
    np.testing.assert_allclose(result["H"], [[0.0, 0.0, -1.0 / 3.0]])

    # Point L2P comparison mode must retain the exact cuboid self P2P field.
    options.use_cuboid_l2p = False
    point_l2p_plan = cdfmm.UniformFmm(positions, positions, options)
    point_l2p_result = point_l2p_plan.evaluate(np.array([[0.0, 0.0, 1.0]]))
    np.testing.assert_allclose(
        point_l2p_result["H"], [[0.0, 0.0, -1.0 / 3.0]]
    )

    options.source_geometry = cdfmm.SourceGeometry.POINT_DIPOLE
    options.source_sizes = []
    options.target_geometry = cdfmm.TargetGeometry.POINT
    options.target_sizes = []
    options.backend = cdfmm.ExecutionBackend.CPU_REFERENCE
    with pytest.raises(ValueError, match="static M2L"):
        cdfmm.UniformFmm(positions, positions, options)


def test_cartesian_spherical_comparison_notebook_is_valid_and_compilable():
    notebook_path = (
        Path(__file__).parents[1]
        / "examples"
        / "simple_notebooks"
        / "simple_cartesian_spherical_fmm_compare.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    for index, source in enumerate(sources):
        compile(source, f"{notebook_path.name}:cell-{index}", "exec")

    combined = "\n".join(sources)
    assert "ORDERS = [1, 2, 3, 4, 5, 6]" in combined
    assert "REPETITIONS = 7" in combined
    assert "ExpansionBasis.SPHERICAL" in combined
    assert "DenseDirectPlan" in combined
    assert "SourceGeometry.UNIFORM_CUBOID" in combined
    assert "TargetGeometry.VOLUME_AVERAGED_CUBOID" in combined


def test_spherical_cuboid_p2m_l2p_notebook_is_valid_and_compilable():
    notebook_path = (
        Path(__file__).parents[1]
        / "examples"
        / "simple_notebooks"
        / "simple_cuboid_p2m_l2p_direct_compare.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    for index, source in enumerate(sources):
        compile(source, f"{notebook_path.name}:cell-{index}", "exec")

    combined = "\n".join(sources)
    assert "ExpansionBasis.SPHERICAL" in combined
    assert "SourceGeometry.UNIFORM_CUBOID" in combined
    assert "TargetGeometry.VOLUME_AVERAGED_CUBOID" in combined
    assert "options.use_cuboid_p2m" in combined
    assert "options.use_cuboid_l2p" in combined
    assert "ORDERS = [4, 6]" in combined
