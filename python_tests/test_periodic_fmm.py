import json
from pathlib import Path

import numpy as np

import cdfmm


REPOSITORY_ROOT = Path(__file__).parents[1]
COMPARISON_NOTEBOOK = (
    REPOSITORY_ROOT / "examples/notebooks/14_periodic_fmm_direct_compare.ipynb"
)


def test_periodic_point_self_retains_nonzero_images():
    options = cdfmm.UniformFmmOptions()
    options.precision = cdfmm.StaticPrecision.FLOAT64
    options.expansion_order = 2
    options.tree.max_level = 0
    options.periodic.enabled = True
    options.fixed_target_source_indices = [0]

    positions = np.zeros((1, 3))
    plan = cdfmm.UniformFmm(positions, positions, options)
    result = plan.evaluate(np.array([[0.0, 0.0, 1.0]]))

    assert plan.periodic_cell.enabled
    np.testing.assert_allclose(
        result["H"], [[0.0, 0.0, 1.0 / 3.0]], rtol=0.0, atol=1.0e-9
    )


def test_periodic_positions_wrap_before_tree_construction():
    options = cdfmm.UniformFmmOptions()
    options.precision = cdfmm.StaticPrecision.FLOAT64
    options.expansion_order = 4
    options.tree.max_level = 2
    options.periodic.enabled = True

    target = np.array([[-0.45, -0.1, 0.2]])
    moment = np.array([[0.2, -0.3, 0.7]])
    wrapped = cdfmm.UniformFmm(np.array([[0.45, 0.1, -0.2]]), target, options)
    shifted = cdfmm.UniformFmm(np.array([[1.45, 0.1, -0.2]]), target, options)

    np.testing.assert_allclose(
        shifted.evaluate(moment)["H"],
        wrapped.evaluate(moment)["H"],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_periodic_comparison_switches_exact_near_field_geometry():
    notebook = json.loads(COMPARISON_NOTEBOOK.read_text(encoding="utf-8"))
    sources = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    combined_source = "\n".join(sources)

    assert "USE_CUBOIDS =" in sources[0]
    assert "moments = CUBOID_VOLUME * magnetisations" in combined_source
    assert "cdfmm.CudaDenseDirectPlan(" in combined_source
    assert "cdfmm.cuda_dense_direct_available()" in combined_source
    assert "SourceGeometry.UNIFORM_CUBOID" in combined_source
    assert "TargetGeometry.VOLUME_AVERAGED_CUBOID" in combined_source
    assert "options.use_cuboid_p2m = False" in combined_source
    assert "options.use_cuboid_l2p = False" in combined_source

    for index, source in enumerate(sources):
        compile(source, f"{COMPARISON_NOTEBOOK.name}:cell-{index}", "exec")
