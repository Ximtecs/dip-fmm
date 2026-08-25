import numpy as np

import cdfmm


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
