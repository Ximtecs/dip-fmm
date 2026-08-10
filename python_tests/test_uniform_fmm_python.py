import numpy as np
import pytest

import cdfmm


POSITIONS = np.array(
    [
        [-0.83, -0.71, -0.64],
        [0.76, -0.58, -0.42],
        [-0.61, 0.69, -0.37],
        [0.57, 0.73, 0.66],
        [-0.14, 0.22, 0.51],
        [0.31, -0.19, 0.12],
    ],
    dtype=float,
)

MOMENTS = np.array(
    [
        [0.7, -0.2, 0.1],
        [-0.4, 0.8, 0.3],
        [0.2, 0.1, -0.6],
        [-0.3, -0.5, 0.9],
        [0.6, 0.4, -0.2],
        [-0.1, 0.3, 0.5],
    ],
    dtype=float,
)


def make_fmm(order=4, level=3):
    tree_options = cdfmm.UniformTreeOptions()
    tree_options.max_level = level
    tree_options.root_centre = cdfmm.Vec3(0.0, 0.0, 0.0)
    tree_options.root_half_width = 1.0

    options = cdfmm.UniformFmmOptions()
    options.expansion_order = order
    options.tree = tree_options
    return cdfmm.UniformFmm(POSITIONS, options)


@pytest.mark.parametrize("order", [1, 3, 5])
@pytest.mark.parametrize("level", [0, 2, 3])
def test_upward_root_matches_direct_p2m(order, level):
    fmm = make_fmm(order, level)
    fmm.upward_pass(MOMENTS)

    centre = np.array(
        [fmm.tree.root_centre.x, fmm.tree.root_centre.y, fmm.tree.root_centre.z]
    )
    direct = cdfmm.p2m_dipole(centre, POSITIONS, MOMENTS, order)
    np.testing.assert_allclose(fmm.root_multipole, direct, rtol=1.0e-13, atol=2.0e-13)
    assert fmm.root_multipole.shape == (len(cdfmm.multi_indices(order)),)


def test_upward_pass_resets_state_and_validates_moment_count():
    fmm = make_fmm()
    fmm.upward_pass(MOMENTS)
    assert np.any(fmm.root_multipole != 0.0)

    fmm.upward_pass(np.zeros_like(MOMENTS))
    for node in fmm.tree.nodes:
        np.testing.assert_array_equal(fmm.multipole(node.index), 0.0)

    with pytest.raises(ValueError, match="one dipole moment per source position"):
        fmm.upward_pass(MOMENTS[:-1])


def test_upward_pass_handles_user_order_without_manual_sorting():
    permutation = np.array([4, 1, 5, 0, 3, 2])
    options = cdfmm.UniformFmmOptions()
    options.expansion_order = 4
    options.tree.max_level = 2
    options.tree.root_centre = cdfmm.Vec3(0.0, 0.0, 0.0)
    options.tree.root_half_width = 1.0

    original = cdfmm.UniformFmm(POSITIONS, options)
    shuffled = cdfmm.UniformFmm(POSITIONS[permutation], options)
    original.upward_pass(MOMENTS)
    shuffled.upward_pass(MOMENTS[permutation])

    np.testing.assert_allclose(
        shuffled.root_multipole,
        original.root_multipole,
        rtol=1.0e-13,
        atol=2.0e-13,
    )


def test_uniform_fmm_rejects_negative_expansion_order():
    options = cdfmm.UniformFmmOptions()
    options.expansion_order = -1
    with pytest.raises(ValueError, match="expansion_order must be >= 0"):
        cdfmm.UniformFmm(POSITIONS, options)


def test_complete_evaluation_shapes_ordering_and_repeated_state():
    targets = np.array(
        [[0.71, 0.66, 0.62], [-0.74, -0.69, -0.57], [0.13, -0.28, 0.45]]
    )
    options = cdfmm.UniformFmmOptions()
    options.expansion_order = 4
    options.tree.max_level = 2
    options.tree.root_centre = cdfmm.Vec3(0.0, 0.0, 0.0)
    options.tree.root_half_width = 1.0
    fmm = cdfmm.UniformFmm(POSITIONS, targets, options)

    result = fmm.evaluate(MOMENTS, output="both")
    assert result["H"].shape == (len(targets), 3)
    assert result["phi"].shape == (len(targets),)
    direct = np.stack(
        [cdfmm.p2p_dipole_sum(target, POSITIONS, MOMENTS)["H"] for target in targets]
    )
    relative = np.linalg.norm(result["H"] - direct, axis=1) / np.linalg.norm(
        direct, axis=1
    )
    assert np.max(relative) < 0.03

    reset = fmm.evaluate(np.zeros_like(MOMENTS))
    np.testing.assert_array_equal(reset["H"], 0.0)
    for node in fmm.tree.nodes:
        np.testing.assert_array_equal(fmm.local(node.index), 0.0)


def test_complete_source_point_evaluation_uses_explicit_identities():
    options = cdfmm.UniformFmmOptions()
    options.expansion_order = 3
    options.tree.max_level = 0
    fmm = cdfmm.UniformFmm(POSITIONS, POSITIONS, options)
    identities = np.arange(len(POSITIONS), dtype=int)
    actual = fmm.evaluate(MOMENTS, target_source_indices=identities)["H"]
    direct = np.stack(
        [
            cdfmm.p2p_dipole_sum(
                target, POSITIONS, MOMENTS, self_index=target_index
            )["H"]
            for target_index, target in enumerate(POSITIONS)
        ]
    )
    np.testing.assert_allclose(actual, direct, rtol=1.0e-14, atol=1.0e-14)
