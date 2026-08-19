import numpy as np
import pytest

import cdfmm


SOURCE_CENTRE = np.array([0.0, 0.0, 0.0])
TARGET_CENTRE = np.array([4.0, 0.5, -0.25])
CHILD_TARGET_CENTRE = np.array([4.1, 0.45, -0.2])
SOURCE_POSITIONS = np.array(
    [
        [-0.20, 0.10, 0.05],
        [0.15, -0.10, 0.20],
        [0.05, 0.20, -0.15],
    ],
    dtype=float,
)
DIPOLE_MOMENTS = np.array(
    [
        [1.00, -0.20, 0.10],
        [-0.40, 0.70, 0.25],
        [0.30, 0.10, -0.60],
    ],
    dtype=float,
)
ORDER = 5


def direct_field(target_position):
    """Return the direct C++ P2P reference field for the shared source set."""
    result = cdfmm.p2p_dipole_sum(
        target_position,
        SOURCE_POSITIONS,
        DIPOLE_MOMENTS,
        output="field",
    )
    return result["H"]


def test_direct_p2p_reference_python():
    identities = np.arange(SOURCE_POSITIONS.shape[0], dtype=np.int32)
    result = cdfmm.direct_p2p_reference(
        SOURCE_POSITIONS,
        SOURCE_POSITIONS,
        DIPOLE_MOMENTS,
        output="both",
        target_source_indices=identities,
    )

    expected_fields = []
    expected_potentials = []
    for target_index, target_position in enumerate(SOURCE_POSITIONS):
        expected = cdfmm.p2p_dipole_sum(
            target_position,
            SOURCE_POSITIONS,
            DIPOLE_MOMENTS,
            output="both",
            self_index=target_index,
        )
        expected_fields.append(expected["H"])
        expected_potentials.append(expected["phi"])

    np.testing.assert_allclose(result["H"], expected_fields)
    np.testing.assert_allclose(result["phi"], expected_potentials)


def test_direct_p2p_reference_python_validates_inputs():
    with pytest.raises(ValueError, match="one moment per source"):
        cdfmm.direct_p2p_reference(
            SOURCE_POSITIONS,
            SOURCE_POSITIONS,
            DIPOLE_MOMENTS[:2],
        )

    with pytest.raises(ValueError, match="one source identity per target"):
        cdfmm.direct_p2p_reference(
            SOURCE_POSITIONS,
            SOURCE_POSITIONS,
            DIPOLE_MOMENTS,
            target_source_indices=[0],
        )


def test_cuda_direct_p2p_reference_python():
    identities = np.arange(SOURCE_POSITIONS.shape[0], dtype=np.int32)

    if not cdfmm.cuda_direct_available():
        with pytest.raises(RuntimeError):
            cdfmm.cuda_direct_p2p_reference(
                SOURCE_POSITIONS,
                SOURCE_POSITIONS,
                DIPOLE_MOMENTS,
                target_source_indices=identities,
            )
        with pytest.raises(RuntimeError):
            cdfmm.CudaDirectPlan(
                SOURCE_POSITIONS,
                SOURCE_POSITIONS,
                target_source_indices=identities,
            )
        return

    expected = cdfmm.direct_p2p_reference(
        SOURCE_POSITIONS,
        SOURCE_POSITIONS,
        DIPOLE_MOMENTS,
        output="both",
        target_source_indices=identities,
    )
    actual = cdfmm.cuda_direct_p2p_reference(
        SOURCE_POSITIONS,
        SOURCE_POSITIONS,
        DIPOLE_MOMENTS,
        output="both",
        target_source_indices=identities,
    )

    np.testing.assert_allclose(actual["H"], expected["H"])
    np.testing.assert_allclose(actual["phi"], expected["phi"])

    plan = cdfmm.CudaDirectPlan(
        SOURCE_POSITIONS,
        SOURCE_POSITIONS,
        target_source_indices=identities,
    )
    assert plan.source_count == SOURCE_POSITIONS.shape[0]
    assert plan.target_count == SOURCE_POSITIONS.shape[0]

    for scale in (1.0, -0.25):
        scaled_moments = scale * DIPOLE_MOMENTS
        expected = cdfmm.direct_p2p_reference(
            SOURCE_POSITIONS,
            SOURCE_POSITIONS,
            scaled_moments,
            output="both",
            target_source_indices=identities,
        )
        actual = plan.evaluate(scaled_moments, output="both")
        np.testing.assert_allclose(actual["H"], expected["H"])
        np.testing.assert_allclose(actual["phi"], expected["phi"])


def test_p2m_python():
    indices = cdfmm.multi_indices(2)
    coefficients = cdfmm.p2m_dipole(
        SOURCE_CENTRE,
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[1.0, 2.0, 3.0]]),
        order=2,
    )

    assert indices.shape == (10, 3)
    assert coefficients.shape == (10,)
    assert coefficients[0] == 0.0

    coefficient_by_index = {
        tuple(alpha): value
        for alpha, value in zip(indices, coefficients)
    }
    assert coefficient_by_index[(1, 0, 0)] == -1.0
    assert coefficient_by_index[(0, 1, 0)] == -2.0
    assert coefficient_by_index[(0, 0, 1)] == -3.0


def test_m2m_python():
    child_centre = np.array([0.1, -0.1, 0.05])
    child_coefficients = cdfmm.p2m_dipole(
        child_centre,
        SOURCE_POSITIONS,
        DIPOLE_MOMENTS,
        order=ORDER,
    )
    translated_coefficients = cdfmm.m2m(
        child_coefficients,
        child_centre,
        SOURCE_CENTRE,
        order=ORDER,
    )
    direct_parent_coefficients = cdfmm.p2m_dipole(
        SOURCE_CENTRE,
        SOURCE_POSITIONS,
        DIPOLE_MOMENTS,
        order=ORDER,
    )

    np.testing.assert_allclose(
        translated_coefficients,
        direct_parent_coefficients,
        rtol=1.0e-13,
        atol=1.0e-14,
    )


def test_m2p_python():
    multipole_coefficients = cdfmm.p2m_dipole(
        SOURCE_CENTRE,
        SOURCE_POSITIONS,
        DIPOLE_MOMENTS,
        order=ORDER,
    )
    target_position = np.array([4.2, 0.4, -0.1])
    result = cdfmm.m2p(
        multipole_coefficients,
        SOURCE_CENTRE,
        target_position,
        order=ORDER,
        output="both",
    )

    np.testing.assert_allclose(
        result["H"],
        direct_field(target_position),
        rtol=2.0e-5,
        atol=1.0e-12,
    )
    assert np.isfinite(result["phi"])


def test_m2l_python():
    multipole_coefficients = cdfmm.p2m_dipole(
        SOURCE_CENTRE,
        SOURCE_POSITIONS,
        DIPOLE_MOMENTS,
        order=ORDER,
    )
    local_coefficients = cdfmm.m2l(
        multipole_coefficients,
        SOURCE_CENTRE,
        TARGET_CENTRE,
        order=ORDER,
    )

    assert local_coefficients.shape == multipole_coefficients.shape
    assert np.all(np.isfinite(local_coefficients))
    np.testing.assert_allclose(
        -local_coefficients[[3, 2, 1]],
        direct_field(TARGET_CENTRE),
        rtol=2.0e-5,
        atol=1.0e-12,
    )


def test_l2l_python():
    multipole_coefficients = cdfmm.p2m_dipole(
        SOURCE_CENTRE,
        SOURCE_POSITIONS,
        DIPOLE_MOMENTS,
        order=ORDER,
    )
    parent_local = cdfmm.m2l(
        multipole_coefficients,
        SOURCE_CENTRE,
        TARGET_CENTRE,
        order=ORDER,
    )
    child_local = cdfmm.l2l(
        parent_local,
        TARGET_CENTRE,
        CHILD_TARGET_CENTRE,
        order=ORDER,
    )
    target_position = CHILD_TARGET_CENTRE + np.array([0.01, -0.02, 0.015])
    result = cdfmm.l2p(
        child_local,
        CHILD_TARGET_CENTRE,
        target_position,
        order=ORDER,
    )

    np.testing.assert_allclose(
        result["H"],
        direct_field(target_position),
        rtol=3.0e-5,
        atol=1.0e-12,
    )


def test_l2p_python():
    multipole_coefficients = cdfmm.p2m_dipole(
        SOURCE_CENTRE,
        SOURCE_POSITIONS,
        DIPOLE_MOMENTS,
        order=ORDER,
    )
    local_coefficients = cdfmm.m2l(
        multipole_coefficients,
        SOURCE_CENTRE,
        TARGET_CENTRE,
        order=ORDER,
    )
    target_position = TARGET_CENTRE + np.array([0.05, -0.03, 0.02])
    result = cdfmm.l2p(
        local_coefficients,
        TARGET_CENTRE,
        target_position,
        order=ORDER,
        output="both",
    )

    np.testing.assert_allclose(
        result["H"],
        direct_field(target_position),
        rtol=3.0e-5,
        atol=1.0e-12,
    )
    assert np.isfinite(result["phi"])


def test_operator_bindings_validate_shapes_and_output_mode():
    with pytest.raises(ValueError, match="shape"):
        cdfmm.p2m_dipole(
            SOURCE_CENTRE,
            np.ones((3, 2)),
            DIPOLE_MOMENTS,
            order=ORDER,
        )

    with pytest.raises(ValueError, match="same number"):
        cdfmm.p2m_dipole(
            SOURCE_CENTRE,
            SOURCE_POSITIONS[:2],
            DIPOLE_MOMENTS,
            order=ORDER,
        )

    with pytest.raises(ValueError, match="output"):
        cdfmm.p2p_dipole_pair(
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            output="invalid",
        )
