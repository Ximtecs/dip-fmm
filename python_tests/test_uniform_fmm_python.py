import numpy as np
import pytest

import cdfmm


def test_cuda_m2l_p2p_backend_aliases_are_compatible():
    assert cdfmm.ExecutionBackend.CUDA_M2L == (
        cdfmm.ExecutionBackend.CUDA_M2L_P2P
    )
    assert cdfmm.ExecutionBackend.CUDA_M2L_STATIC_P2P == (
        cdfmm.ExecutionBackend.CUDA_M2L_P2P
    )
    assert cdfmm.cuda_m2l_available() == cdfmm.cuda_m2l_p2p_available()


def test_static_backend_is_default_and_reference_is_selectable():
    sources = np.array([[-0.75, 0.0, 0.0], [0.75, 0.0, 0.0]])
    moments = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    options = cdfmm.UniformFmmOptions()
    options.expansion_basis = cdfmm.ExpansionBasis.CARTESIAN
    options.precision = cdfmm.StaticPrecision.FLOAT64
    assert options.static_matrix_backend == cdfmm.StaticMatrixBackend.PORTABLE
    options.backend = cdfmm.ExecutionBackend.CPU_STATIC
    options.tree.max_level = 2
    static_fmm = cdfmm.UniformFmm(sources, sources, options)
    assert static_fmm.m2l_backend == cdfmm.M2LBackend.Static
    assert (
        static_fmm.p2p_execution_packing
        == cdfmm.P2PExecutionPacking.POINT_GEOMETRY
    )

    options.m2l_backend = cdfmm.M2LBackend.Reference
    options.backend = cdfmm.ExecutionBackend.CPU_REFERENCE
    reference_fmm = cdfmm.UniformFmm(sources, sources, options)
    identities = np.arange(sources.shape[0], dtype=int)
    static_result = static_fmm.evaluate(
        moments,
        target_source_indices=identities,
    )
    reference_result = reference_fmm.evaluate(
        moments,
        target_source_indices=identities,
    )
    assert np.isfinite(static_result["H"]).all()
    # The two traversal orders can leave different round-off residuals in
    # analytically zero components.  Compare those values at an absolute
    # tolerance close to machine precision as well as relatively elsewhere.
    np.testing.assert_allclose(
        static_result["H"],
        reference_result["H"],
        rtol=1.0e-12,
        atol=1.0e-15,
    )


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
    options.expansion_basis = cdfmm.ExpansionBasis.CARTESIAN
    options.precision = cdfmm.StaticPrecision.FLOAT64
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
    options.expansion_basis = cdfmm.ExpansionBasis.CARTESIAN
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
    options.expansion_basis = cdfmm.ExpansionBasis.CARTESIAN
    options.expansion_order = -1
    with pytest.raises(ValueError, match="expansion_order must be >= 0"):
        cdfmm.UniformFmm(POSITIONS, options)


def test_complete_evaluation_shapes_ordering_and_repeated_state():
    targets = np.array(
        [[0.71, 0.66, 0.62], [-0.74, -0.69, -0.57], [0.13, -0.28, 0.45]]
    )
    options = cdfmm.UniformFmmOptions()
    options.expansion_basis = cdfmm.ExpansionBasis.CARTESIAN
    options.expansion_order = 4
    options.backend = cdfmm.ExecutionBackend.CPU_STATIC
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
    options.expansion_basis = cdfmm.ExpansionBasis.CARTESIAN
    options.precision = cdfmm.StaticPrecision.FLOAT32
    options.expansion_order = 3
    options.backend = cdfmm.ExecutionBackend.CPU_STATIC
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
    np.testing.assert_allclose(actual, direct, rtol=1.0e-6, atol=1.0e-7)


def test_fixed_identity_option_is_used_when_evaluate_omits_the_map():
    identities = np.arange(len(POSITIONS), dtype=int)
    options = cdfmm.UniformFmmOptions()
    options.expansion_basis = cdfmm.ExpansionBasis.CARTESIAN
    options.backend = cdfmm.ExecutionBackend.CPU_STATIC
    options.tree.max_level = 0
    options.fixed_target_source_indices = identities.tolist()
    options.cuda_p2p_bsr_max_bytes = 1234
    fmm = cdfmm.UniformFmm(POSITIONS, POSITIONS, options)

    implicit = fmm.evaluate(MOMENTS)["H"]
    explicit = fmm.evaluate(
        MOMENTS, target_source_indices=identities
    )["H"]

    np.testing.assert_allclose(implicit, explicit, rtol=0.0, atol=0.0)
    assert options.cuda_p2p_bsr_max_bytes == 1234


def test_spatial_layout_option_round_trips():
    sources = np.array([[-0.75, 0.0, 0.0], [0.75, 0.0, 0.0]])
    options = cdfmm.UniformFmmOptions()
    assert options.spatial_layout == cdfmm.SpatialLayout.GENERAL
    options.spatial_layout = cdfmm.SpatialLayout.REGULAR_GRID
    options.expansion_basis = cdfmm.ExpansionBasis.CARTESIAN
    options.precision = cdfmm.StaticPrecision.FLOAT64
    options.backend = cdfmm.ExecutionBackend.CPU_STATIC
    options.tree.max_level = 1
    options.enable_cache = False
    # CPU backends record the hint but keep their own packing policy.
    fmm = cdfmm.UniformFmm(sources, sources, options)
    assert fmm.spatial_layout == cdfmm.SpatialLayout.REGULAR_GRID
    assert fmm.p2p_execution_packing == cdfmm.P2PExecutionPacking.POINT_GEOMETRY


def test_regular_grid_hint_selects_cuda_dictionary():
    if not cdfmm.cuda_full_available():
        pytest.skip("CUDA full backend is unavailable")
    axis = np.linspace(-0.875, 0.875, 8)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    positions = grid.reshape(-1, 3)
    moments = np.tile(np.array([[0.3, -0.2, 0.5]]), (positions.shape[0], 1))
    identities = np.arange(positions.shape[0], dtype=np.int32)
    options = cdfmm.UniformFmmOptions()
    options.precision = cdfmm.StaticPrecision.FLOAT64
    options.expansion_order = 4
    options.tree.max_level = 2
    options.backend = cdfmm.ExecutionBackend.CUDA_FULL
    options.fixed_target_source_indices = identities.tolist()
    options.enable_cache = False
    general = cdfmm.UniformFmm(positions, positions, options)
    assert general.p2p_execution_packing == cdfmm.P2PExecutionPacking.LEAF_BLOCK
    options.spatial_layout = cdfmm.SpatialLayout.REGULAR_GRID
    regular = cdfmm.UniformFmm(positions, positions, options)
    assert regular.spatial_layout == cdfmm.SpatialLayout.REGULAR_GRID
    assert (
        regular.p2p_execution_packing
        == cdfmm.P2PExecutionPacking.TENSOR_DICTIONARY
    )
    expected = general.evaluate(moments, target_source_indices=identities)
    actual = regular.evaluate(moments, target_source_indices=identities)
    np.testing.assert_allclose(actual["H"], expected["H"], rtol=1e-9, atol=1e-12)


def test_explicit_p2p_packing_request_is_honoured_and_reported():
    rng = np.random.default_rng(11)
    positions = rng.uniform(-0.9, 0.9, size=(64, 3))
    moments = rng.uniform(-1.0, 1.0, size=(64, 3))
    identities = np.arange(positions.shape[0], dtype=np.int32)
    options = cdfmm.UniformFmmOptions()
    assert options.p2p_packing == cdfmm.P2PExecutionPacking.AUTO
    options.precision = cdfmm.StaticPrecision.FLOAT64
    options.expansion_order = 3
    options.tree.max_level = 2
    options.backend = cdfmm.ExecutionBackend.CPU_STATIC
    options.fixed_target_source_indices = identities.tolist()
    options.enable_cache = False
    automatic = cdfmm.UniformFmm(positions, positions, options)
    assert automatic.requested_p2p_packing == cdfmm.P2PExecutionPacking.AUTO
    assert automatic.p2p_execution_packing == cdfmm.P2PExecutionPacking.POINT_GEOMETRY
    expected = automatic.evaluate(moments, target_source_indices=identities)
    for packing in (
        cdfmm.P2PExecutionPacking.CANONICAL_AOS,
        cdfmm.P2PExecutionPacking.PARTICLE_ROW_SOA,
        cdfmm.P2PExecutionPacking.TENSOR_DICTIONARY,
        cdfmm.P2PExecutionPacking.POINT_GEOMETRY,
    ):
        options.p2p_packing = packing
        forced = cdfmm.UniformFmm(positions, positions, options)
        assert forced.requested_p2p_packing == packing
        assert forced.p2p_execution_packing == packing
        actual = forced.evaluate(moments, target_source_indices=identities)
        np.testing.assert_allclose(actual["H"], expected["H"], rtol=1e-11, atol=1e-13)


def test_unsupported_explicit_p2p_packing_raises_with_reason():
    positions = np.array([[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0]])
    options = cdfmm.UniformFmmOptions()
    options.backend = cdfmm.ExecutionBackend.CPU_STATIC
    options.enable_cache = False
    options.p2p_packing = cdfmm.P2PExecutionPacking.LEAF_BLOCK
    with pytest.raises(ValueError, match="CUDA execution packings"):
        cdfmm.UniformFmm(positions, positions, options)
    options.p2p_packing = cdfmm.P2PExecutionPacking.POINT_GEOMETRY
    options.source_geometry = cdfmm.SourceGeometry.RECTANGULAR_PRISM
    options.source_sizes = [cdfmm.RectangularPrism(0.05, 0.05, 0.05)]
    with pytest.raises(ValueError, match="PointGeometry"):
        cdfmm.UniformFmm(positions, positions, options)
