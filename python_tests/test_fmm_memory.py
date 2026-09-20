from pathlib import Path

import numpy as np

import cdfmm
from memory_model import (
    coefficient_count,
    estimate_source_point_storage,
    p2m_or_l2p_entries_per_particle,
    shift_entries,
)


REPOSITORY_ROOT = Path(__file__).parents[1]


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
        np.array([[0.25, -0.25, 0.25]]), order=0, depth=1, threads=1
    )

    assert estimate.nodes == 9
    assert estimate.occupied_source_nodes == 1
    assert estimate.occupied_target_nodes == 1
    assert estimate.transfer_classes == 0
    assert estimate.m2l_interactions == 0
    assert estimate.p2p_pairs == 1
    # One thread's P2P position scratch (16 rounded sources x 52 B) 832,
    # packed P2M rows 24, shared M2M/L2L maps 256, level-scaled banks 656,
    # M2L level scalings 32, M2L indices 140, L2P rows 32, multipole and local
    # state 144, moments/results/near fields/identities 84.
    assert estimate.host_static_bytes == 2200
    assert estimate.cuda_partial_bytes == 360
    assert estimate.cuda_full_bytes == 728


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
    # Compare against a cold plan so cache state cannot change the ownership
    # path being measured. Both cold construction and cache hits retain the
    # shared M2M/L2L maps, including when the universal bank is requested.
    estimate = estimate_source_point_storage(
        positions, order=3, depth=2, universal_translation_bank=True
    )
    options = cdfmm.UniformFmmOptions()
    options.expansion_basis = cdfmm.ExpansionBasis.CARTESIAN
    options.precision = cdfmm.StaticPrecision.FLOAT64
    options.expansion_order = 3
    options.tree.max_level = 2
    options.tree.root_centre = cdfmm.Vec3(0.0, 0.0, 0.0)
    options.tree.root_half_width = 1.0
    options.backend = cdfmm.ExecutionBackend.CPU_STATIC
    options.enable_cache = False

    plan = cdfmm.UniformFmm(positions, positions, options)

    assert estimate.host_static_bytes == plan.static_plan_statistics["total_bytes"]
