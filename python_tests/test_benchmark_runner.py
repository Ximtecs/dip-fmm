import pytest

from benchmarks.run_benchmarks import (
    CudaStatus,
    EVALUATION_PHASES,
    M2L_SUBPHASES,
    PROFILES,
    benchmark_backends,
    build_cases,
    comparison_case,
    expanded_cases,
    phase_breakdown_filename,
    profile_backends,
    progress_bar,
    representative_case,
    nested_m2l_phases,
    thread_counts_up_to,
    top_level_evaluation_phases,
    workload_values,
)


def test_thread_counts_respect_non_power_of_two_cap():
    assert thread_counts_up_to(8) == [1, 2, 4, 8]
    assert thread_counts_up_to(6) == [1, 2, 4, 6]


def test_representative_case_uses_largest_size_and_order_nearest_four():
    rows = [
        {"suite": "parameter_grid", "sources": "100", "order": "4"},
        {"suite": "parameter_grid", "sources": "300", "order": "2"},
        {"suite": "parameter_grid", "sources": "300", "order": "5"},
        {"suite": "scaling", "sources": "300", "order": "4"},
    ]

    assert representative_case(rows) == rows[2]


def test_comparison_case_uses_largest_completed_workload_case():
    rows = [
        {
            "suite": "comparison",
            "sources": "100",
            "order": "4",
            "workload_1_total_median": "0.2",
        },
        {
            "suite": "comparison",
            "sources": "300",
            "order": "2",
            "workload_1_total_median": "0.8",
        },
        {
            "suite": "comparison",
            "sources": "300",
            "order": "5",
            "workload_1_total_median": "0.8",
        },
        {
            "suite": "comparison",
            "sources": "1000",
            "order": "4",
            "workload_1_total_median": "nan",
        },
    ]

    assert comparison_case(rows) == rows[2]


def test_workload_values_separate_creation_from_evaluation():
    rows = [
        {
            "workload_1_evaluation_median": "0.1",
            "workload_1_total_median": "0.8",
            "workload_10_evaluation_median": "1.0",
            "workload_10_total_median": "1.7",
        }
    ]

    assert workload_values(rows, include_creation=False) == ([0.1], [1.0])
    assert workload_values(rows, include_creation=True) == ([0.8], [1.7])


def test_case_plan_contains_full_parameter_grid_and_capped_scaling_suite():
    profile = {
        "sizes": [100, 300],
        "orders": [2, 4],
        "depths": [2, 3],
        "evaluations": 2,
        "samples": 2,
    }

    cases = build_cases(profile, max_threads=6)

    grid = [case for case in cases if case.suite == "parameter_grid"]
    assert len(grid) == 8
    assert {
        (case.size, case.order, case.depth) for case in grid
    } == {
        (size, order, depth)
        for size in [100, 300]
        for order in [2, 4]
        for depth in [2, 3]
    }
    assert [case.threads for case in cases if case.suite == "scaling"] == [
        1,
        2,
        4,
        6,
    ]


def test_rough_profile_contains_exactly_four_geometries_and_no_scaling():
    profile = PROFILES["rough"]
    cases = build_cases(profile, max_threads=8)

    assert profile["evaluations"] == 1
    assert profile["samples"] == 1
    assert profile["warmups"] == 1
    assert profile["workload_comparison"] is True
    assert len(cases) == 4
    assert all(case.suite == "parameter_grid" for case in cases)
    assert all(case.order == 4 for case in cases)
    assert all(case.threads == 8 for case in cases)
    assert all(not case.direct for case in cases)
    assert {
        (case.size, case.depth)
        for case in cases
    } == {
        (20_000, 2),
        (20_000, 3),
        (30_000, 2),
        (30_000, 3),
    }


def test_rough_profile_requires_five_backends_and_expands_to_twenty_runs():
    profile = PROFILES["rough"]
    available = [
        "cpu-direct",
        "cuda-direct",
        "cuda-m2l",
        "cpu-static-matrix",
        "cpu-static-matrix-mkl",
    ]
    selected = profile_backends("rough", profile, available)

    assert selected == profile["backends"]
    assert len(expanded_cases(build_cases(profile, 8), selected)) == 20


def test_rough_profile_rejects_a_build_without_cuda_m2l():
    profile = PROFILES["rough"]
    available = [
        "cpu-direct",
        "cuda-direct",
        "cpu-static-matrix",
        "cpu-static-matrix-mkl",
    ]

    with pytest.raises(RuntimeError, match="cuda-m2l"):
        profile_backends("rough", profile, available)


def test_progress_and_phase_filename_identify_the_case():
    row = {
        "suite": "parameter_grid",
        "sources": "300",
        "order": "4",
        "depth": "2",
        "threads": "8",
    }

    assert progress_bar(2, 4) == "[============            ]"
    assert phase_breakdown_filename(row) == (
        "parameter_grid_n300_p4_d2_t8_cpu-static-matrix.png"
    )


def test_cuda_build_runs_every_case_with_cpu_and_direct_backends():
    cases = build_cases(
        {
            "sizes": [100],
            "orders": [4],
            "depths": [2],
            "evaluations": 1,
            "samples": 1,
        },
        max_threads=1,
    )
    backends = benchmark_backends(
        CudaStatus(
            compiled=True,
            available=True,
            direct_available=True,
            m2l_available=False,
            full_available=False,
            mkl_available=True,
            device="Test GPU",
        )
    )

    assert backends == [
        "cpu-direct",
        "cuda-direct",
        "cpu-static-matrix",
        "cpu-static-matrix-mkl",
    ]
    expanded = expanded_cases(cases, backends)
    for case in cases:
        selected = {
            (candidate, backend)
            for candidate, backend in expanded
            if candidate == case
        }
        assert selected == {
            (case, "cpu-direct"),
            (case, "cpu-static-matrix"),
            (case, "cpu-static-matrix-mkl"),
            (case, "cuda-direct"),
        }


def test_cpu_build_runs_only_static_benchmarks():
    status = CudaStatus(
        compiled=False,
        available=False,
        direct_available=False,
        m2l_available=False,
        full_available=False,
        mkl_available=False,
        device="",
    )
    assert benchmark_backends(status) == ["cpu-direct", "cpu-static-matrix"]


def test_cuda_benchmark_fails_clearly_without_a_device():
    status = CudaStatus(
        compiled=True,
        available=False,
        direct_available=False,
        m2l_available=False,
        full_available=False,
        mkl_available=False,
        device="",
    )
    with pytest.raises(RuntimeError, match="no CUDA device"):
        benchmark_backends(status)


def test_phase_plots_include_cuda_transfer_and_kernel_timings():
    assert EVALUATION_PHASES[-3:] == [
        ("cuda_h2d", "CUDA H2D"),
        ("cuda_kernel", "CUDA kernel"),
        ("cuda_d2h", "CUDA D2H"),
    ]


def test_phase_plots_expose_static_matrix_work():
    assert M2L_SUBPHASES == [
        ("m2l_gather", "Gather"),
        ("m2l_multiply", "Matrix multiply"),
        ("m2l_scatter", "Scatter"),
    ]
    assert all(column != "m2l_multiply" for column, _ in EVALUATION_PHASES)


def test_cuda_m2l_device_phases_are_nested_without_double_counting():
    row = {"execution_backend": "cuda-m2l"}
    top_level_columns = {
        column for column, _ in top_level_evaluation_phases(row)
    }
    nested_columns = {
        column for column, _ in nested_m2l_phases(row)
    }

    assert "m2l" in top_level_columns
    assert "cuda_h2d" not in top_level_columns
    assert "cuda_kernel" not in top_level_columns
    assert "cuda_d2h" not in top_level_columns
    assert nested_columns == {
        "cuda_h2d",
        "m2l_gather",
        "m2l_multiply",
        "m2l_scatter",
        "cuda_d2h",
    }
