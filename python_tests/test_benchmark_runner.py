import pytest

from benchmarks.run_benchmarks import (
    CudaStatus,
    EVALUATION_PHASES,
    benchmark_backends,
    build_cases,
    comparison_case,
    expanded_cases,
    phase_breakdown_filename,
    progress_bar,
    representative_case,
    thread_counts_up_to,
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


def test_comparison_case_uses_largest_direct_enabled_case():
    rows = [
        {
            "suite": "comparison",
            "sources": "100",
            "order": "4",
            "p2p_1_total_median": "0.2",
        },
        {
            "suite": "comparison",
            "sources": "300",
            "order": "2",
            "p2p_1_total_median": "0.8",
        },
        {
            "suite": "comparison",
            "sources": "300",
            "order": "5",
            "p2p_1_total_median": "0.8",
        },
        {
            "suite": "comparison",
            "sources": "1000",
            "order": "4",
            "p2p_1_total_median": "nan",
        },
    ]

    assert comparison_case(rows) == rows[2]


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
        "parameter_grid_n300_p4_d2_t8_cpu-static.png"
    )


def test_cuda_build_runs_every_case_with_static_and_cuda_backends():
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
        CudaStatus(compiled=True, available=True, device="Test GPU")
    )

    assert backends == ["cpu-static", "cuda-m2l", "cuda-full"]
    expanded = expanded_cases(cases, backends)
    for case in cases:
        selected = {
            (candidate, backend)
            for candidate, backend in expanded
            if candidate == case
        }
        assert selected == {
            (case, "cpu-static"),
            (case, "cuda-m2l"),
            (case, "cuda-full"),
        }


def test_cpu_build_runs_only_static_benchmarks():
    status = CudaStatus(compiled=False, available=False, device="")
    assert benchmark_backends(status) == ["cpu-static"]


def test_cuda_benchmark_fails_clearly_without_a_device():
    status = CudaStatus(compiled=True, available=False, device="")
    with pytest.raises(RuntimeError, match="no CUDA device"):
        benchmark_backends(status)


def test_phase_plots_include_cuda_transfer_and_kernel_timings():
    assert EVALUATION_PHASES[-3:] == [
        ("cuda_h2d", "CUDA H2D"),
        ("cuda_kernel", "CUDA kernel"),
        ("cuda_d2h", "CUDA D2H"),
    ]
