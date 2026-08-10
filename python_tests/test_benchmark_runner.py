from benchmarks.run_benchmarks import (
    build_cases,
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
        {"suite": "size_order", "sources": "100", "order": "4"},
        {"suite": "size_order", "sources": "300", "order": "2"},
        {"suite": "size_order", "sources": "300", "order": "5"},
        {"suite": "scaling", "sources": "300", "order": "4"},
    ]

    assert representative_case(rows) == rows[2]


def test_case_plan_contains_size_order_depth_and_capped_scaling_suites():
    profile = {
        "sizes": [100, 300],
        "orders": [2, 4],
        "depths": [2, 3],
        "evaluations": 2,
        "samples": 2,
    }

    cases = build_cases(profile, max_threads=6)

    assert [case.suite for case in cases].count("size_order") == 4
    assert [case.depth for case in cases if case.suite == "depth"] == [2, 3]
    assert [case.threads for case in cases if case.suite == "scaling"] == [
        1,
        2,
        4,
        6,
    ]


def test_progress_and_phase_filename_identify_the_case():
    row = {
        "suite": "depth",
        "sources": "300",
        "order": "4",
        "depth": "2",
        "threads": "8",
    }

    assert progress_bar(2, 4) == "[============            ]"
    assert phase_breakdown_filename(row) == "depth_n300_p4_d2_t8.png"
