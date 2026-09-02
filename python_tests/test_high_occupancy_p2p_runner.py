from benchmarks.run_high_occupancy_p2p import build_cases, main


def test_high_occupancy_dry_run_lists_exact_matrix(capsys, tmp_path):
    assert main(["--dry-run", "--output-dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "planned_cases=16" in output
    for particles in (8192, 16384, 32768, 65536):
        for depth in (2, 3):
            assert f"cpu-static-matrix_n{particles}_d{depth}" in output
            assert f"cuda-full_n{particles}_d{depth}" in output
    assert output.count("--precision float32") == 16
    assert output.count("--exact-cuboid-p2p") == 16


def test_high_occupancy_case_matrix_has_independent_backends():
    cases = build_cases()
    assert len(cases) == 16
    assert {case.backend for case in cases} == {
        "cpu-static-matrix", "cuda-full"
    }
