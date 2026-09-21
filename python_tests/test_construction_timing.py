"""Construction-time clocks follow the timing level, and the benchmark driver
keeps its external clock apart from the solver's internal timings.

The C++ tests pin the library contract; these cover the two surfaces the C++
tests cannot: the Python option on ``UniformTreeOptions`` and the CSV a
``benchmark_uniform_fmm`` binary writes.  The benchmark cases need a built
driver and skip, saying so, when none is found.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import cdfmm

REPOSITORY_ROOT = Path(__file__).parents[1]


def _positions(count: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.uniform(-1.0, 1.0, size=(count, 3))


def test_uniform_tree_options_expose_the_build_timing_switch():
    options = cdfmm.UniformTreeOptions()
    # The standalone default keeps the historical behaviour.
    assert options.collect_build_timings is True
    options.max_level = 2
    positions = _positions(300, 11)

    timed = cdfmm.UniformTree(positions, options)
    options.collect_build_timings = False
    silent = cdfmm.UniformTree(positions, options)

    # The switch changes nothing the tree is for.
    assert timed.max_level == silent.max_level
    assert timed.root_half_width == silent.root_half_width
    assert np.array_equal(np.asarray(timed.source_permutation),
                          np.asarray(silent.source_permutation))
    assert np.array_equal(np.asarray(timed.source_inverse_permutation),
                          np.asarray(silent.source_inverse_permutation))
    assert len(timed.nodes) == len(silent.nodes)


def test_uniform_fmm_off_leaves_tree_construction_uncollected():
    positions = _positions(200, 5)
    options = cdfmm.UniformFmmOptions()
    options.expansion_order = 3
    options.tree.max_level = 2
    options.enable_cache = False

    off = cdfmm.UniformFmm(positions, options)
    statistics = off.static_plan_statistics
    assert statistics["timing_level"] == cdfmm.TimingLevel.OFF
    assert statistics["tree_construction_seconds"] == 0.0
    assert off.tree.max_level == 2

    options.timing_level = cdfmm.TimingLevel.DETAILED
    detailed = cdfmm.UniformFmm(positions, options)
    assert detailed.static_plan_statistics["tree_construction_seconds"] > 0.0


# ---------------------------------------------------------------------------
# benchmark_uniform_fmm: external headline clock versus internal phase columns
# ---------------------------------------------------------------------------

INTERNAL_PHASE_COLUMNS = (
    "tree_total", "root_bounds", "node_construction", "topology",
    "source_morton", "source_sorting", "target_morton", "target_sorting",
    "ranges", "interaction_lists", "moment_permutation", "multipole_reset",
    "p2m", "m2m", "local_reset", "l2l", "m2l", "m2l_scale", "m2l_gather",
    "m2l_multiply", "m2l_scatter", "l2p", "p2p", "result_unpermutation",
    "cuda_h2d", "cuda_kernel", "cuda_d2h", "cuda_m2l_h2d", "cuda_m2l_d2h",
    "cuda_p2p_h2d", "cuda_p2p_kernel", "cuda_p2p_d2h", "cuda_p2p_wait",
    "static_plan_seconds", "p2m_plan_seconds", "m2m_plan_seconds",
    "m2l_plan_seconds", "l2l_plan_seconds", "l2p_plan_seconds",
    "p2p_tensor_plan_seconds",
)


def _find_benchmark() -> Path | None:
    """The driver named by ``CDFMM_BENCHMARK_UNIFORM_FMM``, else the newest
    one under a ``build*`` directory of this checkout."""
    named = os.environ.get("CDFMM_BENCHMARK_UNIFORM_FMM")
    if named:
        candidate = Path(named)
        return candidate if candidate.is_file() else None
    candidates = [
        path for path in REPOSITORY_ROOT.glob("build*/benchmarks/benchmark_uniform_fmm")
        if path.is_file() and os.access(path, os.X_OK)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _run_benchmark(binary: Path, output: Path, *arguments: str) -> dict[str, str]:
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "2")
    command = [
        str(binary), "--sources", "300", "--targets", "300",
        "--evaluations", "2", "--samples", "2", "--warmups", "1",
        "--no-direct", "--no-workload-comparison", "--output", str(output),
        *arguments,
    ]
    subprocess.run(command, check=True, env=environment,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    rows = list(csv.DictReader(output.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 1, rows
    return rows[0]


@pytest.fixture(scope="module")
def benchmark_binary() -> Path:
    binary = _find_benchmark()
    if binary is None:
        pytest.skip("no built benchmark_uniform_fmm found (set "
                    "CDFMM_BENCHMARK_UNIFORM_FMM or build the benchmark preset)")
    return binary


def test_direct_reference_rows_carry_no_internal_timing(tmp_path, benchmark_binary):
    """The CPU direct reference has no internal collector, so at every level its
    phase columns stay zero and the headline comes from the external clock."""
    for level in ("off", "detailed"):
        row = _run_benchmark(benchmark_binary, tmp_path / f"direct-{level}.csv",
                             "--backend", "cpu-direct", "--timing", level)
        assert row["execution_backend"] == "cpu-direct"
        assert row["timing_level"] == level
        assert row["internal_timing_source"] == "none"
        assert float(row["evaluation_median"]) > 0.0
        for column in INTERNAL_PHASE_COLUMNS:
            assert float(row[column]) == 0.0, (level, column, row[column])


def test_fmm_rows_obey_the_timing_level(tmp_path, benchmark_binary):
    """At Off the solver's phase columns, including the tree breakdown, are
    uncollected zeros while the external headline columns are measured; at
    Detailed the same columns are populated."""
    common = ("--backend", "cpu-static-matrix", "--depth", "2", "--order", "3")
    off = _run_benchmark(benchmark_binary, tmp_path / "fmm-off.csv",
                         *common, "--timing", "off")
    assert off["internal_timing_source"] == "uniform_fmm"
    assert off["timing_level"] == "off"
    assert float(off["evaluation_median"]) > 0.0
    assert float(off["fmm_setup_seconds"]) > 0.0
    for column in INTERNAL_PHASE_COLUMNS:
        assert float(off[column]) == 0.0, (column, off[column])

    detailed = _run_benchmark(benchmark_binary, tmp_path / "fmm-detailed.csv",
                              *common, "--timing", "detailed")
    assert detailed["internal_timing_source"] == "uniform_fmm"
    assert float(detailed["tree_total"]) > 0.0
    assert float(detailed["interaction_lists"]) > 0.0
    assert float(detailed["p2m"]) > 0.0
    assert float(detailed["p2p"]) > 0.0
    assert float(detailed["static_plan_seconds"]) > 0.0


def test_benchmark_binary_lookup_prefers_the_named_driver(tmp_path, monkeypatch):
    named = tmp_path / "benchmark_uniform_fmm"
    named.write_bytes(b"not run")
    monkeypatch.setenv("CDFMM_BENCHMARK_UNIFORM_FMM", str(named))
    assert _find_benchmark() == named
    monkeypatch.setenv("CDFMM_BENCHMARK_UNIFORM_FMM", str(tmp_path / "missing"))
    assert _find_benchmark() is None


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
