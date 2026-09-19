"""Pin the Phase-3D regression driver's comparison, failure and resume rules.

These are pure-Python properties of the driver and the analyser: they need
neither the compiled extension nor a device, so they run in every
configuration. They exist because each one was a defect found in review.
"""

import csv
import json
import pathlib
import subprocess
import sys

import pytest

from benchmarks.analyse_phase3d_regression import (
    comparison,
    generated_comparisons,
    group_key,
)
from benchmarks.run_phase3d_regression import (
    COLUMNS,
    Case,
    CaseFailure,
    cases,
    manifest_differences,
    run_case,
    session_manifest,
)

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
BASELINE = REPOSITORY / "benchmarks/baselines/phase3d/phase3d_regression.csv"


def full_matrix() -> list[Case]:
    return list(cases("all", cuda=True, mkl=True))


def geometry_class(case: Case) -> str:
    """Whether a case places bodies on a lattice, irregularly, or freely."""
    if "--irregular-bodies" in case.arguments:
        return "irregular"
    if "--regular-grid" in case.arguments:
        return "regular"
    return "free"


def argument_value(case: Case, flag: str, default: str) -> str:
    if flag not in case.arguments:
        return default
    return case.arguments[case.arguments.index(flag) + 1]


# ---------------------------------------------------------------------------
# Comparison metadata
# ---------------------------------------------------------------------------

def test_every_case_carries_comparison_metadata():
    for case in full_matrix():
        assert case.comparison_group, case.name
        assert case.comparison_variant, case.name


def test_group_and_variant_identify_a_case_uniquely():
    """Two rows of one group must differ in the axis under review."""
    seen = {}
    for case in full_matrix():
        key = (case.comparison_group, case.comparison_variant)
        assert key not in seen, f"{case.name} collides with {seen.get(key)}"
        seen[key] = case.name


@pytest.mark.parametrize("flag,default,axis", [
    ("--precision", "float64", "precision"),
    ("--source-geometry", "point", "source geometry"),
    ("--target-geometry", "point", "target geometry"),
])
def test_a_group_never_mixes(flag, default, axis):
    """A ratio is meaningless across a change of precision or geometry."""
    groups = {}
    for case in full_matrix():
        groups.setdefault(case.comparison_group, set()).add(
            argument_value(case, flag, default))
    mixed = {name: values for name, values in groups.items()
             if len(values) > 1}
    assert not mixed, f"groups mixing {axis}: {mixed}"


def test_a_group_never_mixes_lattice_with_irregular_geometry():
    """The defect this metadata exists to prevent.

    Every P-cuda-finite/<kind>/{regular,irregular}/... case shares its first,
    second and last name token, so a rule built from those positions put an
    irregular cloud and a regular lattice in one group.
    """
    groups = {}
    for case in full_matrix():
        groups.setdefault(case.comparison_group, set()).add(
            geometry_class(case))
    mixed = {name: sorted(values) for name, values in groups.items()
             if len(values) > 1}
    assert not mixed, f"groups mixing geometry class: {mixed}"


def test_cuda_finite_regular_and_irregular_are_separate_groups():
    matrix = [c for c in full_matrix()
              if c.workload == "P-cuda-finite" and "/prism/" in c.name]
    regular = {c.comparison_group for c in matrix
               if geometry_class(c) == "regular"}
    irregular = {c.comparison_group for c in matrix
                 if geometry_class(c) == "irregular"}
    assert len(regular) == 1 and len(irregular) == 1
    assert regular.isdisjoint(irregular)


def test_comparison_columns_are_recorded():
    assert "comparison_group" in COLUMNS
    assert "comparison_variant" in COLUMNS


# ---------------------------------------------------------------------------
# The analyser reads metadata, and groups a legacy file the same way
# ---------------------------------------------------------------------------

def test_analyser_prefers_the_recorded_columns():
    row = {"case": "anything/at/all",
           "comparison_group": "recorded/group",
           "comparison_variant": "recorded-variant"}
    assert comparison(row) == ("recorded/group", "recorded-variant")


def test_analyser_groups_a_legacy_row_from_the_generators():
    """A CSV predating the columns is grouped by the same definition."""
    mapping = generated_comparisons()
    name = "P-cuda-finite/prism/irregular/auto/float32"
    assert name in mapping
    legacy = {"case": name, "comparison_group": "", "comparison_variant": ""}
    assert comparison(legacy) == mapping[name]


def test_analyser_isolates_an_unrecognised_case():
    """An unknown case is compared against nothing, never against a guess."""
    row = {"case": "brand/new/case/shape", "comparison_group": "",
           "comparison_variant": ""}
    group, variant = comparison(row)
    assert group == row["case"] and variant == row["case"]


@pytest.mark.skipif(not BASELINE.is_file(), reason="retained baseline absent")
def test_retained_baseline_groups_without_collapsing():
    with BASELINE.open() as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    for row in rows:
        if not row["workload"].startswith("P-"):
            continue
        groups.setdefault(group_key(row), set()).add(
            "irregular" if "/irregular/" in row["case"]
            else "regular" if "/regular/" in row["case"] else "other")
    mixed = {name: sorted(kinds) for name, kinds in groups.items()
             if len(kinds) > 1}
    assert not mixed, f"legacy rows still merged: {mixed}"


@pytest.mark.skipif(not BASELINE.is_file(), reason="retained baseline absent")
def test_retained_baseline_matches_the_generated_case_list():
    """The generators still define exactly the retained matrix."""
    with BASELINE.open() as stream:
        recorded = [row["case"] for row in csv.DictReader(stream)]
    assert [case.name for case in full_matrix()] == recorded


# ---------------------------------------------------------------------------
# A failed case fails the run
# ---------------------------------------------------------------------------

def failing_case() -> Case:
    return Case(name="unit/failing", workload="unit", arguments=[],
                comparison_group="unit", comparison_variant="failing")


def test_a_failed_invocation_raises(tmp_path):
    """It must not return a sentinel the caller can drop on the floor."""
    with pytest.raises(CaseFailure) as failure:
        run_case(sys.executable, failing_case(), evaluations=1, warmups=0,
                 samples=1, threads=1, scratch=tmp_path, resume=False)
    assert failure.value.name == "unit/failing"


def test_an_empty_result_file_raises(tmp_path):
    case = failing_case()
    output = tmp_path / (case.name.replace("/", "_") + ".csv")
    output.write_text("case,workload\n")  # header only, no data row
    with pytest.raises(CaseFailure):
        run_case(sys.executable, case, evaluations=1, warmups=0, samples=1,
                 threads=1, scratch=tmp_path, resume=True)


def test_driver_exits_non_zero_when_a_case_cannot_run(tmp_path):
    """The whole driver, not just run_case, must report the failure."""
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY / "benchmarks/run_phase3d_regression.py"),
         "--binary", sys.executable,
         "--output", str(tmp_path / "out.csv"),
         "--suite", "policy", "--no-cuda", "--no-mkl",
         "--filter", "P-cpu-random/M/auto/float32",
         "--evaluations", "1", "--warmups", "0", "--samples", "1"],
        capture_output=True, text=True, cwd=REPOSITORY)
    assert completed.returncode != 0
    assert "FAILED" in completed.stderr


def test_allow_failures_keeps_going_but_still_reports_incomplete(tmp_path):
    """The opt-in changes where the run stops, not whether it admits a gap.

    Skipping a case and then exiting zero would be the original defect in a
    new costume, so an incomplete matrix is still a non-zero exit.
    """
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY / "benchmarks/run_phase3d_regression.py"),
         "--binary", sys.executable,
         "--output", str(tmp_path / "out.csv"),
         "--suite", "policy", "--no-cuda", "--no-mkl",
         "--filter", "P-cpu-expansion/M/auto",
         "--evaluations", "1", "--warmups", "0", "--samples", "1",
         "--allow-failures"],
        capture_output=True, text=True, cwd=REPOSITORY)
    assert completed.returncode != 0
    assert "SKIPPED" in completed.stderr
    assert "INCOMPLETE" in completed.stderr
    # Both filtered cases were attempted rather than only the first.
    assert completed.stderr.count("SKIPPED") == 2


def test_a_missing_binary_is_reported(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY / "benchmarks/run_phase3d_regression.py"),
         "--binary", str(tmp_path / "absent"),
         "--output", str(tmp_path / "out.csv")],
        capture_output=True, text=True, cwd=REPOSITORY)
    assert completed.returncode != 0
    assert "not found" in completed.stderr


# ---------------------------------------------------------------------------
# Resume is guarded by the session manifest
# ---------------------------------------------------------------------------

class Arguments:
    """The subset of the parsed namespace the manifest reads."""

    def __init__(self, **overrides):
        self.suite = "all"
        self.filter = ""
        self.evaluations = 20
        self.warmups = 3
        self.samples = 5
        self.threads = 8
        self.no_cuda = False
        self.no_mkl = False
        self.__dict__.update(overrides)


def test_manifest_records_the_binary_contents(tmp_path):
    binary = tmp_path / "benchmark"
    binary.write_bytes(b"first build")
    first = session_manifest(binary, Arguments())
    binary.write_bytes(b"second build")
    second = session_manifest(binary, Arguments())
    assert first["binary"]["sha256"] != second["binary"]["sha256"]
    assert manifest_differences(first, second)


def test_manifest_records_the_sampling_settings(tmp_path):
    binary = tmp_path / "benchmark"
    binary.write_bytes(b"one build")
    first = session_manifest(binary, Arguments(samples=5))
    second = session_manifest(binary, Arguments(samples=9))
    differences = manifest_differences(first, second)
    assert any("samples" in difference for difference in differences)


def test_identical_conditions_compare_equal(tmp_path):
    binary = tmp_path / "benchmark"
    binary.write_bytes(b"one build")
    assert not manifest_differences(session_manifest(binary, Arguments()),
                                    session_manifest(binary, Arguments()))


def test_resume_is_refused_against_a_different_session(tmp_path):
    binary = tmp_path / "benchmark"
    binary.write_bytes(b"one build")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    stale = session_manifest(binary, Arguments(samples=99))
    (scratch / "session.json").write_text(json.dumps(stale))

    completed = subprocess.run(
        [sys.executable, str(REPOSITORY / "benchmarks/run_phase3d_regression.py"),
         "--binary", str(binary), "--output", str(tmp_path / "out.csv"),
         "--scratch", str(scratch), "--resume"],
        capture_output=True, text=True, cwd=REPOSITORY)
    assert completed.returncode != 0
    assert "--resume refused" in completed.stderr
    assert "samples" in completed.stderr


def test_resume_without_a_manifest_is_refused(tmp_path):
    binary = tmp_path / "benchmark"
    binary.write_bytes(b"one build")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    completed = subprocess.run(
        [sys.executable, str(REPOSITORY / "benchmarks/run_phase3d_regression.py"),
         "--binary", str(binary), "--output", str(tmp_path / "out.csv"),
         "--scratch", str(scratch), "--resume"],
        capture_output=True, text=True, cwd=REPOSITORY)
    assert completed.returncode != 0
    assert "does not exist" in completed.stderr


def test_a_fresh_run_ignores_an_existing_case_file(tmp_path):
    """Without --resume, an old per-case file must not be believed."""
    case = failing_case()
    output = tmp_path / (case.name.replace("/", "_") + ".csv")
    output.write_text("case,workload\nstale,stale\n")
    # `sys.executable` with no arguments reads stdin and exits non-zero here,
    # so a fresh run reaches the invocation rather than trusting the file.
    with pytest.raises(CaseFailure):
        run_case(sys.executable, case, evaluations=1, warmups=0, samples=1,
                 threads=1, scratch=tmp_path, resume=False)
