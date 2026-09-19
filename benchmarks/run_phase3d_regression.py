#!/usr/bin/env python3
"""Internal Phase-3D cross-backend regression and policy-validation matrix.

ENGINEERING REGRESSION BASELINE -- NOT AN ARTICLE1 PUBLICATION BENCHMARK.

This driver exists to answer three engineering questions at the end of the
v0.2 optimisation phases, on one machine in one session:

  1. does each backend's automatic policy still resolve to the representation
     the Phase-3 measurements chose for it;
  2. does the combined implementation show an integration regression in
     construction, warm loading, evaluation or memory; and
  3. is any remaining bottleneck large enough in absolute time to be worth a
     last targeted fix.

It deliberately does not attempt publication coverage: no external framework
comparison, no polished figures, no large sweeps, and no accuracy campaign.
The numbers it produces must not be copied into the FMM article; the
publication campaign runs after Phase-4 pruning against a frozen
implementation.

Every case is one `benchmark_uniform_fmm` invocation on a fixed geometry with
repeated moment updates, so construction and repeated evaluation are recorded
separately. A run is fresh by default: every case is executed, and any case
that cannot run fails the whole matrix with a non-zero exit, because a
baseline that silently drops rows cannot be told apart from a complete one.
`--resume` reuses the per-case files of an interrupted run, but only after the
scratch manifest shows the same revision, the same binary contents and the
same sampling settings, so "same-session, same-binary" is a checked property
rather than a convention.

Each row also records `comparison_group` and `comparison_variant`: the group
is everything two rows must share before a ratio between them means anything,
and the variant is the single axis under review. They are written down rather
than parsed back out of the display name, whose token positions shift whenever
a case family gains or loses a dimension.

WARNING: this driver leaves the persistent caches enabled, because the policy
question it answers is about repeated evaluation. `fmm_setup_seconds` is
therefore a *warm* number whose value depends on what ran before it -- the
first row of a given order and precision pays the universal operator build and
caches it for every later row. Use it only to catch a gross setup regression,
never as a construction measurement. Cold construction is
`run_construction_matrix.py`, which disables the cache for every row, and the
cold/warm split is `run_phase3d_startup.py`.

Usage:
    python benchmarks/run_phase3d_regression.py \
        --binary build-bench-all/benchmarks/benchmark_uniform_fmm \
        --output phase3d_regression.csv --suite all
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# The columns the Phase-3D review reads. They cover the resolved policy (so a
# stale rule is visible without reading the source), repeated evaluation and
# its phases (the number a policy comparison turns on), construction (so a
# setup regression cannot hide behind a good evaluation), and the persistent
# host and device footprints.
COLUMNS = [
    # identity of the row
    "case", "workload", "comparison_group", "comparison_variant",
    "backend", "precision", "sources", "depth", "order",
    "source_geometry", "target_geometry", "irregular_bodies", "regular_grid",
    "periodic", "layout_hint", "far_field_model",
    # what the policy resolved to
    "p2p_packing_requested", "p2p_packing",
    "point_expansion_requested", "p2m_execution", "l2p_execution",
    "static_multiply_backend", "m2l_strategy",
    # repeated evaluation and its phases
    "evaluation_median", "evaluation_mean", "p2p", "m2l", "p2m", "m2m", "l2l",
    "l2p", "cuda_p2p_kernel", "cuda_p2p_wait", "cuda_kernel",
    # construction
    "fmm_setup_seconds", "static_plan_seconds", "p2p_tensor_plan_seconds",
    "p2m_plan_seconds", "l2p_plan_seconds", "m2l_plan_seconds",
    # memory
    "static_plan_bytes", "near_field_operator_bytes",
    "p2m_operator_bytes", "l2p_operator_bytes",
    "p2p_dictionary_total_bytes", "p2p_canonical_total_bytes",
    "p2p_dictionary_token_width_bytes", "p2p_unique_tensors",
    "cuda_persistent_device_bytes", "cuda_p2p_tensor_bytes",
    # problem shape and a coarse correctness signal
    "near_field_pairs", "occupied_target_leaves", "max_relative_error",
]


class Case(NamedTuple):
    """One benchmark invocation and how it may be compared.

    `comparison_group` is the machine-readable identity that two rows must
    share before a ratio between them means anything: the workload class, the
    geometry and its size, whether the bodies sit on a lattice or are
    irregular, and the precision. `comparison_variant` is the one thing that
    differs inside the group -- the forced alternative under review.

    These are recorded rather than parsed back out of `name`. A display name
    is written for a reader, so its token positions shift whenever a case
    family gains or loses a dimension, and a grouping rule that counts those
    positions silently compares a regular lattice against an irregular cloud.
    """

    name: str
    workload: str
    arguments: list[str]
    comparison_group: str
    comparison_variant: str


class CaseFailure(Exception):
    """A benchmark invocation that did not produce a usable row."""

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"{name}: {reason}")
        self.name = name
        self.reason = reason


def run_case(binary: str, case: Case, evaluations: int, warmups: int,
             samples: int, threads: int, scratch: Path,
             resume: bool) -> dict:
    """Run one case, or reuse its cached CSV, and return the collected row.

    Raises `CaseFailure` when the invocation fails or yields no row. A
    regression baseline that quietly drops a case is worse than no baseline:
    the missing row is indistinguishable from a case that was never defined,
    so nothing downstream can tell a clean matrix from a truncated one.
    """
    output = scratch / (case.name.replace("/", "_") + ".csv")
    if resume and output.exists() and output.stat().st_size > 0:
        print("=", case.name, "(reusing)", file=sys.stderr, flush=True)
    else:
        command = [
            binary, *case.arguments,
            "--evaluations", str(evaluations),
            "--warmups", str(warmups),
            "--samples", str(samples),
            "--threads", str(threads),
            "--no-direct",
            "--no-workload-comparison",
            "--output", str(output),
        ]
        print(">", case.name, file=sys.stderr, flush=True)
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            output.unlink(missing_ok=True)
            tail = completed.stderr.strip().splitlines()[-1:]
            detail = tail[0] if tail else "(no stderr output)"
            raise CaseFailure(case.name,
                              f"exit {completed.returncode}: {detail}")

    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise CaseFailure(case.name, f"{output} holds no data row")

    row = dict(rows[-1])
    row["case"] = case.name
    row["workload"] = case.workload
    row["comparison_group"] = case.comparison_group
    row["comparison_variant"] = case.comparison_variant
    row["p2p_packing_requested"] = row.get("p2p_packing_requested", "")
    row["regular_grid"] = "1" if "--regular-grid" in case.arguments else "0"
    row["layout_hint"] = ("regular_grid"
                          if "--spatial-layout" in case.arguments
                          else "general")
    return {column: row.get(column, "") for column in COLUMNS}


# ---------------------------------------------------------------------------
# Session identity: what a resumed run must match before it may reuse a file.
# ---------------------------------------------------------------------------

MANIFEST_NAME = "session.json"


def file_fingerprint(path: Path) -> dict:
    """Content hash of the benchmark binary, plus supplemental stat metadata.

    The hash is what decides reuse; size and mtime are recorded beside it so a
    mismatch report says something a reader can act on.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    status = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size": status.st_size,
        "mtime_ns": status.st_mtime_ns,
    }


def head_revision() -> str:
    """The checkout's HEAD revision, or "" where it cannot be determined."""
    command = ["git", "-C", str(Path(__file__).resolve().parent),
               "rev-parse", "HEAD"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   check=False)
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def session_manifest(binary: Path, arguments: argparse.Namespace) -> dict:
    """Everything that must agree before two rows belong to one session.

    A per-case CSV records a number, not the conditions that produced it. Two
    such files sitting in one directory therefore look like one matrix even
    when they came from different commits, different binaries or different
    sample counts, so the conditions are written down once and checked before
    any file is reused.
    """
    return {
        "schema": 1,
        "revision": head_revision(),
        "binary": file_fingerprint(binary),
        "suite": arguments.suite,
        "filter": arguments.filter,
        "evaluations": arguments.evaluations,
        "warmups": arguments.warmups,
        "samples": arguments.samples,
        "threads": arguments.threads,
        "cuda": not arguments.no_cuda,
        "mkl": not arguments.no_mkl,
        "columns": COLUMNS,
    }


def manifest_differences(recorded: dict, current: dict) -> list[str]:
    """Human-readable reasons a recorded session is not the current one."""
    differences = []
    for key in sorted(set(recorded) | set(current)):
        if recorded.get(key) != current.get(key):
            differences.append(f"{key}: recorded {recorded.get(key)!r} "
                               f"but this run has {current.get(key)!r}")
    return differences


def geometry_arguments(kind: str) -> list[str]:
    if kind == "point":
        return []
    if kind in ("prism", "tetrahedron"):
        return ["--source-geometry", kind, "--target-geometry", kind]
    raise ValueError(kind)


# `--regular-grid` generates the lattice coordinates; `--spatial-layout` is the
# separate public hint that lets the policy select the signed dictionary. A
# lattice run without the hint is what a caller gets by default, so both are
# measured wherever the lattice rule is the thing under review.
def layout_arguments(layout: str) -> list[str]:
    if layout == "general":
        return []
    if layout == "hint":
        return ["--spatial-layout", "regular-grid"]
    raise ValueError(layout)


# ---------------------------------------------------------------------------
# Phase B: the workload classes the regression matrix must cover.
# ---------------------------------------------------------------------------

def regression_cases(cuda: bool, mkl: bool):
    """Workload classes A-F under the automatic policy on every backend."""
    backends = ["cpu-static-matrix"]
    if mkl:
        backends.append("cpu-static-matrix-mkl")
    if cuda:
        backends += ["cuda-full", "cuda-partial"]

    # A. Random points, small and medium. The medium p=6 case is required.
    for label, size, depth, order in (("S", 10000, 3, 4), ("M", 50000, 4, 6)):
        for backend in backends:
            for precision in ("float32", "float64"):
                yield Case(
                    name=f"A-random/{label}/{backend}/{precision}",
                    workload="A-random",
                    arguments=["--backend", backend, "--sources", str(size),
                               "--targets", str(size), "--depth", str(depth),
                               "--order", str(order),
                               "--precision", precision],
                    comparison_group=f"A-random/{label}/{precision}",
                    comparison_variant=backend)

    # B. Regular point lattice, low occupancy (32^3 at depth 4 is 8 per leaf),
    #    where dictionary compression is strongest.
    # C. The same lattice at depth 3 is 64 per leaf, which is the high
    #    occupancy regime the three-regime dictionary rule was calibrated on.
    for workload, depth, occupancy in (("B-lattice-low", 4, 8),
                                       ("C-lattice-high", 3, 64)):
        for backend in backends:
            for precision in ("float32", "float64"):
                for layout in ("general", "hint"):
                    yield Case(
                        name=(f"{workload}/occ{occupancy}/{backend}/{layout}/"
                              f"{precision}"),
                        workload=workload,
                        arguments=["--backend", backend, "--sources", "32768",
                                   "--targets", "32768", "--depth", str(depth),
                                   "--order", "6", "--regular-grid",
                                   *layout_arguments(layout),
                                   "--precision", precision],
                        comparison_group=(f"{workload}/regular/"
                                          f"occ{occupancy}/{precision}"),
                        comparison_variant=f"{backend}/{layout}")

    # D/E. Regular finite prism and tetrahedron lattices with an exact finite
    #      near field at the reference size N ~ 4096, p = 6.
    for kind in ("prism", "tetrahedron"):
        workload = "D-prism" if kind == "prism" else "E-tetrahedron"
        for backend in backends:
            for precision in ("float32", "float64"):
                for layout in ("general", "hint"):
                    yield Case(
                        name=(f"{workload}/regular/N4096/{backend}/{layout}/"
                              f"{precision}"),
                        workload=workload,
                        arguments=["--backend", backend, "--sources", "4096",
                                   "--targets", "4096", "--depth", "3",
                                   "--order", "6", "--regular-grid",
                                   *geometry_arguments(kind),
                                   *layout_arguments(layout),
                                   "--precision", precision],
                        comparison_group=(f"{workload}/regular/N4096/"
                                          f"point-far-field/{precision}"),
                        comparison_variant=f"{backend}/{layout}")
        # One exact far-field endpoint measurement per family, so the finite
        # P2M/L2P operators are actually built and their cost is recorded.
        yield Case(
            name=(f"{workload}/regular/N4096/exact-far-field/"
                  f"cpu-static-matrix"),
            workload=workload,
            arguments=["--backend", "cpu-static-matrix", "--sources", "4096",
                       "--targets", "4096", "--depth", "3", "--order", "6",
                       "--regular-grid", *geometry_arguments(kind),
                       "--far-field-model", "exact",
                       "--precision", "float32"],
            comparison_group=(f"{workload}/regular/N4096/"
                              f"exact-far-field/float32"),
            comparison_variant="cpu-static-matrix")

    # F. Irregular finite control: low exact tensor reuse, so the general
    #    layout fallback and the construction gate are both exercised.
    for kind in ("prism", "tetrahedron"):
        for backend in backends:
            yield Case(
                name=f"F-irregular/{kind}/N4096/{backend}/float32",
                workload="F-irregular",
                arguments=["--backend", backend, "--sources", "4096",
                           "--targets", "4096", "--depth", "3", "--order", "6",
                           "--irregular-bodies", *geometry_arguments(kind),
                           "--precision", "float32"],
                comparison_group=f"F-irregular/{kind}/irregular/N4096/float32",
                comparison_variant=backend)


# ---------------------------------------------------------------------------
# Phase C: the automatic policy against a small set of credible alternatives.
# ---------------------------------------------------------------------------

def policy_cases(cuda: bool, mkl: bool):
    """Auto against the forced alternatives that could plausibly beat it."""
    # Random CPU points: the position-based executor against the stored rows.
    for packing in ("auto", "point-geometry", "particle-row-soa"):
        for precision in ("float32", "float64"):
            yield Case(
                name=f"P-cpu-random/M/{packing}/{precision}",
                workload="P-cpu-random",
                arguments=["--backend", "cpu-static-matrix",
                           "--sources", "50000", "--targets", "50000",
                           "--depth", "4", "--order", "6",
                           "--p2p-packing", packing,
                           "--precision", precision],
                comparison_group=f"P-cpu-random/M/{precision}",
                comparison_variant=packing)

    # Regular CPU points: the lattice rule against the position-based kernel.
    # `auto` is measured under both layouts, because only the hint can select
    # the dictionary; the forced packings need no hint to be reachable.
    for packing, layout in (("auto", "general"), ("auto", "hint"),
                            ("point-geometry", "general"),
                            ("tensor-dictionary", "general")):
        for precision in ("float32", "float64"):
            yield Case(
                name=f"P-cpu-lattice/occ8/{packing}/{layout}/{precision}",
                workload="P-cpu-lattice",
                arguments=["--backend", "cpu-static-matrix",
                           "--sources", "32768", "--targets", "32768",
                           "--depth", "4", "--order", "6", "--regular-grid",
                           *layout_arguments(layout),
                           "--p2p-packing", packing,
                           "--precision", precision],
                comparison_group=f"P-cpu-lattice/regular/occ8/{precision}",
                comparison_variant=f"{packing}/{layout}")

    # CPU point expansions: procedural against precomputed P2M/L2P.
    for expansion in ("auto", "procedural", "precomputed"):
        for precision in ("float32", "float64"):
            yield Case(
                name=f"P-cpu-expansion/M/{expansion}/{precision}",
                workload="P-cpu-expansion",
                arguments=["--backend", "cpu-static-matrix",
                           "--sources", "50000", "--targets", "50000",
                           "--depth", "4", "--order", "6",
                           "--point-expansion", expansion,
                           "--precision", precision],
                comparison_group=f"P-cpu-expansion/M/{precision}",
                comparison_variant=expansion)

    # CPU far field: portable against oneMKL on a point and a finite case.
    if mkl:
        for backend in ("cpu-static-matrix", "cpu-static-matrix-mkl"):
            for precision in ("float32", "float64"):
                yield Case(
                    name=f"P-cpu-m2l/point-M/{backend}/{precision}",
                    workload="P-cpu-m2l",
                    arguments=["--backend", backend, "--sources", "50000",
                               "--targets", "50000", "--depth", "4",
                               "--order", "6", "--precision", precision],
                    comparison_group=f"P-cpu-m2l/point-M/{precision}",
                    comparison_variant=backend)
                yield Case(
                    name=f"P-cpu-m2l/prism-N4096/{backend}/{precision}",
                    workload="P-cpu-m2l",
                    arguments=["--backend", backend, "--sources", "4096",
                               "--targets", "4096", "--depth", "3",
                               "--order", "6", "--regular-grid",
                               *geometry_arguments("prism"),
                               "--precision", precision],
                    comparison_group=(f"P-cpu-m2l/prism/regular/N4096/"
                                      f"{precision}"),
                    comparison_variant=backend)

    if not cuda:
        return

    # CUDA points, both precisions: the precision-gated rule is the single
    # most important policy to re-verify, so each side is forced explicitly.
    for precision in ("float32", "float64"):
        for packing in ("auto", "point-geometry", "leaf-block"):
            yield Case(
                name=f"P-cuda-random/M/{packing}/{precision}",
                workload="P-cuda-random",
                arguments=["--backend", "cuda-full", "--sources", "50000",
                           "--targets", "50000", "--depth", "4",
                           "--order", "6",
                           "--p2p-packing", packing,
                           "--precision", precision],
                comparison_group=f"P-cuda-random/M/{precision}",
                comparison_variant=packing)
        # The same question on a lattice, where the dictionary is the rival.
        # The FP32 point rule deliberately precedes the layout hint, so the
        # hinted `auto` row is what shows whether that is still true.
        for packing, layout in (("auto", "general"), ("auto", "hint"),
                                ("point-geometry", "general"),
                                ("tensor-dictionary", "general"),
                                ("leaf-block", "general")):
            yield Case(
                name=f"P-cuda-lattice/occ8/{packing}/{layout}/{precision}",
                workload="P-cuda-lattice",
                arguments=["--backend", "cuda-full", "--sources", "32768",
                           "--targets", "32768", "--depth", "4",
                           "--order", "6",
                           "--regular-grid", *layout_arguments(layout),
                           "--p2p-packing", packing,
                           "--precision", precision],
                comparison_group=f"P-cuda-lattice/regular/occ8/{precision}",
                comparison_variant=f"{packing}/{layout}")

    # CUDA point expansions: procedural is the FP32 default and precomputed
    # the FP64 default, so both sides are forced in both precisions.
    for expansion in ("auto", "procedural", "precomputed"):
        for precision in ("float32", "float64"):
            yield Case(
                name=f"P-cuda-expansion/M/{expansion}/{precision}",
                workload="P-cuda-expansion",
                arguments=["--backend", "cuda-full", "--sources", "50000",
                           "--targets", "50000", "--depth", "4",
                           "--order", "6",
                           "--point-expansion", expansion,
                           "--precision", precision],
                comparison_group=f"P-cuda-expansion/M/{precision}",
                comparison_variant=expansion)

    # CUDA finite geometry: the dictionary against leaf blocks on a lattice,
    # and leaf blocks on the general layout.
    for kind in ("prism", "tetrahedron"):
        for packing, layout in (("auto", "general"), ("auto", "hint"),
                                ("tensor-dictionary", "general"),
                                ("leaf-block", "general")):
            yield Case(
                name=(f"P-cuda-finite/{kind}/regular/{packing}/{layout}/"
                      f"float32"),
                workload="P-cuda-finite",
                arguments=["--backend", "cuda-full", "--sources", "4096",
                           "--targets", "4096", "--depth", "3", "--order", "6",
                           "--regular-grid", *geometry_arguments(kind),
                           *layout_arguments(layout),
                           "--p2p-packing", packing,
                           "--precision", "float32"],
                comparison_group=f"P-cuda-finite/{kind}/regular/float32",
                comparison_variant=f"{packing}/{layout}")
        for packing in ("auto", "leaf-block"):
            yield Case(
                name=f"P-cuda-finite/{kind}/irregular/{packing}/float32",
                workload="P-cuda-finite",
                arguments=["--backend", "cuda-full", "--sources", "4096",
                           "--targets", "4096", "--depth", "3", "--order", "6",
                           "--irregular-bodies", *geometry_arguments(kind),
                           "--p2p-packing", packing,
                           "--precision", "float32"],
                comparison_group=f"P-cuda-finite/{kind}/irregular/float32",
                comparison_variant=packing)

    # CudaFull against CudaPartial on the regimes the crossover was last seen
    # in: medium random points and a high-occupancy near field.
    for backend in ("cuda-full", "cuda-partial"):
        for precision in ("float32", "float64"):
            yield Case(
                name=f"P-cuda-backend/M/{backend}/{precision}",
                workload="P-cuda-backend",
                arguments=["--backend", backend, "--sources", "50000",
                           "--targets", "50000", "--depth", "4",
                           "--order", "6",
                           "--precision", precision],
                comparison_group=f"P-cuda-backend/M/{precision}",
                comparison_variant=backend)
        yield Case(
            name=f"P-cuda-backend/occ128/{backend}/float32",
            workload="P-cuda-backend",
            arguments=["--backend", backend, "--sources", "65536",
                       "--targets", "65536", "--depth", "3", "--order", "6",
                       "--precision", "float32"],
            comparison_group="P-cuda-backend/occ128/float32",
            comparison_variant=backend)


def cases(suite: str, cuda: bool, mkl: bool):
    if suite in ("regression", "all"):
        yield from regression_cases(cuda, mkl)
    if suite in ("policy", "all"):
        yield from policy_cases(cuda, mkl)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", default="all",
                        choices=("regression", "policy", "all"))
    parser.add_argument("--evaluations", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--filter", default="",
                        help="substring of case names to run")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--no-mkl", action="store_true")
    parser.add_argument("--scratch", default="",
                        help="directory of per-case CSVs (default: alongside "
                             "the output file)")
    parser.add_argument("--resume", action="store_true",
                        help="reuse per-case CSVs from an interrupted run of "
                             "the same session; refused unless the scratch "
                             "manifest matches this run exactly")
    parser.add_argument("--allow-failures", action="store_true",
                        help="continue past a case that cannot run and write "
                             "an incomplete matrix; exploratory only, never "
                             "for a retained baseline")
    arguments = parser.parse_args()

    binary = Path(arguments.binary)
    if not binary.is_file():
        print(f"benchmark binary not found: {binary}", file=sys.stderr)
        return 2

    output = Path(arguments.output)
    scratch = (Path(arguments.scratch) if arguments.scratch
               else output.parent / (output.stem + "_cases"))
    scratch.mkdir(parents=True, exist_ok=True)

    # Reuse is a claim that an existing file was produced under exactly these
    # conditions. The manifest is what makes that claim checkable; without a
    # matching one, the previous files are from an unknown session and every
    # case is re-run.
    manifest_path = scratch / MANIFEST_NAME
    current = session_manifest(binary, arguments)
    resume = arguments.resume
    if resume:
        if not manifest_path.is_file():
            print(f"--resume: {manifest_path} does not exist, so the existing "
                  f"per-case files cannot be shown to come from this session",
                  file=sys.stderr)
            return 2
        with manifest_path.open() as stream:
            recorded = json.load(stream)
        differences = manifest_differences(recorded, current)
        if differences:
            print("--resume refused: the scratch directory holds a different "
                  "session", file=sys.stderr)
            for difference in differences:
                print(f"  {difference}", file=sys.stderr)
            print("  rerun without --resume to start a fresh matrix",
                  file=sys.stderr)
            return 2
    with manifest_path.open("w") as stream:
        json.dump(current, stream, indent=2)

    selected = [case for case in cases(arguments.suite,
                                       not arguments.no_cuda,
                                       not arguments.no_mkl)
                if not arguments.filter or arguments.filter in case.name]
    expected = len(selected)

    rows = []
    failures: list[CaseFailure] = []
    for case in selected:
        try:
            rows.append(run_case(arguments.binary, case, arguments.evaluations,
                                 arguments.warmups, arguments.samples,
                                 arguments.threads, scratch, resume))
        except CaseFailure as failure:
            if not arguments.allow_failures:
                print(f"FAILED: {failure}", file=sys.stderr)
                print(f"the matrix is incomplete ({len(rows)} of {expected} "
                      f"rows); pass --allow-failures only for exploration, "
                      f"never for a retained baseline", file=sys.stderr)
                return 1
            print(f"  SKIPPED: {failure}", file=sys.stderr)
            failures.append(failure)

    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    complete = len(rows) == expected
    summary = output.with_suffix(".json")
    with summary.open("w") as stream:
        json.dump({
            "kind": "engineering regression baseline",
            "not_for_publication": True,
            "note": "Phase-3D internal matrix; not an Article1 benchmark.",
            "suite": arguments.suite,
            "threads": arguments.threads,
            "evaluations": arguments.evaluations,
            "samples": arguments.samples,
            "cases": len(rows),
            "expected_cases": expected,
            "complete": complete,
            "resumed": resume,
            "skipped": [failure.name for failure in failures],
            "session": current,
        }, stream, indent=2)

    print(f"wrote {len(rows)} of {expected} rows to {output}", file=sys.stderr)
    if not complete:
        print(f"INCOMPLETE: {len(failures)} case(s) did not run:",
              file=sys.stderr)
        for failure in failures:
            print(f"  {failure.name}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
