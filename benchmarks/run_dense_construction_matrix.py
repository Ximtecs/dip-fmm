#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dense all-to-all construction and evaluation matrix.

This driver sweeps `benchmark_dense_direct_construction` over the geometry
combinations, workloads, sizes, precisions and backends the exact dense plan
supports, and appends one CSV row per measured configuration.

One process runs one row, for two reasons: the peak resident set a row reports
then belongs to that row alone, and a row that exhausts memory cannot take the
rest of the sweep with it.  Rows already present in the CSV are skipped, so an
interrupted sweep resumes and a known-bad row can be left out by hand.

Dense storage is `6 * Ns * Nt * scalar_bytes`, which reaches 1.6 GB at
Ns = Nt = 4096 in FP64, so the presets grow the size only where the geometry
is cheap enough for the result to be worth the memory.  `--max-matrix-bytes`
refuses a row whose matrices would exceed a budget.

Examples
--------
    python benchmarks/run_dense_construction_matrix.py \
        --binary build-bench-all/benchmarks/benchmark_dense_direct_construction \
        --output dense-baseline.csv --preset geometry-matrix

    python benchmarks/run_dense_construction_matrix.py \
        --binary build-bench-all/benchmarks/benchmark_dense_direct_construction \
        --output dense-threads.csv --preset threads
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import os
import pathlib
import subprocess
import sys

GEOMETRIES: tuple[str, ...] = ("point", "prism", "tetrahedron")
WORKLOADS: tuple[str, ...] = ("lattice", "lattice-irregular", "random")

# The nine source -> target combinations the dense plan supports.  They are
# enumerated rather than inferred from one another, because the production
# dispatch reaches a different function for several of them.
GEOMETRY_PAIRS: tuple[tuple[str, str], ...] = tuple(
    itertools.product(GEOMETRIES, GEOMETRIES)
)


@dataclasses.dataclass(frozen=True)
class Row:
    """One measured configuration."""

    source: str
    target: str
    workload: str
    sources: int
    targets: int
    precision: str
    backend: str
    threads: int
    identity_map: bool = True

    def key(self) -> tuple:
        """Returns the identity used to resume an existing CSV."""
        return (
            self.source,
            self.target,
            self.workload,
            str(self.sources),
            str(self.targets),
            self.precision,
            self.backend,
            str(self.threads),
            "1" if self.identity_map else "0",
        )

    def matrix_bytes(self) -> int:
        scalar = 4 if self.precision == "fp32" else 8
        return 6 * self.sources * self.targets * scalar


def build_matrix(preset: str, backends: list[str], threads: int) -> list[Row]:
    """Returns the rows of a named preset."""
    rows: list[Row] = []

    def add(
        source: str,
        target: str,
        workload: str,
        sources: int,
        targets: int,
        precisions: tuple[str, ...] = ("fp32", "fp64"),
        row_backends: list[str] | None = None,
        row_threads: int = threads,
        identity_map: bool = True,
    ) -> None:
        for precision in precisions:
            for backend in row_backends if row_backends else backends:
                rows.append(
                    Row(
                        source=source,
                        target=target,
                        workload=workload,
                        sources=sources,
                        targets=targets,
                        precision=precision,
                        backend=backend,
                        threads=row_threads,
                        identity_map=identity_map,
                    )
                )

    if preset == "quick":
        for source, target in GEOMETRY_PAIRS:
            add(source, target, "lattice", 512, 512, precisions=("fp32",))
        return rows

    if preset == "geometry-matrix":
        # Every geometry pair under every workload at one shared size, so the
        # nine combinations and the three redundancy regimes are comparable.
        for source, target in GEOMETRY_PAIRS:
            for workload in WORKLOADS:
                add(source, target, workload, 1024, 1024)
        return rows

    if preset == "sizes":
        # Point pairs are cheap per pair, so they carry the large sizes; the
        # finite combinations stop where the build time stops being useful.
        for size in (256, 512, 1024, 2048, 4096):
            add("point", "point", "lattice", size, size)
            add("point", "point", "random", size, size)
        for size in (256, 512, 1024, 2048):
            add("prism", "prism", "lattice", size, size)
            add("prism", "prism", "lattice-irregular", size, size)
        for size in (256, 512, 1024):
            add("tetrahedron", "tetrahedron", "lattice", size, size)
        for size in (256, 512):
            add("tetrahedron", "tetrahedron", "lattice-irregular", size, size)
        return rows

    if preset == "asymmetric":
        # Independently generated source and target sets, so nothing about the
        # measurement depends on Ns == Nt or on a shared identity map.
        for source, target in GEOMETRY_PAIRS:
            add(source, target, "lattice", 2048, 1024, identity_map=False)
            add(source, target, "random", 2048, 1024, identity_map=False)
        return rows

    if preset == "threads":
        for count in (1, 2, 4, 8):
            add(
                "prism", "prism", "lattice", 1024, 1024,
                precisions=("fp32",), row_threads=count,
            )
            add(
                "prism", "prism", "lattice-irregular", 1024, 1024,
                precisions=("fp32",), row_threads=count,
            )
            add(
                "tetrahedron", "tetrahedron", "lattice-irregular", 512, 512,
                precisions=("fp32",), row_threads=count,
            )
            add(
                "point", "point", "random", 2048, 2048,
                precisions=("fp32",), row_threads=count,
            )
        return rows

    raise SystemExit(f"unknown preset {preset}")


def run_row(
    binary: pathlib.Path,
    row: Row,
    output: pathlib.Path,
    evaluations: int,
    warmups: int,
    repeats: int,
    probe_redundancy: bool,
    checksum: bool,
    label: str,
    commit: str,
) -> bool:
    """Runs one row, appending its CSV line.  Returns whether it succeeded."""
    command = [
        str(binary),
        "--source", row.source,
        "--target", row.target,
        "--workload", row.workload,
        "--sources", str(row.sources),
        "--targets", str(row.targets),
        "--precision", row.precision,
        "--backends", row.backend,
        "--threads", str(row.threads),
        "--evaluations", str(evaluations),
        "--warmups", str(warmups),
        "--construction-repeats", str(repeats),
        "--output", str(output),
        "--label", label,
        "--commit", commit,
    ]
    if not row.identity_map:
        command.append("--no-identity")
    if probe_redundancy:
        command.append("--probe-redundancy")
    if checksum:
        command.append("--checksum")

    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = str(row.threads)

    completed = subprocess.run(
        command, env=environment, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        print(
            f"  FAILED rc={completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}",
            file=sys.stderr,
        )
        return False
    if completed.stderr.strip():
        print(f"  note: {completed.stderr.strip()}", file=sys.stderr)
    return True


def load_done(path: pathlib.Path) -> set[tuple]:
    """Returns the row identities already present in an existing CSV."""
    if not path.exists():
        return set()
    done: set[tuple] = set()
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            done.add(
                (
                    record["source"],
                    record["target"],
                    record["workload"],
                    record["sources"],
                    record["targets"],
                    record["precision"],
                    record["backend"],
                    record["threads"],
                    record["identity_map"],
                )
            )
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--preset",
        default="quick",
        choices=("quick", "geometry-matrix", "sizes", "asymmetric", "threads"),
    )
    parser.add_argument(
        "--backends",
        default="cpu",
        help="comma list of cpu, mkl and cuda (default cpu)",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--evaluations", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--construction-repeats", type=int, default=3)
    parser.add_argument("--probe-redundancy", action="store_true")
    parser.add_argument("--checksum", action="store_true")
    parser.add_argument("--label", default="")
    parser.add_argument("--commit", default="")
    parser.add_argument(
        "--max-matrix-bytes",
        type=int,
        default=4 * 1024**3,
        help="refuse a row whose six matrices would exceed this (default 4 GiB)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list the rows and stop"
    )
    arguments = parser.parse_args()

    backends = [item for item in arguments.backends.split(",") if item]
    rows = build_matrix(arguments.preset, backends, arguments.threads)
    done = load_done(arguments.output)

    pending = []
    for row in rows:
        if row.key() in done:
            continue
        if row.matrix_bytes() > arguments.max_matrix_bytes:
            print(
                f"skipping {row.key()}: "
                f"{row.matrix_bytes() / 1024**3:.2f} GiB of matrices "
                f"exceeds the budget",
                file=sys.stderr,
            )
            continue
        pending.append(row)

    print(f"{len(rows)} rows in preset, {len(pending)} to run")
    if arguments.dry_run:
        for row in pending:
            print(
                f"  {row.source}->{row.target} {row.workload} "
                f"{row.sources}x{row.targets} {row.precision} {row.backend} "
                f"t={row.threads} "
                f"({row.matrix_bytes() / 1024**2:.0f} MiB)"
            )
        return 0

    failures = 0
    for index, row in enumerate(pending, start=1):
        print(
            f"[{index}/{len(pending)}] {row.source}->{row.target} "
            f"{row.workload} {row.sources}x{row.targets} {row.precision} "
            f"{row.backend} t={row.threads}",
            flush=True,
        )
        if not run_row(
            arguments.binary,
            row,
            arguments.output,
            arguments.evaluations,
            arguments.warmups,
            arguments.construction_repeats,
            arguments.probe_redundancy,
            arguments.checksum,
            arguments.label,
            arguments.commit,
        ):
            failures += 1

    print(f"done: {len(pending) - failures} succeeded, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
