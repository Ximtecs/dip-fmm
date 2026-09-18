#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cold static-plan construction matrix.

This driver measures *plan construction*, not repeated evaluation.  It runs
`benchmark_uniform_fmm` once per matrix row with the geometry cache disabled,
so every row is a cold build, and keeps the per-phase `StaticPlanStatistics`
timings that the solver already records.

The evaluation columns are still written out, because a construction change is
only acceptable when repeated evaluation does not regress; `--evaluations`
controls how much evaluation work each row pays for that guard.

Rows are appended to the CSV as they complete and an existing CSV is resumed,
so an interrupted sweep can be restarted and a long row can be skipped without
losing the rest of the matrix.

Examples
--------
    python benchmarks/run_construction_matrix.py \
        --binary build-bench-all/benchmarks/benchmark_uniform_fmm \
        --output construction-baseline.csv --preset quick

    python benchmarks/run_construction_matrix.py \
        --binary build-bench-all/benchmarks/benchmark_uniform_fmm \
        --output construction-baseline.csv --preset full
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
import time

# Phase timings reported by `StaticPlanStatistics`, in the order the solver
# prints them.  Each name is the text after "setup." in the solver summary.
PHASE_KEYS: tuple[str, ...] = (
    "normalisation",
    "tree_construction",
    "topology_construction",
    "universal_cache_lookup",
    "universal_cache_load",
    "universal_operator_build",
    "universal_cache_write",
    "periodic_cache_lookup",
    "periodic_cache_load",
    "periodic_operator_build",
    "geometry_hash",
    "geometry_cache_lookup",
    "geometry_cache_load",
    "geometry_cache_write",
    "p2m",
    "m2m",
    "m2l",
    "l2l",
    "l2p",
    "p2p",
    "p2p_interaction_setup",
    "p2p_canonical_operator",
    "p2p_derived_packing",
    "precision_conversion",
    "backend_packing",
    "far_field_packing",
    "cuda_upload",
    "static_plan",
    "total",
)

# Scalar facts scraped from the same summary block.
SUMMARY_KEYS: tuple[str, ...] = (
    "cache.geometry.hit",
    "cache.universal.hit",
    "cache.bytes_read",
    "cache.bytes_written",
)

# Columns lifted out of the benchmark's own CSV row.
CSV_KEYS: tuple[str, ...] = (
    "near_field_pairs",
    "p2p_static_interactions",
    "p2p_unique_tensors",
    "p2p_canonical_total_bytes",
    "p2p_dictionary_total_bytes",
    "near_field_operator_bytes",
    "p2m_operator_bytes",
    "l2p_operator_bytes",
    "static_plan_bytes",
    "cached_operator_bytes",
    "evaluation_median",
    "occupied_source_leaves",
    "occupied_target_leaves",
    "total_nodes",
    "p2p_packing",
    "p2m_execution",
    "l2p_execution",
    "coefficient_count",
)


@dataclasses.dataclass(frozen=True)
class Row:
    """One matrix cell: a geometry/size/order/backend combination."""

    label: str
    geometry: str
    layout: str
    irregular: bool
    bodies: int
    depth: int
    order: int
    backend: str
    precision: str

    def key(self) -> tuple:
        return (
            self.label,
            self.bodies,
            self.depth,
            self.order,
            self.backend,
            self.precision,
        )


# Body counts and the tree depth that keeps the leaf occupancy comparable
# across sizes.  `--regular-grid` needs a power-of-two-times-odd count, so the
# lattice sizes are cubes of powers of two.
SIZES: dict[str, tuple[int, int]] = {
    "small": (4096, 3),
    "medium": (13824, 4),
    "large": (32768, 4),
}


def build_matrix(preset: str) -> list[Row]:
    """Return the matrix rows for a preset.

    `quick` is the iteration matrix: small sizes only, one order, one backend.
    `baseline` adds the order sweep and the backend spot checks.
    `full` adds the medium and large sizes.
    """

    geometries = (
        # label, geometry, layout, irregular
        ("point-random", "point", "general", False),
        ("point-lattice", "point", "regular-grid", False),
        ("prism-lattice", "prism", "regular-grid", False),
        ("prism-irregular", "prism", "regular-grid", True),
        ("tetra-lattice", "tetrahedron", "regular-grid", False),
        ("tetra-irregular", "tetrahedron", "regular-grid", True),
    )

    rows: list[Row] = []

    def add(size_name: str, order: int, backend: str, precision: str) -> None:
        bodies, depth = SIZES[size_name]
        for label, geometry, layout, irregular in geometries:
            rows.append(
                Row(
                    label=label,
                    geometry=geometry,
                    layout=layout,
                    irregular=irregular,
                    bodies=bodies,
                    depth=depth,
                    order=order,
                    backend=backend,
                    precision=precision,
                )
            )

    # The primary construction comparison: p = 6, FP32, portable CPU.
    add("small", 6, "cpu-static-matrix", "float32")

    if preset == "quick":
        return rows

    # Order sweep at the small size.  Construction cost scales with the
    # expansion order through P2M/L2P, not through P2P.
    for order in (4, 8):
        add("small", order, "cpu-static-matrix", "float32")

    # Backend spot checks.  Most exact geometry construction is host work
    # shared by every backend, so these exist to catch backend-specific
    # packing and upload costs, not to repeat the geometry measurement.
    for backend in ("cuda-full", "cuda-partial", "cpu-static-matrix-mkl"):
        add("small", 6, backend, "float32")

    # FP64 costs the same to construct but converts differently.
    add("small", 6, "cpu-static-matrix", "float64")

    if preset == "baseline":
        return rows

    add("medium", 6, "cpu-static-matrix", "float32")
    add("large", 6, "cpu-static-matrix", "float32")
    add("large", 6, "cuda-full", "float32")

    return rows


def parse_summary(text: str) -> dict[str, str]:
    """Scrape `key: value` lines from the solver's initialisation summary."""

    found: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        found[key.strip()] = value.strip()
    return found


def run_row(
    binary: pathlib.Path,
    row: Row,
    evaluations: int,
    timeout: float,
) -> dict[str, str] | None:
    """Run one matrix cell and return its flattened result record."""

    command = [
        str(binary),
        "--sources",
        str(row.bodies),
        "--targets",
        str(row.bodies),
        "--depth",
        str(row.depth),
        "--order",
        str(row.order),
        "--backend",
        row.backend,
        "--precision",
        row.precision,
        "--source-geometry",
        row.geometry,
        "--target-geometry",
        row.geometry,
        "--spatial-layout",
        row.layout,
        "--evaluations",
        str(evaluations),
        "--samples",
        "1",
        "--warmups",
        "0",
        "--no-direct",
        "--no-workload-comparison",
    ]
    if row.layout == "regular-grid":
        command.append("--regular-grid")
    if row.irregular:
        command.append("--irregular-bodies")

    environment = dict(os.environ)
    # Every row in this matrix is a cold build.
    environment["CDFMM_DISABLE_CACHE"] = "1"

    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout:.0f} s", flush=True)
        return None
    wall = time.monotonic() - started

    if completed.returncode != 0:
        print(f"  FAILED rc={completed.returncode}", flush=True)
        print(completed.stderr.strip()[:2000], flush=True)
        return None

    summary = parse_summary(completed.stdout)

    # The benchmark's own CSV is the last two non-empty lines of stdout: a
    # header and one row.
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    csv_row: dict[str, str] = {}
    for index in range(len(lines) - 1):
        if lines[index].startswith("sources,targets,depth,order,"):
            header = next(csv.reader([lines[index]]))
            values = next(csv.reader([lines[index + 1]]))
            csv_row = dict(zip(header, values))
            break

    record: dict[str, str] = {
        "label": row.label,
        "geometry": row.geometry,
        "layout": row.layout,
        "irregular": "1" if row.irregular else "0",
        "bodies": str(row.bodies),
        "depth": str(row.depth),
        "order": str(row.order),
        "backend": row.backend,
        "precision": row.precision,
        "wall_seconds": f"{wall:.6f}",
    }
    for key in PHASE_KEYS:
        record[f"setup_{key}_s"] = summary.get(f"setup.{key}_seconds", "")
    for key in SUMMARY_KEYS:
        record[key.replace(".", "_")] = summary.get(key, "")
    for key in CSV_KEYS:
        record[key] = csv_row.get(key, "")
    return record


def field_names() -> list[str]:
    names = [
        "label",
        "geometry",
        "layout",
        "irregular",
        "bodies",
        "depth",
        "order",
        "backend",
        "precision",
        "wall_seconds",
    ]
    names += [f"setup_{key}_s" for key in PHASE_KEYS]
    names += [key.replace(".", "_") for key in SUMMARY_KEYS]
    names += list(CSV_KEYS)
    return names


def load_done(path: pathlib.Path) -> set[tuple]:
    """Return the keys of rows an existing CSV already holds."""

    if not path.exists():
        return set()
    done: set[tuple] = set()
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            done.add(
                (
                    record["label"],
                    int(record["bodies"]),
                    int(record["depth"]),
                    int(record["order"]),
                    record["backend"],
                    record["precision"],
                )
            )
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=pathlib.Path,
        required=True,
        help="path to benchmark_uniform_fmm",
    )
    parser.add_argument(
        "--output", type=pathlib.Path, required=True, help="result CSV"
    )
    parser.add_argument(
        "--preset",
        choices=("quick", "baseline", "full"),
        default="baseline",
        help="matrix size (default: baseline)",
    )
    parser.add_argument(
        "--evaluations",
        type=int,
        default=8,
        help="timed evaluations per row, the regression guard (default: 8)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="per-row timeout in seconds (default: 1800)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="restrict to these labels (repeatable)",
    )
    arguments = parser.parse_args()

    if not arguments.binary.exists():
        print(f"no such binary: {arguments.binary}", file=sys.stderr)
        return 2

    rows = build_matrix(arguments.preset)
    if arguments.only:
        wanted = set(arguments.only)
        rows = [row for row in rows if row.label in wanted]

    done = load_done(arguments.output)
    pending = [row for row in rows if row.key() not in done]
    print(
        f"{len(rows)} matrix rows, {len(done)} already recorded, "
        f"{len(pending)} to run",
        flush=True,
    )

    write_header = not arguments.output.exists()
    with arguments.output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names())
        if write_header:
            writer.writeheader()
            handle.flush()
        for index, row in enumerate(pending, start=1):
            print(
                f"[{index}/{len(pending)}] {row.label} N={row.bodies} "
                f"d={row.depth} p={row.order} {row.backend} {row.precision}",
                flush=True,
            )
            record = run_row(
                arguments.binary, row, arguments.evaluations, arguments.timeout
            )
            if record is None:
                continue
            writer.writerow(record)
            handle.flush()
            total = record.get("setup_total_s") or "?"
            p2p = record.get("setup_p2p_s") or "?"
            print(f"  total={total} s  p2p={p2p} s", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
