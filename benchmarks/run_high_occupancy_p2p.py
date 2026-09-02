#!/usr/bin/env python3
"""Run the bounded regular-grid high-occupancy FMM study.

The runner deliberately records every process repetition. It never launches a
case whose conservative pairwise P2P estimate exceeds the selected GPU limit.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import statistics

PARTICLE_COUNTS = (2**13, 2**14, 2**15, 2**16)
DEPTHS = (2, 3)
BACKENDS = ("cpu-static-matrix", "cuda-full")


@dataclass(frozen=True)
class StudyCase:
    particles: int
    depth: int
    backend: str

    @property
    def key(self) -> str:
        return f"{self.backend}_n{self.particles}_d{self.depth}"

    @property
    def mean_occupancy(self) -> float:
        return self.particles / float((2**self.depth) ** 3)


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_options(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", type=Path,
        default=Path("build-bench-all/benchmarks/benchmark_uniform_fmm"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluations", type=positive_integer, default=10)
    parser.add_argument("--cpu-threads", type=positive_integer, default=8)
    parser.add_argument("--gpu-memory-limit-gib", type=float, default=20.0)
    parser.add_argument("--host-memory-limit-gib", type=float, default=50.0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("high_occupancy_p2p_results/raw"),
    )
    return parser.parse_args(argv)


def build_cases() -> list[StudyCase]:
    return [
        StudyCase(particles, depth, backend)
        for particles in PARTICLE_COUNTS
        for depth in DEPTHS
        for backend in BACKENDS
    ]


def pairwise_bytes(case: StudyCase) -> int:
    """Return a conservative interior pairwise Tensor6 storage estimate."""
    q = case.mean_occupancy
    interactions = int(27.0 * case.particles * q)
    if case.backend == "cuda-full":
        # BSR uses nine FP32 values and one source index per interaction.
        return interactions * (9 * 4 + 4)
    # The CPU setup simultaneously retains canonical and SoA representations.
    return interactions * (88 + 6 * 4 + 4 + 1)


def command(case: StudyCase, options: argparse.Namespace, output: Path) -> list[str]:
    return [
        str(options.executable), "--sources", str(case.particles),
        "--targets", str(case.particles), "--depth", str(case.depth),
        "--order", "6", "--expansion-basis", "spherical",
        "--precision", "float32", "--regular-grid", "--exact-cuboid-p2p",
        "--backend", case.backend, "--threads", str(options.cpu_threads),
        "--evaluations", "1", "--samples", "1", "--warmups", "1",
        "--no-direct", "--no-workload-comparison", "--output", str(output),
    ]


def load_completed(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("status") == "OK":
                completed.add((record["case"], int(record["repetition"])))
    return completed


def write_processed_outputs(rows: list[dict[str, str]], output_dir: Path) -> None:
    """Write the compact median table and regenerable study figures."""
    groups: dict[tuple[str, int, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["execution_backend"], int(row["sources"]), int(row["depth"]))
        groups.setdefault(key, []).append(row)
    processed: list[dict[str, float | int | str]] = []
    far_columns = ("p2m", "m2m", "m2l", "l2l", "l2p")
    for (backend, particles, depth), repetitions in sorted(groups.items()):
        median = lambda column: statistics.median(
            float(row[column]) for row in repetitions
        )
        memory_column = (
            "cuda_persistent_device_bytes"
            if backend == "cuda-full" else "static_plan_bytes"
        )
        processed.append({
            "backend": backend, "N": particles, "depth": depth,
            "q": particles / float((2**depth) ** 3),
            "p2p_time": median("p2p"),
            "p2p_interactions_per_second": (
                median("near_field_pairs") / median("p2p")
                if median("p2p") > 0.0 else 0.0
            ),
            "far_time": sum(median(column) for column in far_columns),
            "total_time": median("evaluation_median"),
            "memory": median(memory_column),
        })
    processed_path = output_dir / "processed.csv"
    with processed_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(processed[0]))
        writer.writeheader()
        writer.writerows(processed)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plots = {
        "p2p_runtime_vs_q": ("q", "p2p_time"),
        "p2p_interactions_per_second_vs_q":
            ("q", "p2p_interactions_per_second"),
        "persistent_memory_vs_q": ("q", "memory"),
        "total_fmm_time_vs_q": ("q", "total_time"),
        "near_vs_far_time": ("far_time", "p2p_time"),
        "preferred_depth_vs_n": ("N", "depth"),
    }
    for name, (x_column, y_column) in plots.items():
        figure, axis = plt.subplots()
        for backend in BACKENDS:
            selected = [row for row in processed if row["backend"] == backend]
            axis.plot([row[x_column] for row in selected],
                      [row[y_column] for row in selected], "o-", label=backend)
        axis.set_xlabel(x_column)
        axis.set_ylabel(y_column)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"{name}.png", dpi=160)
        plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    options = parse_options(argv)
    cases = build_cases()
    if options.dry_run:
        print(f"planned_cases={len(cases)}")
        for case in cases:
            output = options.output_dir / f"{case.key}_r000.csv"
            print(
                f"{case.key} q_mean={case.mean_occupancy:g} "
                + shlex.join(command(case, options, output))
            )
        return 0

    options.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = options.output_dir / "repetitions.jsonl"
    completed = load_completed(manifest) if options.resume else set()
    gpu_limit = int(options.gpu_memory_limit_gib * 1024**3)
    host_limit = int(options.host_memory_limit_gib * 1024**3)
    with manifest.open("a", encoding="utf-8") as records:
        for case in cases:
            for repetition in range(options.evaluations):
                if (case.key, repetition) in completed:
                    continue
                output = options.output_dir / f"{case.key}_r{repetition:03d}.csv"
                estimate = pairwise_bytes(case)
                limit = gpu_limit if case.backend == "cuda-full" else host_limit
                if estimate > limit:
                    record = {
                        "case": case.key, "repetition": repetition,
                        "status": "SKIPPED_MEMORY",
                        "estimated_pairwise_bytes": estimate,
                    }
                else:
                    environment = os.environ.copy()
                    environment.update({
                        "OMP_NUM_THREADS": str(options.cpu_threads),
                        "OMP_PROC_BIND": "close", "OMP_PLACES": "cores",
                    })
                    result = subprocess.run(
                        command(case, options, output), env=environment,
                        text=True, capture_output=True, check=False,
                    )
                    record = {
                        "case": case.key, "repetition": repetition,
                        "status": "OK" if result.returncode == 0 else "FAILED",
                        "returncode": result.returncode,
                        "estimated_pairwise_bytes": estimate,
                        "stdout": result.stdout, "stderr": result.stderr,
                        "result_csv": str(output),
                    }
                records.write(json.dumps(record) + "\n")
                records.flush()

    rows: list[dict[str, str]] = []
    for result_path in sorted(options.output_dir.glob("*_r*.csv")):
        with result_path.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    if rows:
        columns = list(rows[0]) + ["raw_file"]
        with (options.output_dir.parent / "all_repetitions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row, result_path in zip(rows, sorted(options.output_dir.glob("*_r*.csv"))):
                row["raw_file"] = str(result_path)
                writer.writerow(row)
        write_processed_outputs(rows, options.output_dir.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
