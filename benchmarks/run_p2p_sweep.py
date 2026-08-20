#!/usr/bin/env python3
"""Run and collect isolated static-P2P benchmark sweeps."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import datetime as dt
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cdfmm-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PARTICLE_COUNTS = tuple(2**exponent for exponent in range(12, 18))
DEFAULT_DEPTHS = (2, 3, 4, 5)


@dataclass(frozen=True)
class SweepCase:
    """One exact-size benchmark geometry at a selected tree depth."""

    depth: int
    particles: int
    irregular: bool

    @property
    def stem(self) -> str:
        """Return a stable file stem describing this case."""
        geometry = "irregular" if self.irregular else "uniform"
        return f"d{self.depth}_n{self.particles}_{geometry}"


def positive_integer(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least one")
    return parsed


def parse_options() -> argparse.Namespace:
    """Parse sweep configuration from the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Run benchmark_p2p over exact particle counts and tree depths, "
            "saving raw output and consolidated CSV tables."
        )
    )
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("build-bench-all/benchmarks/benchmark_p2p"),
        help="benchmark_p2p executable (default: %(default)s)",
    )
    parser.add_argument(
        "--evaluations",
        type=positive_integer,
        default=100,
        help="repeated evaluations per implementation (default: %(default)s)",
    )
    parser.add_argument(
        "--threads",
        type=positive_integer,
        default=8,
        help="OpenMP threads used by CPU implementations (default: %(default)s)",
    )
    parser.add_argument(
        "--particles",
        nargs="+",
        type=positive_integer,
        default=list(DEFAULT_PARTICLE_COUNTS),
        help=(
            "particle counts "
            "(default: 4096 8192 16384 32768 65536 131072)"
        ),
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        type=positive_integer,
        default=list(DEFAULT_DEPTHS),
        help="candidate depths (default: 2 3 4 5)",
    )
    parser.add_argument(
        "--gpu-memory-limit-gib",
        type=float,
        default=20.0,
        help=(
            "maximum estimated persistent CUDA-plan memory in GiB "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--host-memory-limit-gib",
        type=float,
        default=50.0,
        help=(
            "maximum estimated simultaneous host packing memory in GiB "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--irregular",
        action="store_true",
        help="generate irregular rather than uniform point clouds",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="result directory (default: timestamp below p2p_sweep_results)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record a failed case and continue the remaining sweep",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned commands without executing them",
    )
    return parser.parse_args()


def build_cases(options: argparse.Namespace) -> list[SweepCase]:
    """Construct the full particle-count and depth cross-product."""
    return [
        SweepCase(depth, particles, options.irregular)
        for particles in options.particles
        for depth in options.depths
    ]


def estimate_uniform_interactions(case: SweepCase) -> int:
    """Count interactions for the benchmark's even leaf distribution."""
    boxes_per_axis = 1 << case.depth
    leaf_count = boxes_per_axis**3
    leaf_occupancies = [
        (leaf + 1) * case.particles // leaf_count
        - leaf * case.particles // leaf_count
        for leaf in range(leaf_count)
    ]
    interactions = 0
    for target_leaf, target_occupancy in enumerate(leaf_occupancies):
        if target_occupancy == 0:
            continue
        target_x = target_leaf % boxes_per_axis
        target_y = (target_leaf // boxes_per_axis) % boxes_per_axis
        target_z = target_leaf // (boxes_per_axis**2)
        for source_z in range(
            max(0, target_z - 1), min(boxes_per_axis, target_z + 2)
        ):
            for source_y in range(
                max(0, target_y - 1), min(boxes_per_axis, target_y + 2)
            ):
                for source_x in range(
                    max(0, target_x - 1),
                    min(boxes_per_axis, target_x + 2),
                ):
                    source_leaf = (
                        (source_z * boxes_per_axis + source_y)
                        * boxes_per_axis
                        + source_x
                    )
                    interactions += (
                        target_occupancy * leaf_occupancies[source_leaf]
                    )
    return interactions


def estimate_interactions(case: SweepCase) -> int:
    """Estimate interactions before allocating the static representations."""
    if not case.irregular:
        return estimate_uniform_interactions(case)

    boxes_per_axis = 1 << case.depth
    leaf_count = boxes_per_axis**3
    directed_neighbour_pairs = (3 * boxes_per_axis - 2) ** 3
    expected = (
        case.particles**2 * directed_neighbour_pairs / (leaf_count**2)
    )
    # Random occupancy fluctuates around the expectation. Retain a generous
    # margin because this estimate is used as an allocation safety check.
    return max(case.particles, math.ceil(1.5 * expected))


def estimate_memory_bytes(
    case: SweepCase, interaction_count: int
) -> tuple[int, int]:
    """Estimate peak known host storage and the largest CUDA plan."""
    # During construction the canonical, SoA, leaf, BSR, oneMKL index copy,
    # and original interaction pairs overlap in host memory. The per-particle
    # allowance covers moments, fields, identities, positions, and tree data.
    host_bytes = interaction_count * 320 + case.particles * 256

    # cuSPARSE BSR is the largest device packing: nine doubles, one source
    # index, row offsets, and persistent input/output vectors.
    gpu_bytes = (
        interaction_count * (9 * 8 + 4)
        + (case.particles + 1) * 4
        + case.particles * 2 * 3 * 8
    )
    return host_bytes, gpu_bytes


def available_host_memory_bytes() -> int:
    """Return currently available host memory reported by Linux."""
    with Path("/proc/meminfo").open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("could not determine available host memory")


def available_gpu_memory_bytes() -> int:
    """Return free memory on the first GPU visible to nvidia-smi."""
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise RuntimeError(
            "could not query free GPU memory with nvidia-smi: "
            + completed.stderr.strip()
        )
    first_gpu_mebibytes = int(completed.stdout.splitlines()[0].strip())
    return first_gpu_mebibytes * 1024**2


def parse_key_values(line: str) -> dict[str, str]:
    """Parse the comma-separated key=value geometry summary."""
    result: dict[str, str] = {}
    for item in line.split(",")[1:]:
        key, value = item.split("=", maxsplit=1)
        result[key] = value
    return result


def parse_benchmark_output(
    output: str,
) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
    """Extract geometry, CPU rows, and CUDA rows from benchmark output."""
    geometry: dict[str, str] = {}
    cpu_rows: list[dict[str, str]] = []
    cuda_rows: list[dict[str, str]] = []
    header: list[str] | None = None
    destination: list[dict[str, str]] | None = None

    for line in output.splitlines():
        if line.startswith("case,"):
            geometry = parse_key_values(line)
        elif line.startswith("implementation,"):
            header = next(csv.reader([line]))
            destination = cpu_rows
        elif line.startswith("cuda_implementation,"):
            header = next(csv.reader([line]))
            destination = cuda_rows
        elif header is not None and destination is not None and line:
            values = next(csv.reader([line]))
            if len(values) == len(header):
                destination.append(dict(zip(header, values, strict=True)))

    if not geometry or not cpu_rows:
        raise ValueError("benchmark output did not contain a complete CPU table")
    return geometry, cpu_rows, cuda_rows


def add_case_columns(
    rows: list[dict[str, str]],
    case: SweepCase,
    geometry: dict[str, str],
) -> list[dict[str, str]]:
    """Prefix result rows with reproducible case metadata."""
    prefix = {
        "requested_depth": str(case.depth),
        "requested_particles": str(case.particles),
        "irregular": str(case.irregular).lower(),
        **geometry,
    }
    return [{**prefix, **row} for row in rows]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write rows using the union of fields in first-seen order."""
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fastest_cuda_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    """Return the implementation with the shortest reported device time."""
    if not rows:
        return None
    return min(rows, key=lambda row: float(row["device_total_s"]))


def fastest_cpu_row(rows: list[dict[str, str]]) -> dict[str, str]:
    """Return the CPU implementation with the shortest evaluation time."""
    return min(rows, key=lambda row: float(row["evaluation_s"]))


def group_implementation_rows(
    rows: list[dict[str, str]],
    depth: int,
    implementation_column: str,
) -> dict[str, list[dict[str, str]]]:
    """Group one depth's rows by implementation and sort by particle count."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if int(row["requested_depth"]) != depth:
            continue
        grouped.setdefault(row[implementation_column], []).append(row)
    for implementation_rows in grouped.values():
        implementation_rows.sort(key=lambda row: int(row["particles"]))
    return grouped


def plot_scaling_metric(
    cpu_rows: list[dict[str, str]],
    cuda_rows: list[dict[str, str]],
    depths: list[int],
    cpu_metric: str,
    cuda_metric: str,
    ylabel: str,
    output_path: Path,
    scale: float = 1.0,
) -> None:
    """Plot one CPU and CUDA scaling metric for every measured depth."""
    figure, axes = plt.subplots(
        len(depths), 2, figsize=(14.0, 3.8 * len(depths)), squeeze=False
    )
    datasets = (
        (cpu_rows, "implementation", cpu_metric, "CPU and oneMKL"),
        (cuda_rows, "cuda_implementation", cuda_metric, "CUDA"),
    )
    for row_index, depth in enumerate(depths):
        for column_index, (
            rows,
            implementation_column,
            metric,
            backend,
        ) in enumerate(datasets):
            axis = axes[row_index][column_index]
            grouped = group_implementation_rows(
                rows, depth, implementation_column
            )
            for implementation, implementation_rows in grouped.items():
                axis.plot(
                    [int(row["particles"]) for row in implementation_rows],
                    [float(row[metric]) * scale for row in implementation_rows],
                    marker="o",
                    linewidth=1.5,
                    label=implementation,
                )
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_title(f"{backend}, depth {depth}")
            axis.set_xlabel("Particles")
            axis.set_ylabel(ylabel)
            axis.grid(True, which="both", alpha=0.3)
            if grouped:
                axis.legend(fontsize=7)
    figure.suptitle(output_path.stem.replace("_", " ").title())
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_runtime_memory_tradeoff(
    cpu_rows: list[dict[str, str]],
    cuda_rows: list[dict[str, str]],
    depths: list[int],
    output_path: Path,
) -> None:
    """Plot evaluation time against persistent storage for each depth."""
    figure, axes = plt.subplots(
        len(depths), 2, figsize=(14.0, 3.8 * len(depths)), squeeze=False
    )
    datasets = (
        (
            cpu_rows,
            "implementation",
            "total_bytes",
            "evaluation_s",
            "CPU and oneMKL",
        ),
        (
            cuda_rows,
            "cuda_implementation",
            "persistent_device_bytes",
            "device_total_s",
            "CUDA",
        ),
    )
    for row_index, depth in enumerate(depths):
        for column_index, (
            rows,
            implementation_column,
            memory_column,
            runtime_column,
            backend,
        ) in enumerate(datasets):
            axis = axes[row_index][column_index]
            grouped = group_implementation_rows(
                rows, depth, implementation_column
            )
            for implementation, implementation_rows in grouped.items():
                memory = [
                    float(row[memory_column]) / (1024.0**2)
                    for row in implementation_rows
                ]
                runtime = [
                    float(row[runtime_column]) * 1.0e6
                    for row in implementation_rows
                ]
                axis.plot(
                    memory,
                    runtime,
                    marker="o",
                    linewidth=1.5,
                    label=implementation,
                )
                for x_value, y_value, row in zip(
                    memory, runtime, implementation_rows, strict=True
                ):
                    exponent = int(round(math.log2(int(row["particles"]))))
                    axis.annotate(
                        f"2^{exponent}",
                        (x_value, y_value),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=6,
                    )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_title(f"{backend}, depth {depth}")
            axis.set_xlabel("Persistent memory (MiB)")
            axis.set_ylabel("Evaluation time (us)")
            axis.grid(True, which="both", alpha=0.3)
            if grouped:
                axis.legend(fontsize=7)
    figure.suptitle("Runtime Versus Persistent-Memory Trade-off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def generate_figures(
    cpu_rows: list[dict[str, str]],
    cuda_rows: list[dict[str, str]],
    output_directory: Path,
) -> list[Path]:
    """Generate runtime, memory, throughput, and trade-off figures."""
    depths = sorted(
        {
            int(row["requested_depth"])
            for row in [*cpu_rows, *cuda_rows]
        }
    )
    if not depths:
        return []

    figures = [
        output_directory / "runtime_scaling.png",
        output_directory / "memory_scaling.png",
        output_directory / "throughput_scaling.png",
        output_directory / "runtime_memory_tradeoff.png",
    ]
    plot_scaling_metric(
        cpu_rows,
        cuda_rows,
        depths,
        "evaluation_s",
        "device_total_s",
        "Evaluation time (us)",
        figures[0],
        scale=1.0e6,
    )
    plot_scaling_metric(
        cpu_rows,
        cuda_rows,
        depths,
        "total_bytes",
        "persistent_device_bytes",
        "Persistent memory (MiB)",
        figures[1],
        scale=1.0 / (1024.0**2),
    )
    plot_scaling_metric(
        cpu_rows,
        cuda_rows,
        depths,
        "interactions_per_s",
        "interactions_per_kernel_s",
        "Interactions per second",
        figures[2],
    )
    plot_runtime_memory_tradeoff(
        cpu_rows, cuda_rows, depths, figures[3]
    )
    return figures


def main() -> int:
    """Execute the requested sweep and write consolidated results."""
    options = parse_options()
    if options.gpu_memory_limit_gib <= 0.0:
        print(
            "run_p2p_sweep: --gpu-memory-limit-gib must be positive",
            file=sys.stderr,
        )
        return 2
    if options.host_memory_limit_gib <= 0.0:
        print(
            "run_p2p_sweep: --host-memory-limit-gib must be positive",
            file=sys.stderr,
        )
        return 2
    cases = build_cases(options)

    executable = options.executable.expanduser().resolve()
    if not options.dry_run and not executable.is_file():
        print(f"run_p2p_sweep: executable not found: {executable}", file=sys.stderr)
        return 2

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = options.output_dir or Path(
        "p2p_sweep_results", timestamp
    )
    output_directory = output_directory.expanduser().resolve()
    if not options.dry_run:
        output_directory.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(options.threads)
    environment["MKL_NUM_THREADS"] = str(options.threads)

    all_cases: list[dict[str, str]] = []
    all_cpu_rows: list[dict[str, str]] = []
    all_cuda_rows: list[dict[str, str]] = []
    skipped_cases: list[dict[str, str]] = []
    failures = 0

    for index, case in enumerate(cases, start=1):
        interaction_estimate = estimate_interactions(case)
        host_estimate, gpu_estimate = estimate_memory_bytes(
            case, interaction_estimate
        )
        try:
            host_available = available_host_memory_bytes()
            gpu_available = available_gpu_memory_bytes()
        except RuntimeError as error:
            print(f"run_p2p_sweep: {error}", file=sys.stderr)
            return 2
        configured_host_limit = int(
            options.host_memory_limit_gib * 1024**3
        )
        configured_gpu_limit = int(
            options.gpu_memory_limit_gib * 1024**3
        )
        host_limit = min(configured_host_limit, host_available)
        gpu_limit = min(configured_gpu_limit, gpu_available)

        command = [
            str(executable),
            "--depth",
            str(case.depth),
            "--particles",
            str(case.particles),
            "--evaluations",
            str(options.evaluations),
        ]
        if case.irregular:
            command.append("--irregular")
        command.append("--cuda")

        print(
            f"[{index:>2}/{len(cases)}] depth={case.depth} "
            f"particles={case.particles} "
            f"geometry={'irregular' if case.irregular else 'uniform'} "
            f"interactions~{interaction_estimate} "
            f"host~{host_estimate / 1024**3:.2f} GiB "
            f"device~{gpu_estimate / 1024**3:.2f} GiB",
            flush=True,
        )

        skip_reasons = []
        if host_estimate > host_limit:
            if configured_host_limit <= host_available:
                skip_reasons.append(
                    "host estimate exceeds configured "
                    f"{options.host_memory_limit_gib:g} GiB limit"
                )
            else:
                skip_reasons.append(
                    "host estimate exceeds currently available host memory"
                )
        if gpu_estimate > gpu_limit:
            if configured_gpu_limit <= gpu_available:
                skip_reasons.append(
                    "device estimate exceeds configured "
                    f"{options.gpu_memory_limit_gib:g} GiB limit"
                )
            else:
                skip_reasons.append(
                    "device estimate exceeds currently free GPU memory"
                )
        if skip_reasons:
            skipped_cases.append(
                {
                    "requested_particles": str(case.particles),
                    "requested_depth": str(case.depth),
                    "irregular": str(case.irregular).lower(),
                    "estimated_interactions": str(interaction_estimate),
                    "estimated_host_bytes": str(host_estimate),
                    "configured_host_limit_bytes": str(
                        configured_host_limit
                    ),
                    "available_host_bytes": str(host_available),
                    "estimated_device_bytes": str(gpu_estimate),
                    "configured_device_limit_bytes": str(
                        configured_gpu_limit
                    ),
                    "free_device_bytes": str(gpu_available),
                    "reason": "; ".join(skip_reasons),
                }
            )
            print(f"  skipped: {'; '.join(skip_reasons)}")
            if not options.dry_run:
                write_csv(
                    output_directory / "skipped_cases.csv", skipped_cases
                )
            continue
        if options.dry_run:
            print("  " + " ".join(command))
            continue

        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        raw_path = output_directory / f"{case.stem}.txt"
        raw_path.write_text(
            completed.stdout + completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            failures += 1
            print(
                f"  failed with exit status {completed.returncode}; "
                f"see {raw_path}",
                file=sys.stderr,
            )
            if not options.continue_on_error:
                return completed.returncode
            continue

        try:
            geometry, cpu_rows, cuda_rows = parse_benchmark_output(
                completed.stdout
            )
        except ValueError as error:
            failures += 1
            print(f"  could not parse {raw_path}: {error}", file=sys.stderr)
            if not options.continue_on_error:
                return 1
            continue

        cpu_implementations = {row["implementation"] for row in cpu_rows}
        if "onemkl-bsr3" not in cpu_implementations:
            print(
                "  combined benchmark is missing onemkl-bsr3; configure and "
                "build the benchmark-all preset",
                file=sys.stderr,
            )
            return 2
        if not cuda_rows:
            print("  combined benchmark did not produce CUDA rows", file=sys.stderr)
            return 2

        case_geometry = {
            "estimated_interactions": str(interaction_estimate),
            "estimated_host_bytes": str(host_estimate),
            "estimated_device_bytes": str(gpu_estimate),
            **geometry,
        }
        all_cases.append(
            {
                "requested_depth": str(case.depth),
                "requested_particles": str(case.particles),
                "irregular": str(case.irregular).lower(),
                **case_geometry,
            }
        )
        all_cpu_rows.extend(add_case_columns(cpu_rows, case, case_geometry))
        all_cuda_rows.extend(add_case_columns(cuda_rows, case, case_geometry))
        write_csv(output_directory / "cases.csv", all_cases)
        write_csv(output_directory / "cpu_results.csv", all_cpu_rows)
        write_csv(output_directory / "cuda_results.csv", all_cuda_rows)

        cpu_winner = fastest_cpu_row(cpu_rows)
        print(
            f"  CPU winner: {cpu_winner['implementation']} "
            f"evaluation={float(cpu_winner['evaluation_s']) * 1.0e6:.3f} us"
        )
        winner = fastest_cuda_row(cuda_rows)
        if winner is not None:
            print(
                f"  CUDA winner: {winner['cuda_implementation']} "
                f"device_total={float(winner['device_total_s']) * 1.0e6:.3f} us "
                f"speedup={float(winner['total_speedup']):.3f}x"
            )

    if options.dry_run:
        print(
            f"Planned {len(cases)} cases; {len(skipped_cases)} would be "
            "skipped by current memory limits."
        )
        return 0

    print(f"Results written to {output_directory}")
    figures = generate_figures(
        all_cpu_rows, all_cuda_rows, output_directory
    )
    for figure in figures:
        print(f"Figure written to {figure}")
    print(
        f"Completed {len(all_cases)} of {len(cases)} cases "
        f"with {len(skipped_cases)} memory skips and {failures} failures."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
