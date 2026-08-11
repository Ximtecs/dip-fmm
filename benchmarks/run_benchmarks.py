#!/usr/bin/env python3
"""Run deterministic uniform-FMM sweeps and generate performance figures."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import datetime as dt
import os
from pathlib import Path
import platform
import subprocess

import matplotlib.pyplot as plt
import numpy as np

PROFILES = {
    "quick": dict(sizes=[500, 1000], orders=[2, 4], depths=[2, 3], evaluations=2, samples=2),
    "standard": dict(sizes=[1000, 2000, 5000], orders=[2, 3, 4, 5, 6, 8], depths=[2, 3, 4], evaluations=20, samples=5),
    "full": dict(sizes=[1000, 5000, 20000, 50000], orders=[2, 3, 4, 5, 6, 8], depths=[3, 4, 5], evaluations=100, samples=7),
}

EVALUATION_PHASES = [
    ("moment_permutation", "Moment permutation"),
    ("multipole_reset", "Multipole reset"),
    ("p2m", "P2M"),
    ("m2m", "M2M"),
    ("local_reset", "Local reset"),
    ("l2l", "L2L"),
    ("m2l", "M2L"),
    ("l2p", "L2P"),
    ("p2p", "P2P"),
    ("result_unpermutation", "Result unpermutation"),
    ("cuda_h2d", "CUDA H2D"),
    ("cuda_kernel", "CUDA kernel"),
    ("cuda_d2h", "CUDA D2H"),
]


@dataclass(frozen=True)
class BenchmarkCase:
    suite: str
    size: int
    order: int
    depth: int
    threads: int
    direct: bool


@dataclass(frozen=True)
class CudaStatus:
    compiled: bool
    available: bool
    device: str


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def thread_counts_up_to(max_threads: int) -> list[int]:
    counts = []
    thread_count = 1
    while thread_count <= max_threads:
        counts.append(thread_count)
        thread_count *= 2
    if max_threads not in counts:
        counts.append(max_threads)
    return counts


def progress_bar(completed: int, total: int, width: int = 24) -> str:
    filled = width if total == 0 else width * completed // total
    return f"[{'=' * filled}{' ' * (width - filled)}]"


def build_cases(profile: dict, max_threads: int) -> list[BenchmarkCase]:
    cases = []
    for size in profile["sizes"]:
        for order in profile["orders"]:
            for depth in profile["depths"]:
                cases.append(
                    BenchmarkCase(
                        suite="parameter_grid",
                        size=size,
                        order=order,
                        depth=depth,
                        threads=max_threads,
                        direct=size <= 5000,
                    )
                )

    scale_size = profile["sizes"][-1]
    scale_order = min(profile["orders"], key=lambda order: abs(order - 4))
    scale_depth = profile["depths"][-1]
    for threads in thread_counts_up_to(max_threads):
        cases.append(
            BenchmarkCase(
                suite="scaling",
                size=scale_size,
                order=scale_order,
                depth=scale_depth,
                threads=threads,
                direct=False,
            )
        )

    return cases


def benchmark_backends(status: CudaStatus) -> list[str]:
    """Return every backend required for one benchmark geometry."""
    if status.compiled and not status.available:
        raise RuntimeError(
            "The benchmark was compiled with CUDA, but no CUDA device is "
            "available"
        )
    backends = ["cpu-static"]
    if status.compiled:
        backends.extend(["cuda-m2l", "cuda-full"])
    return backends


def expanded_cases(cases: list[BenchmarkCase], backends: list[str]):
    """Pair every planned geometry with every required execution backend."""
    return [(case, backend) for case in cases for backend in backends]


def probe_cuda(executable: Path) -> CudaStatus:
    if not executable.is_file():
        raise FileNotFoundError(f"Benchmark executable not found at {executable}")
    result = subprocess.run(
        [str(executable), "--cuda-status"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = dict(
        line.split("=", maxsplit=1)
        for line in result.stdout.splitlines()
        if "=" in line
    )
    return CudaStatus(
        compiled=values.get("cuda_compiled") == "1",
        available=values.get("cuda_available") == "1",
        device=values.get("cuda_device", ""),
    )


def default_executable() -> Path:
    cuda_executable = Path("build-cuda/benchmarks/benchmark_uniform_fmm")
    if cuda_executable.is_file():
        return cuda_executable
    return Path("build-bench/benchmarks/benchmark_uniform_fmm")


def representative_case(rows: list[dict[str, str]]) -> dict[str, str]:
    base = [row for row in rows if row["suite"] == "parameter_grid"]
    if not base:
        raise ValueError("No parameter-grid benchmark rows are available")
    cpu_rows = [
        row
        for row in base
        if row.get("execution_backend", "cpu-static") == "cpu-static"
    ]
    if cpu_rows:
        base = cpu_rows

    largest_size = max(int(row["sources"]) for row in base)
    candidates = [row for row in base if int(row["sources"]) == largest_size]
    nearest_order = min(
        (int(row["order"]) for row in candidates),
        key=lambda order: abs(order - 4),
    )
    candidates = [
        row for row in candidates if int(row["order"]) == nearest_order
    ]
    if "evaluation_median" in candidates[0]:
        return min(candidates, key=lambda row: float(row["evaluation_median"]))
    return min(candidates, key=lambda row: int(row.get("depth", 0)))


def comparison_case(rows: list[dict[str, str]]) -> dict[str, str]:
    candidates = [
        row
        for row in rows
        if row["suite"] == "comparison"
        and np.isfinite(float(row["p2p_1_total_median"]))
    ]
    if not candidates:
        raise ValueError("No direct-enabled benchmark rows are available")

    largest_size = max(int(row["sources"]) for row in candidates)
    largest = [row for row in candidates if int(row["sources"]) == largest_size]
    return min(largest, key=lambda row: abs(int(row["order"]) - 4))


def case_description(row: dict[str, str]) -> str:
    return (
        f"N={row['sources']}, order={row['order']}, depth={row['depth']}, "
        f"threads={row['threads']}, "
        f"backend={row.get('execution_backend', 'cpu-static')}"
    )


def invoke(executable: Path, output: Path, *, size: int, order: int, depth: int,
           threads: int, evaluations: int, samples: int, direct: bool,
           backend: str = "cpu-static",
           workload_comparison: bool = False) -> dict[str, str]:
    if not executable.is_file():
        raise FileNotFoundError(
            f"Benchmark executable not found at {executable}; run "
            "'cmake --build --preset benchmark' successfully first"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    command = [str(executable), "--sources", str(size), "--targets", str(size),
               "--order", str(order), "--depth", str(depth), "--threads", str(threads),
               "--evaluations", str(evaluations), "--samples", str(samples),
               "--warmups", "1", "--seed", "314159", "--backend", backend,
               "--output", str(output)]
    if not direct:
        command.append("--no-direct")
    if not workload_comparison:
        command.append("--no-workload-comparison")
    subprocess.run(command, check=True)
    if not output.is_file():
        raise RuntimeError(
            f"{executable} did not create {output}; rebuild the benchmark "
            "executable from the current source tree"
        )
    with output.open(newline="", encoding="utf-8") as stream:
        return next(csv.DictReader(stream))


def plot_line(rows, x, y, group, path, title, xlabel, ylabel, log=False):
    plt.figure()
    groups = sorted({row[group] for row in rows})
    for value in groups:
        selected = sorted(
            (row for row in rows if row[group] == value),
            key=lambda row: float(row[x]),
        )
        plt.plot(
            [float(row[x]) for row in selected],
            [float(row[y]) for row in selected],
            "o-",
            label=f"{group}={value}",
        )
    if log:
        plt.xscale("log")
        if all(float(row[y]) > 0.0 for row in rows):
            plt.yscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def phase_breakdown_filename(row: dict[str, str]) -> str:
    return (
        f"{row['suite']}_n{row['sources']}_p{row['order']}_"
        f"d{row['depth']}_t{row['threads']}_"
        f"{row.get('execution_backend', 'cpu-static')}.png"
    )


def generate_phase_breakdown(row: dict[str, str], path: Path) -> None:
    phase_names = [label for _, label in EVALUATION_PHASES]
    phase_values = [float(row[column]) for column, _ in EVALUATION_PHASES]
    recorded_phase_total = sum(phase_values)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(phase_names, phase_values)
    for bar, value in zip(bars, phase_values):
        fraction = value / recorded_phase_total if recorded_phase_total else 0.0
        plt.annotate(
            f"{value:.4g} s ({fraction:.1%})",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
        )
    if phase_values and max(phase_values) > 0.0:
        plt.xlim(0.0, max(phase_values) * 1.35)
    plt.xlabel("Mean phase time per evaluation (s)")
    plt.title(
        f"Evaluation phases: suite={row['suite']}\n{case_description(row)}"
    )
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def generate_work_breakdown(row: dict[str, str], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    values = [
        int(row["m2l_translations"]),
        int(row["near_field_pairs"]),
    ]
    labels = ["M2L translations", "Directed near-field pairs"]
    colours = ["tab:blue", "tab:orange"]
    for axis, value, label, colour in zip(axes, values, labels, colours):
        axis.bar([label], [value], color=colour)
        axis.bar_label(axis.containers[0], labels=[f"{value:,}"], padding=3)
        axis.set_ylabel("Count")
        axis.set_title(label)
    figure.suptitle(f"Geometry work\n{case_description(row)}")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    figure.savefig(path)
    plt.close(figure)


def generate_setup_breakdown(row: dict[str, str], path: Path) -> None:
    setup = float(row.get("fmm_setup_seconds", row["tree_total"]))
    evaluation = float(row["evaluation_median"])
    plt.figure(figsize=(7, 5))
    plt.bar(["one construction + one evaluation"], [setup], label="Tree setup")
    plt.bar(
        ["one construction + one evaluation"],
        [evaluation],
        bottom=[setup],
        label="Evaluation",
    )
    plt.ylabel("Wall time (s)")
    plt.title(f"Setup and evaluation\n{case_description(row)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def configuration_groups(rows: list[dict[str, str]]):
    keys = sorted(
        {
            (
                int(row["sources"]),
                int(row["order"]),
                int(row["threads"]),
                row.get("execution_backend", "cpu-static"),
            )
            for row in rows
        }
    )
    for size, order, threads, backend in keys:
        selected = sorted(
            (
                row
                for row in rows
                if int(row["sources"]) == size
                and int(row["order"]) == order
                and int(row["threads"]) == threads
                and row.get("execution_backend", "cpu-static") == backend
            ),
            key=lambda row: int(row["depth"]),
        )
        yield size, order, threads, backend, selected


def generate_figures(rows: list[dict[str, str]], figures: Path) -> None:
    grid = [row for row in rows if row["suite"] == "parameter_grid"]
    representative = representative_case(rows)
    benchmark_threads = representative["threads"]

    directories = {
        name: figures / name
        for name in [
            "accuracy",
            "comparison",
            "phases",
            "runtime",
            "scaling",
            "setup",
            "work",
        ]
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    per_run_directories = {}
    for name in ["phases", "setup", "work"]:
        per_run_directories[name] = directories[name] / "per_run"
        per_run_directories[name].mkdir(parents=True, exist_ok=True)
    for row in rows:
        filename = phase_breakdown_filename(row)
        generate_phase_breakdown(row, per_run_directories["phases"] / filename)
        generate_setup_breakdown(row, per_run_directories["setup"] / filename)
        generate_work_breakdown(row, per_run_directories["work"] / filename)

    generate_phase_breakdown(
        representative,
        directories["phases"] / "representative.png",
    )

    comparison = comparison_case(rows)
    methods = ["Direct all-to-all P2P", "Reference FMM", "Static FMM"]
    static_backend = comparison["static_multiply_backend"]
    methods[-1] += f" ({static_backend})"
    single_totals = [
        float(comparison["p2p_1_total_median"]),
        float(comparison["reference_1_total_median"]),
        float(comparison["static_1_total_median"]),
    ]
    repeated_totals = [
        float(comparison["p2p_10_total_median"]),
        float(comparison["reference_10_total_median"]),
        float(comparison["static_10_total_median"]),
    ]
    comparison_rows = [
        row
        for row in rows
        if row["suite"] == "comparison"
        and row.get("execution_backend", "cpu-static").startswith("cuda-")
    ]
    for row in comparison_rows:
        backend = row["execution_backend"]
        evaluation = float(row["evaluation_median"])
        setup = (
            float(row["amortised_seconds"]) - evaluation
        ) * int(row["evaluations"])
        methods.append(backend)
        single_totals.append(setup + evaluation)
        repeated_totals.append(setup + 10.0 * evaluation)
    positions = np.arange(len(methods))
    width = 0.36
    plt.figure(figsize=(9, 5))
    plt.bar(positions - width / 2, single_totals, width, label="1 creation + 1 evaluation")
    plt.bar(positions + width / 2, repeated_totals, width, label="1 creation + 10 evaluations")
    plt.xticks(positions, methods)
    plt.ylabel("Median total wall time (s)")
    plt.title(f"Evaluation strategy comparison\n{case_description(comparison)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(directories["comparison"] / "backend_workloads.png")
    plt.close()

    # Combined runtime, accuracy, and work views contain every grid case.
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for size, order, _, backend, selected in configuration_groups(grid):
        label = f"N={size}, p={order}, {backend}"
        depths = [int(row["depth"]) for row in selected]
        axes[0, 0].plot(
            depths,
            [float(row["evaluation_median"]) for row in selected],
            "o-",
            label=label,
        )
        accurate = [
            row for row in selected
            if np.isfinite(float(row["rms_relative_error"]))
        ]
        if accurate:
            axes[0, 1].plot(
                [int(row["depth"]) for row in accurate],
                [float(row["rms_relative_error"]) for row in accurate],
                "o-",
                label=label,
            )
        axes[1, 0].plot(
            depths,
            [int(row["m2l_translations"]) for row in selected],
            "o-",
            label=label,
        )
        axes[1, 1].plot(
            depths,
            [int(row["near_field_pairs"]) for row in selected],
            "o-",
            label=label,
        )
    axes[0, 0].set(
        title="Runtime",
        xlabel="Tree depth",
        ylabel="Median evaluation time (s)",
        yscale="log",
    )
    axes[0, 1].set(
        title="Accuracy",
        xlabel="Tree depth",
        ylabel="RMS relative field error",
        yscale="log",
    )
    axes[1, 0].set(
        title="Far-field work",
        xlabel="Tree depth",
        ylabel="M2L translations",
        yscale="log",
    )
    axes[1, 1].set(
        title="Near-field work",
        xlabel="Tree depth",
        ylabel="Directed pairs",
        yscale="log",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
    )
    figure.suptitle(
        f"All parameter-grid runs (threads={benchmark_threads})"
    )
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.96))
    figure.savefig(figures / "combined_overview.png")
    plt.close(figure)

    runtime_depth_directory = directories["runtime"] / "depth_sweeps"
    accuracy_depth_directory = directories["accuracy"] / "depth_sweeps"
    work_depth_directory = directories["work"] / "depth_sweeps"
    for directory in [
        runtime_depth_directory,
        accuracy_depth_directory,
        work_depth_directory,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    for size, order, threads, backend, selected in configuration_groups(grid):
        depths = [int(row["depth"]) for row in selected]
        stem = f"n{size}_p{order}_t{threads}_{backend}.png"

        plt.figure()
        plt.plot(
            depths,
            [float(row["evaluation_median"]) for row in selected],
            "o-",
        )
        plt.yscale("log")
        plt.xlabel("Tree depth")
        plt.ylabel("Median evaluation time (s)")
        plt.title(
            f"Depth/runtime\nN={size}, order={order}, threads={threads}, "
            f"backend={backend}"
        )
        plt.tight_layout()
        plt.savefig(runtime_depth_directory / stem)
        plt.close()

        accurate = [
            row for row in selected
            if np.isfinite(float(row["rms_relative_error"]))
        ]
        if accurate:
            plt.figure()
            plt.plot(
                [int(row["depth"]) for row in accurate],
                [float(row["rms_relative_error"]) for row in accurate],
                "o-",
            )
            plt.yscale("log")
            plt.xlabel("Tree depth")
            plt.ylabel("RMS relative field error")
            plt.title(
                f"Depth/accuracy\nN={size}, order={order}, threads={threads}, "
                f"backend={backend}"
            )
            plt.tight_layout()
            plt.savefig(accuracy_depth_directory / stem)
            plt.close()

        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(
            depths,
            [int(row["m2l_translations"]) for row in selected],
            "o-",
        )
        axes[0].set(
            xlabel="Tree depth",
            ylabel="M2L translations",
            title="Far-field work",
        )
        axes[1].plot(
            depths,
            [int(row["near_field_pairs"]) for row in selected],
            "o-",
            color="tab:orange",
        )
        axes[1].set(
            xlabel="Tree depth",
            ylabel="Directed pair count",
            title="Near-field work",
        )
        figure.suptitle(
            f"Depth/work balance: N={size}, order={order}, threads={threads}, "
            f"backend={backend}"
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
        figure.savefig(work_depth_directory / stem)
        plt.close(figure)

    # Particle-count plots are separated by order and depth to avoid mixing
    # configurations that perform different amounts of work.
    runtime_particle_directory = directories["runtime"] / "particle_sweeps"
    direct_directory = directories["comparison"] / "direct_vs_fmm"
    runtime_particle_directory.mkdir(parents=True, exist_ok=True)
    direct_directory.mkdir(parents=True, exist_ok=True)
    orders = sorted({int(row["order"]) for row in grid})
    depths = sorted({int(row["depth"]) for row in grid})
    backends = sorted(
        {row.get("execution_backend", "cpu-static") for row in grid}
    )
    configurations = [
        (backend, order, depth)
        for backend in backends
        for order in orders
        for depth in depths
    ]
    for backend, order, depth in configurations:
        selected = sorted(
            (
                row for row in grid
                if int(row["order"]) == order
                and int(row["depth"]) == depth
                and row.get("execution_backend", "cpu-static") == backend
            ),
            key=lambda row: int(row["sources"]),
        )
        stem = f"p{order}_d{depth}_t{benchmark_threads}_{backend}.png"
        plt.figure()
        plt.loglog(
            [int(row["sources"]) for row in selected],
            [float(row["evaluation_median"]) for row in selected],
            "o-",
        )
        plt.xlabel("Sources and targets, N")
        plt.ylabel("Median evaluation time (s)")
        plt.title(
            f"Runtime/particles: order={order}, depth={depth}, "
            f"threads={benchmark_threads}, backend={backend}"
        )
        plt.tight_layout()
        plt.savefig(runtime_particle_directory / stem)
        plt.close()

        accurate = [
            row for row in selected
            if np.isfinite(float(row["direct_seconds"]))
        ]
        if accurate:
            plt.figure()
            plt.loglog(
                [int(row["sources"]) for row in accurate],
                [float(row["evaluation_median"]) for row in accurate],
                "o-",
                label=backend,
            )
            plt.loglog(
                [int(row["sources"]) for row in accurate],
                [float(row["direct_seconds"]) for row in accurate],
                "o-",
                label="Direct all-to-all P2P",
            )
            plt.xlabel("Sources and targets, N")
            plt.ylabel("Wall time (s)")
            plt.title(
                f"Direct/FMM: order={order}, depth={depth}, "
                f"threads={benchmark_threads}, backend={backend}"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(direct_directory / stem)
            plt.close()

    largest = max(int(row["sources"]) for row in grid)
    trade = []
    for row in grid:
        if int(row["sources"]) != largest:
            continue
        labelled = dict(row)
        labelled["series"] = (
            f"{row.get('execution_backend', 'cpu-static')}, "
            f"depth={row['depth']}"
        )
        trade.append(labelled)
    plot_line(
        trade,
        "order",
        "evaluation_median",
        "series",
        directories["runtime"] / "order_runtime_combined.png",
        f"Order/runtime tradeoff (N={largest}, threads={benchmark_threads})",
        "Expansion order",
        "Median evaluation time (s)",
    )
    accurate_trade = [
        row for row in trade
        if np.isfinite(float(row["rms_relative_error"]))
    ]
    if accurate_trade:
        plot_line(
            accurate_trade,
            "order",
            "rms_relative_error",
            "series",
            directories["accuracy"] / "order_accuracy_combined.png",
            f"Order/accuracy tradeoff (N={largest}, threads={benchmark_threads})",
            "Expansion order",
            "RMS relative field error",
            True,
        )

    setup = float(
        representative.get("fmm_setup_seconds", representative["tree_total"])
    )
    evaluation = float(representative["evaluation_median"])
    counts = np.array([1, 2, 5, 10, 100], dtype=float)
    plt.figure()
    plt.plot(counts, evaluation + setup / counts, "o-")
    plt.xscale("log")
    plt.xlabel("Evaluations per geometry setup")
    plt.ylabel("Amortised time per evaluation (s)")
    plt.title(f"Fixed-geometry amortisation\n{case_description(representative)}")
    plt.tight_layout()
    plt.savefig(directories["setup"] / "amortisation_representative.png")
    plt.close()

    scaling_rows = [row for row in rows if row["suite"] == "scaling"]
    for backend in sorted(
        {row.get("execution_backend", "cpu-static") for row in scaling_rows}
    ):
        scaling = sorted(
            (
                row for row in scaling_rows
                if row.get("execution_backend", "cpu-static") == backend
            ),
            key=lambda row: int(row["threads"]),
        )
        scaling_description = (
            f"N={scaling[0]['sources']}, order={scaling[0]['order']}, "
            f"depth={scaling[0]['depth']}, backend={backend}"
        )
        thread_values = np.array([int(row["threads"]) for row in scaling])
        times = np.array([float(row["evaluation_median"]) for row in scaling])
        speedup = times[0] / times
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(thread_values, times, "o-")
        axes[0].set(
            xlabel="Threads",
            ylabel="Median time (s)",
            title="OpenMP runtime",
        )
        axes[1].plot(thread_values, speedup, "o-", label="measured")
        axes[1].plot(thread_values, thread_values, "--", label="ideal")
        axes[1].set(
            xlabel="Threads",
            ylabel="Speedup",
            title="OpenMP scaling",
        )
        axes[1].legend()
        figure.suptitle(f"Thread scaling ({scaling_description})")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        figure.savefig(
            directories["scaling"] / f"thread_scaling_{backend}.png"
        )
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument(
        "--executable",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmark_results"),
    )
    parser.add_argument(
        "--max-threads",
        type=positive_int,
        default=None,
        help="cap both benchmark evaluations and the thread-scaling sweep",
    )
    args = parser.parse_args()
    executable = args.executable or default_executable()
    cuda_status = probe_cuda(executable)
    backends = benchmark_backends(cuda_status)
    profile = PROFILES[args.profile]
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = args.output_root / stamp
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows = []
    temporary = run_dir / "latest.csv"
    available = os.cpu_count() or 1
    max_threads = min(args.max_threads or available, available)
    cases = build_cases(profile, max_threads)
    planned_cases = expanded_cases(cases, backends)
    total_cases = len(planned_cases) + len(backends)
    print(
        f"Benchmark executable: {executable}\n"
        f"CUDA compiled={cuda_status.compiled}, "
        f"available={cuda_status.available}, device={cuda_status.device or 'none'}\n"
        f"Backends: {', '.join(backends)}\n"
        f"Planned {total_cases} benchmark tests; maximum threads={max_threads}.",
        flush=True,
    )

    completed = 0
    for case, backend in planned_cases:
        completed += 1
        print(
            f"\nRunning test {completed}/{total_cases}: suite={case.suite}, "
            f"sources={case.size}, targets={case.size}, order={case.order}, "
            f"depth={case.depth}, threads={case.threads}, backend={backend}",
            flush=True,
        )
        row = invoke(
            executable,
            temporary,
            size=case.size,
            order=case.order,
            depth=case.depth,
            threads=case.threads,
            evaluations=profile["evaluations"],
            samples=profile["samples"],
            direct=case.direct,
            backend=backend,
            workload_comparison=False,
        )
        row["suite"] = case.suite
        rows.append(row)
        print(
            f"Overall {progress_bar(completed, total_cases)} "
            f"{completed}/{total_cases} tests complete",
            flush=True,
        )

    direct_sizes = [size for size in profile["sizes"] if size <= 5000]
    comparison_size = min(direct_sizes)
    comparison_size_index = profile["sizes"].index(comparison_size)
    comparison_depth = profile["depths"][
        min(len(profile["depths"]) - 1, comparison_size_index)
    ]
    comparison_order = min(
        profile["orders"],
        key=lambda order: abs(order - 4),
    )
    for backend in backends:
        completed += 1
        print(
            f"\nRunning test {completed}/{total_cases}: suite=comparison, "
            f"sources={comparison_size}, targets={comparison_size}, "
            f"order={comparison_order}, depth={comparison_depth}, "
            f"threads={max_threads}, backend={backend}",
            flush=True,
        )
        comparison_row = invoke(
            executable,
            temporary,
            size=comparison_size,
            order=comparison_order,
            depth=comparison_depth,
            threads=max_threads,
            evaluations=profile["evaluations"],
            samples=profile["samples"],
            direct=True,
            backend=backend,
            workload_comparison=backend == "cpu-static",
        )
        comparison_row["suite"] = "comparison"
        rows.append(comparison_row)
        print(
            f"Overall {progress_bar(completed, total_cases)} "
            f"{completed}/{total_cases} tests complete",
            flush=True,
        )
    temporary.unlink(missing_ok=True)
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "metadata.txt").write_text(
        f"profile={args.profile}\n"
        f"date={dt.datetime.now().isoformat()}\n"
        f"host={platform.node()}\n"
        f"platform={platform.platform()}\n"
        f"logical_cpus={available}\n"
        f"max_threads={max_threads}\n"
        f"cuda_compiled={cuda_status.compiled}\n"
        f"cuda_available={cuda_status.available}\n"
        f"cuda_device={cuda_status.device}\n"
        f"backends={','.join(backends)}\n",
        encoding="utf-8",
    )
    generate_figures(rows, figures)
    representative = representative_case(rows)
    comparison = comparison_case(rows)
    evaluation = float(representative["evaluation_median"])
    phase_rows = []
    phase_total = sum(
        float(representative[column]) for column, _ in EVALUATION_PHASES
    )
    for column, label in EVALUATION_PHASES:
        seconds = float(representative[column])
        fraction = seconds / phase_total if phase_total else 0.0
        phase_rows.append(f"| {label} | {seconds:.6g} | {fraction:.1%} |")

    (run_dir / "summary.md").write_text(
        "# Benchmark summary\n\n"
        f"- Profile: **{args.profile}**.\n"
        f"- Maximum thread count: **{max_threads}** of {available} logical CPUs.\n"
        f"- Representative case: **{case_description(representative)}**.\n"
        f"- Median complete evaluation: **{evaluation:.6g} s**.\n\n"
        "## Creation/evaluation workloads\n\n"
        f"Comparison case: **{case_description(comparison)}**. Static multiply "
        f"backend: **{comparison['static_multiply_backend']}**. Times are median "
        "totals and include construction where applicable. Direct all-to-all "
        "P2P has no reusable construction phase.\n\n"
        "| Method | 1 creation + 1 evaluation (s) | 1 creation + 10 evaluations (s) | RMS relative error |\n"
        "|---|---:|---:|---:|\n"
        f"| Direct all-to-all P2P | "
        f"{float(comparison['p2p_1_total_median']):.6g} | "
        f"{float(comparison['p2p_10_total_median']):.6g} | 0 |\n"
        f"| Reference FMM | {float(comparison['reference_1_total_median']):.6g} | "
        f"{float(comparison['reference_10_total_median']):.6g} | "
        f"{float(comparison['reference_rms_relative_error']):.6g} |\n"
        f"| Static FMM | {float(comparison['static_1_total_median']):.6g} | "
        f"{float(comparison['static_10_total_median']):.6g} | "
        f"{float(comparison['rms_relative_error']):.6g} |\n\n"
        "## Representative evaluation phases\n\n"
        "Shares are normalised over the recorded phase timers. The median "
        "complete evaluation above is measured independently.\n\n"
        "| Phase | Seconds per evaluation | Recorded share |\n"
        "|---|---:|---:|\n"
        + "\n".join(phase_rows)
        + "\n\n"
        "The full size/order/depth sweep is stored with "
        "`suite=parameter_grid`. `figures/combined_overview.png` provides the "
        "top-level comparison. Figures are organised by result type under "
        "`runtime/`, `accuracy/`, `work/`, `phases/`, `setup/`, `scaling/`, "
        "and `comparison/`; per-run plots are in each applicable `per_run/` "
        "subdirectory.\n",
        encoding="utf-8",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
