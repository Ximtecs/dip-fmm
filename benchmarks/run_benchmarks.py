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
    "minimal": dict(sizes=[500], orders=[2, 4], depths=[2], evaluations=1, samples=1),
    "quick": dict(sizes=[500, 1000], orders=[2, 4], depths=[2, 3], evaluations=2, samples=2),
    "rough": dict(
        sizes=[20_000, 30_000],
        orders=[4],
        depths=[3, 4],
        evaluations=1,
        samples=1,
        warmups=1,
        accuracy_targets=128,
        scaling=False,
        comparison=False,
        workload_comparison=True,
        backends=[
            "cpu-direct",
            "cuda-direct",
            "cpu-static-matrix-mkl",
            "cpu-static-matrix",
            "cuda-partial",
            "cuda-full",
        ],
    ),
    "standard": dict(sizes=[10000, 20000, 30000, 40000], orders=[2, 4, 6, 8], depths=[2, 3, 4], evaluations=20, samples=5),
    "full": dict(sizes=[2**10, 2**12, 2**14, 2**16], orders=[2, 3, 4, 5, 6, 8], depths=[2, 3, 4, 5], evaluations=100, samples=7),
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

M2L_SUBPHASES = [
    ("m2l_gather", "Gather"),
    ("m2l_multiply", "Matrix multiply"),
    ("m2l_scatter", "Scatter"),
]

CUDA_P2P_PHASES = [
    ("cuda_p2p_h2d", "Moment/identity H2D"),
    ("cuda_p2p_kernel", "Sparse list-1 tensor"),
    ("cuda_p2p_d2h", "Field D2H"),
    ("cuda_p2p_wait", "Final host wait"),
]

BACKEND_LABELS = {
    "cpu-direct": "CPU direct P2P",
    "cuda-direct": "CUDA direct P2P",
    "cuda-partial": "Partial CUDA FMM",
    "cuda-full": "Full CUDA FMM",
    "cpu-static-matrix": "FMM static matrix",
    "cpu-static-matrix-mkl": "FMM static matrix + oneMKL",
}

DIRECT_BACKENDS = {"cpu-direct", "cuda-direct"}


def top_level_evaluation_phases(
    row: dict[str, str],
) -> list[tuple[str, str]]:
    """Return non-overlapping top-level phases for one backend row."""
    backend = row.get("execution_backend")
    if backend == "cuda-partial":
        overlapping_cuda = {"p2p", "cuda_h2d", "cuda_kernel", "cuda_d2h"}
        return [
            phase
            for phase in EVALUATION_PHASES
            if phase[0] not in overlapping_cuda
        ]
    if backend == "cuda-full":
        return [
            phase for phase in EVALUATION_PHASES
            if phase[0] != "cuda_kernel"
        ]
    return EVALUATION_PHASES


def nested_m2l_phases(row: dict[str, str]) -> list[tuple[str, str]]:
    """Return the diagnostic partition of the parent M2L phase."""
    if row.get("execution_backend") == "cuda-partial":
        return [
            ("cuda_m2l_h2d", "Multipole H2D"),
            *M2L_SUBPHASES,
            ("cuda_m2l_d2h", "Local D2H"),
        ]
    if row.get("execution_backend") == "cuda-full":
        return []
    return M2L_SUBPHASES


def independent_p2p_phases(
    row: dict[str, str],
) -> list[tuple[str, str]]:
    """Return the independently scheduled CUDA P2P lane."""
    if row.get("execution_backend") == "cuda-partial":
        return CUDA_P2P_PHASES
    return []


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
    direct_available: bool
    m2l_available: bool
    full_available: bool
    mkl_available: bool
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

    if profile.get("scaling", True):
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
    backends = ["cpu-direct"]
    if status.direct_available:
        backends.append("cuda-direct")
    if status.m2l_available:
        backends.append("cuda-partial")
    if status.full_available:
        backends.append("cuda-full")
    backends.append("cpu-static-matrix")
    if status.mkl_available:
        backends.append("cpu-static-matrix-mkl")
    return backends


def profile_backends(
    profile_name: str,
    profile: dict,
    available_backends: list[str],
) -> list[str]:
    """Resolve and validate the backend list requested by a profile."""
    requested_backends = profile.get("backends", available_backends)
    missing_backends = [
        backend
        for backend in requested_backends
        if backend not in available_backends
    ]
    if missing_backends:
        raise RuntimeError(
            f"Profile '{profile_name}' requires unavailable backends: "
            + ", ".join(missing_backends)
        )
    return requested_backends


def expanded_cases(cases: list[BenchmarkCase], backends: list[str]):
    """Expand FMM cases and order-independent direct/scaling cases."""
    tree_backends = [
        backend for backend in backends if backend not in DIRECT_BACKENDS
    ]
    planned = [
        (case, backend)
        for case in cases
        for backend in tree_backends
    ]

    # Direct all-to-all P2P depends only on the particle count. Use zero as the
    # explicit not-applicable value for expansion order and tree depth in CSV.
    direct_cases_by_size = {}
    for case in cases:
        if case.suite != "parameter_grid":
            continue
        direct_cases_by_size.setdefault(
            case.size,
            BenchmarkCase(
                suite="direct",
                size=case.size,
                order=0,
                depth=0,
                threads=case.threads,
                direct=case.direct,
            ),
        )
    planned.extend(
        (direct_cases_by_size[size], backend)
        for size in sorted(direct_cases_by_size)
        for backend in backends
        if backend in DIRECT_BACKENDS
    )
    maximum_threads = max(
        (case.threads for case in cases if case.suite == "parameter_grid"),
        default=0,
    )
    planned.extend(
        (
            BenchmarkCase(
                suite="scaling",
                size=case.size,
                order=0,
                depth=0,
                threads=case.threads,
                direct=False,
            ),
            backend,
        )
        for case in cases
        if case.suite == "scaling" and case.threads != maximum_threads
        for backend in backends
        if backend in DIRECT_BACKENDS
    )
    return planned


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
        direct_available=values.get("cuda_direct_available") == "1",
        m2l_available=(
            values.get("cuda_m2l_p2p_available",
                       values.get("cuda_m2l_available")) == "1"
        ),
        full_available=values.get("cuda_full_available") == "1",
        mkl_available=values.get("one_mkl_available") == "1",
        device=values.get("cuda_device", ""),
    )


def default_executable() -> Path:
    combined_executable = Path(
        "build-bench-all/benchmarks/benchmark_uniform_fmm"
    )
    if combined_executable.is_file():
        return combined_executable
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
        if row.get("execution_backend", "cpu-static-matrix")
        == "cpu-static-matrix"
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
        and np.isfinite(float(row["workload_1_total_median"]))
    ]
    if not candidates:
        raise ValueError("No completed comparison workload rows are available")

    largest_size = max(int(row["sources"]) for row in candidates)
    largest = [row for row in candidates if int(row["sources"]) == largest_size]
    return min(largest, key=lambda row: abs(int(row["order"]) - 4))


def case_description(row: dict[str, str]) -> str:
    backend = row.get("execution_backend", "cpu-static-matrix")
    if backend in DIRECT_BACKENDS:
        return (
            f"N={row['sources']}, threads={row['threads']}, "
            f"backend={backend}"
        )
    return (
        f"N={row['sources']}, order={row['order']}, depth={row['depth']}, "
        f"threads={row['threads']}, "
        f"backend={backend}"
    )


def invoke(executable: Path, output: Path, *, size: int, order: int, depth: int,
           threads: int, evaluations: int, samples: int, direct: bool,
           backend: str = "cpu-static-matrix",
           workload_comparison: bool = True,
           warmups: int = 1,
           accuracy_targets: int = 0) -> dict[str, str]:
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
               "--warmups", str(warmups), "--seed", "314159", "--backend", backend,
               "--output", str(output)]
    if accuracy_targets > 0:
        command.extend(["--accuracy-targets", str(accuracy_targets)])
    if not direct:
        command.append("--no-direct")
    if not workload_comparison:
        command.append("--no-workload-comparison")
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["MKL_NUM_THREADS"] = "1"
    subprocess.run(command, check=True, env=environment)
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
    backend = row.get("execution_backend", "cpu-static-matrix")
    if backend in DIRECT_BACKENDS:
        return (
            f"{row['suite']}_n{row['sources']}_t{row['threads']}_"
            f"{backend}.png"
        )
    return (
        f"{row['suite']}_n{row['sources']}_p{row['order']}_"
        f"d{row['depth']}_t{row['threads']}_"
        f"{backend}.png"
    )


def generate_phase_breakdown(row: dict[str, str], path: Path) -> None:
    top_level_phases = top_level_evaluation_phases(row)
    phase_names = [label for _, label in top_level_phases]
    phase_values = [float(row[column]) for column, _ in top_level_phases]
    recorded_phase_total = sum(phase_values)
    diagnostic_m2l_phases = nested_m2l_phases(row)
    m2l_names = [label for _, label in diagnostic_m2l_phases]
    m2l_values = [
        float(row[column])
        for column, _ in diagnostic_m2l_phases
    ]
    m2l_recorded_total = sum(m2l_values)
    diagnostic_p2p_phases = independent_p2p_phases(row)
    p2p_names = [label for _, label in diagnostic_p2p_phases]
    p2p_values = [float(row[column]) for column, _ in diagnostic_p2p_phases]
    p2p_recorded_total = sum(p2p_values)

    panel_count = (
        1 + bool(diagnostic_m2l_phases) + bool(diagnostic_p2p_phases)
    )
    figure, axes = plt.subplots(1, panel_count, figsize=(7.5 * panel_count, 6))
    axes = np.atleast_1d(axes)

    def draw_breakdown(
        axis, names, values, total, title, *, show_fraction=True
    ):
        bars = axis.barh(names, values)
        for bar, value in zip(bars, values):
            fraction = value / total if total else 0.0
            annotation = f"{value:.4g} s"
            if show_fraction:
                annotation += f" ({fraction:.1%})"
            axis.annotate(
                annotation,
                (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
            )
        if values and max(values) > 0.0:
            axis.set_xlim(0.0, max(values) * 1.45)
        axis.set_xlabel("Mean time per evaluation (s)")
        axis.set_title(title)

    draw_breakdown(
        axes[0],
        phase_names,
        phase_values,
        recorded_phase_total,
        "Top-level evaluation phases",
    )
    next_axis = 1
    if diagnostic_m2l_phases:
        draw_breakdown(
            axes[next_axis],
            m2l_names,
            m2l_values,
            m2l_recorded_total,
            "Nested M2L matrix sub-phases",
        )
        next_axis += 1
    if diagnostic_p2p_phases:
        draw_breakdown(
            axes[next_axis],
            p2p_names,
            p2p_values,
            p2p_recorded_total,
            "Independent CUDA P2P lane",
            show_fraction=False,
        )
    figure.suptitle(
        f"Evaluation timing: suite={row['suite']}\n{case_description(row)}"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    figure.savefig(path)
    plt.close(figure)


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
    tree = min(float(row["tree_total"]), setup)
    static_plan = min(
        float(row.get("static_plan_seconds", 0.0)),
        max(setup - tree, 0.0),
    )
    other_setup = max(setup - tree - static_plan, 0.0)
    evaluation = float(row["evaluation_median"])
    components = [
        ("Tree construction", tree),
        ("Static operator plan", static_plan),
        ("Other backend construction", other_setup),
        ("One evaluation", evaluation),
    ]

    positive_components = [item for item in components if item[1] > 0.0]
    labels = [item[0] for item in positive_components]
    values = [item[1] for item in positive_components]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.barh(labels, values)
    axis.bar_label(bars, labels=[f"{value:.4g} s" for value in values], padding=3)
    axis.set_xscale("log")
    axis.set_xlabel("Wall time (s, logarithmic scale)")
    axis.set_title(
        f"Construction components and one evaluation\n{case_description(row)}"
    )
    axis.margins(x=0.18)
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
                row.get("execution_backend", "cpu-static-matrix"),
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
                and row.get("execution_backend", "cpu-static-matrix") == backend
            ),
            key=lambda row: int(row["depth"]),
        )
        yield size, order, threads, backend, selected


def workload_values(
    rows: list[dict[str, str]],
    include_creation: bool,
) -> tuple[list[float], list[float]]:
    """Return total one- and ten-evaluation workload times."""
    component = "total" if include_creation else "evaluation"
    return (
        [float(row[f"workload_1_{component}_median"]) for row in rows],
        [float(row[f"workload_10_{component}_median"]) for row in rows],
    )


def generate_backend_workload_figures(
    rows: list[dict[str, str]],
    directory: Path,
    description: str,
    suffix: str = "",
) -> None:
    """Plot construction-inclusive and evaluation-only backend workloads."""
    methods = [
        BACKEND_LABELS.get(
            row["execution_backend"],
            row["execution_backend"],
        )
        for row in rows
    ]
    positions = np.arange(len(methods))
    width = 0.36
    workload_plots = [
        (
            True,
            "backend_workloads_with_creation",
            "Construction-inclusive workloads",
            "1 construction + 1 evaluation",
            "1 construction + 10 evaluations",
        ),
        (
            False,
            "backend_workloads_evaluation_only",
            "Evaluation-only workloads (construction excluded)",
            "1 evaluation",
            "10 evaluations",
        ),
    ]
    for (include_creation, stem, title, single_label,
         repeated_label) in workload_plots:
        single_values, repeated_values = workload_values(
            rows,
            include_creation,
        )
        plt.figure(figsize=(11, 5.5))
        plt.bar(
            positions - width / 2,
            single_values,
            width,
            label=single_label,
        )
        plt.bar(
            positions + width / 2,
            repeated_values,
            width,
            label=repeated_label,
        )
        plt.xticks(positions, methods, rotation=15, ha="right")
        plt.yscale("log")
        plt.ylabel("Median total wall time (s, logarithmic scale)")
        plt.title(f"{title}\n{description}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(directory / f"{stem}{suffix}.png")
        plt.close()


def generate_backend_amortisation_figure(
    rows: list[dict[str, str]],
    directory: Path,
    description: str,
    suffix: str = "",
) -> None:
    """Plot setup amortisation over long fixed-geometry evaluations."""
    counts = np.array([1, 10, 100, 1_000, 10_000], dtype=float)
    plt.figure(figsize=(9, 5))
    for row in rows:
        setup = float(row["fmm_setup_seconds"])
        evaluation = float(row["evaluation_median"])
        backend = row["execution_backend"]
        plt.loglog(
            counts,
            evaluation + setup / counts,
            "o-",
            label=BACKEND_LABELS.get(backend, backend),
        )
    plt.xlabel("Evaluations per fixed geometry")
    plt.ylabel("Amortised wall time per evaluation (s)")
    plt.title(f"Backend setup amortisation\n{description}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory / f"backend_amortisation{suffix}.png")
    plt.close()


def generate_figures(rows: list[dict[str, str]], figures: Path) -> None:
    grid = [row for row in rows if row["suite"] == "parameter_grid"]
    direct_rows = [row for row in rows if row["suite"] == "direct"]
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
        if row["suite"] != "direct":
            generate_work_breakdown(row, per_run_directories["work"] / filename)

    generate_phase_breakdown(
        representative,
        directories["phases"] / "representative.png",
    )

    comparison_rows = [
        row
        for row in rows
        if row["suite"] == "comparison"
    ]
    if comparison_rows:
        comparison = comparison_case(rows)
        comparison_rows.extend(
            row
            for row in direct_rows
            if row["sources"] == comparison["sources"]
        )
        comparison_description = (
            f"N={comparison['sources']}, order={comparison['order']}, "
            f"depth={comparison['depth']}, threads={comparison['threads']}"
        )
        generate_backend_workload_figures(
            comparison_rows,
            directories["comparison"],
            comparison_description,
        )
        generate_backend_amortisation_figure(
            comparison_rows,
            directories["comparison"],
            comparison_description,
        )
    else:
        workload_keys = sorted({
            (
                int(row["sources"]),
                int(row["order"]),
                int(row["depth"]),
                int(row["threads"]),
            )
            for row in grid
            if np.isfinite(float(row["workload_1_total_median"]))
        })
        for size, order, depth, threads in workload_keys:
            workload_rows = [
                row
                for row in grid
                if int(row["sources"]) == size
                and int(row["order"]) == order
                and int(row["depth"]) == depth
                and int(row["threads"]) == threads
            ]
            workload_rows.extend(
                row
                for row in direct_rows
                if int(row["sources"]) == size
                and int(row["threads"]) == threads
            )
            description = (
                f"N={size}, order={order}, depth={depth}, threads={threads}"
            )
            suffix = f"_n{size}_p{order}_d{depth}_t{threads}"
            generate_backend_workload_figures(
                workload_rows,
                directories["comparison"],
                description,
                suffix,
            )
            generate_backend_amortisation_figure(
                workload_rows,
                directories["comparison"],
                description,
                suffix,
            )

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
        # CPU direct is itself the all-to-all reference, so plotting it twice
        # would create two differently sampled curves for the same algorithm.
        if accurate and backend != "cpu-direct":
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
        # CPU direct is the reference itself, so a second curve would only
        # compare two timing samples of the same all-to-all implementation.
        if accurate and backend != "cpu-direct":
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
    for backend in sorted({row["execution_backend"] for row in direct_rows}):
        selected_direct = sorted(
            (
                row
                for row in direct_rows
                if row["execution_backend"] == backend
            ),
            key=lambda row: int(row["sources"]),
        )
        plt.figure()
        plt.loglog(
            [int(row["sources"]) for row in selected_direct],
            [float(row["evaluation_median"]) for row in selected_direct],
            "o-",
        )
        plt.xlabel("Sources and targets, N")
        plt.ylabel("Median evaluation time (s)")
        plt.title(
            f"Direct runtime/particles: threads={benchmark_threads}, "
            f"backend={backend}"
        )
        plt.tight_layout()
        plt.savefig(runtime_particle_directory / f"direct_{backend}.png")
        plt.close()

    orders = sorted({int(row["order"]) for row in grid})
    depths = sorted({int(row["depth"]) for row in grid})
    backends = sorted(
        {row.get("execution_backend", "cpu-static-matrix") for row in grid}
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
                and row.get("execution_backend", "cpu-static-matrix") == backend
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
        # CPU direct is the reference itself, so a second curve would only
        # compare two timing samples of the same all-to-all implementation.
        if accurate and backend != "cpu-direct":
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
                label="CPU direct all-to-all P2P",
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
            f"{row.get('execution_backend', 'cpu-static-matrix')}, "
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
    counts = np.array([1, 10, 100, 1_000, 10_000], dtype=float)
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
    if scaling_rows:
        scaling_size = max(int(row["sources"]) for row in scaling_rows)
        scaling_rows.extend(
            row
            for row in direct_rows
            if int(row["sources"]) == scaling_size
        )
    for backend in sorted(
        {
            row.get("execution_backend", "cpu-static-matrix")
            for row in scaling_rows
        }
    ):
        scaling = sorted(
            (
                row for row in scaling_rows
                if row.get("execution_backend", "cpu-static-matrix") == backend
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
    profile = PROFILES[args.profile]
    available_backends = benchmark_backends(cuda_status)
    backends = profile_backends(
        args.profile,
        profile,
        available_backends,
    )
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
    include_comparison = profile.get("comparison", True)
    workload_comparison = profile.get("workload_comparison", True)
    warmups = profile.get("warmups", 1)
    accuracy_targets = profile.get("accuracy_targets", 0)
    comparison_backends = [
        backend for backend in backends if backend not in DIRECT_BACKENDS
    ]
    total_cases = len(planned_cases) + (
        len(comparison_backends) if include_comparison else 0
    )
    print(
        f"Benchmark executable: {executable}\n"
        f"CUDA compiled={cuda_status.compiled}, "
        f"available={cuda_status.available}, "
        f"direct={cuda_status.direct_available}, "
        f"cuda-partial={cuda_status.m2l_available}, "
        f"full={cuda_status.full_available}, "
        f"oneMKL={cuda_status.mkl_available}, "
        f"device={cuda_status.device or 'none'}\n"
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
            workload_comparison=workload_comparison,
            warmups=warmups,
            accuracy_targets=accuracy_targets,
        )
        row["suite"] = case.suite
        rows.append(row)
        print(
            f"Overall {progress_bar(completed, total_cases)} "
            f"{completed}/{total_cases} tests complete",
            flush=True,
        )

    if include_comparison and comparison_backends:
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
        for backend in comparison_backends:
            completed += 1
            print(
                f"\nRunning test {completed}/{total_cases}: "
                f"suite=comparison, sources={comparison_size}, "
                f"targets={comparison_size}, order={comparison_order}, "
                f"depth={comparison_depth}, threads={max_threads}, "
                f"backend={backend}",
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
                workload_comparison=workload_comparison,
                warmups=warmups,
                accuracy_targets=accuracy_targets,
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
        f"cuda_direct_available={cuda_status.direct_available}\n"
        f"cuda_m2l_available={cuda_status.m2l_available}\n"
        f"cuda_m2l_p2p_available={cuda_status.m2l_available}\n"
        f"cuda_full_available={cuda_status.full_available}\n"
        f"one_mkl_available={cuda_status.mkl_available}\n"
        f"cuda_device={cuda_status.device}\n"
        f"backends={','.join(backends)}\n",
        encoding="utf-8",
    )
    generate_figures(rows, figures)
    representative = representative_case(rows)
    evaluation = float(representative["evaluation_median"])
    representative_phases = top_level_evaluation_phases(representative)
    phase_rows = []
    phase_total = sum(
        float(representative[column]) for column, _ in representative_phases
    )
    for column, label in representative_phases:
        seconds = float(representative[column])
        fraction = seconds / phase_total if phase_total else 0.0
        phase_rows.append(f"| {label} | {seconds:.6g} | {fraction:.1%} |")
    m2l_phase_rows = []
    representative_m2l_phases = nested_m2l_phases(representative)
    m2l_phase_total = sum(
        float(representative[column])
        for column, _ in representative_m2l_phases
    )
    for column, label in representative_m2l_phases:
        seconds = float(representative[column])
        fraction = seconds / m2l_phase_total if m2l_phase_total else 0.0
        m2l_phase_rows.append(
            f"| {label} | {seconds:.6g} | {fraction:.1%} |"
        )
    p2p_phase_rows = []
    representative_p2p_phases = independent_p2p_phases(representative)
    for column, label in representative_p2p_phases:
        seconds = float(representative[column])
        p2p_phase_rows.append(f"| {label} | {seconds:.6g} |")
    comparison_rows = [row for row in rows if row["suite"] == "comparison"]
    if comparison_rows:
        comparison = comparison_case(rows)
        comparison_rows.extend(
            row
            for row in rows
            if row["suite"] == "direct"
            and row["sources"] == comparison["sources"]
        )
        workload_rows = []
        for row in comparison_rows:
            workload_rows.append(
                f"| {row['execution_backend']} | {row['m2l_strategy']} | "
                f"{float(row['workload_1_total_median']):.6g} | "
                f"{float(row['workload_10_total_median']):.6g} | "
                f"{float(row['rms_relative_error']):.6g} |"
            )
        comparison_summary = (
            "## Creation/evaluation workloads\n\n"
            f"Comparison geometry: **N={comparison['sources']}, "
            f"order={comparison['order']}, depth={comparison['depth']}, "
            f"threads={comparison['threads']}**. Times are median totals and "
            "include construction.\n\n"
            "| Backend | Stored operator data | 1 creation + 1 evaluation "
            "(s) | 1 creation + 10 evaluations (s) | RMS relative error |\n"
            "|---|---|---:|---:|---:|\n"
            + "\n".join(workload_rows)
            + "\n\n"
        )
    else:
        rough_rows = []
        for row in rows:
            depth = "-" if row["suite"] == "direct" else row["depth"]
            rough_rows.append(
                f"| {row['sources']} | {depth} | "
                f"{row['execution_backend']} | "
                f"{float(row['fmm_setup_seconds']):.6g} | "
                f"{float(row['evaluation_median']):.6g} | "
                f"{float(row['workload_1_total_median']):.6g} | "
                f"{float(row['workload_10_total_median']):.6g} | "
                f"{row['accuracy_targets']} | "
                f"{float(row['rms_relative_error']):.6g} | "
                f"{float(row['max_relative_error']):.6g} |"
            )
        comparison_summary = (
            "## Rough backend sweep\n\n"
            "This profile deliberately omits the additional comparison, "
            "thread-scaling and the additional full-reference suite. Every "
            "process records setup separately, one warmed timed evaluation, "
            "a deterministic sampled direct-reference accuracy check, and "
            "independent 1+1 and 1+10 construction/evaluation workloads.\n\n"
            "| N | Depth | Backend | Setup (s) | Evaluation (s) | 1+1 total "
            "(s) | 1+10 total (s) | Accuracy targets | RMS relative error | "
            "Max relative error |\n"
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|\n"
            + "\n".join(rough_rows)
            + "\n\n"
        )

    (run_dir / "summary.md").write_text(
        "# Benchmark summary\n\n"
        f"- Profile: **{args.profile}**.\n"
        f"- Maximum thread count: **{max_threads}** of {available} logical CPUs.\n"
        f"- Representative case: **{case_description(representative)}**.\n"
        f"- Median complete evaluation: **{evaluation:.6g} s**.\n\n"
        + comparison_summary
        + "## Representative evaluation phases\n\n"
        "Shares are normalised over the recorded phase timers. The median "
        "complete evaluation above is measured independently.\n\n"
        "| Phase | Seconds per evaluation | Recorded share |\n"
        "|---|---:|---:|\n"
        + "\n".join(phase_rows)
        + "\n\n"
        "### Nested M2L matrix phases\n\n"
        "These rows partition the M2L parent phase and are not added to the "
        "top-level phase total.\n\n"
        "| M2L sub-phase | Seconds per evaluation | M2L recorded share |\n"
        "|---|---:|---:|\n"
        + "\n".join(m2l_phase_rows)
        + "\n\n"
        + ("### Independent CUDA P2P lane\n\n"
           "This lane overlaps the dependent far-field chain and is not "
           "added to the top-level phase total. The final host wait reports "
           "only residual blocking at the merge point.\n\n"
           "Device phases and host wait overlap, so they are diagnostic "
           "values rather than an additive partition.\n\n"
           "| P2P diagnostic | Seconds per evaluation |\n"
           "|---|---:|\n"
           + "\n".join(p2p_phase_rows)
           + "\n\n" if p2p_phase_rows else "")
        + "The full size/order/depth FMM sweep is stored with "
        "`suite=parameter_grid`; the one-per-size direct measurements use "
        "`suite=direct` with order and depth set to zero (not applicable). "
        "`figures/combined_overview.png` provides the "
        "top-level comparison. Figures are organised by result type under "
        "`runtime/`, `accuracy/`, `work/`, `phases/`, `setup/`, `scaling/`, "
        "and `comparison/`; per-run plots are in each applicable `per_run/` "
        "subdirectory.\n",
        encoding="utf-8",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
