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
    "quick": dict(sizes=[100, 300], orders=[2, 4], depths=[2, 3], evaluations=2, samples=2),
    "standard": dict(sizes=[500, 2000, 5000], orders=[2, 3, 4, 5, 6, 8], depths=[2, 3, 4], evaluations=20, samples=5),
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
]


@dataclass(frozen=True)
class BenchmarkCase:
    suite: str
    size: int
    order: int
    depth: int
    threads: int
    direct: bool


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
    for size_index, size in enumerate(profile["sizes"]):
        depth_index = min(len(profile["depths"]) - 1, size_index)
        depth = profile["depths"][depth_index]
        for order in profile["orders"]:
            cases.append(
                BenchmarkCase(
                    suite="size_order",
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
    for depth in profile["depths"]:
        cases.append(
            BenchmarkCase(
                suite="depth",
                size=scale_size,
                order=scale_order,
                depth=depth,
                threads=max_threads,
                direct=scale_size <= 5000,
            )
        )

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


def representative_case(rows: list[dict[str, str]]) -> dict[str, str]:
    base = [row for row in rows if row["suite"] == "size_order"]
    if not base:
        raise ValueError("No size/order benchmark rows are available")

    largest_size = max(int(row["sources"]) for row in base)
    candidates = [row for row in base if int(row["sources"]) == largest_size]
    return min(candidates, key=lambda row: abs(int(row["order"]) - 4))


def case_description(row: dict[str, str]) -> str:
    return (
        f"N={row['sources']}, order={row['order']}, depth={row['depth']}, "
        f"threads={row['threads']}"
    )


def invoke(executable: Path, output: Path, *, size: int, order: int, depth: int,
           threads: int, evaluations: int, samples: int, direct: bool) -> dict[str, str]:
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
               "--warmups", "1", "--seed", "314159", "--output", str(output)]
    if not direct:
        command.append("--no-direct")
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
        f"d{row['depth']}_t{row['threads']}.png"
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


def generate_figures(rows: list[dict[str, str]], figures: Path) -> None:
    base = [row for row in rows if row["suite"] == "size_order"]
    representative = representative_case(rows)
    benchmark_threads = representative["threads"]

    plot_line(
        base,
        "sources",
        "evaluation_median",
        "order",
        figures / "runtime_vs_particles.png",
        f"Uniform FMM runtime (threads={benchmark_threads}; depth varies by N)",
        "Sources and targets, N",
        "Median evaluation time (s)",
        True,
    )

    direct = [row for row in base if np.isfinite(float(row["direct_seconds"]))]
    plt.figure()
    selected = sorted(
        (row for row in direct if int(row["order"]) == 4),
        key=lambda row: int(row["sources"]),
    )
    if selected:
        x = [int(row["sources"]) for row in selected]
        plt.loglog(
            x,
            [float(row["evaluation_median"]) for row in selected],
            "o-",
            label="FMM",
        )
        plt.loglog(
            x,
            [float(row["direct_seconds"]) for row in selected],
            "o-",
            label="Direct P2P",
        )
        for row in selected:
            plt.annotate(
                f"depth={row['depth']}",
                (int(row["sources"]), float(row["evaluation_median"])),
                textcoords="offset points",
                xytext=(4, 4),
            )
    plt.title(f"Direct P2P versus FMM (order=4, threads={benchmark_threads})")
    plt.xlabel("Sources and targets, N")
    plt.ylabel("Wall time (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "direct_vs_fmm.png")
    plt.close()

    largest = max(int(row["sources"]) for row in base)
    trade = [row for row in base if int(row["sources"]) == largest]
    trade_depth = trade[0]["depth"]
    plot_line(
        trade,
        "order",
        "evaluation_median",
        "depth",
        figures / "order_runtime.png",
        f"Order/runtime tradeoff (N={largest}, depth={trade_depth}, threads={benchmark_threads})",
        "Expansion order",
        "Median evaluation time (s)",
    )
    plot_line(
        [
            row
            for row in trade
            if np.isfinite(float(row["rms_relative_error"]))
        ],
        "order",
        "rms_relative_error",
        "depth",
        figures / "order_accuracy.png",
        f"Order/accuracy tradeoff (N={largest}, depth={trade_depth}, threads={benchmark_threads})",
        "Expansion order",
        "RMS relative field error",
        True,
    )

    plt.figure()
    setup = float(representative["tree_total"])
    evaluation = float(representative["evaluation_median"])
    plt.bar(["one shot"], [setup], label="tree setup")
    plt.bar(["one shot"], [evaluation], bottom=[setup], label="evaluation")
    plt.ylabel("Wall time (s)")
    plt.title(f"One-shot setup and evaluation\n{case_description(representative)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "one_shot_breakdown.png")
    plt.close()

    generate_phase_breakdown(
        representative,
        figures / "evaluation_breakdown.png",
    )
    breakdown_directory = figures / "evaluation_breakdowns"
    breakdown_directory.mkdir(parents=True, exist_ok=True)
    for row in rows:
        generate_phase_breakdown(
            row,
            breakdown_directory / phase_breakdown_filename(row),
        )

    counts = np.array([1, 2, 5, 10, 100], dtype=float)
    plt.figure()
    plt.plot(counts, evaluation + setup / counts, "o-")
    plt.xscale("log")
    plt.xlabel("Evaluations per geometry setup")
    plt.ylabel("Amortised time per evaluation (s)")
    plt.title(f"Fixed-geometry amortisation\n{case_description(representative)}")
    plt.tight_layout()
    plt.savefig(figures / "amortisation.png")
    plt.close()

    depth_rows = sorted(
        (row for row in rows if row["suite"] == "depth"),
        key=lambda row: int(row["depth"]),
    )
    if depth_rows:
        depth_description = (
            f"N={depth_rows[0]['sources']}, order={depth_rows[0]['order']}, "
            f"threads={depth_rows[0]['threads']}"
        )
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        depths = [int(row["depth"]) for row in depth_rows]
        axes[0].plot(
            depths,
            [float(row["evaluation_median"]) for row in depth_rows],
            "o-",
        )
        axes[0].set(
            xlabel="Tree depth",
            ylabel="Median evaluation time (s)",
            title="Runtime",
        )
        accurate_rows = [
            row
            for row in depth_rows
            if np.isfinite(float(row["direct_seconds"]))
        ]
        if accurate_rows:
            axes[1].semilogy(
                [int(row["depth"]) for row in accurate_rows],
                [float(row["rms_relative_error"]) for row in accurate_rows],
                "o-",
            )
            axes[1].set(
                xlabel="Tree depth",
                ylabel="RMS relative field error",
                title="Accuracy",
            )
        else:
            axes[1].text(
                0.5,
                0.5,
                "Direct reference disabled for this particle count",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
            axes[1].set_axis_off()
        figure.suptitle(f"Tree-depth sweep ({depth_description})")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        figure.savefig(figures / "depth_tradeoff.png")
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(
            depths,
            [int(row["m2l_translations"]) for row in depth_rows],
            "o-",
            label="M2L translations",
        )
        axes[0].set(
            xlabel="Tree depth",
            ylabel="Translation count",
            title="Far-field work",
        )
        axes[1].plot(
            depths,
            [int(row["near_field_pairs"]) for row in depth_rows],
            "o-",
            color="tab:orange",
            label="near-field pairs",
        )
        axes[1].set(
            xlabel="Tree depth",
            ylabel="Directed pair count",
            title="Near-field work",
        )
        figure.suptitle(f"Tree-depth work balance ({depth_description})")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        figure.savefig(figures / "depth_work.png")
        plt.close(figure)

    scaling = sorted(
        (row for row in rows if row["suite"] == "scaling"),
        key=lambda row: int(row["threads"]),
    )
    if scaling:
        scaling_description = (
            f"N={scaling[0]['sources']}, order={scaling[0]['order']}, "
            f"depth={scaling[0]['depth']}"
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
        figure.savefig(figures / "openmp_scaling.png")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("build-bench/benchmarks/benchmark_uniform_fmm"),
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
    total_cases = len(cases)
    print(
        f"Planned {total_cases} benchmark tests; maximum threads={max_threads}.",
        flush=True,
    )

    for case_index, case in enumerate(cases, start=1):
        print(
            f"\nRunning test {case_index}/{total_cases}: suite={case.suite}, "
            f"sources={case.size}, targets={case.size}, order={case.order}, "
            f"depth={case.depth}, threads={case.threads}",
            flush=True,
        )
        row = invoke(
            args.executable,
            temporary,
            size=case.size,
            order=case.order,
            depth=case.depth,
            threads=case.threads,
            evaluations=profile["evaluations"],
            samples=profile["samples"],
            direct=case.direct,
        )
        row["suite"] = case.suite
        rows.append(row)
        print(
            f"Overall {progress_bar(case_index, total_cases)} "
            f"{case_index}/{total_cases} tests complete",
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
        f"max_threads={max_threads}\n",
        encoding="utf-8",
    )
    generate_figures(rows, figures)
    representative = representative_case(rows)
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
        "## Representative evaluation phases\n\n"
        "Shares are normalised over the recorded phase timers. The median "
        "complete evaluation above is measured independently.\n\n"
        "| Phase | Seconds per evaluation | Recorded share |\n"
        "|---|---:|---:|\n"
        + "\n".join(phase_rows)
        + "\n\n"
        "The depth sweep is stored with `suite=depth`; inspect `depth_tradeoff.png` "
        "and `depth_work.png` for its accuracy/runtime and work-count tradeoffs. "
        "Per-case phase plots are stored under `figures/evaluation_breakdowns/`.\n",
        encoding="utf-8",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
