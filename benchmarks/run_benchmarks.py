#!/usr/bin/env python3
"""Run deterministic uniform-FMM sweeps and generate performance figures."""

from __future__ import annotations

import argparse
import csv
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


def invoke(executable: Path, output: Path, *, size: int, order: int, depth: int,
           threads: int, evaluations: int, samples: int, direct: bool) -> dict[str, str]:
    command = [str(executable), "--sources", str(size), "--targets", str(size),
               "--order", str(order), "--depth", str(depth), "--threads", str(threads),
               "--evaluations", str(evaluations), "--samples", str(samples),
               "--warmups", "1", "--seed", "314159", "--output", str(output)]
    if not direct:
        command.append("--no-direct")
    subprocess.run(command, check=True)
    with output.open(newline="", encoding="utf-8") as stream:
        return next(csv.DictReader(stream))


def plot_line(rows, x, y, group, path, title, xlabel, ylabel, log=False):
    plt.figure()
    groups = sorted({row[group] for row in rows})
    for value in groups:
        selected = sorted((row for row in rows if row[group] == value), key=lambda row: float(row[x]))
        plt.plot([float(row[x]) for row in selected], [float(row[y]) for row in selected], "o-", label=f"{group}={value}")
    if log:
        plt.xscale("log")
        plt.yscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def generate_figures(rows: list[dict[str, str]], figures: Path) -> None:
    base = [row for row in rows if row["suite"] == "size_order"]
    plot_line(base, "sources", "evaluation_median", "order", figures / "runtime_vs_particles.png",
              "Uniform FMM runtime versus particle count", "Sources and targets", "Median evaluation time (s)", True)

    direct = [row for row in base if np.isfinite(float(row["direct_seconds"]))]
    plt.figure()
    selected = sorted((row for row in direct if int(row["order"]) == 4), key=lambda row: int(row["sources"]))
    if selected:
        x = [int(row["sources"]) for row in selected]
        plt.loglog(x, [float(row["evaluation_median"]) for row in selected], "o-", label="FMM")
        plt.loglog(x, [float(row["direct_seconds"]) for row in selected], "o-", label="Direct P2P")
    plt.title("Parallel direct P2P versus uniform FMM (order 4)")
    plt.xlabel("Sources and targets")
    plt.ylabel("Wall time (s)")
    plt.legend(); plt.tight_layout(); plt.savefig(figures / "direct_vs_fmm.png"); plt.close()

    largest = max(int(row["sources"]) for row in base)
    trade = [row for row in base if int(row["sources"]) == largest]
    plot_line(trade, "order", "evaluation_median", "depth", figures / "order_runtime.png",
              f"Expansion-order runtime tradeoff (N={largest})", "Expansion order", "Median evaluation time (s)")
    plot_line([row for row in trade if np.isfinite(float(row["rms_relative_error"]))], "order",
              "rms_relative_error", "depth", figures / "order_accuracy.png",
              f"Expansion-order accuracy tradeoff (N={largest})", "Expansion order", "RMS relative field error", True)

    representative = base[len(base) // 2]
    plt.figure()
    setup = float(representative["tree_total"]); evaluation = float(representative["evaluation_median"])
    plt.bar(["one shot"], [setup], label="tree setup")
    plt.bar(["one shot"], [evaluation], bottom=[setup], label="evaluation")
    plt.ylabel("Wall time (s)"); plt.title("One-shot setup and evaluation breakdown")
    plt.legend(); plt.tight_layout(); plt.savefig(figures / "one_shot_breakdown.png"); plt.close()

    phases = ["p2m", "m2m", "m2l", "l2l", "l2p", "p2p"]
    plt.figure(); bottom = 0.0
    for phase in phases:
        value = float(representative[phase]); plt.bar(["evaluation"], [value], bottom=[bottom], label=phase.upper()); bottom += value
    other = max(0.0, evaluation - bottom)
    plt.bar(["evaluation"], [other], bottom=[bottom], label="other")
    plt.ylabel("Wall time (s)"); plt.title("Evaluation phase breakdown")
    plt.legend(ncol=2); plt.tight_layout(); plt.savefig(figures / "evaluation_breakdown.png"); plt.close()

    counts = np.array([1, 2, 5, 10, 100], dtype=float)
    plt.figure(); plt.plot(counts, evaluation + setup / counts, "o-")
    plt.xscale("log"); plt.xlabel("Evaluations per geometry setup"); plt.ylabel("Amortised time per evaluation (s)")
    plt.title("Fixed-geometry repeated-evaluation amortisation"); plt.tight_layout(); plt.savefig(figures / "amortisation.png"); plt.close()

    scaling = sorted((row for row in rows if row["suite"] == "scaling"), key=lambda row: int(row["threads"]))
    if scaling:
        thread_values = np.array([int(row["threads"]) for row in scaling])
        times = np.array([float(row["evaluation_median"]) for row in scaling])
        speedup = times[0] / times
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(thread_values, times, "o-"); axes[0].set(xlabel="Threads", ylabel="Median time (s)", title="OpenMP runtime")
        axes[1].plot(thread_values, speedup, "o-", label="measured"); axes[1].plot(thread_values, thread_values, "--", label="ideal")
        axes[1].set(xlabel="Threads", ylabel="Speedup", title="OpenMP scaling"); axes[1].legend()
        figure.tight_layout(); figure.savefig(figures / "openmp_scaling.png"); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--executable", type=Path, default=Path("build-bench/benchmarks/benchmark_uniform_fmm"))
    parser.add_argument("--output-root", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = args.output_root / stamp
    figures = run_dir / "figures"; figures.mkdir(parents=True)
    rows = []
    temporary = run_dir / "latest.csv"
    available = os.cpu_count() or 1
    default_threads = available
    for size in profile["sizes"]:
        direct = size <= 5000
        for order in profile["orders"]:
            depth = profile["depths"][min(len(profile["depths"]) - 1, profile["sizes"].index(size))]
            row = invoke(args.executable, temporary, size=size, order=order, depth=depth,
                         threads=default_threads, evaluations=profile["evaluations"], samples=profile["samples"], direct=direct)
            row["suite"] = "size_order"; rows.append(row)
    scale_size = profile["sizes"][-1]
    thread_counts = []
    thread = 1
    while thread <= available:
        thread_counts.append(thread); thread *= 2
    if available not in thread_counts:
        thread_counts.append(available)
    for threads in thread_counts:
        row = invoke(args.executable, temporary, size=scale_size, order=4,
                     depth=profile["depths"][-1], threads=threads,
                     evaluations=profile["evaluations"], samples=profile["samples"], direct=False)
        row["suite"] = "scaling"; rows.append(row)
    temporary.unlink(missing_ok=True)
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (run_dir / "metadata.txt").write_text(
        f"profile={args.profile}\ndate={dt.datetime.now().isoformat()}\nhost={platform.node()}\nplatform={platform.platform()}\nlogical_cpus={available}\n",
        encoding="utf-8")
    generate_figures(rows, figures)
    representative = rows[len(rows) // 2]
    evaluation = float(representative["evaluation_median"])
    m2l_fraction = float(representative["m2l"]) / evaluation if evaluation else 0.0
    p2p_fraction = float(representative["p2p"]) / evaluation if evaluation else 0.0
    (run_dir / "summary.md").write_text(
        "# Benchmark summary\n\n"
        f"- Profile: **{args.profile}**.\n"
        f"- Representative measured M2L fraction: **{m2l_fraction:.1%}**.\n"
        f"- Representative measured near-field P2P fraction: **{p2p_fraction:.1%}**.\n"
        "- Inspect `results.csv` for setup, phase, accuracy, and work-counter details.\n",
        encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
