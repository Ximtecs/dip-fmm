#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Figures for a jaxFMM-versus-dip-fmm campaign from ``best.csv``.

Style follows the FMM3D campaign's preliminary figures (recessive grid, small
type, frameless legend, one measure per axis). Series colours are fixed per
framework and never reassigned: jaxFMM orange, dip-fmm point blue, dip-fmm
finite green, FMM3D grey; secondary arms reuse their framework's hue with a
different marker and dash so identity is never colour alone.

Optional FMM3D CPU rows (``--fmm3d-csv``) are drawn in grey and labelled as
CPU so nobody reads them as a GPU result.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e7e6e2"
COLOUR = {"jaxfmm": "#F58518", "dipfmm_point": "#2a78d6", "dipfmm_finite": "#1baf7a", "fmm3d": "#9c9c9c"}

#: Fixed series specification: label, colour key, marker, linestyle.
SERIES = {
    "jaxfmm kifmm p6": ("jaxFMM KIFMM p=6 (GPU)", "jaxfmm", "o", "-"),
    "jaxfmm kifmm p4": ("jaxFMM KIFMM p=4 (GPU)", "jaxfmm", "v", ":"),
    "jaxfmm kifmm p8": ("jaxFMM KIFMM p=8 (GPU)", "jaxfmm", "^", "--"),
    "dipfmm point->point o6": ("dip-fmm order 6, point targets (GPU)", "dipfmm_point", "s", "-"),
    "dipfmm point->point o10": ("dip-fmm order 10, point targets (GPU)", "dipfmm_point", "D", "--"),
    "dipfmm prism->prism o6": ("dip-fmm order 6, prism sources and targets (GPU)", "dipfmm_finite", "P", "-"),
}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK_PRIMARY, "text.color": INK_PRIMARY,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False,
        "lines.linewidth": 1.8, "lines.markersize": 5.5, "figure.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def load_best(path: Path) -> dict[str, list[dict]]:
    series: dict[str, list[dict]] = defaultdict(list)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            series[row["arm"]].append(row)
    for rows in series.values():
        rows.sort(key=lambda r: int(r["n_bodies"]))
    return series


def load_fmm3d(path: Path | None) -> dict[str, list[tuple[int, float, float]]]:
    """FMM3D rows from the Article1 processed CSV: eps -> [(N, seconds, rel L2)]."""
    out: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    if path is None or not path.is_file():
        return out
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row.get("framework") != "fmm3d" or row.get("status") != "success":
                continue
            eps = row.get("case_fmm3d_eps") or row.get("accuracy_control")
            try:
                out[str(eps)].append((int(row["n_bodies"]), float(row["external_evaluation_median_seconds"]),
                                      float(row["relative_l2"])))
            except (KeyError, ValueError):
                continue
    for rows in out.values():
        rows.sort()
    return out


def values(rows: list[dict], column: str, scale: float = 1.0):
    xs, ys = [], []
    for row in rows:
        value = row.get(column)
        if value in (None, ""):
            continue
        xs.append(int(row["n_bodies"]))
        ys.append(float(value) * scale)
    return xs, ys


def plot_series(ax, series, column, scale=1.0, only=None, label_suffix=""):
    for arm, rows in series.items():
        if only and arm not in only:
            continue
        if arm not in SERIES:
            continue
        label, colour, marker, dash = SERIES[arm]
        xs, ys = values(rows, column, scale)
        if xs:
            ax.plot(xs, ys, marker=marker, linestyle=dash, color=COLOUR[colour], label=label + label_suffix,
                    markeredgecolor=SURFACE, markeredgewidth=0.6)


def plot_fmm3d(ax, fmm3d, index):
    for eps, rows in sorted(fmm3d.items()):
        xs = [r[0] for r in rows]
        ys = [r[index] for r in rows]
        dash = "-" if "1e-4" in eps or "0.0001" in eps else "--"
        ax.plot(xs, ys, linestyle=dash, color=COLOUR["fmm3d"], label=f"FMM3D eps={eps} (CPU, 8 threads)")


def finish(ax, ylabel, title):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bodies N (sources = targets)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="best")


def save(fig, out: Path, name: str):
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(out / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--fmm3d-csv", type=Path, default=None)
    parser.add_argument("--stamp", default="")
    args = parser.parse_args()
    style()
    series = load_best(args.results / "best.csv")
    fmm3d = load_fmm3d(args.fmm3d_csv)
    out = args.results / "figures"
    out.mkdir(exist_ok=True)
    note = f"  [{args.stamp}]" if args.stamp else ""
    primary = ("jaxfmm kifmm p6", "dipfmm point->point o6")

    # 1. time per field update, host to host, primary point-target comparison
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    plot_series(ax, series, "host_evaluation_median_seconds", only=primary)
    plot_series(ax, series, "device_evaluation_median_seconds", only=("jaxfmm kifmm p6",),
                label_suffix=", device-resident")
    plot_fmm3d(ax, fmm3d, 1)
    finish(ax, "time per field update (s)", "Warm evaluation, FP32, identical point targets" + note)
    save(fig, out, "time_per_update")

    # 2. ns per body, every arm
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    plot_series(ax, series, "ns_per_body_host")
    for eps, rows in sorted(fmm3d.items()):
        ax.plot([r[0] for r in rows], [1e9 * r[1] / r[0] for r in rows],
                linestyle="-" if "1e-4" in eps else "--", color=COLOUR["fmm3d"], label=f"FMM3D eps={eps} (CPU)")
    finish(ax, "time per body per update (ns)", "Throughput, host to host" + note)
    save(fig, out, "ns_per_body")

    # 3. accuracy
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    plot_series(ax, series, "relative_l2")
    plot_fmm3d(ax, fmm3d, 2)
    finish(ax, "relative L2 error of H (512 sampled targets, FP64 reference)", "Achieved accuracy" + note)
    save(fig, out, "accuracy")

    # 4. setup and first-call (JIT) cost
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    plot_series(ax, series, "setup_seconds", only=primary + ("dipfmm prism->prism o6",))
    for arm in ("jaxfmm kifmm p6",):
        if arm in series:
            label, colour, marker, dash = SERIES[arm]
            xs, ys = values(series[arm], "first_call_seconds")
            ax.plot(xs, ys, marker=marker, linestyle="--", color=COLOUR[colour], markerfacecolor=SURFACE,
                    label=label + ", first call (JIT + run)")
    finish(ax, "seconds", "Setup (tree, operators, upload) and JIT" + note)
    save(fig, out, "setup_and_jit")

    # 5. GPU memory
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    plot_series(ax, series, "gpu_peak_bytes", scale=1 / 2**30, only=("jaxfmm kifmm p6",), label_suffix=", peak in use")
    plot_series(ax, series, "gpu_persistent_bytes", scale=1 / 2**30,
                only=("dipfmm point->point o6", "dipfmm prism->prism o6"), label_suffix=", persistent plan")
    finish(ax, "GPU memory (GiB)", "Device memory" + note)
    save(fig, out, "gpu_memory")

    # 6. speed at accuracy: time per update against achieved error, one point per N
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for arm, rows in series.items():
        if arm not in SERIES:
            continue
        label, colour, marker, dash = SERIES[arm]
        xs = [float(r["relative_l2"]) for r in rows if r.get("relative_l2")]
        ys = [1e9 * float(r["host_evaluation_median_seconds"]) / int(r["n_bodies"]) for r in rows if r.get("relative_l2")]
        ax.plot(xs, ys, marker=marker, linestyle="none", color=COLOUR[colour], label=label,
                markeredgecolor=SURFACE, markeredgewidth=0.6, alpha=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("relative L2 error of H")
    ax.set_ylabel("time per body per update (ns)")
    ax.set_title("Throughput at achieved accuracy (one marker per N)" + note, loc="left")
    ax.legend(loc="best")
    save(fig, out, "throughput_vs_accuracy")
    print(f"figures written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
