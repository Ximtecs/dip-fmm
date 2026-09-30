#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Figures for a jaxFMM-versus-dip-fmm campaign from ``best.csv``.

One figure set per tier (point dipoles, cube bodies, Kuhn tetrahedra): time
per field update, throughput, achieved accuracy and throughput-at-accuracy,
plus setup/JIT and GPU memory. Style follows the FMM3D campaign's preliminary
figures (recessive grid, small type, frameless legend, one measure per axis).
Series colours are fixed per framework and never reassigned: jaxFMM orange,
dip-fmm point-target blue, dip-fmm averaged-target green, FMM3D grey; arms of
one framework differ by marker and dash so identity is never colour alone.

Optional FMM3D CPU rows (``--fmm3d-csv``) are drawn in grey on the point tier
and labelled as CPU so nobody reads them as a GPU result.
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
    "dipfmm point->point o6": ("dip-fmm order 6, point dipoles (GPU, FMM3D campaign rows)", "dipfmm_point", "s", "-"),
    "dipfmm point->point o10": ("dip-fmm order 10, point dipoles (GPU, FMM3D campaign rows)", "dipfmm_point", "D", "--"),
    "dipfmm point->point o15": ("dip-fmm order 15, point dipoles (GPU, FMM3D campaign rows)", "dipfmm_point", "h", ":"),
    "dipfmm point->point o17": ("dip-fmm order 17, point dipoles (GPU, FMM3D campaign rows)", "dipfmm_point", "p", "-."),
    "dipfmm point->point o20": ("dip-fmm order 20, point dipoles (GPU, FMM3D campaign rows)", "dipfmm_point", "*", ":"),
    "dipfmm prism->prism o6": ("dip-fmm order 6, prism sources and targets (GPU)", "dipfmm_finite", "P", "-"),
    "dipfmm tetra->tetra o6": ("dip-fmm order 6, Kuhn tetrahedra sources and targets (GPU)", "dipfmm_finite", "X", "--"),
    "dipfmm prism->point o6": ("dip-fmm order 6, prism sources, point targets (GPU)", "dipfmm_point", "s", "-"),
    "dipfmm tetra->point o6": ("dip-fmm order 6, Kuhn tetrahedra sources, point targets (GPU)", "dipfmm_point", "D", "--"),
    "jaxfmm element p6 n8 prism_faces->point": ("jaxFMM element p=6, near degree 8, cube faces (GPU)", "jaxfmm", "o", "-"),
    "jaxfmm element p8 n10 prism_faces->point": ("jaxFMM element p=8, near degree 10, cube faces (GPU)", "jaxfmm", "^", "--"),
    "jaxfmm element p6 n8 tetra_faces->point": ("jaxFMM element p=6, near degree 8, tetrahedron faces (GPU)", "jaxfmm", "o", "-"),
    "jaxfmm element p8 n10 tetra_faces->point": ("jaxFMM element p=8, near degree 10, tetrahedron faces (GPU)", "jaxfmm", "^", "--"),
}

TIERS = {
    "point": {
        "title": "Point dipoles at lattice sites, identical point targets",
        "arms": ("jaxfmm kifmm p4", "jaxfmm kifmm p6", "jaxfmm kifmm p8",
                 "dipfmm point->point o6", "dipfmm point->point o10", "dipfmm point->point o15",
                 "dipfmm point->point o17", "dipfmm point->point o20"),
        "xlabel": "dipoles N (sources = targets)",
    },
    "prism": {
        "title": "Uniformly magnetised touching cubes, targets at cube centres",
        "arms": ("jaxfmm element p6 n8 prism_faces->point", "jaxfmm element p8 n10 prism_faces->point",
                 "dipfmm prism->point o6", "dipfmm prism->prism o6"),
        "xlabel": "cubes N (jaxFMM integrates about 6N face triangles)",
    },
    "tetra": {
        "title": "Uniformly magnetised Kuhn tetrahedra, targets at centroids",
        "arms": ("jaxfmm element p6 n8 tetra_faces->point", "jaxfmm element p8 n10 tetra_faces->point",
                 "dipfmm tetra->point o6", "dipfmm tetra->tetra o6"),
        "xlabel": "tetrahedra N (jaxFMM integrates about 2N face triangles)",
    },
}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK_PRIMARY, "text.color": INK_PRIMARY,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
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
        try:
            ys.append(float(value) * scale)
        except ValueError:
            continue
        xs.append(int(row["n_bodies"]))
    return xs, ys


def plot_series(ax, series, column, arms, scale=1.0, label_suffix="", hollow=False):
    for arm in arms:
        rows = series.get(arm)
        if not rows or arm not in SERIES:
            continue
        label, colour, marker, dash = SERIES[arm]
        xs, ys = values(rows, column, scale)
        if xs:
            ax.plot(xs, ys, marker=marker, linestyle=dash, color=COLOUR[colour], label=label + label_suffix,
                    markeredgecolor=SURFACE if not hollow else COLOUR[colour],
                    markerfacecolor=SURFACE if hollow else COLOUR[colour], markeredgewidth=0.6 if not hollow else 1.2)


def plot_fmm3d(ax, fmm3d, index):
    for eps, rows in sorted(fmm3d.items()):
        xs = [r[0] for r in rows]
        ys = [r[index] for r in rows]
        dash = "-" if ("1e-4" in eps or "0.0001" in eps) else "--"
        ax.plot(xs, ys, linestyle=dash, color=COLOUR["fmm3d"], label=f"FMM3D eps={eps} (CPU, 8 threads)")


def finish(ax, xlabel, ylabel, title):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", wrap=True)
    ax.legend(loc="best")


def save(fig, out: Path, name: str):
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(out / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def tier_figures(series, fmm3d, out: Path, tier: str, note: str) -> None:
    spec = TIERS[tier]
    arms = spec["arms"]
    if not any(series.get(arm) for arm in arms):
        return
    present = [arm for arm in arms if series.get(arm)]
    point_arms = [arm for arm in present if "->point" in arm or "point->point" in arm]

    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    plot_series(ax, series, "host_evaluation_median_seconds", present)
    jax_arms = [arm for arm in present if arm.startswith("jaxfmm")]
    plot_series(ax, series, "device_evaluation_median_seconds", jax_arms, label_suffix=", device-resident", hollow=True)
    if tier == "point":
        plot_fmm3d(ax, fmm3d, 1)
    finish(ax, spec["xlabel"], "time per field update (s)", f"{spec['title']}: warm evaluation, FP32{note}")
    save(fig, out, f"{tier}_time_per_update")

    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    plot_series(ax, series, "ns_per_body_host", present)
    if tier == "point":
        for eps, rows in sorted(fmm3d.items()):
            ax.plot([r[0] for r in rows], [1e9 * r[1] / r[0] for r in rows],
                    linestyle="-" if "1e-4" in eps else "--", color=COLOUR["fmm3d"], label=f"FMM3D eps={eps} (CPU)")
    finish(ax, spec["xlabel"], "time per body per update (ns)", f"{spec['title']}: throughput, host to host{note}")
    save(fig, out, f"{tier}_ns_per_body")

    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    plot_series(ax, series, "relative_l2", present)
    if tier == "point":
        plot_fmm3d(ax, fmm3d, 2)
    finish(ax, spec["xlabel"], "relative L2 error of H (512 sampled targets, FP64 reference)",
           f"{spec['title']}: achieved accuracy{note}")
    save(fig, out, f"{tier}_accuracy")

    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    for arm in point_arms:
        rows = series[arm]
        label, colour, marker, dash = SERIES[arm]
        xs = [float(r["relative_l2"]) for r in rows if r.get("relative_l2") not in (None, "")]
        ys = [1e9 * float(r["host_evaluation_median_seconds"]) / int(r["n_bodies"])
              for r in rows if r.get("relative_l2") not in (None, "")]
        ax.plot(xs, ys, marker=marker, linestyle="none", color=COLOUR[colour], label=label,
                markeredgecolor=SURFACE, markeredgewidth=0.6, alpha=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("relative L2 error of H")
    ax.set_ylabel("time per body per update (ns)")
    ax.set_title(f"{spec['title']}: throughput at achieved accuracy (one marker per N, point targets){note}",
                 loc="left", wrap=True)
    ax.legend(loc="best")
    save(fig, out, f"{tier}_throughput_vs_accuracy")

    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    plot_series(ax, series, "setup_seconds", present)
    plot_series(ax, series, "first_call_seconds", jax_arms, label_suffix=", first call (JIT + run)", hollow=True)
    finish(ax, spec["xlabel"], "seconds", f"{spec['title']}: setup (tree, operators, upload) and JIT{note}")
    save(fig, out, f"{tier}_setup_and_jit")

    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    plot_series(ax, series, "gpu_peak_bytes", jax_arms, scale=1 / 2**30, label_suffix=", peak in use")
    dip_arms = [arm for arm in present if arm.startswith("dipfmm")]
    plot_series(ax, series, "gpu_persistent_bytes", dip_arms, scale=1 / 2**30, label_suffix=", persistent plan")
    finish(ax, spec["xlabel"], "GPU memory (GiB)", f"{spec['title']}: device memory{note}")
    save(fig, out, f"{tier}_gpu_memory")


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
    for tier in TIERS:
        tier_figures(series, fmm3d, out, tier, note)
    print(f"figures written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
