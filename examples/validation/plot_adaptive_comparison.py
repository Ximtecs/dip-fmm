"""Accuracy-versus-cost comparison of uniform and adaptive trees from adaptive_voronoi_sweep.py CSVs.

    python examples/validation/plot_adaptive_comparison.py sweeps/cmp_*.csv -o sweeps/adaptive_vs_uniform.png --table sweeps/adaptive_vs_uniform.md

One panel per CSV: relative L2 error against the dense reference versus median evaluation time,
uniform-tree plans as circles labelled by depth, adaptive plans as triangles labelled by capacity
(and depth when several were swept); colour = expansion order. Plans with bodies outside their leaf
are drawn hollow. The markdown table lists every plan with build time, near-field size and the
containment count.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

ORDER_COLOURS = {4: "#1f77b4", 6: "#d62728", 8: "#2ca02c", 10: "#9467bd"}


def read(path: Path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = float(v) if v not in ("", None) and not v.isalpha() else v
            except ValueError:
                pass
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="+", type=Path)
    p.add_argument("-o", "--out", type=Path, default=Path("sweeps/adaptive_vs_uniform.png"))
    p.add_argument("--table", type=Path, default=None)
    args = p.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = [(path.stem.replace("cmp_", ""), read(path)) for path in args.csvs if path.exists() and path.stat().st_size]
    sets = [(name, rows) for name, rows in sets if rows]
    ncol = min(3, len(sets))
    nrow = math.ceil(len(sets) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.4 * nrow), constrained_layout=True, squeeze=False)
    lines = []
    for ax, (name, rows) in zip(axes.ravel(), sets):
        depths_swept = len({r.get("depth") for r in rows if r["kind"] == "adaptive"}) > 1
        for r in rows:
            order = int(r["order"])
            colour = ORDER_COLOURS.get(order, "0.3")
            contained = int(r["exceeding"]) == 0
            marker = "o" if r["kind"] == "uniform" else "^"
            ax.scatter(r["eval_median_s"] * 1e3, r["rel_l2"], s=55, marker=marker,
                       facecolors=colour if contained else "none", edgecolors=colour, linewidths=1.4, zorder=3)
            label = f"d{int(r['depth'])}" if r["kind"] == "uniform" else (
                f"c{int(r['capacity'])}" + (f"/d{int(r['depth'])}" if depths_swept else ""))
            ax.annotate(label, (r["eval_median_s"] * 1e3, r["rel_l2"]), textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color="0.25")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("median evaluation time [ms] (8 threads, CPU static)")
        ax.set_ylabel("relative L2 error vs dense")
        n = int(rows[0]["n_bodies"]); prec = rows[0].get("precision", "")
        ax.set_title(f"{name}: {n} prisms ({prec})", fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        lines.append(f"\n### {name} ({n} prisms, {prec})\n")
        lines.append("| tree | order | setting | rel. L2 | eval [ms] | build [s] | tree [s] | P2P pairs | M2L | near field [MB] | far field [MB] | bodies outside leaf |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: (int(r["order"]), r["kind"], r.get("capacity", 0) or 0, r.get("depth", 0) or 0)):
            setting = f"depth {int(r['depth'])}" if r["kind"] == "uniform" else f"capacity {int(r['capacity'])}, max depth {int(r['depth'])}"
            lines.append(f"| {r['kind']} | {int(r['order'])} | {setting} | {r['rel_l2']:.2e} | {r['eval_median_s']*1e3:.1f} | "
                         f"{r['build_s']:.1f} | {r['tree_s']:.2f} | {int(r['p2p_interactions']):,} | {int(r['m2l_interactions']):,} | "
                         f"{r['near_field_MB'] + r['p2p_index_MB']:.0f} | {r['far_field_MB']:.0f} | {int(r['exceeding'])} |")
    for ax in axes.ravel()[len(sets):]:
        ax.axis("off")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color="0.3", label="uniform tree (d = depth)"),
               Line2D([], [], marker="^", ls="", color="0.3", label="adaptive tree (c = capacity)"),
               Line2D([], [], marker="o", ls="", markerfacecolor="none", color="0.3", label="hollow: bodies outside their leaf")]
    handles += [Line2D([], [], marker="s", ls="", color=c, label=f"order {o}") for o, c in ORDER_COLOURS.items()
                if any(int(r["order"]) == o for _, rows in sets for r in rows)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, bbox_to_anchor=(0.5, -0.04))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print("figure:", args.out)
    if args.table:
        args.table.write_text("\n".join(lines) + "\n")
        print("table:", args.table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
