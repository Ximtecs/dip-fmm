#!/usr/bin/env python3
"""Summarise benchmark_operator_representation CSV rows.

For every measured configuration (stage, geometry pair, layout, separation,
mode, precision, order, backend) the script pairs the procedural row with the
best precomputed representation (lowest time per item) and reports

    T_build             construction of the precomputed representation
    T_apply, T_proc     one complete update, precomputed and procedural
    ratio               T_proc / T_apply
    bytes saved         persistent bytes of the precomputed representation
                        minus the procedural invariants
    K_break_even        (T_build - T_proc_setup) / (T_proc - T_apply), in
                        complete field updates, when T_proc > T_apply

Usage:
    python benchmarks/analyse_operator_representation.py results.csv \
        [--markdown summary.md] [--stage p2p]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

KEY = ("stage", "source_geometry", "target_geometry", "layout", "separation",
       "mode", "precision", "order", "backend")

PRIMARY_PROCEDURAL = ("procedural", "procedural-builder", "procedural-cuda")


def read_rows(path: Path) -> list[dict]:
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for column in ("build_seconds", "update_seconds", "ns_per_item",
                       "measured_fraction", "max_relative_error"):
            row[column] = float(row[column]) if row[column] else 0.0
        for column in ("items", "bodies", "leaves", "operator_bytes",
                       "index_bytes", "metadata_bytes", "invariant_bytes",
                       "geometry_bytes", "topology_bytes", "scratch_bytes",
                       "total_persistent_bytes", "unique_operators", "threads"):
            row[column] = int(row[column]) if row[column] else 0
    return rows


def persistent_bytes(row: dict) -> int:
    return (row["operator_bytes"] + row["index_bytes"] + row["metadata_bytes"] +
            row["invariant_bytes"])


def format_seconds(value: float) -> str:
    if value == 0.0:
        return "0"
    if value >= 1.0:
        return f"{value:.3g} s"
    if value >= 1e-3:
        return f"{value * 1e3:.3g} ms"
    if value >= 1e-6:
        return f"{value * 1e6:.3g} us"
    return f"{value * 1e9:.3g} ns"


def format_bytes(value: float) -> str:
    if abs(value) >= 1e9:
        return f"{value / 1e9:.3g} GB"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.3g} MB"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.3g} kB"
    return f"{value:.0f} B"


def format_break_even(value: float | None) -> str:
    if value is None:
        return "never (procedural faster)"
    if value == 0.0:
        return "0"
    if value < 1.0:
        return f"{value:.2g}"
    return f"{value:,.0f}"


def summarise(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[column] for column in KEY)].append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        precomputed = [row for row in group
                       if row["representation_kind"] == "precomputed"]
        procedural = [row for row in group
                      if row["representation_kind"] == "procedural"]
        if not precomputed or not procedural:
            continue
        # The production reconstruction (the exact builder, or the CUDA kernel
        # in the plan's precision) is the primary procedural row; the
        # reduced-precision and alternative-math variants are listed beside it.
        best = min(precomputed, key=lambda row: row["ns_per_item"])
        primary = [row for row in procedural
                   if row["representation"] in PRIMARY_PROCEDURAL]
        proc = primary[-1] if primary else min(procedural,
                                               key=lambda row: row["ns_per_item"])
        variants = "; ".join(
            f"{row['representation']} {row['ns_per_item']:.3g} ns "
            f"(err {row['max_relative_error']:.1e})"
            for row in procedural if row is not proc)
        t_build = best["build_seconds"]
        t_apply = best["update_seconds"]
        t_proc = proc["update_seconds"]
        t_proc_setup = proc["build_seconds"]
        if t_proc > t_apply:
            break_even = max(0.0, (t_build - t_proc_setup) / (t_proc - t_apply))
        else:
            break_even = None
        summaries.append({
            **dict(zip(KEY, key)),
            "items": best["items"],
            "threads": best["threads"],
            "best_precomputed": best["representation"],
            "build_seconds": t_build,
            "build_seconds_per_item": best["build_seconds_per_item"],
            "apply_seconds": t_apply,
            "apply_ns_per_item": best["ns_per_item"],
            "procedural_seconds": t_proc,
            "procedural_ns_per_item": proc["ns_per_item"],
            "measured_fraction": proc["measured_fraction"],
            "ratio": t_proc / t_apply if t_apply > 0 else math.inf,
            "precomputed_bytes": persistent_bytes(best),
            "procedural_bytes": persistent_bytes(proc),
            "bytes_saved": persistent_bytes(best) - persistent_bytes(proc),
            "unique_operators": best["unique_operators"],
            "break_even_updates": break_even,
            "precomputed_error": best["max_relative_error"],
            "procedural_error": proc["max_relative_error"],
            "procedural_note": proc["note"],
            "procedural_variants": variants,
        })
    return summaries


def write_markdown(summaries: list[dict], stream) -> None:
    header = ("| stage | pair | layout | separation | mode | prec | p | backend | items | "
              "best precomputed | T_build | apply / item | procedural / item | ratio | "
              "bytes saved | K_break_even | err pre | err proc | variants |")
    print(header, file=stream)
    print("|" + "---|" * (header.count("|") - 1), file=stream)
    for item in summaries:
        pair = f"{item['source_geometry']}->{item['target_geometry']}"
        order = item["order"] if item["order"] not in ("0", 0) else ""
        fraction = ("" if item["measured_fraction"] >= 1.0
                    else f" ({item['measured_fraction']:.2f} of set)")
        print("| " + " | ".join([
            item["stage"], pair, item["layout"], item["separation"], item["mode"],
            item["precision"], str(order), item["backend"], f"{item['items']:,}",
            item["best_precomputed"], format_seconds(item["build_seconds"]),
            f"{item['apply_ns_per_item']:.3g} ns",
            f"{item['procedural_ns_per_item']:.3g} ns{fraction}",
            f"{item['ratio']:.3g}x", format_bytes(item["bytes_saved"]),
            format_break_even(item["break_even_updates"]),
            f"{item['precomputed_error']:.1e}", f"{item['procedural_error']:.1e}",
            item["procedural_variants"],
        ]) + " |", file=stream)


def write_csv(summaries: list[dict], path: Path) -> None:
    if not summaries:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        for item in summaries:
            writer.writerow({key: ("" if value is None else value)
                             for key, value in item.items()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--stage", default="", help="p2p, p2m or l2p")
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    args = parser.parse_args()
    rows = read_rows(args.csv)
    if args.stage:
        rows = [row for row in rows if row["stage"] == args.stage]
    summaries = summarise(rows)
    write_markdown(summaries, sys.stdout)
    if args.markdown:
        with args.markdown.open("w") as stream:
            write_markdown(summaries, stream)
    if args.summary_csv:
        write_csv(summaries, args.summary_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
