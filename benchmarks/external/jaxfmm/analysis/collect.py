#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect campaign rows into flat tables.

Writes ``rows.csv`` (every case), ``best.csv`` (per framework arm and grid,
the fastest successful configuration by host-side evaluation time) and
``summary.md`` (the headline table) next to the rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

FIELDS = [
    "case_id", "framework", "arm", "role", "dof_per_box", "grid", "n_bodies", "config", "status",
    "setup_seconds", "first_call_seconds", "jit_seconds_estimate",
    "device_evaluation_median_seconds", "host_evaluation_median_seconds", "host_evaluation_min_seconds",
    "ns_per_body_host", "relative_l2", "max_absolute_over_reference_rms", "max_pointwise_relative",
    "gpu_peak_bytes", "gpu_persistent_bytes", "gpu_process_mib", "near_field", "max_depth",
    "failure_reason",
]


def arm_name(row: dict) -> str:
    case = row.get("case", {})
    if row["framework"] == "jaxfmm":
        return f"jaxfmm {case.get('engine')} p{case.get('p')}"
    return f"dipfmm {case.get('source_geometry')}->{case.get('target_geometry')} o{case.get('order')}"


def flatten(row: dict) -> dict:
    case = row.get("case", {})
    grid = int(case.get("grid"))
    metrics = row.get("error_metrics", {})
    dataset = row.get("dataset") or {}
    n_bodies = dataset.get("n_sources") or case.get("n_bodies") or grid**3
    flat = {
        "case_id": row.get("case_id"),
        "framework": row.get("framework"),
        "arm": arm_name(row),
        "role": case.get("role"),
        "dof_per_box": case.get("dof_per_box"),
        "grid": grid,
        "n_bodies": int(n_bodies),
        "config": (f"N_max={case.get('N_max')}" if row["framework"] == "jaxfmm" else f"depth={case.get('depth')}"),
        "status": row.get("status"),
        "setup_seconds": row.get("setup_seconds"),
        "first_call_seconds": row.get("first_call_seconds"),
        "jit_seconds_estimate": row.get("jit_seconds_estimate"),
        "device_evaluation_median_seconds": row.get("device_evaluation_median_seconds"),
        "host_evaluation_median_seconds": row.get("host_evaluation_median_seconds"),
        "host_evaluation_min_seconds": row.get("host_evaluation_min_seconds"),
        "ns_per_body_host": row.get("ns_per_body_host"),
        "relative_l2": metrics.get("relative_l2"),
        "max_absolute_over_reference_rms": metrics.get("max_absolute_over_reference_rms"),
        "max_pointwise_relative": metrics.get("max_pointwise_relative"),
        "failure_reason": (row.get("failure_reason") or "")[:200],
        "near_field": (row.get("setup_info") or {}).get("near_field"),
        "max_depth": (row.get("setup_info") or {}).get("max_depth"),
        "gpu_peak_bytes": None, "gpu_persistent_bytes": None, "gpu_process_mib": None,
    }
    if row["framework"] == "jaxfmm":
        memory = row.get("memory_after_timing") or {}
        flat["gpu_peak_bytes"] = memory.get("peak_bytes_in_use")
        flat["gpu_persistent_bytes"] = (row.get("memory_after_setup") or {}).get("bytes_in_use")
    else:
        stats = (row.get("plan") or {}).get("cuda_plan_statistics") or {}
        flat["gpu_persistent_bytes"] = stats.get("total_bytes") or stats.get("persistent_device_bytes")
        with_plan = row.get("gpu_with_plan") or {}
        before = row.get("gpu_before") or {}
        if with_plan.get("memory_used_mib") is not None and before.get("memory_used_mib") is not None:
            flat["gpu_process_mib"] = with_plan["memory_used_mib"] - before["memory_used_mib"]
    return flat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted((args.results / "rows").glob("*.json")):
        rows.append(flatten(json.loads(path.read_text())))
    rows.sort(key=lambda r: (r["framework"], r["arm"], r["grid"], r["config"]))
    with (args.results / "rows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    best: dict[tuple, dict] = {}
    for row in rows:
        if row["status"] != "success" or row["host_evaluation_median_seconds"] is None:
            continue
        key = (row["arm"], row["grid"])
        if key not in best or row["host_evaluation_median_seconds"] < best[key]["host_evaluation_median_seconds"]:
            best[key] = row
    best_rows = sorted(best.values(), key=lambda r: (r["arm"], r["grid"]))
    with (args.results / "best.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(best_rows)

    statuses: dict[tuple, list[str]] = {}
    for row in rows:
        statuses.setdefault((row["arm"], row["grid"]), []).append(row["status"])
    lines = ["| arm | grid | N | best config | setup s | JIT/first s | eval host s | eval device s | ns/body | rel L2 | max/rms | GPU peak GiB | statuses |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in best_rows:
        peak = row["gpu_peak_bytes"] or row["gpu_persistent_bytes"]
        peak_txt = f"{peak / 2**30:.2f}" if peak else "-"
        dev = row["device_evaluation_median_seconds"]
        lines.append(
            f"| {row['arm']} | {row['grid']} | {row['n_bodies']} | {row['config']} | {row['setup_seconds']:.3g} | "
            f"{(row['jit_seconds_estimate'] if row['jit_seconds_estimate'] is not None else row['first_call_seconds']):.3g} | "
            f"{row['host_evaluation_median_seconds']:.3g} | {dev if dev is None else f'{dev:.3g}'} | "
            f"{row['ns_per_body_host']:.1f} | {row['relative_l2']:.2e} | {row['max_absolute_over_reference_rms']:.2e} | "
            f"{peak_txt} | {','.join(sorted(set(statuses[(row['arm'], row['grid'])])))} |")
    missing = [(arm, grid) for (arm, grid), st in statuses.items() if (arm, grid) not in best]
    if missing:
        lines.append("")
        lines.append("Arms without a successful configuration: " + ", ".join(
            f"{arm} grid {grid} ({','.join(sorted(set(statuses[(arm, grid)])))})" for arm, grid in sorted(missing)))
    (args.results / "summary.md").write_text("\n".join(lines) + "\n")

    # Accuracy-matched pairs: nominal orders are not comparable (KIFMM p is a
    # surface resolution, dip-fmm order a harmonic degree), so for every size
    # pair each jaxFMM arm with the dip-fmm point arm whose achieved error is
    # closest in log space, and quote the speed ratio only for that pair.
    import math

    matched = ["| N | jaxFMM arm | rel L2 | dip-fmm arm | rel L2 | error ratio jax/dip | host time jax s | host time dip s | dip/jax time |",
               "|---|---|---|---|---|---|---|---|---|"]
    grids = sorted({row["grid"] for row in best_rows})
    for grid in grids:
        jax_rows = [r for r in best_rows if r["grid"] == grid and r["framework"] == "jaxfmm" and r["relative_l2"]]
        dip_rows = [r for r in best_rows if r["grid"] == grid and r["framework"] == "dipfmm"
                    and r["arm"].startswith("dipfmm point") and r["relative_l2"]]
        for jrow in jax_rows:
            if not dip_rows:
                continue
            drow = min(dip_rows, key=lambda r: abs(math.log(float(r["relative_l2"])) - math.log(float(jrow["relative_l2"]))))
            jt, dtime = float(jrow["host_evaluation_median_seconds"]), float(drow["host_evaluation_median_seconds"])
            matched.append(
                f"| {grid**3} | {jrow['arm']} | {float(jrow['relative_l2']):.2e} | {drow['arm']} | "
                f"{float(drow['relative_l2']):.2e} | {float(jrow['relative_l2']) / float(drow['relative_l2']):.2f} | "
                f"{jt:.3g} | {dtime:.3g} | {dtime / jt:.2f} |")
    (args.results / "matched.md").write_text("\n".join(matched) + "\n")
    print(f"{len(rows)} rows, {len(best_rows)} best rows -> {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
