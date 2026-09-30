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
        if case.get("engine") == "element":
            near = f" n{case.get('near_deg')}" if case.get("near_deg") else ""
            return f"jaxfmm element p{case.get('p')}{near} {case.get('source_geometry')}->point"
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


def rescore_pending(results: Path) -> int:
    """Score jaxFMM finite rows whose dense reference appeared after they ran."""
    import numpy as np

    sys.path.insert(0, str(ROOT))
    from jaxfmm_campaign import reference

    rescored = 0
    for path in sorted((results / "rows").glob("*.json")):
        row = json.loads(path.read_text())
        ref = row.get("reference") or {}
        if row.get("status") != "success" or ref.get("status") != "pending":
            continue
        cache = results / "reference_cache" / ref["key"]
        field_path = path.with_suffix(".sampled_field.npy")
        if not cache.is_file() or not field_path.is_file():
            continue
        with np.load(cache) as stored:
            row["error_metrics"] = reference.error_metrics(np.load(field_path), stored["field"])
        row["reference"].update({"status": "scored", "source": str(cache)})
        path.write_text(json.dumps(row, indent=2, default=str) + "\n")
        rescored += 1
    return rescored


def article1_point_rows(config: dict, processed: Path) -> list[dict]:
    """dip-fmm point-to-point rows reused from the FMM3D campaign's processed tables."""
    spec = config.get("article1_point_rows")
    if not spec or not processed.is_dir():
        return []
    rows = []
    for name in spec["csv"]:
        path = processed / name
        if not path.is_file():
            continue
        with path.open() as stream:
            for r in csv.DictReader(stream):
                if r.get("status") != "success" or r.get("framework") != "dipfmm":
                    continue
                if r.get("case_backend") != spec["backend"] or r.get("case_precision") != spec["precision"]:
                    continue
                if not str(r.get("case_dataset_id", "")).startswith("lattice_"):
                    continue
                if r.get("case_source_geometry") not in ("point", "") or r.get("case_target_geometry") not in ("point", ""):
                    continue
                grid = int(r["case_grid"])
                rows.append({
                    "case_id": f"article1:{name}:{r['case_id']}", "framework": "dipfmm",
                    "arm": f"dipfmm point->point o{int(r['case_order'])}", "role": "point_target_article1",
                    "dof_per_box": (int(r["case_order"]) + 1) ** 2, "grid": grid, "n_bodies": grid**3,
                    "config": f"depth={r.get('case_depth')}", "status": "success",
                    "setup_seconds": r.get("external_setup_seconds") or None, "first_call_seconds": None,
                    "jit_seconds_estimate": None, "device_evaluation_median_seconds": None,
                    "host_evaluation_median_seconds": float(r["external_evaluation_median_seconds"]),
                    "host_evaluation_min_seconds": r.get("external_evaluation_min_seconds") or None,
                    "ns_per_body_host": 1e9 * float(r["external_evaluation_median_seconds"]) / grid**3,
                    "relative_l2": float(r["relative_l2"]) if r.get("relative_l2") else None,
                    "max_absolute_over_reference_rms": None,
                    "max_pointwise_relative": float(r["max_pointwise_relative"]) if r.get("max_pointwise_relative") else None,
                    "gpu_peak_bytes": None, "gpu_persistent_bytes": r.get("persistent_device_bytes") or None,
                    "gpu_process_mib": None, "near_field": None, "max_depth": None,
                    "failure_reason": f"solver {r.get('frozen_solver_sha', '')[:12]} from {name}",
                })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--article1-processed", type=Path,
                        default=Path("/home/mihaa/MagTense/dip-fmm/Article1/results/processed"))
    args = parser.parse_args()
    print(f"rescored {rescore_pending(args.results)} pending jaxFMM finite rows")
    rows = []
    for path in sorted((args.results / "rows").glob("*.json")):
        rows.append(flatten(json.loads(path.read_text())))
    if args.config:
        rows.extend(article1_point_rows(json.loads(args.config.read_text()), args.article1_processed))
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
