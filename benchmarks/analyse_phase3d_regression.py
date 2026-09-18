#!/usr/bin/env python3
"""Summarise the Phase-3D regression matrix.

ENGINEERING REGRESSION BASELINE -- NOT AN ARTICLE1 PUBLICATION BENCHMARK.

Two views of `run_phase3d_regression.py` output:

  policy   within each policy group, the automatic result against every forced
           alternative that was measured, as a ratio of repeated evaluation.
           A ratio below 1 means the alternative is faster than `Auto`.
  matrix   one line per regression workload row: resolved representation,
           repeated evaluation, construction and retained bytes.

The comparison number is the repeated-evaluation median, because that is what
a production policy is chosen on; construction and memory are printed beside
it so that a policy that wins evaluation while losing setup or footprint is
visible rather than hidden.

Usage:
    python benchmarks/analyse_phase3d_regression.py \
        --input results/phase3d_regression.csv --view policy
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def number(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None  # drop NaN


def group_key(row: dict) -> str:
    """The comparison group: everything but the forced alternative."""
    case = row["case"]
    parts = case.split("/")
    # Case names are <group>/<shape>/<variant...>/<precision>; the variant is
    # what is being compared, so it is dropped from the key.
    if len(parts) < 3:
        return case
    return "/".join([parts[0], parts[1], parts[-1]])


def variant(row: dict) -> str:
    parts = row["case"].split("/")
    return "/".join(parts[2:-1]) if len(parts) >= 4 else parts[-1]


def policy_view(rows: list[dict]) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["workload"].startswith("P-"):
            groups[group_key(row)].append(row)

    for name in sorted(groups):
        members = groups[name]
        automatic = [r for r in members
                     if r.get("p2p_packing_requested", "") in ("auto", "")
                     and r.get("point_expansion_requested", "") in ("auto", "")]
        # An `auto` row measured under both layouts yields two baselines; the
        # general-layout one is the default a caller gets.
        baseline = None
        for row in automatic:
            if row.get("layout_hint") == "general":
                baseline = row
                break
        if baseline is None and automatic:
            baseline = automatic[0]
        if baseline is None:
            continue

        reference = number(baseline["evaluation_median"])
        if reference is None or reference <= 0:
            continue

        print(f"\n== {name}")
        print(f"   {'variant':<34} {'resolved':<18} {'eval [ms]':>10} "
              f"{'ratio':>7} {'setup [s]':>10} {'host MB':>9} {'dev MB':>8}")
        for row in sorted(members, key=lambda r: r["case"]):
            evaluation = number(row["evaluation_median"])
            if evaluation is None:
                continue
            resolved = row.get("p2p_packing", "")
            if row["workload"].endswith("expansion"):
                resolved = (f"{row.get('p2m_execution','')}/"
                            f"{row.get('l2p_execution','')}")
            if row["workload"] == "P-cpu-m2l":
                resolved = row.get("static_multiply_backend", "")
            host = number(row.get("static_plan_bytes", "")) or 0.0
            device = number(row.get("cuda_persistent_device_bytes", "")) or 0.0
            marker = " *" if row is baseline else "  "
            print(f"{marker} {variant(row):<34} {resolved:<18} "
                  f"{evaluation * 1e3:>10.3f} {evaluation / reference:>7.2f} "
                  f"{number(row.get('fmm_setup_seconds','')) or 0.0:>10.3f} "
                  f"{host / 1e6:>9.1f} {device / 1e6:>8.1f}")
    print("\n* = the automatic result used as the ratio's denominator.")


def matrix_view(rows: list[dict]) -> None:
    print(f"{'case':<58} {'packing':<17} {'p2m/l2p':<24} "
          f"{'eval [ms]':>10} {'setup [s]':>10} {'host MB':>9} {'dev MB':>8}")
    for row in rows:
        if row["workload"].startswith("P-"):
            continue
        evaluation = number(row["evaluation_median"])
        if evaluation is None:
            continue
        host = number(row.get("static_plan_bytes", "")) or 0.0
        device = number(row.get("cuda_persistent_device_bytes", "")) or 0.0
        expansion = (f"{row.get('p2m_execution','')}/"
                     f"{row.get('l2p_execution','')}")
        print(f"{row['case']:<58} {row.get('p2p_packing',''):<17} "
              f"{expansion:<24} {evaluation * 1e3:>10.3f} "
              f"{number(row.get('fmm_setup_seconds','')) or 0.0:>10.3f} "
              f"{host / 1e6:>9.1f} {device / 1e6:>8.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--view", default="policy",
                        choices=("policy", "matrix", "both"))
    arguments = parser.parse_args()

    with Path(arguments.input).open() as stream:
        rows = list(csv.DictReader(stream))

    if arguments.view in ("matrix", "both"):
        matrix_view(rows)
    if arguments.view in ("policy", "both"):
        policy_view(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
