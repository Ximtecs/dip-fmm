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

Rows are grouped by the `comparison_group` and `comparison_variant` columns
the runner records, not by parsing the display name. A file written before
those columns existed is grouped through the runner's own case generators; a
case neither source recognises is printed alone rather than compared against
something it does not match.

Usage:
    python benchmarks/analyse_phase3d_regression.py \
        --input results/phase3d_regression.csv --view policy
"""

from __future__ import annotations

import argparse
import csv
import functools
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def number(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None  # drop NaN


@functools.lru_cache(maxsize=1)
def generated_comparisons() -> dict[str, tuple[str, str]]:
    """`case` -> (group, variant) as the runner's own generators define it.

    A CSV written before the comparison columns existed carries only the
    display name. Rather than guess the group back out of that name -- which
    is what previously merged a regular lattice with an irregular cloud --
    the mapping is taken from the generators that produced the names, so a
    legacy file is grouped by the same definition a current one records.
    """
    try:
        import run_phase3d_regression as runner
    except ImportError:
        return {}
    mapping: dict[str, tuple[str, str]] = {}
    for cuda in (True, False):
        for mkl in (True, False):
            for case in runner.cases("all", cuda=cuda, mkl=mkl):
                mapping[case.name] = (case.comparison_group,
                                      case.comparison_variant)
    return mapping


def comparison(row: dict) -> tuple[str, str]:
    """The row's (group, variant), from the CSV or the generator mapping.

    An unrecognised case becomes a group of its own. That prints it without a
    ratio, which is the safe failure: a missing comparison is obvious, while
    a wrong one looks exactly like a real measurement.
    """
    group = row.get("comparison_group", "") or ""
    variant_name = row.get("comparison_variant", "") or ""
    if group and variant_name:
        return group, variant_name
    generated = generated_comparisons().get(row["case"])
    if generated is not None:
        return generated
    return row["case"], row["case"]


def group_key(row: dict) -> str:
    return comparison(row)[0]


def variant(row: dict) -> str:
    return comparison(row)[1]


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
        # The variant names the forced alternative, so an automatic row is the
        # one whose variant is `auto`. Where the compared axis is the backend
        # itself no variant is named `auto` and every row is automatic, which
        # is why the requested-column filter above still decides membership.
        named_auto = [r for r in automatic
                      if variant(r).split("/")[0] == "auto"]
        preferred = named_auto or automatic
        # An `auto` row measured under both layouts yields two candidates; the
        # general-layout one is the default a caller gets.
        baseline = None
        for row in preferred:
            if row.get("layout_hint") == "general":
                baseline = row
                break
        if baseline is None and preferred:
            baseline = preferred[0]
        if baseline is None:
            continue

        reference = number(baseline["evaluation_median"])
        if reference is None or reference <= 0:
            continue

        print(f"\n== {name}")
        print(f"   {'variant':<34} {'resolved':<18} {'eval [ms]':>10} "
              f"{'ratio':>7} {'setup [s]':>10} {'host MB':>9} {'dev MB':>8}")
        for row in sorted(members, key=lambda r: (variant(r), r["case"])):
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
