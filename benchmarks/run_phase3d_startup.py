#!/usr/bin/env python3
"""Phase-3D cold/warm startup decomposition.

ENGINEERING REGRESSION BASELINE -- NOT AN ARTICLE1 PUBLICATION BENCHMARK.

Phase 3C left four startup observations (derived packing on a warm cache, the
canonical near-field data a procedural point plan never reads, the interaction
sort, and the universal operator bank). This driver re-measures them on the
final integrated state by running the same geometry three ways and keeping the
per-stage `StaticPlanStatistics` and cache timings the solver already prints:

  cold-no-cache   `CDFMM_DISABLE_CACHE=1`; nothing is read or written.
  cold-write      an empty cache directory; the build also pays the write.
  warm-hit        the same directory again; the plan is loaded.

The difference between `cold-write` and `warm-hit` is what a returning user
actually saves, and the stages still charged on `warm-hit` are what is left to
optimise. No cache format or key is changed by this script; it only chooses a
cache directory.

Usage:
    python benchmarks/run_phase3d_startup.py \
        --binary build-bench-all/benchmarks/benchmark_uniform_fmm \
        --output startup.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Stages printed by the initialisation summary that the startup review reads.
STAGES = [
    "setup.total_seconds",
    "setup.static_plan_seconds",
    "setup.tree_construction_seconds",
    "setup.topology_construction_seconds",
    "setup.normalisation_seconds",
    "setup.geometry_hash_seconds",
    "setup.p2p_seconds",
    "setup.p2p_interaction_setup_seconds",
    "setup.p2p_canonical_operator_seconds",
    "setup.p2p_derived_packing_seconds",
    "setup.p2m_seconds",
    "setup.m2m_seconds",
    "setup.m2l_seconds",
    "setup.l2l_seconds",
    "setup.l2p_seconds",
    "setup.universal_operator_build_seconds",
    "setup.precision_conversion_seconds",
    "setup.backend_packing_seconds",
    "setup.far_field_packing_seconds",
    "setup.cuda_upload_seconds",
    "setup.universal_cache_lookup_seconds",
    "setup.universal_cache_load_seconds",
    "setup.universal_cache_write_seconds",
    "setup.geometry_cache_lookup_seconds",
    "setup.geometry_cache_load_seconds",
    "setup.geometry_cache_write_seconds",
    "cache.enabled",
    "cache.universal.hit",
    "cache.geometry.hit",
    "cache.bytes_read",
    "cache.bytes_written",
    "p2p_packing",
    "p2m_execution",
    "l2p_execution",
]

COLUMNS = ["case", "mode", *STAGES]


def summary_values(text: str) -> dict[str, str]:
    """Scrape `key: value` lines from the solver's initialisation summary."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        found.setdefault(key.strip(), value.strip())
    return found


def run(binary: str, arguments: list[str], environment: dict) -> dict[str, str]:
    completed = subprocess.run(
        [binary, *arguments, "--evaluations", "2", "--samples", "1",
         "--warmups", "0", "--no-direct", "--no-workload-comparison"],
        capture_output=True, text=True, env=environment)
    if completed.returncode != 0:
        print("  FAILED:", completed.stderr.strip().splitlines()[-1:],
              file=sys.stderr)
        return {}
    return summary_values(completed.stdout)


def cases():
    # One point case and one finite case, as the phase brief requires. The
    # point plan is the one whose canonical near field a procedural executor
    # never reads; the finite plan is the one whose exact tensors dominate.
    yield ("point/N32768/d4/cpu", [
        "--backend", "cpu-static-matrix", "--sources", "32768",
        "--targets", "32768", "--depth", "4", "--order", "6",
        "--precision", "float32"])
    yield ("prism/N4096/d3/regular/cpu", [
        "--backend", "cpu-static-matrix", "--sources", "4096",
        "--targets", "4096", "--depth", "3", "--order", "6",
        "--regular-grid", "--source-geometry", "prism",
        "--target-geometry", "prism", "--precision", "float32"])
    # The size Phase 3C named when it recorded derived packing as the largest
    # warm-cache cost (0.469 s of a 0.978 s warm setup at 32,768 bodies), so
    # that observation is re-measured at the size it was made at.
    yield ("prism/N32768/d4/regular/cpu", [
        "--backend", "cpu-static-matrix", "--sources", "32768",
        "--targets", "32768", "--depth", "4", "--order", "6",
        "--regular-grid", "--source-geometry", "prism",
        "--target-geometry", "prism", "--precision", "float32"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    rows = []
    for name, case_arguments in cases():
        cache_root = Path(tempfile.mkdtemp(prefix="cdfmm-phase3d-cache-"))
        try:
            cold = dict(os.environ, CDFMM_DISABLE_CACHE="1")
            warm = dict(os.environ, CDFMM_CACHE_DIR=str(cache_root))
            warm.pop("CDFMM_DISABLE_CACHE", None)

            for mode, environment in (("cold-no-cache", cold),
                                      ("cold-write", warm),
                                      ("warm-hit", warm)):
                print(">", name, mode, file=sys.stderr, flush=True)
                found = run(arguments.binary, case_arguments, environment)
                if not found:
                    continue
                row = {"case": name, "mode": mode}
                for stage in STAGES:
                    row[stage] = found.get(stage, "")
                rows.append(row)
        finally:
            shutil.rmtree(cache_root, ignore_errors=True)

    with Path(arguments.output).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {arguments.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
