#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cold exact-prism FMM construction, with an optional prior-field comparison."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--order", type=int, default=10)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--backend", choices=("cpu_static", "cuda_full", "cuda_partial"),
                        default="cpu_static")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, help="Prior output JSON; compares saved fields")
    args = parser.parse_args()
    if args.grid < 1 or args.repetitions < 1:
        parser.error("grid and repetitions must be positive")
    sys.path.insert(0, str(args.build.resolve()))
    import cdfmm

    affinity = sorted(os.sched_getaffinity(0))
    repository = Path(__file__).resolve().parents[1]
    axis = np.arange(args.grid, dtype=np.float64) - 0.5 * (args.grid - 1)
    positions = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    size = cdfmm.RectangularPrism(0.8, 0.6, 0.4)
    options = cdfmm.UniformFmmOptions()
    options.backend = getattr(cdfmm.ExecutionBackend, args.backend.upper())
    options.precision = getattr(cdfmm.StaticPrecision, args.precision.upper())
    options.expansion_basis = cdfmm.ExpansionBasis.SPHERICAL
    options.expansion_order = args.order
    tree = options.tree
    tree.max_level = args.depth
    options.tree = tree
    options.enable_cache = False
    options.timing_level = cdfmm.TimingLevel.DETAILED
    options.source_geometry = cdfmm.SourceGeometry.RECTANGULAR_PRISM
    options.target_geometry = cdfmm.TargetGeometry.RECTANGULAR_PRISM
    options.source_sizes = [size]
    options.target_sizes = [size]
    options.spatial_layout = cdfmm.SpatialLayout.GENERAL
    generator = np.random.default_rng(314159)
    moments = generator.normal(size=positions.shape)
    moments /= np.linalg.norm(moments, axis=1)[:, None]
    rows = []
    for _ in range(args.repetitions):
        start = time.perf_counter()
        plan = cdfmm.UniformFmm(positions, positions, options)
        elapsed = time.perf_counter() - start
        stats = plan.static_plan_statistics
        rows.append({"wall_seconds": elapsed, **{
            key: value for key, value in stats.items()
            if key.endswith("seconds") or key.endswith("bytes")}})
        components = plan.evaluate_components(moments)
        del plan
    snapshots = {key: np.asarray(value) for key, value in components.items()}
    geometry_hash = hashlib.sha256(
        positions.tobytes() + np.asarray((0.8, 0.6, 0.4), dtype=np.float64).tobytes()
    ).hexdigest()
    report = {
        "benchmark_eligible": False,
        "kind": "cold exact-prism construction engineering measurement",
        "configuration": {key: value for key, value in vars(args).items()
                          if key not in ("output", "compare")},
        "module": cdfmm.__file__,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        "source_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repository, text=True).strip()),
        "physical_bodies": len(positions),
        "geometry_sha256": geometry_hash,
        "threads": os.environ.get("OMP_NUM_THREADS"),
        "cpu_affinity": affinity,
        "samples": rows,
        "median_seconds": {key: float(np.median([row[key] for row in rows]) )
                           for key in rows[0] if key.endswith("seconds")},
    }
    if args.compare:
        previous = json.loads(args.compare.read_text())
        if previous["geometry_sha256"] != geometry_hash:
            raise ValueError("comparison geometry differs")
        if previous["configuration"]["order"] != args.order:
            raise ValueError("comparison expansion order differs")
        before = np.load(args.compare.with_suffix(".npz"))
        report["field_comparison"] = {}
        for key, actual in snapshots.items():
            expected = before[key]
            difference = actual - expected
            denominator = np.linalg.norm(expected)
            report["field_comparison"][key] = {
                "bit_identical": bool(actual.shape == expected.shape and
                                      actual.dtype == expected.dtype and
                                      actual.tobytes() == expected.tobytes()),
                "relative_l2": float(np.linalg.norm(difference) / denominator)
                if denominator else float(np.linalg.norm(difference)),
                "max_absolute": float(np.max(np.abs(difference))),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, default=str)
        stream.write("\n")
    with args.output.with_suffix(".npz").open("xb") as stream:
        np.savez(stream, **snapshots)
    print(json.dumps({"output": str(args.output), "median_seconds": report["median_seconds"],
                      "field_comparison": report.get("field_comparison")}, indent=2))


if __name__ == "__main__":
    main()
