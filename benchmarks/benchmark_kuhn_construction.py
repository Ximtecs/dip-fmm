#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cold construction of the jaxFMM Kuhn fixture, with optional field snapshots.

This is an engineering construction measurement, not an Article1 comparison.
Run each build in a separate process; --build selects its Python extension.
"""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def kuhn_geometry(grid):
    """Six low-to-high diagonal simplices per cell, in campaign order."""
    axis = np.arange(grid + 1, dtype=np.float64) - 0.5 * grid
    nodes = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    cells = np.indices((grid, grid, grid)).reshape(3, -1).T
    tetrahedra = np.empty((6 * len(cells), 4, 3))
    for slot, permutation in enumerate(itertools.permutations(range(3))):
        first = np.zeros(3, dtype=int)
        first[permutation[0]] = 1
        second = first.copy()
        second[permutation[1]] = 1
        for vertex, offset in enumerate((np.zeros(3, dtype=int), first, second,
                                         np.ones(3, dtype=int))):
            pick = cells + offset
            tetrahedra[slot::6, vertex] = nodes[pick[:, 0], pick[:, 1], pick[:, 2]]
    centres = tetrahedra.mean(axis=1)
    return centres, tetrahedra - centres[:, None, :]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--order", type=int, default=10)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--backend", choices=("cpu_static", "cuda_full", "cuda_partial"),
                        default="cpu_static")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
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
    positions, vertices = kuhn_geometry(args.grid)
    records = [cdfmm.Tetrahedron(np.ascontiguousarray(row)) for row in vertices]
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
    options.source_geometry = cdfmm.SourceGeometry.TETRAHEDRON
    options.target_geometry = cdfmm.TargetGeometry.TETRAHEDRON
    options.source_tetrahedra = records
    options.target_tetrahedra = records
    options.spatial_layout = cdfmm.SpatialLayout.GENERAL
    generator = np.random.default_rng(314159)
    moments = generator.normal(size=positions.shape)
    moments /= 6.0 * np.linalg.norm(moments, axis=1)[:, None]
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
    report = {
        "benchmark_eligible": False,
        "kind": "cold Kuhn-mesh construction engineering measurement",
        "configuration": {key: value for key, value in vars(args).items()
                          if key not in ("output", "compare")},
        "module": cdfmm.__file__,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        "source_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repository, text=True).strip()),
        "physical_bodies": len(positions),
        "geometry_sha256": hashlib.sha256(positions.tobytes() + vertices.tobytes()).hexdigest(),
        "threads": os.environ.get("OMP_NUM_THREADS"),
        "cpu_affinity": affinity,
        "samples": rows,
        "median_seconds": {key: float(np.median([row[key] for row in rows]))
                           for key in rows[0] if key.endswith("seconds")},
    }
    if args.compare:
        previous = json.loads(args.compare.read_text())
        if previous["geometry_sha256"] != report["geometry_sha256"]:
            raise ValueError("comparison geometry differs")
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
