"""Adaptive versus uniform tree on octree-refined Voronoi grain meshes with per-body prisms.

For each configuration the plan is built, evaluated for one moment state and compared with a
dense FP64 reference on a random subset of targets (relative L2 over that subset); construction
and evaluation times and the leaf-containment diagnostics are recorded.

    PYTHONPATH=build python examples/validation/adaptive_voronoi_sweep.py                 # synthetic mesh
    PYTHONPATH=build python examples/validation/adaptive_voronoi_sweep.py --npz mesh.npz  # MagTense adaptive mesh
        (npz with pos (N,3) [m], dims (N,3) [m], is_intergrain (N,), grain_id (N,))
    options: --orders 4 6 8 --capacities 16 32 64 --depths 4 5 6 --uniform-depths 3 4 5 --targets 512 --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

import cdfmm as c

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voronoi_octree_mesh import random_moments, voronoi_octree_mesh  # noqa: E402


def load_npz(path: Path, unit: float):
    z = np.load(path)
    centres = np.asarray(z["pos"], float) / unit
    sizes = np.asarray(z["dims"], float) / unit
    grain = np.asarray(z["grain_id"]).astype(int) - 1
    band = np.asarray(z["is_intergrain"]).astype(bool)
    return centres, sizes, grain, band


BACKEND = c.ExecutionBackend.CPU_STATIC


def make_options(order, sizes, basis, precision):
    options = c.UniformFmmOptions()
    options.expansion_order = order
    options.expansion_basis = basis
    options.precision = precision
    options.backend = BACKEND
    options.source_geometry = c.SourceGeometry.RECTANGULAR_PRISM
    options.target_geometry = c.TargetGeometry.RECTANGULAR_PRISM
    prisms = [c.RectangularPrism(*s) for s in sizes]
    options.source_sizes = prisms
    options.target_sizes = prisms
    return options


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", type=Path, default=None)
    p.add_argument("--unit", type=float, default=1e-9, help="divide npz coordinates by this (default: work in nm)")
    p.add_argument("--grains", type=int, default=30)
    p.add_argument("--base", type=int, default=12)
    p.add_argument("--levels", type=int, default=2)
    p.add_argument("--band", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--orders", type=int, nargs="+", default=[4, 6, 8])
    p.add_argument("--capacities", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--depths", type=int, nargs="+", default=[4, 5, 6])
    p.add_argument("--uniform-depths", type=int, nargs="+", default=[3, 4, 5])
    p.add_argument("--targets", type=int, default=512, help="dense reference on this many random targets")
    p.add_argument("--fp32", action="store_true", help="FP32 plans (reference stays FP64)")
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--repeats", type=int, default=5, help="timed evaluations after one warm-up (median reported)")
    p.add_argument("--backend", choices=("cpu", "cuda"), default="cpu", help="CPU_STATIC or CUDA_FULL")
    p.add_argument("--stop-if-busy", action="store_true",
                   help="stop before each plan if `job-scheduler status` reports a running job")
    args = p.parse_args(argv)
    global BACKEND
    BACKEND = c.ExecutionBackend.CUDA_FULL if args.backend == "cuda" else c.ExecutionBackend.CPU_STATIC

    if args.npz is not None:
        centres, sizes, grain, band = load_npz(args.npz, args.unit)
        rng = np.random.default_rng(1)
        axes = rng.normal(size=(grain.max() + 1, 3)); axes /= np.linalg.norm(axes, axis=1)[:, None]
        moments = axes[grain] * (np.prod(sizes, axis=1) * np.where(band, 0.3, 1.0))[:, None]
        box = float(np.max(centres + sizes / 2) - np.min(centres - sizes / 2))
        label = args.npz.name
    else:
        mesh = voronoi_octree_mesh(args.grains, args.base, args.levels, args.band, seed=args.seed)
        centres, sizes, box, label = mesh.centres, mesh.sizes, mesh.box, f"synthetic g{args.grains} b{args.base} l{args.levels}"
        moments = random_moments(mesh)
    n = len(centres)
    levels = np.unique(np.round(sizes[:, 0] / sizes[:, 0].min()).astype(int), return_counts=True)
    print(f"{label}: {n} prisms, size ratios {dict(zip(levels[0].tolist(), levels[1].tolist()))}, box {box:.4g}", flush=True)

    rng = np.random.default_rng(args.seed)
    subset = np.sort(rng.choice(n, size=min(args.targets, n), replace=False))
    prisms = [c.RectangularPrism(*s) for s in sizes]
    t0 = time.perf_counter()
    dense = c.DenseDirectPlan(centres, centres[subset], c.SourceGeometry.RECTANGULAR_PRISM, c.TargetGeometry.RECTANGULAR_PRISM,
                              prisms, [prisms[i] for i in subset], [], static_precision="float64")
    reference = np.asarray(dense.evaluate(moments, c.DenseDirectBackend.PORTABLE))
    print(f"dense FP64 reference on {len(subset)} targets: {time.perf_counter()-t0:.1f} s", flush=True)
    ref_norm = np.linalg.norm(reference)
    precision = c.StaticPrecision.FLOAT32 if args.fp32 else c.StaticPrecision.FLOAT64
    centre = c.Vec3(*((centres + sizes / 2).max(axis=0) + (centres - sizes / 2).min(axis=0)) / 2)
    half = 0.5 * box * (1 + 1e-6)

    rows = []

    def scheduler_busy() -> bool:
        import subprocess
        try:
            out = subprocess.run(["job-scheduler", "status"], capture_output=True, text=True, timeout=30).stdout
        except Exception:
            return False
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "running":
                return int(parts[1]) > 0
        return False

    def run(kind, order, build, extra):
        if args.stop_if_busy and scheduler_busy():
            print("job-scheduler has a running job: stopping the sweep", flush=True)
            raise SystemExit(3)
        t = time.perf_counter(); built = build(); t_build = time.perf_counter() - t
        plan, t_tree = built if isinstance(built, tuple) else (built, 0.0)
        field = np.asarray(plan.evaluate(moments)["H"])              # warm-up, also the accuracy sample
        times = []
        for _ in range(max(args.repeats, 1)):
            t = time.perf_counter(); plan.evaluate(moments); times.append(time.perf_counter() - t)
        t_eval = float(np.median(times))
        s = plan.static_plan_statistics
        err = float(np.linalg.norm(field[subset] - reference) / ref_norm)
        row = {"kind": kind, "order": order, **extra, "rel_l2": err, "build_s": t_build, "tree_s": t_tree,
               "eval_median_s": t_eval, "eval_min_s": float(min(times)), "eval_max_s": float(max(times)),
               "exceeding": int(s["source_bodies_exceeding_leaf"]), "max_ratio": float(s["max_body_leaf_extent_ratio"]),
               "p2p_interactions": int(s["p2p_interactions"]), "m2l_interactions": int(s["interactions"]),
               "far_field_MB": s["operator_bytes"] / 2**20, "near_field_MB": s["near_field_operator_bytes"] / 2**20,
               "p2p_index_MB": s["p2p_index_bytes"] / 2**20, "n_bodies": n, "precision": "fp32" if args.fp32 else "fp64",
               "backend": args.backend,
               "device_MB": plan.cuda_plan_statistics["persistent_device_bytes"] / 2**20 if args.backend == "cuda" else 0.0}
        rows.append(row)
        print(f"{kind:8s} order {order} {json.dumps(extra)}: rel_l2 {err:.2e}, build {t_build:6.1f} s (tree {t_tree:4.1f}), "
              f"eval median {t_eval*1e3:8.1f} ms [{min(times)*1e3:.1f}-{max(times)*1e3:.1f}], exceeding {row['exceeding']} "
              f"(max ratio {row['max_ratio']:.2f}), p2p {row['p2p_interactions']}, m2l {row['m2l_interactions']}, "
              f"near {row['near_field_MB']:.0f} MB + idx {row['p2p_index_MB']:.0f} MB, far {row['far_field_MB']:.0f} MB, "
              f"device {row['device_MB']:.0f} MB", flush=True)

    for order in args.orders:
        for depth in args.uniform_depths:
            def build_uniform(order=order, depth=depth):
                o = make_options(order, sizes, c.ExpansionBasis.SPHERICAL, precision); o.tree.max_level = depth
                return c.UniformFmm(centres, centres, o)
            run("uniform", order, build_uniform, {"depth": depth})
        for cap in args.capacities:
            for depth in args.depths:
                def build_adaptive(order=order, cap=cap, depth=depth):
                    to = c.AdaptiveTreeOptions(); to.max_particles_per_leaf = cap; to.max_depth = depth
                    to.root_centre = centre; to.root_half_width = half
                    tree = c.AdaptiveTree(centres, to)
                    plan = tree.build_fmm(make_options(order, sizes, c.ExpansionBasis.SPHERICAL, precision))
                    return plan, tree.tree_seconds + tree.interaction_seconds
                run("adaptive", order, build_adaptive, {"capacity": cap, "depth": depth})
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
        print("written", args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
