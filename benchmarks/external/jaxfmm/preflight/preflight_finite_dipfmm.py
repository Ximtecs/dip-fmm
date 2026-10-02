# SPDX-License-Identifier: Apache-2.0
"""dip-fmm half of the finite-source preflight (``cdfmm`` environment).

Same bodies (touching cubes or Kuhn tetrahedra) and states, evaluated with
the FP32 ``CudaFull`` finite-source plan at the body centres (self field
included, as for every finite source) and, for the cell-averaged observable,
at the bodies themselves. The FP64 dense plan supplies the exact reference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaxfmm_campaign import dipfmm_runner, environment, geometry, reference  # noqa: E402
from jaxfmm_campaign.timing import TimingProtocol  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="lattice", choices=("lattice", "kuhn_mesh"))
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--backend", default="cuda_full")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cdfmm = dipfmm_runner.import_cdfmm()
    report: dict = {"kind": args.kind, "grid": args.grid, "process": environment.process_fingerprint(),
                    "module": dipfmm_runner.module_report(cdfmm), "states": {}}
    if "cuda" in args.backend:
        report["module"]["context_warm_seconds"] = dipfmm_runner.warm_cuda_context(cdfmm)

    spec = {"kind": args.kind, "grid": args.grid,
            "body": "prism" if args.kind == "lattice" else "tetra", "body_fill": 1.0}
    dataset = geometry.build_dataset(spec)
    report["counts"] = dataset.counts
    volumes = np.ones(dataset.source_count) if dataset.body == "prism" else np.full(dataset.source_count, 1.0 / 6.0)
    fields: dict[str, np.ndarray] = {"centres": dataset.positions}
    all_targets = np.arange(dataset.source_count, dtype=np.int64)

    plans = {}
    for target_body in ("point", "body"):
        options = dipfmm_runner.make_options(cdfmm, dataset, order=args.order, depth=args.depth,
                                             backend=args.backend, source_body="body", target_body=target_body)
        plan, construction = dipfmm_runner.build_plan(cdfmm, dataset.positions, options)
        plans[target_body] = plan
        report[f"options_{target_body}"] = dipfmm_runner.describe_options(options)
        report[f"construction_seconds_{target_body}"] = construction
        report[f"plan_{target_body}"] = {k: v for k, v in dipfmm_runner.plan_report(plan).items()
                                        if not k.endswith("statistics")}

    protocol = TimingProtocol(warmups=2, samples=3, evaluations=3)
    for name in ("uniform_z", "random", "vortex"):
        M = geometry.magnetisation(dataset.positions, name, geometry.DEFAULT_SEED)
        moments = volumes[:, None] * M          # dip-fmm moment per body = V * M
        fields[f"M_{name}"] = M
        state: dict = {}
        started = time.perf_counter()
        # dense FP64 references with the SAME moments (self field included: no identity map)
        ref_point = _dense(cdfmm, dataset, moments, "point")
        ref_body = _dense(cdfmm, dataset, moments, "body")
        state["dense_reference_seconds"] = time.perf_counter() - started
        fields[f"Href_point_{name}"] = ref_point
        fields[f"Href_body_{name}"] = ref_body
        for target_body, plan in plans.items():
            summary, H, first = dipfmm_runner.time_plan(plan, moments, None, protocol)
            fields[f"H_{target_body}_{name}"] = H
            ref = ref_point if target_body == "point" else ref_body
            state[target_body] = {"first_call_seconds": first, **summary.as_dict("host_evaluation"),
                                  "vs_dense": reference.error_metrics(H, ref)}
        # the uniformly magnetised block: interior H must be close to -M/3 far from the boundary
        if name == "uniform_z":
            state["mean_Hz_over_Mz_point"] = float(np.mean(ref_point[:, 2]))
        report["states"][name] = state

    report["gpu_state_while_plans_alive"] = environment.gpu_state()
    np.savez(out / "dipfmm_finite_fields.npz", **fields)
    (out / "dipfmm_finite_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"counts": dataset.counts,
                      "random_point_vs_dense": report["states"]["random"]["point"]["vs_dense"]["relative_l2"]}))
    return 0


def _dense(cdfmm, dataset: geometry.Dataset, moments: np.ndarray, target_body: str) -> np.ndarray:
    """FP64 dense field of the finite bodies at every centre (point) or body (averaged)."""
    prism = [cdfmm.RectangularPrism(dataset.body_side, dataset.body_side, dataset.body_side)] \
        if dataset.body == "prism" else []
    tets = dipfmm_runner.tetrahedron_records(cdfmm, dataset) if dataset.body == "tetra" else []
    source_kind = cdfmm.SourceGeometry.RECTANGULAR_PRISM if dataset.body == "prism" else cdfmm.SourceGeometry.TETRAHEDRON
    if target_body == "point":
        target_kind, target_sizes, target_tets = cdfmm.TargetGeometry.POINT, [], []
    elif dataset.body == "prism":
        target_kind, target_sizes, target_tets = cdfmm.TargetGeometry.RECTANGULAR_PRISM, prism, []
    else:
        target_kind, target_sizes, target_tets = cdfmm.TargetGeometry.TETRAHEDRON, [], tets
    plan = cdfmm.DenseDirectPlan(dataset.positions, dataset.positions, source_geometry=source_kind,
                                 target_geometry=target_kind, source_sizes=prism, target_sizes=target_sizes,
                                 static_precision="float64", source_tetrahedra=tets, target_tetrahedra=target_tets)
    result = plan.evaluate(moments)
    return np.asarray(result["H"] if isinstance(result, dict) else result, dtype=np.float64)


if __name__ == "__main__":
    sys.exit(main())
