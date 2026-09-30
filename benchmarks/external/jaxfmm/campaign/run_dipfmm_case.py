# SPDX-License-Identifier: Apache-2.0
"""One dip-fmm campaign case in a fresh process (``cdfmm`` environment).

Point cases score against the shared FP64 direct reference; finite cases
(prism or tetrahedron sources and/or targets) score against the solver's own
exact FP64 dense plan evaluated at the same sampled targets, as the FMM3D
campaign's finite arms do. The geometry family, edge count and body fill
come from the command line, never from code.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from jaxfmm_campaign import CAMPAIGN_REVISION, dipfmm_runner, environment, geometry, reference  # noqa: E402
from jaxfmm_campaign.timing import TimingProtocol  # noqa: E402

OOM_MARKERS = ("out of memory", "cuda_error_out_of_memory", "cudaerrormemoryallocation",
               "memoryerror", "cannot allocate memory", "bad_alloc", "failed to allocate")


def classify(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}".casefold()
    return "oom" if any(marker in text for marker in OOM_MARKERS) or isinstance(error, MemoryError) else "failed"


def finite_reference_path(root: Path, dataset: geometry.Dataset, sample_targets: int, sample_seed: int,
                          geometry_tag: str, solver_sha: str) -> Path:
    payload = json.dumps({
        "dataset_sha256": dataset.sha256, "sample_targets": sample_targets, "sample_seed": sample_seed,
        "geometry": geometry_tag, "algorithm": "cdfmm_dense_direct_exact_fp64", "solver": solver_sha,
        "selection": reference.TARGET_SELECTION_VERSION,
        "moments": "V*M",   # finite bodies carry V * M; the jaxFMM finite worker uses the same key
    }, sort_keys=True)
    return root / ("finite_reference_" + hashlib.sha256(payload.encode()).hexdigest()[:24] + ".npz")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", default="lattice", choices=("lattice", "kuhn_mesh"))
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--body-fill", type=float, default=1.0)
    parser.add_argument("--source-body", default="point", choices=("point", "body"))
    parser.add_argument("--target-body", default="point", choices=("point", "body"))
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--precision", default="float32")
    parser.add_argument("--backend", default="cuda_full")
    parser.add_argument("--state", default="random")
    parser.add_argument("--seed", type=int, default=geometry.DEFAULT_SEED)
    parser.add_argument("--sample-targets", type=int, default=512)
    parser.add_argument("--sample-seed", type=int, default=geometry.DEFAULT_SEED)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--evaluations", type=int, default=5)
    parser.add_argument("--reference-cache", action="append", default=[])
    parser.add_argument("--solver-sha", default="unknown")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    finite = args.source_body != "point" or args.target_body != "point"
    if args.geometry == "lattice":
        body = "prism" if finite else "point"
    else:
        body = "tetra"
    spec = {"kind": args.geometry, "grid": args.grid, "spacing": args.spacing, "seed": args.seed,
            "state": args.state, "body": body, "body_fill": args.body_fill}
    row: dict = {
        "framework": "dipfmm",
        "campaign_revision": CAMPAIGN_REVISION,
        "case_id": args.case_id,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "running",
        "case": {
            "framework": "dipfmm", "geometry": args.geometry, "grid": args.grid, "spacing": args.spacing,
            "order": args.order, "depth": args.depth, "precision": args.precision, "backend": args.backend,
            "state": args.state, "source_geometry": "point" if args.source_body == "point" else body,
            "target_geometry": "point" if args.target_body == "point" else body,
            "body_fill": args.body_fill if finite else None,
            "device": "gpu" if "cuda" in args.backend else "cpu",
            "sample_targets": args.sample_targets, "sample_seed": args.sample_seed,
        },
        "protocol": {"warmups": args.warmups, "samples": args.samples, "evaluations": args.evaluations},
        "process": environment.process_fingerprint(),
        "gpu_before": environment.gpu_state(),
        "cuda_toolkit": environment.cuda_toolkit_version(),
        "solver_sha": args.solver_sha,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def write(status: str, **extra) -> None:
        row["status"] = status
        row["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        row.update(extra)
        temporary = out.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(row, indent=2, default=str) + "\n")
        temporary.replace(out)

    try:
        cdfmm = dipfmm_runner.import_cdfmm()
        row["module"] = dipfmm_runner.module_report(cdfmm)
        row["versions"] = {"python": sys.version.split()[0], "numpy": np.__version__}
        if "cuda" in args.backend:
            row["module"]["context_warm_seconds"] = dipfmm_runner.warm_cuda_context(cdfmm)

        dataset = geometry.build_dataset(spec)
        row["dataset"] = {"dataset_id": dataset.dataset_id, "sha256": dataset.sha256,
                          "n_sources": dataset.source_count, "n_targets": dataset.target_count,
                          "counts": dataset.counts, "spec": dataset.spec}
        if finite:
            # Finite bodies carry the moment V * M of a unit magnetisation M, so the
            # same physical bodies are what jaxFMM's face charges describe.
            import dataclasses

            volume = args.spacing**3 * (args.body_fill**3 if body == "prism" else 1.0 / 6.0)
            dataset = dataclasses.replace(dataset, moments=volume * dataset.moments)
            row["dataset"]["moments"] = f"V * M with V = {volume:.6g} per body"
        cache_roots = [Path(p) for p in args.reference_cache] or [HERE.parent / "results" / "reference_cache"]
        reference_started = time.perf_counter()
        if not finite:
            ref = reference.load_or_compute_reference(
                dataset_id=dataset.dataset_id, dataset_sha256=dataset.sha256,
                positions=dataset.positions, moments=dataset.moments,
                sample_targets=args.sample_targets, sample_seed=args.sample_seed,
                cache_roots=cache_roots)
            sample_indices = ref["sample_indices"]
            reference_field = ref["field"]
            row["reference"] = {"kind": "numpy_direct_dipole_fp64", "source": ref["source"]}
        else:
            sample_indices = reference.select_sample_targets(dataset.positions, args.sample_targets, args.sample_seed)
            tag = f"{row['case']['source_geometry']}->{row['case']['target_geometry']}"
            path = finite_reference_path(cache_roots[0], dataset, args.sample_targets, args.sample_seed,
                                         tag, args.solver_sha)
            if path.is_file():
                with np.load(path) as stored:
                    reference_field = stored["field"]
                    assert np.array_equal(stored["sample_indices"], sample_indices)
            else:
                reference_field = dipfmm_runner.finite_dense_reference(
                    cdfmm, dataset, sample_indices, args.source_body, args.target_body)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, field=reference_field, sample_indices=sample_indices)
            row["reference"] = {"kind": "cdfmm_dense_direct_exact_fp64", "source": str(path)}
        row["reference"]["seconds"] = time.perf_counter() - reference_started
        row["reference"]["sample_count"] = int(len(sample_indices))

        options = dipfmm_runner.make_options(
            cdfmm, dataset, order=args.order, depth=args.depth, precision=args.precision, backend=args.backend,
            source_body=args.source_body, target_body=args.target_body)
        row["dipfmm_options"] = dipfmm_runner.describe_options(options)
        plan, construction = dipfmm_runner.build_plan(cdfmm, dataset.positions, options)
        row["setup_seconds"] = construction
        row["plan"] = dipfmm_runner.plan_report(plan)
        row["gpu_with_plan"] = environment.gpu_state()
        protocol = TimingProtocol(args.warmups, args.samples, args.evaluations)
        summary, field, first_call = dipfmm_runner.time_plan(plan, dataset.moments, dataset.identity_map, protocol)
        row["first_call_seconds"] = first_call
        row.update(summary.as_dict("host_evaluation"))
        row["error_metrics"] = reference.error_metrics(field[sample_indices], reference_field)
        row["output_dtype"] = str(np.asarray(plan.evaluate(dataset.moments, target_source_indices=dataset.identity_map)["H"]).dtype)
        row["field_finite"] = bool(np.isfinite(field).all())
        row["ns_per_body_host"] = 1e9 * summary.median / dataset.source_count
        write("success")
        print(json.dumps({"case_id": args.case_id, "status": "success", "host_median_s": summary.median,
                          "setup_s": construction, "relative_l2": row["error_metrics"]["relative_l2"]}))
        return 0
    except BaseException as error:  # noqa: BLE001 - every failure becomes a row
        status = classify(error)
        write(status, failure_reason=f"{type(error).__name__}: {error}"[:4000],
              traceback=traceback.format_exc()[-6000:])
        print(json.dumps({"case_id": args.case_id, "status": status,
                          "error": f"{type(error).__name__}: {str(error)[:300]}"}))
        return 0 if status == "oom" else 1


if __name__ == "__main__":
    sys.exit(main())
