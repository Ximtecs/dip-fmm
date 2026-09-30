# SPDX-License-Identifier: Apache-2.0
"""One jaxFMM finite-source case in a fresh process: face-charge triangles at body centres.

The reference is the dip-fmm FP64 dense field of the same bodies at the same
sampled centres, computed and cached by the dip-fmm worker (``--reference``
names the cache root; the row is marked ``reference_pending`` when it is not
there yet and the orchestrator rescores it later).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from jaxfmm_campaign import CAMPAIGN_REVISION, environment, finite_sources, geometry, reference  # noqa: E402
from jaxfmm_campaign.timing import TimingProtocol  # noqa: E402

OOM_MARKERS = ("resource_exhausted", "out of memory", "cuda_error_out_of_memory",
               "cudaerrormemoryallocation", "memoryerror", "cannot allocate memory", "bad_alloc")


def classify(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}".casefold()
    return "oom" if any(marker in text for marker in OOM_MARKERS) or isinstance(error, MemoryError) else "failed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", default="lattice", choices=("lattice", "kuhn_mesh"))
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--spacing", type=float, default=1.0)
    parser.add_argument("--p", type=int, default=6)
    parser.add_argument("--near-deg", type=int, default=None)
    parser.add_argument("--nmax", type=int, default=None)
    parser.add_argument("--theta", type=float, default=None)
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
    warnings.filterwarnings("ignore", message=".*dtype uint64.*")

    row: dict = {
        "framework": "jaxfmm",
        "campaign_revision": CAMPAIGN_REVISION,
        "case_id": args.case_id,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "running",
        "case": {
            "framework": "jaxfmm", "engine": "element", "tier": "finite", "geometry": args.geometry,
            "grid": args.grid, "spacing": args.spacing, "p": args.p, "near_deg": args.near_deg,
            "N_max": args.nmax, "theta": args.theta, "state": args.state, "precision": "float32", "device": "gpu",
            "source_geometry": ("prism" if args.geometry == "lattice" else "tetra") + "_faces",
            "target_geometry": "point", "sample_targets": args.sample_targets, "sample_seed": args.sample_seed,
        },
        "protocol": {"warmups": args.warmups, "samples": args.samples, "evaluations": args.evaluations},
        "process": environment.process_fingerprint(),
        "jax_environment": environment.jax_environment_variables(),
        "gpu_before": environment.gpu_state(),
        "cuda_toolkit": environment.cuda_toolkit_version(),
        "jaxfmm_git": environment.git_revision(
            Path(os.environ.get("JAXFMM_SOURCE", "/home/mihaa/MagTense/dip-fmm/.external/jaxfmm"))),
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
        import jax
        from importlib.metadata import version as dist_version

        from jaxfmm_campaign import jaxfmm_dipole, jaxfmm_element

        row["precision_config"] = jaxfmm_dipole.configure_precision(fp32=True)
        row["devices"] = jaxfmm_dipole.device_report()
        row["versions"] = {"python": sys.version.split()[0], "jax": jax.__version__,
                           "jaxlib": row["devices"]["jaxlib_version"], "jaxfmm": dist_version("jaxFMM"),
                           "numpy": np.__version__}
        if row["devices"]["default_backend"] != "gpu" and not os.environ.get("JAXFMM_CAMPAIGN_ALLOW_CPU"):
            raise RuntimeError(f"JAX default backend is {row['devices']['default_backend']}, not gpu")

        mesh = finite_sources.build_body_mesh(args.geometry, args.grid, args.spacing)
        dataset = finite_sources.body_dataset(args.geometry, args.grid, args.spacing, args.state, args.seed)
        M, moments = finite_sources.magnetisation_and_moments(mesh, args.state, args.seed)
        sigma = mesh.face_charges(M)
        row["dataset"] = {"dataset_id": dataset.dataset_id, "sha256": dataset.sha256,
                          "n_sources": mesh.counts["sources_jaxfmm"], "n_targets": mesh.counts["targets"],
                          "n_bodies": mesh.counts["bodies"], "counts": mesh.counts, "spec": dataset.spec,
                          "source_representation": "constant face charge (M_left - M_right).n per unique face, "
                                                   "two triangles per cube face, one per tetrahedron face",
                          "total_charge": float(np.sum(sigma * mesh.area))}
        sample_indices = reference.select_sample_targets(mesh.centres, args.sample_targets, args.sample_seed)

        overrides = {"p": args.p}
        for name, value in (("near_deg", args.near_deg), ("N_max", args.nmax), ("theta", args.theta)):
            if value is not None:
                overrides[name] = value
        evaluator = jaxfmm_element.build_element_evaluator(mesh, **overrides)
        row["jaxfmm_parameters"] = {k: (v if isinstance(v, (int, float, bool, str)) or v is None else str(v))
                                    for k, v in evaluator.parameters.items()}
        row["setup_info"] = evaluator.setup_info
        row["setup_seconds"] = evaluator.setup_seconds
        protocol = TimingProtocol(args.warmups, args.samples, args.evaluations)
        record = jaxfmm_element.time_element_evaluator(evaluator, sigma, protocol)
        row.update(record.as_dict())
        row["field_finite"] = bool(np.isfinite(record.field).all())
        row["output_dtype"] = "float32"
        row["ns_per_body_device"] = 1e9 * record.device_timing.median / mesh.counts["bodies"]
        row["ns_per_body_host"] = 1e9 * record.host_timing.median / mesh.counts["bodies"]
        row["ns_per_triangle_host"] = 1e9 * record.host_timing.median / mesh.counts["sources_jaxfmm"]
        np.save(out.with_suffix(".sampled_field.npy"), record.field[sample_indices])
        np.save(out.with_suffix(".sample_indices.npy"), sample_indices)

        # score against the dip-fmm FP64 dense reference of the same bodies at the same centres
        from jaxfmm_campaign.reference import TARGET_SELECTION_VERSION
        import hashlib

        payload = json.dumps({
            "dataset_sha256": dataset.sha256, "sample_targets": args.sample_targets, "sample_seed": args.sample_seed,
            "geometry": f"{'prism' if args.geometry == 'lattice' else 'tetra'}->point",
            "algorithm": "cdfmm_dense_direct_exact_fp64", "solver": args.solver_sha, "selection": TARGET_SELECTION_VERSION,
            "moments": "V*M",
        }, sort_keys=True)
        key = "finite_reference_" + hashlib.sha256(payload.encode()).hexdigest()[:24] + ".npz"
        row["reference"] = {"kind": "cdfmm_dense_direct_exact_fp64", "key": key, "status": "pending"}
        for root in [Path(p) for p in args.reference_cache]:
            path = root / key
            if path.is_file():
                with np.load(path) as stored:
                    assert np.array_equal(stored["sample_indices"], sample_indices)
                    row["error_metrics"] = reference.error_metrics(record.field[sample_indices], stored["field"])
                row["reference"].update({"status": "scored", "source": str(path)})
                break
        write("success")
        print(json.dumps({"case_id": args.case_id, "status": "success",
                          "device_median_s": record.device_timing.median, "host_median_s": record.host_timing.median,
                          "relative_l2": row.get("error_metrics", {}).get("relative_l2"),
                          "triangles": mesh.counts["sources_jaxfmm"]}))
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
