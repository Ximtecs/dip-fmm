# SPDX-License-Identifier: Apache-2.0
"""One jaxFMM campaign case in a fresh process: setup, first call, warm timing, accuracy.

Writes exactly one JSON row. Out-of-memory conditions and other failures are
recorded as rows with ``status`` ``oom`` or ``failed`` rather than propagated,
so the orchestrator never loses a case silently.
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

from jaxfmm_campaign import CAMPAIGN_REVISION, environment, geometry, reference  # noqa: E402
from jaxfmm_campaign.timing import TimingProtocol  # noqa: E402

OOM_MARKERS = ("resource_exhausted", "out of memory", "cuda_error_out_of_memory",
               "cudaerrormemoryallocation", "memoryerror", "cannot allocate memory", "bad_alloc")


def classify(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}".casefold()
    return "oom" if any(marker in text for marker in OOM_MARKERS) or isinstance(error, MemoryError) else "failed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--engine", default="kifmm")
    parser.add_argument("--p", type=int, default=6)
    parser.add_argument("--nmax", type=int, default=None)
    parser.add_argument("--state", default="random")
    parser.add_argument("--sample-targets", type=int, default=512)
    parser.add_argument("--sample-seed", type=int, default=geometry.DEFAULT_SEED)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--evaluations", type=int, default=5)
    parser.add_argument("--reference-cache", action="append", default=[])
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message=".*dtype uint64.*")
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    row: dict = {
        "framework": "jaxfmm",
        "campaign_revision": CAMPAIGN_REVISION,
        "case_id": args.case_id,
        "started_at": started_at,
        "status": "running",
        "case": {
            "framework": "jaxfmm", "engine": args.engine, "grid": args.grid, "p": args.p,
            "N_max": args.nmax, "state": args.state, "precision": "float32",
            "device": "gpu", "sample_targets": args.sample_targets, "sample_seed": args.sample_seed,
            "source_geometry": "point", "target_geometry": "point",
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

        from jaxfmm_campaign import jaxfmm_dipole

        row["precision_config"] = jaxfmm_dipole.configure_precision(fp32=True)
        row["devices"] = jaxfmm_dipole.device_report()
        row["versions"] = {"python": sys.version.split()[0], "jax": jax.__version__,
                           "jaxlib": row["devices"]["jaxlib_version"],
                           "jaxfmm": dist_version("jaxFMM"), "numpy": np.__version__}
        if row["devices"]["default_backend"] != "gpu" and not os.environ.get("JAXFMM_CAMPAIGN_ALLOW_CPU"):
            raise RuntimeError(f"JAX default backend is {row['devices']['default_backend']}, not gpu")
        row["memory_after_import"] = jaxfmm_dipole.memory_stats()

        dataset = geometry.lattice_dataset(args.grid, state=args.state)
        row["dataset"] = {"dataset_id": dataset.dataset_id, "sha256": dataset.sha256,
                          "n_sources": dataset.source_count, "n_targets": dataset.target_count,
                          "spec": dataset.spec()}
        cache_roots = [Path(p) for p in args.reference_cache] or [HERE.parent / "results" / "reference_cache"]
        reference_started = time.perf_counter()
        ref = reference.load_or_compute_reference(
            dataset_id=dataset.dataset_id, dataset_sha256=dataset.sha256,
            positions=dataset.positions, moments=dataset.moments,
            sample_targets=args.sample_targets, sample_seed=args.sample_seed,
            cache_roots=cache_roots)
        row["reference"] = {"kind": "numpy_direct_dipole_fp64", "source": ref["source"],
                            "seconds": time.perf_counter() - reference_started,
                            "sample_count": int(len(ref["sample_indices"]))}

        overrides = {"p": args.p}
        if args.nmax is not None:
            overrides["N_max"] = args.nmax
        evaluator = jaxfmm_dipole.build_evaluator(args.engine, dataset.positions, **overrides)
        row["jaxfmm_parameters"] = {k: (v if isinstance(v, (int, float, bool, str)) or v is None else str(v))
                                    for k, v in evaluator.parameters.items()}
        row["setup_info"] = evaluator.setup_info
        row["setup_seconds"] = evaluator.setup_seconds
        protocol = TimingProtocol(args.warmups, args.samples, args.evaluations)
        record = jaxfmm_dipole.time_evaluator(evaluator, dataset.moments, protocol)
        row.update(record.as_dict())
        field = record.field[ref["sample_indices"]]
        row["error_metrics"] = reference.error_metrics(field, ref["field"])
        row["output_dtype"] = "float32"
        row["field_finite"] = bool(np.isfinite(record.field).all())
        row["ns_per_body_device"] = 1e9 * record.device_timing.median / dataset.source_count
        row["ns_per_body_host"] = 1e9 * record.host_timing.median / dataset.source_count
        write("success")
        print(json.dumps({"case_id": args.case_id, "status": "success",
                          "device_median_s": record.device_timing.median,
                          "host_median_s": record.host_timing.median,
                          "relative_l2": row["error_metrics"]["relative_l2"]}))
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
