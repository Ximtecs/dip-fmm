#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""jaxFMM preflight gate: PASS/FAIL on a tiny problem, without any benchmark.

Runs the jaxFMM half and the dip-fmm half as two *separate* processes, each
confined to the efficiency cores and serialised behind the machine's
benchmark lock, then compares their fields with each other and with the
FP64 direct reference. Between the processes it checks that the GPU has
returned to its baseline memory use and that no worker process survives.

    python benchmarks/external/jaxfmm/preflight/run_preflight.py

exits 0 on PASS and 1 on FAIL and writes ``preflight/out/<stamp>/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from jaxfmm_campaign import environment, locking  # noqa: E402

DEFAULT_JAXFMM_PYTHON = "/home/mihaa/.conda/envs/jaxfmm/bin/python"
DEFAULT_CDFMM_PYTHON = "/home/mihaa/.conda/envs/cdfmm/bin/python"
DEFAULT_CDFMM_BUILD = ("/home/mihaa/MagTense/dip-fmm/Article1/runtime/dip-fmm/"
                       "aa9d75fc37a35fa3846619bc0e13a99d4f117b67/build")

#: Low-memory JAX configuration for debugging only; the campaign documents
#: its own allocator policy separately.
PREFLIGHT_JAX_ENV = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "JAX_DEFAULT_MATMUL_PRECISION": "highest",
    "JAX_PLATFORMS": "cuda,cpu",
}


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def run_worker(argv: list[str], env: dict[str, str], log: Path, cpus: str) -> dict:
    command = ["taskset", "-c", cpus, *argv]
    started = time.perf_counter()
    with log.open("w") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        pid = process.pid
        returncode = process.wait()
    return {"pid": pid, "returncode": returncode, "seconds": time.perf_counter() - started,
            "command": command}


def wait_for_gpu_baseline(baseline: float, pid: int, tolerance: float, seconds: float = 30.0) -> dict:
    deadline = time.time() + seconds
    while True:
        state = environment.gpu_state()
        alive = any(proc["pid"] == pid for proc in state["compute_processes"])
        at_baseline = abs(state["memory_used_mib"] - baseline) <= tolerance
        if (not alive and at_baseline) or time.time() > deadline:
            return {"worker_alive_on_gpu": alive, "memory_used_mib": state["memory_used_mib"],
                    "baseline_mib": baseline, "returned_to_baseline": at_baseline,
                    "compute_processes": state["compute_processes"]}
        time.sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", type=int, default=8, help="lattice edge of the tiny problem (default 8 -> 512)")
    parser.add_argument("--p", type=int, default=6, help="jaxFMM expansion parameter")
    parser.add_argument("--order", type=int, default=6, help="dip-fmm expansion order")
    parser.add_argument("--depth", type=int, default=2, help="dip-fmm tree depth for the tiny problem")
    parser.add_argument("--engines", default="kifmm,flex")
    parser.add_argument("--jaxfmm-python", default=DEFAULT_JAXFMM_PYTHON)
    parser.add_argument("--cdfmm-python", default=DEFAULT_CDFMM_PYTHON)
    parser.add_argument("--cdfmm-build", default=os.environ.get("CDFMM_BUILD_DIR", DEFAULT_CDFMM_BUILD))
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-lock", action="store_true", help="do not take the machine benchmark lock")
    parser.add_argument("--memory-tolerance-mib", type=float, default=64.0)
    args = parser.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out = Path(args.out) if args.out else HERE / "out" / stamp
    out.mkdir(parents=True, exist_ok=True)
    cpus = environment.taskset_list(environment.E_CORE_LOGICAL_CPUS)
    report: dict = {"stamp": stamp, "arguments": vars(args), "efficiency_cores": cpus, "checks": {}}
    checks = report["checks"]

    with locking.exclusive_benchmark_lock(enabled=not args.no_lock):
        before = environment.gpu_state()
        report["gpu_before"] = before
        baseline = before["memory_used_mib"]

        env = dict(os.environ)
        env.update(PREFLIGHT_JAX_ENV)
        env.pop("PYTHONPATH", None)
        jax_run = run_worker(
            [args.jaxfmm_python, str(HERE / "preflight_jaxfmm.py"), "--grid", str(args.grid),
             "--p", str(args.p), "--out", str(out), "--engines", args.engines],
            env, out / "jaxfmm_worker.log", cpus)
        report["jaxfmm_worker"] = jax_run
        report["gpu_after_jaxfmm"] = wait_for_gpu_baseline(baseline, jax_run["pid"], args.memory_tolerance_mib)
        checks["jaxfmm_process_exited_cleanly"] = jax_run["returncode"] == 0
        checks["gpu_memory_back_to_baseline_after_jaxfmm"] = (
            report["gpu_after_jaxfmm"]["returned_to_baseline"]
            and not report["gpu_after_jaxfmm"]["worker_alive_on_gpu"])

        env = dict(os.environ)
        env["PYTHONPATH"] = args.cdfmm_build
        env["OMP_NUM_THREADS"] = "8"
        dip_run = run_worker(
            [args.cdfmm_python, str(HERE / "preflight_dipfmm.py"), "--grid", str(args.grid),
             "--order", str(args.order), "--depth", str(args.depth), "--out", str(out)],
            env, out / "dipfmm_worker.log", cpus)
        report["dipfmm_worker"] = dip_run
        report["gpu_after_dipfmm"] = wait_for_gpu_baseline(baseline, dip_run["pid"], args.memory_tolerance_mib)
        checks["dipfmm_process_exited_cleanly"] = dip_run["returncode"] == 0
        checks["gpu_memory_back_to_baseline_after_dipfmm"] = (
            report["gpu_after_dipfmm"]["returned_to_baseline"]
            and not report["gpu_after_dipfmm"]["worker_alive_on_gpu"])

    # ---- compare the two halves ----------------------------------------------
    jax_report = json.loads((out / "jaxfmm_report.json").read_text()) if (out / "jaxfmm_report.json").exists() else {}
    dip_report = json.loads((out / "dipfmm_report.json").read_text()) if (out / "dipfmm_report.json").exists() else {}
    report["jaxfmm_report"] = jax_report
    report["dipfmm_report"] = dip_report
    stage = jax_report.get("stage", {})
    checks["jaxfmm_default_backend_is_gpu"] = bool(stage.get("A_default_backend_is_gpu"))
    checks["jaxfmm_device_is_cuda"] = bool(stage.get("A_device_is_cuda"))
    checks["jaxfmm_arrays_are_float32"] = bool(stage.get("A_default_dtype_is_float32")) and all(
        rec.get("dipole_output_dtype") == "float32" for rec in jax_report.get("engines", {}).values())
    checks["dipfmm_output_is_float32"] = dip_report.get("output_dtype") == "float32"
    checks["dipfmm_cuda_backend"] = dip_report.get("plan", {}).get("resolved_backend", "").upper().startswith("CUDA")

    comparisons: dict = {}
    if (out / "jaxfmm_fields.npz").exists() and (out / "dipfmm_fields.npz").exists():
        jax_fields = np.load(out / "jaxfmm_fields.npz")
        dip_fields = np.load(out / "dipfmm_fields.npz")
        for engine in jax_report.get("engines", {}):
            for state in ("random", "vortex"):
                comparisons[f"{engine}_vs_dipfmm_{state}"] = relative_l2(
                    jax_fields[f"{engine}_{state}"], dip_fields[f"dipfmm_{state}"])
                comparisons[f"{engine}_vs_direct_{state}"] = relative_l2(
                    jax_fields[f"{engine}_{state}"], dip_fields[f"reference_{state}"])
        for state in ("random", "vortex"):
            comparisons[f"dipfmm_vs_direct_{state}"] = relative_l2(
                dip_fields[f"dipfmm_{state}"], dip_fields[f"reference_{state}"])
    report["comparisons"] = comparisons

    gate = 1.0e-2  # accuracy gate for a tiny FP32 problem at order/p = 6; actual values are reported
    for engine, rec in jax_report.get("engines", {}).items():
        checks[f"{engine}_monopole_normalisation"] = rec.get("monopole_potential_relative_l2", 1.0) < gate
        checks[f"{engine}_dipole_finite"] = bool(rec.get("dipole_finite"))
        checks[f"{engine}_dipole_field_vs_direct"] = rec.get("dipole_field_vs_direct", {}).get("relative_l2", 1.0) < gate
        checks[f"{engine}_dipole_potential_sign"] = rec.get("dipole_potential_vs_direct_relative_l2", 1.0) < gate
        checks[f"{engine}_second_state_fixed_geometry"] = rec.get("second_state_vs_direct", {}).get("relative_l2", 1.0) < gate
        checks[f"{engine}_fixed_equals_rebuilt"] = rec.get("second_state_fixed_vs_rebuilt_relative_l2", 1.0) < 1.0e-4
        checks[f"{engine}_repeat_stable"] = rec.get("repeat_max_relative_difference", 1.0) < 1.0e-4
        checks[f"{engine}_vs_dipfmm"] = comparisons.get(f"{engine}_vs_dipfmm_random", 1.0) < gate
        if engine == "kifmm":
            checks["kifmm_split_driver_matches_stock_doubled_cloud"] = rec.get("split_vs_doubled_cloud_relative_l2", 1.0) < 1.0e-3
    checks["dipfmm_dipole_field_vs_direct"] = dip_report.get("dipole_field_vs_direct", {}).get("relative_l2", 1.0) < gate
    checks["dipfmm_second_state_fixed_geometry"] = dip_report.get("second_state_vs_direct", {}).get("relative_l2", 1.0) < gate

    passed = all(checks.values())
    report["result"] = "PASS" if passed else "FAIL"
    (out / "preflight_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")

    width = max(len(name) for name in checks)
    print(f"jaxFMM preflight  ({out})")
    for name, ok in checks.items():
        print(f"  [{'x' if ok else ' '}] {name.ljust(width)}")
    print("  values:")
    for key, value in sorted(comparisons.items()):
        print(f"    {key:40s} {value:.3e}")
    for engine, rec in jax_report.get("engines", {}).items():
        print(f"    {engine}: monopole_potential {rec.get('monopole_potential_relative_l2'):.2e}"
              f"  dipole_potential {rec.get('dipole_potential_vs_direct_relative_l2'):.2e}"
              f"  setup_info {rec.get('setup_info')}")
    print(f"result: {report['result']}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
