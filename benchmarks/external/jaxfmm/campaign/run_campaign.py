#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a jaxFMM-versus-dip-fmm campaign: one fresh, pinned process per case.

For every case the orchestrator takes the machine-wide benchmark lock, waits
until the GPU shows no compute process and sits at its idle memory baseline,
launches the framework's worker pinned to the performance cores, and after the
worker exits waits for the GPU to return to that baseline before the next
case. A worker that dies without writing its row (for example a kernel OOM
kill) still gets a row, classified from its exit status and stderr.

    python benchmarks/external/jaxfmm/campaign/run_campaign.py \
        --config benchmarks/external/jaxfmm/campaign/config_fp32_order6.json

Run the orchestrator itself on the efficiency cores; it pins the workers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from jaxfmm_campaign import CAMPAIGN_REVISION, environment, locking  # noqa: E402
from jaxfmm_campaign.dipfmm_runner import feasible_depths  # noqa: E402

DEFAULT_JAXFMM_PYTHON = "/home/mihaa/.conda/envs/jaxfmm/bin/python"
DEFAULT_CDFMM_PYTHON = "/home/mihaa/.conda/envs/cdfmm/bin/python"
DEFAULT_CDFMM_BUILD = ("/home/mihaa/MagTense/dip-fmm/Article1/runtime/dip-fmm/"
                       "aa9d75fc37a35fa3846619bc0e13a99d4f117b67/build")
DEFAULT_ARTICLE1_REFERENCE_CACHE = "/home/mihaa/MagTense/dip-fmm/Article1/cache/references"

#: Production JAX configuration: allocator left at JAX's default (preallocation
#: on), true FP32 matmuls, CUDA first. Recorded verbatim in every row.
PRODUCTION_JAX_ENV = {
    "JAX_DEFAULT_MATMUL_PRECISION": "highest",
    "JAX_PLATFORMS": "cuda,cpu",
}

#: The FMM3D campaign's fixed thread/affinity policy for dip-fmm processes.
DIPFMM_THREAD_ENV = {
    "OMP_NUM_THREADS": "8",
    "OMP_PLACES": "{0},{2},{4},{6},{8},{10},{12},{14}",
    "OMP_PROC_BIND": "close",
    "OMP_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "PYTHONUNBUFFERED": "1",
}

OOM_MARKERS = ("resource_exhausted", "out of memory", "cuda_error_out_of_memory",
               "cudaerrormemoryallocation", "memoryerror", "cannot allocate memory", "bad_alloc")


def log(message: str, stream) -> None:
    stamp = dt.datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    stream.write(line + "\n")
    stream.flush()


def build_cases(config: dict) -> list[dict]:
    cases: list[dict] = []
    protocol = config["protocol"]
    for grid in config["grids"]:
        count = grid**3
        jconf = config["jaxfmm"]
        for engine in jconf["engines"]:
            for arm in jconf["arms"]:
                if arm.get("max_grid") and grid > arm["max_grid"]:
                    continue
                for nmax in arm["N_max"]:
                    cases.append({
                        "framework": "jaxfmm", "grid": grid, "engine": engine, "p": arm["p"], "N_max": nmax,
                        "role": arm.get("role", "primary"), "dof_per_box": arm.get("dof_per_box"),
                        "case_id": f"jaxfmm_{engine}_p{arm['p']}_nmax{nmax}_grid{grid}", "protocol": protocol,
                    })
        dconf = config["dipfmm"]
        for arm in dconf["arms"]:
            if arm.get("max_grid") and grid > arm["max_grid"]:
                continue
            order = arm["order"]
            depths = dconf.get("depths") or feasible_depths(
                count, maximum_occupancy=arm.get("max_occupancy", 4096))
            for depth in depths:
                cases.append({
                    "framework": "dipfmm", "grid": grid, "order": order, "depth": depth,
                    "precision": dconf["precision"], "backend": dconf["backend"],
                    "source_geometry": arm["source"], "target_geometry": arm["target"],
                    "body_fill": arm.get("body_fill", 1.0), "role": arm.get("role", "primary"),
                    "dof_per_box": arm.get("dof_per_box"),
                    "case_id": (f"dipfmm_{arm['source']}2{arm['target']}_{dconf['precision']}_"
                                f"{dconf['backend']}_o{order}_d{depth}_grid{grid}"),
                    "protocol": protocol,
                })
    return cases


def worker_command(case: dict, config: dict, args, out: Path) -> tuple[list[str], dict]:
    protocol = case["protocol"]
    common = ["--grid", str(case["grid"]), "--case-id", case["case_id"], "--out", str(out),
              "--sample-targets", str(config["sample_targets"]), "--state", config["state"],
              "--warmups", str(protocol["warmups"]), "--samples", str(protocol["samples"]),
              "--evaluations", str(protocol["evaluations"]),
              "--reference-cache", str(args.results / "reference_cache")]
    if args.article1_reference_cache and Path(args.article1_reference_cache).is_dir():
        common += ["--reference-cache", args.article1_reference_cache]
    cpus = environment.taskset_list(environment.P_CORE_LOGICAL_CPUS)
    if case["framework"] == "jaxfmm":
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.update(PRODUCTION_JAX_ENV)
        env.update(config.get("jaxfmm", {}).get("environment", {}))
        argv = ["taskset", "-c", cpus, args.jaxfmm_python, str(HERE / "run_jaxfmm_case.py"),
                "--engine", case["engine"], "--p", str(case["p"]), "--nmax", str(case["N_max"]), *common]
        return argv, env
    env = dict(os.environ)
    env["PYTHONPATH"] = args.cdfmm_build
    env.update(DIPFMM_THREAD_ENV)
    env["CDFMM_CACHE_DIR"] = str(args.results / "scratch_cache")
    argv = ["taskset", "-c", cpus, args.cdfmm_python, str(HERE / "run_dipfmm_case.py"),
            "--order", str(case["order"]), "--depth", str(case["depth"]),
            "--precision", case["precision"], "--backend", case["backend"],
            "--source-geometry", case["source_geometry"], "--target-geometry", case["target_geometry"],
            "--body-fill", str(case["body_fill"]), "--solver-sha", args.solver_sha, *common]
    return argv, env


def wait_for_clean_gpu(baseline: float, tolerance: float, timeout: float, stream) -> dict:
    deadline = time.time() + timeout
    while True:
        clean, state = locking.gpu_is_clean(baseline, tolerance)
        if clean or time.time() > deadline:
            return {"clean": clean, **state}
        log(f"GPU busy ({state['memory_used_mib']:.0f} MiB, {len(state['compute_processes'])} processes); waiting", stream)
        time.sleep(10.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--results", type=Path, default=None, help="results root (default results/<campaign>)")
    parser.add_argument("--jaxfmm-python", default=DEFAULT_JAXFMM_PYTHON)
    parser.add_argument("--cdfmm-python", default=DEFAULT_CDFMM_PYTHON)
    parser.add_argument("--cdfmm-build", default=os.environ.get("CDFMM_BUILD_DIR", DEFAULT_CDFMM_BUILD))
    parser.add_argument("--article1-reference-cache", default=DEFAULT_ARTICLE1_REFERENCE_CACHE)
    parser.add_argument("--only", default=None, help="substring filter on case ids")
    parser.add_argument("--frameworks", default="jaxfmm,dipfmm")
    parser.add_argument("--max-grid", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-lock", action="store_true")
    parser.add_argument("--case-timeout", type=float, default=3 * 3600.0)
    parser.add_argument("--gpu-tolerance-mib", type=float, default=256.0)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    campaign = config["campaign"]
    args.results = args.results or (ROOT / "results" / campaign)
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "rows").mkdir(exist_ok=True)
    args.solver_sha = Path(args.cdfmm_build).parent.name

    cases = build_cases(config)
    frameworks = set(args.frameworks.split(","))
    cases = [c for c in cases if c["framework"] in frameworks]
    if args.only:
        cases = [c for c in cases if args.only in c["case_id"]]
    if args.max_grid:
        cases = [c for c in cases if c["grid"] <= args.max_grid]

    stream = (args.results / "campaign.log").open("a")
    log(f"campaign {campaign} ({CAMPAIGN_REVISION}): {len(cases)} cases, results in {args.results}", stream)
    manifest = {
        "campaign": campaign, "campaign_revision": CAMPAIGN_REVISION, "config": config,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cdfmm_build": args.cdfmm_build, "solver_sha": args.solver_sha,
        "jaxfmm_python": args.jaxfmm_python, "cdfmm_python": args.cdfmm_python,
        "production_jax_env": PRODUCTION_JAX_ENV, "dipfmm_thread_env": DIPFMM_THREAD_ENV,
        "performance_cores": list(environment.P_CORE_LOGICAL_CPUS),
        "orchestrator": environment.process_fingerprint(),
        "gpu": environment.gpu_state(),
        "cuda_toolkit": environment.cuda_toolkit_version(),
        "dipfmm_git": environment.git_revision(ROOT.parents[2]),
        "jaxfmm_git": environment.git_revision(Path("/home/mihaa/MagTense/dip-fmm/.external/jaxfmm")),
    }
    (args.results / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    if args.dry_run:
        for case in cases:
            print(case["case_id"])
        return 0

    baseline = None
    for index, case in enumerate(cases, start=1):
        out = args.results / "rows" / f"{case['case_id']}.json"
        if out.exists():
            status = json.loads(out.read_text()).get("status")
            if status in ("success", "oom") or (status == "failed" and not args.retry_failed):
                log(f"[{index}/{len(cases)}] skip {case['case_id']} ({status})", stream)
                continue
        argv, env = worker_command(case, config, args, out)
        with locking.exclusive_benchmark_lock(enabled=not args.no_lock):
            if baseline is None:
                state = wait_for_clean_gpu(0.0, 1e9, 0.0, stream)
                baseline = state["memory_used_mib"]
                log(f"GPU idle baseline {baseline:.0f} MiB ({state['gpu_model']})", stream)
            before = wait_for_clean_gpu(baseline, args.gpu_tolerance_mib, 3600.0, stream)
            if not before["clean"]:
                log(f"[{index}/{len(cases)}] GPU never became clean; recording and continuing", stream)
            log(f"[{index}/{len(cases)}] run {case['case_id']}", stream)
            started = time.perf_counter()
            with (args.results / "rows" / f"{case['case_id']}.stderr").open("w") as err:
                try:
                    completed = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=err,
                                               text=True, timeout=args.case_timeout)
                    returncode, stdout = completed.returncode, completed.stdout
                except subprocess.TimeoutExpired:
                    returncode, stdout = -1, "timeout"
            elapsed = time.perf_counter() - started
            after = wait_for_clean_gpu(baseline, args.gpu_tolerance_mib, 120.0, stream)
        stderr_text = (args.results / "rows" / f"{case['case_id']}.stderr").read_text()
        if not out.exists():
            haystack = stderr_text.casefold()
            status = "oom" if (returncode == 137 or any(m in haystack for m in OOM_MARKERS)) else \
                ("timeout" if returncode == -1 else "failed")
            out.write_text(json.dumps({
                "framework": case["framework"], "campaign_revision": CAMPAIGN_REVISION,
                "case_id": case["case_id"], "case": case, "status": status,
                "failure_reason": f"worker exited {returncode} without a row",
                "stderr_tail": stderr_text[-4000:],
            }, indent=2, default=str) + "\n")
        row = json.loads(out.read_text())
        row.setdefault("case", {}).update({"role": case.get("role"), "dof_per_box": case.get("dof_per_box")})
        row["orchestrator"] = {"returncode": returncode, "wall_seconds": elapsed,
                               "gpu_before": before, "gpu_after": after, "stdout": stdout[-2000:]}
        out.write_text(json.dumps(row, indent=2, default=str) + "\n")
        log(f"[{index}/{len(cases)}] {row.get('status')} {case['case_id']} in {elapsed:.0f}s: {stdout.strip()[-300:]}", stream)
    shutil.rmtree(args.results / "scratch_cache", ignore_errors=True)
    log("campaign finished", stream)
    return 0


if __name__ == "__main__":
    sys.exit(main())
