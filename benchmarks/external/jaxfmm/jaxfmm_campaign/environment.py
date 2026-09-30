# SPDX-License-Identifier: Apache-2.0
"""Machine, process and framework fingerprints recorded with every row."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

#: Physical performance cores of the i9-14900KF, one logical CPU per core,
#: exactly as the FMM3D campaign pins its measured processes.
P_CORE_LOGICAL_CPUS = (0, 2, 4, 6, 8, 10, 12, 14)

#: Efficiency cores: every non-measured activity (installation, compilation
#: experiments, preflight, analysis) is confined to these.
E_CORE_LOGICAL_CPUS = tuple(range(16, 32))


def taskset_list(cpus: tuple[int, ...]) -> str:
    return ",".join(str(cpu) for cpu in cpus)


def current_affinity() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        return []


def core_class(affinity: list[int]) -> str:
    """Name the affinity set: performance, efficiency, mixed or unrestricted."""
    cpus = set(affinity)
    if not cpus or cpus == set(range(os.cpu_count() or 0)):
        return "unrestricted"
    if cpus <= set(P_CORE_LOGICAL_CPUS):
        return "performance"
    if cpus <= set(E_CORE_LOGICAL_CPUS):
        return "efficiency"
    return "mixed"


def git_revision(path: Path) -> dict[str, Any]:
    """Commit, describe string and dirtiness of a checkout (or ``None`` fields)."""
    result: dict[str, Any] = {"path": str(path), "commit": None, "describe": None, "dirty": None}
    git = shutil.which("git")
    if git is None or not Path(path).exists():
        return result
    try:
        result["commit"] = subprocess.run(
            [git, "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        result["describe"] = subprocess.run(
            [git, "-C", str(path), "describe", "--tags", "--always", "--dirty"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            [git, "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout
        result["dirty"] = bool(status.strip())
    except subprocess.CalledProcessError:
        pass
    return result


def nvidia_smi_query(fields: str, extra: list[str] | None = None) -> list[list[str]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    command = [executable, f"--query-{extra[0] if extra else 'gpu'}={fields}", "--format=csv,noheader,nounits"]
    try:
        output = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in output.strip().splitlines():
        if line.strip():
            rows.append([item.strip() for item in line.split(",")])
    return rows


def gpu_state() -> dict[str, Any]:
    """Current GPU identity, memory use and compute processes from nvidia-smi."""
    gpu_rows = nvidia_smi_query("name,driver_version,memory.used,memory.total")
    process_rows = nvidia_smi_query("pid,process_name,used_memory", ["compute-apps"])
    gpu = gpu_rows[0] if gpu_rows else ["unknown", "unknown", "nan", "nan"]
    processes = [
        {"pid": int(row[0]), "name": row[1], "used_mib": float(row[2]) if row[2] != "[N/A]" else None}
        for row in process_rows if len(row) >= 3
    ]
    return {
        "gpu_model": gpu[0],
        "driver_version": gpu[1],
        "memory_used_mib": float(gpu[2]),
        "memory_total_mib": float(gpu[3]),
        "compute_processes": processes,
    }


def cuda_toolkit_version() -> str | None:
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        return None
    try:
        output = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        return None
    for line in output.splitlines():
        if "release" in line:
            return line.split("release", 1)[1].split(",")[0].strip()
    return None


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def process_fingerprint() -> dict[str, Any]:
    affinity = current_affinity()
    return {
        "hostname": platform.node(),
        "os": platform.platform(),
        "cpu_model": cpu_model(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "affinity": affinity,
        "core_class": core_class(affinity),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "pid": os.getpid(),
    }


def jax_environment_variables() -> dict[str, str | None]:
    """Every allocator/precision knob that can change JAX behaviour."""
    names = (
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "JAX_PLATFORMS",
        "JAX_ENABLE_X64",
        "JAX_DEFAULT_MATMUL_PRECISION",
        "XLA_FLAGS",
        "JAX_COMPILATION_CACHE_DIR",
        "CUDA_VISIBLE_DEVICES",
    )
    return {name: os.environ.get(name) for name in names}
