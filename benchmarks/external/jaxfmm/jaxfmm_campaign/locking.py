# SPDX-License-Identifier: Apache-2.0
"""Machine-wide benchmark exclusion and GPU cleanliness checks.

The FMM3D campaign serialises every timed measurement on this machine through
one ``flock`` file. Holding the same lock here guarantees that no Article1
job measures while a jaxFMM or dip-fmm case runs, and vice versa. The lock
path is configurable so the campaign still works on a machine without the
Article1 checkout.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import os
from pathlib import Path
from typing import Any, Iterator

from . import environment

DEFAULT_LOCK_PATH = Path("/home/mihaa/MagTense/dip-fmm/Article1/.article1_benchmark.lock")


def lock_path() -> Path:
    return Path(os.environ.get("CDFMM_BENCHMARK_LOCK", str(DEFAULT_LOCK_PATH)))


@contextlib.contextmanager
def exclusive_benchmark_lock(enabled: bool = True) -> Iterator[None]:
    if not enabled:
        yield
        return
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        stamp = dt.datetime.now(dt.timezone.utc).isoformat()
        os.write(handle, f"{os.getpid()} jaxfmm-campaign {stamp}\n".encode())
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        os.close(handle)


def gpu_is_clean(baseline_mib: float, tolerance_mib: float = 256.0) -> tuple[bool, dict[str, Any]]:
    """True when no compute process is on the GPU and memory sits at the baseline."""
    state = environment.gpu_state()
    processes = state["compute_processes"]
    memory_ok = abs(state["memory_used_mib"] - baseline_mib) <= tolerance_mib
    return (not processes) and memory_ok, state
