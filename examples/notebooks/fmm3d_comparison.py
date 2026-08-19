"""Small data-conversion helpers for the FMM3D comparison notebook."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite, sqrt
from time import perf_counter
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RelativeErrorMetrics:
    """Aggregate pointwise relative vector errors."""

    rms: float
    maximum: float


def is_out_of_memory_error(error: BaseException) -> bool:
    """Return whether an exception reports a host or CUDA allocation failure.

    pybind11 maps ``std::bad_alloc`` to ``MemoryError``, while CUDA allocation
    failures reach Python as ``RuntimeError`` with a runtime-specific message.
    Keep this predicate deliberately narrow so unrelated CUDA failures are not
    hidden by long benchmark sweeps.
    """

    if isinstance(error, MemoryError):
        return True

    message = str(error).casefold()
    allocation_markers = (
        "out of memory",
        "cannot allocate memory",
        "failed to allocate",
        "memory allocation",
        "cudaerrormemoryallocation",
        "cuda_error_out_of_memory",
        "std::bad_alloc",
        "bad allocation",
    )
    return any(marker in message for marker in allocation_markers)


def fmm3d_laplace_nterms(eps: float) -> int:
    """Return FMM3D v2.1.0's Laplace expansion order for ``eps``.

    This mirrors ``src/Laplace/l3dterms.f`` from the pinned source release.
    The public Python wrapper does not expose the selected ``nterms`` value.
    """

    if not isfinite(eps) or eps <= 0.0:
        raise ValueError("FMM3D eps must be a positive finite number")

    jfun = sqrt(3.0) / 2.0
    hfun = 1.0 / (1.5**2)
    for order in range(2, 1001):
        hfun /= 1.5
        jfun *= sqrt(3.0) / 2.0
        if jfun * hfun < eps:
            return order
    return 1


def validate_source_point_geometry(
    source_positions: np.ndarray,
    target_positions: np.ndarray,
    target_source_indices: np.ndarray,
) -> None:
    """Validate coincident source-point geometry and its identity map."""

    sources = np.asarray(source_positions, dtype=np.float64)
    targets = np.asarray(target_positions, dtype=np.float64)
    identities = np.asarray(target_source_indices)
    if sources.ndim != 2 or sources.shape[1:] != (3,):
        raise ValueError("source positions must have shape (N, 3)")
    if targets.shape != sources.shape:
        raise ValueError("source and target positions must have matching shapes")
    if not np.array_equal(targets, sources):
        raise ValueError("source and target positions must be identical")
    expected = np.arange(len(sources), dtype=np.int64)
    if identities.shape != expected.shape or not np.array_equal(
        identities, expected
    ):
        raise ValueError("target-source identities must equal arange(N)")


def fmm3d_source_fields(output: Any) -> np.ndarray:
    """Convert FMM3D source gradients to cdfmm magnetic-field ordering.

    FMM3D returns grad(phi) with component-major storage. cdfmm returns
    H = -grad(phi) with one row per target.
    """

    gradient = np.asarray(output.grad, dtype=np.float64)
    if gradient.ndim == 2 and gradient.shape[0] == 3:
        return np.ascontiguousarray(-gradient.T)
    if gradient.ndim == 3 and gradient.shape[1] == 3:
        return np.ascontiguousarray(-np.transpose(gradient, (0, 2, 1)))
    raise ValueError(
        "FMM3D gradient must have shape (3, N) or (nd, 3, N); "
        f"received {gradient.shape}"
    )


def relative_error_metrics(
    approximate: np.ndarray,
    reference: np.ndarray,
    denominator_floor: float = 1.0e-30,
) -> RelativeErrorMetrics:
    """Return RMS and maximum pointwise relative vector errors."""

    approximate = np.asarray(approximate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if approximate.shape != reference.shape or approximate.shape[-1] != 3:
        raise ValueError("field arrays must have matching (..., 3) shapes")
    numerator = np.linalg.norm(approximate - reference, axis=-1)
    denominator = np.maximum(
        np.linalg.norm(reference, axis=-1), denominator_floor
    )
    relative = numerator / denominator
    return RelativeErrorMetrics(
        rms=float(np.sqrt(np.mean(relative * relative))),
        maximum=float(np.max(relative, initial=0.0)),
    )


def timed_call(function: Callable[[], Any]) -> tuple[Any, float]:
    """Call a synchronous function and return its result and wall time."""

    start = perf_counter()
    result = function()
    return result, perf_counter() - start
