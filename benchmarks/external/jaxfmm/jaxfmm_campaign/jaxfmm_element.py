# SPDX-License-Identifier: Apache-2.0
"""jaxFMM element-path adapter for uniformly magnetised bodies.

The bodies' unique faces become constant-charge triangles (per-triangle P1
nodal values all equal to ``sigma_f``), the targets are the body centres, and
jaxFMM's element tree (``jaxfmm.fem.element_farfield_setup`` +
``element_compile(field=True)``) evaluates ``H = -grad(phi)`` there. Stock
jaxFMM code throughout: nothing is re-implemented here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

import jax
import jax.numpy as jnp

from .finite_sources import BodyMesh
from .timing import TimingProtocol, TimingSummary

ELEMENT_DEFAULTS = {"p": 4, "theta": 0.5, "N_max": 128, "s": 3, "near_deg": None, "duffy_deg": 4,
                    "jit_connectivity": True, "mem_limit": 4 * 1024**3}


@dataclass
class ElementEvaluator:
    parameters: dict[str, Any]
    setup: dict
    forward: Callable
    setup_seconds: float
    setup_info: dict[str, Any] = field(default_factory=dict)

    def apply_device(self, surf_nodal_device):
        return self.forward(None, surf_nodal_device)

    def apply_host(self, sigma_host: np.ndarray) -> np.ndarray:
        surf_nodal = jnp.asarray(np.repeat(np.asarray(sigma_host, np.float32)[:, None], 3, axis=1))
        return np.asarray(self.apply_device(surf_nodal))


def constant_triangle_charges(sigma: np.ndarray) -> np.ndarray:
    """``(N_tri, 3)`` per-triangle nodal values, all equal, so the charge is constant."""
    return np.repeat(np.asarray(sigma, np.float32)[:, None], 3, axis=1)


def build_element_evaluator(mesh: BodyMesh, **overrides) -> ElementEvaluator:
    from jaxfmm.fem import element_compile, element_farfield_setup

    parameters = {**ELEMENT_DEFAULTS, **overrides}
    kwargs = {k: v for k, v in parameters.items() if v is not None and k != "mem_limit"}
    nodes = jnp.asarray(mesh.nodes.astype(np.float32))
    tris = jnp.asarray(mesh.triangles.astype(np.int32))
    eval_pts = jnp.asarray(mesh.centres.astype(np.float32))
    started = time.perf_counter()
    setup = element_farfield_setup(nodes, None, tris, eval_pts, **kwargs)
    # The stock default is unbounded, which lets large near/far stages materialise
    # enormous intermediates.  A fixed transient budget keeps the same native
    # jaxFMM algorithm but makes those stages batch their work deterministically.
    forward = element_compile(setup, field=True, mem_limit=parameters["mem_limit"])
    jax.block_until_ready([v for v in setup.values() if hasattr(v, "shape")])
    elapsed = time.perf_counter() - started
    info = {
        "near_deg_effective": int(setup["near_deg"]) if "near_deg" in setup else None,
        "theta": float(setup["theta"]) if "theta" in setup else None,
        "n_levels": int(len(setup["lvl_info"])) if "lvl_info" in setup else None,
        "near_pairs": int(np.asarray(setup["dir_cnct"]).shape[0]) if "dir_cnct" in setup else None,
        "mpl_pairs": int(np.asarray(setup["mpl_cnct"]).shape[0]) if "mpl_cnct" in setup else None,
        "tri_quad_points_far": int(setup["tri_qpos"].shape[1]) if "tri_qpos" in setup and hasattr(setup["tri_qpos"], "shape") and setup["tri_qpos"].ndim > 1 else None,
        "tri_quad_points_near": int(setup["tri_qpos_n"].shape[1]) if "tri_qpos_n" in setup and hasattr(setup["tri_qpos_n"], "shape") and setup["tri_qpos_n"].ndim > 1 else None,
        "evaluation_mem_limit_bytes": int(parameters["mem_limit"]),
    }
    return ElementEvaluator(parameters=parameters, setup=setup, forward=forward, setup_seconds=elapsed,
                            setup_info=info)


@dataclass
class ElementEvaluationRecord:
    first_call_seconds: float
    device_timing: TimingSummary
    host_timing: TimingSummary
    field: np.ndarray
    memory_after_setup: dict[str, Any] | None
    memory_after_first_call: dict[str, Any] | None
    memory_after_timing: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        row = {
            "first_call_seconds": self.first_call_seconds,
            "jit_seconds_estimate": self.first_call_seconds - self.device_timing.median,
            "memory_after_setup": self.memory_after_setup,
            "memory_after_first_call": self.memory_after_first_call,
            "memory_after_timing": self.memory_after_timing,
        }
        row.update(self.device_timing.as_dict("device_evaluation"))
        row.update(self.host_timing.as_dict("host_evaluation"))
        return row


def time_element_evaluator(evaluator: ElementEvaluator, sigma: np.ndarray,
                           protocol: TimingProtocol) -> ElementEvaluationRecord:
    """First call (compile + run), warmups, device-resident and host-to-host samples.

    ``host_evaluation`` starts from the per-triangle charges on the host (the
    state a micromagnetic loop would hand over) and ends with a NumPy field.
    Computing the face charges from M is a cheap NumPy step outside the clock
    for both codes (dip-fmm receives moments, jaxFMM receives sigma).
    """
    from .jaxfmm_dipole import memory_stats

    sigma32 = np.asarray(sigma, np.float32)
    memory_after_setup = memory_stats()
    surf_device = jax.device_put(constant_triangle_charges(sigma32))
    surf_device.block_until_ready()
    started = time.perf_counter()
    result = evaluator.apply_device(surf_device)
    result.block_until_ready()
    first_call = time.perf_counter() - started
    memory_after_first_call = memory_stats()
    for _ in range(protocol.warmups):
        evaluator.apply_device(surf_device).block_until_ready()
    device_timing = TimingSummary()
    for _ in range(protocol.samples):
        started = time.perf_counter()
        for _ in range(protocol.evaluations):
            result = evaluator.apply_device(surf_device)
            result.block_until_ready()
        device_timing.per_sample_seconds.append((time.perf_counter() - started) / protocol.evaluations)
    host_timing = TimingSummary()
    host_result = None
    for _ in range(protocol.samples):
        started = time.perf_counter()
        for _ in range(protocol.evaluations):
            host_result = evaluator.apply_host(sigma32)
        host_timing.per_sample_seconds.append((time.perf_counter() - started) / protocol.evaluations)
    return ElementEvaluationRecord(first_call, device_timing, host_timing, np.asarray(host_result, np.float64),
                                   memory_after_setup, memory_after_first_call, memory_stats())
