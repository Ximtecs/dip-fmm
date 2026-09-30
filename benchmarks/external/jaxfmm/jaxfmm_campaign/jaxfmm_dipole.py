# SPDX-License-Identifier: Apache-2.0
r"""jaxFMM adapters that evaluate the exact point-dipole field ``H = -grad(phi)``.

jaxFMM's kernels are scalar (one charge per source), so a vector dipole is
not a native source. The exact route used here is the derivative of the
*monopole* field with respect to the source positions: for unit charges,

    phi_charge(x) = sum_j G(x - y_j),           G(r) = 1/(4 pi |r|)
    d/d eps phi_charge(x)|_{y_j -> y_j + eps m_j}
        = sum_j m_j . grad_y G(x - y_j)
        = sum_j m_j . (x - y_j) / (4 pi |x - y_j|^3)  = phi_dipole(x),

and the same forward-mode derivative of the monopole field ``-grad_x phi``
gives ``H_dipole`` term by term. Applied to the FMM this is the classical
dipole extension of a kernel-independent or harmonic FMM: the P2M and P2P
stages are differentiated with respect to the source coordinates while the
box centres, the tree and the target coordinates stay fixed, and every
translation stage is linear. JAX's ``jvp`` performs exactly that; the
tangent flows only through the source positions.

Two engines are supported:

* ``flex`` — jaxFMM's rotation-based solid-harmonic engine. Its evaluator
  takes ``pts`` and ``eval_pts`` as separate arguments, so the stock
  ``eval_potential`` is differentiated directly (nothing re-implemented).
* ``kifmm`` — jaxFMM's kernel-independent engine on the cubic octree. Its
  driver takes one ``positions`` array for sources and targets, so a
  position derivative of the stock driver would also move the targets. This
  module therefore assembles the same phases through jaxFMM's stable
  ``CubicOperators``/``SourceData`` interface with separate source and
  target ``SourceData`` records. Every operator, pair list, batching and
  precision choice is the one ``kifmm.setup`` made; the padded (non-lean)
  layout is required, which holds far beyond the campaign's largest case.

The doubled-cloud check (``kifmm_doubled_cloud_field``) validates the split
driver against the *stock* evaluator on tiny problems: it appends a
zero-charge copy of the points as targets so the stock driver's position
derivative only moves the sources. That trick doubles the problem and is
never used for timing.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

import numpy as np

import jax
import jax.numpy as jnp

from .timing import TimingProtocol, TimingSummary

#: The knobs jaxFMM exposes per engine, recorded with every row.
KIFMM_DEFAULTS = {"p": 6, "N_max": 64, "max_depth": None, "batch_size": 16384,
                  "mem_limit": None, "p2p_mixed": True, "bin_gate": 1.2, "bin_K": 16,
                  "lean_threshold": 2e9}
FLEX_DEFAULTS = {"p": 6, "N_max": 128, "theta": 0.77, "s": 3, "m2l_mixed": None,
                 "p2p_mixed": None, "jit_cnct": True}


def configure_precision(fp32: bool = True) -> dict[str, Any]:
    """Pin the JAX precision policy and report it.

    ``jax_enable_x64=False`` makes every array float32. jaxFMM's architecture
    notes warn that recent JAX defaults float32 matmuls to TF32 on NVIDIA
    GPUs; ``JAX_DEFAULT_MATMUL_PRECISION=highest`` restores true FP32 so the
    comparison with dip-fmm's FP32 cuBLAS path is like for like. The value
    is read from the environment so the launcher, not this module, owns it.
    """
    jax.config.update("jax_enable_x64", not fp32)
    return {
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_default_matmul_precision": os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        "jax_compilation_cache_dir": jax.config.jax_compilation_cache_dir,
    }


def device_report() -> dict[str, Any]:
    devices = jax.devices()
    first = devices[0]
    return {
        "jax_version": jax.__version__,
        "jaxlib_version": __import__("jaxlib").__version__,
        "default_backend": jax.default_backend(),
        "devices": [f"{d.platform}:{d.id}:{getattr(d, 'device_kind', '?')}" for d in devices],
        "device_kind": getattr(first, "device_kind", None),
        "platform": first.platform,
    }


def memory_stats() -> dict[str, Any] | None:
    stats = jax.devices()[0].memory_stats()
    if not stats:
        return None
    keys = ("bytes_in_use", "peak_bytes_in_use", "bytes_limit", "largest_free_block_bytes",
            "num_allocs", "pool_bytes")
    return {key: int(stats[key]) for key in keys if key in stats}


# --------------------------------------------------------------------------
# KIFMM: source/target-split forward built from jaxFMM's stable phase interface
# --------------------------------------------------------------------------

def _kifmm_tree_and_operators(s: dict[str, Any]):
    from jaxfmm import core
    from jaxfmm.kifmm.assemble import make_operators

    tree = core.CubicTree(s["m_id"], s["p_id"], s["lvls"], s["is_leaf"], s["pil"],
                          s["domain_extent"], s["pmin"], int(s["max_depth"]))
    mem_limit = s["mem_limit"]
    op_set = make_operators(
        s["ops"], s["V"], W_list=s["W"], X_list=s["X"], batch_size=s["batch_size"],
        m2l_groups=s["m2l_groups"],
        mem_limit=(float(mem_limit) if mem_limit != jnp.inf else jnp.inf),
        field=True, pbc_op=s.get("pbc_op"), img_cnct=s.get("img_cnct", jnp.array([[]])),
    )
    return tree, op_set


def kifmm_requires_lean(s: dict[str, Any]) -> bool:
    sorted_meta = s.get("sorted_meta")
    if sorted_meta is None:
        return False
    pos_sorted = len(sorted_meta) > 4 and bool(sorted_meta[4])
    padded_bytes = int(np.prod(s["pil"].shape)) * 4 * jnp.asarray(s["positions"]).dtype.itemsize
    return pos_sorted or padded_bytes >= 2e9


def kifmm_split_monopole_field(s: dict[str, Any], source_positions, target_positions, charges):
    """The stock cubic driver's padded path with separate source/target records."""
    from jaxfmm import core
    from jaxfmm.near import _build_padded, eval_direct, eval_direct_binned
    from jaxfmm.trees import box_geometry

    tree, ops = _kifmm_tree_and_operators(s)
    centres, h = box_geometry(tree.m_id, tree.lvls, tree.domain_extent, tree.pmin)
    centres = centres.astype(source_positions.dtype)
    h = h.astype(source_positions.dtype)
    padded_sources, padded_charges, rev = _build_padded(source_positions, charges, tree.pil)
    padded_targets, _, _ = _build_padded(target_positions, charges, tree.pil)
    src = core.SourceData(source_positions, charges, padded_sources, padded_charges, rev, centres, h)
    tgt = core.SourceData(target_positions, charges, padded_targets, padded_charges, rev, centres, h)

    up = ops.s2m(src, tree)
    up = ops.m2m(up, tree)
    dn = ops.m2l(up, tree)
    if ops.pbc is not None:
        dn = ops.pbc(dn, up, tree)
    if ops.p2l is not None:
        dn = ops.p2l(dn, src, tree)
    dn = ops.l2l(dn, tree)
    far = ops.l2p(dn, tgt, tree)

    dir_cnct = s["dir_cnct"]
    mixed = bool(s["p2p_mixed"])
    cen = centres if mixed else None
    img_cnct = s.get("img_cnct", jnp.array([[]]))
    if jnp.asarray(dir_cnct).size == 0:
        near = jnp.zeros((target_positions.shape[0], 3), dtype=target_positions.dtype)
    elif s["p2p_bins"] is not None:
        near = eval_direct_binned(padded_sources, padded_charges, padded_targets, rev, dir_cnct,
                                  s["p2p_bins"][0], s["p2p_bins"][1], img_cnct=img_cnct,
                                  field=True, mem_limit=s["mem_limit"], mixed=mixed, centres=cen)
    else:
        near = eval_direct(padded_sources, padded_charges, padded_targets, rev, dir_cnct,
                           img_cnct=img_cnct, field=True, mem_limit=s["mem_limit"], mixed=mixed,
                           centres=cen)
    out = far + near
    if ops.m2p is not None:
        out = out + ops.m2p(up, tgt, tree)
    return out


@partial(jax.jit, static_argnames=["skel"])
def _kifmm_dipole_fwd(arrs, moments, skel):
    from jaxfmm.util import join_static

    s = join_static(skel, arrs)
    positions = s["positions"]
    charges = jnp.ones((positions.shape[0],), positions.dtype)

    def monopole_field(source_positions):
        return kifmm_split_monopole_field(s, source_positions, positions, charges)

    return jax.jvp(monopole_field, (positions,), (moments,))[1]


@partial(jax.jit, static_argnames=["skel"])
def _kifmm_stock_dipole_fwd(arrs, charges, tangent, skel):
    """Position derivative of the STOCK driver (moves sources and targets alike)."""
    from jaxfmm.kifmm.setup import evaluate_setup
    from jaxfmm.util import join_static

    s = join_static(skel, arrs)

    def monopole_field(positions):
        return evaluate_setup({**s, "positions": positions}, charges, field=True)

    return jax.jvp(monopole_field, (s["positions"],), (tangent,))[1]


# --------------------------------------------------------------------------
# flex: the stock evaluator already separates pts from eval_pts
# --------------------------------------------------------------------------

@partial(jax.jit, static_argnames=["skel"])
def _flex_dipole_fwd(arrs, moments, skel):
    from jaxfmm.flex.fmm import eval_potential
    from jaxfmm.util import join_static

    h = join_static(skel, arrs)
    positions = h["pts"]
    charges = jnp.ones((positions.shape[0],), positions.dtype)

    def monopole_field(source_positions):
        return eval_potential(charges, **{**h, "pts": source_positions}, field=True)

    return jax.jvp(monopole_field, (positions,), (moments,))[1]


# --------------------------------------------------------------------------
# evaluator objects and timing
# --------------------------------------------------------------------------

@dataclass
class DipoleEvaluator:
    engine: str
    parameters: dict[str, Any]
    skel: Any
    arrs: tuple
    forward: Callable
    setup_seconds: float
    setup_info: dict[str, Any] = field(default_factory=dict)

    def apply_device(self, moments_device):
        return self.forward(self.arrs, moments_device, self.skel)

    def apply_host(self, moments_host: np.ndarray) -> np.ndarray:
        moments_device = jnp.asarray(moments_host)
        return np.asarray(self.apply_device(moments_device))


def build_evaluator(engine: str, positions: np.ndarray, **overrides) -> DipoleEvaluator:
    """Run the engine's own ``setup`` on the point cloud and freeze it."""
    from jaxfmm.util import split_static

    positions = jnp.asarray(np.ascontiguousarray(positions, dtype=np.float32))
    started = time.perf_counter()
    if engine == "kifmm":
        from jaxfmm import kifmm

        parameters = {**KIFMM_DEFAULTS, **overrides}
        setup = kifmm.setup(positions, **parameters)
        info = {
            "max_depth": int(setup["max_depth"]),
            "maxleaf": int(setup["maxleaf"]),
            "adaptive": bool(setup["adaptive"]),
            "bin_ratio": float(setup["bin_ratio"]),
            "near_field": "binned" if setup["p2p_bins"] is not None else "flat",
            "lean": kifmm_requires_lean(setup),
            "mem_limit_bytes": float(setup["mem_limit"]),
            "p2p_mixed": bool(setup["p2p_mixed"]),
            "n_boxes": int(setup["m_id"].shape[0]),
            "near_pairs": int(np.asarray(setup["dir_cnct"]).shape[0]),
        }
        if info["lean"]:
            raise NotImplementedError(
                "kifmm.setup selected the lean leaf-major layout; the split source/target "
                "driver supports the padded layout only"
            )
        forward = _kifmm_dipole_fwd
    elif engine == "flex":
        from jaxfmm import flex

        parameters = {**FLEX_DEFAULTS, **overrides}
        setup_kwargs = {k: v for k, v in parameters.items() if k not in ("m2l_mixed", "p2p_mixed")}
        setup = flex.setup(positions, **setup_kwargs)
        # The mixed-precision flags are evaluation-time arguments in the stock
        # frozen evaluator; carry the production defaults (None = auto).
        setup = {**setup, "m2l_mixed": parameters["m2l_mixed"], "p2p_mixed": parameters["p2p_mixed"]}
        info = {
            "n_levels": int(len(setup["lvl_info"])) if "lvl_info" in setup else None,
            "near_pairs": int(np.asarray(setup["dir_cnct"]).shape[0]) if "dir_cnct" in setup else None,
            "mpl_pairs": int(np.asarray(setup["mpl_cnct"]).shape[0]) if "mpl_cnct" in setup else None,
        }
        forward = _flex_dipole_fwd
    else:
        raise ValueError(f"unknown jaxFMM engine: {engine}")
    skel, arrs = split_static(setup)
    arrs = tuple(jnp.asarray(a) for a in arrs)
    jax.block_until_ready(arrs)
    elapsed = time.perf_counter() - started
    return DipoleEvaluator(engine=engine, parameters=parameters, skel=skel, arrs=arrs,
                           forward=forward, setup_seconds=elapsed, setup_info=info)


def kifmm_doubled_cloud_field(positions: np.ndarray, moments: np.ndarray, **overrides) -> np.ndarray:
    """Dipole field through the STOCK kifmm driver on a doubled cloud (validation only)."""
    from jaxfmm import kifmm
    from jaxfmm.util import split_static

    count = len(positions)
    doubled = np.concatenate([positions, positions]).astype(np.float32)
    charges = np.concatenate([np.ones(count), np.zeros(count)]).astype(np.float32)
    tangent = np.concatenate([moments, np.zeros_like(moments)]).astype(np.float32)
    parameters = {**KIFMM_DEFAULTS, **overrides}
    setup = kifmm.setup(jnp.asarray(doubled), **parameters)
    skel, arrs = split_static(setup)
    arrs = tuple(jnp.asarray(a) for a in arrs)
    out = _kifmm_stock_dipole_fwd(arrs, jnp.asarray(charges), jnp.asarray(tangent), skel)
    return np.asarray(out)[count:]


@dataclass
class EvaluationRecord:
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


def time_evaluator(evaluator: DipoleEvaluator, moments: np.ndarray,
                   protocol: TimingProtocol) -> EvaluationRecord:
    """First call (compile + run), warmups, then device-resident and host samples.

    Every timed call ends with ``block_until_ready`` so asynchronous dispatch
    cannot hide device work. ``device_evaluation`` keeps the moments and the
    result on the device; ``host_evaluation`` starts from a NumPy moment array
    and ends with a NumPy result, so it includes the H2D and D2H copies.
    """
    moments32 = np.ascontiguousarray(moments, dtype=np.float32)
    memory_after_setup = memory_stats()
    moments_device = jax.device_put(moments32)
    moments_device.block_until_ready()

    started = time.perf_counter()
    result = evaluator.apply_device(moments_device)
    result.block_until_ready()
    first_call = time.perf_counter() - started
    memory_after_first_call = memory_stats()

    for _ in range(protocol.warmups):
        evaluator.apply_device(moments_device).block_until_ready()

    device_timing = TimingSummary()
    for _ in range(protocol.samples):
        started = time.perf_counter()
        for _ in range(protocol.evaluations):
            result = evaluator.apply_device(moments_device)
            result.block_until_ready()
        device_timing.per_sample_seconds.append((time.perf_counter() - started) / protocol.evaluations)

    host_timing = TimingSummary()
    host_result = None
    for _ in range(protocol.samples):
        started = time.perf_counter()
        for _ in range(protocol.evaluations):
            host_result = evaluator.apply_host(moments32)
        host_timing.per_sample_seconds.append((time.perf_counter() - started) / protocol.evaluations)

    return EvaluationRecord(
        first_call_seconds=first_call,
        device_timing=device_timing,
        host_timing=host_timing,
        field=np.asarray(host_result, dtype=np.float64),
        memory_after_setup=memory_after_setup,
        memory_after_first_call=memory_after_first_call,
        memory_after_timing=memory_stats(),
    )
