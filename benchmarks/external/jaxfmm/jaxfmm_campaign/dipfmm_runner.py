# SPDX-License-Identifier: Apache-2.0
"""dip-fmm adapters for the jaxFMM comparison (imports the ``cdfmm`` extension).

Point-target cases use point-dipole sources on the FP32 ``CudaFull`` backend
with the production FP32 point policy stated explicitly: position-based
``PointGeometry`` P2P, procedural point expansions, general layout, spherical
basis. Finite cases use touching unit cubes (``body_fill = 1``) as sources and
targets, the ``prism_cells`` geometry of the FMM3D campaign's finite arms.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .timing import TimingProtocol, TimingSummary


def import_cdfmm():
    import cdfmm  # noqa: WPS433 - only importable inside the cdfmm environment

    return cdfmm


def module_report(cdfmm) -> dict[str, Any]:
    return {
        "cdfmm_module": cdfmm.__file__,
        "cuda_available": bool(cdfmm.cuda_available()),
        "cuda_full_available": bool(cdfmm.cuda_full_available()),
        "cuda_device": cdfmm.cuda_device_description() if cdfmm.cuda_available() else None,
        "one_mkl_available": bool(cdfmm.one_mkl_available()),
    }


def _enum(cdfmm, enum_name: str, value: str):
    return getattr(getattr(cdfmm, enum_name), value.upper())


def make_options(cdfmm, *, order: int, depth: int, precision: str = "float32",
                 backend: str = "cuda_full", source_geometry: str = "point",
                 target_geometry: str = "point", body_side: float | None = None,
                 timing_level: str = "off"):
    options = cdfmm.UniformFmmOptions()
    options.backend = _enum(cdfmm, "ExecutionBackend", backend)
    options.precision = _enum(cdfmm, "StaticPrecision", precision)
    options.expansion_basis = cdfmm.ExpansionBasis.SPHERICAL
    options.expansion_order = int(order)
    tree = options.tree
    tree.max_level = int(depth)
    options.tree = tree
    options.p2p_packing = cdfmm.P2PExecutionPacking.POINT_GEOMETRY if source_geometry == "point" \
        and target_geometry == "point" else cdfmm.P2PExecutionPacking.AUTO
    options.point_expansion_execution = cdfmm.PointExpansionExecution.PROCEDURAL if source_geometry == "point" \
        and target_geometry == "point" else cdfmm.PointExpansionExecution.AUTO
    options.spatial_layout = cdfmm.SpatialLayout.GENERAL
    options.timing_level = _enum(cdfmm, "TimingLevel", timing_level)
    options.enable_cache = False
    if source_geometry == "point":
        options.source_geometry = cdfmm.SourceGeometry.POINT_DIPOLE
        options.source_sizes = []
    elif source_geometry == "prism":
        options.source_geometry = cdfmm.SourceGeometry.RECTANGULAR_PRISM
        options.source_sizes = [cdfmm.RectangularPrism(body_side, body_side, body_side)]
    else:
        raise ValueError(f"unsupported source geometry: {source_geometry}")
    if target_geometry == "point":
        options.target_geometry = cdfmm.TargetGeometry.POINT
        options.target_sizes = []
    elif target_geometry == "prism":
        options.target_geometry = cdfmm.TargetGeometry.RECTANGULAR_PRISM
        options.target_sizes = [cdfmm.RectangularPrism(body_side, body_side, body_side)]
    else:
        raise ValueError(f"unsupported target geometry: {target_geometry}")
    return options


def describe_options(options) -> dict[str, Any]:
    return {
        "backend": str(options.backend).split(".")[-1],
        "precision": str(options.precision).split(".")[-1],
        "expansion_basis": str(options.expansion_basis).split(".")[-1],
        "expansion_order": int(options.expansion_order),
        "depth": int(options.tree.max_level),
        "p2p_packing": str(options.p2p_packing).split(".")[-1],
        "point_expansion_execution": str(options.point_expansion_execution).split(".")[-1],
        "spatial_layout": str(options.spatial_layout).split(".")[-1],
        "source_geometry": str(options.source_geometry).split(".")[-1],
        "target_geometry": str(options.target_geometry).split(".")[-1],
        "timing_level": str(options.timing_level).split(".")[-1],
        "enable_cache": bool(options.enable_cache),
    }


def warm_cuda_context(cdfmm) -> float:
    """Build and run a throw-away CUDA plan so context creation is not timed."""
    started = time.perf_counter()
    rng = np.random.default_rng(0)
    positions = rng.uniform(-1.0, 1.0, size=(512, 3))
    moments = rng.normal(size=(512, 3))
    options = make_options(cdfmm, order=4, depth=2)
    plan = cdfmm.UniformFmm(positions, positions, options)
    plan.evaluate(moments, target_source_indices=np.arange(512, dtype=np.int32))
    del plan
    return time.perf_counter() - started


def build_plan(cdfmm, positions: np.ndarray, options):
    """Construct the static plan (tree, operators, upload); returns (plan, seconds)."""
    started = time.perf_counter()
    plan = cdfmm.UniformFmm(positions, positions, options)
    return plan, time.perf_counter() - started


def plan_report(plan) -> dict[str, Any]:
    def plain(mapping):
        return {key: (float(value) if isinstance(value, (int, float)) else str(value))
                for key, value in dict(mapping).items()}

    report = {
        "resolved_backend": str(plan.backend).split(".")[-1],
        "resolved_p2p_packing": str(plan.p2p_execution_packing).split(".")[-1],
        "resolved_p2m_execution": str(plan.p2m_execution).split(".")[-1],
        "resolved_l2p_execution": str(plan.l2p_execution).split(".")[-1],
        "resolved_precision": str(plan.precision).split(".")[-1],
        "static_plan_statistics": plain(plan.static_plan_statistics),
    }
    try:
        report["cuda_plan_statistics"] = plain(plan.cuda_plan_statistics)
    except Exception as error:  # noqa: BLE001 - CPU plans have none
        report["cuda_plan_statistics"] = {"unavailable": str(error)}
    return report


def time_plan(plan, moments: np.ndarray, identities: np.ndarray | None,
              protocol: TimingProtocol) -> tuple[TimingSummary, np.ndarray, float]:
    """Warmups then timed host-to-host field updates through ``evaluate``.

    Each call passes NumPy moments in and receives a NumPy ``H`` back, so it
    includes the moment upload, the device evaluation, the synchronisation and
    the result download, the same boundary as the FMM3D campaign's dip-fmm
    driver. The first call after construction is timed separately.
    """
    kwargs = {} if identities is None else {"target_source_indices": identities}
    started = time.perf_counter()
    result = plan.evaluate(moments, **kwargs)["H"]
    first_call = time.perf_counter() - started
    for _ in range(protocol.warmups):
        result = plan.evaluate(moments, **kwargs)["H"]
    summary = TimingSummary()
    for _ in range(protocol.samples):
        started = time.perf_counter()
        for _ in range(protocol.evaluations):
            result = plan.evaluate(moments, **kwargs)["H"]
        summary.per_sample_seconds.append((time.perf_counter() - started) / protocol.evaluations)
    return summary, np.asarray(result, dtype=np.float64), first_call


def finite_dense_reference(cdfmm, positions: np.ndarray, moments: np.ndarray,
                           sample_indices: np.ndarray, body_side: float,
                           source_geometry: str, target_geometry: str,
                           pairs_per_block: float = 2.0e7) -> np.ndarray:
    """Exact FP64 finite field at the sampled targets from the solver's dense plan.

    ``DenseDirectPlan`` holds one pair tensor per source-target pair, so the
    sampled targets are processed in blocks sized from the source count, the
    same budget the FMM3D campaign's finite reference uses.
    """
    prism = cdfmm.RectangularPrism(body_side, body_side, body_side)
    source_kind = cdfmm.SourceGeometry.RECTANGULAR_PRISM if source_geometry == "prism" \
        else cdfmm.SourceGeometry.POINT_DIPOLE
    target_kind = cdfmm.TargetGeometry.RECTANGULAR_PRISM if target_geometry == "prism" \
        else cdfmm.TargetGeometry.POINT
    count = len(positions)
    block = max(1, int(pairs_per_block // max(count, 1)))
    field = np.zeros((len(sample_indices), 3), dtype=np.float64)
    for start in range(0, len(sample_indices), block):
        stop = min(start + block, len(sample_indices))
        indices = np.asarray(sample_indices[start:stop], dtype=np.int64)
        plan = cdfmm.DenseDirectPlan(
            positions, positions[indices],
            source_geometry=source_kind, target_geometry=target_kind,
            source_sizes=[prism] if source_geometry == "prism" else [],
            target_sizes=[prism] if target_geometry == "prism" else [],
            target_source_indices=[int(i) for i in indices],
        )
        result = plan.evaluate(moments)
        field[start:stop] = np.asarray(result["H"] if isinstance(result, dict) else result, dtype=np.float64)
        del plan
    return field


def feasible_depths(count: int, minimum_occupancy: float = 2.0,
                    maximum_occupancy: float = 4096.0) -> list[int]:
    """Depths whose mean leaf occupancy lies in the FMM3D campaign's window."""
    depths = []
    for depth in range(1, 12):
        occupancy = count / 8.0**depth
        if minimum_occupancy <= occupancy <= maximum_occupancy:
            depths.append(depth)
    return depths
