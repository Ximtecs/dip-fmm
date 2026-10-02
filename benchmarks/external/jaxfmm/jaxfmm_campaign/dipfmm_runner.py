# SPDX-License-Identifier: Apache-2.0
"""dip-fmm adapters for the jaxFMM comparison (imports the ``cdfmm`` extension).

Point-target cases use point-dipole sources on the FP32 ``CudaFull`` backend
with the production FP32 point policy stated explicitly: position-based
``PointGeometry`` P2P, procedural point expansions, general layout, spherical
basis. Finite cases use the dataset's bodies (touching cubes of the lattice,
or the face-touching Kuhn tetrahedra of the mesh) as sources and/or targets.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .geometry import Dataset
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


def _uniform(array: np.ndarray) -> bool:
    return bool(np.all(array == array[0]))


def prism_records(cdfmm, dataset: Dataset) -> list:
    """One ``RectangularPrism`` record when every body is the same cube."""
    side = dataset.body_side
    return [cdfmm.RectangularPrism(side, side, side)]


def tetrahedron_records(cdfmm, dataset: Dataset) -> list:
    """``Tetrahedron`` records: one when all bodies coincide, else one per body."""
    array = dataset.tetrahedra
    if _uniform(array):
        return [cdfmm.Tetrahedron(np.ascontiguousarray(array[0], dtype=np.float64))]
    return [cdfmm.Tetrahedron(np.ascontiguousarray(row, dtype=np.float64)) for row in array]


def make_options(cdfmm, dataset: Dataset, *, order: int, depth: int, precision: str = "float32",
                 backend: str = "cuda_full", source_body: str = "point", target_body: str = "point",
                 timing_level: str = "off"):
    """Build ``UniformFmmOptions`` for the dataset's bodies.

    ``source_body``/``target_body`` are ``point`` or ``body``; ``body`` takes
    the dataset's finite body (prism or tetrahedron).
    """
    options = cdfmm.UniformFmmOptions()
    options.backend = _enum(cdfmm, "ExecutionBackend", backend)
    options.precision = _enum(cdfmm, "StaticPrecision", precision)
    options.expansion_basis = cdfmm.ExpansionBasis.SPHERICAL
    options.expansion_order = int(order)
    tree = options.tree
    tree.max_level = int(depth)
    options.tree = tree
    point_pair = source_body == "point" and target_body == "point"
    options.p2p_packing = cdfmm.P2PExecutionPacking.POINT_GEOMETRY if point_pair \
        else cdfmm.P2PExecutionPacking.AUTO
    options.point_expansion_execution = cdfmm.PointExpansionExecution.PROCEDURAL if point_pair \
        else cdfmm.PointExpansionExecution.AUTO
    options.spatial_layout = cdfmm.SpatialLayout.GENERAL
    options.timing_level = _enum(cdfmm, "TimingLevel", timing_level)
    options.enable_cache = False
    options.source_sizes = []
    options.target_sizes = []
    options.source_tetrahedra = []
    options.target_tetrahedra = []
    if source_body == "point":
        options.source_geometry = cdfmm.SourceGeometry.POINT_DIPOLE
    elif dataset.body == "prism":
        options.source_geometry = cdfmm.SourceGeometry.RECTANGULAR_PRISM
        options.source_sizes = prism_records(cdfmm, dataset)
    elif dataset.body == "tetra":
        options.source_geometry = cdfmm.SourceGeometry.TETRAHEDRON
        options.source_tetrahedra = tetrahedron_records(cdfmm, dataset)
    else:
        raise ValueError(f"dataset has no finite body for source_body={source_body}")
    if target_body == "point":
        options.target_geometry = cdfmm.TargetGeometry.POINT
    elif dataset.body == "prism":
        options.target_geometry = cdfmm.TargetGeometry.RECTANGULAR_PRISM
        options.target_sizes = prism_records(cdfmm, dataset)
    elif dataset.body == "tetra":
        options.target_geometry = cdfmm.TargetGeometry.TETRAHEDRON
        options.target_tetrahedra = tetrahedron_records(cdfmm, dataset)
    else:
        raise ValueError(f"dataset has no finite body for target_body={target_body}")
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
        "source_records": len(options.source_sizes) + len(options.source_tetrahedra),
        "target_records": len(options.target_sizes) + len(options.target_tetrahedra),
        "timing_level": str(options.timing_level).split(".")[-1],
        "enable_cache": bool(options.enable_cache),
    }


def warm_cuda_context(cdfmm) -> float:
    """Build and run a throw-away CUDA plan so context creation is not timed."""
    from .geometry import lattice_dataset

    started = time.perf_counter()
    dataset = lattice_dataset(8)
    options = make_options(cdfmm, dataset, order=4, depth=2)
    plan = cdfmm.UniformFmm(dataset.positions, dataset.positions, options)
    plan.evaluate(dataset.moments, target_source_indices=dataset.identity_map)
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


def finite_dense_reference(cdfmm, dataset: Dataset, sample_indices: np.ndarray,
                           source_body: str, target_body: str,
                           pairs_per_block: float = 2.0e7) -> np.ndarray:
    """Exact FP64 finite field at the sampled targets from the solver's dense plan.

    ``DenseDirectPlan`` holds one pair tensor per source-target pair, so the
    sampled targets are processed in blocks sized from the source count, the
    same budget the FMM3D campaign's finite reference uses.
    """
    positions = dataset.positions
    source_kind = cdfmm.SourceGeometry.POINT_DIPOLE
    target_kind = cdfmm.TargetGeometry.POINT
    source_sizes: list = []
    target_sizes: list = []
    source_tets: list = []
    target_tets: list = []
    if source_body != "point":
        if dataset.body == "prism":
            source_kind, source_sizes = cdfmm.SourceGeometry.RECTANGULAR_PRISM, prism_records(cdfmm, dataset)
        else:
            source_kind, source_tets = cdfmm.SourceGeometry.TETRAHEDRON, tetrahedron_records(cdfmm, dataset)
    count = len(positions)
    block = max(1, int(pairs_per_block // max(count, 1)))
    field = np.zeros((len(sample_indices), 3), dtype=np.float64)
    for start in range(0, len(sample_indices), block):
        stop = min(start + block, len(sample_indices))
        indices = np.asarray(sample_indices[start:stop], dtype=np.int64)
        if target_body != "point":
            if dataset.body == "prism":
                target_kind, target_sizes = cdfmm.TargetGeometry.RECTANGULAR_PRISM, prism_records(cdfmm, dataset)
            else:
                target_kind = cdfmm.TargetGeometry.TETRAHEDRON
                records = tetrahedron_records(cdfmm, dataset)
                target_tets = records if len(records) == 1 else [records[int(i)] for i in indices]
        plan = cdfmm.DenseDirectPlan(
            positions, positions[indices],
            source_geometry=source_kind, target_geometry=target_kind,
            source_sizes=source_sizes, target_sizes=target_sizes,
            target_source_indices=[int(i) for i in indices],
            static_precision="float64",
            source_tetrahedra=source_tets, target_tetrahedra=target_tets,
        )
        result = plan.evaluate(dataset.moments)
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
