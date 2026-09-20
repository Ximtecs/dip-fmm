"""Geometry-derived storage estimates for the uniform FMM notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

import cdfmm


import os

STATIC_ENTRY_BYTES = 16
# StaticDipoleBlock stores two indices, nine FP64 coefficients, one identity
# marker, and the padding required by its native C++ layout.
P2P_BLOCK_BYTES = 88
# The SoA packing stores nine FP64 coefficients, one source index, and one
# byte-sized identity marker per interaction.
P2P_SOA_INTERACTION_BYTES = 77
# The CPU position-based P2P executor keeps, per OpenMP thread, six FP64 rows
# (position and moment components) and one sorted index per gathered source.
P2P_POSITION_SCRATCH_BYTES = 6 * 8 + 4
# Portable M2L block schedule: one (source, target slot, source level) record
# per interaction; the schedule exists once the transfer matrices exceed 1 MiB.
M2L_SCHEDULE_INTERACTION_BYTES = 12
M2L_SCHEDULE_MATRIX_LIMIT_BYTES = 1 << 20
M2L_SCHEDULE_MAX_BLOCK_VALUES = 4096
M2L_SCHEDULE_MAX_BLOCK_TARGETS = 128
VEC3_BYTES = 24
DOUBLE_BYTES = 8
INT_BYTES = 4
# Four contiguous int fields: target, target level, row begin, and row end.
CUDA_M2L_ACTIVE_ROW_BYTES = 4 * INT_BYTES


@dataclass(frozen=True)
class StorageEstimate:
    """Storage counts and byte-level estimates for one source-point tree."""

    particles: int
    order: int
    depth: int
    coefficients: int
    nodes: int
    occupied_source_nodes: int
    occupied_target_nodes: int
    transfer_classes: int
    m2l_interactions: int
    p2p_pairs: int
    host_static: dict[str, int]
    cuda_partial: dict[str, int]
    cuda_full: dict[str, int]

    @property
    def host_static_bytes(self) -> int:
        return sum(self.host_static.values())

    @property
    def cuda_partial_bytes(self) -> int:
        return sum(self.cuda_partial.values())

    @property
    def cuda_full_bytes(self) -> int:
        return sum(self.cuda_full.values())


def coefficient_count(order: int) -> int:
    """Return the total-degree Cartesian basis size through ``order``."""

    if order < 0:
        raise ValueError("order must be non-negative")
    return comb(order + 3, 3)


def p2m_or_l2p_entries_per_particle(order: int) -> int:
    """Return the generic number of field entries for one particle map."""

    if order < 0:
        raise ValueError("order must be non-negative")
    return 3 * comb(order + 2, 3)


def shift_entries(order: int) -> int:
    """Return entries in one triangular M2M or L2L translation."""

    if order < 0:
        raise ValueError("order must be non-negative")
    return comb(order + 6, 6)


def default_thread_count() -> int:
    """Return the OpenMP team size a plan built in this process would use."""

    value = os.environ.get("OMP_NUM_THREADS")
    if value:
        try:
            return max(1, int(value.split(",")[0]))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def translation_bank_bytes(order: int, depth: int) -> int:
    """Return the resident bytes of the level-scaled M2M and L2L column banks.

    The CPU hierarchy packs every child-class translation, for child levels
    ``1..depth``, as one value per (input column, output) pair inside the
    contiguous output range of that column. The ranges are structural, so
    they are probed once per input with a unit vector through the exposed
    dynamic translations; all eight classes share one structure.
    """

    coefficients = coefficient_count(order)
    child = np.array([-0.25, -0.25, -0.25])
    origin = np.zeros(3)
    values = 0
    for apply in (
        lambda unit: cdfmm.m2m(unit, child, origin, order),
        lambda unit: cdfmm.l2l(unit, origin, child, order),
    ):
        for index in range(coefficients):
            unit = np.zeros(coefficients)
            unit[index] = 1.0
            nonzero = np.flatnonzero(np.asarray(apply(unit)))
            if nonzero.size:
                values += int(nonzero.max() - nonzero.min() + 1)
    # `values` already sums the M2M and the L2L structure; each bank stores
    # per class: begin/end ints and (C + 1) offsets, plus one block offset per
    # (level, class) and a terminating entry.
    per_bank_metadata = 8 * (2 * coefficients * INT_BYTES + (coefficients + 1) * 8) + (
        depth * 8 + 1
    ) * 8
    return depth * 8 * values * DOUBLE_BYTES + 2 * per_bank_metadata


def _m2l_schedule_bytes(nodes, depth: int, coefficients: int, threads: int) -> int:
    """Return the bytes of the transfer-class-sorted portable M2L block schedule."""

    block_targets = min(max(M2L_SCHEDULE_MAX_BLOCK_VALUES // coefficients, 1),
                        M2L_SCHEDULE_MAX_BLOCK_TARGETS)
    minimum_blocks = 4 * max(1, threads)
    total_blocks = 0
    total_runs = 0
    total_interactions = 0
    total_targets = 0
    for level in range(depth + 1):
        level_nodes = [node for node in nodes if node.level == level]
        total_targets += len(level_nodes)
        if not level_nodes:
            continue
        level_block_targets = min(
            max((len(level_nodes) + minimum_blocks - 1) // minimum_blocks, 1),
            block_targets,
        )
        for begin in range(0, len(level_nodes), level_block_targets):
            block = level_nodes[begin:begin + level_block_targets]
            classes = set()
            for target in block:
                if target.target_count == 0:
                    continue
                for source_index in target.list2:
                    source = nodes[source_index]
                    if source.source_count == 0:
                        continue
                    classes.add((target.ix - source.ix, target.iy - source.iy,
                                 target.iz - source.iz))
                    total_interactions += 1
            total_runs += len(classes)
            total_blocks += 1
    return (
        (depth + 2) * INT_BYTES
        + 2 * (total_blocks + 1) * INT_BYTES
        + total_targets * INT_BYTES
        + (total_runs + 1) * INT_BYTES
        + total_runs * INT_BYTES
        + total_interactions * M2L_SCHEDULE_INTERACTION_BYTES
    )


def _make_tree(positions: np.ndarray, depth: int) -> cdfmm.UniformTree:
    options = cdfmm.UniformTreeOptions()
    options.max_level = depth
    options.root_centre = cdfmm.Vec3(0.0, 0.0, 0.0)
    options.root_half_width = 1.0
    return cdfmm.UniformTree(positions, positions, options)


def estimate_source_point_storage(
    positions: np.ndarray,
    order: int,
    depth: int,
    *,
    universal_translation_bank: bool = False,
    threads: int | None = None,
) -> StorageEstimate:
    """Estimate current host and CUDA storage without constructing an FMM plan.

    The estimate follows the containers retained by the present implementation,
    including the shared M2M/L2L templates on both cold construction and
    universal-cache hits. The host (``CpuStatic``) figures describe the derived
    CPU execution packing: dense P2M rows, level-scaled M2M/L2L column banks,
    flat L2P rows, the transfer-class M2L block schedule (when the matrix set
    exceeds 1 MiB) and the per-thread position-based P2P scratch that replaces
    stored pair tensors for point sources; ``threads`` defaults to the OpenMP
    team size of this process. It intentionally excludes allocator metadata,
    CUDA context storage, pinned staging buffers, and Python object overhead.
    """

    if not hasattr(cdfmm, "static_m2l_matrix"):
        raise RuntimeError(
            "The active cdfmm Python extension predates static_m2l_matrix. "
            "From the repository root, run `cmake --fresh --preset notebooks` "
            "and `cmake --build --preset notebooks -j`, then restart the "
            "notebook kernel."
        )

    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("positions must have shape (N, 3)")
    if depth < 1:
        raise ValueError("depth must be at least one")
    if np.any(points < -1.0) or np.any(points > 1.0):
        raise ValueError("positions must lie inside the fixed [-1, 1]^3 root")

    tree = _make_tree(points, depth)
    nodes = tree.nodes
    particle_count = len(points)
    coefficients = coefficient_count(order)
    particle_entries = p2m_or_l2p_entries_per_particle(order)
    translation_entries = shift_entries(order)

    groups: dict[tuple[int, int, int], int] = {}
    active_m2l_rows = 0
    for target in nodes:
        if target.level == 0 or target.target_count == 0:
            continue
        row_has_interaction = False
        for source_index in target.list2:
            source = nodes[source_index]
            if source.source_count == 0:
                continue
            row_has_interaction = True
            key = (
                target.ix - source.ix,
                target.iy - source.iy,
                target.iz - source.iz,
            )
            groups[key] = groups.get(key, 0) + 1
        active_m2l_rows += int(row_has_interaction)

    p2p_pairs = 0
    widest_neighbourhood = 0
    for target in nodes:
        if not target.is_leaf or target.target_count == 0:
            continue
        neighbours = sum(
            nodes[source_index].source_count for source_index in target.list1
        )
        p2p_pairs += target.target_count * neighbours
        widest_neighbourhood = max(widest_neighbourhood, neighbours)
    thread_count = default_thread_count() if threads is None else max(1, threads)
    # The per-thread stride is rounded up to whole cache lines of 16 sources.
    p2p_scratch_bytes = (
        thread_count * ((widest_neighbourhood + 15) // 16 * 16) * P2P_POSITION_SCRATCH_BYTES
    )

    m2l_interactions = sum(groups.values())
    occupied_source_nodes = sum(
        node.level > 0 and node.source_count > 0 for node in nodes
    )
    occupied_target_nodes = sum(
        node.level > 0 and node.target_count > 0 for node in nodes
    )

    p2p_static_bytes = (
        p2p_pairs * P2P_BLOCK_BYTES + (particle_count + 1) * INT_BYTES
    )
    p2p_soa_bytes = (
        p2p_pairs * P2P_SOA_INTERACTION_BYTES
        + (particle_count + 1) * INT_BYTES
    )
    # A populated far-field plan constructs the complete 316-class M2L bank.
    # ``universal_translation_bank`` also requests that bank for a geometry
    # with no active M2L rows. The eight shared M2M/L2L templates remain
    # resident regardless of whether they were built or loaded from cache.
    matrix_count = 316 if universal_translation_bank or groups else 0
    shared_translation_bytes = 2 * 8 * translation_entries * STATIC_ENTRY_BYTES
    cached_matrix_bytes = matrix_count * coefficients**2 * DOUBLE_BYTES
    packed_translation_bytes = translation_bank_bytes(order, depth)
    schedule_bytes = (
        _m2l_schedule_bytes(nodes, depth, coefficients, thread_count)
        if cached_matrix_bytes > M2L_SCHEDULE_MATRIX_LIMIT_BYTES
        else 0
    )
    # CUDA stores compact active-row descriptors, source and matrix indices,
    # and one explicit level per node. Endpoint levels are validated while the
    # device packing is built and need not be uploaded separately.
    cuda_interaction_index_bytes = (
        active_m2l_rows * CUDA_M2L_ACTIVE_ROW_BYTES
        + (2 * m2l_interactions + len(nodes)) * INT_BYTES
    )
    # The canonical host plan retains explicit source and target levels plus
    # the legacy same-level array. It also owns target schedules grouped by
    # level and a level value for every compact node ID.
    interaction_index_bytes = (
        (
            len(nodes) + 1
            + 5 * m2l_interactions
            + 2 * (depth + 1)
            + (depth + 2)
            + 2 * len(nodes)
        )
        * INT_BYTES
    )
    level_scaling_bytes = 2 * (depth + 1) * coefficients * DOUBLE_BYTES
    host_static = {
        "P2P position scratch": p2p_scratch_bytes,
        "packed P2M rows": 3 * coefficients * particle_count * DOUBLE_BYTES,
        "shared M2M/L2L maps": shared_translation_bytes,
        "level-scaled M2M/L2L banks": packed_translation_bytes,
        "cached M2L matrices": cached_matrix_bytes + level_scaling_bytes,
        "M2L interaction indices": interaction_index_bytes + schedule_bytes,
        "L2P rows": 4 * coefficients * particle_count * DOUBLE_BYTES,
        "multipole and local state": (
            2 * len(nodes) * coefficients * DOUBLE_BYTES
        ),
        "moments, results, near fields, and identities": (
            particle_count * (VEC3_BYTES + 32 + VEC3_BYTES + INT_BYTES)
        ),
    }

    coefficient_buffer_bytes = 2 * len(nodes) * coefficients * DOUBLE_BYTES
    cuda_partial = {
        "P2P tensors": p2p_static_bytes,
        "P2P dynamic buffers": (
            2 * particle_count * VEC3_BYTES + particle_count * INT_BYTES
        ),
        "M2L class matrices and indices": (
            cached_matrix_bytes
            + cuda_interaction_index_bytes
            + level_scaling_bytes
        ),
        "M2L multipole and local buffers": coefficient_buffer_bytes,
    }

    cuda_full = {
        "P2P tensors": p2p_static_bytes,
        "shared M2L matrices": cached_matrix_bytes,
        "M2L interaction metadata": cuda_interaction_index_bytes,
        "M2L level scalings": level_scaling_bytes,
        "shared M2M/L2L matrices": (
            shared_translation_bytes
        ),
        "other static operator entries": (
            2 * particle_entries * particle_count * STATIC_ENTRY_BYTES
        ),
        "M2M/L2L interaction metadata": (
            (occupied_source_nodes + occupied_target_nodes) * 4 * INT_BYTES
        ),
        "geometry indices": 2 * particle_count * INT_BYTES,
        "moments, fields, and identities": (
            2 * particle_count * VEC3_BYTES
            + 3 * particle_count * VEC3_BYTES
            + particle_count * INT_BYTES
        ),
        "multipole and local buffers": coefficient_buffer_bytes,
    }

    return StorageEstimate(
        particles=particle_count,
        order=order,
        depth=depth,
        coefficients=coefficients,
        nodes=len(nodes),
        occupied_source_nodes=occupied_source_nodes,
        occupied_target_nodes=occupied_target_nodes,
        transfer_classes=len(groups),
        m2l_interactions=m2l_interactions,
        p2p_pairs=p2p_pairs,
        host_static=host_static,
        cuda_partial=cuda_partial,
        cuda_full=cuda_full,
    )
