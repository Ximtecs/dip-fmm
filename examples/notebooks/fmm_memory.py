"""Geometry-derived storage estimates for the uniform FMM notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

import cdfmm


STATIC_ENTRY_BYTES = 16
P2P_BLOCK_BYTES = 56
VEC3_BYTES = 24
DOUBLE_BYTES = 8
INT_BYTES = 4


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
) -> StorageEstimate:
    """Estimate current host and CUDA storage without constructing an FMM plan.

    The estimate follows the containers allocated by the present implementation.
    It intentionally excludes allocator metadata, CUDA context storage, pinned
    staging buffers, and Python object overhead.
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
    for target in nodes:
        if target.level == 0 or target.target_count == 0:
            continue
        for source_index in target.list2:
            source = nodes[source_index]
            if source.source_count == 0:
                continue
            key = (
                target.ix - source.ix,
                target.iy - source.iy,
                target.iz - source.iz,
            )
            groups[key] = groups.get(key, 0) + 1

    p2p_pairs = 0
    for target in nodes:
        if not target.is_leaf or target.target_count == 0:
            continue
        p2p_pairs += target.target_count * sum(
            nodes[source_index].source_count for source_index in target.list1
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
    cached_matrix_bytes = len(groups) * coefficients**2 * DOUBLE_BYTES
    # The canonical target-row plan stores one row offset per tree node;
    # source, matrix-class, and level indices for every interaction; and two
    # node-index bounds for every level. The latter let portable M2L visit only
    # the targets owned by the requested level.
    cuda_interaction_index_bytes = (
        len(nodes) + 1 + 3 * m2l_interactions
    ) * INT_BYTES
    interaction_index_bytes = (
        cuda_interaction_index_bytes + 2 * (depth + 1) * INT_BYTES
    )
    level_scaling_bytes = 2 * (depth + 1) * coefficients * DOUBLE_BYTES
    host_static = {
        "P2P tensors": p2p_static_bytes,
        "P2M maps": particle_entries * particle_count * STATIC_ENTRY_BYTES,
        "shared M2M/L2L maps": (
            2 * depth * 8 * translation_entries * STATIC_ENTRY_BYTES
        ),
        "cached M2L matrices": cached_matrix_bytes + level_scaling_bytes,
        "M2L interaction indices": interaction_index_bytes,
        "L2P rows": 4 * coefficients * particle_count * DOUBLE_BYTES,
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
            2 * depth * 8 * translation_entries * STATIC_ENTRY_BYTES
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
