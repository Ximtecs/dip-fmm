# SPDX-License-Identifier: Apache-2.0
"""Deterministic benchmark geometries and source states.

Two geometry families, both parametrised by an edge count ``grid`` and a
``spacing`` and generated from a spec dictionary rather than a fixed list:

* ``lattice``: a centred cubic point lattice of ``grid**3`` bodies. The bodies
  are point dipoles, or touching cubes of side ``body_fill * spacing`` when
  the prism body is requested.
* ``kuhn_mesh``: a conforming, face-touching tetrahedral mesh of the cubic
  domain, each of the ``grid**3`` cells split into the six Kuhn (Freudenthal)
  tetrahedra sharing its main diagonal, ``6 * grid**3`` bodies in all.

These reproduce, bit for bit, the conventions of the FMM3D comparison
campaign (``Article1/scripts/datasets.py``): unit spacing, random unit
dipole moments from ``default_rng(seed + 1)``, coincident sources and
targets, tetrahedron vertices stored as offsets from the centroid, and the
binary ``A1DS0002`` container whose SHA256 identifies a dataset. Every
dataset reports its counts explicitly (cells, vertices, tetrahedra, faces,
sources, targets) so a scaling axis is never ambiguous.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Lattice edge counts of the FMM3D external campaign (N = grid**3).
FMM3D_CAMPAIGN_GRIDS = (10, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128)

#: The dataset seed shared with the FMM3D campaign.
DEFAULT_SEED = 314159

DATASET_MAGIC = b"A1DS0002"

FIELD_CONVENTION = {
    "greens_function": "G(r) = 1/(4*pi*|r|)",
    "potential": "phi(x_i) = sum_j m_j . r_ij / (4*pi*|r_ij|^3), r_ij = x_i - x_j",
    "field": "H = -grad(phi)",
    "field_closed_form": "H_i = (1/(4*pi)) * sum_j [ 3 r_ij (m_j.r_ij)/|r_ij|^5 - m_j/|r_ij|^3 ]",
    "moment_units": "dimensionless unit vectors (finite bodies: moment per body, not per volume)",
    "coordinate_scale": "lattice spacing 1.0",
    "self_interaction": "excluded by explicit index identity, never by coordinate equality",
    "primary_output": "H",
}


# --------------------------------------------------------------------------
# point lattice
# --------------------------------------------------------------------------

def regular_lattice(grid: int, spacing: float = 1.0) -> np.ndarray:
    """Centred lattice of ``grid**3`` points in C order (x slowest)."""
    axis = (np.arange(grid, dtype=np.float64) - 0.5 * (grid - 1)) * spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


# --------------------------------------------------------------------------
# Kuhn tetrahedral mesh
# --------------------------------------------------------------------------

def _kuhn_corner_indices() -> list[tuple[int, int, int, int]]:
    """Six congruent tetrahedra of the unit cube sharing the (0,0,0)-(1,1,1) diagonal."""
    def bit(corner: tuple[int, int, int]) -> int:
        return corner[0] * 4 + corner[1] * 2 + corner[2]

    cells: list[tuple[int, int, int, int]] = []
    for permutation in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        first = [0, 0, 0]
        first[permutation[0]] = 1
        second = list(first)
        second[permutation[1]] = 1
        cells.append((bit((0, 0, 0)), bit(tuple(first)), bit(tuple(second)), bit((1, 1, 1))))
    return cells


KUHN_TETRAHEDRA = _kuhn_corner_indices()
_CORNER_OFFSETS = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=np.int64)


def kuhn_mesh(grid: int, spacing: float = 1.0) -> dict[str, np.ndarray]:
    """Face-touching Kuhn mesh of a ``grid**3``-cell cube, centred at the origin.

    Returns the node coordinates ``(n_nodes, 3)``, the connectivity
    ``(6 * grid**3, 4)`` into the nodes, the per-tetrahedron centroids and the
    vertex offsets from the centroid ``(n_tets, 4, 3)`` (the ``Tetrahedron``
    record convention), plus the unique face count.
    """
    axis = (np.arange(grid + 1, dtype=np.float64) - 0.5 * grid) * spacing
    nodes = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    cells = np.indices((grid, grid, grid)).reshape(3, -1).T
    node_index = np.arange((grid + 1) ** 3).reshape(grid + 1, grid + 1, grid + 1)
    corner_ids = np.empty((len(cells), 8), dtype=np.int64)
    corners = np.empty((len(cells), 8, 3), dtype=np.float64)
    for index, offset in enumerate(_CORNER_OFFSETS):
        pick = cells + offset
        corner_ids[:, index] = node_index[pick[:, 0], pick[:, 1], pick[:, 2]]
        corners[:, index, :] = nodes[pick[:, 0], pick[:, 1], pick[:, 2]]
    connectivity = np.empty((len(cells) * 6, 4), dtype=np.int64)
    vertices = np.empty((len(cells) * 6, 4, 3), dtype=np.float64)
    for slot, quad in enumerate(KUHN_TETRAHEDRA):
        connectivity[slot::6] = corner_ids[:, list(quad)]
        vertices[slot::6] = corners[:, list(quad), :]
    centroids = vertices.mean(axis=1)
    faces = np.concatenate([
        np.sort(connectivity[:, [0, 1, 2]], axis=1),
        np.sort(connectivity[:, [0, 1, 3]], axis=1),
        np.sort(connectivity[:, [0, 2, 3]], axis=1),
        np.sort(connectivity[:, [1, 2, 3]], axis=1),
    ])
    n_faces = int(len(np.unique(faces, axis=0)))
    return {
        "nodes": nodes.reshape(-1, 3),
        "connectivity": connectivity,
        "centroids": centroids,
        "vertex_offsets": vertices - centroids[:, None, :],
        "n_faces": n_faces,
    }


# --------------------------------------------------------------------------
# face charges of uniformly magnetised bodies (the jaxFMM source representation)
# --------------------------------------------------------------------------

#: The six quad faces of a cube as corner-bit indices (same order as _CORNER_OFFSETS),
#: each split into two triangles.
_CUBE_FACE_TRIANGLES = (
    (0, 1, 3), (0, 3, 2),   # x = 0 face (bits a=0)
    (4, 6, 7), (4, 7, 5),   # x = 1
    (0, 4, 5), (0, 5, 1),   # y = 0
    (2, 3, 7), (2, 7, 6),   # y = 1
    (0, 2, 6), (0, 6, 4),   # z = 0
    (1, 5, 7), (1, 7, 3),   # z = 1
)


def cube_lattice_mesh(grid: int, spacing: float = 1.0) -> dict[str, np.ndarray]:
    """Nodes and per-cube corner ids of a ``grid**3`` lattice of touching unit cells."""
    axis = (np.arange(grid + 1, dtype=np.float64) - 0.5 * grid) * spacing
    nodes = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    node_index = np.arange((grid + 1) ** 3).reshape(grid + 1, grid + 1, grid + 1)
    cells = np.indices((grid, grid, grid)).reshape(3, -1).T
    corner_ids = np.empty((len(cells), 8), dtype=np.int64)
    for index, offset in enumerate(_CORNER_OFFSETS):
        pick = cells + offset
        corner_ids[:, index] = node_index[pick[:, 0], pick[:, 1], pick[:, 2]]
    centres = nodes[corner_ids].mean(axis=1)
    return {"nodes": nodes, "corner_ids": corner_ids, "centres": centres}


def face_triangles(nodes: np.ndarray, body_triangles: np.ndarray, body_centres: np.ndarray) -> dict[str, np.ndarray]:
    """Unique boundary/interface triangles of a body mesh with left/right adjacency.

    ``body_triangles`` is ``(n_bodies, k, 3)``: the ``k`` triangles bounding each
    body as node ids. Triangles shared by two bodies are kept once. The returned
    normal of every triangle points away from its ``left`` body, so the charge of
    a uniformly magnetised mesh is ``sigma = (M_left - M_right) . n`` with
    ``M_right = 0`` on the outer boundary (``right == -1``).
    """
    n_bodies, k, _ = body_triangles.shape
    flat = body_triangles.reshape(-1, 3)
    owner = np.repeat(np.arange(n_bodies), k)
    key = np.sort(flat, axis=1)
    _, first, inverse, counts = np.unique(key, axis=0, return_index=True, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    n_faces = len(first)
    left = owner[first]
    right = np.full(n_faces, -1, dtype=np.int64)
    order = np.argsort(inverse, kind="stable")
    sorted_faces = inverse[order]
    sorted_owner = owner[order]
    second = np.ones(n_faces, dtype=bool)
    # the second occurrence of each shared face names the right body
    seen = np.zeros(n_faces, dtype=bool)
    for face, body in zip(sorted_faces, sorted_owner):
        if seen[face]:
            right[face] = body
        seen[face] = True
    del second
    triangles = flat[first]
    a, b, c = nodes[triangles[:, 0]], nodes[triangles[:, 1]], nodes[triangles[:, 2]]
    normal = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(normal, axis=1)
    normal /= (2.0 * area)[:, None]
    outward = np.einsum("ij,ij->i", normal, (a + b + c) / 3.0 - body_centres[left]) > 0.0
    normal[~outward] *= -1.0
    return {"triangles": triangles, "left": left, "right": right, "normal": normal, "area": area,
            "shared_count": int(np.count_nonzero(counts == 2))}


def body_face_charges(faces: dict[str, np.ndarray], magnetisation: np.ndarray) -> np.ndarray:
    """Constant face charge per triangle, ``(M_left - M_right) . n``."""
    right_m = np.where(faces["right"][:, None] >= 0, magnetisation[np.maximum(faces["right"], 0)], 0.0)
    return np.einsum("ij,ij->i", magnetisation[faces["left"]] - right_m, faces["normal"])


def cube_body_triangles(corner_ids: np.ndarray) -> np.ndarray:
    """``(n_cubes, 12, 3)`` node ids of the twelve boundary triangles of each cube."""
    return corner_ids[:, np.array(_CUBE_FACE_TRIANGLES)]


def tetra_body_triangles(connectivity: np.ndarray) -> np.ndarray:
    """``(n_tets, 4, 3)`` node ids of the four faces of each tetrahedron."""
    return connectivity[:, np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])]


# --------------------------------------------------------------------------
# source states
# --------------------------------------------------------------------------

def unit_dipole_moments(count: int, seed: int) -> np.ndarray:
    """Deterministic random unit-vector dipole moments (seed + 1 generator)."""
    generator = np.random.default_rng(seed + 1)
    directions = generator.normal(size=(count, 3))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return directions / norms


def magnetisation(positions: np.ndarray, state: str, seed: int) -> np.ndarray:
    """Unit moments for one of the three source states of the FMM3D campaign."""
    if state == "random":
        return unit_dipole_moments(len(positions), seed)
    if state == "uniform_z":
        moments = np.zeros_like(positions)
        moments[:, 2] = 1.0
        return moments
    if state == "vortex":
        centred = positions - positions.mean(axis=0)
        radius = np.hypot(centred[:, 0], centred[:, 1])
        extent = max(float(np.abs(centred[:, :2]).max()), 1.0)
        core = np.exp(-((radius / (0.2 * extent)) ** 2))
        in_plane = np.sqrt(np.clip(1.0 - core**2, 0.0, 1.0))
        safe = np.where(radius > 0.0, radius, 1.0)
        moments = np.column_stack((
            -centred[:, 1] / safe * in_plane,
            centred[:, 0] / safe * in_plane,
            core,
        ))
        return moments / np.linalg.norm(moments, axis=1, keepdims=True)
    raise ValueError(f"unknown magnetisation state: {state}")


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Dataset:
    """One problem: coincident sources and targets with an explicit body kind."""

    dataset_id: str
    spec: dict[str, Any]
    positions: np.ndarray
    moments: np.ndarray
    identity_map: np.ndarray
    body: str                      # point | prism | tetra
    body_side: float | None        # prism full side length
    tetrahedra: np.ndarray | None  # (N, 4, 3) vertex offsets from the centroid
    counts: dict[str, int] = field(default_factory=dict)
    sha256: str = ""

    @property
    def source_count(self) -> int:
        return int(len(self.positions))

    @property
    def target_count(self) -> int:
        return int(len(self.positions))

    @property
    def spacing(self) -> float:
        return float(self.spec["spacing"])


def container_bytes(positions: np.ndarray, moments: np.ndarray, prisms: np.ndarray | None,
                    tetrahedra: np.ndarray | None) -> bytes:
    """Serialise a coincident dataset in the ``A1DS0002`` layout (flags: identity,
    source prisms, target prisms, source tetrahedra, target tetrahedra)."""
    positions = np.ascontiguousarray(positions, dtype="<f8")
    moments = np.ascontiguousarray(moments, dtype="<f8")
    identity = np.arange(len(positions), dtype="<i4")
    flags = 1
    blocks = []
    if prisms is not None:
        flags |= 2 | 4
        block = np.ascontiguousarray(prisms, dtype="<f8").reshape(-1, 3)
        blocks += [block, block]
    if tetrahedra is not None:
        flags |= 8 | 16
        block = np.ascontiguousarray(tetrahedra, dtype="<f8").reshape(-1, 12)
        blocks += [block, block]
    parts = [
        DATASET_MAGIC,
        struct.pack("<QQQ", len(positions), len(positions), flags),
        positions.tobytes(order="C"),
        moments.tobytes(order="C"),
        positions.tobytes(order="C"),
        identity.tobytes(order="C"),
    ] + [block.tobytes(order="C") for block in blocks]
    return b"".join(parts)


def build_dataset(spec: dict[str, Any]) -> Dataset:
    """Build a dataset from its spec.

    Spec keys: ``kind`` (``lattice`` or ``kuhn_mesh``), ``grid``, ``spacing``
    (default 1.0), ``seed`` (default 314159), ``state`` (default ``random``),
    ``body`` (``point``/``prism`` for a lattice, ``tetra`` for a mesh) and
    ``body_fill`` (prism side as a fraction of the spacing, default 1.0).
    """
    kind = spec.get("kind", "lattice")
    grid = int(spec["grid"])
    spacing = float(spec.get("spacing", 1.0))
    seed = int(spec.get("seed", DEFAULT_SEED))
    state = spec.get("state", "random")
    full = {"kind": kind, "grid": grid, "spacing": spacing, "seed": seed, "state": state}
    if kind == "lattice":
        body = spec.get("body", "point")
        positions = regular_lattice(grid, spacing)
        counts = {"nx": grid, "ny": grid, "nz": grid, "cells": grid**3, "vertices": (grid + 1) ** 3,
                  "tetrahedra": 0, "faces": 0, "sources": grid**3, "targets": grid**3}
        prisms = None
        tetrahedra = None
        body_side = None
        if body == "prism":
            body_fill = float(spec.get("body_fill", 1.0))
            body_side = body_fill * spacing
            prisms = np.full((len(positions), 3), body_side)
            full["body_fill"] = body_fill
            dataset_id = f"prism_cells_{grid}" if body_fill == 1.0 else f"prism_fill{body_fill}_{grid}"
        elif body == "point":
            dataset_id = f"lattice_{grid}"
        else:
            raise ValueError(f"lattice body must be point or prism, not {body}")
    elif kind == "kuhn_mesh":
        body = "tetra"
        mesh = kuhn_mesh(grid, spacing)
        positions = mesh["centroids"]
        tetrahedra = mesh["vertex_offsets"]
        prisms = None
        body_side = None
        counts = {"nx": grid, "ny": grid, "nz": grid, "cells": grid**3, "vertices": int(len(mesh["nodes"])),
                  "tetrahedra": int(len(tetrahedra)), "faces": mesh["n_faces"],
                  "sources": int(len(positions)), "targets": int(len(positions))}
        dataset_id = f"tetra_mesh_{grid}"
    else:
        raise ValueError(f"unknown geometry kind: {kind}")
    full["body"] = body
    if state != "random":
        dataset_id += f"_{state}"
    moments = magnetisation(positions, state, seed)
    digest = hashlib.sha256(container_bytes(positions, moments, prisms, tetrahedra)).hexdigest()
    return Dataset(dataset_id=dataset_id, spec=full, positions=positions, moments=moments,
                   identity_map=np.arange(len(positions), dtype=np.int32), body=body,
                   body_side=body_side, tetrahedra=tetrahedra, counts=counts, sha256=digest)


def lattice_dataset(grid: int, *, spacing: float = 1.0, seed: int = DEFAULT_SEED,
                    state: str = "random") -> Dataset:
    """The point lattice ``lattice_<grid>`` (convenience wrapper)."""
    return build_dataset({"kind": "lattice", "grid": grid, "spacing": spacing, "seed": seed, "state": state})
