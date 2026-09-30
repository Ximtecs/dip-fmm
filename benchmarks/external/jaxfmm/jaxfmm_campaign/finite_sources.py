# SPDX-License-Identifier: Apache-2.0
"""Uniformly magnetised bodies as face charges (NumPy only).

A body mesh is the union of touching bodies: the cubes of a lattice or the
Kuhn tetrahedra of a conforming mesh. With a constant magnetisation ``M_b``
inside each body the volume charge vanishes and the field is carried by the
faces, ``sigma_f = (M_left - M_right) . n_f`` (``M_right = 0`` on the outer
boundary). Each unique face is one constant-charge triangle: that is what
jaxFMM's element path integrates, and its triangle count is jaxFMM's true
source count, reported next to the body count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import geometry


@dataclass(frozen=True)
class BodyMesh:
    kind: str                      # lattice | kuhn_mesh
    grid: int
    spacing: float
    nodes: np.ndarray              # (n_nodes, 3)
    centres: np.ndarray            # (n_bodies, 3) body centres = point targets
    volumes: np.ndarray            # (n_bodies,)
    triangles: np.ndarray          # (n_tri, 3) node ids of the unique faces
    left: np.ndarray               # (n_tri,) body owning the face
    right: np.ndarray              # (n_tri,) neighbouring body or -1
    normal: np.ndarray             # (n_tri, 3) unit normal pointing away from left
    area: np.ndarray               # (n_tri,)
    counts: dict[str, int]

    def face_charges(self, magnetisation: np.ndarray) -> np.ndarray:
        faces = {"left": self.left, "right": self.right, "normal": self.normal}
        return geometry.body_face_charges(faces, magnetisation)


def build_body_mesh(kind: str, grid: int, spacing: float = 1.0) -> BodyMesh:
    if kind == "lattice":
        mesh = geometry.cube_lattice_mesh(grid, spacing)
        body_triangles = geometry.cube_body_triangles(mesh["corner_ids"])
        centres = mesh["centres"]
        volumes = np.full(len(centres), spacing**3)
        counts = {"nx": grid, "ny": grid, "nz": grid, "cells": grid**3, "bodies": int(len(centres)),
                  "vertices": int(len(mesh["nodes"])), "tetrahedra": 0}
    elif kind == "kuhn_mesh":
        mesh = geometry.kuhn_mesh(grid, spacing)
        body_triangles = geometry.tetra_body_triangles(mesh["connectivity"])
        centres = mesh["centroids"]
        volumes = np.full(len(centres), spacing**3 / 6.0)
        counts = {"nx": grid, "ny": grid, "nz": grid, "cells": grid**3, "bodies": int(len(centres)),
                  "vertices": int(len(mesh["nodes"])), "tetrahedra": int(len(centres)), "faces": mesh["n_faces"]}
    else:
        raise ValueError(f"unknown body mesh kind: {kind}")
    faces = geometry.face_triangles(mesh["nodes"], body_triangles, centres)
    counts.update({"unique_faces": int(len(faces["triangles"])) if kind == "kuhn_mesh" else int(len(faces["triangles"]) // 2),
                   "triangles": int(len(faces["triangles"])), "shared_triangles": faces["shared_count"],
                   "sources_jaxfmm": int(len(faces["triangles"])), "sources_dipfmm": int(len(centres)),
                   "targets": int(len(centres))})
    return BodyMesh(kind=kind, grid=grid, spacing=spacing, nodes=mesh["nodes"], centres=centres, volumes=volumes,
                    triangles=faces["triangles"], left=faces["left"], right=faces["right"],
                    normal=faces["normal"], area=faces["area"], counts=counts)


def body_dataset(kind: str, grid: int, spacing: float = 1.0, state: str = "random",
                 seed: int = geometry.DEFAULT_SEED) -> geometry.Dataset:
    """The dip-fmm dataset of the same bodies (moments are ``V * M`` per body)."""
    spec = {"kind": kind, "grid": grid, "spacing": spacing, "seed": seed, "state": state,
            "body": "prism" if kind == "lattice" else "tetra", "body_fill": 1.0}
    return geometry.build_dataset(spec)


def magnetisation_and_moments(mesh: BodyMesh, state: str, seed: int = geometry.DEFAULT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Per-body unit magnetisation and the dip-fmm moments ``V * M``."""
    M = geometry.magnetisation(mesh.centres, state, seed)
    return M, mesh.volumes[:, None] * M


def describe(mesh: BodyMesh) -> dict[str, Any]:
    return {"kind": mesh.kind, "grid": mesh.grid, "spacing": mesh.spacing, **mesh.counts}
