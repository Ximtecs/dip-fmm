# SPDX-License-Identifier: Apache-2.0
"""Deterministic point-lattice geometry and source states.

These functions reproduce, bit for bit, the conventions of the FMM3D
comparison campaign (``Article1/scripts/datasets.py``): a centred cubic
lattice with unit spacing, random unit dipole moments drawn from
``default_rng(seed + 1)``, coincident sources and targets, and the binary
``A1DS0002`` container whose SHA256 identifies a dataset. Re-implementing
them here keeps the jaxFMM campaign self-contained inside the repository
while letting it prove, by hash, that it evaluates the same data as the
FMM3D rows.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
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
    "moment_units": "dimensionless unit vectors",
    "coordinate_scale": "lattice spacing 1.0",
    "self_interaction": "excluded by explicit index identity, never by coordinate equality",
    "primary_output": "H",
}


def regular_lattice(grid: int, spacing: float = 1.0) -> np.ndarray:
    """Centred lattice of ``grid**3`` points in C order (x slowest)."""
    axis = (np.arange(grid, dtype=np.float64) - 0.5 * (grid - 1)) * spacing
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


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


@dataclass(frozen=True)
class LatticeDataset:
    """One point-lattice problem: coincident sources and targets."""

    dataset_id: str
    grid: int
    spacing: float
    seed: int
    state: str
    positions: np.ndarray
    moments: np.ndarray
    identity_map: np.ndarray
    sha256: str

    @property
    def source_count(self) -> int:
        return int(len(self.positions))

    @property
    def target_count(self) -> int:
        return int(len(self.positions))

    def spec(self) -> dict[str, Any]:
        return {
            "kind": "lattice",
            "grid": self.grid,
            "spacing": self.spacing,
            "seed": self.seed,
            "state": self.state,
        }


def container_bytes(positions: np.ndarray, moments: np.ndarray) -> bytes:
    """Serialise a coincident point dataset in the ``A1DS0002`` layout."""
    positions = np.ascontiguousarray(positions, dtype="<f8")
    moments = np.ascontiguousarray(moments, dtype="<f8")
    identity = np.arange(len(positions), dtype="<i4")
    flags = 1  # identity map present, no finite-body blocks
    parts = [
        DATASET_MAGIC,
        struct.pack("<QQQ", len(positions), len(positions), flags),
        positions.tobytes(order="C"),
        moments.tobytes(order="C"),
        positions.tobytes(order="C"),
        identity.tobytes(order="C"),
    ]
    return b"".join(parts)


def lattice_dataset(grid: int, *, spacing: float = 1.0, seed: int = DEFAULT_SEED,
                    state: str = "random") -> LatticeDataset:
    """Build the lattice dataset ``lattice_<grid>`` and hash its container."""
    positions = regular_lattice(grid, spacing)
    moments = magnetisation(positions, state, seed)
    digest = hashlib.sha256(container_bytes(positions, moments)).hexdigest()
    suffix = "" if state == "random" else f"_{state}"
    return LatticeDataset(
        dataset_id=f"lattice_{grid}{suffix}",
        grid=grid,
        spacing=spacing,
        seed=seed,
        state=state,
        positions=positions,
        moments=moments,
        identity_map=np.arange(len(positions), dtype=np.int32),
        sha256=digest,
    )
