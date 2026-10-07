"""Octree-refined cuboid meshes of a Voronoi grain structure, for FMM validation.

Mimics the MagTense multigrain setup: random seeds in a cube, an unweighted Voronoi
tessellation, and a uniform base grid whose tiles are split (up to ``levels`` times) wherever
any part of the tile lies within half the band width of a grain-boundary plane or the tile
straddles two grains. Self-contained (numpy only) so that dip-fmm tests need no MagTense.

Distances to the grain boundary are the exact bisector-plane distances of the unweighted
Voronoi tessellation, taken over all other seeds (the minimum is attained at a Voronoi
neighbour, so this equals the facet-plane distance of the own cell).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CORNERS = np.array([[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)])
CHILDREN = np.array([[sx, sy, sz] for sz in (-0.25, 0.25) for sy in (-0.25, 0.25) for sx in (-0.25, 0.25)])


@dataclass
class OctreeMesh:
    centres: np.ndarray     # (N, 3)
    sizes: np.ndarray       # (N, 3) full side lengths
    level: np.ndarray       # (N,)
    grain: np.ndarray       # (N,) 0-based grain of the centre
    in_band: np.ndarray     # (N,) centre within half the band width of a boundary
    seeds: np.ndarray       # (n_grains, 3)
    box: float

    @property
    def volumes(self) -> np.ndarray:
        return np.prod(self.sizes, axis=1)

    def summary(self) -> dict:
        levels, counts = np.unique(self.level, return_counts=True)
        return {"n_bodies": int(len(self.centres)), "tiles_per_level": {int(l): int(c) for l, c in zip(levels, counts)},
                "band_fraction": float(self.volumes[self.in_band].sum() / self.box**3),
                "size_min": float(self.sizes.min()), "size_max": float(self.sizes.max())}


def boundary_distance(points: np.ndarray, seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest seed and the distance to the nearest Voronoi face (bisector-plane form)."""
    d2 = ((points[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)       # (P, S)
    own = np.argmin(d2, axis=1)
    d_own = d2[np.arange(len(points)), own]
    seed_dist = np.linalg.norm(seeds[:, None, :] - seeds[None, :, :], axis=2)   # (S, S)
    plane = (d2 - d_own[:, None]) / (2.0 * np.where(seed_dist[own] > 0, seed_dist[own], np.inf))
    plane[np.arange(len(points)), own] = np.inf
    return own, plane.min(axis=1)


def voronoi_octree_mesh(n_grains: int = 20, base: int = 8, levels: int = 2, band: float = 0.03,
                        box: float = 1.0, seed: int = 0, min_seed_spacing: float | None = None) -> OctreeMesh:
    """Build the mesh in a cube of side ``box`` centred on the origin.

    Seeds are drawn uniformly with a minimum spacing (default half the mean grain size) so
    that no grain is degenerate; ``band`` is the full width of the intergrain band.
    """
    rng = np.random.default_rng(seed)
    spacing = min_seed_spacing if min_seed_spacing is not None else 0.5 * box / n_grains ** (1 / 3)
    seeds = []
    for _ in range(100000):
        p = rng.uniform(-0.5 * box, 0.5 * box, 3)
        if all(np.linalg.norm(p - q) >= spacing for q in seeds):
            seeds.append(p)
        if len(seeds) == n_grains:
            break
    seeds = np.array(seeds)
    if len(seeds) < n_grains:
        raise RuntimeError("could not place the seeds; lower min_seed_spacing")

    h0 = box / base
    g = -0.5 * box + (np.arange(base) + 0.5) * h0
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    centres = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    sizes = np.full(len(centres), h0)
    level = np.zeros(len(centres), dtype=int)
    out_c, out_s, out_l = [], [], []
    for lev in range(levels + 1):
        if lev == levels or len(centres) == 0:
            out_c.append(centres); out_s.append(sizes); out_l.append(level)
            break
        pts = np.concatenate([centres[:, None, :] + CORNERS[None] * sizes[:, None, None], centres[:, None, :]], axis=1)
        own, dist = boundary_distance(pts.reshape(-1, 3), seeds)
        own = own.reshape(len(centres), 9); dist = dist.reshape(len(centres), 9)
        refine = np.any(own != own[:, :1], axis=1) | (dist.min(axis=1) <= 0.5 * band)
        keep = ~refine
        out_c.append(centres[keep]); out_s.append(sizes[keep]); out_l.append(level[keep])
        r = np.flatnonzero(refine)
        centres = (centres[r][:, None, :] + CHILDREN[None] * sizes[r][:, None, None]).reshape(-1, 3)
        sizes = np.repeat(sizes[r] / 2, 8)
        level = np.full(len(centres), lev + 1)
    centres = np.concatenate(out_c); sizes = np.concatenate(out_s); level = np.concatenate(out_l)
    own, dist = boundary_distance(centres, seeds)
    return OctreeMesh(centres=centres, sizes=np.repeat(sizes[:, None], 3, axis=1), level=level, grain=own,
                      in_band=dist <= 0.5 * band, seeds=seeds, box=box)


def random_moments(mesh: OctreeMesh, seed: int = 1, ms_grain: float = 1.0, ms_band: float = 0.3) -> np.ndarray:
    """Total moments m = V Ms mhat with one random unit direction per grain and a softer band."""
    rng = np.random.default_rng(seed)
    axes = rng.normal(size=(len(mesh.seeds), 3))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    mhat = axes[mesh.grain]
    ms = np.where(mesh.in_band, ms_band, ms_grain)
    return mhat * (mesh.volumes * ms)[:, None]
