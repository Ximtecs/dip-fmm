# SPDX-License-Identifier: Apache-2.0
"""FP64 direct reference, error metrics and the sampled-target selection.

The formulas mirror ``Article1/scripts/reference.py`` exactly, so a relative
L2 error reported here is comparable with the FMM3D campaign's rows: the
reference is an independent NumPy FP64 direct dipole sum over every source,
evaluated at a Morton-stride subset of the targets, and the headline metric
is the Frobenius-norm ratio of the error over the reference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

FOUR_PI = 4.0 * np.pi

REFERENCE_SCHEMA_VERSION = 2
TARGET_SELECTION_VERSION = "morton_stride_v1"


def direct_dipole_field(
    source_positions: np.ndarray,
    moments: np.ndarray,
    target_positions: np.ndarray,
    target_source_indices: np.ndarray | None = None,
    block: int | None = None,
) -> np.ndarray:
    """Exact O(N*K) direct evaluation of ``H = -grad(phi)`` in FP64.

    ``target_source_indices[i]`` names the source that target ``i`` *is*, and
    that pair is skipped by index identity. Targets that correspond to no
    source pass ``-1``.
    """
    sources = np.ascontiguousarray(source_positions, dtype=np.float64)
    dipoles = np.ascontiguousarray(moments, dtype=np.float64)
    targets = np.ascontiguousarray(target_positions, dtype=np.float64)
    identities = (
        None if target_source_indices is None
        else np.ascontiguousarray(target_source_indices, dtype=np.int64).ravel()
    )
    if block is None:
        block = max(1, min(512, int(2.0e7 // max(len(sources), 1))))
    field = np.zeros_like(targets)
    for start in range(0, len(targets), block):
        stop = min(start + block, len(targets))
        separation = targets[start:stop, None, :] - sources[None, :, :]
        distance_squared = np.einsum("ijk,ijk->ij", separation, separation)
        if identities is not None:
            rows = np.arange(start, stop)
            excluded = identities[rows]
            valid = excluded >= 0
            distance_squared[rows[valid] - start, excluded[valid]] = np.inf
        else:
            distance_squared[distance_squared == 0.0] = np.inf
        inverse_distance = 1.0 / np.sqrt(distance_squared)
        inverse_cubed = inverse_distance**3
        inverse_fifth = inverse_cubed * inverse_distance * inverse_distance
        moment_dot_separation = np.einsum("ijk,jk->ij", separation, dipoles)
        first = 3.0 * separation * (moment_dot_separation * inverse_fifth)[:, :, None]
        second = dipoles[None, :, :] * inverse_cubed[:, :, None]
        field[start:stop] = (first - second).sum(axis=1) / FOUR_PI
    return field


def direct_dipole_potential(
    source_positions: np.ndarray,
    moments: np.ndarray,
    target_positions: np.ndarray,
    target_source_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Exact FP64 scalar potential ``phi`` of the same dipole system."""
    sources = np.ascontiguousarray(source_positions, dtype=np.float64)
    dipoles = np.ascontiguousarray(moments, dtype=np.float64)
    targets = np.ascontiguousarray(target_positions, dtype=np.float64)
    separation = targets[:, None, :] - sources[None, :, :]
    distance_squared = np.einsum("ijk,ijk->ij", separation, separation)
    if target_source_indices is not None:
        identities = np.asarray(target_source_indices, dtype=np.int64).ravel()
        rows = np.arange(len(targets))
        valid = identities >= 0
        distance_squared[rows[valid], identities[valid]] = np.inf
    else:
        distance_squared[distance_squared == 0.0] = np.inf
    inverse_cubed = distance_squared ** -1.5
    moment_dot_separation = np.einsum("ijk,jk->ij", separation, dipoles)
    return (moment_dot_separation * inverse_cubed).sum(axis=1) / FOUR_PI


def error_metrics(field: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Accuracy of ``field`` against ``reference``, both ``(K, 3)`` fields.

    ``relative_l2`` is the headline: the Frobenius norm of the difference over
    the Frobenius norm of the reference. The pointwise relative metrics use a
    floor so near-zero reference vectors cannot dominate them, and
    ``max_absolute_over_reference_rms`` is the useful maximum metric: the
    largest per-target error normalised by the RMS reference magnitude.
    """
    field = np.asarray(field, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if field.shape != reference.shape:
        raise ValueError(f"shape mismatch {field.shape} vs {reference.shape}")
    difference = field - reference
    per_target_absolute = np.linalg.norm(difference, axis=1)
    per_target_reference = np.linalg.norm(reference, axis=1)
    floor = np.finfo(np.float64).eps * max(1.0, float(per_target_reference.max(initial=0.0)))
    per_target_relative = per_target_absolute / np.maximum(per_target_reference, floor)
    reference_norm = np.linalg.norm(reference)
    reference_rms = reference_norm / np.sqrt(max(reference.shape[0], 1))
    return {
        "relative_l2": float(np.linalg.norm(difference) / max(reference_norm, floor)),
        "max_absolute_over_reference_rms": float(per_target_absolute.max(initial=0.0) / max(reference_rms, floor)),
        "max_pointwise_relative": float(per_target_relative.max(initial=0.0)),
        "mean_pointwise_relative": float(per_target_relative.mean()) if per_target_relative.size else 0.0,
        "median_pointwise_relative": float(np.median(per_target_relative)) if per_target_relative.size else 0.0,
        "p99_pointwise_relative": float(np.percentile(per_target_relative, 99)) if per_target_relative.size else 0.0,
        "rms_absolute": float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0,
        "max_absolute": float(per_target_absolute.max(initial=0.0)),
        "error_metric_definition": (
            "relative_l2 = ||H-Href||_F / ||Href||_F; "
            "max_absolute_over_reference_rms = max_i |H_i-Href_i| / rms_i |Href_i|"
        ),
    }


def _part1by2(values: np.ndarray, bits: int) -> np.ndarray:
    n = values.astype(np.uint64) & np.uint64((1 << bits) - 1)
    n = (n | (n << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    n = (n | (n << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    n = (n | (n << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    n = (n | (n << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    n = (n | (n << np.uint64(2))) & np.uint64(0x1249249249249249)
    return n


def morton_order(positions: np.ndarray, bits: int = 20) -> np.ndarray:
    minimum = positions.min(axis=0)
    span = np.maximum(positions.max(axis=0) - minimum, 1.0e-12)
    scale = (1 << bits) - 1
    integer = np.clip(np.round((positions - minimum) / span * scale), 0, scale).astype(np.uint64)
    codes = (
        _part1by2(integer[:, 0], bits)
        | (_part1by2(integer[:, 1], bits) << np.uint64(1))
        | (_part1by2(integer[:, 2], bits) << np.uint64(2))
    )
    return np.argsort(codes, kind="stable")


def select_sample_targets(positions: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Deterministic, spatially spread target subset (Morton stride)."""
    total = len(positions)
    if count >= total:
        return np.arange(total, dtype=np.int64)
    order = morton_order(positions)
    generator = np.random.default_rng(seed)
    stride = total / count
    offset = generator.uniform(0.0, stride)
    picks = np.clip(np.floor(offset + stride * np.arange(count)).astype(np.int64), 0, total - 1)
    chosen: list[int] = []
    seen: set[int] = set()
    for index in order[picks].tolist():
        if index not in seen:
            seen.add(index)
            chosen.append(index)
    for index in order.tolist():
        if len(chosen) >= count:
            break
        if index not in seen:
            seen.add(index)
            chosen.append(index)
    return np.array(sorted(chosen[:count]), dtype=np.int64)


def reference_cache_key(*, dataset_id: str, dataset_sha256: str, sample_targets: int,
                        sample_seed: int) -> str:
    """The FMM3D campaign's reference key, so its cached references are reusable."""
    payload = {
        "schema": REFERENCE_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "dataset_sha256": dataset_sha256,
        "sample_targets": sample_targets,
        "sample_seed": sample_seed,
        "target_selection_version": TARGET_SELECTION_VERSION,
        "source_geometry": "point",
        "target_geometry": "point",
        "source_target_relation": "identity",
        "precision": "float64",
        "algorithm": "numpy_direct_dipole",
        "field_convention": "H=-grad(phi), G=1/(4 pi |r|)",
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "reference_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def load_or_compute_reference(
    *,
    dataset_id: str,
    dataset_sha256: str,
    positions: np.ndarray,
    moments: np.ndarray,
    sample_targets: int,
    sample_seed: int,
    cache_roots: list[Path],
) -> dict[str, Any]:
    """One exact FP64 reference per (geometry, sampled target set).

    ``cache_roots`` are searched in order; the first is where a new reference
    is written. Passing the FMM3D campaign's cache first lets the jaxFMM rows
    score against byte-identical reference fields.
    """
    key = reference_cache_key(
        dataset_id=dataset_id, dataset_sha256=dataset_sha256,
        sample_targets=sample_targets, sample_seed=sample_seed,
    )
    for root in cache_roots:
        path = Path(root) / f"{key}.npz"
        if path.is_file():
            with np.load(path) as stored:
                return {
                    "field": stored["field"],
                    "sample_indices": stored["sample_indices"],
                    "source": str(path),
                }
    sample_indices = select_sample_targets(positions, sample_targets, sample_seed)
    field = direct_dipole_field(positions, moments, positions[sample_indices], sample_indices)
    root = Path(cache_roots[0])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{key}.npz"
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, field=field, sample_indices=sample_indices)
    temporary.replace(path)
    return {"field": field, "sample_indices": sample_indices, "source": str(path)}
