"""Plotting and diagnostic helpers shared by the cdfmm example notebooks."""

from collections.abc import Iterable
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

import cdfmm


def vector_norm(vectors):
    """Return Euclidean norms along the final array axis."""
    return np.linalg.norm(np.asarray(vectors), axis=-1)


def relative_error(approximate, reference, floor=1.0e-30):
    """Return pointwise vector relative errors with a safe denominator."""
    absolute_error = vector_norm(np.asarray(approximate) - np.asarray(reference))
    reference_norm = np.maximum(vector_norm(reference), floor)
    return absolute_error / reference_norm


def random_unit_vectors(rng, count):
    """Draw deterministic unit vectors when supplied a seeded generator."""
    directions = rng.normal(size=(count, 3))
    return directions / vector_norm(directions)[:, np.newaxis]


def direct_fields(target_positions, source_positions, dipole_moments):
    """Evaluate the C++ direct sum at each target in an array."""
    fields = []
    for target_position in np.asarray(target_positions):
        result = cdfmm.p2p_dipole_sum(
            target_position,
            source_positions,
            dipole_moments,
            output="field",
        )
        fields.append(result["H"])
    return np.asarray(fields)


def multipole_fields(
    target_positions,
    multipole_coefficients,
    source_centre,
    order,
):
    """Evaluate one C++ multipole expansion at several targets."""
    fields = []
    for target_position in np.asarray(target_positions):
        result = cdfmm.m2p(
            multipole_coefficients,
            source_centre,
            target_position,
            order=order,
            output="field",
        )
        fields.append(result["H"])
    return np.asarray(fields)


def local_fields(
    target_positions,
    local_coefficients,
    local_centre,
    order,
):
    """Evaluate one C++ local expansion at several targets."""
    fields = []
    for target_position in np.asarray(target_positions):
        result = cdfmm.l2p(
            local_coefficients,
            local_centre,
            target_position,
            order=order,
            output="field",
        )
        fields.append(result["H"])
    return np.asarray(fields)


def error_metrics(approximate, reference):
    """Return mean, RMS, and maximum pointwise relative field errors."""
    errors = relative_error(approximate, reference)
    return {
        "mean": float(np.mean(errors)),
        "rms": float(np.sqrt(np.mean(errors**2))),
        "maximum": float(np.max(errors)),
    }


def plot_coefficients_by_degree(
    axes,
    coefficients,
    order,
    title,
    colour="tab:blue",
):
    """Plot absolute Cartesian coefficient values grouped by total degree."""
    multi_indices = cdfmm.multi_indices(order)
    degrees = np.sum(multi_indices, axis=1)
    axes.scatter(degrees, np.abs(coefficients), color=colour, alpha=0.8)
    axes.set_yscale("symlog", linthresh=1.0e-16)
    axes.set_xlabel("Total degree")
    axes.set_ylabel("Coefficient magnitude")
    axes.set_title(title)
    axes.grid(alpha=0.25)


def cube_edges(centre, half_width):
    """Return the twelve line segments defining an axis-aligned cube."""
    centre = np.asarray(centre, dtype=float)
    offsets = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    vertices = centre + half_width * offsets

    edge_indices = []
    for first in range(8):
        for second in range(first + 1, 8):
            # Cube-edge vertices differ in exactly one sign coordinate.
            if np.count_nonzero(offsets[first] != offsets[second]) == 1:
                edge_indices.append((first, second))

    return [(vertices[first], vertices[second]) for first, second in edge_indices]


def draw_box_3d(
    axes,
    centre,
    half_width,
    colour="0.5",
    linewidth=0.8,
    alpha=0.7,
    label=None,
):
    """Draw all cube boundaries on a Matplotlib three-dimensional axes."""
    first_segment = True
    for start, end in cube_edges(centre, half_width):
        axes.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=colour,
            linewidth=linewidth,
            alpha=alpha,
            label=label if first_segment else None,
        )
        first_segment = False


def set_axes_equal(axes, points: Optional[Iterable] = None):
    """Set equal data ranges on all axes of a three-dimensional plot."""
    if points is not None:
        point_array = np.asarray(list(points), dtype=float)
        minimum = np.min(point_array, axis=0)
        maximum = np.max(point_array, axis=0)
    else:
        minimum = np.array(
            [axes.get_xlim3d()[0], axes.get_ylim3d()[0], axes.get_zlim3d()[0]]
        )
        maximum = np.array(
            [axes.get_xlim3d()[1], axes.get_ylim3d()[1], axes.get_zlim3d()[1]]
        )

    centre = 0.5 * (minimum + maximum)
    radius = 0.5 * np.max(maximum - minimum)
    radius = max(radius, 1.0e-12)
    axes.set_xlim3d(centre[0] - radius, centre[0] + radius)
    axes.set_ylim3d(centre[1] - radius, centre[1] + radius)
    axes.set_zlim3d(centre[2] - radius, centre[2] + radius)


def vec3_to_array(vector):
    """Convert a bound cdfmm.Vec3 to a NumPy vector."""
    return np.array([vector.x, vector.y, vector.z], dtype=float)


def nodes_at_level(tree, level):
    """Return complete-tree nodes at one level in Morton order."""
    return [node for node in tree.nodes if node.level == level]


def finish_3d_axes(axes, title):
    """Apply consistent geometry labels and layout-friendly styling."""
    axes.set_title(title)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    set_axes_equal(axes)


def new_3d_figure(figsize=(8, 7)):
    """Create a consistently sized three-dimensional figure and axes."""
    figure = plt.figure(figsize=figsize)
    axes = figure.add_subplot(111, projection="3d")
    return figure, axes
