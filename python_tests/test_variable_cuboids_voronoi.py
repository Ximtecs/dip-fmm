"""Per-body prism plans on an octree-refined Voronoi grain mesh: uniform and adaptive trees
against the dense reference, and the leaf-containment diagnostics."""

from pathlib import Path
import sys

import numpy as np
import pytest

import cdfmm as c

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "examples" / "validation"))

from voronoi_octree_mesh import random_moments, voronoi_octree_mesh  # noqa: E402


def prism_options(order, basis=c.ExpansionBasis.SPHERICAL):
    options = c.UniformFmmOptions()
    options.expansion_order = order
    options.expansion_basis = basis
    options.precision = c.StaticPrecision.FLOAT64
    options.backend = c.ExecutionBackend.CPU_STATIC
    options.source_geometry = c.SourceGeometry.RECTANGULAR_PRISM
    options.target_geometry = c.TargetGeometry.RECTANGULAR_PRISM
    return options


def with_sizes(options, mesh):
    sizes = [c.RectangularPrism(*s) for s in mesh.sizes]
    options.source_sizes = sizes
    options.target_sizes = sizes
    return options


SUBSET = 256   # dense FP64 reference on this many targets keeps the test under a minute


def subset_indices(mesh):
    n = len(mesh.centres)
    return np.arange(0, n, max(1, n // SUBSET))[:SUBSET]


def dense_field(mesh, moments):
    sizes = [c.RectangularPrism(*s) for s in mesh.sizes]
    sub = subset_indices(mesh)
    plan = c.DenseDirectPlan(mesh.centres, mesh.centres[sub], c.SourceGeometry.RECTANGULAR_PRISM,
                             c.TargetGeometry.RECTANGULAR_PRISM, sizes, [sizes[i] for i in sub], [], static_precision="float64")
    return np.asarray(plan.evaluate(moments, c.DenseDirectBackend.PORTABLE))


def field(plan, moments):
    return np.asarray(plan.evaluate(moments)["H"])


def relative_l2(mesh, full_field, dense_subset):
    sub = subset_indices(mesh)
    return float(np.linalg.norm(full_field[sub] - dense_subset) / np.linalg.norm(dense_subset))


@pytest.fixture(scope="module")
def mesh():
    return voronoi_octree_mesh(n_grains=8, base=4, levels=2, band=0.06, seed=3)   # ~2.9k prisms, three levels


@pytest.fixture(scope="module")
def reference(mesh):
    moments = random_moments(mesh)
    return moments, dense_field(mesh, moments)


def test_mesh_is_graded_and_tiles_the_box(mesh):
    s = mesh.summary()
    assert set(s["tiles_per_level"]) == {0, 1, 2}
    assert np.isclose(mesh.volumes.sum(), mesh.box**3)
    assert 0.05 < s["band_fraction"] < 0.6


def test_adaptive_prism_plan_matches_dense_and_keeps_bodies_in_leaves(mesh, reference):
    moments, dense = reference
    tree_options = c.AdaptiveTreeOptions()
    assert tree_options.max_depth == 5 and tree_options.max_particles_per_leaf == 32   # new defaults
    tree_options.root_centre = c.Vec3(0.0, 0.0, 0.0)
    tree_options.root_half_width = 0.5 * mesh.box * (1 + 1e-6)
    errors = []
    for order in (4, 8):
        plan = c.AdaptiveTree(mesh.centres, tree_options).build_fmm(with_sizes(prism_options(order), mesh))
        stats = plan.static_plan_statistics
        # With the defaults a coarse tile (box/6) can land in a box split by its fine neighbours
        # (down to box/32): the statistics report it and the plan still evaluates. The count is
        # geometry dependent, so it is recorded, not pinned; the error bound below holds anyway
        # for this scene.
        assert stats["source_bodies_exceeding_leaf"] == stats["target_bodies_exceeding_leaf"]
        assert np.isfinite(stats["max_body_leaf_extent_ratio"])
        errors.append(relative_l2(mesh, field(plan, moments), dense))
    assert errors[1] < errors[0] < 0.05


def test_capacity_large_enough_keeps_all_bodies_contained(mesh, reference):
    moments, dense = reference
    tree_options = c.AdaptiveTreeOptions()
    tree_options.max_depth = 1                   # leaves of box/2 contain every tile (largest = box/4)
    tree_options.max_particles_per_leaf = 32
    tree_options.root_centre = c.Vec3(0.0, 0.0, 0.0)
    tree_options.root_half_width = 0.5 * mesh.box * (1 + 1e-6)
    plan = c.AdaptiveTree(mesh.centres, tree_options).build_fmm(with_sizes(prism_options(6), mesh))
    stats = plan.static_plan_statistics
    assert stats["source_bodies_exceeding_leaf"] == 0
    assert stats["target_bodies_exceeding_leaf"] == 0
    assert stats["max_body_leaf_extent_ratio"] <= 1.0 + 1e-6
    assert relative_l2(mesh, field(plan, moments), dense) < 1e-2


def test_uniform_prism_plan_reports_protruding_coarse_tiles(mesh, reference):
    moments, dense = reference
    options = with_sizes(prism_options(6), mesh)
    options.tree.max_level = 1        # leaf = box/2 >= largest tile (box/4): contained
    contained = c.UniformFmm(mesh.centres, mesh.centres, options)
    assert contained.static_plan_statistics["source_bodies_exceeding_leaf"] == 0
    options.tree.max_level = 4        # leaf = box/16 < base tile box/4: coarse tiles protrude
    protruding = c.UniformFmm(mesh.centres, mesh.centres, options)
    n_coarse = int(np.sum(mesh.level == 0))
    assert protruding.static_plan_statistics["source_bodies_exceeding_leaf"] >= n_coarse
    assert protruding.static_plan_statistics["max_body_leaf_extent_ratio"] > 1.5
    # both still evaluate; accuracy of the protruding plan is reported, not asserted
    for plan in (contained, protruding):
        assert np.all(np.isfinite(field(plan, moments)))
    assert relative_l2(mesh, field(contained, moments), dense) < 1e-2
