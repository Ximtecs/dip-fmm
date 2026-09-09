"""Structural contract for the MagTense geometry all-to-all notebook.

The notebook intentionally contains a large (1000 x 1000) comparison.  This
test validates its executable structure and documented cases without running
any notebook cell.
"""

import re
from pathlib import Path

import nbformat


NOTEBOOK = (
    Path(__file__).parents[1]
    / "examples"
    / "simple_notebooks"
    / "simple_geometry_magtense_all_to_all_compare.ipynb"
)


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is not None


def test_geometry_magtense_all_to_all_notebook_contract():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)

    code_cells = [
        cell for cell in notebook.cells if cell.cell_type == "code"
    ]
    assert code_cells
    for index, cell in enumerate(code_cells):
        compile("".join(cell.source), f"{NOTEBOOK.name}:cell-{index}", "exec")

    sources = ["".join(cell.source) for cell in notebook.cells]
    combined = "\n".join(sources)
    lowered = combined.lower()

    # Visible workload: one thousand particles on a 10 x 10 x 10 grid.  The
    # alternatives accept either named constants or literal configuration.
    assert _has(r"\b1000\b", combined)
    assert _has(r"10\s*[x,]\s*10\s*[x,]\s*10|\(\s*10\s*,\s*10\s*,\s*10\s*\)", combined)

    # The comparison must use the persistent dense direct plan.  A one-level
    # UniformFmm is not an O(N^2) dense reference and is not an equivalent
    # substitute for the direct geometry plan.
    assert "DenseDirectPlan" in combined
    assert "DenseDirectBackend.PORTABLE" in combined
    assert "UniformFmm" not in combined
    assert "max_level" not in combined
    assert _has(r"def\s+run_dense\s*\(", combined)

    # Two public-MagTense accuracy comparisons and two CDFMM target-policy
    # comparisons are represented with explicit labels.
    for label in (
        "prism -> point",
        "prism target averaging impact (vs point target)",
        "tetrahedron -> point",
        "tetra geometry record + point target model",
    ):
        assert label in combined
    assert "physical-model impact" in lowered
    assert "not a numerical error" in lowered
    assert "prism" in lowered and "point" in lowered
    assert "tetra" in lowered
    assert _has(r"prism.{0,80}point|point.{0,80}prism", lowered)
    assert _has(r"tetra.{0,80}point|point.{0,80}tetra", lowered)

    # Use only tile types and arguments available in public MagTense 2.2.0.
    for tile_type in (2, 5):
        assert _has(rf"tile_type\s*=\s*{tile_type}\b", combined)
    assert not _has(r"tile_type\s*=\s*8\b", combined)
    assert "obs_size" not in lowered

    # The CDFMM physical geometry/model API is exercised explicitly by the
    # dense plan, including the tetrahedral records carried by the target.
    for token in (
        "SourceGeometry.RECTANGULAR_PRISM",
        "SourceGeometry.TETRAHEDRON",
        "TargetGeometry.POINT",
        "TargetGeometry.RECTANGULAR_PRISM",
        "TargetGeometry.TETRAHEDRON",
        "SourceModel.EXACT_GEOMETRY",
        "TargetModel.EXACT_GEOMETRY",
        "TargetModel.POINT",
    ):
        assert token in combined
    assert "target_source_indices" in combined

    # Self/coincident behavior must be asserted, and the physical tetrahedral
    # target + point target model comparison must be explained explicitly.
    assert _has(r"assert[^\n]*(self|coincident|identity)", lowered)
    assert any(
        "physical" in cell.lower()
        and "tetra" in cell.lower()
        and "targetmodel.point" in cell.lower()
        for cell in sources
    )
    assert _has(r"tetra.{0,120}(tetra|tetrahedron).{0,120}unsupported", lowered)

    # Figures cover the physical setup, MagTense parity, and the distinction
    # between numerical error and changing the target model.
    assert "matplotlib.pyplot" in combined
    assert "Poly3DCollection" in combined
    assert "setup-figure" in {cell.id for cell in notebook.cells}
    assert "result-figures" in {cell.id for cell in notebook.cells}
    assert _has(r"scatter\s*\(", combined)
    assert _has(r"quiver\s*\(", combined)
    assert "ideal parity" in lowered
    assert "target-model effect (not numerical error)" in lowered
    assert "no tetrahedron target-averaging curve is shown" in lowered

    # Exact tetra-target averaging is unsupported.  The equal tetra curves
    # are explicitly attributed to TargetModel.POINT, not to small tiles.
    assert "by construction" in lowered
    assert "not evidence that these tiles are small enough" in lowered
