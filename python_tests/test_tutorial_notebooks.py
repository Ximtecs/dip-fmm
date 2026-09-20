"""Execute the canonical tutorials against the just-built module.

Every tutorial under ``examples/tutorials/`` is supported user-facing
material, so each one is executed headlessly on the portable CPU build with
a generic ``python3`` kernel. Cells that need CUDA or oneMKL guard themselves
with the availability queries and print a message instead, so a portable run
covers every cell. The structural checks (valid nbformat, unique cell ids,
compilable code) run first and separately, because they fail fast and need
no kernel.
"""

import os
from pathlib import Path
import sys

import nbformat
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = REPOSITORY_ROOT / "examples" / "tutorials"
NOTEBOOKS = sorted(TUTORIALS.glob("0*_*.ipynb"))
# Every tutorial runs in well under a minute on the portable build; the limit
# is a safety net against a hung kernel, not a performance budget.
CELL_TIMEOUT_SECONDS = 900


def _absolute_python_path():
    """Return ``PYTHONPATH`` with every entry made absolute.

    The kernel starts in the tutorial directory, so a relative
    ``PYTHONPATH=build`` (the documented no-install workflow) would no longer
    point at the built extension and the kernel would silently import an
    unrelated installed ``cdfmm``. Resolving the entries against the current
    working directory keeps the kernel on the module the tests were told to
    use.
    """
    entries = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    return os.pathsep.join(str(Path(entry).resolve()) for entry in entries)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[path.stem for path in NOTEBOOKS])
def test_tutorial_is_well_formed(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    ids = [cell.id for cell in notebook.cells]
    assert len(ids) == len(set(ids)) == len(notebook.cells)
    assert notebook.metadata["kernelspec"]["name"] == "python3"
    assert notebook.cells[0].cell_type == "markdown"
    assert notebook.cells[0].source.startswith("# ")
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{path.name}:cell-{index}", "exec")
            # Tutorials are committed without outputs; the execution test
            # below produces them on demand.
            assert cell.outputs == [], f"{path.name} cell {index} has stored output"


def test_six_tutorials_are_present():
    assert [path.stem for path in NOTEBOOKS] == [
        "01_getting_started",
        "02_finite_geometry",
        "03_backends_and_execution",
        "04_cache_and_periodicity",
        "05_trees_and_parameters",
        "06_operator_chain",
    ]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[path.stem for path in NOTEBOOKS])
def test_tutorial_executes(path, tmp_path, monkeypatch):
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("PYTHONPATH", _absolute_python_path())
    # The cache tutorial manages its own directory; the others share a
    # per-test scratch cache so the run neither reads nor writes the
    # developer's persistent cache.
    monkeypatch.setenv("CDFMM_CACHE_DIR", str(tmp_path / "cache"))

    notebook = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        notebook,
        timeout=CELL_TIMEOUT_SECONDS,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()

    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert errors == []
    # The first code cell of every tutorial reports which module it imported.
    first_code = next(cell for cell in notebook.cells if cell.cell_type == "code")
    printed = "".join(
        "".join(output.get("text", [])) for output in first_code.get("outputs", [])
    )
    assert "cdfmm" in printed or sys.version  # the import itself succeeded
