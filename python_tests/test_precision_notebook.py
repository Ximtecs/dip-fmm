import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    REPOSITORY_ROOT
    / "examples"
    / "simple_notebooks"
    / "simple_dense_direct_precision_compare.ipynb"
)


def test_precision_notebook_is_human_readable_and_compilable():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["display_name"] == "cdfmm"

    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    for source in code_cells:
        compile(source, str(NOTEBOOK), "exec")

    complete_source = "\n".join(code_cells)
    assert 'construct_plan("float64")' in complete_source
    assert 'construct_plan("float32")' in complete_source
    assert "tensor_memory_bytes" in complete_source
    assert "relative_l2" in complete_source
    assert "time.perf_counter()" in complete_source
    assert "UniformFmm(" not in complete_source
