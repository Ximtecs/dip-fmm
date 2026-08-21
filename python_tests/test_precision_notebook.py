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
    assert "static_precision='float64'" in complete_source
    assert "cdfmm.StaticPrecision.FLOAT32" in complete_source
    assert "cdfmm.StaticPrecision.FLOAT64" in complete_source
    assert "relative_l2" in complete_source
    assert "time.perf_counter()" in complete_source
    assert "cdfmm.CudaDirectPlan(" in complete_source
    assert "cdfmm.UniformFmm(" in complete_source
    assert "cdfmm.ExecutionBackend.CUDA_FULL" in complete_source
    assert "cdfmm.ExecutionBackend.CUDA_PARTIAL" in complete_source
    assert "cdfmm.ExecutionBackend.CPU_REFERENCE" in complete_source
    assert "cdfmm.StaticMatrixBackend.ONE_MKL" in complete_source
    assert "ORDERS = (2, 4, 6)" in complete_source
    assert "TREE_DEPTHS = (2, 3, 4)" in complete_source
    assert "persistent_device_bytes" in complete_source
    assert "evaluation_h2d_bytes" in complete_source
    assert "evaluation_d2h_bytes" in complete_source
    assert "static_plan_statistics" in complete_source
