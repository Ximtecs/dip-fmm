import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from examples.notebooks.fmm3d_comparison import (
    fmm3d_laplace_nterms,
    fmm3d_source_fields,
    is_out_of_memory_error,
    relative_error_metrics,
    timed_call,
    validate_source_point_geometry,
)


REPOSITORY_ROOT = Path(__file__).parents[1]
NOTEBOOK = REPOSITORY_ROOT / "examples/notebooks/11_fmm3d_comparison.ipynb"
INSTALLER = REPOSITORY_ROOT / "examples/notebooks/install_fmm3d.sh"


class Fmm3dOutput:
    def __init__(self, gradient):
        self.grad = gradient


@pytest.mark.parametrize(
    ("eps", "nterms"),
    [(1.0e-3, 12), (1.0e-6, 25), (1.0e-9, 37)],
)
def test_fmm3d_laplace_order_matches_pinned_upstream_routine(eps, nterms):
    assert fmm3d_laplace_nterms(eps) == nterms


@pytest.mark.parametrize("eps", [0.0, -1.0, np.inf, np.nan])
def test_fmm3d_laplace_order_rejects_invalid_tolerances(eps):
    with pytest.raises(ValueError, match="positive finite"):
        fmm3d_laplace_nterms(eps)


def test_fmm3d_single_density_gradient_maps_to_cdfmm_field_rows():
    gradient = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    actual = fmm3d_source_fields(Fmm3dOutput(gradient))

    np.testing.assert_array_equal(
        actual,
        np.array([[-1.0, -3.0, -5.0], [-2.0, -4.0, -6.0]]),
    )
    assert actual.flags.c_contiguous


def test_fmm3d_vectorized_gradients_map_to_density_target_component_order():
    gradient = np.arange(24.0).reshape(2, 3, 4)

    actual = fmm3d_source_fields(Fmm3dOutput(gradient))

    assert actual.shape == (2, 4, 3)
    np.testing.assert_array_equal(actual[1, 2], -gradient[1, :, 2])


def test_fmm3d_gradient_adapter_rejects_unknown_shapes():
    with pytest.raises(ValueError, match="FMM3D gradient"):
        fmm3d_source_fields(Fmm3dOutput(np.zeros((4, 3))))


def test_relative_error_metrics_report_rms_and_maximum():
    reference = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    approximate = np.array([[1.1, 0.0, 0.0], [0.0, 2.4, 0.0]])

    metrics = relative_error_metrics(approximate, reference)

    assert metrics.rms == pytest.approx(np.sqrt((0.1**2 + 0.2**2) / 2.0))
    assert metrics.maximum == pytest.approx(0.2)


def test_timed_call_returns_the_synchronous_result():
    result, seconds = timed_call(lambda: "complete")

    assert result == "complete"
    assert seconds >= 0.0


@pytest.mark.parametrize(
    "error",
    [
        MemoryError(),
        RuntimeError("CUDA failure: out of memory"),
        RuntimeError("cudaErrorMemoryAllocation"),
        RuntimeError("std::bad_alloc"),
    ],
)
def test_out_of_memory_detection_accepts_host_and_cuda_failures(error):
    assert is_out_of_memory_error(error)


def test_out_of_memory_detection_rejects_unrelated_runtime_failures():
    assert not is_out_of_memory_error(
        RuntimeError("CUDA kernel launch failed: invalid configuration")
    )


def test_source_point_geometry_accepts_coincident_identity_mapping():
    positions = np.array([[0.0, 0.0, 0.0], [0.5, -0.25, 0.75]])

    validate_source_point_geometry(positions, positions.copy(), np.arange(2))


def test_source_point_geometry_rejects_invalid_source_shape():
    with pytest.raises(ValueError, match="source positions"):
        validate_source_point_geometry(
            np.zeros(3), np.zeros((1, 3)), np.arange(1)
        )


@pytest.mark.parametrize(
    ("targets", "identities", "message"),
    [
        (np.zeros((3, 3)), np.arange(2), "matching shapes"),
        (
            np.array([[0.0, 0.0, 0.0], [0.5, -0.25, 0.70]]),
            np.arange(2),
            "identical",
        ),
        (
            np.array([[0.0, 0.0, 0.0], [0.5, -0.25, 0.75]]),
            np.array([1, 0]),
            "arange",
        ),
    ],
)
def test_source_point_geometry_rejects_invalid_targets_or_identities(
    targets, identities, message
):
    sources = np.array([[0.0, 0.0, 0.0], [0.5, -0.25, 0.75]])

    with pytest.raises(ValueError, match=message):
        validate_source_point_geometry(sources, targets, identities)


def test_comparison_notebook_is_valid_json_with_compilable_code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["display_name"] == "cdfmm"
    sources = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    assert "GPU_ONLY = True" in sources[0]
    assert "PARTICLE_COUNTS = [20_000, 40_000, 60_000]" in sources[0]
    assert "EXPANSION_ORDERS = [4, 6, 8]" in sources[0]
    assert "TREE_DEPTHS = [3, 4]" in sources[0]
    assert "FMM3D_EPS_VALUES = [1.0e-3, 1.0e-4]" in sources[0]
    assert "VECTOR_BATCH_SIZE is added by this notebook" in sources[0]
    combined = "\n".join(sources)
    assert "ExecutionBackend.CUDA_FULL" in combined
    assert "ExecutionBackend.CUDA_PARTIAL" in combined
    assert "StaticMatrixBackend.ONE_MKL" in combined
    assert "if not GPU_ONLY and not cdfmm.one_mkl_available()" in combined
    assert "if not GPU_ONLY:" in combined
    assert "is_out_of_memory_error(error)" in combined
    assert "skipped_cases.append" in combined
    assert "median_timed_evaluations" in combined
    assert "plan = make_cdfmm_plan" in combined
    assert "options.fixed_target_source_indices = identities.tolist()" in combined
    assert "target_source_indices=identities" not in combined
    assert "for _ in range(WARMUP_EVALUATIONS)" in combined
    assert "nd=len(batch)" in combined
    for index, source in enumerate(sources):
        compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec")


def test_fmm3d_installer_rejects_an_inactive_environment():
    environment = os.environ.copy()
    environment.pop("CONDA_DEFAULT_ENV", None)
    environment.pop("CONDA_PREFIX", None)

    result = subprocess.run(
        ["bash", str(INSTALLER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Activate the cdfmm Conda environment" in result.stderr


def test_fmm3d_installer_short_circuits_when_pinned_version_is_present(
    tmp_path,
):
    conda_prefix = tmp_path / "cdfmm"
    conda_bin = conda_prefix / "bin"
    conda_bin.mkdir(parents=True)
    fake_python = conda_bin / "python"
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    (conda_prefix / ".cdfmm-fmm3d-2.1.0").write_text(
        "v2.1.0\n", encoding="utf-8"
    )
    environment.update(
        CONDA_DEFAULT_ENV="cdfmm",
        CONDA_PREFIX=str(conda_prefix),
        PATH=f"{conda_bin}:{environment['PATH']}",
    )

    for _ in range(2):
        result = subprocess.run(
            ["bash", str(INSTALLER)],
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "already installed" in result.stdout
