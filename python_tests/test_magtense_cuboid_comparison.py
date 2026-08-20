# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    REPOSITORY_ROOT
    / "examples"
    / "simple_notebooks"
    / "simple_cuboid_magtense_compare.ipynb"
)
MAGTENSE_ENVIRONMENT = REPOSITORY_ROOT / "environment-magtense.yml"
CMAKE_PRESETS = REPOSITORY_ROOT / "CMakePresets.json"


def test_magtense_environment_pins_compatible_packages():
    environment = MAGTENSE_ENVIRONMENT.read_text(encoding="utf-8")

    assert "name: cdfmm-magtense" in environment
    assert "- python=3.11" in environment
    assert "- magtense=2.2.0=py3.11_cpu" in environment
    assert "- mkl-devel=2023.1.0" in environment
    assert "- numpy=1.25.2" in environment
    assert "- setuptools<81" in environment
    assert "cmt-dtu-energy/label/cpu" in environment


def test_magtense_build_preset_installs_python_module():
    presets = json.loads(CMAKE_PRESETS.read_text(encoding="utf-8"))
    configure = {
        preset["name"]: preset for preset in presets["configurePresets"]
    }
    builds = {preset["name"]: preset for preset in presets["buildPresets"]}

    assert configure["magtense"]["binaryDir"] == "${sourceDir}/build-magtense"
    assert configure["magtense"]["inherits"] == "notebooks"
    assert configure["magtense"]["cacheVariables"]["MKL_THREADING"] == (
        "intel_thread"
    )
    assert configure["magtense"]["cacheVariables"]["CDFMM_ENABLE_OPENMP"] == (
        "OFF"
    )
    assert (
        configure["magtense"]["cacheVariables"][
            "CDFMM_INSTALL_PYTHON_TO_ENV"
        ]
        == "ON"
    )
    assert builds["magtense"]["targets"] == ["install"]


def test_magtense_cuboid_notebook_is_valid_and_compilable():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert (
        notebook["metadata"]["kernelspec"]["display_name"]
        == "cdfmm-magtense"
    )

    sources = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    combined_source = "\n".join(sources)

    assert "cdfmm.DenseDirectPlan" in combined_source
    assert "cdfmm.CudaDenseDirectPlan" in combined_source
    assert 'CDFMM_BACKEND = "cuda"' in combined_source
    assert '"normal-cpu", "mkl-cpu", "cuda"' in combined_source
    assert "cdfmm.DenseDirectBackend.PORTABLE" in combined_source
    assert "cdfmm.DenseDirectBackend.ONE_MKL" in combined_source
    assert "cdfmm.SourceGeometry.UNIFORM_CUBOID" in combined_source
    assert "cdfmm.TargetGeometry.POINT" in combined_source
    assert "magstatics.get_H_field" in combined_source
    assert "H_magtense = np.asarray" in combined_source
    assert "CUBE_VOLUME_M3 * magnetisations" in combined_source
    assert "N_EVALUATION_POINTS = 10_000" in combined_source
    assert "TIMING_REPEATS = 20" in combined_source
    assert "magstatics.get_demag_tensor" in combined_source
    assert "difference[:, component]" in combined_source
    assert "np.maximum(absolute_error" in combined_source
    assert "np.maximum(pointwise_relative_error" in combined_source
    assert "print(H_cdfmm)" not in combined_source
    assert "print(H_magtense)" not in combined_source

    for source in sources:
        compile(source, str(NOTEBOOK), "exec")


@pytest.mark.skipif(
    importlib.util.find_spec("magtense") is None,
    reason="MagTense is an optional validation dependency",
)
def test_uniform_cube_direct_field_matches_magtense():
    import cdfmm
    from magtense import magstatics

    cube_side = 1.0
    cube_volume = cube_side**3
    centres = 3.0 * np.array(
        [
            [x, y, z]
            for x in (-0.5, 0.5)
            for y in (-0.5, 0.5)
            for z in (-0.5, 0.5)
        ],
        dtype=np.float64,
    )
    magnetisations = 1.0e6 * np.array(
        [
            [1.0, 0.2, -0.1],
            [-0.3, 0.8, 0.4],
            [0.1, -0.6, 0.9],
            [0.7, 0.2, 0.5],
            [-0.4, -0.3, 0.8],
            [0.6, -0.7, -0.2],
            [-0.8, 0.4, -0.3],
            [0.2, 0.9, -0.5],
        ],
        dtype=np.float64,
    )

    cube_size = cdfmm.CuboidSize(cube_side, cube_side, cube_side)
    plan = cdfmm.DenseDirectPlan(
        centres,
        centres,
        cdfmm.SourceGeometry.UNIFORM_CUBOID,
        cdfmm.TargetGeometry.POINT,
        [cube_size],
    )
    cdfmm_field = np.asarray(plan.evaluate(cube_volume * magnetisations))

    tiles = magstatics.Tiles(
        n=len(centres),
        tile_type=2,
        size=[cube_side, cube_side, cube_side],
        offset=np.asfortranarray(centres),
        rot=[0.0, 0.0, 0.0],
        M_rem=0.0,
    )
    tiles.M = np.asfortranarray(magnetisations)
    magtense_field = np.asarray(
        magstatics.get_H_field(tiles, np.asfortranarray(centres))
    )

    relative_l2_error = (
        np.linalg.norm(cdfmm_field - magtense_field)
        / np.linalg.norm(magtense_field)
    )
    assert relative_l2_error < 5.0e-5
