import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[1]


def test_notebook_preset_supports_gnu_mkl_without_intel_header_leakage():
    presets = json.loads(
        (REPOSITORY_ROOT / "CMakePresets.json").read_text(encoding="utf-8")
    )
    configure_presets = {
        preset["name"]: preset for preset in presets["configurePresets"]
    }
    cmake_source = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )

    cuda = configure_presets["cuda"]
    notebooks = configure_presets["notebooks"]

    assert "project(cartesian_dipole_fmm LANGUAGES C CXX)" in cmake_source
    assert notebooks["cacheVariables"]["MKL_THREADING"] == "gnu_thread"
    assert cuda["environment"]["CPATH"] == ""
    assert cuda["environment"]["CPLUS_INCLUDE_PATH"] == ""
    assert cuda["environment"]["CXXFLAGS"] == ""


def test_combined_profiling_preset_enables_nvtx_with_debug_symbols():
    presets = json.loads(
        (REPOSITORY_ROOT / "CMakePresets.json").read_text(encoding="utf-8")
    )
    configure_presets = {
        preset["name"]: preset for preset in presets["configurePresets"]
    }
    build_presets = {
        preset["name"]: preset for preset in presets["buildPresets"]
    }

    profile = configure_presets["profile-all"]
    assert profile["inherits"] == "benchmark-all"
    assert profile["binaryDir"] == "${sourceDir}/build-profile-all"
    assert profile["cacheVariables"]["CMAKE_BUILD_TYPE"] == "RelWithDebInfo"
    assert profile["cacheVariables"]["CDFMM_ENABLE_PROFILING"] == "ON"
    assert build_presets["profile-all"]["configurePreset"] == "profile-all"


def test_cuda_environment_and_source_use_header_only_nvtx3():
    cuda_environment = (
        REPOSITORY_ROOT / "environment-cuda.yml"
    ).read_text(encoding="utf-8")
    cmake_source = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    profile_header = (REPOSITORY_ROOT / "src/profile.hpp").read_text(
        encoding="utf-8"
    )

    assert "cuda-nvtx-dev=13.3.*" in cuda_environment
    assert "CUDA::nvtx3" in cmake_source
    assert "CUDA::nvToolsExt" not in cmake_source
    assert "#include <nvtx3/nvToolsExt.h>" in profile_header


def test_profile_mode_defaults_to_ten_consecutive_evaluations():
    benchmark_source = (
        REPOSITORY_ROOT / "benchmarks" / "benchmark_uniform_fmm.cpp"
    ).read_text(encoding="utf-8")
    profile_defaults_start = benchmark_source.index("if (profile_requested)")
    argument_parsing_start = benchmark_source.index(
        "for (int index = 1; index < argc; ++index)",
        profile_defaults_start,
    )
    profile_defaults = benchmark_source[
        profile_defaults_start:argument_parsing_start
    ]

    assert "options.evaluations = 10;" in profile_defaults
    assert "options.warmups = 1;" in profile_defaults
    assert "options.samples = 1;" in profile_defaults
