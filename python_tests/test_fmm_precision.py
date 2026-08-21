"""Precision selection and dtype behaviour for complete FMM plans."""

import numpy as np
import pytest

import cdfmm


def geometry(count=40):
    rng = np.random.default_rng(941)
    positions = rng.uniform(-0.9, 0.9, size=(count, 3))
    moments = rng.normal(size=(count, 3))
    identities = np.arange(count, dtype=np.int32)
    return positions, moments, identities


def make_options(precision, backend=cdfmm.ExecutionBackend.CPU_STATIC):
    options = cdfmm.UniformFmmOptions()
    options.expansion_order = 4
    options.tree.max_level = 3
    options.backend = backend
    options.precision = precision
    return options


def test_default_precision_is_float32():
    positions, moments, identities = geometry()
    plan = cdfmm.UniformFmm(positions, positions, cdfmm.UniformFmmOptions())
    result = plan.evaluate(moments, target_source_indices=identities)
    assert plan.precision == cdfmm.StaticPrecision.FLOAT32
    assert result["H"].dtype == np.float32
    assert result["phi"].dtype == np.float32
    assert plan.root_multipole.dtype == np.float32


@pytest.mark.parametrize("input_dtype", [np.float32, np.float64])
def test_float32_plan_accepts_both_input_dtypes_and_returns_float32(input_dtype):
    positions, moments, identities = geometry()
    plan = cdfmm.UniformFmm(
        positions.astype(input_dtype),
        positions.astype(input_dtype),
        make_options(cdfmm.StaticPrecision.FLOAT32),
    )
    result = plan.evaluate(
        moments.astype(input_dtype), target_source_indices=identities
    )
    assert plan.precision == cdfmm.StaticPrecision.FLOAT32
    assert result["H"].dtype == np.float32
    assert result["phi"].dtype == np.float32
    assert plan.root_multipole.dtype == np.float32
    assert plan.multipole(0).dtype == np.float32
    assert plan.local(0).dtype == np.float32
    assert plan.static_plan_statistics["scalar_bytes"] == 4


def test_float32_and_float64_fmm_agree_and_state_storage_halves():
    positions, moments, identities = geometry()
    fp32 = cdfmm.UniformFmm(
        positions,
        positions,
        make_options(cdfmm.StaticPrecision.FLOAT32),
    )
    fp64 = cdfmm.UniformFmm(
        positions,
        positions,
        make_options(cdfmm.StaticPrecision.FLOAT64),
    )
    field32 = fp32.evaluate(moments, target_source_indices=identities)["H"]
    field64 = fp64.evaluate(moments, target_source_indices=identities)["H"]
    np.testing.assert_allclose(field32, field64, rtol=2.0e-5, atol=2.0e-5)
    assert fp32.static_plan_statistics["state_bytes"] * 2 == (
        fp64.static_plan_statistics["state_bytes"] + identities.nbytes
    )
    assert fp32.static_plan_statistics["operator_bytes"] < (
        fp64.static_plan_statistics["operator_bytes"]
    )


@pytest.mark.parametrize(
    "backend,matrix_backend",
    [
        (cdfmm.ExecutionBackend.CPU_REFERENCE, None),
        (cdfmm.ExecutionBackend.CPU_STATIC, cdfmm.StaticMatrixBackend.PORTABLE),
        pytest.param(
            cdfmm.ExecutionBackend.CPU_STATIC,
            cdfmm.StaticMatrixBackend.ONE_MKL,
            marks=pytest.mark.skipif(
                not cdfmm.one_mkl_available(), reason="oneMKL is unavailable"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "precision", [cdfmm.StaticPrecision.FLOAT32, cdfmm.StaticPrecision.FLOAT64]
)
def test_cpu_backends_support_both_precisions(backend, matrix_backend, precision):
    positions, moments, identities = geometry(24)
    options = make_options(precision, backend)
    if matrix_backend is not None:
        options.static_matrix_backend = matrix_backend
    plan = cdfmm.UniformFmm(positions, positions, options)
    result = plan.evaluate(moments, target_source_indices=identities)["H"]
    expected_dtype = (
        np.float32
        if precision == cdfmm.StaticPrecision.FLOAT32
        else np.float64
    )
    assert result.dtype == expected_dtype
    assert np.isfinite(result).all()


@pytest.mark.skipif(not cdfmm.cuda_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "backend",
    [cdfmm.ExecutionBackend.CUDA_PARTIAL, cdfmm.ExecutionBackend.CUDA_FULL],
)
@pytest.mark.parametrize(
    "precision", [cdfmm.StaticPrecision.FLOAT32, cdfmm.StaticPrecision.FLOAT64]
)
def test_cuda_fmm_backends_support_both_precisions(backend, precision):
    positions, moments, identities = geometry(24)
    options = make_options(precision, backend)
    options.fixed_target_source_indices = identities.tolist()
    plan = cdfmm.UniformFmm(positions, positions, options)
    result = plan.evaluate(moments, target_source_indices=identities)["H"]
    expected_dtype = (
        np.float32
        if precision == cdfmm.StaticPrecision.FLOAT32
        else np.float64
    )
    assert result.dtype == expected_dtype
    assert plan.cuda_plan_statistics["scalar_bytes"] in (4, 8)
    assert np.isfinite(result).all()
