"""The timing level is an accounting switch: it decides which timing fields a
plan collects and must never change results or the resolved policy."""
import numpy as np
import pytest

import cdfmm


def _plan(level, backend=cdfmm.ExecutionBackend.CPU_STATIC):
    rng = np.random.default_rng(11)
    positions = rng.uniform(-0.9, 0.9, size=(150, 3))
    moments = rng.normal(size=(150, 3))
    identities = np.arange(150, dtype=np.int32)
    options = cdfmm.UniformFmmOptions()
    options.expansion_order = 4
    options.tree.max_level = 3
    options.tree.root_centre = cdfmm.Vec3(0.0, 0.0, 0.0)
    options.tree.root_half_width = 1.0
    options.backend = backend
    options.fixed_target_source_indices = identities.tolist()
    options.enable_cache = False
    options.timing_level = level
    plan = cdfmm.UniformFmm(positions, positions, options)
    return plan, moments, identities


def test_timing_level_is_off_by_default_and_exposed_as_an_enum():
    options = cdfmm.UniformFmmOptions()
    assert options.timing_level == cdfmm.TimingLevel.OFF
    assert set(cdfmm.TimingLevel.__members__) == {"OFF", "COARSE", "DETAILED"}
    plan, moments, identities = _plan(cdfmm.TimingLevel.OFF)
    assert plan.timing_level == cdfmm.TimingLevel.OFF
    plan.evaluate(moments, target_source_indices=identities)
    timings = plan.last_timings
    assert timings["timing_level"] == cdfmm.TimingLevel.OFF
    assert timings["evaluations"] == 1
    assert timings["total"] == 0.0
    assert timings["far_field"] == 0.0
    assert timings["p2m"] == 0.0
    statistics = plan.static_plan_statistics
    assert statistics["timing_level"] == cdfmm.TimingLevel.OFF
    assert statistics["total_setup_seconds"] == 0.0
    assert statistics["operator_bytes"] > 0


def test_coarse_and_detailed_levels_populate_their_fields():
    plan, moments, identities = _plan(cdfmm.TimingLevel.COARSE)
    plan.evaluate(moments, target_source_indices=identities)
    coarse = plan.last_timings
    assert coarse["timing_level"] == cdfmm.TimingLevel.COARSE
    assert coarse["total"] >= coarse["far_field"] >= 0.0
    assert coarse["p2m"] == 0.0 and coarse["m2l"] == 0.0
    assert plan.static_plan_statistics["total_setup_seconds"] > 0.0
    assert plan.static_plan_statistics["p2m_construction_seconds"] == 0.0

    plan, moments, identities = _plan(cdfmm.TimingLevel.DETAILED)
    plan.evaluate(moments, target_source_indices=identities)
    detailed = plan.last_timings
    assert detailed["timing_level"] == cdfmm.TimingLevel.DETAILED
    assert detailed["total"] > 0.0
    assert all(detailed[key] >= 0.0
               for key in ("moment_permutation", "p2m", "m2m", "m2l", "l2l",
                           "l2p", "p2p", "result_unpermutation"))
    assert plan.static_plan_statistics["p2p_construction_seconds"] >= 0.0
    assert plan.static_plan_statistics["total_setup_seconds"] > 0.0


def test_runtime_level_change_keeps_results_and_resets_the_aggregate():
    plan, moments, identities = _plan(cdfmm.TimingLevel.OFF)
    reference = plan.evaluate(moments, target_source_indices=identities)["H"]
    packing = plan.p2p_execution_packing
    for level in (cdfmm.TimingLevel.DETAILED, cdfmm.TimingLevel.COARSE,
                  cdfmm.TimingLevel.OFF):
        plan.set_timing_level(level)
        assert plan.timing_level == level
        assert plan.aggregate_timings["evaluations"] == 0
        result = plan.evaluate(moments, target_source_indices=identities)["H"]
        np.testing.assert_array_equal(result, reference)
        assert plan.p2p_execution_packing == packing
        assert plan.last_timings["timing_level"] == level
    plan.evaluate(moments, target_source_indices=identities)
    assert plan.aggregate_timings["evaluations"] == 2
    plan.reset_timings()
    assert plan.aggregate_timings["evaluations"] == 0


@pytest.mark.skipif(not cdfmm.cuda_full_available(),
                    reason="the full CUDA backend is unavailable")
def test_cuda_full_results_do_not_depend_on_the_timing_level():
    plan, moments, identities = _plan(cdfmm.TimingLevel.OFF,
                                      cdfmm.ExecutionBackend.CUDA_FULL)
    reference = plan.evaluate(moments, target_source_indices=identities)["H"]
    assert plan.last_timings["cuda_p2p_kernel"] == 0.0
    # Device accumulation with atomics is not bitwise reproducible between
    # evaluations, so the level is held to the backend's FP32 precision.
    plan.set_timing_level(cdfmm.TimingLevel.DETAILED)
    detailed = plan.evaluate(moments, target_source_indices=identities)["H"]
    scale = np.abs(reference).max()
    np.testing.assert_allclose(detailed, reference, rtol=0.0, atol=1e-5 * scale)
    assert plan.last_timings["total"] > 0.0
    assert plan.last_timings["cuda_d2h"] >= 0.0
