import numpy as np

import cdfmm


def geometry():
    rng = np.random.default_rng(7201)
    sources = rng.uniform(-1.0, 1.0, (24, 3))
    targets = rng.uniform(-0.9, 0.9, (12, 3))
    moments = rng.normal(size=(24, 3))
    return sources, targets, moments


def test_performance_adviser_returns_a_tested_depth():
    sources, targets, moments = geometry()
    result = cdfmm.suggest_depth_for_performance(
        sources,
        targets,
        moments,
        order=3,
        candidate_depths=[1, 2],
        repetitions=1,
    )
    assert result["suggested_depth"] in {1, 2}
    assert result["branches_concurrent"] is False
    assert {candidate["depth"] for candidate in result["candidates"]} == {1, 2}
    assert all(candidate["evaluation_seconds"] > 0.0
               for candidate in result["candidates"])


def test_accuracy_sampling_is_deterministic_and_selected_pair_passes():
    sources, targets, moments = geometry()
    arguments = dict(
        desired_accuracy=2.0,
        candidate_orders=[2, 3],
        candidate_depths=[1],
        sample_size=7,
        repetitions=1,
    )
    first = cdfmm.suggest_parameters_for_accuracy(
        sources, targets, moments, **arguments
    )
    second = cdfmm.suggest_parameters_for_accuracy(
        sources, targets, moments, **arguments
    )
    assert first["reference_target_indices"] == second["reference_target_indices"]
    assert first["reference_target_count"] == 7
    selected = next(
        candidate for candidate in first["candidates"]
        if candidate["order"] == first["suggested_order"]
        and candidate["depth"] == first["suggested_depth"]
    )
    assert selected["satisfies_accuracy"]
    assert selected["rms_relative_error"] <= arguments["desired_accuracy"]


def test_unreasonable_candidate_is_recorded_not_raised():
    sources, targets, moments = geometry()
    result = cdfmm.suggest_depth_for_performance(
        sources, targets, moments, order=2,
        candidate_depths=[1, 9], repetitions=1
    )
    failed = next(candidate for candidate in result["candidates"]
                  if candidate["depth"] == 9)
    assert failed["status"] == "failed"
    assert failed["reason"]
