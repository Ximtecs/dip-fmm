import json
from pathlib import Path

import numpy as np

import cdfmm


REPOSITORY_ROOT = Path(__file__).parents[1]
NOTEBOOK = REPOSITORY_ROOT / "examples/notebooks/13_parameter_selection.ipynb"


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


def test_parameter_selection_notebook_sweeps_particles_orders_and_depths():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["display_name"] == "cdfmm"
    sources = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    combined_source = "\n".join(sources)
    assert "particle_counts = [1000, 5000, 10000, 30000]" in sources[0]
    assert "performance_orders = [2, 4, 6, 8]" in sources[0]
    assert "candidate_depths = [2, 3, 4]" in sources[0]
    assert "for particle_count in particle_counts" in combined_source
    assert "for order in performance_orders" in combined_source
    assert "plt.subplots(" in combined_source
    for index, source in enumerate(sources):
        compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec")
