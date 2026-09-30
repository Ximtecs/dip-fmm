# SPDX-License-Identifier: Apache-2.0
"""dip-fmm half of the preflight: runs alone in the ``cdfmm`` environment.

Evaluates the same tiny lattice with the FP32 ``CudaFull`` point-target plan
at the campaign order and writes its fields next to the jaxFMM ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaxfmm_campaign import dipfmm_runner, environment, geometry, reference  # noqa: E402
from jaxfmm_campaign.timing import TimingProtocol  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cdfmm = dipfmm_runner.import_cdfmm()
    report: dict = {"process": environment.process_fingerprint(), "module": dipfmm_runner.module_report(cdfmm)}
    report["module"]["context_warm_seconds"] = dipfmm_runner.warm_cuda_context(cdfmm)

    dataset = geometry.lattice_dataset(args.grid, state="random")
    dataset_second = geometry.lattice_dataset(args.grid, state="vortex")
    positions = dataset.positions
    reference_first = reference.direct_dipole_field(positions, dataset.moments, positions, dataset.identity_map)
    reference_second = reference.direct_dipole_field(positions, dataset_second.moments, positions, dataset.identity_map)

    options = dipfmm_runner.make_options(cdfmm, order=args.order, depth=args.depth)
    report["options"] = dipfmm_runner.describe_options(options)
    plan, construction = dipfmm_runner.build_plan(cdfmm, positions, options)
    report["construction_seconds"] = construction
    report["plan"] = dipfmm_runner.plan_report(plan)

    protocol = TimingProtocol(warmups=2, samples=3, evaluations=3)
    summary, field_first, first_call = dipfmm_runner.time_plan(plan, dataset.moments, dataset.identity_map, protocol)
    raw = plan.evaluate(dataset.moments, target_source_indices=dataset.identity_map)
    report["output_dtype"] = str(raw["H"].dtype)
    report["output_keys"] = sorted(raw.keys())
    report["first_call_seconds"] = first_call
    report.update(summary.as_dict("host_evaluation"))
    report["dipole_field_vs_direct"] = reference.error_metrics(field_first, reference_first)
    report["dipole_finite"] = bool(np.isfinite(field_first).all())
    try:
        both = plan.evaluate(dataset.moments, output="both", target_source_indices=dataset.identity_map)
        phi = np.asarray(both["phi"], dtype=np.float64)
        phi_reference = reference.direct_dipole_potential(positions, dataset.moments, positions, dataset.identity_map)
        report["dipole_potential_vs_direct_relative_l2"] = float(
            np.linalg.norm(phi - phi_reference) / np.linalg.norm(phi_reference))
    except Exception as error:  # noqa: BLE001 - potential output is optional
        report["dipole_potential_vs_direct_relative_l2"] = f"unavailable: {error}"

    field_second = np.asarray(plan.evaluate(dataset_second.moments, target_source_indices=dataset.identity_map)["H"],
                              dtype=np.float64)
    report["second_state_vs_direct"] = reference.error_metrics(field_second, reference_second)
    repeats = [np.asarray(plan.evaluate(dataset_second.moments, target_source_indices=dataset.identity_map)["H"],
                          dtype=np.float64) for _ in range(3)]
    report["repeat_max_relative_difference"] = max(
        float(np.linalg.norm(r - field_second) / np.linalg.norm(field_second)) for r in repeats)
    report["gpu_state_while_plan_alive"] = environment.gpu_state()

    np.savez(out / "dipfmm_fields.npz", dipfmm_random=field_first, dipfmm_vortex=field_second,
             reference_random=reference_first, reference_vortex=reference_second)
    (out / "dipfmm_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"relative_l2": report["dipole_field_vs_direct"]["relative_l2"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
