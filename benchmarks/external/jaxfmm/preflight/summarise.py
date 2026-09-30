#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print the essential numbers of a preflight output directory."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    out = Path(sys.argv[1])
    jax_path = out / "jaxfmm_report.json"
    dip_path = out / "dipfmm_report.json"
    if jax_path.exists():
        r = json.loads(jax_path.read_text())
        print("versions", r.get("versions"))
        print("devices", r.get("devices"))
        print("precision", r.get("precision"))
        print("jax_environment", r.get("jax_environment"))
        print("jaxfmm_git", r.get("jaxfmm_git"))
        print("process", {k: r["process"].get(k) for k in ("affinity", "core_class")})
        print("stage", r.get("stage"))
        print("memory_after_import", r.get("memory_after_import"))
        print("memory_at_exit", r.get("memory_at_exit"))
        for engine, rec in r.get("engines", {}).items():
            print(f"--- {engine}")
            print("  parameters", rec.get("parameters"))
            print("  setup_info", rec.get("setup_info"))
            print("  setup_seconds", rec.get("setup_seconds"))
            print("  monopole_potential_rel_l2", rec.get("monopole_potential_relative_l2"))
            print("  monopole_field_rel_l2", rec.get("monopole_field_relative_l2"))
            print("  dipole_potential_rel_l2", rec.get("dipole_potential_vs_direct_relative_l2"))
            m = rec.get("dipole_field_vs_direct", {})
            print("  dipole_field_vs_direct rel_l2", m.get("relative_l2"), "max/rms", m.get("max_absolute_over_reference_rms"),
                  "max_pointwise", m.get("max_pointwise_relative"))
            print("  output dtype/shape", rec.get("dipole_output_dtype"), rec.get("dipole_output_shape"), "finite", rec.get("dipole_finite"))
            print("  doubled_cloud_vs_direct", rec.get("doubled_cloud_vs_direct_relative_l2"),
                  "split_vs_doubled", rec.get("split_vs_doubled_cloud_relative_l2"))
            print("  second_state rel_l2", rec.get("second_state_vs_direct", {}).get("relative_l2"),
                  "fixed_vs_rebuilt", rec.get("second_state_fixed_vs_rebuilt_relative_l2"),
                  "repeat_max_diff", rec.get("repeat_max_relative_difference"))
            t = rec.get("timing", {})
            print("  timing first_call", t.get("first_call_seconds"), "device median", t.get("device_evaluation_median_seconds"),
                  "host median", t.get("host_evaluation_median_seconds"))
            print("  memory_after_engine", rec.get("memory_after_engine"))
    if dip_path.exists():
        r = json.loads(dip_path.read_text())
        print("=== dip-fmm")
        print("module", r.get("module"))
        print("process", {k: r["process"].get(k) for k in ("affinity", "core_class")})
        print("options", r.get("options"))
        print("plan", {k: v for k, v in r.get("plan", {}).items() if not k.endswith("statistics")})
        print("cuda_plan_statistics", r.get("plan", {}).get("cuda_plan_statistics"))
        print("construction_seconds", r.get("construction_seconds"), "first_call", r.get("first_call_seconds"),
              "host median", r.get("host_evaluation_median_seconds"))
        print("output dtype", r.get("output_dtype"), "keys", r.get("output_keys"))
        m = r.get("dipole_field_vs_direct", {})
        print("dipole_field_vs_direct rel_l2", m.get("relative_l2"), "max/rms", m.get("max_absolute_over_reference_rms"))
        print("dipole_potential_rel_l2", r.get("dipole_potential_vs_direct_relative_l2"))
        print("second_state rel_l2", r.get("second_state_vs_direct", {}).get("relative_l2"),
              "repeat_max_diff", r.get("repeat_max_relative_difference"))
        print("gpu_state_while_plan_alive", r.get("gpu_state_while_plan_alive"))
    pre = out / "preflight_report.json"
    if pre.exists():
        r = json.loads(pre.read_text())
        print("=== gate")
        print("gpu_before", r.get("gpu_before"))
        print("gpu_after_jaxfmm", r.get("gpu_after_jaxfmm"))
        print("gpu_after_dipfmm", r.get("gpu_after_dipfmm"))
        print("workers", {k: {kk: vv for kk, vv in r[k].items() if kk != "command"} for k in ("jaxfmm_worker", "dipfmm_worker") if k in r})
        print("comparisons", r.get("comparisons"))
        print("result", r.get("result"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
