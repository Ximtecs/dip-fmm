# SPDX-License-Identifier: Apache-2.0
"""jaxFMM half of the finite-source preflight (``jaxfmm`` environment).

Uniformly magnetised bodies (touching unit cubes of a lattice, or the Kuhn
tetrahedra of a conforming mesh) are represented by their face charges
``sigma_f = (M_left - M_right) . n_f`` on constant-charge triangles and
evaluated with jaxFMM's element path at the body centres. Writes the fields
for every (state, engine setting) so the orchestrator can compare them with
dip-fmm's exact analytic operators and the FP64 dense reference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaxfmm_campaign import environment, geometry  # noqa: E402
from jaxfmm_campaign.timing import TimingProtocol, TimingSummary  # noqa: E402


def build_body_mesh(kind: str, grid: int) -> dict:
    """Nodes, per-body boundary triangles, body centres and volumes for both families."""
    if kind == "lattice":
        mesh = geometry.cube_lattice_mesh(grid)
        triangles = geometry.cube_body_triangles(mesh["corner_ids"])
        centres = mesh["centres"]
        volumes = np.ones(len(centres))
        counts = {"cells": grid**3, "bodies": len(centres), "vertices": len(mesh["nodes"])}
    elif kind == "kuhn_mesh":
        mesh = geometry.kuhn_mesh(grid)
        triangles = geometry.tetra_body_triangles(mesh["connectivity"])
        centres = mesh["centroids"]
        volumes = np.full(len(centres), 1.0 / 6.0)
        counts = {"cells": grid**3, "bodies": len(centres), "vertices": len(mesh["nodes"]),
                  "faces": mesh["n_faces"]}
    else:
        raise ValueError(kind)
    faces = geometry.face_triangles(mesh["nodes"], triangles, centres)
    counts.update({"triangles": int(len(faces["triangles"])), "shared_triangles": faces["shared_count"],
                   "targets": int(len(centres))})
    return {"nodes": mesh["nodes"], "faces": faces, "centres": centres, "volumes": volumes, "counts": counts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="lattice", choices=("lattice", "kuhn_mesh"))
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--settings", default="p4,p6,p6n4,p6n6,p8n6",
                        help="comma list of pN[nK]: multipole order N, near-field degree K")
    parser.add_argument("--nmax", type=int, default=16,
                        help="elements per leaf; small so a tiny mesh still has far-field levels")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", message=".*dtype uint64.*")

    report: dict = {"kind": args.kind, "grid": args.grid, "settings": {}, "states": {}}
    report["process"] = environment.process_fingerprint()
    report["jax_environment"] = environment.jax_environment_variables()

    import jax
    import jax.numpy as jnp
    from jaxfmm_campaign import jaxfmm_dipole

    report["precision"] = jaxfmm_dipole.configure_precision(fp32=True)
    report["devices"] = jaxfmm_dipole.device_report()
    from jaxfmm.fem import element_compile, element_farfield_setup

    body = build_body_mesh(args.kind, args.grid)
    report["counts"] = body["counts"]
    faces = body["faces"]
    centres = body["centres"]
    n_bodies = len(centres)

    # magnetisation states (per body, unit vectors); dip-fmm moments are V * M
    states = {
        "uniform_z": geometry.magnetisation(centres, "uniform_z", geometry.DEFAULT_SEED),
        "random": geometry.magnetisation(centres, "random", geometry.DEFAULT_SEED),
        "vortex": geometry.magnetisation(centres, "vortex", geometry.DEFAULT_SEED),
    }
    fields: dict[str, np.ndarray] = {"centres": centres, "volumes": body["volumes"]}
    for name, M in states.items():
        sigma = geometry.body_face_charges(faces, M)
        total_charge = float(np.sum(sigma * faces["area"]))
        report["states"][name] = {
            "total_charge": total_charge,
            "charged_triangles": int(np.count_nonzero(np.abs(sigma) > 1e-12)),
            "moments_are_V_times_M": True,
        }
        fields[f"M_{name}"] = M
        fields[f"sigma_{name}"] = sigma

    nodes32 = jnp.asarray(body["nodes"].astype(np.float32))
    tris = jnp.asarray(faces["triangles"].astype(np.int32))
    eval_pts = jnp.asarray(centres.astype(np.float32))
    protocol = TimingProtocol(warmups=2, samples=3, evaluations=3)
    for setting in [s.strip() for s in args.settings.split(",") if s.strip()]:
        p = int(setting[1:].split("n")[0])
        near_deg = int(setting.split("n")[1]) if "n" in setting else None
        record: dict = {"p": p, "near_deg": near_deg}
        try:
            started = time.perf_counter()
            kwargs = {"p": p, "N_max": args.nmax}
            if near_deg is not None:
                kwargs["near_deg"] = near_deg
            setup = element_farfield_setup(nodes32, None, tris, eval_pts, **kwargs)
            evaluator = element_compile(setup, field=True)
            record["setup_seconds"] = time.perf_counter() - started
            record["setup_keys"] = sorted(k for k in setup.keys() if not hasattr(setup[k], "shape"))[:40]
            record["near_deg_effective"] = int(setup["near_deg"]) if "near_deg" in setup else None
            record["theta"] = float(setup["theta"]) if "theta" in setup else None
            record["n_levels"] = int(len(setup["lvl_info"])) if "lvl_info" in setup else None
            for name in states:
                sigma = fields[f"sigma_{name}"].astype(np.float32)
                surf_nodal = jnp.asarray(np.repeat(sigma[:, None], 3, axis=1))   # constant per triangle
                started = time.perf_counter()
                H = evaluator(None, surf_nodal)
                H.block_until_ready()
                first = time.perf_counter() - started
                for _ in range(protocol.warmups):
                    evaluator(None, surf_nodal).block_until_ready()
                summary = TimingSummary()
                for _ in range(protocol.samples):
                    started = time.perf_counter()
                    for _ in range(protocol.evaluations):
                        H = evaluator(None, surf_nodal)
                        H.block_until_ready()
                    summary.per_sample_seconds.append((time.perf_counter() - started) / protocol.evaluations)
                H = np.asarray(H, dtype=np.float64)
                fields[f"H_{setting}_{name}"] = H
                record[name] = {"first_call_seconds": first, "output_dtype": str(evaluator(None, surf_nodal).dtype),
                                "finite": bool(np.isfinite(H).all()), **summary.as_dict("device_evaluation")}
            record["memory"] = jaxfmm_dipole.memory_stats()
            record["status"] = "success"
        except Exception as error:  # noqa: BLE001 - recorded per setting
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"[:2000]
        report["settings"][setting] = record

    np.savez(out / "jaxfmm_finite_fields.npz", **fields)
    (out / "jaxfmm_finite_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"counts": body["counts"], "settings": {k: v.get("status") for k, v in report["settings"].items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
