# SPDX-License-Identifier: Apache-2.0
"""jaxFMM half of the preflight: runs alone in the ``jaxfmm`` environment.

Stages A, B and D of the preflight on a tiny lattice. Writes a JSON report
and an ``.npz`` with every field it computed so the orchestrator can compare
them with the dip-fmm process without either framework being imported twice.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaxfmm_campaign import environment, geometry, reference  # noqa: E402
from jaxfmm_campaign.timing import TimingProtocol  # noqa: E402


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
                 / max(np.linalg.norm(np.asarray(b, dtype=np.float64)), 1e-300))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--p", type=int, default=6)
    parser.add_argument("--nmax", type=int, default=8,
                        help="leaf size; small so a 512-point tree still has far-field pairs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--engines", default="kifmm,flex")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report: dict = {"stage": {}, "engines": {}}
    report["process"] = environment.process_fingerprint()
    report["jax_environment"] = environment.jax_environment_variables()
    report["jaxfmm_git"] = environment.git_revision(
        Path(os.environ.get("JAXFMM_SOURCE", "/home/mihaa/MagTense/dip-fmm/.external/jaxfmm")))

    # ---- Stage A: import, backend, precision -------------------------------
    # jaxFMM's 32-bit mode (x64 disabled) narrows its Morton codes to uint32 and
    # JAX warns once per call site; the mode itself is recorded in ``precision``.
    import warnings

    warnings.filterwarnings("ignore", message=".*dtype uint64.*")
    import jax
    import jax.numpy as jnp
    import jaxfmm  # noqa: F401
    from importlib.metadata import version as dist_version

    from jaxfmm_campaign import jaxfmm_dipole

    precision = jaxfmm_dipole.configure_precision(fp32=True)
    devices = jaxfmm_dipole.device_report()
    report["versions"] = {
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jaxlib": devices["jaxlib_version"],
        "jaxfmm_distribution": dist_version("jaxFMM"),
        "numpy": np.__version__,
    }
    report["devices"] = devices
    report["precision"] = precision
    probe = jnp.ones((4,)) * 1.5
    report["stage"]["A_default_backend_is_gpu"] = devices["default_backend"] == "gpu"
    report["stage"]["A_device_is_cuda"] = devices["platform"] in ("gpu", "cuda")
    report["stage"]["A_default_dtype_is_float32"] = str(probe.dtype) == "float32"
    report["memory_after_import"] = jaxfmm_dipole.memory_stats()

    # ---- tiny problem -------------------------------------------------------
    dataset = geometry.lattice_dataset(args.grid, state="random")
    dataset_second = geometry.lattice_dataset(args.grid, state="vortex")
    positions = dataset.positions
    report["problem"] = {
        "dataset_id": dataset.dataset_id,
        "grid": args.grid,
        "n_sources": dataset.source_count,
        "n_targets": dataset.target_count,
        "sha256": dataset.sha256,
        "second_state": dataset_second.state,
    }
    reference_first = reference.direct_dipole_field(positions, dataset.moments, positions, dataset.identity_map)
    reference_second = reference.direct_dipole_field(positions, dataset_second.moments, positions, dataset.identity_map)
    potential_first = reference.direct_dipole_potential(positions, dataset.moments, positions, dataset.identity_map)
    fields = {"reference_random": reference_first, "reference_vortex": reference_second}

    # ---- Stage B: monopole normalisation and dipole sign through each engine -
    from jaxfmm import flex, kifmm
    from jaxfmm.util import split_static

    positions32 = jnp.asarray(positions.astype(np.float32))
    charges = jnp.ones((len(positions),), jnp.float32)
    # monopole potential of unit charges: sum_j 1/(4 pi r_ij), self excluded
    separation = positions[:, None, :] - positions[None, :, :]
    distance = np.sqrt(np.einsum("ijk,ijk->ij", separation, separation))
    np.fill_diagonal(distance, np.inf)
    monopole_reference = (1.0 / distance).sum(axis=1) / (4.0 * np.pi)
    monopole_field_reference = (separation / (distance**3)[:, :, None]).sum(axis=1) / (4.0 * np.pi)

    engines = [name.strip() for name in args.engines.split(",") if name.strip()]
    protocol = TimingProtocol(warmups=2, samples=3, evaluations=3)
    for engine in engines:
        record: dict = {}
        if engine == "kifmm":
            setup = kifmm.setup(positions32, p=args.p, N_max=args.nmax)
            stock_pot = np.asarray(kifmm.compile_evaluator(setup)(charges))
            stock_field = np.asarray(kifmm.compile_evaluator(setup, field=True)(charges))
        else:
            setup = flex.setup(positions32, p=args.p, N_max=args.nmax)
            stock_pot = np.asarray(flex.compile_evaluator(setup)(charges))
            stock_field = np.asarray(flex.compile_evaluator(setup, field=True)(charges))
        record["monopole_potential_relative_l2"] = relative_l2(stock_pot, monopole_reference)
        record["monopole_field_relative_l2"] = relative_l2(stock_field, monopole_field_reference)
        record["stock_output_dtype"] = str(stock_field.dtype)

        evaluator = jaxfmm_dipole.build_evaluator(engine, positions, p=args.p, N_max=args.nmax)
        record["parameters"] = {k: (None if v is None else (v if isinstance(v, (int, float, bool, str)) else str(v)))
                                for k, v in evaluator.parameters.items()}
        record["setup_info"] = evaluator.setup_info
        record["setup_seconds"] = evaluator.setup_seconds
        timing = jaxfmm_dipole.time_evaluator(evaluator, dataset.moments, protocol)
        field_first = timing.field
        device_out = evaluator.apply_device(jnp.asarray(dataset.moments.astype(np.float32)))
        record["dipole_output_dtype"] = str(device_out.dtype)
        record["dipole_output_shape"] = list(device_out.shape)
        record["dipole_finite"] = bool(np.isfinite(field_first).all())
        metrics = reference.error_metrics(field_first, reference_first)
        record["dipole_field_vs_direct"] = metrics
        record["timing"] = timing.as_dict()
        fields[f"{engine}_random"] = field_first

        # potential sign convention through the same derivative on the potential
        if engine == "kifmm":
            from jaxfmm.kifmm.setup import evaluate_setup

            skel, arrs = split_static(setup)
            arrs = tuple(jnp.asarray(a) for a in arrs)
            from jaxfmm.util import join_static

            def dipole_potential(m, skel=skel, arrs=arrs):
                s = join_static(skel, arrs)
                f = lambda P: evaluate_setup({**s, "positions": P}, charges, field=False)  # noqa: E731
                return jax.jvp(f, (s["positions"],), (m,))[1]

            # this stock derivative also moves the targets; on the potential the
            # extra term is m_i . grad phi_mono(x_i) = -m_i . H_mono(x_i)
            phi_stock = np.asarray(dipole_potential(jnp.asarray(dataset.moments.astype(np.float32))))
            phi_corrected = phi_stock + np.einsum("ij,ij->i", dataset.moments, stock_field)
            record["dipole_potential_vs_direct_relative_l2"] = relative_l2(phi_corrected, potential_first)
            # cross-check of the split driver against the stock driver on a doubled cloud
            doubled = jaxfmm_dipole.kifmm_doubled_cloud_field(positions, dataset.moments, p=args.p,
                                                              N_max=2 * args.nmax)
            record["doubled_cloud_vs_direct_relative_l2"] = relative_l2(doubled, reference_first)
            record["split_vs_doubled_cloud_relative_l2"] = relative_l2(field_first, doubled)
        else:
            from jaxfmm.flex.fmm import eval_potential
            from jaxfmm.util import join_static

            skel, arrs = split_static(setup)
            arrs = tuple(jnp.asarray(a) for a in arrs)

            def dipole_potential_flex(m, skel=skel, arrs=arrs):
                h = join_static(skel, arrs)
                f = lambda P: eval_potential(charges, **{**h, "pts": P}, field=False)  # noqa: E731
                return jax.jvp(f, (h["pts"],), (m,))[1]

            phi = np.asarray(dipole_potential_flex(jnp.asarray(dataset.moments.astype(np.float32))))
            record["dipole_potential_vs_direct_relative_l2"] = relative_l2(phi, potential_first)

        # ---- Stage D: new moments on the frozen geometry ----------------------
        field_second = evaluator.apply_host(dataset_second.moments.astype(np.float32)).astype(np.float64)
        rebuilt = jaxfmm_dipole.build_evaluator(engine, positions, p=args.p, N_max=args.nmax)
        field_second_rebuilt = rebuilt.apply_host(dataset_second.moments.astype(np.float32)).astype(np.float64)
        repeats = [evaluator.apply_host(dataset_second.moments.astype(np.float32)).astype(np.float64)
                   for _ in range(3)]
        record["second_state_vs_direct"] = reference.error_metrics(field_second, reference_second)
        record["second_state_fixed_vs_rebuilt_relative_l2"] = relative_l2(field_second, field_second_rebuilt)
        record["repeat_max_relative_difference"] = max(relative_l2(r, field_second) for r in repeats)
        fields[f"{engine}_vortex"] = field_second
        record["memory_after_engine"] = jaxfmm_dipole.memory_stats()
        report["engines"][engine] = record

    report["memory_at_exit"] = jaxfmm_dipole.memory_stats()
    np.savez(out / "jaxfmm_fields.npz", **fields)
    (out / "jaxfmm_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"engines": list(report["engines"]), "backend": devices["default_backend"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
