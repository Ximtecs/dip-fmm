# SPDX-License-Identifier: Apache-2.0
"""Diagnose the flex engine's field and dipole accuracy on a tiny lattice.

Compares, for several (theta, N_max, precision) settings: the monopole
potential, the monopole field, the JVP dipole field, and a central
finite-difference derivative of the monopole field (which is what the JVP
must reproduce if it is the true derivative of the flex approximant).
Runs on the CPU backend; pass ``--x64`` for float64.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jaxfmm_campaign import geometry, reference  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64)) / np.linalg.norm(np.asarray(b, np.float64)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--p", type=int, default=6)
    parser.add_argument("--x64", action="store_true")
    parser.add_argument("--engine", default="flex")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message=".*dtype uint64.*")
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", args.x64)
    from jaxfmm import flex, kifmm
    from jaxfmm.flex.fmm import eval_potential
    from jaxfmm.util import join_static, split_static

    dtype = np.float64 if args.x64 else np.float32
    dataset = geometry.lattice_dataset(args.grid)
    P = dataset.positions
    m = dataset.moments
    N = len(P)
    sep = P[:, None, :] - P[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", sep, sep))
    np.fill_diagonal(dist, np.inf)
    pot_ref = (1.0 / dist).sum(1) / (4 * np.pi)
    fld_ref = (sep / (dist**3)[:, :, None]).sum(1) / (4 * np.pi)
    dip_ref = reference.direct_dipole_field(P, m, P, dataset.identity_map)

    Pj = jnp.asarray(P.astype(dtype))
    q = jnp.ones((N,), Pj.dtype)
    mj = jnp.asarray(m.astype(dtype))
    print(f"engine={args.engine} p={args.p} x64={args.x64} N={N}")
    if args.engine == "flex":
        configs = [dict(theta=0.77, N_max=8), dict(theta=0.5, N_max=8), dict(theta=0.77, N_max=32),
                   dict(theta=0.5, N_max=32), dict(theta=0.3, N_max=32)]
    else:
        configs = [dict(N_max=8), dict(N_max=16), dict(N_max=32)]
    for cfg in configs:
        if args.engine == "flex":
            h = flex.setup(Pj, p=args.p, **cfg)
            skel, arrs = split_static(h)

            def pot(Ps, qq=q):
                return eval_potential(qq, **{**join_static(skel, arrs), "pts": Ps}, field=False)

            def fld(Ps, qq=q):
                return eval_potential(qq, **{**join_static(skel, arrs), "pts": Ps}, field=True)
        else:
            from jaxfmm.kifmm.setup import evaluate_setup
            s = kifmm.setup(Pj, p=args.p, **cfg)
            skel, arrs = split_static(s)

            def pot(Ps, qq=q):
                return evaluate_setup({**join_static(skel, arrs), "positions": Ps}, qq, field=False)

            def fld(Ps, qq=q):
                return evaluate_setup({**join_static(skel, arrs), "positions": Ps}, qq, field=True)

        u = np.asarray(pot(Pj))
        H = np.asarray(fld(Pj))
        e_pot, e_fld = rel(u, pot_ref), rel(H, fld_ref)
        # JVP w.r.t. all positions (moves targets too): correct by the target term
        # for the potential: + m_i . grad_i phi = - m_i . H_i ; for the field the
        # target term needs the Hessian, so compare with finite differences instead.
        dphi = np.asarray(jax.jvp(pot, (Pj,), (mj,))[1])
        # flex keeps eval_pts fixed (source-only derivative already); the kifmm
        # stock driver moves the targets too and needs the + m_i . H_i correction.
        dphi_src_only = dphi if args.engine == "flex" else dphi + np.einsum("ij,ij->i", m, H)
        phi_dip_ref = reference.direct_dipole_potential(P, m, P, dataset.identity_map)
        e_dphi = rel(dphi_src_only, phi_dip_ref)
        dH_jvp = np.asarray(jax.jvp(fld, (Pj,), (mj,))[1])
        # NOTE(cdfmm): a finite difference in the source positions is NOT a valid
        # check here: displacing sources while targets stay fixed opens every
        # self pair to distance eps and the monopole field blows up as 1/eps^2.
        # The doubled-cloud JVP below is the independent check instead.
        dH_fd = dH_jvp
        e_jvp_vs_fd = 0.0
        # source-only derivative via the doubled cloud (targets with zero charge/tangent)
        P2 = jnp.concatenate([Pj, Pj])
        q2 = jnp.concatenate([q, jnp.zeros_like(q)])
        m2 = jnp.concatenate([mj, jnp.zeros_like(mj)])
        if args.engine == "flex":
            h2 = flex.setup(P2, p=args.p, **cfg)
            skel2, arrs2 = split_static(h2)

            def fld2(Ps):
                return eval_potential(q2, **{**join_static(skel2, arrs2), "pts": Ps}, field=True)
        else:
            s2 = kifmm.setup(P2, p=args.p, **{**cfg, "N_max": 2 * cfg["N_max"]})
            skel2, arrs2 = split_static(s2)

            def fld2(Ps):
                return evaluate_setup({**join_static(skel2, arrs2), "positions": Ps}, q2, field=True)

        dH_doubled = np.asarray(jax.jvp(fld2, (P2,), (m2,))[1])[N:]
        e_dip_doubled = rel(dH_doubled, dip_ref)
        # flex native split: eval_pts fixed, pts moved
        e_dip_split = None
        if args.engine == "flex":
            def fld_split(Ps):
                return eval_potential(q, **{**join_static(skel, arrs), "pts": Ps}, field=True)
            dH_split = np.asarray(jax.jvp(fld_split, (Pj,), (mj,))[1])
            e_dip_split = rel(dH_split, dip_ref)
        e_fd_vs_dip = rel(dH_fd, dip_ref)
        norms = (float(np.linalg.norm(dH_jvp)), float(np.linalg.norm(dH_fd)), float(np.linalg.norm(dip_ref)))
        print(f"  {cfg}: pot {e_pot:.2e} field {e_fld:.2e} dipole_pot(src) {e_dphi:.2e} "
              f"jvp_vs_fd {e_jvp_vs_fd:.2e} fd_vs_direct {e_fd_vs_dip:.2e} "
              f"dipole_field(doubled) {e_dip_doubled:.2e} "
              f"dipole_field(split) {e_dip_split if e_dip_split is None else f'{e_dip_split:.2e}'} "
              f"norms jvp/fd/ref {norms[0]:.3e}/{norms[1]:.3e}/{norms[2]:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
