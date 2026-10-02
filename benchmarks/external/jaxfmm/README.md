# jaxFMM GPU comparison campaign

dip-fmm against jaxFMM, a JIT-compiled GPU fast multipole method in JAX, on
the same point-dipole lattices and finite-body meshes as the FMM3D CPU
campaign. **The production home of this comparison is the Article1 campaign
tree** (`Article1/`, kept outside git): campaigns `external_jaxfmm` and
`finite_jaxfmm`, `scripts/adapters/jaxfmm_adapter.py`, `scripts/setup_jaxfmm.sh`,
`analysis/figures_jaxfmm.py`, with rows under `Article1/results` and figures
under `Article1/figures`. The adapter runs the `jaxfmm_campaign` package of
this directory, exported at a pinned commit by `setup_jaxfmm.sh`; the
standalone preflight, orchestrator and analysis kept here are the development
harness that validated the formulation and the tooling. The package
re-implements the Article1 geometry, source state, sampled FP64 reference and
error metrics bit for bit so both are directly comparable.

| Directory | Role |
|---|---|
| `jaxfmm_campaign/` | shared package: geometry (bit-identical to the FMM3D datasets), FP64 reference and metrics, environment fingerprints, timing statistics, the machine benchmark lock, the jaxFMM dipole adapters, the dip-fmm adapters |
| `preflight/` | the tiny isolated gate (`run_preflight.py`): PASS/FAIL without any benchmark |
| `campaign/` | one fresh pinned process per case (`run_jaxfmm_case.py`, `run_dipfmm_case.py`) and the orchestrator (`run_campaign.py`) with its JSON configuration |
| `results/<campaign>/` | one JSON row per case plus `manifest.json`; `collect.py` output and figures |
| `analysis/` | `collect.py` (rows to CSV and Markdown) and `make_figures.py` |
| `install_jaxfmm.sh` | pinned environment: jaxFMM v0.3.3 from the local checkout under `.external/`, JAX with the CUDA 13 wheels |

## The dipole formulation in jaxFMM

jaxFMM kernels are scalar (one charge per source; `jaxfmm.kernels.spec.Kernel`
is documented as "a scalar, translation-invariant kernel"), so a vector dipole
is not a native source and the stray-field application (`jaxfmm.apps.mag`)
is a P1 finite-element discretisation, not a point-dipole path. The exact
route used here is the derivative of the monopole field with respect to the
source positions, which is the classical dipole extension of a
kernel-independent FMM and is available through JAX's forward-mode autodiff:

    H_dipole(x) = d/d eps [ -grad_x sum_j G(x - (y_j + eps m_j)) ]_{eps = 0}

The tangent flows through P2M and P2P only; box centres, tree and targets
stay fixed, and every translation is linear. For the KIFMM engine the stock
driver takes one `positions` array for sources and targets, so the campaign
assembles the same phases through jaxFMM's stable `CubicOperators`/`SourceData`
interface with separate source and target records (`jaxfmm_dipole.py`); the
preflight checks that against the stock driver on a doubled cloud. The flex
engine separates `pts` from `eval_pts` natively and is differentiated as is.

Precision: jaxFMM's documented 32-bit mode (`jax_enable_x64 = False`, uint32
Morton codes, every array float32), with `JAX_DEFAULT_MATMUL_PRECISION=highest`
so float32 matmuls are not silently run as TF32 (jaxFMM's own architecture
notes flag this); dip-fmm's FP32 CUDA path uses cuBLAS in default FP32 mode.

## The three tiers

1. **Point to point.** jaxFMM KIFMM (position-derivative dipoles, above) at
   the lattice sites. dip-fmm's point rows are *reused* from the FMM3D
   campaign's processed tables (`analysis/collect.py --config`), not
   re-measured.
2. **Finite sources, point targets** (the apples-to-apples finite
   comparison). Uniformly magnetised bodies (touching unit cubes, or the six
   Kuhn tetrahedra per cell of a conforming mesh) with random unit
   magnetisation per body. dip-fmm evaluates its exact prism/tetrahedron
   sources at the body centres; jaxFMM's element path evaluates the same
   bodies as constant-charge triangles on their unique faces,
   `sigma = (M_left - M_right) . n` (`jaxfmm_campaign/finite_sources.py`,
   `jaxfmm_element.py`), at identical centres. The self field is included by
   both (dip-fmm includes a finite body's own field regardless of the
   identity map, which only marks point-source self pairs). Every row reports
   bodies, unique faces, triangles (jaxFMM's true source count) and targets.
   jaxFMM's accuracy at body centres is set by its near-field Gauss degree
   (`near_deg`), recorded per row and chosen by achieved error in the
   preflight.
3. **Finite sources, finite targets.** dip-fmm prism-to-prism and
   tetra-to-tetra (cell-averaged H), which jaxFMM cannot express; reported on
   their own, never inside a point-target ratio.

Both finite tiers score against the solver's FP64 dense plan at 512 sampled
centres (moments `V * M`), cached by the dip-fmm worker and picked up by the
jaxFMM worker (or rescored by `collect.py`).

## Nominal orders are not comparable

KIFMM's `p` is the number of points per cube-face edge of the equivalent
surface: p=4, 6 and 8 place 56, 152 and 296 equivalent charges per box.
dip-fmm's spherical expansion of order n carries (n+1)^2 coefficients: 49 at
order 6, 121 at order 10. On the preflight lattice KIFMM p=6 reaches 1.9e-5
relative L2 while dip-fmm order 6 reaches 1.6e-3 and order 10 reaches 2.8e-4
on the same 64-leaf tree, so the campaign runs every arm at every size and the
analysis pairs arms by *achieved* error (`matched.md`, the
throughput-versus-accuracy figure) instead of by the integer parameter. The
order-6 pair is reported, but as a non-equivalent comparison with its error
gap alongside.

## Running

```bash
# 1. environment (once)
./benchmarks/external/jaxfmm/install_jaxfmm.sh

# 2. tiny isolated gate on the efficiency cores, behind the benchmark lock
python benchmarks/external/jaxfmm/preflight/run_preflight.py --engines kifmm

# 3. production campaign (orchestrator on the E-cores; it pins workers to the P-cores)
taskset -c 16-31 python benchmarks/external/jaxfmm/campaign/run_campaign.py \
    --config benchmarks/external/jaxfmm/campaign/config_fp32_order6.json

# 4. tables and figures
python benchmarks/external/jaxfmm/analysis/collect.py --results benchmarks/external/jaxfmm/results/fp32_order6_point_lattice
python benchmarks/external/jaxfmm/analysis/make_figures.py --results benchmarks/external/jaxfmm/results/fp32_order6_point_lattice \
    --fmm3d-csv Article1/results/processed/external_fmm3d.csv
```

The dip-fmm worker runs in the `cdfmm` Conda environment with `PYTHONPATH`
pointing at a frozen build of the solver commit under test
(`Article1/runtime/dip-fmm/<sha>/build`, built by
`Article1/runtime/build_frozen_runtime.sh`); the jaxFMM worker runs in the
`jaxfmm` environment. Both are launched with `taskset` on the eight physical
performance cores, and every timed case holds the machine-wide benchmark lock
that the FMM3D campaign uses, so no two measurements overlap.

## What is measured

* `setup_seconds`: jaxFMM `setup` (tree, pair lists, operator tables, device
  upload) or dip-fmm plan construction (tree, operators, upload).
* `first_call_seconds`: the first jaxFMM call, JIT compilation plus one run;
  `jit_seconds_estimate` subtracts the warm median.
* `device_evaluation_*`: jaxFMM with moments and result resident on the device,
  ended by `block_until_ready`.
* `host_evaluation_*`: NumPy moments in, NumPy `H` out, for both codes; this is
  the comparable field-update boundary and includes the H2D and D2H copies.
* Accuracy: relative L2 error of `H` on 512 Morton-stride sampled targets
  against the FP64 direct sum (the FMM3D campaign's cached references are
  reused when present), plus the maximum per-target error over the RMS
  reference magnitude.
* Memory: JAX `memory_stats` (bytes in use, peak) for jaxFMM; the persistent
  device plan bytes and the nvidia-smi process footprint for dip-fmm.

Statistics follow the FMM3D campaign: warmups, then `samples` timed blocks
of `evaluations` updates each, headline = upper median over samples.
