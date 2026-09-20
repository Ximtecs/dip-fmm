# cdfmm tutorials

Six notebooks cover the supported Python API in the order a new user needs
it. Each is self-contained, deterministic, runs on the portable CPU build in
well under a minute, and guards its CUDA and oneMKL cells with the
availability queries so it also runs without a device.

| Notebook | What it covers |
|---|---|
| `01_getting_started.ipynb` | the plan lifecycle: geometry, one reusable `UniformFmm`, repeated evaluation with changing moments, units, array shapes, the exact direct reference, what triggers a rebuild |
| `02_finite_geometry.ipynb` | rectangular prisms and tetrahedra, total moments `m = V M`, the four near/far model selectors, finite self fields versus point self exclusion, all nine source/target pairs against the dense direct plan |
| `03_backends_and_execution.ipynb` | precision, `CPU_STATIC`/oneMKL/`CUDA_PARTIAL`/`CUDA_FULL`/`CPU_REFERENCE`, forced P2P packings, precomputed versus procedural point expansions, expansion basis, the diagnostics, and which options to leave on `AUTO` |
| `04_cache_and_periodicity.ipynb` | the persistent cache (keys, cold/warm, which geometries share a plan, `CDFMM_CACHE_DIR`), fully periodic cubic cells with the zero-`k` convention, and a 27-image direct check |
| `05_trees_and_parameters.ipynb` | the uniform tree (Morton order, `list1`/`list2`), depth against near/far work, the two parameter advisers, and `AdaptiveTree` feeding the same static plan |
| `06_operator_chain.ipynb` | P2P, P2M, M2M, M2P, M2L, L2L and L2P one by one with the public operator functions, every route checked against the direct sum, and the coefficients of a real static plan |

`tutorial_utils.py` holds plotting and error-metric helpers;
`adaptive_showcase.py` holds the grain-geometry generators and topology
summaries used by tutorial 5 (and by `python_tests/test_adaptive_showcase.py`).
Neither reimplements any operator.

## Running

From the repository root, after a build that produced the Python module:

```console
conda activate cdfmm
jupyter lab examples/tutorials
```

Select the kernel of the environment that can import `cdfmm` (the tutorials
are saved with the generic `python3` kernel name). The native initialisation
summary printed by every plan construction goes to the terminal that launched
Jupyter, not into the cell.

The notebooks are committed without outputs and are executed by
`python_tests/test_tutorial_notebooks.py` on every CI run:

```console
PYTHONPATH=build python -m pytest python_tests/test_tutorial_notebooks.py -v
```

External comparisons that need another package live elsewhere:
`examples/validation/` (MagTense) and `benchmarks/external/fmm3d/` (FMM3D).
